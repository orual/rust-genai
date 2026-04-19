use crate::adapter::adapters::support::get_api_key;
use crate::adapter::anthropic::AnthropicStreamer;
use crate::adapter::{Adapter, AdapterKind, ServiceType, WebRequestData};
use crate::chat::{
	Binary, BinarySource, CacheControl, CacheCreationDetails, ChatOptionsSet, ChatRequest, ChatResponse,
	ChatResponseFormat, ChatRole, ChatStream, ChatStreamResponse, ContentPart, MessageContent, PromptTokensDetails,
	ReasoningEffort, StopReason, SystemBlock, Tool, ToolCall, ToolConfig, ToolName, Usage,
};
use crate::resolver::{AuthData, Endpoint};
use crate::webc::{EventSourceStream, WebResponse};
use crate::{Headers, ModelIden};
use crate::{Result, ServiceTarget};
use reqwest::RequestBuilder;
use serde_json::{Map, Value, json};
use std::sync::OnceLock;
use tracing::info;
use tracing::warn;
use value_ext::JsonValueExt;

pub struct AnthropicAdapter;

const REASONING_LOW: u32 = 1024;
const REASONING_MEDIUM: u32 = 8000;
const REASONING_HIGH: u32 = 24000;

// NOTE: For now, those are opt-ins, but should become opt-out when well supported.
// see: effort doc: https://platform.claude.com/docs/en/build-with-claude/effort
// NOTE (pattern fork patch): the `claude-opus-4-7` entries below are added ahead
// of upstream so pattern's primary model gets the same reasoning/adaptive-thinking
// treatment as 4-6. Drop them when upstream catches up. Sonnet 4.7 has not yet
// shipped — revisit when it does. Tracked in REBASE_NOTES_v3_foundation.md.
const SUPPORT_EFFORT_MODELS: &[&str] = &["claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-5"];
const SUPPORT_REASONING_MAX_MODELS: &[&str] = &["claude-opus-4-7", "claude-opus-4-6"];
// see:adaptive thinking: https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
const SUPPORT_ADAPTTIVE_THINK_MODELS: &[&str] = &["claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6"];

fn has_model(model_prefixes: &[&str], model_name: &str) -> bool {
	model_prefixes.iter().any(|prefix| model_name.contains(prefix))
}

/// Returns true when the given model name looks like a Claude Opus model with
/// version >= 4.7 (e.g. `claude-opus-4-7`, `claude-opus-5-0`, ...).
///
/// The regex is unanchored and tolerates arbitrary prefixes/suffixes around the
/// core `claude-opus-<major>-<minor>` portion. Any parse or regex failure is
/// treated as a conservative `false`.
fn is_opus_4_7_or_higher(model_name: &str) -> bool {
	static RE: OnceLock<Option<regex::Regex>> = OnceLock::new();
	let re = RE.get_or_init(|| regex::Regex::new(r"claude-opus-(\d+)-(\d+)").ok());
	let Some(re) = re.as_ref() else {
		return false;
	};
	let Some(caps) = re.captures(model_name) else {
		return false;
	};
	let major = caps.get(1).and_then(|m| m.as_str().parse::<u32>().ok());
	let minor = caps.get(2).and_then(|m| m.as_str().parse::<u32>().ok());
	match (major, minor) {
		(Some(major), Some(minor)) => (major, minor) >= (4, 7),
		_ => false,
	}
}

fn insert_anthropic_reasoning(
	payload: &mut Value,
	output_config: &mut Map<String, Value>,
	model_name: &str,
	effort: &ReasoningEffort,
) -> Result<()> {
	let mut budget: Option<u32> = None;
	let support_effort = has_model(SUPPORT_EFFORT_MODELS, model_name);
	let support_reasoning_max = has_model(SUPPORT_REASONING_MAX_MODELS, model_name);
	let support_adaptive = has_model(SUPPORT_ADAPTTIVE_THINK_MODELS, model_name);
	let support_xhigh = is_opus_4_7_or_higher(model_name);

	// if support effort, we default with effor
	if support_effort {
		let effort = match effort {
			ReasoningEffort::Minimal => "low",
			ReasoningEffort::Low => "low",
			ReasoningEffort::Medium => "medium",
			ReasoningEffort::High => "high",
			ReasoningEffort::XHigh if support_xhigh => "xhigh",
			ReasoningEffort::Max | ReasoningEffort::XHigh if support_reasoning_max => "max",
			ReasoningEffort::Max if support_xhigh => "max",
			ReasoningEffort::XHigh => "high",
			ReasoningEffort::Max => "high",
			// we capture for later
			ReasoningEffort::Budget(val) => {
				budget = Some(*val); // not very elegant
				""
			}
			ReasoningEffort::None => "",
		};

		// if we have an effort, write into the shared output_config map
		if !effort.is_empty() {
			output_config.insert("effort".to_string(), json!(effort));
		}
	}

	// -- if support adaptive, we add it (with the eventual budget tokens)
	// if not (but support effort), it should be fined without the thinking object.
	if support_adaptive {
		let thinking = match budget {
			Some(budget) => json!({
						"type": "adaptive",
						"budget_tokens": budget // if None, should be ok.
			}),
			None => json!({
				"type": "adaptive"}),
		};

		// if support adaptive, we set the thinking type to "adaptive" and let the model decide how to use the budget (if any)
		payload.x_insert("thinking", thinking)?;
	}

	// -- If it does not support effort, fall back on the legacy with with budget
	if !support_effort {
		let thinking_budget = match effort {
			ReasoningEffort::None => None,
			ReasoningEffort::Budget(budget) => Some(*budget),
			ReasoningEffort::Low | ReasoningEffort::Minimal => Some(REASONING_LOW),
			ReasoningEffort::Medium => Some(REASONING_MEDIUM),
			ReasoningEffort::High | ReasoningEffort::Max | ReasoningEffort::XHigh => Some(REASONING_HIGH),
		};

		if let Some(thinking_budget) = thinking_budget {
			payload.x_insert(
				"thinking",
				json!({
					"type": "enabled",
					"budget_tokens": thinking_budget
				}),
			)?;
		}
	}

	Ok(())
}

// NOTE: For Anthropic, the max_tokens must be specified.
//       To avoid surprises, the default value for genai is the maximum for a given model.
// Current logic:
// - if model contains `3-opus` or `3-haiku` 4x max token limit,
// - otherwise assume 8k model
//
// NOTE: Will need to add the thinking option: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
// For max model tokens see: https://docs.anthropic.com/en/docs/about-claude/models/overview
//
// fall back
pub(in crate::adapter) const MAX_TOKENS_64K: u32 = 64000; // claude-opus-4-5 claude-sonnet... (4 and above), claude-haiku..., claude-3-7-sonnet,
// custom
pub(in crate::adapter) const MAX_TOKENS_32K: u32 = 32000; // claude-opus-4
pub(in crate::adapter) const MAX_TOKENS_8K: u32 = 8192; // claude-3-5-sonnet, claude-3-5-haiku
pub(in crate::adapter) const MAX_TOKENS_4K: u32 = 4096; // claude-3-opus, claude-3-haiku

const ANTHROPIC_VERSION: &str = "2023-06-01";

impl AnthropicAdapter {
	pub const API_KEY_DEFAULT_ENV_NAME: &str = "ANTHROPIC_API_KEY";

	pub(in crate::adapter::adapters) async fn list_model_names_for_end_target(
		kind: AdapterKind,
		endpoint: Endpoint,
		auth: AuthData,
	) -> Result<Vec<String>> {
		// -- url
		let base_url = endpoint.base_url();
		let url = format!("{base_url}models");

		// -- auth / headers
		let api_key = auth.single_key_value().ok();
		let headers = api_key
			.map(|api_key| {
				Headers::from(vec![
					("x-api-key".to_string(), api_key),
					("anthropic-version".to_string(), ANTHROPIC_VERSION.to_string()),
				])
			})
			.unwrap_or_default();

		// -- Exec request
		let web_c = crate::webc::WebClient::default();
		let mut res = web_c
			.do_get(&url, &headers)
			.await
			.map_err(|webc_error| crate::Error::WebAdapterCall {
				adapter_kind: kind,
				webc_error,
			})?;

		// -- Format result
		let mut models: Vec<String> = Vec::new();

		if let Value::Array(models_value) = res.body.x_take("data")? {
			for mut model in models_value {
				let model_name: String = model.x_take("id")?;
				models.push(model_name);
			}
		}

		Ok(models)
	}
}

impl Adapter for AnthropicAdapter {
	const DEFAULT_API_KEY_ENV_NAME: Option<&'static str> = Some(Self::API_KEY_DEFAULT_ENV_NAME);

	fn default_endpoint() -> Endpoint {
		const BASE_URL: &str = "https://api.anthropic.com/v1/";
		Endpoint::from_static(BASE_URL)
	}

	fn default_auth() -> AuthData {
		match Self::DEFAULT_API_KEY_ENV_NAME {
			Some(env_name) => AuthData::from_env(env_name),
			None => AuthData::None,
		}
	}

	async fn all_model_names(kind: AdapterKind, endpoint: Endpoint, auth: AuthData) -> Result<Vec<String>> {
		Self::list_model_names_for_end_target(kind, endpoint, auth).await
	}

	fn get_service_url(_model: &ModelIden, service_type: ServiceType, endpoint: Endpoint) -> Result<String> {
		let base_url = endpoint.base_url();
		let url = match service_type {
			ServiceType::Chat | ServiceType::ChatStream => format!("{base_url}messages"),
			ServiceType::Embed => format!("{base_url}embeddings"), // Anthropic doesn't support embeddings yet
		};

		Ok(url)
	}

	fn to_web_request_data(
		target: ServiceTarget,
		service_type: ServiceType,
		chat_req: ChatRequest,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<WebRequestData> {
		let ServiceTarget { endpoint, auth, model } = target;

		// -- api_key
		let api_key = get_api_key(auth, &model)?;

		// -- url
		let url = Self::get_service_url(&model, service_type, endpoint)?;

		// -- headers
		let headers = Headers::from(vec![
			("x-api-key".to_string(), api_key),
			("anthropic-version".to_string(), ANTHROPIC_VERSION.to_string()),
		]);

		// -- Parts
		let AnthropicRequestParts {
			system,
			messages,
			tools,
		} = Self::into_anthropic_request_parts(chat_req)?;

		// -- Extract Model Name and Reasoning
		let (_, raw_model_name) = model.model_name.namespace_and_name();

		// -- Reasoning Budget
		let (model_name, computed_reasoning_effort) = match (raw_model_name, options_set.reasoning_effort()) {
			// No explicity reasoning_effor, try to infer from model name suffix (supports -zero)
			(model, None) => {
				// let model_name: &str = &model.model_name;
				if let Some((prefix, last)) = raw_model_name.rsplit_once('-') {
					let reasoning = match last {
						"zero" => None,
						"None" => Some(ReasoningEffort::Low),
						"minimal" => Some(ReasoningEffort::Low),
						"low" => Some(ReasoningEffort::Low),
						"medium" => Some(ReasoningEffort::Medium),
						"high" => Some(ReasoningEffort::High),
						"xhigh" => Some(ReasoningEffort::XHigh),
						"max" => Some(ReasoningEffort::Max),
						_ => None,
					};
					// create the model name if there was a `-..` reasoning suffix
					let model = if reasoning.is_some() { prefix } else { model };

					(model, reasoning)
				} else {
					(model, None)
				}
			}
			// If reasoning effort, turn the low, medium, budget ones into Budget
			(model, Some(effort)) => (model, Some(effort.clone())),
		};

		// -- Build the basic payload
		let stream = matches!(service_type, ServiceType::ChatStream);
		let mut payload = json!({
			"model": model_name.to_string(),
			"messages": messages,
			"stream": stream
		});

		if let Some(system) = system {
			payload.x_insert("system", system)?;
		}

		if let Some(tools) = tools {
			payload.x_insert("/tools", tools)?;
		}

		// -- Set the reasoning effort
		// Both reasoning effort and structured-output format write into `output_config`.
		// Build a shared map so both contributions end up in the same object.
		let mut output_config: Map<String, Value> = Map::new();

		if let Some(computed_reasoning_effort) = computed_reasoning_effort {
			insert_anthropic_reasoning(&mut payload, &mut output_config, model_name, &computed_reasoning_effort)?;
		}

		if let Some(cache_control) = options_set.cache_control() {
			info!(
				"Anthropic request-level cache_control '{cache_control:?}' is currently ignored. Use message-level cache_control instead."
			);
		}

		// -- Add supported ChatOptions
		if let Some(ChatResponseFormat::JsonSpec(st_json)) = options_set.response_format() {
			// https://platform.claude.com/docs/en/build-with-claude/structured-outputs#json-outputs
			// Note: Anthropic's json_schema format does not use a schema name; JsonSpec.name is intentionally omitted.
			output_config.insert(
				"format".to_string(),
				json!({
					"type": "json_schema",
					"schema": st_json.schema_with_additional_properties_false(),
				}),
			);
		}

		// Insert output_config once, merging effort + format into a single object.
		if !output_config.is_empty() {
			payload.x_insert("output_config", Value::Object(output_config))?;
		}

		if let Some(temperature) = options_set.temperature() {
			payload.x_insert("temperature", temperature)?;
		}

		if !options_set.stop_sequences().is_empty() {
			payload.x_insert("stop_sequences", options_set.stop_sequences())?;
		}

		let max_tokens = Self::resolve_max_tokens(model_name, &options_set);
		payload.x_insert("max_tokens", max_tokens)?; // required for Anthropic

		if let Some(top_p) = options_set.top_p() {
			payload.x_insert("top_p", top_p)?;
		}

		Ok(WebRequestData { url, headers, payload })
	}

	fn to_chat_response(
		model_iden: ModelIden,
		web_response: WebResponse,
		_options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatResponse> {
		let WebResponse { mut body, .. } = web_response;

		// -- Capture the provider_model_iden
		// TODO: Need to be implemented (if available), for now, just clone model_iden
		let provider_model_name: Option<String> = body.x_remove("model").ok();
		let provider_model_iden = model_iden.from_optional_name(provider_model_name);

		// -- Capture the usage
		let usage = body.x_take::<Value>("usage");

		let usage = usage.map(Self::into_usage).unwrap_or_default();
		let stop_reason = body
			.x_take::<Option<String>>("stop_reason")
			.ok()
			.flatten()
			.map(StopReason::from);

		// -- Capture the content
		let mut content: MessageContent = MessageContent::default();

		// NOTE: Here we are going to concatenate all of the Anthropic text content items into one
		//       genai MessageContent::Text. This is more in line with the OpenAI API style,
		//       but loses the fact that they were originally separate items.
		let json_content_items: Vec<Value> = body.x_take("content")?;

		let mut reasoning_content: Vec<String> = Vec::new();

		for mut item in json_content_items {
			let typ: String = item.x_take("type")?;
			match typ.as_ref() {
				"text" => {
					let part = ContentPart::from_text(item.x_take::<String>("text")?);
					content.push(part);
				}
				"thinking" => reasoning_content.push(item.x_take("thinking")?),
				"tool_use" => {
					let call_id = item.x_take::<String>("id")?;
					let fn_name = item.x_take::<String>("name")?;
					// if not found, will be Value::Null
					let fn_arguments = item.x_take::<Value>("input").unwrap_or_default();
					let tool_call = ToolCall {
						call_id,
						fn_name,
						fn_arguments,
						thought_signatures: None,
					};

					let part = ContentPart::ToolCall(tool_call);
					content.push(part);
				}
				other_typ => {
					// insert it back
					item.x_insert("type", other_typ)?;
					content.push(ContentPart::from_custom(item, Some(model_iden.clone())))
				}
			}
		}

		let reasoning_content = if !reasoning_content.is_empty() {
			Some(reasoning_content.join("\n"))
		} else {
			None
		};

		Ok(ChatResponse {
			content,
			reasoning_content,
			model_iden,
			provider_model_iden,
			stop_reason,
			usage,
			captured_raw_body: None, // Set by the client exec_chat
			response_id: None,
		})
	}

	fn to_chat_stream(
		model_iden: ModelIden,
		reqwest_builder: RequestBuilder,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatStreamResponse> {
		let event_source = EventSourceStream::new(reqwest_builder);
		let anthropic_stream = AnthropicStreamer::new(event_source, model_iden.clone(), options_set);
		let chat_stream = ChatStream::from_inter_stream(anthropic_stream);
		Ok(ChatStreamResponse {
			model_iden,
			stream: chat_stream,
		})
	}

	fn to_embed_request_data(
		_service_target: crate::ServiceTarget,
		_embed_req: crate::embed::EmbedRequest,
		_options_set: crate::embed::EmbedOptionsSet<'_, '_>,
	) -> Result<crate::adapter::WebRequestData> {
		Err(crate::Error::AdapterNotSupported {
			adapter_kind: crate::adapter::AdapterKind::Anthropic,
			feature: "embeddings".to_string(),
		})
	}

	fn to_embed_response(
		_model_iden: crate::ModelIden,
		_web_response: crate::webc::WebResponse,
		_options_set: crate::embed::EmbedOptionsSet<'_, '_>,
	) -> Result<crate::embed::EmbedResponse> {
		Err(crate::Error::AdapterNotSupported {
			adapter_kind: crate::adapter::AdapterKind::Anthropic,
			feature: "embeddings".to_string(),
		})
	}
}

// region:    --- Support

impl AnthropicAdapter {
	/// Resolves the max_tokens value for an Anthropic model, using the user-provided
	/// value if set, or a model-appropriate default.
	pub(in crate::adapter) fn resolve_max_tokens(model_name: &str, options_set: &ChatOptionsSet) -> u32 {
		options_set.max_tokens().unwrap_or_else(|| {
			// most likely models used, so put first. Also a little wider with `claude-sonnet` (since name from version 4)
			if model_name.contains("claude-sonnet")
				|| model_name.contains("claude-haiku")
				|| model_name.contains("claude-3-7-sonnet")
				|| model_name.contains("claude-opus-4-5")
			{
				MAX_TOKENS_64K
			} else if model_name.contains("claude-opus-4") {
				MAX_TOKENS_32K
			} else if model_name.contains("claude-3-5") {
				MAX_TOKENS_8K
			} else if model_name.contains("3-opus") || model_name.contains("3-haiku") {
				MAX_TOKENS_4K
			}
			// for now, fall back on the 64K by default (might want to be more conservative)
			else {
				MAX_TOKENS_64K
			}
		})
	}

	pub(in crate::adapter) fn into_usage(mut usage_value: Value) -> Usage {
		// IMPORTANT: For Anthropic, the `input_tokens` does not include `cache_creation_input_tokens` or `cache_read_input_tokens`.
		// Therefore, it must be normalized in the OpenAI style, where it includes both cached and written tokens (for symmetry).
		let input_tokens: i32 = usage_value.x_take("input_tokens").ok().unwrap_or(0);
		let cache_creation_input_tokens: i32 = usage_value.x_take("cache_creation_input_tokens").unwrap_or(0);
		let cache_read_input_tokens: i32 = usage_value.x_take("cache_read_input_tokens").unwrap_or(0);
		let completion_tokens: i32 = usage_value.x_take("output_tokens").ok().unwrap_or(0);

		// Parse cache_creation breakdown if present (TTL-specific breakdown)
		let cache_creation_details = usage_value.get("cache_creation").and_then(parse_cache_creation_details);

		// compute the prompt_tokens
		let prompt_tokens = input_tokens + cache_creation_input_tokens + cache_read_input_tokens;

		// Compute total_tokens
		let total_tokens = prompt_tokens + completion_tokens;

		// For now the logic is to have a Some of PromptTokensDetails if at least one of those value is not 0
		// TODO: Needs to be normalized across adapters.
		let prompt_tokens_details =
			if cache_creation_input_tokens > 0 || cache_read_input_tokens > 0 || cache_creation_details.is_some() {
				Some(PromptTokensDetails {
					cache_creation_tokens: Some(cache_creation_input_tokens),
					cache_creation_details,
					cached_tokens: Some(cache_read_input_tokens),
					audio_tokens: None,
				})
			} else {
				None
			};

		Usage {
			prompt_tokens: Some(prompt_tokens),
			prompt_tokens_details,

			completion_tokens: Some(completion_tokens),
			// for now, None for Anthropic
			completion_tokens_details: None,

			total_tokens: Some(total_tokens),
		}
	}

	/// Takes the GenAI ChatMessages and constructs the System string and JSON Messages for Anthropic.
	/// - Will push the `ChatRequest.system` and system message to `AnthropicRequestParts.system`
	pub(in crate::adapter) fn into_anthropic_request_parts(chat_req: ChatRequest) -> Result<AnthropicRequestParts> {
		let mut messages: Vec<Value> = Vec::new();
		// (content, cache_control)
		let mut systems: Vec<(String, Option<CacheControl>)> = Vec::new();

		// Track TTL ordering for validation (1h must come before 5m)
		let mut seen_5m_cache = false;

		// `system_blocks`, when present, is authoritative — it fully replaces
		// `chat_req.system` and any `ChatRole::System` messages for system-
		// prompt construction. Callers using `system_blocks` are taking explicit
		// control over system-prompt shape (per-block cache_control, ordering).
		let explicit_system_blocks: Option<Vec<SystemBlock>> = chat_req.system_blocks;
		let use_explicit_blocks = explicit_system_blocks.is_some();

		// NOTE: For now, this means the first System cannot have a cache control
		//       so that we do not change too much.
		if !use_explicit_blocks
			&& let Some(system) = chat_req.system
		{
			systems.push((system, None));
		}

		// -- Process the messages
		for msg in chat_req.messages {
			let cache_control = msg.options.and_then(|o| o.cache_control);

			// Check TTL ordering constraint
			if let Some(ref cc) = cache_control {
				match cc {
					CacheControl::Memory | CacheControl::Ephemeral | CacheControl::Ephemeral5m => {
						seen_5m_cache = true;
					}
					CacheControl::Ephemeral1h | CacheControl::Ephemeral24h => {
						if seen_5m_cache {
							warn!(
								"Anthropic cache TTL ordering violation: Ephemeral1h appears after Ephemeral/Ephemeral5m. \
								1-hour cache entries must appear before 5-minute cache entries. \
								See: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#mixing-different-ttls"
							);
						}
					}
				}
			}

			match msg.role {
				// Collect only text for system; other content parts are ignored by Anthropic here.
				// When explicit `system_blocks` are set, they are authoritative and
				// system-role messages are dropped from system-prompt construction.
				ChatRole::System => {
					if !use_explicit_blocks
						&& let Some(system_text) = msg.content.joined_texts()
					{
						systems.push((system_text, cache_control));
					}
				}

				// User message: text, binary (image/document), and tool_result supported.
				ChatRole::User => {
					if msg.content.is_text_only() {
						let text = msg.content.joined_texts().unwrap_or_else(String::new);
						let content = apply_cache_control_to_text(cache_control.as_ref(), text);
						messages.push(json!({"role": "user", "content": content}));
					} else {
						let mut values: Vec<Value> = Vec::new();
						for part in msg.content {
							match part {
								ContentPart::Text(text) => {
									values.push(json!({"type": "text", "text": text}));
								}
								ContentPart::Binary(binary) => {
									let is_image = binary.is_image();
									let Binary {
										content_type, source, ..
									} = binary;

									if is_image {
										match &source {
											BinarySource::Url(_) => {
												// As of this API version, Anthropic doesn't support images by URL directly in messages.
												warn!(
													"Anthropic doesn't support images from URL, need to handle it gracefully"
												);
											}
											BinarySource::Base64(content) => {
												values.push(json!({
													"type": "image",
													"source": {
														"type": "base64",
														"media_type": content_type,
														"data": content,
													}
												}));
											}
										}
									} else {
										match &source {
											BinarySource::Url(url) => {
												values.push(json!({
													"type": "document",
													"source": {
														"type": "url",
														"url": url,
													}
												}));
											}
											BinarySource::Base64(b64) => {
												values.push(json!({
													"type": "document",
													"source": {
														"type": "base64",
														"media_type": content_type,
														"data": b64,
													}
												}));
											}
										}
									}
								}
								// ToolCall is not valid in user content for Anthropic; skip gracefully.
								ContentPart::ToolCall(_tc) => {}
								ContentPart::ToolResponse(tool_response) => {
									values.push(json!({
										"type": "tool_result",
										"content": tool_response.content,
										"tool_use_id": tool_response.call_id,
									}));
								}
								ContentPart::ThoughtSignature(_) => {}
								ContentPart::ReasoningContent(_) => {}
								// Custom are ignored for this logic
								ContentPart::Custom(_) => {}
							}
						}
						let values = apply_cache_control_to_parts(cache_control.as_ref(), values);
						messages.push(json!({"role": "user", "content": values}));
					}
				}

				// Assistant can mix text, thinking, and tool_use entries.
				ChatRole::Assistant => {
					let mut values: Vec<Value> = Vec::new();
					let mut has_tool_use = false;
					let mut has_text = false;

					// Collect thought signatures and reasoning content separately so
					// we can pair them into signed thinking blocks for the wire format.
					// Anthropic requires each thinking block to carry the original signed
					// text (`thinking`) alongside the opaque `signature` it issued.
					//
					// Pairing strategy: zip signatures with reasoning chunks by index.
					// The first signature is paired with all concatenated reasoning text;
					// any additional signatures (from multiple thinking blocks in a single
					// turn) are emitted with an empty `thinking` field. This is lossless
					// for the common single-block case and makes a best-effort attempt for
					// multi-block interleaved thinking where per-block text was merged.
					let mut thought_signatures: Vec<String> = Vec::new();
					let mut reasoning_texts: Vec<String> = Vec::new();

					// First pass: collect thinking-related parts and build non-thinking
					// content blocks in the order they appear.
					for part in msg.content {
						match part {
							ContentPart::Text(text) => {
								has_text = true;
								values.push(json!({"type": "text", "text": text}));
							}
							ContentPart::ToolCall(tool_call) => {
								has_tool_use = true;
								// Anthropic API requires `input` to be an object, never null.
								// Streaming parsers may produce null arguments when deltas are
								// missing or empty; fall back to an empty object in that case.
								let input = if tool_call.fn_arguments.is_null() {
									Value::Object(Map::new())
								} else {
									tool_call.fn_arguments
								};
								// see: https://docs.anthropic.com/en/docs/build-with-claude/tool-use#example-of-successful-tool-result
								values.push(json!({
									"type": "tool_use",
									"id": tool_call.call_id,
									"name": tool_call.fn_name,
									"input": input,
								}));
							}
							ContentPart::ThoughtSignature(sig) => {
								thought_signatures.push(sig);
							}
							ContentPart::ReasoningContent(text) => {
								reasoning_texts.push(text);
							}
							// Unsupported for assistant role in Anthropic message content
							ContentPart::Binary(_) => {}
							ContentPart::ToolResponse(_) => {}
							// Custom are ignored for this logic
							ContentPart::Custom(_) => {}
						}
					}

					// Build signed thinking blocks and prepend them before other content.
					// Anthropic requires thinking blocks to appear before tool_use blocks.
					if !thought_signatures.is_empty() {
						// Combine all reasoning text into one string; for single-block
						// responses (the common case) this is the exact original text.
						let combined_reasoning = reasoning_texts.join("");
						let mut thinking_blocks: Vec<Value> = thought_signatures
							.into_iter()
							.enumerate()
							.map(|(i, sig)| {
								// Assign the full reasoning text to the first block; subsequent
								// blocks from multi-block interleaved thinking get an empty string
								// because the per-block text was merged on the inbound path.
								let thinking_text = if i == 0 { combined_reasoning.clone() } else { String::new() };
								json!({
									"type": "thinking",
									"thinking": thinking_text,
									"signature": sig,
								})
							})
							.collect();
						thinking_blocks.extend(values);
						values = thinking_blocks;
					}

					if !has_tool_use && has_text && cache_control.is_none() && values.len() == 1 {
						// Optimize to simple string when it's only one text part and no cache control.
						let text = values
							.first()
							.and_then(|v| v.get("text"))
							.and_then(|v| v.as_str())
							.unwrap_or_default()
							.to_string();
						let content = apply_cache_control_to_text(None, text);
						messages.push(json!({"role": "assistant", "content": content}));
					} else {
						let values = apply_cache_control_to_parts(cache_control.as_ref(), values);
						messages.push(json!({"role": "assistant", "content": values}));
					}
				}

				// Tool responses are represented as user tool_result items in Anthropic.
				ChatRole::Tool => {
					let mut values: Vec<Value> = Vec::new();
					for part in msg.content {
						if let ContentPart::ToolResponse(tool_response) = part {
							values.push(json!({
								"type": "tool_result",
								"content": tool_response.content,
								"tool_use_id": tool_response.call_id,
							}));
						}
					}
					if !values.is_empty() {
						let values = apply_cache_control_to_parts(cache_control.as_ref(), values);
						messages.push(json!({"role": "user", "content": values}));
					}
				}
			}
		}

		// -- Create the Anthropic system
		// NOTE: Anthropic does not have a "role": "system", just a single optional system property
		let system = if let Some(blocks) = explicit_system_blocks {
			// Explicit `system_blocks` path: always emit array shape, honouring
			// each block's cache_control verbatim. Empty vec → no system at all
			// (caller explicitly asked for no system prompt).
			if blocks.is_empty() {
				None
			} else {
				let parts: Vec<Value> = blocks
					.iter()
					.map(|block| match &block.cache_control {
						Some(cc) => json!({
							"type": "text",
							"text": block.text,
							"cache_control": cache_control_to_json(cc),
						}),
						None => json!({"type": "text", "text": block.text}),
					})
					.collect();
				Some(json!(parts))
			}
		} else if !systems.is_empty() {
			let has_any_cache = systems.iter().any(|(_, cc)| cc.is_some());
			let system: Value = if has_any_cache {
				// Build multi-part system with per-part cache_control
				let parts: Vec<Value> = systems
					.iter()
					.map(|(content, cc)| {
						if let Some(cc) = cc {
							json!({"type": "text", "text": content, "cache_control": cache_control_to_json(cc)})
						} else {
							json!({"type": "text", "text": content})
						}
					})
					.collect();
				json!(parts)
			} else {
				let content_buff = systems.iter().map(|(content, _)| content.as_str()).collect::<Vec<&str>>();
				// we add empty line in between each system
				let content = content_buff.join("\n\n");
				json!(content)
			};
			Some(system)
		} else {
			None
		};

		// -- Process the tools

		let tools: Option<Vec<Value>> = chat_req
			.tools
			.map(|tools| {
				tools
					.into_iter()
					.map(Self::tool_to_anthropic_tool)
					.collect::<Result<Vec<Value>>>()
			})
			.transpose()?;

		Ok(AnthropicRequestParts {
			system,
			messages,
			tools,
		})
	}

	fn tool_to_anthropic_tool(tool: Tool) -> Result<Value> {
		let Tool {
			name,
			description,
			schema,
			config,
			..
		} = tool;

		let name = match name {
			ToolName::WebSearch => "web_search".to_string(),
			ToolName::Custom(name) => name,
		};

		let mut tool_value = json!({"name": name});

		// -- Add type for builtin tool
		#[allow(clippy::single_match)] // will have more
		match name.as_str() {
			"web_search" => {
				tool_value.x_insert("type", "web_search_20250305")?;
			}
			_ => (),
		}

		// NOTE: Fo now, if tool_value.type then, assume bultin and set config as propertie
		if tool_value.get("type").is_some() {
			if let Some(config) = config {
				match config {
					ToolConfig::WebSearch(config) => {
						if let Some(max_uses) = config.max_uses {
							let _ = tool_value.x_insert("max_uses", max_uses);
						}
						if let Some(allowed_domains) = config.allowed_domains {
							let _ = tool_value.x_insert("allowed_domains", allowed_domains);
						}
						if let Some(blocked_domains) = config.blocked_domains {
							let _ = tool_value.x_insert("blocked_domains", blocked_domains);
						}
					}
					// if custom, we assume we flatten the config properties since we are in a builtin
					ToolConfig::Custom(config) => {
						// NOTE: For now, ignore if not object
						tool_value.x_merge(config)?;
					}
				}
			}
		} else {
			tool_value.x_insert("input_schema", schema)?;
			if let Some(description) = description {
				// TODO: need to handle error
				let _ = tool_value.x_insert("description", description);
			}
		}

		Ok(tool_value)
	}
}

/// Convert CacheControl to Anthropic JSON format.
///
/// See: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#1-hour-cache-duration
fn cache_control_to_json(cache_control: &CacheControl) -> Value {
	match cache_control {
		CacheControl::Ephemeral => {
			json!({"type": "ephemeral"})
		}
		CacheControl::Memory => {
			json!({"type": "ephemeral"})
		}
		CacheControl::Ephemeral5m => {
			json!({"type": "ephemeral", "ttl": "5m"})
		}
		CacheControl::Ephemeral1h => {
			json!({"type": "ephemeral", "ttl": "1h"})
		}
		CacheControl::Ephemeral24h => {
			json!({"type": "ephemeral", "ttl": "1h"})
		}
	}
}

/// Parse cache_creation breakdown from Anthropic API response.
///
/// The API returns TTL-specific token counts in the `cache_creation` object:
/// ```json
/// "cache_creation": {
///     "ephemeral_5m_input_tokens": 456,
///     "ephemeral_1h_input_tokens": 100
/// }
/// ```
pub(super) fn parse_cache_creation_details(cache_creation: &Value) -> Option<CacheCreationDetails> {
	let ephemeral_5m_tokens = cache_creation
		.get("ephemeral_5m_input_tokens")
		.and_then(|v| v.as_i64())
		.map(|v| v as i32);
	let ephemeral_1h_tokens = cache_creation
		.get("ephemeral_1h_input_tokens")
		.and_then(|v| v.as_i64())
		.map(|v| v as i32);

	// Only return Some if at least one TTL has tokens
	if ephemeral_5m_tokens.is_some() || ephemeral_1h_tokens.is_some() {
		Some(CacheCreationDetails {
			ephemeral_5m_tokens,
			ephemeral_1h_tokens,
		})
	} else {
		None
	}
}

/// Apply the cache control logic to a text content
fn apply_cache_control_to_text(cache_control: Option<&CacheControl>, content: String) -> Value {
	if let Some(cc) = cache_control {
		let value = json!({"type": "text", "text": content, "cache_control": cache_control_to_json(cc)});
		json!(vec![value])
	}
	// simple return
	else {
		json!(content)
	}
}

/// Apply the cache control logic to a text content
fn apply_cache_control_to_parts(cache_control: Option<&CacheControl>, parts: Vec<Value>) -> Vec<Value> {
	let mut parts = parts;
	if let Some(cc) = cache_control
		&& !parts.is_empty()
	{
		let len = parts.len();
		if let Some(last_value) = parts.get_mut(len - 1) {
			// NOTE: For now, if it fails, then, no cache
			let _ = last_value.x_insert("cache_control", cache_control_to_json(cc));
			// TODO: Should warn
		}
	}
	parts
}

pub(in crate::adapter) struct AnthropicRequestParts {
	pub system: Option<Value>,
	pub messages: Vec<Value>,
	pub tools: Option<Vec<Value>>,
}

// endregion: --- Support

// region:    --- Tests

#[cfg(test)]
mod tests {
	use super::*;
	use crate::ServiceTarget;
	use crate::adapter::{Adapter, ServiceType};
	use crate::chat::{ChatOptions, ChatRequest, JsonSpec};
	use crate::resolver::AuthData;

	/// Regression guard: when both `reasoning_effort` and `JsonSpec` response format are set
	/// on a model that uses the `output_config` effort API (e.g. `claude-sonnet-4-6`), both
	/// `effort` and `format` must appear inside the same `output_config` JSON object.
	#[test]
	fn test_output_config_merges_effort_and_format() {
		let chat_options = ChatOptions {
			reasoning_effort: Some(ReasoningEffort::High),
			response_format: Some(ChatResponseFormat::JsonSpec(JsonSpec::new(
				"anthropic_ignores_name", // NOTE: Anthropic doesn't recognize a "name" field
				json!({"type": "object", "properties": {}}),
			))),
			..Default::default()
		};

		let model_iden = ModelIden::new(AdapterKind::Anthropic, "claude-sonnet-4-6");
		let target = ServiceTarget {
			endpoint: AnthropicAdapter::default_endpoint(),
			auth: AuthData::from_single("test-key"),
			model: model_iden,
		};
		let options_set = ChatOptionsSet::default().with_chat_options(Some(&chat_options));

		let result = AnthropicAdapter::to_web_request_data(
			target,
			ServiceType::Chat,
			ChatRequest::from_user("hello"),
			options_set,
		);

		let web_req = result.expect("to_web_request_data should succeed");
		let output_config = web_req.payload.get("output_config").expect("output_config must be present");

		assert_eq!(
			output_config.get("effort").and_then(|v| v.as_str()),
			Some("high"),
			"effort must be present in output_config"
		);
		assert_eq!(
			output_config.get("format").and_then(|f| f.get("type")).and_then(|v| v.as_str()),
			Some("json_schema"),
			"format.type must be present in output_config"
		);
	}

	#[test]
	fn test_cache_control_to_json_ephemeral() {
		let result = cache_control_to_json(&CacheControl::Ephemeral);
		assert_eq!(result, json!({"type": "ephemeral"}));
	}

	#[test]
	fn test_cache_control_to_json_ephemeral_5m() {
		let result = cache_control_to_json(&CacheControl::Ephemeral5m);
		assert_eq!(result, json!({"type": "ephemeral", "ttl": "5m"}));
	}

	#[test]
	fn test_cache_control_to_json_memory() {
		let result = cache_control_to_json(&CacheControl::Memory);
		assert_eq!(result, json!({"type": "ephemeral"}));
	}

	#[test]
	fn test_cache_control_to_json_ephemeral_1h() {
		let result = cache_control_to_json(&CacheControl::Ephemeral1h);
		assert_eq!(result, json!({"type": "ephemeral", "ttl": "1h"}));
	}

	#[test]
	fn test_cache_control_to_json_ephemeral_24h() {
		let result = cache_control_to_json(&CacheControl::Ephemeral24h);
		assert_eq!(result, json!({"type": "ephemeral", "ttl": "1h"}));
	}

	#[test]
	fn test_parse_cache_creation_details_with_both_ttls() {
		let cache_creation = json!({
			"ephemeral_5m_input_tokens": 456,
			"ephemeral_1h_input_tokens": 100
		});
		let result = parse_cache_creation_details(&cache_creation);
		assert!(result.is_some());
		let details = result.unwrap();
		assert_eq!(details.ephemeral_5m_tokens, Some(456));
		assert_eq!(details.ephemeral_1h_tokens, Some(100));
	}

	#[test]
	fn test_parse_cache_creation_details_with_5m_only() {
		let cache_creation = json!({
			"ephemeral_5m_input_tokens": 456
		});
		let result = parse_cache_creation_details(&cache_creation);
		assert!(result.is_some());
		let details = result.unwrap();
		assert_eq!(details.ephemeral_5m_tokens, Some(456));
		assert_eq!(details.ephemeral_1h_tokens, None);
	}

	#[test]
	fn test_parse_cache_creation_details_with_1h_only() {
		let cache_creation = json!({
			"ephemeral_1h_input_tokens": 100
		});
		let result = parse_cache_creation_details(&cache_creation);
		assert!(result.is_some());
		let details = result.unwrap();
		assert_eq!(details.ephemeral_5m_tokens, None);
		assert_eq!(details.ephemeral_1h_tokens, Some(100));
	}

	#[test]
	fn test_parse_cache_creation_details_empty() {
		let cache_creation = json!({});
		let result = parse_cache_creation_details(&cache_creation);
		assert!(result.is_none());
	}

	// -- system_blocks path (Task 3 fork patch)

	use crate::chat::{ChatMessage, SystemBlock};

	/// Single explicit `SystemBlock` without cache_control still renders as an
	/// array (caller asked for explicit control — we honour it verbatim).
	#[test]
	fn test_system_blocks_single_no_cache_renders_array() {
		let req = ChatRequest::from_user("hi").with_system_blocks(vec![SystemBlock::new("you are helpful")]);
		let parts = AnthropicAdapter::into_anthropic_request_parts(req).expect("request parts");
		let system = parts.system.expect("system must be present");

		let arr = system.as_array().expect("system must be an array when blocks are used");
		assert_eq!(arr.len(), 1);
		assert_eq!(arr[0], json!({"type": "text", "text": "you are helpful"}));
		assert!(arr[0].get("cache_control").is_none(), "no cache_control expected");
	}

	/// Multiple blocks with per-block cache_control, including mixed TTLs in the
	/// 1h-before-5m order Anthropic requires. Each block must render its own
	/// `cache_control` entry.
	#[test]
	fn test_system_blocks_multiple_with_mixed_ttls() {
		let req = ChatRequest::from_user("hi").with_system_blocks(vec![
			SystemBlock::new("stable identity").with_cache_control(CacheControl::Ephemeral1h),
			SystemBlock::new("less stable base"),
			SystemBlock::new("volatile state").with_cache_control(CacheControl::Ephemeral5m),
		]);
		let parts = AnthropicAdapter::into_anthropic_request_parts(req).expect("request parts");
		let system = parts.system.expect("system present");
		let arr = system.as_array().expect("array");

		assert_eq!(arr.len(), 3);
		assert_eq!(
			arr[0],
			json!({
				"type": "text",
				"text": "stable identity",
				"cache_control": {"type": "ephemeral", "ttl": "1h"},
			}),
		);
		assert_eq!(arr[1], json!({"type": "text", "text": "less stable base"}));
		assert!(
			arr[1].get("cache_control").is_none(),
			"middle block has no cache_control"
		);
		assert_eq!(
			arr[2],
			json!({
				"type": "text",
				"text": "volatile state",
				"cache_control": {"type": "ephemeral", "ttl": "5m"},
			}),
		);
	}

	/// `system_blocks` is authoritative: when set, `chat_req.system` is ignored,
	/// as are any `ChatRole::System` messages. Caller has full control.
	#[test]
	fn test_system_blocks_overrides_system_string_and_system_messages() {
		let req = ChatRequest::new(vec![
			ChatMessage::system("ignored system message"),
			ChatMessage::user("real user turn"),
		])
		.with_system("ignored system string")
		.with_system_blocks(vec![SystemBlock::new("only this wins")]);

		let parts = AnthropicAdapter::into_anthropic_request_parts(req).expect("request parts");
		let system = parts.system.expect("system present");
		let arr = system.as_array().expect("array");

		assert_eq!(arr.len(), 1);
		assert_eq!(arr[0].get("text").and_then(|v| v.as_str()), Some("only this wins"));

		// And the user turn should still be in messages (system messages dropped but
		// user messages preserved).
		let user_msgs: Vec<_> = parts
			.messages
			.iter()
			.filter(|m| m.get("role").and_then(|v| v.as_str()) == Some("user"))
			.collect();
		assert_eq!(user_msgs.len(), 1, "user message must survive");
	}

	/// Explicit empty `system_blocks` vec → no system field emitted (caller
	/// actively asked for no system prompt).
	#[test]
	fn test_system_blocks_empty_vec_produces_no_system() {
		let req = ChatRequest::from_user("hi")
			.with_system("ignored")
			.with_system_blocks(Vec::<SystemBlock>::new());
		let parts = AnthropicAdapter::into_anthropic_request_parts(req).expect("request parts");
		assert!(parts.system.is_none(), "empty system_blocks means no system");
	}

	/// Without `system_blocks`, the existing system-string + system-message
	/// behaviour must be preserved (regression guard for the unchanged path).
	#[test]
	fn test_system_blocks_none_preserves_legacy_behaviour() {
		let req = ChatRequest::from_user("hi").with_system("legacy-system");
		let parts = AnthropicAdapter::into_anthropic_request_parts(req).expect("request parts");
		let system = parts.system.expect("legacy system present");

		// Legacy path with no cache_control on any system renders as a plain string,
		// not an array.
		assert_eq!(system, json!("legacy-system"));
	}

	// -- Task 4 fork patch: Opus 4.7 routing

	/// Regression guard: `claude-opus-4-7` must land in every reasoning-support
	/// array (effort, reasoning-max, adaptive-thinking). If upstream eventually
	/// adds its own entries and the fork-patch line is dropped, the assertions
	/// still hold via the upstream definitions.
	#[test]
	fn test_opus_4_7_reasoning_support_arrays() {
		assert!(
			has_model(SUPPORT_EFFORT_MODELS, "claude-opus-4-7"),
			"claude-opus-4-7 must be in SUPPORT_EFFORT_MODELS"
		);
		assert!(
			has_model(SUPPORT_REASONING_MAX_MODELS, "claude-opus-4-7"),
			"claude-opus-4-7 must be in SUPPORT_REASONING_MAX_MODELS"
		);
		assert!(
			has_model(SUPPORT_ADAPTTIVE_THINK_MODELS, "claude-opus-4-7"),
			"claude-opus-4-7 must be in SUPPORT_ADAPTTIVE_THINK_MODELS"
		);
	}

	/// `ANTHROPIC_VERSION` is still the current stable API version per
	/// claude-code + cliproxy (2023-06-01). This test pins the value; if a
	/// future upstream bump is desired, update this assertion deliberately.
	#[test]
	fn test_anthropic_version_constant_pinned() {
		assert_eq!(
			ANTHROPIC_VERSION, "2023-06-01",
			"ANTHROPIC_VERSION bump should be a deliberate decision"
		);
	}

	/// Wire-shape verification: a ToolResponse with `content: Value::Array`
	/// must serialize as a native JSON array on the Anthropic wire, NOT as
	/// a stringified JSON string.
	///
	/// This guards the pattern where segment-3 text is folded INTO the
	/// ToolResponse's content as a nested block array (matching claude-code's
	/// `smooshIntoToolResult` pattern). If `json!({"content": value_array})`
	/// ever emitted `"content": "[{...}]"` (stringified) instead of
	/// `"content": [{...}]` (native), Anthropic would reject the payload.
	///
	/// Verified: serde_json's `json!` macro inlines a `Value::Array` as a
	/// native JSON array — not a string — so `tool_result.content` is emitted
	/// correctly.
	#[test]
	fn test_tool_response_array_content_emits_native_json_array_on_wire() {
		use crate::chat::{ChatMessage, ChatRequest, MessageContent, ToolResponse};

		// Build a ToolResponse whose content is a Value::Array of text blocks —
		// the shape produced by the segment-3 splice in pattern_runtime.
		let folded_content = json!([
			{"type": "text", "text": "seg3 memory context"},
			{"type": "text", "text": "original tool output"},
		]);
		let tool_response = ToolResponse::new_content("toolu_spliced", folded_content);

		// Build an assistant(tool_use) + tool(tool_result) message pair.
		let assistant_msg = ChatMessage::assistant(MessageContent::from_parts(vec![ContentPart::ToolCall(
			crate::chat::ToolCall {
				call_id: "toolu_spliced".into(),
				fn_name: "code".into(),
				fn_arguments: json!({"code": "pure ()"}),
				thought_signatures: None,
			},
		)]));
		let tool_msg = ChatMessage::tool(tool_response);

		let req = ChatRequest::from_user("run this")
			.append_message(assistant_msg)
			.append_message(tool_msg);

		let parts = AnthropicAdapter::into_anthropic_request_parts(req).expect("wire serialization should succeed");

		// The last message in the Anthropic wire format is the user message
		// containing the tool_result block (Tool-role messages are emitted as
		// user-role on the wire with tool_result content blocks).
		let last_wire_msg = parts.messages.last().expect("must have at least one message");
		let wire_content = last_wire_msg.get("content").expect("message must have content");
		let content_arr = wire_content.as_array().expect("top-level content must be an array");

		// Find the tool_result block.
		let tool_result = content_arr
			.iter()
			.find(|block| block.get("type").and_then(|t| t.as_str()) == Some("tool_result"))
			.expect("must have a tool_result block");

		// The nested content of the tool_result must be a native JSON array,
		// NOT a string. A stringified array like "[{...}]" would be a bug.
		let nested_content = tool_result.get("content").expect("tool_result must have a content field");

		assert!(
			nested_content.is_array(),
			"tool_result.content must be a native JSON array, got: {nested_content:?}. \
			If this fails, Value::Array content is being stringified — see the seg3 splice \
			in pattern_runtime/src/agent_loop.rs"
		);

		let nested_arr = nested_content.as_array().unwrap();
		assert_eq!(nested_arr.len(), 2, "must have exactly 2 blocks (seg3 + original)");
		assert_eq!(nested_arr[0]["type"], "text");
		assert_eq!(nested_arr[0]["text"], "seg3 memory context");
		assert_eq!(nested_arr[1]["type"], "text");
		assert_eq!(nested_arr[1]["text"], "original tool output");
	}

	/// Regression guard: thinking blocks captured on turn N must appear verbatim as
	/// `{"type": "thinking", "thinking": "<text>", "signature": "<sig>"}` content
	/// blocks in the outbound Anthropic request payload for turn N+1.
	///
	/// This exercises the end-to-end thinking-block round-trip path:
	/// `ContentPart::ThoughtSignature` + `ContentPart::ReasoningContent` in an
	/// assistant message → wire `thinking` block in the next request.
	///
	/// This test is the primary regression guard for the thinking-block preservation
	/// feature introduced to support Extended Thinking across tool-use cycles.
	#[test]
	fn test_thinking_blocks_preserved_across_tool_use_turns() {
		// Turn N assistant response: the model produced a thinking block, then called a tool.
		// This is the shape produced by `StreamEnd::into_assistant_message_for_tool_use`:
		//   [ThoughtSignature("sig-abc"), ToolCall(...), ReasoningContent("step 1 think")]
		let thinking_text = "step 1: I need to look up the weather.";
		let signature = "sig-ABCDEFG1234567";

		let assistant_msg = ChatMessage::assistant(MessageContent::from_parts(vec![
			ContentPart::ThoughtSignature(signature.to_string()),
			ContentPart::ToolCall(ToolCall {
				call_id: "toolu_01XYZ".to_string(),
				fn_name: "get_weather".to_string(),
				fn_arguments: json!({"city": "Paris"}),
				thought_signatures: None,
			}),
			// ReasoningContent is appended by with_reasoning_content in into_assistant_message_for_tool_use.
			ContentPart::ReasoningContent(thinking_text.to_string()),
		]));

		// Turn N tool result (user-role on the Anthropic wire).
		let tool_result_msg = ChatMessage::tool(
			crate::chat::ToolResponse::new("toolu_01XYZ", "Sunny, 22°C"),
		);

		// Build the multi-turn request for turn N+1.
		let req = ChatRequest::new(vec![
			ChatMessage::user("What's the weather in Paris?"),
			assistant_msg,
			tool_result_msg,
		]);

		let parts = AnthropicAdapter::into_anthropic_request_parts(req).expect("request parts");

		// Find the assistant message in the wire format.
		let assistant_wire = parts
			.messages
			.iter()
			.find(|m| m.get("role").and_then(|v| v.as_str()) == Some("assistant"))
			.expect("must have an assistant message in the wire payload");

		let wire_content = assistant_wire.get("content").expect("assistant message must have content");
		let content_arr = wire_content.as_array().expect("assistant content must be an array");

		// The thinking block must appear BEFORE the tool_use block.
		let thinking_block = content_arr
			.iter()
			.find(|block| block.get("type").and_then(|t| t.as_str()) == Some("thinking"))
			.expect("must have a thinking block in the outbound payload for turn N+1");

		assert_eq!(
			thinking_block.get("thinking").and_then(|v| v.as_str()),
			Some(thinking_text),
			"thinking block must carry the verbatim reasoning text from turn N"
		);
		assert_eq!(
			thinking_block.get("signature").and_then(|v| v.as_str()),
			Some(signature),
			"thinking block must carry the verbatim signature from turn N"
		);

		// The tool_use block must also be present (thinking precedes it).
		let tool_use_block = content_arr
			.iter()
			.find(|block| block.get("type").and_then(|t| t.as_str()) == Some("tool_use"))
			.expect("must have a tool_use block");
		assert_eq!(
			tool_use_block.get("id").and_then(|v| v.as_str()),
			Some("toolu_01XYZ"),
		);

		// Thinking block must appear before tool_use block (Anthropic wire requirement).
		let thinking_idx = content_arr
			.iter()
			.position(|b| b.get("type").and_then(|t| t.as_str()) == Some("thinking"))
			.unwrap();
		let tool_use_idx = content_arr
			.iter()
			.position(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_use"))
			.unwrap();
		assert!(
			thinking_idx < tool_use_idx,
			"thinking block (idx={thinking_idx}) must appear before tool_use block (idx={tool_use_idx})"
		);
	}

	/// Verify that multiple thinking blocks from a single turn are each emitted as
	/// separate `{"type": "thinking", ...}` blocks on the outbound wire. For the
	/// multi-block case, the full reasoning text is assigned to the first signature;
	/// subsequent signatures get empty thinking text (best-effort, since the per-block
	/// text is merged on the inbound streaming path).
	#[test]
	fn test_multiple_thinking_blocks_are_emitted_separately() {
		let sig1 = "sig-FIRST-0001";
		let sig2 = "sig-SECOND-0002";
		let combined_reasoning = "block one text block two text";

		let assistant_msg = ChatMessage::assistant(MessageContent::from_parts(vec![
			ContentPart::ThoughtSignature(sig1.to_string()),
			ContentPart::ThoughtSignature(sig2.to_string()),
			ContentPart::ToolCall(ToolCall {
				call_id: "toolu_multi".to_string(),
				fn_name: "search".to_string(),
				fn_arguments: json!({"q": "test"}),
				thought_signatures: None,
			}),
			ContentPart::ReasoningContent(combined_reasoning.to_string()),
		]));

		let req = ChatRequest::new(vec![
			ChatMessage::user("search for test"),
			assistant_msg,
		]);

		let parts = AnthropicAdapter::into_anthropic_request_parts(req).expect("request parts");

		let assistant_wire = parts
			.messages
			.iter()
			.find(|m| m.get("role").and_then(|v| v.as_str()) == Some("assistant"))
			.expect("must have an assistant message");

		let content_arr = assistant_wire
			.get("content")
			.and_then(|v| v.as_array())
			.expect("content must be an array");

		let thinking_blocks: Vec<&Value> = content_arr
			.iter()
			.filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("thinking"))
			.collect();

		assert_eq!(thinking_blocks.len(), 2, "must have exactly 2 thinking blocks");

		// First block gets the combined reasoning text.
		assert_eq!(
			thinking_blocks[0].get("thinking").and_then(|v| v.as_str()),
			Some(combined_reasoning),
			"first thinking block must carry the combined reasoning text"
		);
		assert_eq!(
			thinking_blocks[0].get("signature").and_then(|v| v.as_str()),
			Some(sig1),
		);

		// Second block gets an empty thinking field (merged-text limitation).
		assert_eq!(
			thinking_blocks[1].get("thinking").and_then(|v| v.as_str()),
			Some(""),
			"second thinking block must have empty thinking field (merged-text limitation)"
		);
		assert_eq!(
			thinking_blocks[1].get("signature").and_then(|v| v.as_str()),
			Some(sig2),
		);
	}
}

// endregion: --- Tests
