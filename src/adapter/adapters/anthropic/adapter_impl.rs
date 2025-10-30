use crate::adapter::adapters::support::get_api_key;
use crate::adapter::anthropic::AnthropicStreamer;
use crate::adapter::{Adapter, AdapterKind, ServiceType, WebRequestData};
use crate::chat::{
	ChatOptionsSet, ChatRequest, ChatResponse, ChatRole, ChatStream, ChatStreamResponse, ContentBlock, ContentPart,
	ImageSource, MessageContent, PromptTokensDetails, ReasoningEffort, ToolCall, Usage,
};
use crate::resolver::{AuthData, Endpoint};
use crate::webc::WebResponse;
use crate::{Headers, ModelIden};
use crate::{Result, ServiceTarget};
use reqwest::RequestBuilder;
use reqwest_eventsource::EventSource;
use serde_json::{Value, json};
use tracing::warn;
use value_ext::JsonValueExt;

pub struct AnthropicAdapter;

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
const MAX_TOKENS_64K: u32 = 64000; // claude-3-7-sonnet, claude-sonnet-4
// custom
const MAX_TOKENS_32K: u32 = 32000; // claude-opus-4
const MAX_TOKENS_8K: u32 = 8192; // claude-3-5-sonnet, claude-3-5-haiku
const MAX_TOKENS_4K: u32 = 4096; // claude-3-opus, claude-3-haiku

const ANTHROPIC_VERSION: &str = "2023-06-01";
const MODELS: &[&str] = &[
	"claude-opus-4-1-20250805",
	"claude-opus-4-20250514",
	"claude-sonnet-4-5-20250929",
	"claude-sonnet-4-20250514",
	"claude-3-7-sonnet-latest",
	"claude-haiku-4-5-20251001",
	"claude-3-5-haiku-latest",
	"claude-3-opus-20240229",
	"claude-3-haiku-20240307",
];

impl AnthropicAdapter {
	pub const API_KEY_DEFAULT_ENV_NAME: &str = "ANTHROPIC_API_KEY";
}

impl Adapter for AnthropicAdapter {
	fn default_endpoint() -> Endpoint {
		const BASE_URL: &str = "https://api.anthropic.com/v1/";
		Endpoint::from_static(BASE_URL)
	}

	fn default_auth() -> AuthData {
		AuthData::from_env(Self::API_KEY_DEFAULT_ENV_NAME)
	}

	/// Note: For now, it returns the common models (see above)
	async fn all_model_names(_kind: AdapterKind) -> Result<Vec<String>> {
		Ok(MODELS.iter().map(|s| s.to_string()).collect())
	}

	fn get_service_url(_model: &ModelIden, service_type: ServiceType, endpoint: Endpoint) -> String {
		let base_url = endpoint.base_url();
		match service_type {
			ServiceType::Chat | ServiceType::ChatStream => format!("{base_url}messages"),
			ServiceType::Embed => format!("{base_url}embeddings"), // Anthropic doesn't support embeddings yet
		}
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
		let url = Self::get_service_url(&model, service_type, endpoint);

		// -- Detect OAuth by checking if api_key starts with "Bearer "
		let is_oauth = api_key.starts_with("Bearer ");

		// -- headers
		let headers = if is_oauth {
			// OAuth uses Authorization header and requires anthropic-beta header
			Headers::from(vec![
				("Authorization".to_string(), api_key),
				("anthropic-version".to_string(), ANTHROPIC_VERSION.to_string()),
				("anthropic-beta".to_string(), "oauth-2025-04-20".to_string()),
			])
		} else {
			// Regular API key uses x-api-key header
			Headers::from(vec![
				("x-api-key".to_string(), api_key),
				("anthropic-version".to_string(), ANTHROPIC_VERSION.to_string()),
			])
		};

		// -- Calculate thinking_enabled early to pass to message formatting
		let (model_name, _) = model.model_name.as_model_name_and_namespace();
		let supports_thinking = model_name.contains("claude-opus-4")
			|| model_name.contains("claude-sonnet-4")
			|| model_name.contains("claude-3-7-sonnet")
			|| model_name.contains("claude-haiku-4-5");

		let thinking_enabled = if supports_thinking {
			match options_set.reasoning_effort() {
				Some(ReasoningEffort::Low) => true,
				Some(ReasoningEffort::Medium) => true,
				Some(ReasoningEffort::High) => true,
				Some(ReasoningEffort::Budget(b)) => *b > 0,
				None => false,
			}
		} else {
			false
		};

		// -- Parts
		let AnthropicRequestParts {
			system,
			messages,
			tools,
		} = Self::into_anthropic_request_parts(chat_req, is_oauth, thinking_enabled)?;

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

		// -- Calculate max_tokens first (required for Anthropic)
		let max_tokens = options_set.max_tokens().unwrap_or_else(|| {
			// most likely models used, so put first. Also a little wider with `claude-sonnet` (since name from version 4)
			if model_name.contains("claude-sonnet")
				|| model_name.contains("claude-3-7-sonnet")
				|| model_name.contains("claude-haiku-4-5")
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
		});
		payload.x_insert("max_tokens", max_tokens)?; // required for Anthropic

		// -- Add thinking configuration if enabled
		if thinking_enabled {
			// Convert reasoning effort to budget tokens
			let budget_tokens = match options_set.reasoning_effort() {
				Some(ReasoningEffort::Low) => 4096,     // 4k tokens
				Some(ReasoningEffort::Medium) => 16384, // 16k tokens (recommended starting point)
				Some(ReasoningEffort::High) => 32768,   // 32k tokens
				Some(ReasoningEffort::Budget(b)) => *b as u32,
				None => 16384, // Default to medium if thinking is enabled
			};

			// Ensure budget is at least 1024 (Anthropic minimum)
			let budget_tokens = budget_tokens.max(1024);

			// Ensure budget is less than max_tokens
			let budget_tokens = budget_tokens.min(max_tokens.saturating_sub(100));

			let thinking = json!({
				"type": "enabled",
				"budget_tokens": budget_tokens
			});
			payload.x_insert("thinking", thinking)?;
		}

		// -- Add other supported ChatOptions
		// Temperature cannot be set when thinking is enabled
		if !thinking_enabled {
			if let Some(temperature) = options_set.temperature() {
				payload.x_insert("temperature", temperature)?;
			}
		}

		if !options_set.stop_sequences().is_empty() {
			payload.x_insert("stop_sequences", options_set.stop_sequences())?;
		}

		// top_p restrictions when thinking is enabled
		if let Some(top_p) = options_set.top_p() {
			if thinking_enabled {
				// When thinking is enabled, top_p must be between 0.95 and 1
				if top_p >= 0.95 && top_p <= 1.0 {
					payload.x_insert("top_p", top_p)?;
				}
				// Otherwise skip setting top_p
			} else {
				// Normal top_p when thinking is disabled
				payload.x_insert("top_p", top_p)?;
			}
		}

		Ok(WebRequestData { url, headers, payload })
	}

	fn to_chat_response(
		model_iden: ModelIden,
		web_response: WebResponse,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatResponse> {
		let WebResponse { mut body, .. } = web_response;
		let captured_raw_body = options_set.capture_raw_body().unwrap_or_default().then(|| body.clone());

		// -- Capture the provider_model_iden
		// TODO: Need to be implemented (if available), for now, just clone model_iden
		let provider_model_name: Option<String> = body.x_remove("model").ok();
		let provider_model_iden = model_iden.from_optional_name(provider_model_name);

		// -- Capture the usage
		let usage = body.x_take::<Value>("usage");
		let usage = usage.map(Self::into_usage).unwrap_or_default();

		// -- Capture the content
		let mut content: Vec<MessageContent> = Vec::new();

		// -- Process content items
		let json_content_items: Vec<Value> = body.x_take("content")?;

		// Check if we have thinking blocks mixed with other content
		let has_thinking_blocks = json_content_items.iter().any(|item| {
			matches!(
				item.get("type").and_then(|v| v.as_str()),
				Some("thinking" | "redacted_thinking")
			)
		});

		if has_thinking_blocks {
			// When thinking blocks are present, preserve exact block sequence
			let mut blocks: Vec<ContentBlock> = Vec::new();
			let mut reasoning_content = String::new();

			for mut item in json_content_items {
				let typ: &str = item.x_get_as("type")?;
				match typ {
					"text" => {
						let text = item.x_take("text")?;
						blocks.push(ContentBlock::Text {
							text,
							thought_signature: None,
						});
					}
					"thinking" => {
						// Thinking blocks might have "thinking" field instead of "text"
						let text: String = item.x_take("thinking").or_else(|_| item.x_take("text"))?;
						let signature = item.x_take("signature").ok();
						reasoning_content.push_str(&text);
						reasoning_content.push('\n');
						blocks.push(ContentBlock::Thinking { text, signature });
					}
					"redacted_thinking" => {
						let data = item.x_take("data")?;
						blocks.push(ContentBlock::RedactedThinking { data });
					}
					"tool_use" => {
						let id = item.x_take("id")?;
						let name = item.x_take("name")?;
						let input = item.x_take("input").unwrap_or_default();
						blocks.push(ContentBlock::ToolUse {
							id,
							name,
							input,
							thought_signature: None,
						});
					}
					_ => {
						// Skip unknown block types
						warn!("Unknown content block type in Anthropic response: {}", typ);
					}
				}
			}

			content.push(MessageContent::Blocks(blocks));

			Ok(ChatResponse {
				content,
				reasoning_content: if reasoning_content.is_empty() {
					None
				} else {
					Some(reasoning_content.trim_end().to_string())
				},
				model_iden,
				provider_model_iden,
				usage,
				captured_raw_body,
			})
		} else {
			// No thinking blocks - use traditional parsing for backward compatibility
			let mut text_content: Vec<String> = Vec::new();
			let mut tool_calls: Vec<ToolCall> = vec![];

			for mut item in json_content_items {
				let typ: &str = item.x_get_as("type")?;
				if typ == "text" {
					text_content.push(item.x_take("text")?);
				} else if typ == "tool_use" {
					let call_id = item.x_take::<String>("id")?;
					let fn_name = item.x_take::<String>("name")?;
					let fn_arguments = item.x_take::<Value>("input").unwrap_or_default();
					let tool_call = ToolCall {
						call_id,
						fn_name,
						fn_arguments,
					};
					tool_calls.push(tool_call);
				}
			}

			if !tool_calls.is_empty() {
				content.push(MessageContent::from(tool_calls))
			}

			if !text_content.is_empty() {
				content.push(MessageContent::from(text_content.join("\n")))
			}

			Ok(ChatResponse {
				content,
				reasoning_content: None,
				model_iden,
				provider_model_iden,
				usage,
				captured_raw_body,
			})
		}
	}

	fn to_chat_stream(
		model_iden: ModelIden,
		reqwest_builder: RequestBuilder,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatStreamResponse> {
		let event_source = EventSource::new(reqwest_builder)?;
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
	pub(super) fn into_usage(mut usage_value: Value) -> Usage {
		// IMPORTANT: For Anthropic, the `input_tokens` does not include `cache_creation_input_tokens` or `cache_read_input_tokens`.
		// Therefore, it must be normalized in the OpenAI style, where it includes both cached and written tokens (for symmetry).
		let input_tokens: i32 = usage_value.x_take("input_tokens").ok().unwrap_or(0);
		let cache_creation_input_tokens: i32 = usage_value.x_take("cache_creation_input_tokens").unwrap_or(0);
		let cache_read_input_tokens: i32 = usage_value.x_take("cache_read_input_tokens").unwrap_or(0);
		let completion_tokens: i32 = usage_value.x_take("output_tokens").ok().unwrap_or(0);

		// compute the prompt_tokens
		let prompt_tokens = input_tokens + cache_creation_input_tokens + cache_read_input_tokens;

		// Compute total_tokens
		let total_tokens = prompt_tokens + completion_tokens;

		// For now the logic is to have a Some of PromptTokensDetails if at least one of those value is not 0
		// TODO: Needs to be normalized across adapters.
		let prompt_tokens_details = if cache_creation_input_tokens > 0 || cache_read_input_tokens > 0 {
			Some(PromptTokensDetails {
				cache_creation_tokens: Some(cache_creation_input_tokens),
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
	/// - When is_oauth is true, forces array format for system prompts
	/// - When thinking_enabled is true, adds thinking blocks to assistant messages before tool calls
	fn into_anthropic_request_parts(
		chat_req: ChatRequest,
		is_oauth: bool,
		_thinking_enabled: bool,
	) -> Result<AnthropicRequestParts> {
		let mut messages: Vec<Value> = Vec::new();
		// (content, is_cache_control)
		let mut systems: Vec<(String, bool)> = Vec::new();

		// NOTE: For now, this means the first System cannot have a cache control
		//       so that we do not change too much.
		if let Some(system) = chat_req.system {
			systems.push((system, false));
		}

		// -- Process the messages
		for msg in chat_req.messages {
			let is_cache_control = msg.options.map(|o| o.cache_control.is_some()).unwrap_or(false);

			match msg.role {
				// for now, system and tool messages go to the system
				ChatRole::System => {
					if let MessageContent::Text(content) = msg.content {
						systems.push((content, is_cache_control))
					}
					// TODO: Needs to trace/warn that other types are not supported
				}
				ChatRole::User => {
					let content = match msg.content {
						MessageContent::Text(content) => apply_cache_control_to_text(is_cache_control, content),
						MessageContent::Parts(parts) => {
							let values = parts
								.iter()
								.filter_map(|part| match part {
									ContentPart::Text(text) => Some(json!({"type": "text", "text": text})),
									ContentPart::Image { content_type, source } => match source {
										ImageSource::Url(url) => {
											// Anthropic now supports URL images
											Some(json!({
												"type": "image",
												"source": {
													"type": "url",
													"url": url,
												}
											}))
										}
										ImageSource::Base64(content) => Some(json!({
											"type": "image",
											"source": {
												"type": "base64",
												"media_type": content_type,
												"data": content,
											},
										})),
									},
								})
								.collect::<Vec<Value>>();

							let values = apply_cache_control_to_parts(is_cache_control, values);

							json!(values)
						}
						MessageContent::Blocks(blocks) => {
							// Convert ContentBlocks to Anthropic format
							let values = blocks
								.into_iter()
								.map(|block| match block {
									ContentBlock::Text { text, .. } => json!({"type": "text", "text": text}),
									ContentBlock::Thinking { text, signature } => {
										let mut obj = json!({"type": "thinking", "thinking": text});
										if let Some(sig) = signature {
											obj["signature"] = json!(sig);
										}
										obj
									}
									ContentBlock::RedactedThinking { data } => json!({
										"type": "redacted_thinking",
										"data": data,
									}),
									ContentBlock::ToolUse { id, name, input, .. } => json!({
										"type": "tool_use",
										"id": id,
										"name": name,
										"input": input,
									}),
									ContentBlock::ToolResult {
										tool_use_id, content, ..
									} => json!({
										"type": "tool_result",
										"tool_use_id": tool_use_id,
										"content": content,
									}),
								})
								.collect::<Vec<Value>>();

							let values = apply_cache_control_to_parts(is_cache_control, values);
							json!(values)
						}
						// Use `match` instead of `if let`. This will allow to future-proof this
						// implementation in case some new message content types would appear,
						// this way the library would not compile if not all methods are implemented
						// continue would allow to gracefully skip pushing unserializable message
						// TODO: Probably need to warn if it is a ToolCalls type of content
						MessageContent::ToolCalls(_) => continue,
						MessageContent::ToolResponses(_) => continue,
					};
					messages.push(json! ({"role": "user", "content": content}));
				}
				ChatRole::Assistant => {
					//
					match msg.content {
						MessageContent::Text(content) => {
							let content = apply_cache_control_to_text(is_cache_control, content);
							messages.push(json! ({"role": "assistant", "content": content}))
						}
						MessageContent::ToolCalls(tool_calls) => {
							let tool_calls = tool_calls
								.into_iter()
								.map(|tool_call| {
									// see: https://docs.anthropic.com/en/docs/build-with-claude/tool-use#example-of-successful-tool-result
									json!({
										"type": "tool_use",
										"id": tool_call.call_id,
										"name": tool_call.fn_name,
										"input": tool_call.fn_arguments,
									})
								})
								.collect::<Vec<Value>>();
							let tool_calls = apply_cache_control_to_parts(is_cache_control, tool_calls);
							messages.push(json! ({
								"role": "assistant",
								"content": tool_calls
							}));
						}
						MessageContent::Blocks(blocks) => {
							// For assistant messages with blocks, convert directly
							let values = blocks
								.into_iter()
								.map(|block| match block {
									ContentBlock::Text { text, .. } => json!({"type": "text", "text": text}),
									ContentBlock::Thinking { text, signature } => {
										let mut obj = json!({"type": "thinking", "thinking": text});
										if let Some(sig) = signature {
											obj["signature"] = json!(sig);
										}
										obj
									}
									ContentBlock::RedactedThinking { data } => json!({
										"type": "redacted_thinking",
										"data": data,
									}),
									ContentBlock::ToolUse { id, name, input, .. } => json!({
										"type": "tool_use",
										"id": id,
										"name": name,
										"input": input,
									}),
									ContentBlock::ToolResult {
										tool_use_id, content, ..
									} => json!({
										"type": "tool_result",
										"tool_use_id": tool_use_id,
										"content": content,
									}),
								})
								.collect::<Vec<Value>>();

							let values = apply_cache_control_to_parts(is_cache_control, values);
							messages.push(json! ({
								"role": "assistant",
								"content": values
							}));
						}
						// TODO: Probably need to trace/warn that this will be ignored
						MessageContent::Parts(_) => (),
						MessageContent::ToolResponses(_) => (),
					}
				}
				ChatRole::Tool => {
					if let MessageContent::ToolResponses(tool_responses) = msg.content {
						let tool_responses = tool_responses
							.into_iter()
							.map(|tool_response| {
								json!({
									"type": "tool_result",
									"content": tool_response.content,
									"tool_use_id": tool_response.call_id,
								})
							})
							.collect::<Vec<Value>>();
						let tool_responses = apply_cache_control_to_parts(is_cache_control, tool_responses);
						// FIXME: MessageContent::ToolResponse should be MessageContent::ToolResponses (even if OpenAI does require multi Tool message)
						messages.push(json!({
							"role": "user",
							"content": tool_responses
						}));
					}
					// TODO: Probably need to trace/warn that this will be ignored
				}
			}
		}

		// -- Create the Anthropic system
		// NOTE: Anthropic does not have a "role": "system", just a single optional system property
		let system = if !systems.is_empty() {
			// OAuth always requires array format
			if is_oauth {
				// Build array format for OAuth
				let mut parts: Vec<Value> = Vec::new();

				// OAuth requires Claude Code identification as first system prompt
				parts.push(json!({
					"type": "text",
					"text": "You are Claude Code, Anthropic's official CLI for Claude."
				}));

				// Add user's system prompts, clarifying they override Claude Code identity
				for (idx, (content, is_cache_control)) in systems.iter().enumerate() {
					let text = if idx == 0 {
						// Prepend clarification to first user system prompt
						format!("You are NOT Claude Code. {}", content)
					} else {
						content.clone()
					};

					let mut part = json!({"type": "text", "text": text});
					// Apply cache control if specified or on the last system message
					if *is_cache_control || (idx == systems.len() - 1) {
						part["cache_control"] = json!({"type": "ephemeral", "ttl": "1h"});
					}
					parts.push(part);
				}
				Some(json!(parts))
			} else {
				// Non-OAuth uses existing logic
				let mut last_cache_idx = -1;
				// first determine the last cache control index
				for (idx, (_, is_cache_control)) in systems.iter().enumerate() {
					if *is_cache_control {
						last_cache_idx = idx as i32;
					}
				}
				// Now build the system multi part
				let system: Value = if last_cache_idx > 0 {
					let mut parts: Vec<Value> = Vec::new();
					for (idx, (content, _)) in systems.iter().enumerate() {
						let idx = idx as i32;
						if idx == last_cache_idx {
							let part = json!({"type": "text", "text": content, "cache_control": {"type": "ephemeral", "ttl": "1h"}});
							parts.push(part);
						} else {
							let part = json!({"type": "text", "text": content});
							parts.push(part);
						}
					}
					json!(parts)
				} else {
					let content_buff = systems.iter().map(|(content, _)| content.as_str()).collect::<Vec<&str>>();
					// we add empty line in between each system
					let content = content_buff.join("\n\n");
					json!(content)
				};
				Some(system)
			}
		} else {
			None
		};

		// -- Process the tools
		let mut tools = chat_req.tools.map(|tools| {
			tools
				.into_iter()
				.map(|tool| {
					// TODO: Need to handle the error correctly
					// TODO: Needs to have a custom serializer (tool should not have to match to a provider)
					// NOTE: Right now, low probability, so we just return null if cannot convert to value.
					let mut tool_value = json!({
						"name": tool.name,
						"input_schema": tool.schema,
					});

					if let Some(description) = tool.description {
						// TODO: need to handle error
						let _ = tool_value.x_insert("description", description);
					}
					tool_value
				})
				.collect::<Vec<Value>>()
		});

		if let Some(tool) = tools.as_mut().and_then(|t| t.last_mut()).and_then(|t| t.as_object_mut()) {
			tool.insert("cache_control".to_string(), json!({"type": "ephemeral", "ttl": "1h"}));
		}

		Ok(AnthropicRequestParts {
			system,
			messages,
			tools,
		})
	}
}

/// Apply the cache control logic to a text content
fn apply_cache_control_to_text(is_cache_control: bool, content: String) -> Value {
	if is_cache_control {
		let value = json!({"type": "text", "text": content, "cache_control": {"type": "ephemeral", "ttl": "1h"}});
		json!(vec![value])
	}
	// simple return
	else {
		json!(content)
	}
}

/// Apply the cache control logic to a text content
fn apply_cache_control_to_parts(is_cache_control: bool, parts: Vec<Value>) -> Vec<Value> {
	let mut parts = parts;
	if is_cache_control && !parts.is_empty() {
		let len = parts.len();
		if let Some(last_value) = parts.get_mut(len - 1) {
			// NOTE: For now, if it fails, then, no cache
			let _ = last_value.x_insert("cache_control", json!( {"type": "ephemeral", "ttl": "1h"}));
			// TODO: Should warn
		}
	}
	parts
}

struct AnthropicRequestParts {
	system: Option<Value>,
	messages: Vec<Value>,
	tools: Option<Vec<Value>>,
}

// endregion: --- Support
