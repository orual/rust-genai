use crate::adapter::AdapterKind;
use crate::adapter::adapters::support::{StreamerCapturedData, StreamerOptions};
use crate::adapter::anthropic::parse_cache_creation_details;
use crate::adapter::inter_stream::{InterStreamEnd, InterStreamEvent};
use crate::chat::{ChatOptionsSet, PromptTokensDetails, StopReason, ToolCall, Usage};
use crate::webc::{Event, EventSourceStream};
use crate::{Error, ModelIden, Result};
use serde_json::{Map, Value};
use std::pin::Pin;
use std::task::{Context, Poll};
use value_ext::JsonValueExt;

pub struct AnthropicStreamer {
	inner: EventSourceStream,
	options: StreamerOptions,

	// -- Set by the poll_next
	/// Flag to prevent polling the EventSource after a MessageStop event
	done: bool,

	captured_data: StreamerCapturedData,
	in_progress_block: InProgressBlock,
}

enum InProgressBlock {
	Text,
	ToolUse {
		id: String,
		name: String,
		input: String,
	},
	/// Accumulates the thinking text and signature for the current thinking block.
	/// Anthropic delivers the signature as a single `delta/signature` event at the
	/// end of each thinking block, but we store it on the block so it stays paired
	/// with the block's reasoning text when we capture it at `content_block_stop`.
	Thinking {
		reasoning: String,
		signature: Option<String>,
	},
}

impl AnthropicStreamer {
	pub fn new(inner: EventSourceStream, model_iden: ModelIden, options_set: ChatOptionsSet<'_, '_>) -> Self {
		Self {
			inner,
			done: false,
			options: StreamerOptions::new(model_iden, options_set),
			captured_data: Default::default(),
			in_progress_block: InProgressBlock::Text,
		}
	}
}

impl futures::Stream for AnthropicStreamer {
	type Item = Result<InterStreamEvent>;

	fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		if self.done {
			return Poll::Ready(None);
		}

		while let Poll::Ready(event) = Pin::new(&mut self.inner).poll_next(cx) {
			// NOTE: At this point, we capture more events than needed for genai::StreamItem, but it serves as documentation.
			match event {
				Some(Ok(Event::Open)) => return Poll::Ready(Some(Ok(InterStreamEvent::Start))),
				Some(Ok(Event::Message(message))) => {
					let message_type = message.event.as_str();

					match message_type {
						"message_start" => {
							self.capture_usage(message_type, &message.data)?;
							continue;
						}
						"message_delta" => {
							self.capture_usage(message_type, &message.data)?;
							// Capture stop_reason from delta (e.g., "end_turn", "max_tokens", "tool_use")
							if let Ok(data) = self.parse_message_data(&message.data)
								&& let Ok(reason) = data.x_get::<String>("/delta/stop_reason")
							{
								self.captured_data.stop_reason = Some(reason);
							}
							continue;
						}
						"content_block_start" => {
							let mut data: Value =
								serde_json::from_str(&message.data).map_err(|serde_error| Error::StreamParse {
									model_iden: self.options.model_iden.clone(),
									serde_error,
								})?;

							match data.x_get_str("/content_block/type") {
								Ok("text") => self.in_progress_block = InProgressBlock::Text,
								Ok("thinking") => {
									self.in_progress_block = InProgressBlock::Thinking {
										reasoning: String::new(),
										signature: None,
									};
								}
								Ok("tool_use") => {
									let id: String = data.x_take("/content_block/id")?;
									let name: String = data.x_take("/content_block/name")?;

									// Emit an initial ToolCallChunk with name and empty args,
									// matching OpenAI's incremental streaming behaviour.
									let tc = ToolCall {
										call_id: id.clone(),
										fn_name: name.clone(),
										fn_arguments: Value::String(String::new()),
										thought_signatures: None,
										thought_signatures_provenance: None,
									};

									self.in_progress_block = InProgressBlock::ToolUse {
										id,
										name,
										input: String::new(),
									};

									return Poll::Ready(Some(Ok(InterStreamEvent::ToolCallChunk(tc))));
								}
								Ok(txt) => {
									tracing::warn!("unhandled content type: {txt}");
								}
								Err(e) => {
									tracing::error!("{e:?}");
								}
							}

							continue;
						}
						"content_block_delta" => {
							let mut data: Value =
								serde_json::from_str(&message.data).map_err(|serde_error| Error::StreamParse {
									model_iden: self.options.model_iden.clone(),
									serde_error,
								})?;

							match &mut self.in_progress_block {
								InProgressBlock::Text => {
									let content: String = data.x_take("/delta/text")?;

									// Add to the captured_content if chat options say so
									if self.options.capture_content {
										match self.captured_data.content {
											Some(ref mut c) => c.push_str(&content),
											None => self.captured_data.content = Some(content.clone()),
										}
									}

									return Poll::Ready(Some(Ok(InterStreamEvent::Chunk(content))));
								}
								InProgressBlock::ToolUse { id, name, input } => {
									let partial = data.x_get_str("/delta/partial_json")?;
									input.push_str(partial);

									// Emit incremental ToolCallChunk with accumulated args
									// (as Value::String, same convention as OpenAI adapter).
									let tc = ToolCall {
										call_id: id.clone(),
										fn_name: name.clone(),
										fn_arguments: Value::String(input.clone()),
										thought_signatures: None,
										thought_signatures_provenance: None,
									};

									return Poll::Ready(Some(Ok(InterStreamEvent::ToolCallChunk(tc))));
								}
								InProgressBlock::Thinking { reasoning, signature } => {
									if let Ok(thinking) = data.x_take::<String>("/delta/thinking") {
										// Accumulate reasoning text on the in-progress block so we
										// have the full text available at content_block_stop for pairing
										// with the signature before capturing.
										reasoning.push_str(&thinking);

										// Also add to captured_reasoning_content if the option is set.
										if self.options.capture_reasoning_content {
											match self.captured_data.reasoning_content {
												Some(ref mut r) => r.push_str(&thinking),
												None => self.captured_data.reasoning_content = Some(thinking.clone()),
											}
										}

										return Poll::Ready(Some(Ok(InterStreamEvent::ReasoningChunk(thinking))));
									} else if let Ok(sig) = data.x_take::<String>("/delta/signature") {
										// Record the signature on the block. Anthropic delivers this
										// as a single event at the end of the thinking block's deltas.
										*signature = Some(sig.clone());

										return Poll::Ready(Some(Ok(InterStreamEvent::ThoughtSignatureChunk(sig))));
									} else {
										// If it is thinking but no thinking or signature field, we log and skip.
										tracing::warn!(
											"content_block_delta for thinking block but no thinking or signature found: {data:?}"
										);
										continue;
									}
								}
							}
						}
						"content_block_stop" => {
							match std::mem::replace(&mut self.in_progress_block, InProgressBlock::Text) {
								InProgressBlock::ToolUse { id, name, input } => {
									// ToolCallChunks were already emitted incrementally
									// during content_block_start and content_block_delta.
									// Here we only finalize capture with parsed arguments.
									if self.options.capture_tool_calls {
										let fn_arguments = if input.is_empty() {
											Value::Object(Map::new())
										} else {
											serde_json::from_str(&input)?
										};

										let tc = ToolCall {
											call_id: id,
											fn_name: name,
											fn_arguments,
											thought_signatures: None,
											thought_signatures_provenance: None,
										};

										match self.captured_data.tool_calls {
											Some(ref mut t) => t.push(tc),
											None => self.captured_data.tool_calls = Some(vec![tc]),
										}
									}
								}
								InProgressBlock::Thinking { reasoning, signature } => {
									// Capture the signature for this thinking block so the adapter
									// can reconstruct signed thinking blocks on the next turn.
									// We capture whenever we have a signature, regardless of other
									// capture flags, because signature capture is necessary for
									// continued thinking across tool-use cycles.
									if let Some(sig) = signature {
										self.captured_data.push_thought_signature(sig, AdapterKind::Anthropic);
									}
									// The per-block reasoning text has already been accumulated
									// into captured_data.reasoning_content during content_block_delta.
									// We drop the per-block copy here; the combined string is kept.
									let _ = reasoning;
								}
								InProgressBlock::Text => {
									// no-op for text blocks
								}
							}

							continue;
						}
						// -- END MESSAGE
						"message_stop" => {
							// Ensure we do not poll the EventSource anymore on the next poll.
							// NOTE: This way, the last MessageStop event is still sent,
							//       but then, on the next poll, it will be stopped.
							self.done = true;

							// Capture the usage
							let captured_usage = if self.options.capture_usage {
								self.captured_data.usage.take().map(|mut usage| {
									// Compute the total if any of input/output are not null
									if usage.prompt_tokens.is_some() || usage.completion_tokens.is_some() {
										usage.total_tokens = Some(
											usage.prompt_tokens.unwrap_or(0) + usage.completion_tokens.unwrap_or(0),
										);
									}
									usage
								})
							} else {
								None
							};

							let inter_stream_end = InterStreamEnd {
								captured_usage,
								captured_stop_reason: self.captured_data.stop_reason.take().map(StopReason::from),
								captured_text_content: self.captured_data.content.take(),
								captured_reasoning_content: self.captured_data.reasoning_content.take(),
								captured_tool_calls: self.captured_data.tool_calls.take(),
								// Populate with the per-block signatures accumulated during the stream.
								// These are used by the adapter to reconstruct signed thinking blocks
								// on the follow-up request so the model can continue its reasoning chain.
								captured_thought_signatures: self.captured_data.take_thought_signatures(),
								captured_response_id: None,
							};

							// TODO: Need to capture the data as needed
							return Poll::Ready(Some(Ok(InterStreamEvent::End(inter_stream_end))));
						}

						"ping" => continue, // Loop to the next event
						other => tracing::warn!("UNKNOWN MESSAGE TYPE: {other}"),
					}
				}
				Some(Err(err)) => {
					tracing::error!("Error: {}", err);
					return Poll::Ready(Some(Err(Error::WebStream {
						model_iden: self.options.model_iden.clone(),
						cause: err.to_string(),
						error: err,
					})));
				}
				None => return Poll::Ready(None),
			}
		}
		Poll::Pending
	}
}

// Support
impl AnthropicStreamer {
	fn capture_usage(&mut self, message_type: &str, message_data: &str) -> Result<()> {
		if self.options.capture_usage {
			let data = self.parse_message_data(message_data)?;
			// TODO: Might want to exit early if usage is not found

			let (input_path, output_path) = if message_type == "message_start" {
				("/message/usage/input_tokens", "/message/usage/output_tokens")
			} else if message_type == "message_delta" {
				("/usage/input_tokens", "/usage/output_tokens")
			} else {
				// TODO: Use tracing
				tracing::debug!(
					"TRACING DEBUG - Anthropic message type not supported for input/output tokens: {message_type}"
				);
				return Ok(()); // For now permissive
			};

			// -- Capture/Add the eventual input_tokens
			// NOTE: Permissive on this one; if an error occurs, treat it as nonexistent (for now)
			if let Ok(input_tokens) = data.x_get::<i32>(input_path) {
				let val = self
					.captured_data
					.usage
					.get_or_insert(Usage::default())
					.prompt_tokens
					.get_or_insert(0);
				*val += input_tokens;
			}

			if let Ok(output_tokens) = data.x_get::<i32>(output_path) {
				let val = self
					.captured_data
					.usage
					.get_or_insert(Usage::default())
					.completion_tokens
					.get_or_insert(0);
				*val += output_tokens;
			}

			// -- Capture cache tokens (only present in message_start)
			// NOTE: Anthropic's input_tokens does NOT include cached tokens, so we must add them.
			// See also: AnthropicAdapter::into_usage() for non-streaming equivalent.
			if message_type == "message_start" {
				let cache_creation: i32 = data.x_get("/message/usage/cache_creation_input_tokens").unwrap_or(0);
				let cache_read: i32 = data.x_get("/message/usage/cache_read_input_tokens").unwrap_or(0);

				// Parse cache_creation breakdown if present (TTL-specific breakdown)
				// Use x_get with JSON pointer to navigate to /message/usage/cache_creation
				let cache_creation_details = data
					.x_get::<Value>("/message/usage/cache_creation")
					.ok()
					.as_ref()
					.and_then(parse_cache_creation_details);

				if cache_creation > 0 || cache_read > 0 || cache_creation_details.is_some() {
					let usage = self.captured_data.usage.get_or_insert(Usage::default());

					// Add cache tokens to prompt_tokens (same as into_usage does)
					if let Some(ref mut pt) = usage.prompt_tokens {
						*pt += cache_creation + cache_read;
					}

					// Set prompt_tokens_details (match into_usage behavior: always Some(value))
					usage.prompt_tokens_details = Some(PromptTokensDetails {
						cache_creation_tokens: Some(cache_creation),
						cache_creation_details,
						cached_tokens: Some(cache_read),
						audio_tokens: None,
					});
				}
			}
		}

		Ok(())
	}

	/// Simple wrapper for now, with the corresponding map_err.
	/// Might have more logic later.
	fn parse_message_data(&self, payload: &str) -> Result<Value> {
		serde_json::from_str(payload).map_err(|serde_error| Error::StreamParse {
			model_iden: self.options.model_iden.clone(),
			serde_error,
		})
	}
}
