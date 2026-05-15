//! This support module is for common constructs and utilities for all the adapter implementations.
//! It should be private to the `crate::adapter::adapters` module.

use crate::ModelIden;
use crate::adapter::AdapterKind;
use crate::chat::{ChatOptionsSet, ThinkingBlock, Usage};
use crate::resolver::AuthData;
use crate::{Error, Result};

pub fn get_api_key(auth: AuthData, model: &ModelIden) -> Result<String> {
	auth.single_key_value().map_err(|resolver_error| Error::Resolver {
		model_iden: model.clone(),
		resolver_error,
	})
}

// region:    --- StreamerChatOptions

#[derive(Debug)]
pub struct StreamerOptions {
	pub capture_usage: bool,
	pub capture_reasoning_content: bool,
	pub capture_content: bool,
	pub capture_tool_calls: bool,
	pub model_iden: ModelIden,
}

impl StreamerOptions {
	pub fn new(model_iden: ModelIden, options_set: ChatOptionsSet<'_, '_>) -> Self {
		Self {
			capture_usage: options_set.capture_usage().unwrap_or(false),
			capture_content: options_set.capture_content().unwrap_or(false),
			capture_reasoning_content: options_set.capture_reasoning_content().unwrap_or(false),
			capture_tool_calls: options_set.capture_tool_calls().unwrap_or(false),
			model_iden,
		}
	}
}

// endregion: --- StreamerChatOptions

// region:    --- Streamer Captured Data

#[derive(Debug, Default)]
pub struct StreamerCapturedData {
	pub usage: Option<Usage>,
	pub stop_reason: Option<String>,
	pub content: Option<String>,
	pub reasoning_content: Option<String>,
	pub tool_calls: Option<Vec<crate::chat::ToolCall>>,
	/// Per-block thinking payloads (text + signature) captured across the stream.
	/// Elements pair with their originating content block 1:1 so that Anthropic's
	/// byte-exact signature validation holds on replay.
	pub thought_blocks: Option<Vec<ThinkingBlock>>,
	pub thought_signatures_provenance: Option<AdapterKind>,
}

impl StreamerCapturedData {
	/// Append a thought block (per-block text paired with its signature) and
	/// record its provenance. The pairing must be preserved through reconstruction
	/// and outbound replay — Anthropic validates signatures byte-exact against
	/// the paired thinking text.
	///
	/// All blocks on a single stream are expected to come from the same adapter;
	/// the provenance is set on the first call and then verified (mismatches are silently
	/// overwritten in favour of the latest, which should never happen in practice).
	pub fn push_thought_block(&mut self, text: String, sig: String, provenance: AdapterKind) {
		self.thought_blocks
			.get_or_insert_with(Vec::new)
			.push(ThinkingBlock::signed(provenance, text, sig));
		self.thought_signatures_provenance = Some(provenance);
	}

	/// Drain accumulated thought blocks and their provenance, returning them as a
	/// tuple ready to be stored in `InterStreamEnd::captured_thought_blocks`.
	pub fn take_thought_blocks(&mut self) -> Option<(Vec<ThinkingBlock>, AdapterKind)> {
		let blocks = self.thought_blocks.take()?;
		let provenance = self.thought_signatures_provenance.take().unwrap_or(AdapterKind::Anthropic);
		Some((blocks, provenance))
	}
}

// endregion: --- Streamer Captured Data

// region:    --- Tool Content Helpers

/// Flatten a tool response's Vec<ContentPart> to a plain string.
///
/// Used by adapters whose tool-result content field is string-only (OpenAI Chat
/// Completions legacy API; Gemini 2.x; Ollama; Cohere) or as a temporary bridge
/// while per-adapter multi-modal serialization is being implemented.
///
/// - Text parts are joined with newlines.
/// - Binary parts are replaced with [attachment: name-or-content-type] placeholders.
/// - Other ContentPart variants (ToolCall, ToolResponse, ThinkingBlock, Custom)
///   are not semantically meaningful inside a tool result and are dropped silently.
pub fn tool_response_parts_as_string(parts: &[crate::chat::ContentPart]) -> String {
	use crate::chat::ContentPart;
	let mut buf: Vec<String> = Vec::new();
	for part in parts {
		match part {
			ContentPart::Text(s) => buf.push(s.clone()),
			ContentPart::Binary(b) => {
				let label = b.name.clone().unwrap_or_else(|| b.content_type.clone());
				buf.push(format!("[attachment: {label}]"));
			}
			ContentPart::ToolCall(_)
			| ContentPart::ToolResponse(_)
			| ContentPart::ThinkingBlock(_)
			| ContentPart::Custom(_) => {}
		}
	}
	buf.join("\n")
}

// endregion: --- Tool Content Helpers
