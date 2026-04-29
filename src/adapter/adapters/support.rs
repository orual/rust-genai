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

/// Return a tool response content value as a flat string.
///
/// Providers that require plain-string tool content (OpenAI, Gemini, Ollama)
/// use this helper. A `Value::String` is returned verbatim; any other JSON
/// shape is serialized to its compact JSON string representation. This is a
/// lossless fallback — structured content (e.g. Anthropic nested block arrays)
/// round-trips through JSON rather than being silently dropped.
pub fn content_as_string(content: &serde_json::Value) -> String {
	match content {
		serde_json::Value::String(s) => s.clone(),
		other => other.to_string(),
	}
}

// endregion: --- Tool Content Helpers
