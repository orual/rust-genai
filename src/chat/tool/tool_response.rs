use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Response produced by a tool invocation, paired with the originating tool call ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResponse {
	/// Identifier of the originating tool call.
	pub call_id: String,
	/// Tool output payload.
	///
	/// For simple text results this is a `Value::String`. For providers that
	/// accept structured content (e.g. Anthropic's nested text/image block
	/// arrays inside `tool_result.content`) this may be a `Value::Array` of
	/// typed block objects. Other providers that require a flat string receive
	/// the value serialized to its JSON string form via `content_as_string`.
	pub content: Value,
}

/// Constructors
impl ToolResponse {
	/// Creates a new `ToolResponse` with a plain-text payload.
	///
	/// This is the backwards-compatible convenience constructor. The `content`
	/// string is stored as `Value::String` internally. Providers that accept
	/// only flat-string tool content (OpenAI, Gemini, Ollama) receive it
	/// verbatim; Anthropic receives it as a JSON string value, which is also
	/// valid there.
	pub fn new(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
		Self {
			call_id: tool_call_id.into(),
			content: Value::String(content.into()),
		}
	}

	/// Creates a new `ToolResponse` with a structured content payload.
	///
	/// Use this when the content is a nested array of typed blocks (e.g.
	/// Anthropic's `[{"type":"text","text":"..."}]` format inside
	/// `tool_result.content`). Non-Anthropic adapters will stringify the
	/// value to its JSON representation via `content_as_string`.
	pub fn new_content(tool_call_id: impl Into<String>, content: impl Into<Value>) -> Self {
		Self {
			call_id: tool_call_id.into(),
			content: content.into(),
			is_error: None,
		}
	}
}

/// Computed accessors
impl ToolResponse {
	/// Returns an approximate in-memory size of this `ToolResponse`, in bytes,
	/// computed as the sum of the UTF-8 lengths of:
	/// - `call_id`
	/// - JSON serialization of `content` (approximate; falls back to 0 on error)
	pub fn size(&self) -> usize {
		self.call_id.len() + serde_json::to_string(&self.content).map(|s| s.len()).unwrap_or(0)
	}
}

/// Getters
#[allow(unused)]
impl ToolResponse {
	fn tool_call_id(&self) -> &str {
		&self.call_id
	}

	fn content(&self) -> &Value {
		&self.content
	}
}
