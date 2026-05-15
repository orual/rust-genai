use crate::chat::ContentPart;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Response produced by a tool invocation, paired with the originating tool call ID.
///
/// `content` is a `Vec<ContentPart>` allowing multi-modal results (text, images,
/// documents). Single-text results — by far the common case — are constructed via
/// `ToolResponse::new(call_id, text)` and stored as a one-element vec containing a
/// single `ContentPart::Text`. Per-adapter serialization (in
/// `adapter::adapters::*::adapter_impl`) maps the vec to each provider's native
/// tool-result content shape:
///
/// - **Anthropic**: array of typed content blocks inside `tool_result.content`.
/// - **OpenAI Responses**: array of typed content inside `function_call_output.output`.
/// - **Gemini 3.x+**: text parts concatenated into `functionResponse.response`; binary
///   parts emitted as `functionResponse.parts: [...]`.
/// - **OpenAI Chat Completions** (legacy) / **Gemini 2.x** / **Ollama** / **Cohere**:
///   stringified — text parts joined, binary parts replaced with
///   `[attachment: <name>]` placeholders. See `tool_response_parts_as_string`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResponse {
	/// Identifier of the originating tool call.
	pub call_id: String,

	/// Tool output payload as a vec of content parts.
	///
	/// For text-only results this contains a single `ContentPart::Text`.
	/// Multi-modal results may contain a mix of Text and Binary parts.
	/// Other ContentPart variants (ToolCall, ToolResponse, ThinkingBlock) are not
	/// semantically meaningful inside a tool result; adapters MAY emit a warning
	/// or silently drop such parts.
	///
	/// Custom Deserialize handles legacy shapes for back-compat with messages
	/// stored before the multi-modal upgrade (2026-05-15) — see
	/// `deserialize_content_backcompat`.
	#[serde(deserialize_with = "deserialize_content_backcompat", serialize_with = "serialize_content_backcompat")]
	pub content: Vec<ContentPart>,
}

/// Constructors
impl ToolResponse {
	/// Create a new `ToolResponse` with a plain-text payload.
	///
	/// Common case for tools that return text. The text is wrapped in a single
	/// `ContentPart::Text`.
	pub fn new(call_id: impl Into<String>, text: impl Into<String>) -> Self {
		Self {
			call_id: call_id.into(),
			content: vec![ContentPart::Text(text.into())],
		}
	}

	/// Create a `ToolResponse` from a vec of content parts.
	///
	/// Use this for multi-modal results — e.g. a tool that reads an image file
	/// and wants to return the image alongside text metadata.
	pub fn from_parts(call_id: impl Into<String>, content: Vec<ContentPart>) -> Self {
		Self {
			call_id: call_id.into(),
			content,
		}
	}

	/// Back-compat constructor accepting a `serde_json::Value`.
	///
	/// Pre-multi-modal callers used `new_content` to stuff arbitrary JSON into the
	/// `content` field (commonly an Anthropic-shaped array of typed blocks). This
	/// constructor preserves that shape by converting the Value to a single
	/// `ContentPart::Text` carrying the JSON-stringified form. New code should use
	/// `from_parts` with typed `ContentPart::Binary` for multi-modal results.
	#[deprecated(note = "use `from_parts` for multi-modal results, or `new` for plain text")]
	pub fn new_content(call_id: impl Into<String>, content: impl Into<Value>) -> Self {
		// Coerce the legacy Value-shaped content using the same logic as the
		// back-compat deserializer, so any caller passing an anthropic-block-shaped
		// array (the old seg3 splice shape) round-trips to the correct wire output.
		let value = content.into();
		let content = match value {
			Value::String(s) => vec![ContentPart::Text(s)],
			Value::Array(items) => items.into_iter().map(coerce_item_to_content_part).collect(),
			Value::Null => Vec::new(),
			other => vec![ContentPart::Text(other.to_string())],
		};
		Self {
			call_id: call_id.into(),
			content,
		}
	}
}

/// Computed accessors
impl ToolResponse {
	/// Approximate in-memory size in bytes.
	pub fn size(&self) -> usize {
		self.call_id.len()
			+ self.content.iter().map(content_part_size).sum::<usize>()
	}

	/// Returns true if the content is exactly a single Text part.
	pub fn is_text_only(&self) -> bool {
		matches!(self.content.as_slice(), [ContentPart::Text(_)])
	}

	/// Returns the legacy-shaped `serde_json::Value` representation of the content.
	///
	/// Mirrors what the custom `serialize_content_backcompat` emits: bare `Value::String`
	/// for single-Text content, anthropic-block-shaped `Value::Array` otherwise. Use this
	/// when you need the legacy wire shape for TUI/plugin emission, or anywhere that was
	/// previously consuming `Value::String(text)` content directly.
	pub fn content_as_legacy_value(&self) -> serde_json::Value {
		use crate::chat::{Binary, BinarySource};
		use serde_json::{Value, json};

		if let [ContentPart::Text(s)] = self.content.as_slice() {
			return Value::String(s.clone());
		}

		let mut blocks: Vec<Value> = Vec::new();
		for part in &self.content {
			match part {
				ContentPart::Text(s) => blocks.push(json!({"type": "text", "text": s})),
				ContentPart::Binary(Binary { content_type, source, name }) => {
					let (block_type, source_value) = if content_type.starts_with("image/") {
						let src = match source {
							BinarySource::Base64(data) => json!({
								"type": "base64",
								"media_type": content_type,
								"data": data,
							}),
							BinarySource::Url(url) => json!({"type": "url", "url": url}),
						};
						("image", src)
					} else {
						let src = match source {
							BinarySource::Base64(data) => json!({
								"type": "base64",
								"media_type": content_type,
								"data": data,
							}),
							BinarySource::Url(url) => json!({"type": "url", "url": url}),
						};
						("document", src)
					};
					let mut block = json!({"type": block_type, "source": source_value});
					if let Some(n) = name {
						block["name"] = json!(n);
					}
					blocks.push(block);
				}
				ContentPart::ToolCall(_)
				| ContentPart::ToolResponse(_)
				| ContentPart::ThinkingBlock(_)
				| ContentPart::Custom(_) => {}
			}
		}
		Value::Array(blocks)
	}

	/// Returns the joined text of all Text parts, ignoring non-text parts.
	/// Returns None if there are no text parts.
	pub fn joined_text(&self) -> Option<String> {
		let texts: Vec<&str> = self
			.content
			.iter()
			.filter_map(|p| match p {
				ContentPart::Text(s) => Some(s.as_str()),
				_ => None,
			})
			.collect();
		if texts.is_empty() { None } else { Some(texts.join("\n")) }
	}
}

fn content_part_size(part: &ContentPart) -> usize {
	use crate::chat::BinarySource;
	match part {
		ContentPart::Text(s) => s.len(),
		ContentPart::Binary(b) => (match &b.source {
			BinarySource::Base64(s) => s.len(),
			BinarySource::Url(s) => s.len(),
		}) + b.content_type.len() + b.name.as_ref().map(|n| n.len()).unwrap_or(0),
		ContentPart::ToolCall(_)
		| ContentPart::ToolResponse(_)
		| ContentPart::ThinkingBlock(_)
		| ContentPart::Custom(_) => 0,
	}
}

/// Deserialize `ToolResponse.content` from one of several legacy shapes.
///
/// Pattern's database has historical messages serialized with the OLD
/// `ToolResponse.content: serde_json::Value` type. Those rows have one of:
/// - `Value::String("text")` — from `ToolResponse::new(id, text)` (common)
/// - `Value::Array([{type: text, text: "..."}, ...])` — anthropic-shaped blocks
///   produced by the seg3 splice (`new_content` path)
/// - `Value::Object({...})` — arbitrary JSON tool result
/// - `Value::Array([{Text: "..."}, ...])` — new shape (forward-compat sanity)
///
/// All shapes are coerced into `Vec<ContentPart>`. Anthropic-shaped text blocks
/// (`{type: "text", text}`) are translated to `ContentPart::Text`. Anthropic-shaped
/// image blocks (`{type: "image", source: {...}}`) are translated to
/// `ContentPart::Binary`. Anything we can't recognize is preserved as a single
/// `ContentPart::Text` carrying the stringified JSON so no data is lost.
fn deserialize_content_backcompat<'de, D>(deserializer: D) -> Result<Vec<ContentPart>, D::Error>
where
	D: serde::Deserializer<'de>,
{
	use serde::Deserialize;

	let raw = Value::deserialize(deserializer)?;
	match raw {
		Value::String(s) => Ok(vec![ContentPart::Text(s)]),
		Value::Array(items) => {
			let mut parts: Vec<ContentPart> = Vec::with_capacity(items.len());
			for item in items {
				parts.push(coerce_item_to_content_part(item));
			}
			Ok(parts)
		}
		Value::Null => Ok(Vec::new()),
		other => Ok(vec![ContentPart::Text(other.to_string())]),
	}
}

/// Convert a single item from a legacy content array into a `ContentPart`.
///
/// Tries each known shape in order:
/// 1. Anthropic-shaped text block `{type: "text", text: "..."}` → Text
/// 2. Anthropic-shaped image block `{type: "image", source: {type: base64, media_type, data}}` → Binary
/// 3. New externally-tagged ContentPart shape `{Text: "..."}` etc → standard deserialize
/// 4. Anything else → fall back to stringified Text so no data is lost
fn coerce_item_to_content_part(item: Value) -> ContentPart {
	use crate::chat::{Binary, BinarySource};
	use std::sync::Arc;

	// Try anthropic-shaped text block first.
	if let Some(obj) = item.as_object() {
		let type_tag = obj.get("type").and_then(|v| v.as_str());
		match type_tag {
			Some("text") => {
				if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
					return ContentPart::Text(text.to_string());
				}
			}
			Some("image") => {
				if let Some(source_obj) = obj.get("source").and_then(|v| v.as_object()) {
					let src_type = source_obj.get("type").and_then(|v| v.as_str());
					let media_type = source_obj
						.get("media_type")
						.and_then(|v| v.as_str())
						.unwrap_or("application/octet-stream")
						.to_string();
					match src_type {
						Some("base64") => {
							if let Some(data) = source_obj.get("data").and_then(|v| v.as_str()) {
								return ContentPart::Binary(Binary {
									content_type: media_type,
									source: BinarySource::Base64(Arc::from(data)),
									name: None,
								});
							}
						}
						Some("url") => {
							if let Some(url) = source_obj.get("url").and_then(|v| v.as_str()) {
								return ContentPart::Binary(Binary {
									content_type: media_type,
									source: BinarySource::Url(url.to_string()),
									name: None,
								});
							}
						}
						_ => {}
					}
				}
			}
			_ => {}
		}
	}

	// Try standard externally-tagged ContentPart deserialize.
	if let Ok(part) = serde_json::from_value::<ContentPart>(item.clone()) {
		return part;
	}

	// Last-resort: stringify the JSON as a Text part so no data is lost.
	ContentPart::Text(item.to_string())
}

/// Serialize `ToolResponse.content` into the legacy shape so token-count paths
/// (which stringify ChatMessage for counting) stay stable across the multi-modal
/// upgrade. Old DB rows had ToolResponse.content as either:
/// - `Value::String("text")` — text-only results (~all today)
/// - `Value::Array([{type: text, text}, ...])` — multi-text from seg3 splice
///
/// We mirror that:
/// - `vec![Text(s)]` → bare `Value::String(s)` (matches old new(id, text))
/// - everything else → anthropic-block-shaped array (matches old splice shape and
///   is what Anthropic accepts on the wire too)
///
/// This means existing serialized rows round-trip byte-for-byte for text-only
/// tool responses, and token counts based on stringified JSON stay consistent.
fn serialize_content_backcompat<S>(parts: &[ContentPart], serializer: S) -> Result<S::Ok, S::Error>
where
	S: serde::Serializer,
{
	use crate::chat::{Binary, BinarySource};
	use serde::ser::SerializeSeq;

	// Shortcut: single Text → bare string.
	if let [ContentPart::Text(s)] = parts {
		return serializer.serialize_str(s);
	}

	// Multi-part or non-Text: emit as anthropic-block-shaped array.
	let mut seq = serializer.serialize_seq(Some(parts.len()))?;
	for part in parts {
		match part {
			ContentPart::Text(s) => {
				seq.serialize_element(&serde_json::json!({"type": "text", "text": s}))?;
			}
			ContentPart::Binary(Binary { content_type, source, name }) => {
				let (block_type, source_value) = if content_type.starts_with("image/") {
					let src = match source {
						BinarySource::Base64(data) => serde_json::json!({
							"type": "base64",
							"media_type": content_type,
							"data": data,
						}),
						BinarySource::Url(url) => serde_json::json!({"type": "url", "url": url}),
					};
					("image", src)
				} else {
					let src = match source {
						BinarySource::Base64(data) => serde_json::json!({
							"type": "base64",
							"media_type": content_type,
							"data": data,
						}),
						BinarySource::Url(url) => serde_json::json!({"type": "url", "url": url}),
					};
					("document", src)
				};
				let mut block = serde_json::json!({"type": block_type, "source": source_value});
				if let Some(n) = name {
					block["name"] = serde_json::json!(n);
				}
				seq.serialize_element(&block)?;
			}
			// Non-meaningful inside tool_result — silently drop.
			ContentPart::ToolCall(_)
			| ContentPart::ToolResponse(_)
			| ContentPart::ThinkingBlock(_)
			| ContentPart::Custom(_) => {}
		}
	}
	seq.end()
}

#[cfg(test)]
mod backcompat_tests {
	use super::*;
	use serde_json::json;

	#[test]
	fn legacy_string_content_deserializes_as_single_text_part() {
		let raw = json!({"call_id": "call_01", "content": "hello world"});
		let tr: ToolResponse = serde_json::from_value(raw).expect("should deserialize");
		assert_eq!(tr.call_id, "call_01");
		assert_eq!(tr.content.len(), 1);
		match &tr.content[0] {
			ContentPart::Text(s) => assert_eq!(s, "hello world"),
			other => panic!("expected Text, got {other:?}"),
		}
	}

	#[test]
	fn legacy_anthropic_array_deserializes_to_text_parts() {
		let raw = json!({
			"call_id": "call_02",
			"content": [
				{"type": "text", "text": "seg3 memory"},
				{"type": "text", "text": "original output"},
			]
		});
		let tr: ToolResponse = serde_json::from_value(raw).expect("should deserialize");
		assert_eq!(tr.content.len(), 2);
		match &tr.content[0] {
			ContentPart::Text(s) => assert_eq!(s, "seg3 memory"),
			other => panic!("expected Text, got {other:?}"),
		}
	}

	#[test]
	fn legacy_object_content_falls_back_to_stringified_text() {
		let raw = json!({"call_id": "call_03", "content": {"result": 42}});
		let tr: ToolResponse = serde_json::from_value(raw).expect("should deserialize");
		assert_eq!(tr.content.len(), 1);
		match &tr.content[0] {
			ContentPart::Text(s) => assert!(s.contains("42"), "got: {s}"),
			other => panic!("expected Text, got {other:?}"),
		}
	}


	#[test]
	fn serialize_text_only_matches_legacy_bare_string() {
		let tr = ToolResponse::new("call_x", "hi");
		let v = serde_json::to_value(&tr).expect("serialize");
		// content must be a bare string, not ["hi"] or [{Text: "hi"}].
		assert_eq!(v["content"], serde_json::Value::String("hi".into()));
	}

	#[test]
	fn serialize_multi_text_matches_anthropic_block_array() {
		let tr = ToolResponse::from_parts(
			"call_y",
			vec![
				ContentPart::Text("seg3".to_string()),
				ContentPart::Text("orig".to_string()),
			],
		);
		let v = serde_json::to_value(&tr).expect("serialize");
		assert_eq!(v["content"][0]["type"], "text");
		assert_eq!(v["content"][0]["text"], "seg3");
		assert_eq!(v["content"][1]["text"], "orig");
	}

	#[test]
	fn serialize_image_emits_anthropic_image_block() {
		use crate::chat::{Binary, BinarySource};
		use std::sync::Arc;
		let tr = ToolResponse::from_parts(
			"call_z",
			vec![ContentPart::Binary(Binary {
				content_type: "image/png".to_string(),
				source: BinarySource::Base64(Arc::from("BASE64DATA")),
				name: None,
			})],
		);
		let v = serde_json::to_value(&tr).expect("serialize");
		assert_eq!(v["content"][0]["type"], "image");
		assert_eq!(v["content"][0]["source"]["type"], "base64");
		assert_eq!(v["content"][0]["source"]["media_type"], "image/png");
		assert_eq!(v["content"][0]["source"]["data"], "BASE64DATA");
	}

	#[test]
	fn deprecated_new_content_with_anthropic_array_preserves_blocks() {
		// Legacy callers using new_content with an anthropic-shaped array (the seg3
		// splice shape) must produce a ToolResponse whose Vec<ContentPart> round-trips
		// back through the custom Serialize to the same wire shape.
		#[allow(deprecated)]
		let tr = ToolResponse::new_content(
			"toolu_spliced",
			serde_json::json!([
				{"type": "text", "text": "seg3 memory"},
				{"type": "text", "text": "original output"},
			]),
		);
		assert_eq!(tr.content.len(), 2, "must preserve both blocks");
		match &tr.content[0] {
			ContentPart::Text(s) => assert_eq!(s, "seg3 memory"),
			other => panic!("first should be Text(seg3), got {other:?}"),
		}
		// Serialize and verify it produces the same anthropic-array shape on the wire.
		let v = serde_json::to_value(&tr).expect("serialize");
		assert_eq!(v["content"][0]["type"], "text");
		assert_eq!(v["content"][0]["text"], "seg3 memory");
		assert_eq!(v["content"][1]["text"], "original output");
	}

	#[test]
	fn new_externally_tagged_array_round_trips() {
		let tr = ToolResponse::from_parts(
			"call_04",
			vec![
				ContentPart::Text("hello".to_string()),
				ContentPart::Text("world".to_string()),
			],
		);
		let serialized = serde_json::to_value(&tr).expect("serialize");
		let roundtrip: ToolResponse = serde_json::from_value(serialized).expect("deserialize");
		assert_eq!(roundtrip.content.len(), 2);
	}
}
