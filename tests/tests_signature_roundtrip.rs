//! Round-trip tests for Anthropic thinking-block signatures — inbound path.
//!
//! Anthropic validates thinking-block signatures byte-exact against the
//! `thinking` text in the block. Any mutation of the text between inbound
//! streaming decode and outbound replay invalidates the signature and
//! causes the API to reject the request with
//! "thinking or redacted_thinking blocks cannot be modified".
//!
//! The companion outbound tests live as unit tests inside
//! `src/adapter/adapters/anthropic/adapter_impl.rs` because they need
//! crate-private access to `AnthropicAdapter::to_web_request_data`.

mod support;

use genai::chat::*;
use support::yakbak::replay_client;
use support::{TestResult, extract_stream_end};

/// Inbound: a streamed Anthropic response with two distinct thinking blocks
/// must produce two `ContentPart::ThinkingBlock` entries whose `text` and
/// `signature` pair 1:1 with the fixture.
#[tokio::test]
async fn test_inbound_multi_block_preserves_per_block_text_and_signature() -> TestResult<()> {
	// -- Fixture contains 2 thinking blocks with distinct text + signatures, then a tool_use.
	let (client, _server) = replay_client("anthropic", "thinking_multi_block_stream").await?;

	let chat_req = ChatRequest::new(vec![ChatMessage::user("What is the weather in Paris?")]);

	let options = ChatOptions::default()
		.with_capture_content(true)
		.with_capture_reasoning_content(true)
		.with_capture_tool_calls(true);

	let stream_res = client
		.exec_chat_stream("anthropic::claude-opus-4-6", chat_req, Some(&options))
		.await?;
	let extract = extract_stream_end(stream_res.stream).await?;

	let captured = extract
		.stream_end
		.captured_content
		.as_ref()
		.ok_or("Should have captured_content")?;

	let thinking_blocks: Vec<&ThinkingBlock> = captured
		.parts()
		.iter()
		.filter_map(|p| p.as_thinking_block())
		.collect();

	assert_eq!(
		thinking_blocks.len(),
		2,
		"expected exactly 2 thinking blocks, got {}",
		thinking_blocks.len()
	);

	assert_eq!(
		thinking_blocks[0].text.as_deref(),
		Some("First block: analyzing the request."),
		"block 0 text must match fixture exactly"
	);
	assert_eq!(
		thinking_blocks[0].signature.as_deref(),
		Some("sig-BLOCK0-AAAAAAAAAAAAAAAA"),
		"block 0 signature must match fixture"
	);
	assert_eq!(
		thinking_blocks[1].text.as_deref(),
		Some("Second block: deciding on the tool call."),
		"block 1 text must match fixture exactly"
	);
	assert_eq!(
		thinking_blocks[1].signature.as_deref(),
		Some("sig-BLOCK1-BBBBBBBBBBBBBBBB"),
		"block 1 signature must match fixture"
	);

	Ok(())
}
