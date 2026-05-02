//! The Adapter layer allows adapting client requests/responses to various AI providers.
//! Currently, it employs a static dispatch pattern with the `Adapter` trait and `AdapterDispatcher` implementation.
//! Adapter implementations are organized by adapter type under the `adapters` submodule.
//!
//! Notes:
//! - All `Adapter` trait methods take the `AdapterKind` as an argument, and for now, the `Adapter` trait functions
//!   are all static (i.e., no `&self`). This reduces state management and ensures that all states are passed as arguments.
//! - Only `AdapterKind` from `AdapterConfig` is publicly exported.

// region:    --- Modules

mod adapter_kind;
mod adapter_types;
mod adapters;
mod dispatcher;

// -- Flatten (private, crate, public)
use adapters::*;

pub(crate) use adapter_types::*;
pub(crate) use dispatcher::*;

pub use adapter_kind::*;

// Re-export the Anthropic adapter's wire-conversion entry point so external
// crates can produce request bodies in the exact shape Anthropic expects
// (e.g. for the `/v1/messages/count_tokens` endpoint, which shares the wire
// shape with `/v1/messages`). See `AnthropicRequestParts` doc for rationale.
pub use adapters::anthropic::{AnthropicAdapter, AnthropicRequestParts};

// -- Crate modules
pub(crate) mod inter_stream;

// endregion: --- Modules
