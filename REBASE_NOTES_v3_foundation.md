# Rebase notes — pattern v3-foundation

**Date:** 2026-04-17
**Produced by:** Phase 4 Task 1 of `docs/implementation-plans/2026-04-16-v3-foundation/phase_04.md`.

## Chosen rebase target

**`v0.6.0-beta.17`** (commit `13a7348`).

Rationale: tagged, stable, and currently equivalent to `upstream/main` tip. `0a65190` (xhigh reasoning for Opus 4.7+) is the most recent content change before the tag, so no unreleased upstream improvements justify pinning to a post-tag commit. Pinning to the tag keeps pattern's `Cargo.lock` reference human-legible.

Fork currently diverges 5 commits ahead / 347 commits behind upstream at `0.4.0-alpha.8-WIP`. Approach: fresh-clone upstream, apply minimal patches forward, verify, then force-push to `orual/rust-genai` after explicit sign-off — **not** `git reset --hard` on the existing working copy (preserves old fork as `../rust-genai-legacy-pre-v3/`).

## Upstream API surface verification

All queries from Phase 4 Task 1 Step 3 run against `upstream/main` (commit `13a7348`):

| Required API | Present? | Location |
|---|---|---|
| `AuthData::RequestOverride { url, headers }` | ✅ | `src/resolver/auth_data.rs:15`; applied in `src/client/client_impl.rs:86` (chat) and `:160` (chat_stream) — fully replaces adapter-produced url + headers |
| Adaptive thinking (`type: "adaptive"`) | ✅ | `src/adapter/adapters/anthropic/adapter_impl.rs:31` (`SUPPORT_ADAPTTIVE_THINK_MODELS`), emitted at `:103-107` |
| `ReasoningEffort::XHigh` for Opus 4.7+ | ✅ | `is_opus_4_7_or_higher()` regex dispatch at `adapter_impl.rs:37-38`; XHigh maps to `"xhigh"` effort string |
| `CacheControl` enum with TTL variants | ✅ | `src/chat/chat_message.rs:148` — `Ephemeral` (5m default), `Ephemeral5m`, `Ephemeral1h`, `Ephemeral24h` |
| `extra_headers` on `ChatOptions` | ✅ | `src/chat/chat_options.rs:71` + builder at `:180`; merged in `client_impl.rs` before RequestOverride |
| `ChatStreamEvent::ReasoningChunk` | ✅ | `src/chat/chat_stream.rs:76` |
| `ChatStreamEvent::ToolCallChunk` | ✅ | `src/chat/chat_stream.rs:82` |
| `Usage::cache_creation_tokens` | ✅ | `src/chat/usage.rs:70` |
| `Usage::cached_tokens` | ✅ | `src/chat/usage.rs:76` |
| `Usage::reasoning_tokens` | ✅ | `src/chat/usage.rs:100` |
| `Usage::ephemeral_5m_tokens` / `ephemeral_1h_tokens` | ✅ | `src/chat/usage.rs:50,53` |

## Opus/Sonnet 4.7 registry survey

`SUPPORT_ADAPTTIVE_THINK_MODELS` lists only `claude-opus-4-6` and `claude-sonnet-4-6`. 4.7 coverage is **partial**:

- **`ReasoningEffort::XHigh`** — handled dynamically via `is_opus_4_7_or_higher()` regex; works for any 4.7+ model.
- **Adaptive thinking itself** — gated by literal membership in `SUPPORT_ADAPTTIVE_THINK_MODELS`; `claude-opus-4-7` and `claude-sonnet-4-7` do **not** currently receive `"type": "adaptive"`.

**Conclusion:** pattern's fork adds `claude-opus-4-7` to `SUPPORT_ADAPTTIVE_THINK_MODELS` as Phase 4 Task 4. (Sonnet 4.7 does not currently exist — only Opus has shipped a 4.7.) Comment at the array will remind future maintainers to drop the addition once upstream covers it, and to revisit when Sonnet 4.7+ ships. `SUPPORT_EFFORT_MODELS` / `SUPPORT_REASONING_MAX_MODELS` get the same `claude-opus-4-7` addition for consistency (pattern wants 4.7 to support Max effort end-to-end).

## System-prompt-array gap (drives Task 3 fork patch)

`ChatRequest.system` on upstream (`src/chat/chat_request.rs:13`) is still `Option<String>`. Upstream's Anthropic adapter emits array-shaped system prompts **only** as a side effect of cache-control usage (single `cache_control: ephemeral` on the last block), with no per-block TTL control.

Phase 5's three-segment cache layout requires:
- multiple explicit system blocks with independent content
- independent `cache_control` on each (including `Ephemeral1h` on segment 1)

→ **Task 3 fork patch required**: introduce `SystemPrompt::{Single(String), Blocks(Vec<SystemBlock>)}` enum with `SystemBlock { text, cache_control: Option<CacheControl> }`. Adapter serializes `Blocks` variant to the full Anthropic array shape with per-block `cache_control`. `From<&str>` / `From<String>` impls preserve backward compat for upstream consumers.

## Kept / dropped / added patches summary

**Dropped (upstream subsumes):**
- `18225ac` Anthropic OAuth workaround — upstream's `AuthData::RequestOverride` handles Bearer auth headers. Literal "You are Claude Code" + "You are NOT Claude Code" injection moves to pattern-side `RequestShaper::SubscriptionRoutingShape` (Phase 4 Task 12), not fork-side.
- `e0e16db` Proper anthropic-style thinking — upstream adaptive thinking subsumes.
- `7cc71e3` Extended thinking via reasoning budget — upstream `ReasoningEffort::Budget(val)` + adaptive emission covers.
- `9e5c1d7` Extended thinking bug fix — upstream handles natively.
- `db3dd51` Gemini failure softening — unrelated to pattern's Anthropic path. Re-evaluate if pattern adds Gemini later; can cherry-pick forward at that time.

**Added (new fork patches):**
- **Task 3** — `SystemPrompt` enum with per-block `cache_control`. New commit.
- **Task 4** — `claude-opus-4-7` added to `SUPPORT_ADAPTTIVE_THINK_MODELS`, `SUPPORT_EFFORT_MODELS`, and `SUPPORT_REASONING_MAX_MODELS`. (Sonnet 4.7 not yet released; revisit when it ships.) `ANTHROPIC_VERSION` constant check (no bump expected; `2023-06-01` remains current per claude-code + cliproxy).

**Kept from old fork:** none. All pre-rebase patches are superseded by upstream features or migrating pattern-side.

## Executor next steps (Task 2)

1. `mv /home/orual/Projects/PatternProject/rust-genai /home/orual/Projects/PatternProject/rust-genai-legacy-pre-v3/` — preserve old fork as reference.
2. `git clone https://github.com/orual/rust-genai.git /home/orual/Projects/PatternProject/rust-genai` — fresh clone retaining origin remote.
3. `cd /home/orual/Projects/PatternProject/rust-genai && git remote add upstream https://github.com/jeremychone/rust-genai.git && git fetch upstream --tags`.
4. `git checkout -b rebase/pattern-v3-foundation v0.6.0-beta.17`.
5. Apply Task 3 + Task 4 patches in follow-up commits.
6. `cargo nextest run --all-features` — verify fork tests pass.
7. Bump fork `Cargo.toml` version to `0.6.0-beta.17+pattern.1`.
8. Wire pattern's workspace via path dep: `genai = { path = "../rust-genai" }`. Fork is a flat crate (not a workspace); root `Cargo.toml` declares `name = "genai"`.
9. After end-to-end pattern-side validation, force-push `rebase/pattern-v3-foundation` to `origin`. Preserve pre-v3 fork history by tagging `pre-v3-fork-tip` at the current `origin/main` commit before any `main` move.
