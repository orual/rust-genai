use crate::ModelIden;
use crate::adapter::openai::OpenAIAdapter;
use crate::adapter::{Adapter, AdapterKind, ServiceType, WebRequestData};
use crate::chat::{ChatOptionsSet, ChatRequest, ChatResponse, ChatStreamResponse};
use crate::resolver::{AuthData, Endpoint};
use crate::webc::WebResponse;
use crate::{Result, ServiceTarget};
use reqwest::RequestBuilder;

pub struct OpenRouterAdapter;

/// Popular models available on OpenRouter.
/// OpenRouter supports 100+ models from various providers, but these are some commonly used ones.
/// Users can use any model by specifying its OpenRouter model ID.
// ~ newer on top when/if possible
pub(in crate::adapter) const MODELS: &[&str] = &[
	// -- Anthropic Claude models
    "anthropic/claude-opus-4.5",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5",
	"anthropic/claude-opus-4",
	"anthropic/claude-sonnet-4",
    "anthropic/claude-3.7-sonnet:thinking",
    "anthropic/claude-3.7-sonnet",
	"anthropic/claude-3.5-sonnet",
	"anthropic/claude-3.5-haiku",
	"anthropic/claude-3-opus",
	"anthropic/claude-3-sonnet",
	"anthropic/claude-3-haiku",
	// -- OpenAI models
	"openai/gpt-4o",
	"openai/gpt-4o-mini",
	"openai/gpt-4-turbo",
	"openai/gpt-4",
	"openai/o1-preview",
	"openai/o1-mini",
	// -- Google models
	"google/gemini-2.0-flash-exp",
	"google/gemini-exp-1206",
	"google/gemini-pro-1.5",
	"google/gemini-flash-1.5",
	// -- Meta Llama models
	"meta-llama/llama-3.3-70b-instruct",
	"meta-llama/llama-3.1-405b-instruct",
	"meta-llama/llama-3.1-70b-instruct",
	"meta-llama/llama-3.1-8b-instruct",
	// -- Mistral models
	"mistralai/mistral-large",
	"mistralai/mistral-medium",
	"mistralai/mixtral-8x22b-instruct",
	"mistralai/mixtral-8x7b-instruct",
	// -- DeepSeek models
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-v3.1-terminus",
    "deepseek/deepseek-chat-v3-0324",
	"deepseek/deepseek-chat",
	"deepseek/deepseek-r1",
    "deepseek/deepseek-r1-0528",
    // Moonshot models
    "moonshotai/kimi-k2-thinking",
    "moonshotai/kimi-k2-0905",
    // GLM models
    "z-ai/glm-4.7",
    "z-ai/glm-4.6",
    "z-ai/glm-4.5",
	// -- Other popular models
	"qwen/qwen-2.5-72b-instruct",
	"cohere/command-r-plus",
];

impl OpenRouterAdapter {
	/// Default environment variable name for OpenRouter API key
	pub const API_KEY_DEFAULT_ENV_NAME: &str = "OPENROUTER_API_KEY";
	/// Environment variable name for app URL (used for HTTP-Referer header)
	pub const APP_URL_ENV_NAME: &str = "OPENROUTER_APP_URL";
	/// Environment variable name for app title (used for X-Title header)
	pub const APP_TITLE_ENV_NAME: &str = "OPENROUTER_APP_TITLE";
}

// The OpenRouter API adapter is modeled after the OpenAI adapter, as OpenRouter uses an OpenAI-compatible API.
impl Adapter for OpenRouterAdapter {
	fn default_endpoint() -> Endpoint {
		const BASE_URL: &str = "https://openrouter.ai/api/v1/";
		Endpoint::from_static(BASE_URL)
	}

	fn default_auth() -> AuthData {
		AuthData::from_env(Self::API_KEY_DEFAULT_ENV_NAME)
	}

	async fn all_model_names(_kind: AdapterKind) -> Result<Vec<String>> {
		Ok(MODELS.iter().map(|s| s.to_string()).collect())
	}

	fn get_service_url(model: &ModelIden, service_type: ServiceType, endpoint: Endpoint) -> String {
		OpenAIAdapter::util_get_service_url(model, service_type, endpoint)
	}

	fn to_web_request_data(
		target: ServiceTarget,
		service_type: ServiceType,
		chat_req: ChatRequest,
		chat_options: ChatOptionsSet<'_, '_>,
	) -> Result<WebRequestData> {
		let mut web_request_data =
			OpenAIAdapter::util_to_web_request_data(target, service_type, chat_req, chat_options)?;

		// Add OpenRouter-specific headers from environment variables if set
		if let Ok(app_url) = std::env::var(Self::APP_URL_ENV_NAME) {
			web_request_data.headers.merge(("HTTP-Referer", app_url));
		}
		if let Ok(app_title) = std::env::var(Self::APP_TITLE_ENV_NAME) {
			web_request_data.headers.merge(("X-Title", app_title));
		}

		Ok(web_request_data)
	}

	fn to_chat_response(
		model_iden: ModelIden,
		web_response: WebResponse,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatResponse> {
		OpenAIAdapter::to_chat_response(model_iden, web_response, options_set)
	}

	fn to_chat_stream(
		model_iden: ModelIden,
		reqwest_builder: RequestBuilder,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatStreamResponse> {
		OpenAIAdapter::to_chat_stream(model_iden, reqwest_builder, options_set)
	}

	fn to_embed_request_data(
		_service_target: crate::ServiceTarget,
		_embed_req: crate::embed::EmbedRequest,
		_options_set: crate::embed::EmbedOptionsSet<'_, '_>,
	) -> Result<crate::adapter::WebRequestData> {
		// OpenRouter does support embeddings for some models, but implementation would require
		// additional testing. For now, we return not supported.
		Err(crate::Error::AdapterNotSupported {
			adapter_kind: crate::adapter::AdapterKind::OpenRouter,
			feature: "embeddings".to_string(),
		})
	}

	fn to_embed_response(
		_model_iden: crate::ModelIden,
		_web_response: crate::webc::WebResponse,
		_options_set: crate::embed::EmbedOptionsSet<'_, '_>,
	) -> Result<crate::embed::EmbedResponse> {
		Err(crate::Error::AdapterNotSupported {
			adapter_kind: crate::adapter::AdapterKind::OpenRouter,
			feature: "embeddings".to_string(),
		})
	}
}
