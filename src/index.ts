// Override native loader before anything else imports it
import { setNativeAddon } from 'local-llm-js-core/native';
import { createReactNativeAddon } from './native-bridge';
setNativeAddon(createReactNativeAddon());

// Re-export the shared runtime surface from core where available.
export { Model, InferenceContext, EmbeddingContext, ModelPool, ChatCompletions, Embeddings, LocalLLMProvider } from 'local-llm-js-core';
export { LocalLLM, ModelManager } from './local-llm';
export type {
  LocalLLMOptions,
  ChatMessage, ContentPart, ComputeMode, GenerateOptions, FlashAttentionMode,
  KvCacheType, ContextOverflowStrategy, ContextOverflowConfig, EmbeddingPoolingType,
  ResponseFormat, ChatCompletionTool, ChatCompletionToolChoice,
  ModelOptions, ContextOptions, EmbeddingContextOptions, QuantizationType, QuantizeOptions,
  BatchResult, InferenceMetrics, BenchmarkOptions, BenchmarkResult,
} from 'local-llm-js-core';
export type { DownloadOptions } from './local-llm';

// RN-specific exports
export { getDeviceCapabilities, canRunModel, recommendQuantization } from './device';
export type { DeviceCapabilities } from './device';
export { createNativeDownloader } from './rn-downloader';
export type { DownloadAdapter } from 'local-llm-js-core';
export { LocalLLMError, LocalLLMErrorCode } from './errors';
