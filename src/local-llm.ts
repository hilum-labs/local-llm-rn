import {
  BaseLocalLLM,
  ChatCompletions,
  LocalLLMProvider,
  Model,
  getNativeAddon,
  type LocalLLMOptions,
  type LocalLLMRuntimeAdapter,
  type LogCallback,
  type LogLevel,
} from 'local-llm-js-core';
import { ModelManager } from './model-manager';
import { buildVisionPrompt } from './vision';
import { LocalLLMError, LocalLLMErrorCode } from './errors';

function isLocalPath(path: string): boolean {
  return (
    path.startsWith('/') ||
    path.startsWith('./') ||
    path.startsWith('../') ||
    path.startsWith('file://') ||
    path.startsWith('content://')
  );
}

const runtimeAdapter: LocalLLMRuntimeAdapter = {
  async loadModel(path, options) {
    return Model.load(path, options);
  },
  async resolveFilePath(path: string, options: LocalLLMOptions): Promise<string> {
    if (isLocalPath(path)) {
      return path;
    }

    const manager = new ModelManager(options.cacheDir);
    return manager.downloadModel(path, {
      onProgress: options.onProgress
        ? (_downloaded, _total, percent) => options.onProgress!(percent)
        : undefined,
    });
  },
  setLogCallback(callback: LogCallback | null, minLevel?: LogLevel): void {
    LocalLLM.setLogCallback(callback, minLevel);
  },
  createChatCompletions(model, context, modelName, config) {
    return new ChatCompletions(model, context, modelName, {
      ...config,
      visionPromptBuilder: config?.visionPromptBuilder ?? buildVisionPrompt,
    });
  },
};

export { ModelManager };
export type { DownloadOptions } from './model-manager';
export type { LocalLLMOptions } from 'local-llm-js-core';

/**
 * On-device LLM inference for React Native.
 *
 * Use {@link LocalLLM.create} to load a model and start chatting:
 * ```ts
 * const ai = await LocalLLM.create({ model: 'user/repo/model.gguf', compute: 'gpu' });
 * const res = await ai.chat.completions.create({ messages: [...], stream: true });
 * ```
 */
export class LocalLLM extends BaseLocalLLM {
  constructor(options: LocalLLMOptions) {
    super(options, runtimeAdapter);
  }

  /** Create and initialize a LocalLLM instance (downloads the model if needed). */
  static async create(options: LocalLLMOptions): Promise<LocalLLM> {
    const instance = new LocalLLM(options);
    await instance.init();
    return instance;
  }

  /**
   * Pre-download a model (and optional vision projector) without creating an inference context.
   * Returns the local file path to the downloaded model.
   */
  static async preload(
    model: string,
    options?: { cacheDir?: string; projector?: string; onProgress?: (percent: number) => void },
  ): Promise<string> {
    const manager = new ModelManager(options?.cacheDir);
    const progressCb = options?.onProgress
      ? (_downloaded: number, _total: number, percent: number) => options.onProgress!(percent)
      : undefined;

    const projectorPromise = options?.projector && !isLocalPath(options.projector)
      ? manager.downloadModel(options.projector, { onProgress: progressCb })
      : undefined;

    if (isLocalPath(model)) {
      await projectorPromise;
      return model;
    }

    const [modelPath] = await Promise.all(
      projectorPromise
        ? [manager.downloadModel(model, { onProgress: progressCb }), projectorPromise]
        : [manager.downloadModel(model, { onProgress: progressCb })],
    );

    return modelPath;
  }

  private static readonly LOG_LEVEL_MAP: Record<LogLevel, number> = {
    debug: 1,
    info: 2,
    warn: 3,
    error: 4,
  };

  private static readonly LOG_LEVEL_REVERSE: Record<number, LogLevel> = {
    1: 'debug',
    2: 'info',
    3: 'warn',
    4: 'error',
  };

  /** Set a callback to receive engine log messages. Pass `null` to disable. */
  static setLogCallback(callback: LogCallback | null, minLevel: LogLevel = 'info'): void {
    const addon = getNativeAddon();
    if (callback) {
      addon.setLogLevel(LocalLLM.LOG_LEVEL_MAP[minLevel]);
      addon.setLogCallback((level: number, text: string) => {
        callback(LocalLLM.LOG_LEVEL_REVERSE[level] ?? 'info', text);
      });
    } else {
      addon.setLogCallback(null);
    }
  }

  /** Create a Vercel AI SDK-compatible language model provider. */
  languageModel(id?: string): LocalLLMProvider {
    if (!this._model) {
      throw new LocalLLMError(LocalLLMErrorCode.NOT_INITIALIZED, 'LocalLLM not initialized. Call await init() first, or use await LocalLLM.create().');
    }
    this.ensureContext();
    const modelId = id ?? this._modelName ?? 'local';
    return new LocalLLMProvider(this._model, this._context!, modelId, {
      visionPromptBuilder: buildVisionPrompt,
    });
  }
}
