import { beforeEach, describe, expect, it, vi } from 'vitest';

const addon = {
  setLogLevel: vi.fn(),
  setLogCallback: vi.fn(),
};

const initSpy = vi.fn(async function (this: { initialized?: boolean }) {
  this.initialized = true;
});

const buildVisionPrompt = vi.fn(async () => ({ text: 'prompt', imageBuffers: [] }));
const downloadModel = vi.fn(async (url: string) => `/cache/${encodeURIComponent(url)}`);

vi.mock('local-llm-js-core', () => {
  class BaseLocalLLM {
    options: unknown;
    adapter: unknown;
    _model: unknown = { nativeHandle: 'model-handle' };
    _context: unknown = { nativeHandle: 'context-handle' };
    _modelName = 'local';

    constructor(options: unknown, adapter: unknown) {
      this.options = options;
      this.adapter = adapter;
    }

    async init(): Promise<void> {
      await initSpy.call(this);
    }

    ensureContext(): void {}
  }

  class LocalLLMProvider {
    args: unknown[];

    constructor(...args: unknown[]) {
      this.args = args;
    }
  }

  return {
    BaseLocalLLM,
    ChatCompletions: class ChatCompletions {},
    LocalLLMProvider,
    Model: {
      load: vi.fn(async (path: string, options: unknown) => ({ path, options })),
    },
    getNativeAddon: () => addon,
  };
});

vi.mock('../src/model-manager', () => ({
  ModelManager: class ModelManager {
    downloadModel = downloadModel;
  },
}));

vi.mock('../src/vision', () => ({
  buildVisionPrompt,
}));

describe('LocalLLM wrapper', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('creates and initializes an instance', async () => {
    const { LocalLLM } = await import('../src/local-llm');

    const instance = await LocalLLM.create({ model: 'user/repo/model.gguf' });
    expect(instance).toBeInstanceOf(LocalLLM);
    expect(initSpy).toHaveBeenCalledOnce();
  });

  it('preloads local paths without downloading', async () => {
    const { LocalLLM } = await import('../src/local-llm');

    await expect(LocalLLM.preload('./local-model.gguf')).resolves.toBe('./local-model.gguf');
    expect(downloadModel).not.toHaveBeenCalled();
  });

  it('preloads remote models and projectors through the model manager', async () => {
    const { LocalLLM } = await import('../src/local-llm');

    const result = await LocalLLM.preload('user/repo/model.gguf', {
      projector: 'user/repo/projector.gguf',
    });

    expect(result).toContain('/cache/');
    expect(downloadModel).toHaveBeenCalledTimes(2);
  });

  it('maps log levels and native callbacks', async () => {
    const { LocalLLM } = await import('../src/local-llm');
    const callback = vi.fn();

    LocalLLM.setLogCallback(callback, 'warn');

    expect(addon.setLogLevel).toHaveBeenCalledWith(3);
    expect(addon.setLogCallback).toHaveBeenCalledTimes(1);

    const nativeCallback = addon.setLogCallback.mock.calls[0]?.[0] as ((level: number, text: string) => void);
    nativeCallback(4, 'native-error');
    expect(callback).toHaveBeenCalledWith('error', 'native-error');
  });

  it('builds language models with the RN vision prompt builder', async () => {
    const { LocalLLM } = await import('../src/local-llm');

    const instance = new LocalLLM({ model: './local.gguf' });
    const provider = instance.languageModel('smoke');

    expect(provider).toBeDefined();
    expect(buildVisionPrompt).not.toHaveBeenCalled();
  });
});
