import { beforeEach, describe, expect, it, vi } from 'vitest';
import { LocalLLMErrorCode } from '../src/errors';

const startStream = vi.fn();
const stopStream = vi.fn();

let tokenHandler: ((event: any) => void) | null = null;

const addListener = vi.fn((eventName: string, handler: (event: any) => void) => {
  if (eventName === 'onToken') tokenHandler = handler;
  return { remove: vi.fn(() => { tokenHandler = null; }) };
});

class MockNativeEventEmitter {
  addListener = addListener;
}

vi.mock('react-native', () => ({
  NativeEventEmitter: MockNativeEventEmitter,
  NativeModules: { LocalLLM: {} },
}));

const nativeMock = {
  backendInfo: vi.fn(() => 'test'),
  backendVersion: vi.fn(() => '0.0.1'),
  apiVersion: vi.fn(() => 1),
  loadModel: vi.fn(async () => 'model-1'),
  getModelSize: vi.fn(() => 1000),
  freeModel: vi.fn(),
  createContext: vi.fn(() => 'ctx-1'),
  getContextSize: vi.fn(() => 2048),
  freeContext: vi.fn(),
  warmup: vi.fn(),
  kvCacheClear: vi.fn(),
  startStream,
  stopStream,
  generate: vi.fn(async () => 'result'),
  loadProjector: vi.fn(() => 'mtmd-1'),
  supportVision: vi.fn(() => true),
  freeMtmdContext: vi.fn(),
  generateVision: vi.fn(async () => 'vision-result'),
  startStreamVision: vi.fn(),
  tokenize: vi.fn(() => [1, 2, 3]),
  detokenize: vi.fn(() => 'hello'),
  applyChatTemplate: vi.fn(() => '<template>'),
  jsonSchemaToGrammar: vi.fn(() => 'root ::= "test"'),
  getEmbeddingDimension: vi.fn(() => 384),
  createEmbeddingContext: vi.fn(() => 'emb-1'),
  embed: vi.fn(() => [0.1, 0.2]),
  embedBatch: vi.fn(() => [[0.1], [0.2]]),
  startBatch: vi.fn(),
  quantize: vi.fn(),
  setLogLevel: vi.fn(),
  enableLogEvents: vi.fn(),
  getPerf: vi.fn(() => ({ promptTokensPerSec: 100 })),
  optimalThreadCount: vi.fn(() => 4),
  benchmark: vi.fn(() => ({
    promptTokensPerSec: 100,
    generatedTokensPerSec: 50,
    ttftMs: 200,
    totalMs: 1000,
    iterations: 5,
  })),
  getDeviceCapabilities: vi.fn(() => ({
    totalRAM: 8 * 1024 * 1024 * 1024,
    availableRAM: 6 * 1024 * 1024 * 1024,
    gpuName: 'Apple GPU',
    isLowPowerMode: false,
  })),
  downloadModel: vi.fn(),
  cancelDownload: vi.fn(),
  getModelStoragePath: vi.fn(() => '/models'),
  fileExists: vi.fn(() => true),
  getFileSize: vi.fn(() => 1000),
  readTextFile: vi.fn(() => null),
  writeTextFile: vi.fn(),
  removePath: vi.fn(),
};

vi.mock('../src/NativeLocalLLM', () => ({
  default: nativeMock,
}));

describe('native-bridge streaming edge cases', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    tokenHandler = null;
  });

  it('handles error events during streaming with typed LocalLLMError', async () => {
    const { createReactNativeAddon } = await import('../src/native-bridge');
    const addon = createReactNativeAddon();

    const model = await addon.loadModel('/test.gguf', {});
    const ctx = addon.createContext(model, {});

    const callback = vi.fn();
    addon.inferStream(model, ctx, 'test prompt', {}, callback);

    // Simulate error event from native
    tokenHandler?.({ contextId: (ctx as any).__id, error: 'Out of context', done: false });

    expect(callback).toHaveBeenCalledOnce();
    const [err] = callback.mock.calls[0];
    expect(err).toBeInstanceOf(Error);
    expect(err.code).toBe(LocalLLMErrorCode.STREAM_FAILED);
    expect(err.message).toBe('Out of context');
  });

  it('handles immediate done without tokens', async () => {
    const { createReactNativeAddon } = await import('../src/native-bridge');
    const addon = createReactNativeAddon();

    const model = await addon.loadModel('/test.gguf', {});
    const ctx = addon.createContext(model, {});

    const callback = vi.fn();
    addon.inferStream(model, ctx, '', {}, callback);

    // Simulate immediate completion
    tokenHandler?.({ contextId: (ctx as any).__id, done: true });

    expect(callback).toHaveBeenCalledOnce();
    expect(callback).toHaveBeenCalledWith(null, null);
  });

  it('ignores events for different context IDs', async () => {
    const { createReactNativeAddon } = await import('../src/native-bridge');
    const addon = createReactNativeAddon();

    const model = await addon.loadModel('/test.gguf', {});
    const ctx = addon.createContext(model, {});

    const callback = vi.fn();
    addon.inferStream(model, ctx, 'test', {}, callback);

    // Simulate event for a different context
    tokenHandler?.({ contextId: 'other-context-id', token: 'wrong', done: false });

    expect(callback).not.toHaveBeenCalled();
  });

  it('raises typed error for benchmark failures', async () => {
    nativeMock.benchmark.mockReturnValueOnce({
      promptTokensPerSec: 0,
      generatedTokensPerSec: 0,
      ttftMs: 0,
      totalMs: 0,
      iterations: 0,
      error: 'Context too small',
    });

    const { createReactNativeAddon } = await import('../src/native-bridge');
    const addon = createReactNativeAddon();

    const model = await addon.loadModel('/test.gguf', {});
    const ctx = addon.createContext(model, {});

    try {
      addon.benchmark(model, ctx, {});
      expect.unreachable('Should have thrown');
    } catch (e: any) {
      expect(e.code).toBe(LocalLLMErrorCode.INFERENCE_FAILED);
      expect(e.message).toBe('Context too small');
    }
  });

  it('stopStream can be called before the first token arrives', async () => {
    const { createReactNativeAddon } = await import('../src/native-bridge');
    const addon = createReactNativeAddon();

    const model = await addon.loadModel('/test.gguf', {});
    const ctx = addon.createContext(model, {});

    const callback = vi.fn();
    addon.inferStream(model, ctx, 'test', {}, callback);

    // Stop immediately before any token events arrive
    addon.stopStream(ctx);
    expect(stopStream).toHaveBeenCalledWith((ctx as any).__id);

    // Native sends done after cancel is processed
    tokenHandler?.({ contextId: (ctx as any).__id, done: true });

    expect(callback).toHaveBeenCalledOnce();
    expect(callback).toHaveBeenCalledWith(null, null);
  });

  it('freeModel during streaming does not crash — error event is delivered', async () => {
    const { createReactNativeAddon } = await import('../src/native-bridge');
    const addon = createReactNativeAddon();

    const model = await addon.loadModel('/test.gguf', {});
    const ctx = addon.createContext(model, {});

    const callback = vi.fn();
    addon.inferStream(model, ctx, 'test', {}, callback);

    // Simulate: user frees the model while streaming is active
    addon.freeModel(model);
    expect(nativeMock.freeModel).toHaveBeenCalledWith((model as any).__id);

    // Native layer detects the freed model and sends an error event
    tokenHandler?.({ contextId: (ctx as any).__id, error: 'Model was freed', done: false });

    expect(callback).toHaveBeenCalledOnce();
    const [err] = callback.mock.calls[0];
    expect(err.code).toBe(LocalLLMErrorCode.STREAM_FAILED);
  });

  it('delivers tokens normally then completes', async () => {
    const { createReactNativeAddon } = await import('../src/native-bridge');
    const addon = createReactNativeAddon();

    const model = await addon.loadModel('/test.gguf', {});
    const ctx = addon.createContext(model, {});

    const callback = vi.fn();
    addon.inferStream(model, ctx, 'Hello', {}, callback);

    const ctxId = (ctx as any).__id;
    tokenHandler?.({ contextId: ctxId, token: 'Hi', done: false });
    tokenHandler?.({ contextId: ctxId, token: ' there', done: false });
    tokenHandler?.({ contextId: ctxId, done: true });

    expect(callback).toHaveBeenCalledTimes(3);
    expect(callback.mock.calls[0]).toEqual([null, 'Hi']);
    expect(callback.mock.calls[1]).toEqual([null, ' there']);
    expect(callback.mock.calls[2]).toEqual([null, null]);
  });

  it('forwards chat template and schema conversion calls unchanged', async () => {
    const { createReactNativeAddon } = await import('../src/native-bridge');
    const addon = createReactNativeAddon();

    const model = await addon.loadModel('/test.gguf', {});
    const messages = [{ role: 'user', content: 'hi' }];
    const schema = '{"type":"object","properties":{"answer":{"type":"string"}}}';

    expect(addon.applyChatTemplate(model, messages, true)).toBe('<template>');
    expect(addon.jsonSchemaToGrammar(schema)).toBe('root ::= "test"');

    expect(nativeMock.applyChatTemplate).toHaveBeenCalledWith('model-1', messages, true);
    expect(nativeMock.jsonSchemaToGrammar).toHaveBeenCalledWith(schema);
  });
});
