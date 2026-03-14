import { describe, expect, it, vi } from 'vitest';

const nativeModule = {
  backendInfo: vi.fn(() => 'backend-info'),
  backendVersion: vi.fn(() => 'backend-version'),
  apiVersion: vi.fn(() => 0x010000),
  loadModel: vi.fn(async () => 'model-1'),
  getModelSize: vi.fn(() => 1),
  freeModel: vi.fn(),
  createContext: vi.fn(() => 'ctx-1'),
  getContextSize: vi.fn(() => 1),
  freeContext: vi.fn(),
  warmup: vi.fn(),
  kvCacheClear: vi.fn(),
  startStream: vi.fn(),
  stopStream: vi.fn(),
  generate: vi.fn(async () => 'hello'),
  loadProjector: vi.fn(() => 'mtmd-1'),
  supportVision: vi.fn(() => true),
  freeMtmdContext: vi.fn(),
  generateVision: vi.fn(async () => 'vision'),
  startStreamVision: vi.fn(),
  tokenize: vi.fn(() => [1, 2]),
  detokenize: vi.fn(() => 'hello'),
  applyChatTemplate: vi.fn(() => '<prompt>'),
  jsonSchemaToGrammar: vi.fn(() => 'root ::= "ok"'),
  getEmbeddingDimension: vi.fn(() => 4),
  createEmbeddingContext: vi.fn(() => 'emb-1'),
  embed: vi.fn(() => [0.1, 0.2]),
  embedBatch: vi.fn(() => [[0.1], [0.2]]),
  startBatch: vi.fn(),
  quantize: vi.fn(),
  setLogLevel: vi.fn(),
  enableLogEvents: vi.fn(),
  getPerf: vi.fn(() => ({ promptTokensPerSec: 1 })),
  optimalThreadCount: vi.fn(() => 3),
  benchmark: vi.fn(() => ({
    promptTokensPerSec: 1,
    generatedTokensPerSec: 1,
    ttftMs: 1,
    totalMs: 1,
    iterations: 1,
  })),
};

class MockNativeEventEmitter {
  addListener = vi.fn(() => ({ remove: vi.fn() }));
}

vi.mock('react-native', () => ({
  NativeEventEmitter: MockNativeEventEmitter,
  NativeModules: { LocalLLM: {} },
}));

vi.mock('../src/NativeLocalLLM', () => ({
  default: nativeModule,
}));

describe('native contract smoke', () => {
  it('exposes the bridge methods the package relies on', async () => {
    const { createReactNativeAddon } = await import('../src/native-bridge');
    const addon = createReactNativeAddon() as Record<string, unknown>;

    const expectedMethods = [
      'apiVersion',
      'applyChatTemplate',
      'backendInfo',
      'backendVersion',
      'benchmark',
      'createContext',
      'createEmbeddingContext',
      'createMtmdContext',
      'detokenize',
      'embed',
      'embedBatch',
      'freeContext',
      'freeModel',
      'freeMtmdContext',
      'getEmbeddingDimension',
      'getModelSize',
      'getPerf',
      'inferBatch',
      'inferStream',
      'inferStreamVision',
      'inferSync',
      'inferSyncVision',
      'jsonSchemaToGrammar',
      'kvCacheClear',
      'loadModel',
      'optimalThreadCount',
      'quantize',
      'setLogCallback',
      'setLogLevel',
      'stopStream',
      'supportVision',
      'tokenize',
      'warmup',
    ];

    for (const method of expectedMethods) {
      expect(typeof addon[method]).toBe('function');
    }
  });

  it('maps draft_model handles to draft_model_id for the native bridge', async () => {
    const { createReactNativeAddon } = await import('../src/native-bridge');
    const addon = createReactNativeAddon();

    const model = await addon.loadModel('/tmp/model.gguf', {});
    addon.createContext(model, {
      n_ctx: 256,
      draft_model: model,
    });

    expect(nativeModule.createContext).toHaveBeenCalledWith('model-1', {
      n_ctx: 256,
      draft_model_id: 'model-1',
    });
  });
});
