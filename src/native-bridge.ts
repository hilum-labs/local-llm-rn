import NativeLocalLLM from './NativeLocalLLM';
import { NativeEventEmitter, NativeModules } from 'react-native';
import type { NativeAddon, NativeModel, NativeContext, NativeMtmdContext } from 'local-llm-js-core/native';
import { getDeviceCapabilities } from './device';
import { LocalLLMError, LocalLLMErrorCode } from './errors';

const emitter = new NativeEventEmitter(NativeModules.LocalLLM);

// ── Power-aware defaults ────────────────────────────────────────────────────

function getOptimizedContextOptions(options?: Record<string, any>): Record<string, any> {
  const opts = { ...(options ?? {}) };

  try {
    const caps = getDeviceCapabilities();
    const availableGB = caps.availableRAM / (1024 * 1024 * 1024);

    // 3d: Power-aware thread management — halve threads in low power mode
    if (caps.isLowPowerMode && !opts.n_threads) {
      // Use 2 threads in low power mode (engine defaults to all cores otherwise)
      opts.n_threads = 2;
    }

    // 3b: Smart n_batch defaults based on available RAM
    if (!opts.n_batch) {
      if (availableGB >= 6) {
        opts.n_batch = 512;   // Plenty of RAM — large batches
      } else if (availableGB >= 3) {
        opts.n_batch = 256;   // Mid-range — moderate batches
      } else {
        opts.n_batch = 128;   // Low RAM — small batches to avoid spikes
      }
    }
  } catch {
    // Device capabilities unavailable — use engine defaults
  }

  return opts;
}

function getOptimalThreadCount(): number {
  try {
    return NativeLocalLLM.optimalThreadCount();
  } catch {
    try {
      const caps = getDeviceCapabilities();
      const availableGB = caps.availableRAM / (1024 * 1024 * 1024);

      if (caps.isLowPowerMode) return 2;
      if (availableGB >= 6) return 4;
      if (availableGB >= 3) return 3;
      return 2;
    } catch {
      return 2;
    }
  }
}

type BrandedId<B extends string> = { __id: string; __brand: B };
const wrapId = <B extends string>(id: string, brand: B) => ({ __id: id, __brand: brand } as unknown);
const unwrapId = (obj: any): string => obj.__id;

function toBase64(bytes: Uint8Array): string {
  const runtime = globalThis as {
    Buffer?: { from(input: Uint8Array): { toString(encoding: 'base64'): string } };
    btoa?: (input: string) => string;
  };

  if (runtime.Buffer) {
    return runtime.Buffer.from(bytes).toString('base64');
  }

  if (runtime.btoa) {
    let binary = '';
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
    }
    return runtime.btoa(binary);
  }

  throw new LocalLLMError(LocalLLMErrorCode.VISION_FAILED, 'Base64 encoding is not available in this runtime');
}

export function createReactNativeAddon(): NativeAddon {
  return {
    backendInfo: () => NativeLocalLLM.backendInfo(),
    backendVersion: () => NativeLocalLLM.backendVersion(),
    apiVersion: () => NativeLocalLLM.apiVersion(),

    loadModel: (path, options) => {
      const promise = NativeLocalLLM.loadModel(path, options ?? {});
      return promise.then((id) => wrapId(id, 'NativeModel')) as any;
    },
    loadModelFromBuffer: () => { throw new LocalLLMError(LocalLLMErrorCode.MODEL_LOAD_FAILED, 'loadModelFromBuffer is not available on React Native'); },
    getModelSize: (model) => NativeLocalLLM.getModelSize(unwrapId(model)),
    freeModel: (model) => NativeLocalLLM.freeModel(unwrapId(model)),

    createContext: (model, options) => {
      const optimized = getOptimizedContextOptions(options as Record<string, any>);
      // Convert draft_model handle to its native string ID for the bridge
      if (optimized.draft_model && typeof optimized.draft_model === 'object') {
        optimized.draft_model_id = unwrapId(optimized.draft_model as any);
        delete optimized.draft_model;
      }
      const id = NativeLocalLLM.createContext(unwrapId(model), optimized);
      return wrapId(id, 'NativeContext') as NativeContext;
    },
    getContextSize: (ctx) => NativeLocalLLM.getContextSize(unwrapId(ctx)),
    freeContext: (ctx) => NativeLocalLLM.freeContext(unwrapId(ctx)),

    warmup: (model, ctx) => {
      NativeLocalLLM.warmup(unwrapId(model), unwrapId(ctx));
    },

    kvCacheClear: (ctx, fromPos) => NativeLocalLLM.kvCacheClear(unwrapId(ctx), fromPos),
    stopStream: (ctx) => NativeLocalLLM.stopStream(unwrapId(ctx)),

    getPerf: (ctx) => NativeLocalLLM.getPerf(unwrapId(ctx)) as any,

    tokenize: (model, text, addSpecial, parseSpecial) => {
      const arr = NativeLocalLLM.tokenize(unwrapId(model), text, addSpecial ?? true, parseSpecial ?? false);
      return new Int32Array(arr);
    },
    detokenize: (model, tokens) => {
      const plain = tokens instanceof Int32Array ? Array.from(tokens) : tokens;
      return NativeLocalLLM.detokenize(unwrapId(model), plain);
    },
    applyChatTemplate: (model, messages, addAssistant) =>
      NativeLocalLLM.applyChatTemplate(unwrapId(model), messages, addAssistant ?? true),

    inferSync: (model, ctx, prompt, options) =>
      NativeLocalLLM.generate(unwrapId(model), unwrapId(ctx), prompt, options ?? {}) as any,

    inferStream: (model, ctx, prompt, options, callback) => {
      const ctxId = unwrapId(ctx);
      const sub = emitter.addListener('onToken', (event) => {
        if (event.contextId !== ctxId) return;
        if (event.error) { callback(new LocalLLMError(LocalLLMErrorCode.STREAM_FAILED, event.error), null); sub.remove(); return; }
        if (event.done) { callback(null, null); sub.remove(); return; }
        callback(null, event.token);
      });
      NativeLocalLLM.startStream(unwrapId(model), ctxId, prompt, options ?? {});
    },

    createMtmdContext: (model, projectorPath, options) => {
      const id = NativeLocalLLM.loadProjector(unwrapId(model), projectorPath, options ?? {});
      return wrapId(id, 'NativeMtmdContext') as NativeMtmdContext;
    },
    supportVision: (ctx) => NativeLocalLLM.supportVision(unwrapId(ctx)),
    freeMtmdContext: (ctx) => NativeLocalLLM.freeMtmdContext(unwrapId(ctx)),

    inferSyncVision: async (model, ctx, mtmdCtx, prompt, imageBuffers, options) => {
      const base64s = imageBuffers.map((buf) => toBase64(buf));
      return NativeLocalLLM.generateVision(
        unwrapId(model), unwrapId(ctx), unwrapId(mtmdCtx), prompt, base64s, options ?? {},
      );
    },

    inferStreamVision: (model, ctx, mtmdCtx, prompt, imageBuffers, options, callback) => {
      const ctxId = unwrapId(ctx);
      const base64s = imageBuffers.map((buf) => toBase64(buf));
      const sub = emitter.addListener('onToken', (event) => {
        if (event.contextId !== ctxId) return;
        if (event.error) { callback(new LocalLLMError(LocalLLMErrorCode.VISION_FAILED, event.error), null); sub.remove(); return; }
        if (event.done) { callback(null, null); sub.remove(); return; }
        callback(null, event.token);
      });
      NativeLocalLLM.startStreamVision(
        unwrapId(model), ctxId, unwrapId(mtmdCtx), prompt, base64s, options ?? {},
      );
    },

    jsonSchemaToGrammar: (schemaJson) => NativeLocalLLM.jsonSchemaToGrammar(schemaJson),

    getEmbeddingDimension: (model) => NativeLocalLLM.getEmbeddingDimension(unwrapId(model)),
    createEmbeddingContext: (model, options) => {
      const id = NativeLocalLLM.createEmbeddingContext(unwrapId(model), options ?? {});
      return wrapId(id, 'NativeContext') as NativeContext;
    },
    embed: (ctx, model, tokens) => {
      const plain = tokens instanceof Int32Array ? Array.from(tokens) : tokens;
      return new Float32Array(NativeLocalLLM.embed(unwrapId(ctx), unwrapId(model), plain));
    },
    embedBatch: (ctx, model, tokenArrays) => {
      const plainArrays = tokenArrays.map((t) => (t instanceof Int32Array ? Array.from(t) : t));
      const results = NativeLocalLLM.embedBatch(unwrapId(ctx), unwrapId(model), plainArrays) as number[][];
      return results.map((r) => new Float32Array(r));
    },

    inferBatch: (model, ctx, prompts, options, callback) => {
      const ctxId = unwrapId(ctx);
      let completedCount = 0;
      const sub = emitter.addListener('onBatchToken', (event) => {
        if (event.contextId !== ctxId) return;
        if (event.error) { callback(new LocalLLMError(LocalLLMErrorCode.INFERENCE_FAILED, event.error), null, event.seqIndex, null); return; }
        if (event.done) {
          callback(null, null, event.seqIndex, event.finishReason ?? null);
          completedCount++;
          if (completedCount >= prompts.length) sub.remove();
          return;
        }
        callback(null, event.token, event.seqIndex, null);
      });
      NativeLocalLLM.startBatch(unwrapId(model), ctxId, prompts, options as any);
    },

    quantize: (inputPath, outputPath, options, callback) => {
      const sub = emitter.addListener('onQuantizeComplete', (event) => {
        sub.remove();
        callback(event.error ? new LocalLLMError(LocalLLMErrorCode.QUANTIZE_FAILED, event.error) : null);
      });
      NativeLocalLLM.quantize(inputPath, outputPath, options);
    },

    setLogCallback: (() => {
      let logSub: { remove(): void } | null = null;
      return (cb: ((level: number, text: string) => void) | null) => {
        if (logSub) { logSub.remove(); logSub = null; }
        if (cb) {
          logSub = emitter.addListener('onLog', (event) => cb(event.level, event.text));
          NativeLocalLLM.enableLogEvents(true);
        } else {
          NativeLocalLLM.enableLogEvents(false);
        }
      };
    })(),
    setLogLevel: (level) => NativeLocalLLM.setLogLevel(level),
    optimalThreadCount: () => getOptimalThreadCount(),
    benchmark: (model, ctx, options) => {
      const result = NativeLocalLLM.benchmark(unwrapId(model), unwrapId(ctx), options ?? {}) as {
        promptTokensPerSec: number;
        generatedTokensPerSec: number;
        ttftMs: number;
        totalMs: number;
        iterations: number;
        error?: string;
      };
      if (result.error) {
        throw new LocalLLMError(LocalLLMErrorCode.INFERENCE_FAILED, result.error);
      }
      return result;
    },
  };
}
