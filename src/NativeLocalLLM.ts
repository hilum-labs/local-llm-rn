import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';
import type { UnsafeObject } from 'react-native/Libraries/Types/CodegenTypes';

export interface Spec extends TurboModule {
  backendInfo(): string;
  backendVersion(): string;
  apiVersion(): number;

  loadModel(path: string, options: Object): Promise<string>;
  getModelSize(modelId: string): number;
  freeModel(modelId: string): void;

  createContext(modelId: string, options: Object): string;
  getContextSize(contextId: string): number;
  freeContext(contextId: string): void;

  warmup(modelId: string, contextId: string): Promise<void>;
  kvCacheClear(contextId: string, fromPos: number): void;

  tokenize(modelId: string, text: string, addSpecial: boolean, parseSpecial: boolean): number[];
  detokenize(modelId: string, tokens: number[]): string;
  applyChatTemplate(modelId: string, messages: UnsafeObject[], addAssistant: boolean): string;

  generate(modelId: string, contextId: string, prompt: string, options: Object): Promise<string>;
  startStream(modelId: string, contextId: string, prompt: string, options: Object): void;
  stopStream(contextId: string): void;

  loadProjector(modelId: string, path: string, options: Object): string;
  supportVision(mtmdId: string): boolean;
  freeMtmdContext(mtmdId: string): void;
  generateVision(
    modelId: string, contextId: string, mtmdId: string,
    prompt: string, imageBase64s: string[], options: Object,
  ): Promise<string>;
  startStreamVision(
    modelId: string, contextId: string, mtmdId: string,
    prompt: string, imageBase64s: string[], options: Object,
  ): void;

  getPerf(contextId: string): UnsafeObject;
  optimalThreadCount(): number;
  benchmark(
    modelId: string,
    contextId: string,
    options: Object,
  ): {
    promptTokensPerSec: number;
    generatedTokensPerSec: number;
    ttftMs: number;
    totalMs: number;
    iterations: number;
    error?: string;
  };

  jsonSchemaToGrammar(schemaJson: string): string;

  getEmbeddingDimension(modelId: string): number;
  createEmbeddingContext(modelId: string, options: Object): string;
  embed(contextId: string, modelId: string, tokens: number[]): number[];
  embedBatch(contextId: string, modelId: string, tokenArrays: number[][]): number[][];

  startBatch(modelId: string, contextId: string, prompts: string[], options: Object): void;

  quantize(inputPath: string, outputPath: string, options: Object): void;

  setLogLevel(level: number): void;
  enableLogEvents(enabled: boolean): void;

  downloadModel(url: string, destPath: string): void;
  resumeDownload(url: string, destPath: string): void;
  cancelDownload(url: string): void;

  getDeviceCapabilities(): Object;
  getModelStoragePath(): string;
  fileExists(path: string): boolean;
  getFileSize(path: string): number;
  readTextFile(path: string): string | null;
  writeTextFile(path: string, content: string): void;
  removePath(path: string): void;
  sha256File(path: string): Promise<string>;
}

export default TurboModuleRegistry.getEnforcing<Spec>('LocalLLM');
