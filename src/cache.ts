import { extractModelFilename, hashModelUrl, resolveModelUrl, type CacheEntry, type CacheIndex, type ModelRegistry } from 'local-llm-js-core';
import NativeLocalLLM from './NativeLocalLLM';
import { LocalLLMError, LocalLLMErrorCode } from './errors';

export type { CacheEntry } from 'local-llm-js-core';

function joinPath(...parts: string[]): string {
  return parts
    .filter(Boolean)
    .map((part, index) => {
      if (index === 0) {
        return part.replace(/\/+$/, '');
      }
      return part.replace(/^\/+/, '').replace(/\/+$/, '');
    })
    .join('/');
}

export class ModelCache implements ModelRegistry {
  readonly cacheDir: string;
  private indexPath: string;

  constructor(cacheDir?: string) {
    this.cacheDir = cacheDir ?? NativeLocalLLM.getModelStoragePath();
    this.indexPath = joinPath(this.cacheDir, 'index.json');
  }

  /** Ensure a path is within the cache directory to prevent path traversal. */
  private assertSafePath(path: string): void {
    const normalized = path.replace(/\/+/g, '/').replace(/\/\.\.\//g, '/');
    if (!normalized.startsWith(this.cacheDir)) {
      throw new LocalLLMError(
        LocalLLMErrorCode.INVALID_PATH,
        `Path "${path}" is outside the cache directory "${this.cacheDir}"`,
      );
    }
  }

  static hashUrl(url: string): string {
    return hashModelUrl(url);
  }

  async getCachedModel(url: string): Promise<CacheEntry | null> {
    const resolvedUrl = resolveModelUrl(url);
    const index = this.readIndex();
    const key = ModelCache.hashUrl(resolvedUrl);
    const entry = index.models[key];
    if (!entry) return null;

    if (!NativeLocalLLM.fileExists(entry.path)) {
      delete index.models[key];
      this.writeIndex(index);
      return null;
    }

    entry.lastUsedAt = new Date().toISOString();
    this.writeIndex(index);
    return entry;
  }

  async cacheModel(url: string, filePath: string, size: number): Promise<CacheEntry> {
    this.assertSafePath(filePath);
    const resolvedUrl = resolveModelUrl(url);
    const index = this.readIndex();
    const key = ModelCache.hashUrl(resolvedUrl);
    const now = new Date().toISOString();
    const entry: CacheEntry = {
      url: resolvedUrl,
      path: filePath,
      size,
      downloadedAt: now,
      lastUsedAt: now,
    };
    index.models[key] = entry;
    this.writeIndex(index);
    return entry;
  }

  async listModels(): Promise<CacheEntry[]> {
    const index = this.readIndex();
    return Object.values(index.models).filter((entry) => NativeLocalLLM.fileExists(entry.path));
  }

  async removeModel(url: string): Promise<boolean> {
    const resolvedUrl = resolveModelUrl(url);
    const index = this.readIndex();
    const key = ModelCache.hashUrl(resolvedUrl);
    if (!index.models[key]) return false;

    const dir = this.modelDir(resolvedUrl);
    this.assertSafePath(dir);
    NativeLocalLLM.removePath(dir);
    delete index.models[key];
    this.writeIndex(index);
    return true;
  }

  modelDir(url: string): string {
    return joinPath(this.cacheDir, ModelCache.hashUrl(resolveModelUrl(url)));
  }

  targetPath(url: string): string {
    const resolvedUrl = resolveModelUrl(url);
    return joinPath(this.modelDir(resolvedUrl), extractModelFilename(resolvedUrl));
  }

  private readIndex(): CacheIndex {
    const raw = NativeLocalLLM.readTextFile(this.indexPath);
    if (!raw) {
      return { models: {} };
    }

    try {
      return JSON.parse(raw) as CacheIndex;
    } catch {
      const backupPath = this.indexPath + '.corrupt.' + Date.now();
      NativeLocalLLM.writeTextFile(backupPath, raw);
      NativeLocalLLM.removePath(this.indexPath);
      console.warn(
        `[local-llm-rn] Cache index was corrupt and has been reset. Backup saved to: ${backupPath}`,
      );
      return { models: {} };
    }
  }

  private writeIndex(index: CacheIndex): void {
    NativeLocalLLM.writeTextFile(this.indexPath, JSON.stringify(index, null, 2) + '\n');
  }
}
