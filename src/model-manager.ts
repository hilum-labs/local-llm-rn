import { resolveModelUrl, type CacheEntry } from 'local-llm-js-core';
import NativeLocalLLM from './NativeLocalLLM';
import type { DownloadAdapter } from 'local-llm-js-core';
import { ModelCache } from './cache';
import { createNativeDownloader } from './rn-downloader';
import { LocalLLMError, LocalLLMErrorCode } from './errors';

export interface DownloadOptions {
  onProgress?: (downloaded: number, total: number, percent: number) => void;
  /** Expected file size in bytes. If set, the download is verified against this value. */
  expectedSize?: number;
  /**
   * Expected SHA256 hex digest of the downloaded file.
   * If set, the file is hashed after download and verified against this value.
   * Protects against corrupted or tampered GGUF files.
   */
  expectedSha256?: string;
}

/** Manages model downloads and local caching. */
export class ModelManager {
  private cache: ModelCache;
  private downloader: DownloadAdapter;

  constructor(cacheDir?: string, downloader?: DownloadAdapter) {
    this.cache = new ModelCache(cacheDir);
    this.downloader = downloader ?? createNativeDownloader();
  }

  /**
   * Download a model (or return its cached path if already downloaded).
   *
   * Pass `options.expectedSize` to verify the download wasn't truncated or corrupted.
   */
  async downloadModel(url: string, options?: DownloadOptions): Promise<string> {
    const resolvedUrl = resolveModelUrl(url);
    const cached = await this.cache.getCachedModel(resolvedUrl);
    if (cached) return cached.path;

    const targetPath = this.cache.targetPath(resolvedUrl);
    await this.downloader.download(resolvedUrl, targetPath, options?.onProgress);

    const size = NativeLocalLLM.getFileSize(targetPath);
    if (size === 0) {
      NativeLocalLLM.removePath(targetPath);
      throw new LocalLLMError(
        LocalLLMErrorCode.DOWNLOAD_INTEGRITY_MISMATCH,
        `Downloaded file is empty: ${resolvedUrl}`,
      );
    }
    if (options?.expectedSize && size !== options.expectedSize) {
      NativeLocalLLM.removePath(targetPath);
      throw new LocalLLMError(
        LocalLLMErrorCode.DOWNLOAD_INTEGRITY_MISMATCH,
        `Downloaded file size mismatch for ${resolvedUrl}: expected ${options.expectedSize} bytes, got ${size} bytes`,
      );
    }
    if (options?.expectedSha256) {
      const actualHash = await NativeLocalLLM.sha256File(targetPath);
      if (actualHash !== options.expectedSha256.toLowerCase()) {
        NativeLocalLLM.removePath(targetPath);
        throw new LocalLLMError(
          LocalLLMErrorCode.DOWNLOAD_INTEGRITY_MISMATCH,
          `SHA256 mismatch for ${resolvedUrl}: expected ${options.expectedSha256}, got ${actualHash}`,
        );
      }
    }

    await this.cache.cacheModel(resolvedUrl, targetPath, size);
    return targetPath;
  }

  /** List all cached models. */
  async listModels(): Promise<CacheEntry[]> {
    return this.cache.listModels();
  }

  /** Remove a cached model by its original URL. Returns true if it was found and removed. */
  async removeModel(url: string): Promise<boolean> {
    return this.cache.removeModel(resolveModelUrl(url));
  }
}
