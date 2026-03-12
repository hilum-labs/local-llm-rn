import { beforeEach, describe, expect, it, vi } from 'vitest';
import { LocalLLMErrorCode } from '../src/errors';

const nativeMock = {
  getFileSize: vi.fn(() => 1234),
  removePath: vi.fn(),
  sha256File: vi.fn(async () => 'abc123'),
};

const getCachedModel = vi.fn();
const cacheModel = vi.fn();
const listModels = vi.fn();
const removeModel = vi.fn();
const targetPath = vi.fn((url: string) => `/cache/${encodeURIComponent(url)}/model.gguf`);

const download = vi.fn();

vi.mock('../src/NativeLocalLLM', () => ({
  default: nativeMock,
}));

vi.mock('../src/cache', () => ({
  ModelCache: class ModelCache {
    getCachedModel = getCachedModel;
    cacheModel = cacheModel;
    listModels = listModels;
    removeModel = removeModel;
    targetPath = targetPath;
  },
}));

vi.mock('../src/rn-downloader', () => ({
  createNativeDownloader: vi.fn(() => ({ download })),
}));

describe('ModelManager', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns cached model paths without downloading again', async () => {
    getCachedModel.mockResolvedValueOnce({ path: '/cache/model.gguf' });

    const { ModelManager } = await import('../src/model-manager');
    const manager = new ModelManager('/cache');

    await expect(manager.downloadModel('user/repo/model.gguf')).resolves.toBe('/cache/model.gguf');
    expect(download).not.toHaveBeenCalled();
    expect(cacheModel).not.toHaveBeenCalled();
  });

  it('downloads and caches a model on cache miss', async () => {
    getCachedModel.mockResolvedValueOnce(null);
    download.mockResolvedValueOnce(undefined);
    cacheModel.mockResolvedValueOnce(undefined);

    const { ModelManager } = await import('../src/model-manager');
    const manager = new ModelManager('/cache');

    const result = await manager.downloadModel('user/repo/model.gguf');
    expect(result).toContain('/cache/');
    expect(download).toHaveBeenCalledOnce();
    expect(nativeMock.getFileSize).toHaveBeenCalledOnce();
    expect(cacheModel).toHaveBeenCalledOnce();
  });

  it('rejects empty downloads with DOWNLOAD_INTEGRITY_MISMATCH', async () => {
    getCachedModel.mockResolvedValueOnce(null);
    download.mockResolvedValueOnce(undefined);
    nativeMock.getFileSize.mockReturnValueOnce(0);

    const { ModelManager } = await import('../src/model-manager');
    const manager = new ModelManager('/cache');

    try {
      await manager.downloadModel('user/repo/model.gguf');
      expect.unreachable('Should have thrown');
    } catch (e: any) {
      expect(e.code).toBe(LocalLLMErrorCode.DOWNLOAD_INTEGRITY_MISMATCH);
      expect(nativeMock.removePath).toHaveBeenCalledOnce();
    }
  });

  it('rejects size-mismatched downloads when expectedSize is set', async () => {
    getCachedModel.mockResolvedValueOnce(null);
    download.mockResolvedValueOnce(undefined);
    nativeMock.getFileSize.mockReturnValueOnce(500);

    const { ModelManager } = await import('../src/model-manager');
    const manager = new ModelManager('/cache');

    try {
      await manager.downloadModel('user/repo/model.gguf', { expectedSize: 1234 });
      expect.unreachable('Should have thrown');
    } catch (e: any) {
      expect(e.code).toBe(LocalLLMErrorCode.DOWNLOAD_INTEGRITY_MISMATCH);
      expect(e.message).toContain('expected 1234 bytes, got 500 bytes');
    }
  });

  it('rejects SHA256-mismatched downloads when expectedSha256 is set', async () => {
    getCachedModel.mockResolvedValueOnce(null);
    download.mockResolvedValueOnce(undefined);
    nativeMock.getFileSize.mockReturnValueOnce(1234);
    nativeMock.sha256File.mockResolvedValueOnce('deadbeef');

    const { ModelManager } = await import('../src/model-manager');
    const manager = new ModelManager('/cache');

    try {
      await manager.downloadModel('user/repo/model.gguf', { expectedSha256: 'abc123' });
      expect.unreachable('Should have thrown');
    } catch (e: any) {
      expect(e.code).toBe(LocalLLMErrorCode.DOWNLOAD_INTEGRITY_MISMATCH);
      expect(e.message).toContain('SHA256 mismatch');
      expect(e.message).toContain('expected abc123');
      expect(e.message).toContain('got deadbeef');
      expect(nativeMock.removePath).toHaveBeenCalledOnce();
    }
  });

  it('passes SHA256 check when hash matches', async () => {
    getCachedModel.mockResolvedValueOnce(null);
    download.mockResolvedValueOnce(undefined);
    nativeMock.getFileSize.mockReturnValueOnce(1234);
    nativeMock.sha256File.mockResolvedValueOnce('abc123def456');
    cacheModel.mockResolvedValueOnce(undefined);

    const { ModelManager } = await import('../src/model-manager');
    const manager = new ModelManager('/cache');

    const result = await manager.downloadModel('user/repo/model.gguf', { expectedSha256: 'ABC123DEF456' });
    expect(result).toContain('/cache/');
    expect(nativeMock.removePath).not.toHaveBeenCalled();
    expect(cacheModel).toHaveBeenCalledOnce();
  });

  it('delegates list and remove operations to the cache', async () => {
    const fakeEntries = [{ url: 'https://example.com/model.gguf', path: '/cache/model.gguf', size: 1, downloadedAt: '', lastUsedAt: '' }];
    listModels.mockResolvedValueOnce(fakeEntries);
    removeModel.mockResolvedValueOnce(true);

    const { ModelManager } = await import('../src/model-manager');
    const manager = new ModelManager('/cache');

    await expect(manager.listModels()).resolves.toEqual(fakeEntries);
    await expect(manager.removeModel('user/repo/model.gguf')).resolves.toBe(true);
    expect(removeModel).toHaveBeenCalledOnce();
  });
});
