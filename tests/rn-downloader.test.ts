import { beforeEach, describe, expect, it, vi } from 'vitest';
import { LocalLLMErrorCode } from '../src/errors';

let eventHandlers: Record<string, ((event: any) => void)[]> = {};

class MockNativeEventEmitter {
  addListener(eventName: string, handler: (event: any) => void) {
    if (!eventHandlers[eventName]) eventHandlers[eventName] = [];
    eventHandlers[eventName].push(handler);
    return {
      remove: () => {
        const arr = eventHandlers[eventName];
        if (arr) {
          const idx = arr.indexOf(handler);
          if (idx >= 0) arr.splice(idx, 1);
        }
      },
    };
  }
}

vi.mock('react-native', () => ({
  NativeEventEmitter: MockNativeEventEmitter,
  NativeModules: { LocalLLM: {} },
}));

const nativeMock = {
  downloadModel: vi.fn(),
  resumeDownload: vi.fn(),
  cancelDownload: vi.fn(),
};

vi.mock('../src/NativeLocalLLM', () => ({
  default: nativeMock,
}));

function emit(eventName: string, data: any) {
  for (const handler of (eventHandlers[eventName] ?? [])) {
    handler(data);
  }
}

let testId = 0;
function uniqueUrl() {
  return `https://example.com/model-${++testId}.gguf`;
}

describe('createNativeDownloader', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    eventHandlers = {};
  });

  it('calls downloadModel on first download', async () => {
    const url = uniqueUrl();
    const { createNativeDownloader } = await import('../src/rn-downloader');
    const downloader = createNativeDownloader();

    const promise = downloader.download(url, '/dest/model.gguf');
    emit('onDownloadComplete', { url });

    await promise;
    expect(nativeMock.downloadModel).toHaveBeenCalledWith(url, '/dest/model.gguf');
    expect(nativeMock.resumeDownload).not.toHaveBeenCalled();
  });

  it('rejects with DOWNLOAD_FAILED and marks resumable', async () => {
    const url = uniqueUrl();
    const { createNativeDownloader } = await import('../src/rn-downloader');
    const downloader = createNativeDownloader();

    const promise = downloader.download(url, '/dest/model.gguf');
    emit('onDownloadError', { url, error: 'Network lost', resumable: true });

    try {
      await promise;
      expect.unreachable('Should have thrown');
    } catch (e: any) {
      expect(e.code).toBe(LocalLLMErrorCode.DOWNLOAD_FAILED);
      expect(e.message).toBe('Network lost');
    }
  });

  it('calls resumeDownload on retry after a resumable error', async () => {
    const url = uniqueUrl();
    const { createNativeDownloader } = await import('../src/rn-downloader');
    const downloader = createNativeDownloader();

    // First attempt fails with resumable: true
    const p1 = downloader.download(url, '/dest/model.gguf');
    emit('onDownloadError', { url, error: 'Interrupted', resumable: true });
    await p1.catch(() => {});

    // Second attempt should use resumeDownload
    const p2 = downloader.download(url, '/dest/model.gguf');
    emit('onDownloadComplete', { url });
    await p2;

    expect(nativeMock.downloadModel).toHaveBeenCalledTimes(1);
    expect(nativeMock.resumeDownload).toHaveBeenCalledTimes(1);
    expect(nativeMock.resumeDownload).toHaveBeenCalledWith(url, '/dest/model.gguf');
  });

  it('calls fresh downloadModel on retry after a non-resumable error', async () => {
    const url = uniqueUrl();
    const { createNativeDownloader } = await import('../src/rn-downloader');
    const downloader = createNativeDownloader();

    // First attempt fails with resumable: false
    const p1 = downloader.download(url, '/dest/model.gguf');
    emit('onDownloadError', { url, error: '404', resumable: false });
    await p1.catch(() => {});

    // Second attempt should start fresh
    const p2 = downloader.download(url, '/dest/model.gguf');
    emit('onDownloadComplete', { url });
    await p2;

    expect(nativeMock.downloadModel).toHaveBeenCalledTimes(2);
    expect(nativeMock.resumeDownload).not.toHaveBeenCalled();
  });

  it('reports supportsResume as true', async () => {
    const { createNativeDownloader } = await import('../src/rn-downloader');
    const downloader = createNativeDownloader();
    expect(downloader.supportsResume).toBe(true);
  });

  it('fires progress callbacks', async () => {
    const url = uniqueUrl();
    const { createNativeDownloader } = await import('../src/rn-downloader');
    const downloader = createNativeDownloader();
    const progress = vi.fn();

    const promise = downloader.download(url, '/dest/model.gguf', progress);
    emit('onDownloadProgress', { url, downloaded: 500, total: 1000, percent: 50 });
    emit('onDownloadComplete', { url });

    await promise;
    expect(progress).toHaveBeenCalledWith(500, 1000, 50);
  });

  it('cancel delegates to native', async () => {
    const url = uniqueUrl();
    const { createNativeDownloader } = await import('../src/rn-downloader');
    const downloader = createNativeDownloader();
    downloader.cancel(url);
    expect(nativeMock.cancelDownload).toHaveBeenCalledWith(url);
  });
});
