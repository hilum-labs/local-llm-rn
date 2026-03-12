import NativeLocalLLM from './NativeLocalLLM';
import { NativeEventEmitter, NativeModules } from 'react-native';
import type { DownloadAdapter } from 'local-llm-js-core';
import { LocalLLMError, LocalLLMErrorCode } from './errors';

const emitter = new NativeEventEmitter(NativeModules.LocalLLM);

/** Set of URLs whose last error was resumable (native reported `resumable: true`). */
const resumableUrls = new Set<string>();

export function createNativeDownloader(): DownloadAdapter {
  return {
    download(url, destPath, onProgress) {
      return new Promise((resolve, reject) => {
        const subs = [
          emitter.addListener('onDownloadProgress', (e) => {
            if (e.url !== url) return;
            onProgress?.(e.downloaded, e.total, e.percent);
          }),
          emitter.addListener('onDownloadComplete', (e) => {
            if (e.url !== url) return;
            subs.forEach((s) => s.remove());
            resumableUrls.delete(url);
            resolve();
          }),
          emitter.addListener('onDownloadError', (e) => {
            if (e.url !== url) return;
            subs.forEach((s) => s.remove());
            if (e.resumable) {
              resumableUrls.add(url);
            }
            reject(new LocalLLMError(LocalLLMErrorCode.DOWNLOAD_FAILED, e.error));
          }),
        ];

        if (resumableUrls.has(url)) {
          resumableUrls.delete(url);
          NativeLocalLLM.resumeDownload(url, destPath);
        } else {
          NativeLocalLLM.downloadModel(url, destPath);
        }
      });
    },

    cancel(url) {
      NativeLocalLLM.cancelDownload(url);
    },

    supportsResume: true,
  };
}
