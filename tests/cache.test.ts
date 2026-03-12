import { beforeEach, describe, expect, it, vi } from 'vitest';
import { LocalLLMErrorCode } from '../src/errors';

const fileStore = new Map<string, string>();
const dirStore = new Set<string>();

function joinPath(...parts: string[]): string {
  return parts
    .filter(Boolean)
    .map((part, index) => {
      if (index === 0) return part.replace(/\/+$/, '');
      return part.replace(/^\/+/, '').replace(/\/+$/, '');
    })
    .join('/');
}

function removePathRecursive(target: string): void {
  for (const file of [...fileStore.keys()]) {
    if (file === target || file.startsWith(`${target}/`)) {
      fileStore.delete(file);
    }
  }

  for (const dir of [...dirStore]) {
    if (dir === target || dir.startsWith(`${target}/`)) {
      dirStore.delete(dir);
    }
  }
}

vi.mock('../src/NativeLocalLLM', () => ({
  default: {
    getModelStoragePath: vi.fn(() => '/models'),
    fileExists: vi.fn((path: string) => fileStore.has(path)),
    getFileSize: vi.fn((path: string) => (fileStore.get(path) ?? '').length),
    readTextFile: vi.fn((path: string) => fileStore.get(path) ?? null),
    writeTextFile: vi.fn((path: string, content: string) => {
      fileStore.set(path, content);
      dirStore.add(path.split('/').slice(0, -1).join('/') || '/');
    }),
    removePath: vi.fn((path: string) => removePathRecursive(path)),
  },
}));

describe('ModelCache', () => {
  beforeEach(() => {
    fileStore.clear();
    dirStore.clear();
  });

  it('caches and lists downloaded models', async () => {
    const { ModelCache } = await import('../src/cache');
    const cache = new ModelCache('/models');
    const url = 'unsloth/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf';
    const resolvedUrl = 'https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf';
    const targetPath = cache.targetPath(url);

    fileStore.set(targetPath, 'model-bytes');

    const entry = await cache.cacheModel(url, targetPath, 11);
    expect(entry.url).toBe(resolvedUrl);

    const cached = await cache.getCachedModel(url);
    expect(cached?.path).toBe(targetPath);

    const list = await cache.listModels();
    expect(list).toHaveLength(1);
    expect(list[0]?.path).toBe(targetPath);
  });

  it('removes cache entries and files', async () => {
    const { ModelCache } = await import('../src/cache');
    const cache = new ModelCache('/models');
    const url = 'unsloth/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf';
    const targetPath = cache.targetPath(url);

    fileStore.set(targetPath, 'model-bytes');
    await cache.cacheModel(url, targetPath, 11);

    await expect(cache.removeModel(url)).resolves.toBe(true);
    await expect(cache.listModels()).resolves.toEqual([]);
    expect(fileStore.has(targetPath)).toBe(false);
  });

  it('resets a corrupt cache index and preserves a backup', async () => {
    const { ModelCache } = await import('../src/cache');
    const cache = new ModelCache('/models');
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const backupPathPrefix = joinPath('/models', 'index.json.corrupt.');

    fileStore.set(joinPath('/models', 'index.json'), '{not-json');

    const list = await cache.listModels();
    expect(list).toEqual([]);
    expect(fileStore.get(joinPath('/models', 'index.json'))).toBeUndefined();
    expect([...fileStore.keys()].some((key) => key.startsWith(backupPathPrefix))).toBe(true);
    expect(warnSpy).toHaveBeenCalledOnce();
  });

  it('rejects paths outside the cache directory', async () => {
    const { ModelCache } = await import('../src/cache');
    const cache = new ModelCache('/models');

    try {
      await cache.cacheModel('https://example.com/model.gguf', '/etc/passwd', 100);
      expect.unreachable('Should have thrown');
    } catch (e: any) {
      expect(e.code).toBe(LocalLLMErrorCode.INVALID_PATH);
      expect(e.message).toContain('outside the cache directory');
    }
  });
});
