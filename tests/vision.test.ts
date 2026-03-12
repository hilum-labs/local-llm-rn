import { afterEach, describe, expect, it, vi } from 'vitest';
import { LocalLLMErrorCode } from '../src/errors';

describe('vision helpers', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('normalizes local image paths and collects media buffers', async () => {
    const fetchMock = vi.fn(async (url: string) => ({
      ok: true,
      arrayBuffer: async () => new TextEncoder().encode(url).buffer,
    }));
    vi.stubGlobal('fetch', fetchMock);

    const { buildVisionPrompt } = await import('../src/vision');
    const result = await buildVisionPrompt([
      { role: 'user', content: [{ type: 'text', text: 'Look' }, { type: 'image_url', image_url: { url: '/tmp/image.png' } }] },
      { role: 'assistant', content: 'Done' },
    ]);

    expect(fetchMock).toHaveBeenCalledWith('file:///tmp/image.png');
    expect(result.text).toContain('<__media__>');
    expect(result.imageBuffers).toHaveLength(1);
    expect(result.imageBuffers[0]).toBeInstanceOf(Uint8Array);
  });

  it('throws when image fetch fails', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: false,
      status: 404,
      statusText: 'Not Found',
    })));

    const { resolveImageToBytes } = await import('../src/vision');
    try {
      await resolveImageToBytes('https://example.com/missing.png');
      expect.unreachable('Should have thrown');
    } catch (e: any) {
      expect(e.code).toBe(LocalLLMErrorCode.VISION_FETCH_FAILED);
      expect(e.message).toContain('404 Not Found');
    }
  });
});
