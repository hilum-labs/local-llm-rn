import type { ChatMessage } from 'local-llm-js-core';
import { LocalLLMError, LocalLLMErrorCode } from './errors';

const MEDIA_MARKER = '<__media__>';

function normalizeImageUrl(url: string): string {
  if (
    url.startsWith('data:') ||
    url.startsWith('http://') ||
    url.startsWith('https://') ||
    url.startsWith('file://') ||
    url.startsWith('content://')
  ) {
    return url;
  }

  if (url.startsWith('/')) {
    return `file://${url}`;
  }

  return url;
}

export async function resolveImageToBytes(url: string): Promise<Uint8Array> {
  const response = await fetch(normalizeImageUrl(url));
  if (!response.ok) {
    throw new LocalLLMError(
      LocalLLMErrorCode.VISION_FETCH_FAILED,
      `Failed to fetch image: ${response.status} ${response.statusText}`,
    );
  }
  return new Uint8Array(await response.arrayBuffer());
}

export async function buildVisionPrompt(
  messages: ChatMessage[],
): Promise<{ text: string; imageBuffers: Uint8Array[] }> {
  const imageBuffers: Uint8Array[] = [];
  const textParts: string[] = [];

  for (const msg of messages) {
    if (typeof msg.content === 'string') {
      textParts.push(msg.content);
      continue;
    }

    const msgParts: string[] = [];
    for (const part of msg.content) {
      if (part.type === 'text') {
        msgParts.push(part.text);
      } else if (part.type === 'image_url') {
        imageBuffers.push(await resolveImageToBytes(part.image_url.url));
        msgParts.push(MEDIA_MARKER);
      }
    }
    textParts.push(msgParts.join(''));
  }

  return { text: textParts.join('\n'), imageBuffers };
}
