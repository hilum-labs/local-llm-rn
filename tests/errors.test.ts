import { describe, expect, it } from 'vitest';
import { LocalLLMError, LocalLLMErrorCode } from '../src/errors';

describe('LocalLLMError', () => {
  it('includes error code and message', () => {
    const error = new LocalLLMError(LocalLLMErrorCode.MODEL_LOAD_FAILED, 'bad model');
    expect(error.code).toBe('MODEL_LOAD_FAILED');
    expect(error.message).toBe('bad model');
    expect(error.name).toBe('LocalLLMError');
    expect(error).toBeInstanceOf(Error);
  });

  it('supports cause via ErrorOptions', () => {
    const cause = new Error('root cause');
    const error = new LocalLLMError(LocalLLMErrorCode.DOWNLOAD_FAILED, 'download broke', { cause });
    expect(error.cause).toBe(cause);
  });

  it('can be caught by code', () => {
    try {
      throw new LocalLLMError(LocalLLMErrorCode.INSUFFICIENT_MEMORY, 'not enough RAM');
    } catch (e) {
      if (e instanceof LocalLLMError && e.code === LocalLLMErrorCode.INSUFFICIENT_MEMORY) {
        expect(true).toBe(true);
        return;
      }
    }
    expect.unreachable('Should have caught the error');
  });
});
