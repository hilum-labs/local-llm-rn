import { beforeEach, describe, expect, it, vi } from 'vitest';

const getDeviceCapabilities = vi.fn();

vi.mock('../src/NativeLocalLLM', () => ({
  default: {
    getDeviceCapabilities,
  },
}));

describe('device helpers', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('reports device capabilities from the native module', async () => {
    const fakeCaps = {
      totalRAM: 8 * 1024 * 1024 * 1024,
      availableRAM: 6 * 1024 * 1024 * 1024,
      gpuName: 'Apple GPU',
      isLowPowerMode: false,
    };
    getDeviceCapabilities.mockReturnValueOnce(fakeCaps);

    const device = await import('../src/device');
    expect(device.getDeviceCapabilities()).toEqual(fakeCaps);
  });

  it('rejects models that exceed estimated available memory', async () => {
    getDeviceCapabilities.mockReturnValueOnce({
      totalRAM: 4 * 1024 * 1024 * 1024,
      availableRAM: 512 * 1024 * 1024,
      gpuName: 'Adreno',
      isLowPowerMode: false,
    });

    const device = await import('../src/device');
    const result = device.canRunModel(700 * 1024 * 1024);

    expect(result.canRun).toBe(false);
    expect(result.reason).toContain('available');
  });

  it('recommends quantization tiers from total RAM', async () => {
    const device = await import('../src/device');

    getDeviceCapabilities.mockReturnValueOnce({
      totalRAM: 9 * 1024 * 1024 * 1024,
      availableRAM: 7 * 1024 * 1024 * 1024,
      gpuName: 'Apple GPU',
      isLowPowerMode: false,
    });
    expect(device.recommendQuantization()).toBe('Q8_0');

    getDeviceCapabilities.mockReturnValueOnce({
      totalRAM: 6 * 1024 * 1024 * 1024,
      availableRAM: 4 * 1024 * 1024 * 1024,
      gpuName: 'Apple GPU',
      isLowPowerMode: false,
    });
    expect(device.recommendQuantization()).toBe('Q6_K');

    getDeviceCapabilities.mockReturnValueOnce({
      totalRAM: 4 * 1024 * 1024 * 1024,
      availableRAM: 3 * 1024 * 1024 * 1024,
      gpuName: 'Apple GPU',
      isLowPowerMode: false,
    });
    expect(device.recommendQuantization()).toBe('Q4_K_M');
  });
});
