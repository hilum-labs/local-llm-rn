import NativeLocalLLM from './NativeLocalLLM';

export interface DeviceCapabilities {
  /** Total physical RAM in bytes. */
  totalRAM: number;
  /** Currently available RAM in bytes (respects iOS jetsam limits). */
  availableRAM: number;
  /** GPU name, e.g. "Apple A16 GPU" or "Adreno 740". */
  gpuName: string;
  /** Apple GPU family: 5 = A12+, 7 = A14+, 9 = A17+. iOS only. */
  metalFamily?: number;
  /** Metal API version: 1, 2, or 3. iOS only. */
  metalVersion?: number;
  /** OS version string, e.g. "17.2.1". iOS only. */
  iosVersion?: string;
  /** Vulkan API version, e.g. "1.3". Android only. */
  vulkanVersion?: string;
  /** SoC model string from Build.SOC_MODEL, e.g. "SM8550". Android only. */
  chipset?: string;
  /** Android version string, e.g. "14". Android only. */
  androidVersion?: string;
  /** Whether the device is in low-power / battery saver mode. */
  isLowPowerMode: boolean;
}

/** Returns the device's hardware capabilities (RAM, GPU, platform info). */
export function getDeviceCapabilities(): DeviceCapabilities {
  return NativeLocalLLM.getDeviceCapabilities() as DeviceCapabilities;
}

/**
 * Check whether the device has enough available RAM to load a model of the given size.
 * Returns `{ canRun: true }` or `{ canRun: false, reason, suggestion }`.
 */
export function canRunModel(modelSizeBytes: number): {
  canRun: boolean;
  reason?: string;
  suggestion?: string;
} {
  const caps = getDeviceCapabilities();
  const availableMB = caps.availableRAM / (1024 * 1024);
  const modelSizeMB = modelSizeBytes / (1024 * 1024);

  // Model needs ~1.2x its file size in RAM (weights + KV cache + overhead)
  const estimatedUsageMB = modelSizeMB * 1.2;

  if (estimatedUsageMB > availableMB * 0.8) {
    return {
      canRun: false,
      reason: `Model needs ~${Math.round(estimatedUsageMB)} MB but only ${Math.round(availableMB)} MB available`,
      suggestion: modelSizeMB > 2000
        ? 'Try a Q4_K_M quantized variant or a smaller model'
        : 'Close other apps to free memory',
    };
  }

  return { canRun: true };
}

/**
 * Recommend a quantization level based on the device's total RAM.
 *
 * - 8+ GB: `Q8_0` (iPhone 16 Pro, Pixel 8 Pro)
 * - 6 GB: `Q6_K` (iPhone 14/15 Pro, flagship Android)
 * - 4 GB: `Q4_K_M` (iPhone 11-13, mid-range Android)
 * - <3 GB: `Q3_K_S` (low-RAM devices)
 */
export function recommendQuantization(): string {
  const caps = getDeviceCapabilities();
  const totalGB = caps.totalRAM / (1024 * 1024 * 1024);

  if (totalGB >= 8) return 'Q8_0';
  if (totalGB >= 6) return 'Q6_K';
  if (totalGB >= 4) return 'Q4_K_M';
  return 'Q3_K_S';
}
