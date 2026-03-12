# local-llm-rn

Run LLMs on-device in React Native with Metal (iOS) and Vulkan (Android) GPU acceleration. Same OpenAI-compatible API as [`local-llm`](https://www.npmjs.com/package/local-llm).

[![npm](https://img.shields.io/npm/v/local-llm-rn)](https://www.npmjs.com/package/local-llm-rn)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Platform: iOS | Android](https://img.shields.io/badge/platform-iOS%20%7C%20Android-lightgrey)

```bash
npm install local-llm-rn
```

```typescript
import { LocalLLM } from 'local-llm-rn';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
  compute: 'gpu',
});

const response = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Hello!' }],
  stream: true,
});

for await (const chunk of response) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? '');
}
```

> Need to run on **Node.js** instead? Check out [`local-llm`](https://www.npmjs.com/package/local-llm) for macOS, Linux, and Windows.

## Why local-llm-rn?

- **On-device.** Models run entirely on the phone. No server, no API keys, no data leaves the device.
- **GPU accelerated.** Metal on iOS, Vulkan on Android. Not just CPU inference.
- **OpenAI-compatible API.** Same `chat.completions.create()` you already know from `local-llm` and OpenAI.
- **Device-aware.** Built-in helpers to check RAM, recommend quantization, and prevent OOM crashes.
- **Auto download.** Pass a HuggingFace URL, models are downloaded and cached on-device automatically.
- **Speculative decoding.** Use a small draft model for 2-3x faster generation with zero quality loss.

## Platform Support

| Platform | GPU Backend | Min Version | Notes |
|---|---|---|---|
| iOS | Metal | iOS 16+ | BF16 + Accelerate BLAS |
| Android | Vulkan | Android 8+ (API 26) | CPU fallback on devices without Vulkan |

### Tested Compatibility

| | Versions |
|---|---|
| React Native | 0.76 - 0.83 |
| Expo SDK | 53 - 55 |
| Xcode | 15+ |
| NDK | 27.x |
| CMake | 3.22.1+ |

## Setup

### Expo (recommended)

```bash
npm install local-llm-rn
npx expo prebuild
```

### Bare React Native

```bash
npm install local-llm-rn
cd ios && pod install
```

Requires React Native 0.76+ (New Architecture / Turbo Modules).
Examples and CI are pinned to React Native 0.83 / Expo SDK 55.
Examples target iOS 16.0 and Android SDK levels compatible with the native module.

> **Note:** `local-llm-rn` ships raw TypeScript source (`src/index.ts`) — no pre-compiled JS. This is intentional: Metro (the React Native bundler) handles TypeScript natively, and shipping `.ts` gives consumers full source maps, accurate go-to-definition, and smaller npm tarballs. This package is designed exclusively for the React Native / Metro ecosystem.

## Quick Start

### 1. Check device capabilities

Before loading a model, check if the device can handle it:

```typescript
import { canRunModel, getDeviceCapabilities, recommendQuantization } from 'local-llm-rn';

const caps = getDeviceCapabilities();
console.log(caps.gpuName);       // "Apple A16 GPU"
console.log(caps.totalRAM);      // 6442450944 (6 GB)
console.log(caps.metalFamily);   // 9 (A17+)

const quant = recommendQuantization();
console.log(quant);              // "Q6_K"

const check = canRunModel(1_800_000_000); // 1.8 GB model
if (!check.canRun) {
  console.warn(check.reason);    // "Model needs ~2160 MB but only 1500 MB available"
  console.warn(check.suggestion); // "Try a Q4_K_M quantized variant or a smaller model"
}
```

### 2. Load a model

```typescript
import { LocalLLM } from 'local-llm-rn';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
  compute: 'gpu',
  contextSize: 2048,
  onProgress: (pct) => console.log(`Downloading: ${pct.toFixed(1)}%`),
});
```

### 3. Chat with streaming

```typescript
const response = await ai.chat.completions.create({
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'What is the capital of France?' },
  ],
  stream: true,
});

let text = '';
for await (const chunk of response) {
  text += chunk.choices[0]?.delta?.content ?? '';
  // Update your UI here
}
```

### 4. Check performance

Every response includes inference speed metrics:

```typescript
console.log(`Speed: ${response._timing?.generatedTokensPerSec.toFixed(1)} tok/s`);
console.log(`TTFT: ${response._timing?.promptEvalMs.toFixed(0)} ms`);
```

When streaming, `_timing` is on the final chunk:

```typescript
for await (const chunk of response) {
  const content = chunk.choices[0]?.delta?.content;
  if (content) setText((t) => t + content);
  if (chunk._timing) {
    console.log(`Generation: ${chunk._timing.generatedTokensPerSec.toFixed(1)} tok/s`);
  }
}
```

### 5. Clean up

```typescript
ai.dispose();
```

## Recommended Models

| Model | Quant | Size | Good for |
|---|---|---|---|
| [SmolLM2 1.7B](https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF) | Q4_K_M | ~1.0 GB | Fast, works on all devices |
| [TinyLlama 1.1B](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) | Q4_K_M | ~636 MB | Testing, development |
| [Llama 3.2 3B](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) | Q4_K_M | ~1.8 GB | Best quality for flagship phones |
| [Phi-3 Mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) | Q4_K_M | ~2.2 GB | Great balance of speed and quality |

**Quantization guide by device RAM:**

| Device RAM | Recommended | Examples |
|---|---|---|
| 8 GB | Q8_0 | iPhone 16 Pro |
| 6 GB | Q6_K | iPhone 14/15 Pro |
| 4 GB | Q4_K_M | iPhone 11-13, iPhone 14/15 base |
| 3 GB | Q3_K_S | iPhone X, older devices |

## Device Helpers API

```typescript
import { getDeviceCapabilities, canRunModel, recommendQuantization } from 'local-llm-rn';
```

### `getDeviceCapabilities()`

Returns device hardware info:

```typescript
{
  totalRAM: number;        // Total RAM in bytes
  availableRAM: number;    // Available RAM (respects iOS jetsam limits)
  gpuName: string;         // e.g. "Apple A16 GPU"
  metalFamily: number;     // Apple GPU family (5=A12+, 7=A14+, 9=A17+)
  metalVersion: number;    // Metal version (1, 2, or 3)
  iosVersion: string;      // e.g. "17.2.1"
  isLowPowerMode: boolean;
}
```

### `canRunModel(modelSizeBytes)`

Checks if the device has enough RAM to run a model:

```typescript
const result = canRunModel(1_800_000_000);
// { canRun: true }
// or { canRun: false, reason: "...", suggestion: "..." }
```

### `recommendQuantization()`

Suggests the best quantization level based on device RAM:

```typescript
const quant = recommendQuantization();
// "Q8_0" | "Q6_K" | "Q4_K_M" | "Q3_K_S"
```

## Configuration

```typescript
const ai = await LocalLLM.create({
  model: 'user/repo/file.gguf',   // HuggingFace shorthand or local path

  compute: 'gpu',                  // 'gpu' | 'cpu' | 'auto'
  contextSize: 2048,               // Context window size
  batchSize: 512,                  // Batch size for prompt processing

  warmup: true,                    // Warmup on load — eliminates cold-start (default: true)

  // Speculative decoding (optional — 2-3x faster generation)
  // draftModel: 'user/repo/small-model.gguf',  // Small model from same family
  // draftNMax: 16,                              // Max draft tokens per step

  onProgress: (pct) => {},         // Download progress callback (0-100)
});
```

## Error Handling

All errors thrown by `local-llm-rn` are instances of `LocalLLMError` with a typed `code` property:

```typescript
import { LocalLLMError, LocalLLMErrorCode } from 'local-llm-rn';

try {
  const ai = await LocalLLM.create({ model: 'user/repo/model.gguf' });
} catch (e) {
  if (e instanceof LocalLLMError) {
    switch (e.code) {
      case LocalLLMErrorCode.MODEL_LOAD_FAILED:
        // Handle model loading failure
        break;
      case LocalLLMErrorCode.DOWNLOAD_FAILED:
        // Handle download failure
        break;
      case LocalLLMErrorCode.INSUFFICIENT_MEMORY:
        // Suggest a smaller model
        break;
    }
  }
}
```

Available error codes: `MODEL_LOAD_FAILED`, `MODEL_TOO_LARGE`, `CONTEXT_CREATE_FAILED`, `CONTEXT_EXHAUSTED`, `INFERENCE_FAILED`, `STREAM_FAILED`, `DOWNLOAD_FAILED`, `DOWNLOAD_INTEGRITY_MISMATCH`, `VISION_FAILED`, `VISION_FETCH_FAILED`, `EMBEDDING_FAILED`, `NOT_INITIALIZED`, `INVALID_PATH`, `CACHE_CORRUPT`, `QUANTIZE_FAILED`, `INSUFFICIENT_MEMORY`.

## Device + Performance Combo

Combine device capabilities with inference metrics:

```typescript
import { LocalLLM, getDeviceCapabilities } from 'local-llm-rn';

const caps = getDeviceCapabilities();
console.log(`Device: ${caps.gpuName}, ${(caps.totalRAM / 1e9).toFixed(1)} GB RAM`);

const ai = await LocalLLM.create({
  model: modelPath,
  compute: 'gpu',
});

const response = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Hello!' }],
});

console.log(response.choices[0].message.content);
console.log(`Speed: ${response._timing?.generatedTokensPerSec.toFixed(1)} tok/s on ${caps.gpuName}`);

ai.dispose();
```

## Examples

- **[Expo example](./examples/expo-test/)** — Complete chat UI with device detection, model downloading, and streaming responses
- **[Bare RN example](./examples/react-native-test/)** — Minimal bare React Native test app

## Ecosystem

| Package | Description | Install |
|---|---|---|
| [`local-llm`](https://www.npmjs.com/package/local-llm) | Node.js / Bun / Electron | `npm install local-llm` |
| [`local-llm-rn`](https://www.npmjs.com/package/local-llm-rn) | React Native / Expo (this package) | `npm install local-llm-rn` |
| [`local_llm`](https://pub.dev/packages/local_llm) | Flutter | `flutter pub add local_llm` |
| [`hilum-local-llm-engine`](https://github.com/hilum-labs/hilum-local-llm-engine) | Core C++ engine | Vendored automatically |

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for setup instructions.

## Contact

Questions, feedback, or partnership inquiries: [info@hilumlabs.com](mailto:info@hilumlabs.com)

## License

MIT — See [LICENSE](./LICENSE) for details.

Made by [Hilum Labs](https://github.com/hilum-labs).
