# local-llm-rn Test App

Minimal bare React Native app for testing `local-llm-rn` on iOS and Android.

## Prerequisites

- Xcode 16+ and CocoaPods for iOS
- Android Studio / Android SDK for Android
- Node.js 20+

## Setup

```bash
# Install dependencies
npm install

# Generate the native projects (iOS + Android)
# This enforces iOS 16.0 and Android SDK levels matching the library.
npm run bootstrap:native

# Install pods (compiles the C++ engine via CocoaPods)
cd ios && pod install && cd ..
```

## Run

```bash
# Start Metro bundler
npm start

# In another terminal — build and run on iOS simulator or device
npm run ios

# Or build and run on Android
npm run android
```

## What it tests

1. **Device capabilities** — native bridge is reachable
2. **Model download** — HuggingFace GGUF download and progress reporting
3. **Cache index** — `listModels()` and `removeModel()` round-trip
4. **Model load** — real `LocalLLM.create()` on device
5. **Inference** — one deterministic completion on CPU
6. **Memory cleanup** — `dispose()` frees model and context

## CI smoke automation

The app exposes a deterministic smoke-test screen that CI drives with Maestro using:

```bash
maestro test .maestro/smoke.yaml
```
