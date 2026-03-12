# Expo Test App

Test app for `local-llm-rn` using Expo with expo-router.

## Expo Go is NOT supported

This app requires a **development build** — it will not work in Expo Go.

`local-llm-rn` ships ~20 MB of C++ source that Xcode compiles via
CocoaPods. Expo Go is a prebuilt binary with a fixed set of native modules and
cannot load custom native code at runtime. This is the same limitation as any
React Native library with native code (react-native-reanimated, expo-camera, etc).

Use `expo run:ios` (local Xcode build) or `eas build` (cloud build) instead.

## Setup

```bash
# Install dependencies
npm install

# Generate native project (required for native modules)
npx expo prebuild

# Run on iOS simulator or device
npx expo run:ios
```

## What it does

1. Shows device capabilities (RAM, GPU, Metal version)
2. Recommends quantization based on device RAM
3. Downloads TinyLlama Q4_K_M on first launch (~700 MB)
4. Chat UI with streaming responses
5. All inference runs on-device via Metal GPU
