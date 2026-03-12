# Contributing to local-llm-rn

Thank you for contributing! Here's how to get started.

## Development setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/hilum-labs/local-llm-rn.git
cd local-llm-rn

# Install dependencies for this repo
pnpm install

# The prepare script symlinks cpp/ → vendor/hilum-local-llm-engine
# This runs automatically after pnpm install
```

## Project structure

```
local-llm-rn/
├── src/              TypeScript source (RN bridge + re-exports)
├── ios/              Objective-C++ Turbo Module (LocalLLM.mm)
├── vendor/           Engine submodule (hilum-local-llm-engine)
├── cpp/              Symlink to vendor/ (or real copy in npm)
├── scripts/          Build scripts (prepare.sh, prepack.sh)
├── examples/         Example apps (Expo + bare RN)
└── local-llm-rn.podspec
```

## Key files

- **`ios/LocalLLM.mm`** — The native Objective-C++ Turbo Module (~1,267 lines). This is where all native logic lives: model loading, inference, streaming, embeddings, vision, quantization, device capabilities.
- **`src/native-bridge.ts`** — TypeScript adapter that maps the `NativeAddon` interface to Turbo Module calls.
- **`src/index.ts`** — Entry point that injects the RN native backend and re-exports the shared `local-llm` API.

## Running examples

### Expo

```bash
cd examples/expo-test
npm install
npx expo prebuild
npx expo run:ios
```

### Bare React Native

```bash
cd examples/react-native-test
npm install
cd ios && pod install && cd ..
npx react-native run-ios
```

## Submitting changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Make your changes
4. Run type checking: `npm run typecheck`
   Root repo commands use `pnpm`, so prefer `pnpm typecheck` during development.
5. Submit a pull request

## Engine changes

If your change requires modifications to the C++ engine, submit a separate PR to [hilum-local-llm-engine](https://github.com/hilum-labs/hilum-local-llm-engine) first. Once merged, update the submodule pointer here.

## Code style

- TypeScript: Follow existing patterns, no unnecessary comments
- Objective-C++: Match the existing `LocalLLM.mm` style
- Keep the native bridge as thin as possible — business logic belongs in `local-llm` or the engine
