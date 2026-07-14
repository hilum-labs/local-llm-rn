# Changelog

## Unreleased

### Fixed

- Retain native model ownership until queued iOS and Android context operations finish.
- Use the dedicated embedding-context destructor across TypeScript, iOS, and Android.
- Revalidate cached model size and optional SHA-256 metadata before reuse.
- Reject traversal, sibling-prefix, symlink, and out-of-storage file destinations.
- Remove terminal stream and batch listeners on completion, failure, and synchronous startup errors.
- Report Android batch failures through the same event contract as iOS.

### Changed

- Derive the CocoaPods version and tag from `package.json`.
- Run locked tests, dependency audits, native builds, and vendoring checks in release gates.

## 0.1.0

Initial release — split from [`local-llm`](https://github.com/hilum-labs/local-llm) monorepo into standalone package.

### Features

- Metal GPU acceleration on iOS (BF16, Accelerate BLAS)
- Turbo Module native bridge (React Native 0.76+ New Architecture)
- OpenAI-compatible `chat.completions.create()` API
- Streaming text generation
- Vision / multimodal support
- Embeddings API (single + batch)
- Grammar / structured JSON output
- Function / tool calling
- Batch inference (multi-sequence)
- Model quantization utilities
- Context window management (sliding window, truncation)
- Device capability checks (RAM, Metal family, quantization recommendations)
- Background model downloads via native `NSURLSession`
- Expo and bare React Native support
