# Integration Tests

These tests require a real device or simulator with a downloaded model.
They are NOT run in CI — run them manually during development.

## Setup

```bash
# Download a small test model (~85 MB)
curl -L -o /tmp/tinyllama-q2.gguf \
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
```

## Run

```bash
# From the example app directory
cd examples/react-native-test
npx react-native run-ios   # or run-android

# Maestro smoke tests (device must be booted)
maestro test .maestro/smoke.yaml
```

## What to verify manually

1. **Model loads** — No crash, progress callback fires
2. **Streaming works** — Tokens arrive one at a time via `onToken` events
3. **stopStream mid-generation** — Call `stopStream` after 2-3 tokens, verify it stops cleanly (no crash, done event fires)
4. **Model freed during generation** — Free the model while streaming, verify error event (not a crash)
5. **Memory pressure** — Load a model on a device near its RAM limit, verify `canRunModel` returns false
6. **Re-download after clear** — Clear cache, download again, verify cache hit on second load
