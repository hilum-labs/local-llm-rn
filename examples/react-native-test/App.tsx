import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { LocalLLM, ModelManager, getDeviceCapabilities } from 'local-llm-rn';

const SMOKE_MODEL_URL =
  'unsloth/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf';
const SMOKE_PROMPT = 'Reply with exactly: OK';

type StepState = 'idle' | 'running' | 'pass' | 'fail';
type StepKey =
  | 'device'
  | 'download'
  | 'listAfterDownload'
  | 'load'
  | 'benchmark'
  | 'infer'
  | 'remove'
  | 'listAfterRemove';

const STEP_LABELS: Record<StepKey, string> = {
  device: 'device',
  download: 'download',
  listAfterDownload: 'list-after-download',
  load: 'load',
  benchmark: 'benchmark',
  infer: 'infer',
  remove: 'remove',
  listAfterRemove: 'list-after-remove',
};

const INITIAL_STEPS: Record<StepKey, StepState> = {
  device: 'idle',
  download: 'idle',
  listAfterDownload: 'idle',
  load: 'idle',
  benchmark: 'idle',
  infer: 'idle',
  remove: 'idle',
  listAfterRemove: 'idle',
};

type OverallState = 'idle' | 'running' | 'passed' | 'failed';

export default function App() {
  const manager = useMemo(() => new ModelManager(), []);
  const hasAutoStarted = useRef(false);
  const [overall, setOverall] = useState<OverallState>('idle');
  const [steps, setSteps] = useState<Record<StepKey, StepState>>({ ...INITIAL_STEPS });
  const [logs, setLogs] = useState<string[]>(['Smoke test ready.']);
  const [progress, setProgress] = useState('0.0%');
  const [loading, setLoading] = useState(false);

  const appendLog = useCallback((line: string) => {
    setLogs((current) => [...current, line]);
  }, []);

  const setStep = useCallback((step: StepKey, state: StepState, detail?: string) => {
    setSteps((current) => ({ ...current, [step]: state }));
    if (detail) {
      appendLog(`${STEP_LABELS[step]}: ${detail}`);
    }
  }, [appendLog]);

  const reset = useCallback(() => {
    setOverall('idle');
    setSteps({ ...INITIAL_STEPS });
    setLogs(['Smoke test ready.']);
    setProgress('0.0%');
  }, []);

  const runSmokeTest = useCallback(async () => {
    if (loading) {
      return;
    }

    reset();
    setLoading(true);
    setOverall('running');

    let ai: LocalLLM | null = null;
    let downloadedPath = '';
    let currentStep: StepKey = 'device';

    try {
      currentStep = 'device';
      setStep('device', 'running', 'probing device capabilities');
      const caps = getDeviceCapabilities();
      setStep(
        'device',
        'pass',
        `ram=${(caps.totalRAM / 1e9).toFixed(1)}GB gpu=${caps.gpuName}`,
      );

      appendLog('pre-cleaning model cache entry');
      await manager.removeModel(SMOKE_MODEL_URL).catch(() => false);

      currentStep = 'download';
      setStep('download', 'running', 'downloading smoke model');
      downloadedPath = await manager.downloadModel(SMOKE_MODEL_URL, {
        onProgress: (_downloaded, _total, percent) => {
          setProgress(`${percent.toFixed(1)}%`);
        },
      });
      setStep('download', 'pass', `path=${downloadedPath}`);

      currentStep = 'listAfterDownload';
      setStep('listAfterDownload', 'running', 'listing cached models');
      const cachedAfterDownload = await manager.listModels();
      if (!cachedAfterDownload.some((entry) => entry.path === downloadedPath)) {
        throw new Error('Downloaded model was not present in cache index');
      }
      setStep('listAfterDownload', 'pass', `entries=${cachedAfterDownload.length}`);

      currentStep = 'load';
      setStep('load', 'running', 'loading model on CPU');
      ai = await LocalLLM.create({
        model: SMOKE_MODEL_URL,
        compute: 'cpu',
        contextSize: 512,
      });
      setStep('load', 'pass', 'model initialized');

      currentStep = 'benchmark';
      setStep('benchmark', 'running', 'running cpu benchmark');
      const benchmark = await ai.benchmark({
        promptTokens: 128,
        generateTokens: 64,
        iterations: 3,
      });
      const benchmarkSummary =
        `prompt_tps=${benchmark.promptTokensPerSec.toFixed(2)} ` +
        `gen_tps=${benchmark.generatedTokensPerSec.toFixed(2)} ` +
        `ttft_ms=${benchmark.ttftMs.toFixed(2)} total_ms=${benchmark.totalMs.toFixed(2)}`;
      console.log(`BENCHMARK_RESULT ${benchmarkSummary}`);
      setStep('benchmark', 'pass', benchmarkSummary);

      currentStep = 'infer';
      setStep('infer', 'running', 'running chat completion');
      const response = await ai.chat.completions.create({
        messages: [{ role: 'user', content: SMOKE_PROMPT }],
        max_tokens: 8,
        temperature: 0,
        stream: false,
      });
      const content = response.choices[0]?.message?.content?.trim() ?? '';
      if (!content) {
        throw new Error('Inference returned empty content');
      }
      setStep('infer', 'pass', `response=${JSON.stringify(content)}`);

      ai.dispose();
      ai = null;

      currentStep = 'remove';
      setStep('remove', 'running', 'removing cached model');
      const removed = await manager.removeModel(SMOKE_MODEL_URL);
      if (!removed) {
        throw new Error('removeModel returned false');
      }
      setStep('remove', 'pass', 'cache entry removed');

      currentStep = 'listAfterRemove';
      setStep('listAfterRemove', 'running', 'verifying cache cleanup');
      const cachedAfterRemove = await manager.listModels();
      if (cachedAfterRemove.some((entry) => entry.path === downloadedPath)) {
        throw new Error('Model still present after removeModel');
      }
      setStep('listAfterRemove', 'pass', `entries=${cachedAfterRemove.length}`);

      setOverall('passed');
      appendLog('smoke test passed');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setOverall('failed');
      appendLog(`failure: ${message}`);
      setSteps((current) => ({ ...current, [currentStep]: 'fail' }));
      if (ai) {
        ai.dispose();
      }
    } finally {
      setLoading(false);
    }
  }, [appendLog, loading, manager, reset, setStep]);

  useEffect(() => {
    if (hasAutoStarted.current) {
      return;
    }
    hasAutoStarted.current = true;
    void runSmokeTest();
  }, [runSmokeTest]);

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>local-llm-rn CLI Smoke Test</Text>
      <Text style={styles.meta}>Model: {SMOKE_MODEL_URL}</Text>
      <Text style={styles.meta}>Prompt: {SMOKE_PROMPT}</Text>
      <Text testID="smoke-overall" style={styles.overall}>
        SMOKE STATUS: {overall.toUpperCase()}
      </Text>
      <Text testID="smoke-progress" style={styles.progress}>
        DOWNLOAD PROGRESS: {progress}
      </Text>

      <View style={styles.actions}>
        <TouchableOpacity
          testID="run-smoke-test"
          style={[styles.button, loading && styles.buttonDisabled]}
          onPress={runSmokeTest}
          disabled={loading}
        >
          {loading ? <ActivityIndicator color="#fff" /> : <Text style={styles.buttonText}>Run Smoke Test</Text>}
        </TouchableOpacity>
        <TouchableOpacity
          testID="reset-smoke-test"
          style={[styles.button, styles.secondaryButton]}
          onPress={reset}
          disabled={loading}
        >
          <Text style={styles.buttonText}>Reset</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.steps}>
        {(Object.keys(INITIAL_STEPS) as StepKey[]).map((step) => (
          <Text key={step} testID={`step-${STEP_LABELS[step]}`} style={styles.step}>
            STEP {STEP_LABELS[step].toUpperCase()}: {steps[step].toUpperCase()}
          </Text>
        ))}
      </View>

      <ScrollView style={styles.logBox}>
        {logs.map((line, index) => (
          <Text key={`${index}-${line}`} testID={index === logs.length - 1 ? 'smoke-log-last' : undefined} style={styles.logLine}>
            {line}
          </Text>
        ))}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f6f0e8',
    padding: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    marginBottom: 12,
    color: '#1c1a17',
  },
  meta: {
    fontSize: 12,
    color: '#5f584f',
    marginBottom: 4,
  },
  overall: {
    marginTop: 12,
    fontSize: 16,
    fontWeight: '700',
    color: '#1c1a17',
  },
  progress: {
    marginTop: 6,
    fontSize: 13,
    color: '#5f584f',
  },
  actions: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 20,
    marginBottom: 20,
  },
  button: {
    flex: 1,
    minHeight: 48,
    backgroundColor: '#0d5c63',
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 12,
  },
  secondaryButton: {
    backgroundColor: '#7a6f62',
  },
  buttonDisabled: {
    opacity: 0.7,
  },
  buttonText: {
    color: '#fff',
    fontWeight: '700',
    fontSize: 15,
  },
  steps: {
    backgroundColor: '#fffdf9',
    borderRadius: 12,
    padding: 14,
    gap: 8,
    marginBottom: 16,
  },
  step: {
    fontSize: 13,
    color: '#1c1a17',
  },
  logBox: {
    flex: 1,
    backgroundColor: '#1f2430',
    borderRadius: 12,
    padding: 14,
  },
  logLine: {
    color: '#d8dee9',
    fontSize: 12,
    lineHeight: 18,
    marginBottom: 6,
  },
});
