import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  SafeAreaView,
  ScrollView,
  TextInput,
  Text,
  Pressable,
  View,
  StyleSheet,
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import {
  LocalLLM,
  getDeviceCapabilities,
  recommendQuantization,
  canRunModel,
} from 'local-llm-rn';
import { DeviceInfoCard } from '../components/DeviceInfoCard';
import { ChatBubble } from '../components/ChatBubble';

const MODEL_URL = 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf';
const MODEL_SIZE_BYTES = 669_000_000;

type Message = { role: 'user' | 'assistant'; content: string };

export default function ChatScreen() {
  const [status, setStatus] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<number | null>(null);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [loading, setLoading] = useState(false);
  const [ai, setAi] = useState<LocalLLM | null>(null);
  const [showDeviceInfo, setShowDeviceInfo] = useState(false);
  const scrollRef = useRef<ScrollView>(null);

  useEffect(() => {
    return () => { ai?.dispose(); };
  }, [ai]);

  const loadModel = useCallback(async () => {
    setLoading(true);
    setStatus('Checking device...');

    const check = canRunModel(MODEL_SIZE_BYTES);
    if (!check.canRun) {
      Alert.alert('Cannot run model', `${check.reason}\n\n${check.suggestion}`);
      setStatus(null);
      setLoading(false);
      return;
    }

    const caps = getDeviceCapabilities();
    const quant = recommendQuantization();
    setStatus(`${caps.gpuName} | ${(caps.totalRAM / 1e9).toFixed(1)} GB RAM | Using ${quant}`);

    setStatus('Downloading model...');
    try {
      const instance = await LocalLLM.create({
        model: MODEL_URL,
        compute: 'gpu',
        contextSize: 2048,
        onProgress: (pct) => setDownloadProgress(pct),
      });

      setAi(instance);
      setDownloadProgress(null);
      setStatus('Ready');
    } catch (err: any) {
      setStatus(`Error: ${err.message}`);
      setDownloadProgress(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const sendMessage = useCallback(async () => {
    if (!ai || !input.trim() || streaming) return;

    const userMsg: Message = { role: 'user', content: input.trim() };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput('');
    setStreaming(true);

    // Add empty assistant message for streaming
    const assistantMsg: Message = { role: 'assistant', content: '' };
    setMessages([...newMessages, assistantMsg]);

    try {
      const response = await ai.chat.completions.create({
        messages: newMessages.map((m) => ({ role: m.role, content: m.content })),
        stream: true,
      });

      let text = '';
      for await (const chunk of response) {
        const delta = chunk.choices[0]?.delta?.content ?? '';
        text += delta;
        setMessages([...newMessages, { role: 'assistant', content: text }]);
      }
    } catch (err: any) {
      setMessages([...newMessages, { role: 'assistant', content: `Error: ${err.message}` }]);
    } finally {
      setStreaming(false);
    }
  }, [ai, input, messages, streaming]);

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>local-llm</Text>
          <Pressable onPress={() => setShowDeviceInfo(!showDeviceInfo)}>
            <Text style={styles.infoButton}>i</Text>
          </Pressable>
        </View>

        {showDeviceInfo && <DeviceInfoCard />}

        {status && (
          <View style={styles.statusBar}>
            <Text style={styles.statusText}>{status}</Text>
            {downloadProgress !== null && (
              <View style={styles.progressBar}>
                <View style={[styles.progressFill, { width: `${downloadProgress}%` }]} />
              </View>
            )}
          </View>
        )}

        {/* Chat messages */}
        <ScrollView
          ref={scrollRef}
          style={styles.chatArea}
          contentContainerStyle={styles.chatContent}
          onContentSizeChange={() => scrollRef.current?.scrollToEnd({ animated: true })}
        >
          {messages.length === 0 && !ai && (
            <View style={styles.emptyState}>
              <Text style={styles.emptyTitle}>On-Device LLM</Text>
              <Text style={styles.emptySubtitle}>
                Run language models locally on your iPhone with Metal GPU acceleration.
              </Text>
              <Pressable
                style={[styles.loadButton, loading && styles.buttonDisabled]}
                onPress={loadModel}
                disabled={loading}
              >
                {loading ? (
                  <ActivityIndicator color="#fff" />
                ) : (
                  <Text style={styles.loadButtonText}>Download & Load Model</Text>
                )}
              </Pressable>
            </View>
          )}

          {messages.map((msg, i) => (
            <ChatBubble key={i} role={msg.role} content={msg.content} />
          ))}
        </ScrollView>

        {/* Input */}
        {ai && (
          <View style={styles.inputRow}>
            <TextInput
              style={styles.input}
              placeholder="Message..."
              placeholderTextColor="#999"
              value={input}
              onChangeText={setInput}
              onSubmitEditing={sendMessage}
              editable={!streaming}
              returnKeyType="send"
              multiline
            />
            <Pressable
              style={[styles.sendButton, (!input.trim() || streaming) && styles.sendDisabled]}
              onPress={sendMessage}
              disabled={!input.trim() || streaming}
            >
              {streaming ? (
                <ActivityIndicator color="#fff" size="small" />
              ) : (
                <Text style={styles.sendIcon}>↑</Text>
              )}
            </Pressable>
          </View>
        )}
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f2f2f7' },
  flex: { flex: 1 },
  header: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingHorizontal: 16, paddingVertical: 12, borderBottomWidth: 0.5, borderBottomColor: '#ddd',
    backgroundColor: '#fff',
  },
  title: { fontSize: 20, fontWeight: '700' },
  infoButton: {
    width: 28, height: 28, borderRadius: 14, backgroundColor: '#e5e5ea',
    textAlign: 'center', lineHeight: 28, fontSize: 14, fontWeight: '600', color: '#666',
  },
  statusBar: { paddingHorizontal: 16, paddingVertical: 8, backgroundColor: '#fff', borderBottomWidth: 0.5, borderBottomColor: '#ddd' },
  statusText: { fontSize: 12, color: '#666', textAlign: 'center' },
  progressBar: { height: 3, backgroundColor: '#e5e5ea', borderRadius: 1.5, marginTop: 6 },
  progressFill: { height: 3, backgroundColor: '#007AFF', borderRadius: 1.5 },
  chatArea: { flex: 1 },
  chatContent: { padding: 16, gap: 12 },
  emptyState: { alignItems: 'center', paddingTop: 80, paddingHorizontal: 32 },
  emptyTitle: { fontSize: 28, fontWeight: '700', marginBottom: 8 },
  emptySubtitle: { fontSize: 15, color: '#666', textAlign: 'center', lineHeight: 22, marginBottom: 32 },
  loadButton: { backgroundColor: '#007AFF', paddingHorizontal: 24, paddingVertical: 14, borderRadius: 12 },
  loadButtonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  buttonDisabled: { opacity: 0.5 },
  inputRow: {
    flexDirection: 'row', alignItems: 'flex-end', gap: 8,
    paddingHorizontal: 12, paddingVertical: 8, backgroundColor: '#fff',
    borderTopWidth: 0.5, borderTopColor: '#ddd',
  },
  input: {
    flex: 1, backgroundColor: '#f2f2f7', borderRadius: 20, paddingHorizontal: 16,
    paddingVertical: 10, fontSize: 16, maxHeight: 100,
  },
  sendButton: {
    width: 36, height: 36, borderRadius: 18, backgroundColor: '#007AFF',
    alignItems: 'center', justifyContent: 'center',
  },
  sendDisabled: { backgroundColor: '#c7c7cc' },
  sendIcon: { color: '#fff', fontSize: 18, fontWeight: '700' },
});
