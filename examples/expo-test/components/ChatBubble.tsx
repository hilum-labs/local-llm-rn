import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

interface Props {
  role: 'user' | 'assistant';
  content: string;
}

export function ChatBubble({ role, content }: Props) {
  const isUser = role === 'user';
  return (
    <View style={[styles.row, isUser && styles.rowUser]}>
      <View style={[styles.bubble, isUser ? styles.bubbleUser : styles.bubbleAssistant]}>
        <Text style={[styles.text, isUser && styles.textUser]}>
          {content || (isUser ? '' : '...')}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: 'row', justifyContent: 'flex-start' },
  rowUser: { justifyContent: 'flex-end' },
  bubble: { maxWidth: '80%', paddingHorizontal: 14, paddingVertical: 10, borderRadius: 18 },
  bubbleUser: { backgroundColor: '#007AFF', borderBottomRightRadius: 4 },
  bubbleAssistant: { backgroundColor: '#e5e5ea', borderBottomLeftRadius: 4 },
  text: { fontSize: 15, lineHeight: 21, color: '#000' },
  textUser: { color: '#fff' },
});
