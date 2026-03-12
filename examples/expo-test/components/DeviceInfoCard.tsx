import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { getDeviceCapabilities, recommendQuantization } from 'local-llm-rn';

export function DeviceInfoCard() {
  try {
    const caps = getDeviceCapabilities();
    const quant = recommendQuantization();

    const rows = [
      ['Total RAM', `${(caps.totalRAM / 1e9).toFixed(1)} GB`],
      ['Available RAM', `${(caps.availableRAM / 1e9).toFixed(1)} GB`],
      ['GPU', caps.gpuName],
      ['Metal Family', `Apple ${caps.metalFamily}`],
      ['Metal Version', `${caps.metalVersion}`],
      ['iOS Version', caps.iosVersion],
      ['Low Power Mode', caps.isLowPowerMode ? 'Yes' : 'No'],
      ['Recommended Quant', quant],
    ];

    return (
      <View style={styles.card}>
        {rows.map(([label, value]) => (
          <View key={label} style={styles.row}>
            <Text style={styles.label}>{label}</Text>
            <Text style={styles.value}>{value}</Text>
          </View>
        ))}
      </View>
    );
  } catch {
    return (
      <View style={styles.card}>
        <Text style={styles.label}>Device info unavailable (native module not loaded)</Text>
      </View>
    );
  }
}

const styles = StyleSheet.create({
  card: {
    marginHorizontal: 16, marginVertical: 8, padding: 12,
    backgroundColor: '#fff', borderRadius: 12,
    shadowColor: '#000', shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05, shadowRadius: 4,
  },
  row: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 4 },
  label: { fontSize: 13, color: '#666' },
  value: { fontSize: 13, fontWeight: '500', color: '#000' },
});
