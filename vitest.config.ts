import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    clearMocks: true,
    restoreMocks: true,
    include: ['tests/**/*.test.ts'],
    server: {
      deps: {
        // react-native uses Flow types that Vite's parser cannot handle.
        // Externalizing it lets vi.mock() provide the implementation.
        external: ['react-native'],
      },
    },
  },
});
