import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const packageJson = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8')) as {
  version: string;
};
const podspec = readFileSync(new URL('../local-llm-rn.podspec', import.meta.url), 'utf8');

describe('package metadata', () => {
  it('derives the CocoaPods version and v-prefixed source tag from package.json', () => {
    expect(podspec).toContain('s.version      = package["version"]');
    expect(podspec).toContain(' :tag => "v#{s.version}"');
    expect(packageJson.version).toMatch(/^\d+\.\d+\.\d+$/);
  });
});
