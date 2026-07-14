import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

const minimumVersion = '0.3.0';
const manifestPath = resolve(process.argv[2] ?? 'package.json');
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
const declaredVersion = manifest.dependencies?.['local-llm-js-core'];

function parseExactVersion(version, source) {
  const match = /^(\d+)\.(\d+)\.(\d+)$/.exec(version ?? '');
  if (!match) {
    throw new Error(`${source} must pin local-llm-js-core to an exact semantic version; received ${JSON.stringify(version)}`);
  }
  return match.slice(1).map(Number);
}

function compareVersions(left, right) {
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) return left[index] - right[index];
  }
  return 0;
}

const declared = parseExactVersion(declaredVersion, manifestPath);
const minimum = parseExactVersion(minimumVersion, 'minimum core version');
if (compareVersions(declared, minimum) < 0) {
  throw new Error(`local-llm-js-core ${declaredVersion} is ABI-incompatible; release ${minimumVersion} or newer is required`);
}

const installedManifestPath = resolve(dirname(manifestPath), 'node_modules/local-llm-js-core/package.json');
const installedManifest = JSON.parse(readFileSync(installedManifestPath, 'utf8'));
if (installedManifest.version !== declaredVersion) {
  throw new Error(`installed local-llm-js-core ${installedManifest.version} does not match the ${declaredVersion} package pin`);
}

console.log(`Verified local-llm-js-core ${declaredVersion}`);
