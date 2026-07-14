import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const args = process.argv.slice(2).filter((arg) => arg !== '--');
const tag = args.find((arg) => arg !== '--check-registry') ?? process.env.GITHUB_REF_NAME;
const checkRegistry = args.includes('--check-registry');
if (!tag?.startsWith('v') || !/^\d+\.\d+\.\d+$/.test(tag.slice(1))) {
  throw new Error(`expected a v-prefixed semantic version tag; received ${JSON.stringify(tag)}`);
}

const manifest = JSON.parse(readFileSync('package.json', 'utf8'));
const version = tag.slice(1);
if (manifest.version !== version) {
  throw new Error(`package.json has version ${manifest.version}; expected ${version} from tag ${tag}`);
}

if (checkRegistry) {
  const spec = `${manifest.name}@${version}`;
  const result = spawnSync('npm', ['view', spec, 'version'], { encoding: 'utf8' });
  if (result.status === 0) {
    throw new Error(`${spec} is already published`);
  }

  const output = `${result.stdout ?? ''}\n${result.stderr ?? ''}`;
  if (!/E404|404 Not Found/i.test(output)) {
    process.stderr.write(output);
    throw new Error(`could not verify npm availability for ${spec}`);
  }
}

console.log(`Validated ${manifest.name}@${version}`);
