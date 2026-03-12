#!/usr/bin/env node

import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const pkgPath = "package.json";

function run(cmd, args, opts = {}) {
  const res = spawnSync(cmd, args, {
    cwd: repoRoot,
    stdio: opts.capture ? ["ignore", "pipe", "pipe"] : "inherit",
    encoding: "utf8",
  });
  if (res.status !== 0) {
    if (opts.capture) {
      const stderr = (res.stderr || "").trim();
      const stdout = (res.stdout || "").trim();
      if (stdout) console.error(stdout);
      if (stderr) console.error(stderr);
    }
    throw new Error(`Command failed: ${cmd} ${args.join(" ")}`);
  }
  return (res.stdout || "").trim();
}

function readJson(relPath) {
  return JSON.parse(readFileSync(path.join(repoRoot, relPath), "utf8"));
}

function writeJson(relPath, obj) {
  writeFileSync(path.join(repoRoot, relPath), `${JSON.stringify(obj, null, 2)}\n`, "utf8");
}

function parseSemver(v) {
  const m = /^(\d+)\.(\d+)\.(\d+)$/.exec(v);
  if (!m) return null;
  return { major: Number(m[1]), minor: Number(m[2]), patch: Number(m[3]) };
}

function bump(v, kind) {
  const p = parseSemver(v);
  if (!p) throw new Error(`Unsupported version format: ${v}`);
  if (kind === "patch") return `${p.major}.${p.minor}.${p.patch + 1}`;
  if (kind === "minor") return `${p.major}.${p.minor + 1}.0`;
  if (kind === "major") return `${p.major + 1}.0.0`;
  throw new Error(`Unsupported bump kind: ${kind}`);
}

function ensureCleanGit() {
  const status = run("git", ["status", "--porcelain"], { capture: true });
  if (status) throw new Error("Git working tree is not clean. Commit or stash changes first.");
}

function ensureMainBranch() {
  const branch = run("git", ["branch", "--show-current"], { capture: true });
  if (branch !== "main") throw new Error(`Current branch is '${branch}'. Switch to 'main' before releasing.`);
}

async function main() {
  const rl = createInterface({ input, output });
  try {
    const pkg = readJson(pkgPath);
    const current = pkg.version;
    console.log(`Current version: ${current}`);
    console.log("Select release type:");
    console.log("1) patch");
    console.log("2) minor");
    console.log("3) major");
    console.log("4) custom");

    const choice = (await rl.question("Choice [1]: ")).trim() || "1";
    let nextVersion;
    if (choice === "1") nextVersion = bump(current, "patch");
    else if (choice === "2") nextVersion = bump(current, "minor");
    else if (choice === "3") nextVersion = bump(current, "major");
    else if (choice === "4") {
      const custom = (await rl.question("Enter version (x.y.z): ")).trim();
      if (!parseSemver(custom)) throw new Error(`Invalid version: ${custom}`);
      nextVersion = custom;
    } else {
      throw new Error(`Invalid choice: ${choice}`);
    }

    const tag = `v${nextVersion}`;
    console.log(`\nPlanned release: ${nextVersion}`);
    console.log(`Tag: ${tag}`);
    const confirm = (await rl.question("Proceed? [y/N]: ")).trim().toLowerCase();
    if (confirm !== "y" && confirm !== "yes") {
      console.log("Cancelled.");
      return;
    }

    ensureCleanGit();
    ensureMainBranch();
    run("git", ["pull", "--rebase"]);

    pkg.version = nextVersion;
    writeJson(pkgPath, pkg);

    run("git", ["add", pkgPath]);
    run("git", ["commit", "-m", `chore(release): v${nextVersion}`]);
    run("git", ["tag", "-a", tag, "-m", tag]);
    run("git", ["push", "origin", "HEAD"]);
    run("git", ["push", "origin", tag]);

    console.log(`\nRelease pushed: ${nextVersion}`);
    console.log("Now monitor CI release workflow for npm publish.");
  } finally {
    rl.close();
  }
}

main().catch((err) => {
  console.error(`Release failed: ${err.message}`);
  process.exit(1);
});
