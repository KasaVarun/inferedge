import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import {
  REQUIRED,
  OPTIONAL,
  assertSafeName,
  inspectExistingDestPath,
  syncPublications,
} from "./sync-public-artifacts.mjs";

const websiteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = path.resolve(websiteRoot, "..");
const publicRoot = path.join(websiteRoot, "public");
const publishedRoot = path.join(repoRoot, "docs", "benchmarks");
const scriptPath = path.join(websiteRoot, "scripts", "sync-public-artifacts.mjs");

function sha256(filePath) {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

function snapshotPublic() {
  const files = [];
  for (const dirName of ["data", "figures"]) {
    const dir = path.join(publicRoot, dirName);
    if (!fs.existsSync(dir)) {
      continue;
    }
    for (const name of fs.readdirSync(dir)) {
      const full = path.join(dir, name);
      const stat = fs.lstatSync(full);
      files.push({
        full,
        name,
        dirName,
        hash: stat.isFile() ? sha256(full) : null,
        mtimeNs: stat.mtimeNs,
        size: stat.size,
      });
    }
  }
  return files;
}

function runCli(args) {
  return spawnSync(process.execPath, [scriptPath, ...args], {
    encoding: "utf8",
    cwd: websiteRoot,
  });
}

function makeTempDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "inferedge-sync-"));
}

test("unexpected filename is rejected", () => {
  assert.throws(() => assertSafeName("../secret.json"), /path-like/);
  assert.throws(() => assertSafeName("a/b.json"), /path-like/);
  assert.throws(() => assertSafeName(".."), /path-like/);
});

test("destination symlink is rejected", () => {
  const dir = makeTempDir();
  const target = path.join(dir, "target");
  const link = path.join(dir, "link");
  fs.mkdirSync(target);
  fs.symlinkSync(target, link);
  assert.throws(() => inspectExistingDestPath(link, "Destination root", "directory"), /symbolic link/);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("destination escape is rejected", () => {
  const before = snapshotPublic();
  const escaped = runCli(["--dest", path.join(websiteRoot, "..", "docs")]);
  assert.notEqual(escaped.status, 0);
  assert.match(escaped.stderr, /Destination must be/);
  const tmp = runCli(["--dest", os.tmpdir()]);
  assert.notEqual(tmp.status, 0);
  assert.deepEqual(snapshotPublic(), before);
});

test("malformed source fails closed", () => {
  const before = snapshotPublic();
  const missingValue = runCli(["--source"]);
  assert.notEqual(missingValue.status, 0);
  const unknown = runCli(["--wat"]);
  assert.notEqual(unknown.status, 0);
  const missingDir = runCli(["--source", path.join(os.tmpdir(), "inferedge-does-not-exist")]);
  assert.notEqual(missingDir.status, 0);
  const asFile = makeTempDir();
  const filePath = path.join(asFile, "not-a-dir");
  fs.writeFileSync(filePath, "x");
  const fileSource = runCli(["--source", filePath]);
  assert.notEqual(fileSource.status, 0);
  fs.rmSync(asFile, { recursive: true, force: true });
  assert.deepEqual(snapshotPublic(), before);
});

test("required source missing causes zero writes", () => {
  const before = snapshotPublic();
  const empty = makeTempDir();
  const result = runCli(["--source", empty]);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /Required publication source is missing/);
  assert.deepEqual(snapshotPublic(), before);
  fs.rmSync(empty, { recursive: true, force: true });
});

test("source directory symlink is rejected with no writes", () => {
  const before = snapshotPublic();
  const dir = makeTempDir();
  const link = path.join(dir, "src-link");
  fs.symlinkSync(publishedRoot, link);
  const result = runCli(["--source", link]);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /symbolic link/);
  assert.deepEqual(snapshotPublic(), before);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("allowlisted source symlink is rejected with no writes", () => {
  const before = snapshotPublic();
  const src = makeTempDir();
  const secret = path.join(src, "secret.bin");
  fs.writeFileSync(secret, "SECRET");
  fs.symlinkSync(secret, path.join(src, "arrival-aware-sparse-gemm.md"));
  const result = runCli(["--source", src]);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /symbolic link/);
  assert.deepEqual(snapshotPublic(), before);
  fs.rmSync(src, { recursive: true, force: true });
});

const ORIGINAL_EIGHT = Object.freeze([
  { name: "arrival-aware-sparse-gemm.md", destDir: "data" },
  { name: "arrival-aware-sparse-gemm-results.json", destDir: "data" },
  { name: "arrival-aware-sparse-gemm-p50.png", destDir: "figures" },
  { name: "arrival-aware-sparse-gemm-paired-deltas.png", destDir: "figures" },
  { name: "neural-batching-gemm.md", destDir: "data" },
  { name: "neural-batching-gemm-results.json", destDir: "data" },
  { name: "adaptive-neural-batching-gemm.md", destDir: "data" },
  { name: "adaptive-neural-batching-gemm-results.json", destDir: "data" },
]);

const GENERATIVE_FILES = Object.freeze([
  "generative-overhead-qwen3-0.6b.md",
  "generative-overhead-qwen3-0.6b-results.json",
]);

function stageRequiredSource(dir) {
  for (const entry of REQUIRED) {
    fs.copyFileSync(path.join(publishedRoot, entry.name), path.join(dir, entry.name));
  }
}

const EXPECTED_PUBLICATION_FILES = Object.freeze([
  ...ORIGINAL_EIGHT,
  ...GENERATIVE_FILES.map((name) => ({ name, destDir: "data" })),
]);

function publicInventory() {
  const files = [];
  for (const dirName of ["data", "figures"]) {
    const dir = path.join(publicRoot, dirName);
    if (!fs.existsSync(dir)) {
      continue;
    }
    for (const name of fs.readdirSync(dir)) {
      const full = path.join(dir, name);
      const stat = fs.lstatSync(full);
      files.push({ dirName, name, symlink: stat.isSymbolicLink(), directory: stat.isDirectory() });
    }
  }
  return files;
}

function ensurePublicDir(dirName) {
  const dir = path.join(publicRoot, dirName);
  fs.mkdirSync(dir, { recursive: true });
  const stat = fs.lstatSync(dir);
  if (stat.isSymbolicLink() || !stat.isDirectory()) {
    throw new Error(`Refusing non-directory public fixture: ${dir}`);
  }
  return dir;
}

function removePathNoFollow(target) {
  let stat;
  try {
    stat = fs.lstatSync(target);
  } catch (error) {
    if (error && error.code === "ENOENT") {
      return;
    }
    throw error;
  }
  if (stat.isSymbolicLink() || stat.isFile()) {
    fs.unlinkSync(target);
    return;
  }
  if (stat.isDirectory()) {
    fs.rmSync(target, { recursive: true, force: true });
    return;
  }
  throw new Error(`Refusing to remove unexpected path: ${target}`);
}

test("ten curated publication artifacts copy byte-identically", () => {
  assert.equal(REQUIRED.length, 6);
  assert.equal(OPTIONAL.length, 4);
  const expectedNames = EXPECTED_PUBLICATION_FILES.map((entry) => `${entry.destDir}/${entry.name}`).sort();
  const implemented = [...REQUIRED, ...OPTIONAL].map((entry) => `${entry.destDir}/${entry.name}`).sort();
  assert.deepEqual(implemented, expectedNames);

  const copied = syncPublications([]);
  assert.equal(copied.length, 10);
  for (const entry of EXPECTED_PUBLICATION_FILES) {
    const source = path.join(publishedRoot, entry.name);
    const dest = path.join(publicRoot, entry.destDir, entry.name);
    assert.equal(fs.lstatSync(dest).isSymbolicLink(), false);
    assert.equal(fs.lstatSync(dest).isFile(), true);
    assert.equal(sha256(source), sha256(dest));
    assert.deepEqual(fs.readFileSync(source), fs.readFileSync(dest));
  }
  assert.deepEqual(fs.readdirSync(publicRoot).sort(), ["data", "figures"]);
  const inventory = publicInventory();
  assert.equal(inventory.length, 10);
  assert.equal(inventory.every((item) => !item.symlink && !item.directory), true);
  assert.deepEqual(inventory.map((item) => `${item.dirName}/${item.name}`).sort(), expectedNames);
  assert.equal(inventory.filter((item) => item.dirName === "data").length, 8);
  assert.equal(inventory.filter((item) => item.dirName === "figures").length, 2);
});

test("missing generative markdown fails with no writes", () => {
  const before = snapshotPublic();
  const src = makeTempDir();
  stageRequiredSource(src);
  fs.unlinkSync(path.join(src, "generative-overhead-qwen3-0.6b.md"));
  const result = runCli(["--source", src]);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /generative-overhead-qwen3-0\.6b\.md/);
  assert.deepEqual(snapshotPublic(), before);
  fs.rmSync(src, { recursive: true, force: true });
});

test("missing generative json fails with no writes", () => {
  const before = snapshotPublic();
  const src = makeTempDir();
  stageRequiredSource(src);
  fs.unlinkSync(path.join(src, "generative-overhead-qwen3-0.6b-results.json"));
  const result = runCli(["--source", src]);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /generative-overhead-qwen3-0\.6b-results\.json/);
  assert.deepEqual(snapshotPublic(), before);
  fs.rmSync(src, { recursive: true, force: true });
});

test("source symlink for either new generative file fails with no writes", () => {
  for (const name of GENERATIVE_FILES) {
    const before = snapshotPublic();
    const src = makeTempDir();
    stageRequiredSource(src);
    const secret = path.join(src, "secret.bin");
    fs.writeFileSync(secret, "SECRET");
    fs.unlinkSync(path.join(src, name));
    fs.symlinkSync(secret, path.join(src, name));
    const result = runCli(["--source", src]);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /symbolic link/);
    assert.equal(fs.readFileSync(secret, "utf8"), "SECRET");
    assert.deepEqual(snapshotPublic(), before);
    fs.rmSync(src, { recursive: true, force: true });
  }
});

test("destination symlink for a new generative file is rejected", () => {
  const outside = makeTempDir();
  const secret = path.join(outside, "secret");
  fs.writeFileSync(secret, "SECRET");
  const link = path.join(ensurePublicDir("data"), "generative-overhead-qwen3-0.6b-results.json");
  removePathNoFollow(link);
  fs.symlinkSync(secret, link);
  try {
    const result = runCli([]);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /symbolic link/);
    assert.equal(fs.readFileSync(secret, "utf8"), "SECRET");
    assert.equal(fs.lstatSync(link).isSymbolicLink(), true);
    assert.equal(fs.readlinkSync(link), secret);
  } finally {
    removePathNoFollow(link);
    fs.rmSync(outside, { recursive: true, force: true });
    syncPublications([]);
  }
  const dest = path.join(publicRoot, "data", "generative-overhead-qwen3-0.6b-results.json");
  assert.equal(fs.lstatSync(dest).isSymbolicLink(), false);
  assert.equal(fs.lstatSync(dest).isFile(), true);
  for (const entry of EXPECTED_PUBLICATION_FILES) {
    const source = path.join(publishedRoot, entry.name);
    const generated = path.join(publicRoot, entry.destDir, entry.name);
    assert.deepEqual(fs.readFileSync(source), fs.readFileSync(generated));
  }
});

test("unallowlisted files are not copied and unexpected destinations are pruned", () => {
  const src = makeTempDir();
  const planted = path.join(publicRoot, "data", "not-a-publication.json");
  const extraDir = path.join(publicRoot, "data", "extra-dir");
  try {
    stageRequiredSource(src);
    fs.writeFileSync(path.join(src, "not-a-publication.json"), "{\"no\":true}\n");
    ensurePublicDir("data");
    removePathNoFollow(planted);
    removePathNoFollow(extraDir);
    fs.writeFileSync(planted, "PLANTED");
    fs.mkdirSync(extraDir);
    fs.writeFileSync(path.join(extraDir, "note.txt"), "nope");
    const result = runCli(["--source", src]);
    assert.equal(result.status, 0);
    assert.equal(fs.existsSync(planted), false);
    assert.equal(fs.existsSync(extraDir), false);
    const expected = REQUIRED.map((entry) => `${entry.destDir}/${entry.name}`).sort();
    const actual = [];
    assert.deepEqual(fs.readdirSync(publicRoot).sort(), ["data", "figures"]);
    for (const dirName of ["data", "figures"]) {
      for (const name of fs.readdirSync(path.join(publicRoot, dirName))) {
        const full = path.join(publicRoot, dirName, name);
        const stat = fs.lstatSync(full);
        assert.equal(stat.isSymbolicLink(), false);
        assert.equal(stat.isDirectory(), false);
        assert.equal(stat.isFile(), true);
        actual.push(`${dirName}/${name}`);
        assert.deepEqual(fs.readFileSync(full), fs.readFileSync(path.join(src, name)));
      }
    }
    assert.deepEqual(actual.sort(), expected);
    assert.equal(actual.includes("data/not-a-publication.json"), false);
  } finally {
    fs.rmSync(src, { recursive: true, force: true });
    removePathNoFollow(planted);
    removePathNoFollow(extraDir);
    syncPublications([]);
  }
});

test("existing generated copies are overwritten only in the approved output area", () => {
  const marker = path.join(ensurePublicDir("data"), "arrival-aware-sparse-gemm.md");
  const resolvedMarker = path.resolve(marker);
  const resolvedPublic = path.resolve(publicRoot);
  assert.ok(resolvedMarker.startsWith(`${resolvedPublic}${path.sep}`));
  removePathNoFollow(marker);
  fs.writeFileSync(marker, "STALE");
  syncPublications([]);
  const restored = fs.readFileSync(marker);
  const source = fs.readFileSync(path.join(publishedRoot, "arrival-aware-sparse-gemm.md"));
  assert.deepEqual(restored, source);
  assert.notEqual(restored.toString("utf8"), "STALE");
  assert.equal(path.resolve(marker).startsWith(`${resolvedPublic}${path.sep}`), true);
  const websiteFiles = fs.readdirSync(websiteRoot);
  assert.equal(websiteFiles.includes("arrival-aware-sparse-gemm.md"), false);
});
