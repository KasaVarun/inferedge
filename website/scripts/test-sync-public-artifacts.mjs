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

test("all eight legitimate published files copy byte-identically", () => {
  const copied = syncPublications([]);
  assert.equal(copied.length, REQUIRED.length + OPTIONAL.length);
  for (const entry of [...REQUIRED, ...OPTIONAL]) {
    const source = path.join(publishedRoot, entry.name);
    const dest = path.join(publicRoot, entry.destDir, entry.name);
    assert.equal(sha256(source), sha256(dest));
    assert.deepEqual(fs.readFileSync(source), fs.readFileSync(dest));
  }
});

test("existing generated copies are overwritten only in the approved output area", () => {
  const marker = path.join(publicRoot, "data", "arrival-aware-sparse-gemm.md");
  fs.writeFileSync(marker, "STALE");
  syncPublications([]);
  const restored = fs.readFileSync(marker);
  const source = fs.readFileSync(path.join(publishedRoot, "arrival-aware-sparse-gemm.md"));
  assert.deepEqual(restored, source);
  const websiteFiles = fs.readdirSync(websiteRoot);
  assert.ok(!websiteFiles.includes("arrival-aware-sparse-gemm.md"));
});
