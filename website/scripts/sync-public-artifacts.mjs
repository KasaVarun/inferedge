#!/usr/bin/env node
/**
 * Copy only allowlisted files from docs/benchmarks/ into website/public/.
 * Preflight all paths, then write temps, then rename. No symlink follow.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const websiteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = path.resolve(websiteRoot, "..");
const PUBLIC_ROOT = path.resolve(websiteRoot, "public");
const ALLOWED_DEST_DIRS = Object.freeze(["data", "figures"]);

export const REQUIRED = Object.freeze([
  { name: "arrival-aware-sparse-gemm.md", destDir: "data" },
  { name: "arrival-aware-sparse-gemm-results.json", destDir: "data" },
  { name: "arrival-aware-sparse-gemm-p50.png", destDir: "figures" },
  { name: "arrival-aware-sparse-gemm-paired-deltas.png", destDir: "figures" },
  { name: "generative-overhead-qwen3-0.6b.md", destDir: "data" },
  { name: "generative-overhead-qwen3-0.6b-results.json", destDir: "data" },
]);

export const OPTIONAL = Object.freeze([
  { name: "neural-batching-gemm.md", destDir: "data" },
  { name: "neural-batching-gemm-results.json", destDir: "data" },
  { name: "adaptive-neural-batching-gemm.md", destDir: "data" },
  { name: "adaptive-neural-batching-gemm-results.json", destDir: "data" },
]);

export function assertSafeName(name) {
  if (typeof name !== "string" || name.length === 0) {
    throw new Error("Refusing empty artifact name");
  }
  if (name.includes("\0") || name.includes("/") || name.includes("\\") || name.includes("..")) {
    throw new Error(`Refusing path-like artifact name: ${name}`);
  }
  if (path.basename(name) !== name) {
    throw new Error(`Refusing path-like artifact name: ${name}`);
  }
}

function parseArgs(argv) {
  let sourceDir = path.join(repoRoot, "docs", "benchmarks");
  let destPublic = PUBLIC_ROOT;
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (flag === "--source") {
      if (typeof value !== "string" || value.length === 0 || value.startsWith("--")) {
        throw new Error("Malformed --source argument");
      }
      sourceDir = value;
      i += 1;
    } else if (flag === "--dest") {
      if (typeof value !== "string" || value.length === 0 || value.startsWith("--")) {
        throw new Error("Malformed --dest argument");
      }
      destPublic = value;
      i += 1;
    } else {
      throw new Error(`Unknown or incomplete argument: ${flag}`);
    }
  }
  if (sourceDir.includes("\0") || destPublic.includes("\0")) {
    throw new Error("Malformed CLI path");
  }
  return { sourceDir, destPublic };
}

function lstatOrThrow(target, label) {
  try {
    return fs.lstatSync(target);
  } catch (error) {
    const code = error && typeof error === "object" && "code" in error ? error.code : undefined;
    if (code === "ENOENT") {
      throw new Error(`${label} is missing: ${target}`);
    }
    throw error;
  }
}

function assertNotSymlink(stat, label, target) {
  if (stat.isSymbolicLink()) {
    throw new Error(`${label} is a symbolic link: ${target}`);
  }
}

function assertRegularFile(stat, label, target) {
  assertNotSymlink(stat, label, target);
  if (!stat.isFile()) {
    throw new Error(`${label} is not a regular file: ${target}`);
  }
}

function assertDirectory(stat, label, target) {
  assertNotSymlink(stat, label, target);
  if (!stat.isDirectory()) {
    throw new Error(`${label} is not a directory: ${target}`);
  }
}

function assertResolvedInside(child, parent, label) {
  const relative = path.relative(parent, child);
  if (relative === "" ) {
    return;
  }
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`${label} escapes ${parent}: ${child}`);
  }
}

function resolveConfinedDest(destPublic) {
  const requested = path.resolve(destPublic);
  if (requested !== PUBLIC_ROOT) {
    throw new Error(`Destination must be ${PUBLIC_ROOT}`);
  }
  return requested;
}

function readRegularFileBytes(filePath) {
  const stat = lstatOrThrow(filePath, "Source file");
  assertRegularFile(stat, "Source file", filePath);
  const flags = fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW ?? 0);
  const fd = fs.openSync(filePath, flags);
  try {
    const bytes = Buffer.alloc(stat.size);
    let offset = 0;
    while (offset < bytes.length) {
      const n = fs.readSync(fd, bytes, offset, bytes.length - offset, offset);
      if (n === 0) {
        break;
      }
      offset += n;
    }
    if (offset !== stat.size) {
      throw new Error(`Short read of ${filePath}`);
    }
    return bytes;
  } finally {
    fs.closeSync(fd);
  }
}

function inspectSourceFile(sourceDir, entry, required) {
  assertSafeName(entry.name);
  if (!ALLOWED_DEST_DIRS.includes(entry.destDir)) {
    throw new Error(`Unexpected destination directory: ${entry.destDir}`);
  }
  const sourcePath = path.join(sourceDir, entry.name);
  assertResolvedInside(sourcePath, sourceDir, "Source file");
  let stat;
  try {
    stat = fs.lstatSync(sourcePath);
  } catch (error) {
    const code = error && typeof error === "object" && "code" in error ? error.code : undefined;
    if (code === "ENOENT") {
      if (required) {
        throw new Error(`Required publication source is missing: ${sourcePath}`);
      }
      return null;
    }
    throw error;
  }
  assertRegularFile(stat, required ? "Required source" : "Optional source", sourcePath);
  const destDir = path.join(PUBLIC_ROOT, entry.destDir);
  const destPath = path.join(destDir, entry.name);
  assertResolvedInside(destDir, PUBLIC_ROOT, "Destination directory");
  assertResolvedInside(destPath, destDir, "Destination file");
  return { entry, sourcePath, destDir, destPath };
}

export function inspectExistingDestPath(target, label, kind) {
  let stat;
  try {
    stat = fs.lstatSync(target);
  } catch (error) {
    const code = error && typeof error === "object" && "code" in error ? error.code : undefined;
    if (code === "ENOENT") {
      return;
    }
    throw error;
  }
  assertNotSymlink(stat, label, target);
  if (kind === "directory" && !stat.isDirectory()) {
    throw new Error(`${label} is not a directory: ${target}`);
  }
  if (kind === "file" && !stat.isFile()) {
    throw new Error(`${label} is not a regular file: ${target}`);
  }
}

function removeRealTree(target) {
  const stat = fs.lstatSync(target);
  if (stat.isSymbolicLink()) {
    throw new Error(`Refusing to prune through a symbolic link: ${target}`);
  }
  if (stat.isDirectory()) {
    for (const name of fs.readdirSync(target)) {
      removeRealTree(path.join(target, name));
    }
    fs.rmdirSync(target);
    return;
  }
  if (stat.isFile()) {
    fs.unlinkSync(target);
    return;
  }
  throw new Error(`Refusing unexpected destination entry: ${target}`);
}

function pruneUnexpectedDestinations(destRoot, planned) {
  const allowedByDir = new Map(ALLOWED_DEST_DIRS.map((dirName) => [dirName, new Set()]));
  for (const item of planned) {
    allowedByDir.get(item.entry.destDir).add(item.entry.name);
  }
  const rootStat = lstatOrThrow(destRoot, "Destination root");
  assertDirectory(rootStat, "Destination root", destRoot);
  for (const name of fs.readdirSync(destRoot)) {
    const full = path.join(destRoot, name);
    if (!ALLOWED_DEST_DIRS.includes(name)) {
      removeRealTree(full);
      continue;
    }
    const dirStat = lstatOrThrow(full, "Destination subdirectory");
    assertDirectory(dirStat, "Destination subdirectory", full);
    const allowed = allowedByDir.get(name);
    for (const childName of fs.readdirSync(full)) {
      const child = path.join(full, childName);
      const childStat = fs.lstatSync(child);
      if (childStat.isSymbolicLink()) {
        throw new Error(`Destination is a symbolic link: ${child}`);
      }
      if (allowed.has(childName)) {
        if (!childStat.isFile()) {
          throw new Error(`Destination file is not a regular file: ${child}`);
        }
        continue;
      }
      removeRealTree(child);
    }
  }
}

function writeTempThenRename(sourcePath, destPath, destDir) {
  const bytes = readRegularFileBytes(sourcePath);
  const tempPath = path.join(
    destDir,
    `.${path.basename(destPath)}.${process.pid}.${Date.now()}.${Math.random().toString(16).slice(2)}.tmp`,
  );
  assertResolvedInside(tempPath, destDir, "Temporary destination");
  const flags = fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_EXCL;
  const fd = fs.openSync(tempPath, flags, 0o644);
  try {
    let offset = 0;
    while (offset < bytes.length) {
      offset += fs.writeSync(fd, bytes, offset, bytes.length - offset);
    }
    fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
  }
  const written = fs.readFileSync(tempPath);
  if (!written.equals(bytes)) {
    fs.unlinkSync(tempPath);
    throw new Error(`Byte mismatch after copy: ${destPath}`);
  }
  return { tempPath, destPath, bytes };
}

export function syncPublications(argv = process.argv.slice(2)) {
  const { sourceDir: sourceArg, destPublic } = parseArgs(argv);
  const destRoot = resolveConfinedDest(destPublic);
  const sourceDir = path.resolve(sourceArg);

  const sourceStat = lstatOrThrow(sourceDir, "Publication source directory");
  assertDirectory(sourceStat, "Publication source directory", sourceDir);

  inspectExistingDestPath(destRoot, "Destination root", "directory");
  for (const dirName of ALLOWED_DEST_DIRS) {
    inspectExistingDestPath(path.join(destRoot, dirName), "Destination subdirectory", "directory");
  }

  const planned = [];
  const destSeen = new Set();
  for (const entry of REQUIRED) {
    const item = inspectSourceFile(sourceDir, entry, true);
    if (item === null) {
      throw new Error(`Required publication source is missing: ${entry.name}`);
    }
    if (destSeen.has(item.destPath)) {
      throw new Error(`Duplicate destination mapping: ${item.destPath}`);
    }
    destSeen.add(item.destPath);
    planned.push(item);
  }
  for (const entry of OPTIONAL) {
    const item = inspectSourceFile(sourceDir, entry, false);
    if (item === null) {
      continue;
    }
    if (destSeen.has(item.destPath)) {
      throw new Error(`Duplicate destination mapping: ${item.destPath}`);
    }
    destSeen.add(item.destPath);
    planned.push(item);
  }

  if (!fs.existsSync(destRoot)) {
    fs.mkdirSync(destRoot, { recursive: false, mode: 0o755 });
  }
  const publicStat = lstatOrThrow(destRoot, "Destination root");
  assertDirectory(publicStat, "Destination root", destRoot);

  for (const dirName of ALLOWED_DEST_DIRS) {
    const destDir = path.join(destRoot, dirName);
    if (!fs.existsSync(destDir)) {
      fs.mkdirSync(destDir, { recursive: false, mode: 0o755 });
    }
    const dirStat = lstatOrThrow(destDir, "Destination subdirectory");
    assertDirectory(dirStat, "Destination subdirectory", destDir);
  }

  const temps = [];
  try {
    for (const item of planned) {
      inspectExistingDestPath(item.destPath, "Destination file", "file");
      temps.push(writeTempThenRename(item.sourcePath, item.destPath, item.destDir));
    }
    for (const temp of temps) {
      fs.renameSync(temp.tempPath, temp.destPath);
      const finalBytes = fs.readFileSync(temp.destPath);
      if (!finalBytes.equals(temp.bytes)) {
        throw new Error(`Byte mismatch after rename: ${temp.destPath}`);
      }
    }
    pruneUnexpectedDestinations(destRoot, planned);
  } catch (error) {
    for (const temp of temps) {
      try {
        if (fs.existsSync(temp.tempPath)) {
          fs.unlinkSync(temp.tempPath);
        }
      } catch {
        // Best-effort cleanup of unpublished temps only.
      }
    }
    throw error;
  }

  return planned.map((item) => path.relative(websiteRoot, item.destPath));
}

function isMain() {
  const entry = process.argv[1];
  if (typeof entry !== "string") {
    return false;
  }
  return pathToFileURL(path.resolve(entry)).href === import.meta.url;
}

if (isMain()) {
  try {
    const copied = syncPublications();
    const requiredCount = REQUIRED.length;
    const optionalCount = copied.length - requiredCount;
    process.stdout.write(
      `Copied ${requiredCount} required and ${optionalCount} optional publications.\n`,
    );
    for (const file of copied) {
      process.stdout.write(`  ${file}\n`);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`error: ${message}\n`);
    process.exit(1);
  }
}
