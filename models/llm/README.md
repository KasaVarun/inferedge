# InferEdge local LLM packages

This directory documents the experimental generative path. The feature is
default OFF. Ordinary InferEdge builds, CI, and the float32 ONNX engine do
not use it.

## Status

- Experimental. `INFEREDGE_ENABLE_GENERATIVE` defaults to OFF.
- ONNX Runtime GenAI 0.17.0 is rejected and must not be introduced.
- No model weights, GGUF files, or converted packages belong in Git or Pages.
- No model downloads automatically. `trust_remote_code` remains false.
- Qwen is the first planned manual integration. Llama requires a separate
  license-acceptance decision and is not enabled by this phase.
- Phase 9-B1 does not claim that a real Qwen or Llama model has passed.
- Phase 9-B2 will perform real-model correctness. Benchmarks are a later
  audited phase. Offline learning is later. No performance claims are
  authorized here.

## llama.cpp pin

The only accepted revision is:

- Project: llama.cpp
- Tag: `v0.7.0`
- Commit: `620d2225cc436ab0b001e3200e590e3b286ef9ab`
- Source archive: `https://github.com/ggml-org/llama.cpp/archive/refs/tags/v0.7.0.tar.gz`
- Source archive SHA-256: `55d67acf3ad2ad747a9bbeb75f9f0e173dbcc0d1753a471038a90c819421c5e1`

Configure and ordinary CI must not download llama.cpp. Supply a locally
built and installed prefix through `INFEREDGE_LLAMA_CPP_ROOT`.

## Manual llama.cpp build

Work in an isolated temporary directory. Do not vendor sources into this
repository.

1. Download the exact archive URL above.
2. Verify the SHA-256 of the archive bytes before extraction.
3. Configure a CPU-only Release install for Apple arm64. Disable Metal,
   CoreML, CUDA, Vulkan, server, examples, and tests.
4. Install into a prefix you control.
5. Create `share/inferedge/llama-cpp-provenance.txt` in that prefix.

The provenance file is operator-attested. It is not a cryptographic
signature.

The ten lines must appear in that exact order as 252 raw UTF-8/ASCII
bytes, LF only, with exactly one terminal newline. CMake compares the
hexadecimal file bytes and does not normalize newlines. Reordered keys,
blank lines, extra terminal content, CR, CRLF, BOM, and NUL are rejected.
CMake and the C++ loader enforce the same bytes.

```text
schema_version=1
project=llama.cpp
tag=v0.7.0
commit=620d2225cc436ab0b001e3200e590e3b286ef9ab
source_archive_sha256=55d67acf3ad2ad747a9bbeb75f9f0e173dbcc0d1753a471038a90c819421c5e1
build_type=Release
architecture=arm64
backend=cpu
metal=off
server=off
```

Configure InferEdge:

```text
cmake -S . -B build-generative -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DINFEREDGE_ENABLE_GENERATIVE=ON \
  -DINFEREDGE_LLAMA_CPP_ROOT=/absolute/canonical/prefix
```

## External model-package structure

Place local packages only under `models/llm/local/`. That directory is
gitignored. A package is one directory containing exactly the allowlisted
files. No hidden files. No nested directories. No tokenizer source files.
The GGUF must carry its own tokenizer metadata.

Required files:

- `manifest.txt`
- the single manifest-declared `.gguf` file
- the single manifest-declared chat-template file
- the required license file

`NOTICE` is permitted only when `notice_file` names it. Use
`notice_file=none` when the package has no NOTICE file.

`inferedge-generate` never downloads a model and has no default package
path. The CLI requires `--model-package`, `--max-new-tokens`, `--threads`,
and exactly one of `--prompt` or `--prompt-file`. Stdin is not read.
Empty and ASCII-whitespace-only prompts (space, tab, LF) are rejected.
Unicode whitespace is not treated as empty. `--prompt` may be visible
to local process-listing tools and shell history; use `--prompt-file` for
sensitive prompts. Prompt files must be canonical absolute regular files
without symlink leaves or ancestors, without extra hard links, and without
special-file types. InferEdge reads the prompt into memory for the
synchronous request and does not persist it or send it over the network.
Prompt files remain the operator's responsibility after the process exits.
Prompt and output logging is not implemented.

## Manifest format

See `manifest.schema.txt` and `example-manifest.txt`. The format is a
fixed UTF-8 `key=value` document. There is no JSON parser. Extra keys,
missing keys, duplicates, whitespace, CR, NUL, path separators, and
unsafe file names are rejected.

Fixed runtime fields:

```text
runtime_project=llama.cpp
runtime_tag=v0.7.0
runtime_commit=620d2225cc436ab0b001e3200e590e3b286ef9ab
```

Phase 9-B1 added `license_file`, `license_sha256`, `notice_file`, and
`notice_sha256` so every allowlisted file is integrity-checked.
`notice_file=none` requires `notice_sha256=none` and means the NOTICE
file is absent. A file named `none` is not a NOTICE file in that case.

The GGUF SHA-256 pins embedded tokenizer metadata. Phase 9-B2 must still
compare InferEdge tokens with the pinned official tokenizer oracle.

The reserved Qwen non-thinking template identity is
`qwen3-non-thinking-v1.txt`. The template is validated completely before
`{user}` substitution. User text is inert data and is not scanned for
template syntax after substitution. Empty user prompts are rejected.
Ordinary words such as include or important are allowed in the frozen
template when they are not executable constructs.

Generation enforces `prompt_token_count <= effective_context` and
`max_new_tokens <= effective_context - prompt_token_count`. The effective
limit is the minimum of the manifest, the session context, and llama
context capacity. Zero generated tokens are rejected. `max_new_tokens` is
not silently reduced. Generated text is capped at
`kGenerativeMaxGeneratedBytes` and is not silently truncated.

`GenerativeSession` supports sequential `generate()` calls only. Concurrent
calls on one session are unsupported. A caller-owned `CancellationFlag`
must outlive `generate()`; the pointer may be null. Cancellation may be
requested from another thread through atomic state. Session destruction
must not overlap `generate()`.

Phase 9-B1 does not claim that template is
correct against the official tokenizer. Real-model generation remains
unproven. No performance claims are authorized.

Validation and load use separate file opens. That cannot fully defeat a
malicious concurrent replacement without descriptor-based loading.
