# Phase 9-B1 correctness protocol

This document freezes the correctness-only scope of the default-OFF
`inferedge-generate` foundation. It does not authorize benchmarks,
quality scores, or speed claims. Real-model correctness is still unproven.

## Scope

Phase 9-B1 implements package validation, a local llama.cpp provenance
gate, and a separate generative executable. It does not download, convert,
or run Qwen or Llama weights. Existing ONNX float32 inference, scheduling,
benchmarks, the public website, and deployment are unchanged.

ONNX Runtime GenAI 0.17.0 remains rejected.

## Runtime limits

- CPU only
- Apple arm64
- One loaded model
- Batch size one
- Context at most 4096
- Combined prompt-token and output-token context is enforced
- Maximum generated tokens bounded and validated; never silently reduced
- Zero generated tokens are rejected
- Greedy decoding only
- Request-local greedy sampler; sequential `generate()` only
- No sampling controls
- No speculative decoding
- No adapters
- No embeddings mode
- No multimodal input
- No server mode
- No network
- No prompt or output persistence
- Prompt and output logging is not implemented
- No telemetry
- No benchmark mode

## Empty-prompt contract

Empty user prompts and empty prompt files are invalid requests. ASCII
whitespace-only content is rejected. The ASCII whitespace set is space
(0x20), tab (0x09), and LF (0x0a). Other Unicode whitespace is not treated
as empty. Tokenizer behavior does not decide emptiness.

## Templates and user data

Validate the frozen template completely before substitution. The template
must be valid UTF-8, within its size bound, and contain exactly one `{user}`
placeholder. Other placeholders, `{{`, `}}`, `{%`, `%}`, `${`, `$`,
environment expansion, filesystem references, URL or network
references, and recursive-template constructs are rejected. Ordinary
English words such as include or important are not executable constructs.

After the template passes, replace the single `{user}` occurrence exactly
once. Do not scan the expanded prompt for template syntax. User text is
inert data. Ordinary characters such as `{`, `}`, `{user}`, `${HOME}`,
URL text, and `../` remain literal. User text is validated as UTF-8 against
its own input-size bound. Expanded size is checked with overflow-safe
arithmetic against the expanded-prompt byte limit.

## Context capacity

After tokenization and before prompt decode:

```text
prompt_token_count <= effective_context_limit
max_new_tokens <= effective_context_limit - prompt_token_count
```

The effective limit is the minimum of the manifest context limit, the
session or request configured context, and the actual llama context
capacity. Requests that cannot generate the requested maximum without
exceeding that limit fail with a typed context-limit error.

## UTF-8 and token pieces

Manifest text, template text, user prompts, prompt-file contents, and
successful generated output use one strict UTF-8 validator. Overlong
encodings, UTF-16 surrogates, code points above U+10FFFF, truncated
sequences, invalid continuations, and prohibited NULs are rejected.

Token pieces may split a UTF-8 code point. Generation accumulates a
bounded incomplete trailing sequence. Successful EOS or maximum-token
completion requires complete valid UTF-8. Cancellation preserves generated
token IDs and returns only the longest complete valid UTF-8 prefix.

Token pieces use the pinned llama.cpp required-size API with an explicit
maximum. Pieces are not truncated. Accumulated generated text is capped at
`kGenerativeMaxGeneratedBytes` (4096 tokens × 4096-byte pieces). The bound
is checked before each append. Exceeding it is a decode error, not silent
truncation. A real model is not claimed to reach this bound.

## Session, cancellation, and privacy

`GenerativeSession` supports sequential calls. Concurrent `generate()`,
`load()`, or destruction overlapping `generate()` is unsupported. The
caller-owned `CancellationFlag` must outlive `generate()`. The pointer may
be null. Cancellation may be requested from another thread through atomic
state. The abort callback is installed only while that flag is alive and
is cleared before return.

`--prompt` may be visible to local process-listing tools and shell history.
`--prompt-file` is recommended for sensitive prompts. Prompt files must be
canonical absolute regular files without symlink leaves or ancestors,
without extra hard links, and without special-file types. InferEdge does
not persist prompt or output content and does not send prompts over the
network. Prompt files remain the operator's responsibility after exit.

## Qwen non-thinking requirement

The first manual integration target is Qwen in non-thinking mode. The
reserved frozen template identity is `qwen3-non-thinking-v1.txt`. Phase
9-B1 does not claim that template is correct. Phase 9-B2 must compare
rendered prompts and token IDs with the pinned official tokenizer output
before any Qwen correctness claim.

Llama remains gated on a separate license-acceptance decision.

## Token and template oracle

The GGUF SHA-256 covers embedded tokenizer metadata. There is no separate
`tokenizer_sha256` field. Phase 9-B2 independently verifies tokenizer
behavior against the pinned official tokenizer oracle.

Phase 9-B2 must record:

- The exact package manifest and file hashes, including license and NOTICE
- The frozen template bytes and SHA-256
- Official tokenizer prompt tokens for each fixture prompt
- InferEdge prompt tokens after frozen-template expansion
- Greedy generated token IDs
- UTF-8 decoded text
- Finish reason

A mismatch against the official tokenizer oracle is a correctness failure.

## License and NOTICE hashes

Every accepted package file is integrity-checked. `license_file` must name
a safe regular file whose bytes match `license_sha256`. `notice_file=none`
requires `notice_sha256=none`. A declared NOTICE file requires a safe
filename and a matching lowercase 64-hex SHA-256. Those files are hashed
before llama.cpp model load.

## Provenance

The llama.cpp provenance file is exactly 252 UTF-8/ASCII bytes: ten
LF-terminated lines in the documented order, with one terminal LF. CMake
compares the raw hexadecimal file bytes. `file(READ)` text mode is not used.
Reordered keys, blank lines, extra terminal content, CR, CRLF, BOM, NUL,
and size mismatches are rejected. CMake and C++ accept only that byte
sequence.

## Repetition requirements

A Phase 9-B2 correctness run is not a single lucky decode. Repeat each
accepted fixture enough times to show deterministic greedy identity on
the same package, same prompt, same thread count, and same maximum-token
bound. Any divergence is a stop condition. Phase 9-B1 does not claim
real-model repeated determinism.

## Finish behavior

- EOS or EOG ends generation with finish reason EOS
- Reaching the validated maximum ends generation with finish reason
  maximum tokens
- A caller-owned cancellation flag may end generation with finish reason
  canceled
- Phase 9-B1 tests the cancellation adapter without a real model
- Bounded real-model cancellation latency is not claimed until measured
  in a later audited phase

## What this phase does not claim

- Cross-quant equality is not assumed. A Q4 result is not evidence for
  Q8 or F16
- No quality claim
- No speed claim
- No arrival-fidelity or scheduler claim
- No trained or learned policy
- No real-model generation

## Phase 9-B2 stop conditions

Stop Phase 9-B2 and do not proceed to benchmarks if any of the following
occur:

- Official tokenizer tokens disagree with InferEdge prompt tokens
- Greedy outputs are not repeatable on the same package
- UTF-8 validation fails
- Context or maximum-token bounds are exceeded
- Package path, hash, or allowlist checks are bypassed
- A model is downloaded automatically
- Metal, CoreML, CUDA, Vulkan, server, or telemetry support appears
- Any performance number is treated as publication evidence
