# InferEdge

InferEdge is a high-performance C++ inference runtime and scheduler. The repository has a C++20 build, a command-line executable, unit tests, formatting, and Linux CI. Configure downloads the pinned ONNX Runtime 1.31.0 CPU library and links it into `inferedge` and the ONNX test executable.

`inferedge_onnx` loads one ONNX model into a reusable CPU session. The `inferedge` executable parses a command and calls that session. `inferedge_core` does not link ONNX Runtime.

```text
./inferedge --help
./inferedge --version
```

## Project status

The executable runs one fixed rank-1 float32 model and still answers `--help` and `--version`.

| Present | Not present yet |
| --- | --- |
| CMake + Ninja build | Scheduler, request queue, or dynamic batching |
| `inferedge --model` and `--input` | HTTP API, database, frontend, Python bindings |
| GoogleTest unit tests | CUDA, TensorRT, or CoreML provider setup |
| Synchronous float32 `InferenceEngine` | Concurrent `run()` calls |
| clang-format 23.1.3 and Linux GitHub Actions | A production inference API |

Version: **0.1.0**

## Architecture roadmap

Each phase should stay reviewable on its own. Later phases are intentionally not started.

1. **Phase 0 — Foundation.** Compiler settings, layout, CLI, tests, and CI.
2. **Phase 1 — Core types.** Ownership, status/error reporting, and the data types later phases will share.
3. **Phase 2 — Model loading.** Phase 2.0 links the pinned ONNX Runtime CPU library. Phase 2.1 runs one committed float32 Add graph. Phase 2.2 moves that execution into `InferenceEngine`. Phase 2.3 calls that engine from `inferedge`. Scheduling comes later.
4. **Phase 3 — Execution.** Run one inference and return outputs.
5. **Phase 4 — Scheduling.** Threads and dynamic batching, measured against a real executor.
6. **Phase 5 — Service boundary.** A process API only after the runtime has a stable in-process interface.

## Prerequisites

On this Mac:

| Tool | Used for |
| --- | --- |
| Apple Clang with C++20 | Compiling |
| CMake 3.24 or newer | Generating the build |
| Ninja | Running the build |
| clang-format 23.1.3 | Formatting C++ sources |
| Git | Version control |

Install the formatter if it is missing:

```bash
brew install clang-format
```

Linux CI installs `g++`, `cmake`, `ninja-build`, and the same clang-format version. The workflow runs on `ubuntu-24.04`.

The first configure step downloads GoogleTest 1.18.0 and the ONNX Runtime 1.31.0 CPU archive for the host platform. Both downloads require network access. Neither archive is committed.

## ONNX Runtime

Version **1.31.0**, CPU packages only:

| Host | Archive | SHA-256 |
| --- | --- | --- |
| macOS arm64 | `onnxruntime-osx-arm64-1.31.0.tgz` | `1031b36dd3bdaa8d976a6130d9cf2a05e9be320b93ae804f8427b91dc3432afc` |
| Linux x86-64 | `onnxruntime-linux-x64-1.31.0.tgz` | `cc5c72baf5ae5c8238a6841f0897227be2d02826b9cf98eaf02fdefaeeb03a57` |

CMake downloads the one archive that matches the host, checks its SHA-256, and extracts it under the build directory. A checksum mismatch or an unsupported operating system and processor stops configuration. The version does not float to a newer release. The CMake files shipped inside the archive point at directories the archive does not contain, so InferEdge defines the imported target `onnxruntime::onnxruntime` from `include/` and `lib/` itself.

Set `INFEREDGE_ONNXRUNTIME_ROOT` to an already extracted package when configure should not download. That directory must contain `include/onnxruntime_cxx_api.h`, the shared library, its runtime soname, and a `VERSION_NUMBER` file that reads `1.31.0`.

`inferedge` and the smoke-test binary record the extracted `lib` directory as their build RPATH. macOS loads `@rpath/libonnxruntime.1.dylib`. Linux loads `libonnxruntime.so.1`.

Official macOS and Linux builds include telemetry. Tests and CI set `ORT_DISABLE_TELEMETRY=1` before the library is initialized. Export the same variable before running `inferedge` or `inferedge_onnx_smoke` outside CTest. This does not rebuild ONNX Runtime with telemetry compiled out.

ONNX Runtime is MIT licensed, copyright Microsoft Corporation. Each archive includes `LICENSE` and `ThirdPartyNotices.txt`. Those notices stay with the extracted package in the build tree.

The smoke test constructs `Ort::Env` and checks that `Ort::GetVersionString()` is `1.31.0`. The same executable loads `models/add_f32.onnx` through `InferenceEngine` and runs it. The inspected macOS library requires macOS 14.0 or newer.

## Add fixture

`models/add_f32.onnx` is a hand-written graph, not a trained model. It has no weights.

| Item | Value |
| --- | --- |
| Graph | `add_f32` |
| Inputs | `A` and `B`, `float32[2]` |
| Output | `C`, `float32[2]` |
| Operation | `C = Add(A, B)` |
| ONNX IR | 10 |
| Opset | 13 |
| Producer | `inferedge` `0.1.0` |
| Generator | `onnx==1.22.0` |
| Size | 203 bytes |
| SHA-256 | `37eb1b1b1adee439d2d14b91a0b217bd15ab809610d1271ea821d56e6400a447` |

CMake checks that hash with `file(SHA256)` and stops if the file is missing or different. Configuration does not run Python.

The C++ test creates `Ort::Env`, then `Ort::SessionOptions`, then `Ort::Session`. Options select the CPU execution provider, one intra-op thread, and `ORT_DISABLE_ALL`. The test checks the input and output names, `float32` types, and shapes `[2]`. Input tensors borrow `std::vector<float>` storage. `Ort::Session::Run` produces the output, and the test reads that tensor before destroying it.

Two successful cases use exact `float32` values:

| A | B | C |
| --- | --- | --- |
| `[1.0, 2.0]` | `[3.0, 4.0]` | `[4.0, 6.0]` |
| `[-1.5, 0.0]` | `[2.5, -4.0]` | `[1.0, -4.0]` |

The same test expects `Ort::Exception` for a missing file, `tests/data/not_a_model.bin`, a shape of `[3]`, and an `int64` input. Those failures do not stop the process.

To regenerate the fixture on purpose:

```bash
python3 -m venv .venv
.venv/bin/pip install -r models/requirements.txt
.venv/bin/python models/generate_add_model.py
shasum -a 256 models/add_f32.onnx
```

Copy the printed digest into `INFEREDGE_ADD_MODEL_SHA256` in `CMakeLists.txt`. The generator refuses any `onnx` version other than `1.22.0`, passes IR 10 and opset 13 explicitly, and runs `onnx.checker.check_model` before saving. Metadata has no timestamp.

`models/int64_echo.onnx` is a second untrained fixture. It is an Identity of `int64[1]` from `X` to `Y`, IR 10, opset 13, 201 bytes, SHA-256 `4bae136df39291cfccfb7a1b9b0f03186598326f441f755095459ab03805b145`. The engine rejects it because only `float32` tensors are supported. Regenerate it with `models/generate_int64_echo_model.py` and update `INFEREDGE_INT64_MODEL_SHA256`.

`models/rank2_f32.onnx` is an Identity of `float32[1, 2]` from `M` to `N`, IR 10, opset 13, 212 bytes, SHA-256 `193fc307deda81dee0e3b0f5999809a80d92ec46070c66c187b4578daefb5bc2`. The CLI rejects it because the input is not a fixed rank-1 tensor. Regenerate it with `models/generate_rank2_f32_model.py` and update `INFEREDGE_RANK2_MODEL_SHA256`.

## Inference engine

`inferedge_onnx` is a static library. The `inferedge` executable links it. `inferedge_core` does not. Public declarations live in `include/inferedge/inference_engine.hpp` and do not include ONNX Runtime headers. `src/inference_engine.cpp` owns the runtime types behind `std::unique_ptr`.

`load_inference_engine` checks that the path is a file, then creates one `Ort::Env` and one `Ort::Session`. Session options stay alive through construction, select the CPU provider, set one intra-op thread, and disable graph optimizations. The environment is destroyed after the session. A missing file returns `kModelNotFound`. A file ONNX Runtime cannot load returns `kModelInvalid`. A model input or output that is not a `float32` tensor returns `kUnsupportedType`. On success the returned engine is non-null and `run` reuses that session.

`run` accepts any number of named `float32` tensors. Names are matched to the model, so input order does not matter. The first problem is returned as `InferenceError`: empty name, duplicate name, unexpected name, missing name, negative dimension, overflowing shape product, element-count mismatch, then rank and fixed-dimension mismatch. A negative model dimension is dynamic and accepts a non-negative caller size. A failed `run` leaves the session loaded.

`NamedTensorView::values` is a `std::span<const float>`. It must stay valid for the whole `run` call. ONNX Runtime 1.31 `CreateTensor` takes `float*`, not `const float*`, so `run` copies each input into mutable storage for that call and does not cast away const. The copy is destroyed after `Run` returns. Each output is copied into `NamedTensor::values` before the ONNX Runtime output value is destroyed. Returned tensors own those floats.

`InferenceEngine` is move-only. A moved-from engine has no session. `inputs()` and `outputs()` then return an empty vector, and `run` returns `kExecution`. Those accessors do not dereference an empty implementation.

The engine is not thread-safe. One thread should call a given engine. Per-call input spans and returned vectors are distinct objects, and they share one `Ort::Session`. Concurrent `run` calls have not been tested.

## Command line

```bash
export ORT_DISABLE_TELEMETRY=1
./build/inferedge --model models/add_f32.onnx --input A=1,2 --input B=3,4
```

```text
Model: models/add_f32.onnx
Backend: ONNX Runtime CPU

Output C:
[4, 6]
```

`--model <path>` and `--model=<path>` select the file. `--input <name>=<values>` and `--input=<name>=<values>` supply one named tensor, and the flag can be repeated. Values are comma-separated finite `float32` numbers in decimal or scientific form. A negative number stays inside that argument, as in `--input A=-1.5,0`. A separate argument that starts with `-` is an option.

`--help`, `-h`, `--version`, and `-V` print to stdout and exit 0. With no arguments, the program prints the same help text and exits 0. Those two requests cannot be combined with any other argument.

This command runs a model only when every input is a fixed rank-1 `float32` tensor. The shape is the dimension stored in the model. A different rank, or a dynamic dimension, is rejected before `run`. The engine checks the element count and reports a missing or unexpected name. Printed numbers are the floats `InferenceEngine::run` returned. `std::to_chars` writes them in general format, so a whole number has no trailing decimal.

| Exit code | Meaning |
| --- | --- |
| 0 | Help, version, or a successful run |
| 2 | Invalid command line |
| 3 | The model file is missing, invalid, or not a float32 tensor |
| 4 | An input does not match the model, or an input shape is not fixed rank-1 |
| 5 | Inference failed while the session was running |

Successful text goes to stdout. Errors go to stderr as `error: ...`. A usage error also reprints the help text. The `Model:` line repeats the path argument as it was typed.

One command accepts at most 64 named inputs and 1,048,576 floats in total, with that same cap on each input. A numeric token is at most 64 characters. The values text of one `--input` is at most 1,048,576 × 65 characters, which is one maximum-length token plus a comma for every allowed value. An input name is at most 1024 characters. `std::from_chars` must consume each token. Overflow and underflow reported by that parser are usage errors. `inf` and `nan` are usage errors.

`parse_cli` does not load a model. `execute_inference` loads it once and calls `run` once. The parsed `vector<float>` stays alive for that call.

## Current limitations

- The CLI runs fixed rank-1 float32 inputs only. It does not accept a dynamic dimension or a rank other than 1.
- There is no scheduler, request queue, thread pool, or batching API.
- The engine runs one synchronous `float32` model on CPU, with one intra-op thread and graph optimizations disabled.
- Each `run` copies input floats and output floats. This is not zero-copy execution.
- Concurrent `run()` calls are not supported.
- The committed graphs are hand-written. They are not trained models.
- CUDA, TensorRT, and CoreML providers are not configured.
- Python is used only to regenerate fixtures. The C++ build does not install it.

## Build

From the repository root:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
./build/inferedge --version
```

Use `Release` for an optimized build. Ninja is a single-configuration generator, so the build type is chosen at configure time. Reconfigure into a separate directory if you want both trees at once:

```bash
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-release
```

`Debug` keeps symbols and disables optimization. `Release` enables optimization and defines `NDEBUG`. CMake's defaults for those two configurations are used; this project does not replace them.

Project targets compile as C++20 with extensions off (`-std=c++20`, not `gnu++20` on Clang and GCC). Warnings are errors for InferEdge code: `-Wall -Wextra -Wpedantic -Werror`, plus `-Wshadow` and `-Wnon-virtual-dtor`. MSVC uses `/W4`.

## Test

```bash
ctest --test-dir build --output-on-failure
```

`ctest` runs the project tests, the ONNX Runtime tests (version check, direct Add execution, and `InferenceEngine`), process tests that spawn `inferedge`, and two direct checks: `inferedge --help` and `inferedge --version`.

To run the binaries directly:

```bash
export ORT_DISABLE_TELEMETRY=1
./build/inferedge_tests
./build/inferedge_onnx_smoke
```

## Formatting

```bash
find include src tests -type f \( -name '*.hpp' -o -name '*.cpp' \) -print0 \
  | xargs -0 clang-format --dry-run --Werror
```

Apply the style with `--dry-run` removed and `-i` added. CI uses clang-format **23.1.3**, matching `.github/workflows/ci.yml`.

## Layout

```text
cmake/               pinned ONNX Runtime download
include/inferedge/   public headers
models/              committed ONNX fixtures and their generators
src/                 library and executable sources
tests/               GoogleTest sources
.github/workflows/   Linux CI
```

`inferedge_core` is a static library. It contains `parse_cli` and does not link ONNX Runtime. The `inferedge` executable links `inferedge_core` and `inferedge_onnx`. `src/run_command.cpp` is compiled into the executable, and `main` only parses arguments, runs a model when asked, and prints the result.

## How a C++ build works

### Compilation

The compiler translates one `.cpp` file at a time into an object file. `src/cli.cpp` becomes an object that contains `parse_cli`. `src/main.cpp` becomes an object that contains `main`. A header is not compiled by itself. When a `.cpp` file includes `inferedge/cli.hpp`, the compiler pastes that declaration into the translation unit it is compiling.

### Headers and source files

A header says what other files are allowed to call: names, argument types, and return types. A source file says how the function behaves. `parse_cli` is declared in `include/inferedge/cli.hpp` and defined in `src/cli.cpp`. The tests include the header and link the library, so they exercise the same function `main` calls.

`version.hpp` is header-only because the version is a compile-time string. CMake passes `-DINFEREDGE_VERSION="0.1.0"` from `project(VERSION 0.1.0)`. One version number lives in `CMakeLists.txt`.

### Linking

The linker combines object files and libraries into an executable. It matches the call to `parse_cli` in `main.cpp` with the definition compiled from `cli.cpp`. If the definition were missing, compilation of `main.cpp` could still succeed and the link step would fail with an undefined reference. That split is why a header and a source file are different files.

### What CMake does

CMake does not compile the project. It reads `CMakeLists.txt` and writes a Ninja build file under `build/`. Ninja then runs Clang, the linker, and the test discovery commands. Changing a source file and running `cmake --build build` recompiles only what is out of date.

`FetchContent` downloads GoogleTest during configuration and adds its targets to this build. InferEdge does not vendor a copy of GoogleTest or ONNX Runtime in the repository.

### Why C++20

C++20 is the newest standard this project needs. It is supported by the installed Apple Clang and by current GCC on GitHub-hosted Ubuntu runners. The code uses `std::string_view` and `constexpr` functions that are part of that standard. Newer standards can wait until a language feature earns a place in the runtime.

## License

MIT. See [LICENSE](LICENSE).
