# InferEdge

InferEdge is a high-performance C++ inference runtime and scheduler. The repository has a C++20 build, a small command-line executable, unit tests, formatting, and Linux CI. Configure downloads the pinned ONNX Runtime 1.31.0 CPU library and links it into a smoke test.

The executable can introduce itself. It does not load models or run inference yet.

```text
./inferedge --help
./inferedge --version
```

## Project status

The command-line executable still only answers `--help` and `--version`.

| Present | Not present yet |
| --- | --- |
| CMake + Ninja build | Model loading or tensor inference |
| `inferedge` with `--help` and `--version` | Scheduler, threads, or dynamic batching |
| GoogleTest unit tests | HTTP API, database, frontend, Python bindings |
| ONNX Runtime 1.31.0 CPU, linked by the smoke test | CUDA, TensorRT, or CoreML provider setup |
| clang-format 23.1.3 and Linux GitHub Actions | A production inference API |

Version: **0.1.0**

## Architecture roadmap

Each phase should stay reviewable on its own. Later phases are intentionally not started.

1. **Phase 0 — Foundation.** Compiler settings, layout, CLI, tests, and CI.
2. **Phase 1 — Core types.** Ownership, status/error reporting, and the data types later phases will share.
3. **Phase 2 — Model loading.** Phase 2.0 links the pinned ONNX Runtime CPU library. Loading a model and running it come later.
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

The smoke-test binary records the extracted `lib` directory as its build RPATH. macOS loads `@rpath/libonnxruntime.1.dylib`. Linux loads `libonnxruntime.so.1`. The `inferedge` executable does not link ONNX Runtime.

Official macOS and Linux builds include telemetry. Tests and CI set `ORT_DISABLE_TELEMETRY=1` before the library is initialized. Export the same variable before running `inferedge_onnx_smoke` outside CTest. This does not rebuild ONNX Runtime with telemetry compiled out.

ONNX Runtime is MIT licensed, copyright Microsoft Corporation. Each archive includes `LICENSE` and `ThirdPartyNotices.txt`. Those notices stay with the extracted package in the build tree.

The smoke test constructs `Ort::Env` and checks that `Ort::GetVersionString()` is `1.31.0`. It does not load a model. The inspected macOS library requires macOS 14.0 or newer.

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

`ctest` runs the project tests, the ONNX Runtime smoke test, and two executable checks: `inferedge --help` and `inferedge --version`.

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
src/                 library and executable sources
tests/               GoogleTest sources
.github/workflows/   Linux CI
```

`inferedge_core` is a static library. The `inferedge` executable links that library and does not contain the parsing logic.

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
