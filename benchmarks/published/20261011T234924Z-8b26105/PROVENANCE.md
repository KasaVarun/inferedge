# Provenance for the sanitized baseline

This directory is a sanitized derivative of the local run set `20261011T234924Z-8b26105`. The local directory `benchmarks/baselines/20261011T234924Z-8b26105/` remains the authoritative experiment record. These files were copied from that record after collection and analysis had already finished. No trial was rerun, and no measurement was recomputed.

## Experiment

| Fact | Value |
| --- | --- |
| Run-set id | `20261011T234924Z-8b26105` |
| Git commit that produced the measurements | `8b26105f5971035a38af6227bc34ca11589327f7` |
| Hardware model | Mac17,8 |
| macOS | 27.0.1 (Darwin, arm64) |
| Compiler | Apple clang version 21.0.0 (clang-2100.3.34.2), target arm64-apple-darwin27.0.0 |
| Build | Release, `CMAKE_CXX_FLAGS_RELEASE` `-O3 -DNDEBUG`, sanitizers absent |
| ONNX Runtime | 1.31.0 CPU |
| Add model SHA-256 | `37eb1b1b1adee439d2d14b91a0b217bd15ab809610d1271ea821d56e6400a447` |
| MNIST model SHA-256 | `3473a4b2673fe4353c2b3a04a7d432717a4dfcdc42fc217565c99743542d8cce` |
| `inferedge-bench` SHA-256 | `871c094371fc8678fe11534f80bb5e5a21adbe51ab4dfd59570b01b2a68c9e39` |
| Original `aggregate.json` SHA-256 | `dd0b2fb30189b6bc3e2d0a86cd14fc65bd1e85b21c321e772d9a6c6c12f3e5b9` |
| Sanitized `aggregate.json` SHA-256 | `8c6643ea5020a01a5f6738618b295080ff274d924186b4340db3bb738b6b9793` |

The two aggregate digests differ because the sanitized file replaces repository-absolute paths with repository-relative paths. Latency, throughput, percentiles, counts, correctness, hashes, the Git revision, trial ids, and timestamps were not edited. A published JSON file cannot keep the original digest when one of those path fields changes.

## What was sanitized

Each JSON document was parsed. A string was rewritten only when it belonged to one of the fields below and contained the absolute repository prefix. The rewrite removed that prefix, leaving a repository-relative path. The rest of each file, including number spelling, was left as copied. Stdout files are the benchmark JSON reports, so the same `model` edit was applied there. Stderr files were empty and were copied unchanged.

| File | Fields |
| --- | --- |
| `environment.json` | `queried.benchmark_binary_path`, `queried.onnxruntime_link_output`, `queried.onnxruntime_resolved_library`, `verified_configuration.add_model_path`, `verified_configuration.mnist_model_path`, `verified_configuration.mnist_fixture_directory` |
| `aggregate.json` | `models.add.path`, `models.mnist.path` |
| `runs/*.json` and `runs/*.stdout` | `model` |
| `runs/*.meta.json` | `command` elements that held the benchmark binary, the model path, or the MNIST fixture directory; `stdout_path`; `stderr_path` |

`stdout_path` and `stderr_path` now name `benchmarks/baselines/20261011T234924Z-8b26105/runs/<trial>.stdout` and the matching `.stderr` file. That is the collector's original location, expressed relative to the repository. The copies in this directory live under `runs/`.

Structural comparison of every original JSON document with its published copy found differences only in those fields. Every counted and pilot report still has `correctness` `passed`. Mean, minimum, maximum, P50, P95, P99, population standard deviation, active-call throughput, and observed sequential throughput match the originals exactly. The aggregate workload block matches the original numbers exactly. Pilots stay outside that block.

## How to read this copy

Inspect these JSON files directly. Do not pass this directory to `benchmarks/analyze_baseline.py`. That tool validates a collector run set and publishes `aggregate.json` only when the file is absent. This directory already contains the sanitized aggregate, and it was not produced by the collector. Analysis of this campaign was completed on the original local run set before any path was rewritten. The collector and analyzer were not changed to accept this derivative.

A later campaign must use the committed collector. It will record a new run-set id and its own absolute paths. Those paths stay in the local run set until a separate sanitization step is reviewed.

## SHA-256 of the sanitized data files

This list covers the copied data files. It does not cover `PROVENANCE.md`. Each `runs/<trial>.stdout` digest matches `runs/<trial>.json` because both are the same report after the same `model` edit. Every stderr file is empty, so each has the empty-file digest `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

| SHA-256 | File |
| --- | --- |
| `8c6643ea5020a01a5f6738618b295080ff274d924186b4340db3bb738b6b9793` | `aggregate.json` |
| `1992b99c57283bc88d88ce46f587983c6f92ce70c73b6ce5272bbf6587d0bf15` | `environment.json` |
| `5b9223e66d84a63d2aacd16146b7b114ff39e3107f78b7712853454342dcbadc` | `runs/add-01.json` |
| `5d8435ac00a115bfcc3da32e65705af235267ad36ee967e26db6c6068339bd7c` | `runs/add-01.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-01.stderr` |
| `5b9223e66d84a63d2aacd16146b7b114ff39e3107f78b7712853454342dcbadc` | `runs/add-01.stdout` |
| `3b2fcaa5bec8df4bbcfd958a3790b752a39225ec893e8fbe62da8e4e55e999a0` | `runs/add-02.json` |
| `8b456b680a207027ecc0a9f1919bd39f993e65aaff9d71459b25312be9d4dac8` | `runs/add-02.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-02.stderr` |
| `3b2fcaa5bec8df4bbcfd958a3790b752a39225ec893e8fbe62da8e4e55e999a0` | `runs/add-02.stdout` |
| `735825ad1c8cbe426bfc4f66324368b03e4f40859e7cf33c9d69db96abf7f72a` | `runs/add-03.json` |
| `b8ec6cd849ee87dde82dff93b06a5aa3f45fdd37dd08626ff956c88d5ef239be` | `runs/add-03.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-03.stderr` |
| `735825ad1c8cbe426bfc4f66324368b03e4f40859e7cf33c9d69db96abf7f72a` | `runs/add-03.stdout` |
| `092718b94776db1253e5f60a51d3b3318ce191e06d474d26be5e9bef00ffd3f1` | `runs/add-04.json` |
| `e107427fe083e53af24409a84759244a61aa98e553d96dba28ad24868544dd1b` | `runs/add-04.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-04.stderr` |
| `092718b94776db1253e5f60a51d3b3318ce191e06d474d26be5e9bef00ffd3f1` | `runs/add-04.stdout` |
| `2de2833d0587616d17655eb8b1441df6d452f150b2987f19cab1ecdf85b5e6c7` | `runs/add-05.json` |
| `b8d048aa507de4ab7f6f0c8602b75a917682b0f6fc4b803198e0839795b9d88d` | `runs/add-05.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-05.stderr` |
| `2de2833d0587616d17655eb8b1441df6d452f150b2987f19cab1ecdf85b5e6c7` | `runs/add-05.stdout` |
| `02654e403cb75e3df4225dc33ad80b50ead84022ffef6332989da58788ea2178` | `runs/mnist-01.json` |
| `09bc9249a510274d4727cd894278af17749b8d24a0300059a683ad96dc4b17db` | `runs/mnist-01.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-01.stderr` |
| `02654e403cb75e3df4225dc33ad80b50ead84022ffef6332989da58788ea2178` | `runs/mnist-01.stdout` |
| `684e708108b026dfcff34b5f1e5b68a835d47149ac19716273808a85b5ff59da` | `runs/mnist-02.json` |
| `639c3310bacedcb7e60438faf9d905d7a94e0435516256ee471591191c0a80c6` | `runs/mnist-02.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-02.stderr` |
| `684e708108b026dfcff34b5f1e5b68a835d47149ac19716273808a85b5ff59da` | `runs/mnist-02.stdout` |
| `0c6dfad04cde01e6764ef8943d318fd619f913df6c1e1a706ae9c39f18cead16` | `runs/mnist-03.json` |
| `c30b88c908020e3c04c6b0c2023daff23a454b95ef20e41b9968252e416f6fb6` | `runs/mnist-03.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-03.stderr` |
| `0c6dfad04cde01e6764ef8943d318fd619f913df6c1e1a706ae9c39f18cead16` | `runs/mnist-03.stdout` |
| `3a05778d51f61468e3cca8e985957e66fdfc8c41c6e21f4a5275df3a1d9aef18` | `runs/mnist-04.json` |
| `eb2aa1cc0d83b5c559408390811ced4c739af5b077b41eaea4f2629de6e256b7` | `runs/mnist-04.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-04.stderr` |
| `3a05778d51f61468e3cca8e985957e66fdfc8c41c6e21f4a5275df3a1d9aef18` | `runs/mnist-04.stdout` |
| `e8246ce97afa06a158fe84cafba3b6fdc75601d0403aa0e70c3b71e7da868195` | `runs/mnist-05.json` |
| `7f6868158d1bc0954027b48060ec8673cec4c18ab95bc72031f468b2608af102` | `runs/mnist-05.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-05.stderr` |
| `e8246ce97afa06a158fe84cafba3b6fdc75601d0403aa0e70c3b71e7da868195` | `runs/mnist-05.stdout` |
| `a7105bf416ae73a5e15d2dba429947c562c780cec896d1d57caade744fe27b5f` | `runs/pilot-add.json` |
| `5e522086373e7b7cf73e456bd42da529eba31de6b6abe05237d793a8bdbc641b` | `runs/pilot-add.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/pilot-add.stderr` |
| `a7105bf416ae73a5e15d2dba429947c562c780cec896d1d57caade744fe27b5f` | `runs/pilot-add.stdout` |
| `a8bc770548600942585af3cfde8f95b5d496dd0dce9b40fd05f2eb3050b45dc9` | `runs/pilot-mnist.json` |
| `ec93ff63d8c4b61c8d922484dcde90d85ce23464a853a088df533a6a62d51cb9` | `runs/pilot-mnist.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/pilot-mnist.stderr` |
| `a8bc770548600942585af3cfde8f95b5d496dd0dce9b40fd05f2eb3050b45dc9` | `runs/pilot-mnist.stdout` |
