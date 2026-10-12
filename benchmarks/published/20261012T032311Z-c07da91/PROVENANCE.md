# Provenance for the sanitized concurrent run set

This directory is a sanitized derivative of the private run set `20261012T032311Z-c07da91`. The private directory `benchmarks/concurrent_runs/20261012T032311Z-c07da91/` remains the authoritative experiment record. These files were copied from that record after collection and analysis had already finished. No trial was rerun, and no measurement was recomputed. This publication is a sanitized derivative, not a new experiment.

## Experiment

| Fact | Value |
| --- | --- |
| Run-set id | `20261012T032311Z-c07da91` |
| Measurement commit | `c07da912da109c09b6e3ea6f9ae8f1a89e97a79e` |
| Hardware model | Mac17,8 |
| macOS | 27.0.1 (Darwin 27.0.0, arm64) |
| Logical and physical CPU counts | 18 and 18 |
| Compiler | Apple clang version 21.0.0 (clang-2100.3.34.2), target arm64-apple-darwin27.0.0 |
| Compiler path | `/usr/bin/c++` |
| Compiler installed directory | `/Library/Developer/CommandLineTools/usr/bin` |
| Build | Release, `CMAKE_CXX_FLAGS_RELEASE` `-O3 -DNDEBUG`, sanitizers absent |
| ONNX Runtime | 1.31.0 CPU |
| Add model SHA-256 | `37eb1b1b1adee439d2d14b91a0b217bd15ab809610d1271ea821d56e6400a447` |
| MNIST model SHA-256 | `3473a4b2673fe4353c2b3a04a7d432717a4dfcdc42fc217565c99743542d8cce` |
| `inferedge-bench-concurrent` SHA-256 | `614564d20a12d298d0cf453cfe447e32b5645acaf9725bbe6dc3958e7cf37baf` |
| Original `aggregate.json` SHA-256 | `767a85f9e6d49b23e5ba8f2950ae4f7cb9111b39476d83731c77b7901dace7f8` |
| Original `environment.json` SHA-256 | `330a87092ddf31957dc71b14a20e590e16d18aba746fc67583e92289e942d195` |
| Published `aggregate.json` SHA-256 | `4ca76583e18a405e40091bfb2e693ab601f3f23013cfca652fb6e4ff66e0aafe` |
| Published `environment.json` SHA-256 | `eaa6441168ad88c3b1c06ca9820a832ec051b70f10264e9c240168b273ecaa0c` |

The published aggregate and environment digests differ from the original digests because the sanitized files replace repository-absolute paths with repository-relative paths. No numeric measurement was changed. Latency, aggregate throughput, active-call throughput, percentiles, worker counts, iteration counts, per-worker request counts, scaling ratios, scaling efficiencies, correctness, model hashes, binary hash, hardware facts, timestamps, trial identifiers, trial order, and pilot classification were not edited. A published JSON file cannot keep the original digest when one of those path fields changes.

The original private run set remains unmodified.

## What was sanitized

Each JSON document was parsed. A string was rewritten only when it belonged to one of the fields below and contained the absolute repository prefix. The rewrite removed that prefix, leaving a repository-relative path. The rest of each file, including number spelling, was left as copied. Stdout files are the benchmark JSON reports, so the same `model` edit was applied there. Stderr files were empty and were copied unchanged.

| File | Fields |
| --- | --- |
| `environment.json` | `queried.benchmark_binary_path`, `queried.onnxruntime_link_output`, `queried.onnxruntime_resolved_library`, `verified_configuration.add_model_path`, `verified_configuration.mnist_model_path`, `verified_configuration.mnist_fixture_directory` |
| `aggregate.json` | `models.add.path`, `models.mnist.path` |
| `runs/*.json` and `runs/*.stdout` | `model` |
| `runs/*.meta.json` | `command` elements that held the benchmark binary, the model path, or the MNIST fixture directory; `stdout_path`; `stderr_path` |

Approved results of that rewrite:

| Original role | Published repository-relative path |
| --- | --- |
| Benchmark binary | `build-concurrent/inferedge-bench-concurrent` |
| Resolved ONNX Runtime library | `build-concurrent/_deps/onnxruntime/onnxruntime-osx-arm64-1.31.0/lib/libonnxruntime.1.31.0.dylib` |
| Add model | `models/add_f32.onnx` |
| MNIST model | `models/mnist_mlp.onnx` |
| MNIST fixture directory | `models/fixtures/mnist` |
| Trial stdout recorded by the collector | `benchmarks/concurrent_runs/20261012T032311Z-c07da91/runs/<trial>.stdout` |
| Trial stderr recorded by the collector | `benchmarks/concurrent_runs/20261012T032311Z-c07da91/runs/<trial>.stderr` |

`stdout_path` and `stderr_path` name the collector's original location, expressed relative to the repository. The copies in this directory live under `runs/`.

`queried.onnxruntime_link_output` keeps the generic system libraries `/usr/lib/libc++.1.dylib` and `/usr/lib/libSystem.B.dylib`. `queried.compiler_path` remains `/usr/bin/c++`. The compiler version text still records `InstalledDir: /Library/Developer/CommandLineTools/usr/bin`. Those strings were not rewritten.

Command flags, concurrency values, warmup, iteration counts, fixture id `test_00003`, and every numeric field were left unchanged. Only command elements that contained the repository prefix were rewritten.

Structural comparison of every original JSON document with its published copy found differences only in the approved path fields. Every counted and pilot report still has `correctness` `passed`, tool version `0.1.0`, and schema version 2.

## How to read this copy

Inspect these JSON files directly, or read [`../../CONCURRENT_RESULTS.md`](../../CONCURRENT_RESULTS.md). Do not pass this directory to `benchmarks/analyze_concurrent.py`. That tool validates a collector run set and publishes `aggregate.json` only when the file is absent. This directory already contains the sanitized aggregate, and it was not produced by the collector. Analysis of this campaign was completed on the original private run set before any path was rewritten.

A later campaign must use the committed collector. It will record a new run-set id and its own absolute paths. Those paths stay in the private run set until a separate sanitization step is reviewed.

## SHA-256 of the sanitized data files

This list covers the copied data files. It does not cover `PROVENANCE.md`. Each `runs/<trial>.stdout` digest matches `runs/<trial>.json` because both are the same report after the same `model` edit. Every stderr file is empty, so each has the empty-file digest `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

| SHA-256 | File |
| --- | --- |
| `4ca76583e18a405e40091bfb2e693ab601f3f23013cfca652fb6e4ff66e0aafe` | `aggregate.json` |
| `eaa6441168ad88c3b1c06ca9820a832ec051b70f10264e9c240168b273ecaa0c` | `environment.json` |
| `a4341886adbdc8156474ba2d6798ccc9d6f80ccb7a7c2e792a7e03aa2a4261e3` | `runs/add-c01-01.json` |
| `7ee5ef0ec0cb36090359a7d97fa0b7e031ce94327894e3accc171baffee18391` | `runs/add-c01-01.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-c01-01.stderr` |
| `a4341886adbdc8156474ba2d6798ccc9d6f80ccb7a7c2e792a7e03aa2a4261e3` | `runs/add-c01-01.stdout` |
| `e1c36b2eb24c6ff1567ba522cab5a0b4b4a693529d2d49999edb2de18a6dea79` | `runs/add-c01-02.json` |
| `754202768fef8772102e7820249a0b38996302257e20ddd83f01c80955b9b2d8` | `runs/add-c01-02.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-c01-02.stderr` |
| `e1c36b2eb24c6ff1567ba522cab5a0b4b4a693529d2d49999edb2de18a6dea79` | `runs/add-c01-02.stdout` |
| `e40bb5fa708b9c307b26a85951aeebf1623fcea4d66f099b7ef59743b427cd1b` | `runs/add-c01-03.json` |
| `fc336e85711fdbec1e9c34869b79bfd67d65fd56d8d489ba5c8695c5d28156b1` | `runs/add-c01-03.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-c01-03.stderr` |
| `e40bb5fa708b9c307b26a85951aeebf1623fcea4d66f099b7ef59743b427cd1b` | `runs/add-c01-03.stdout` |
| `f22339657fcb8145a12b3067c508f9bc8df63b6844a5f616e735d87b3ff717a9` | `runs/add-c01-04.json` |
| `dd9b610f88583bdf821a1c48032abee435e238d5745d1a8e8b1e6d130fdb9795` | `runs/add-c01-04.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-c01-04.stderr` |
| `f22339657fcb8145a12b3067c508f9bc8df63b6844a5f616e735d87b3ff717a9` | `runs/add-c01-04.stdout` |
| `a18ec412b7eef9c2038b671043151f3a2914771d815ae52325af0ac318902242` | `runs/add-c01-05.json` |
| `9afd4d68033ab8a9c1c9dc73a09988aa933ece64dbe185515ab7dd90e12ced60` | `runs/add-c01-05.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-c01-05.stderr` |
| `a18ec412b7eef9c2038b671043151f3a2914771d815ae52325af0ac318902242` | `runs/add-c01-05.stdout` |
| `e60bb197196ad6fcdcff6412384f35664dbe8f8c59debb145e639654dea80796` | `runs/add-c02-01.json` |
| `66cdd7debb341e446e8bf32dbc913a40c47d4c4a2d5338b70e1da508071c4df6` | `runs/add-c02-01.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-c02-01.stderr` |
| `e60bb197196ad6fcdcff6412384f35664dbe8f8c59debb145e639654dea80796` | `runs/add-c02-01.stdout` |
| `2929d1e72e39ca6fd377b80f63760167f56ee960c5b1be0be4c70b4451a5560e` | `runs/add-c02-02.json` |
| `06b7d30ba1289b312846637720db150661cb2a74070bb8ba662973c9c1822b49` | `runs/add-c02-02.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-c02-02.stderr` |
| `2929d1e72e39ca6fd377b80f63760167f56ee960c5b1be0be4c70b4451a5560e` | `runs/add-c02-02.stdout` |
| `fe2d377a4164d98e73ddf4dbbc8dc2918bbb1eb985be738be4cdf724ac6a6e12` | `runs/add-c02-03.json` |
| `f76b82045024748b86dceabb86b24763901959178b4ecafe40ba3cd2f8f53d12` | `runs/add-c02-03.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-c02-03.stderr` |
| `fe2d377a4164d98e73ddf4dbbc8dc2918bbb1eb985be738be4cdf724ac6a6e12` | `runs/add-c02-03.stdout` |
| `35dc482f31052413f678e5a9b6b16b5c3dd67cd14ecaf4146e981b285e5db690` | `runs/add-c02-04.json` |
| `e40d3b3e75e7b1ce681d13f6c2c336e418f30ec7e4d42203e0a34ef2d66adf14` | `runs/add-c02-04.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-c02-04.stderr` |
| `35dc482f31052413f678e5a9b6b16b5c3dd67cd14ecaf4146e981b285e5db690` | `runs/add-c02-04.stdout` |
| `caa7013fc199e6197ad7c6af649c3ca8a5a91bb4dfe5c89487e098096b97215d` | `runs/add-c02-05.json` |
| `e7bf9ebd6be9c79e4b1572fe5b096e15911e9c8acc06779f870575ad6e86a300` | `runs/add-c02-05.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-c02-05.stderr` |
| `caa7013fc199e6197ad7c6af649c3ca8a5a91bb4dfe5c89487e098096b97215d` | `runs/add-c02-05.stdout` |
| `84d6e8fbea063db23fe2dbca8995e93e5191ff95416494d48b53e2623b556262` | `runs/add-c04-01.json` |
| `f532d62a517a913dd0e2fee8fc004826b4d4799c346c7c6d0435b0335f9e6f73` | `runs/add-c04-01.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-c04-01.stderr` |
| `84d6e8fbea063db23fe2dbca8995e93e5191ff95416494d48b53e2623b556262` | `runs/add-c04-01.stdout` |
| `c4ddb84f55deff4f0b1ba0201d28bfb195b62378276a24675aba8d00d38d36a1` | `runs/add-c04-02.json` |
| `671c119bea5c822fd67006ce56d2e6977ac89e4bd21cf3cc4a1aaa38f4a4a9f5` | `runs/add-c04-02.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-c04-02.stderr` |
| `c4ddb84f55deff4f0b1ba0201d28bfb195b62378276a24675aba8d00d38d36a1` | `runs/add-c04-02.stdout` |
| `54ce50ce6876b809fd3273f7d99b51dd68a3da04ff2fd513d0790414040566b0` | `runs/add-c04-03.json` |
| `c074f572754a8bb7890cc9fad88fe76034aac459257cb6db8a732813be99e873` | `runs/add-c04-03.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-c04-03.stderr` |
| `54ce50ce6876b809fd3273f7d99b51dd68a3da04ff2fd513d0790414040566b0` | `runs/add-c04-03.stdout` |
| `0a875278d337f59776df21525053ce5e9340332cdd5c67f09fa8f8e004c2e6fa` | `runs/add-c04-04.json` |
| `65e337af789e823ceecd843b75f9552ac8208b3e54a847b744698f2a27858c8e` | `runs/add-c04-04.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-c04-04.stderr` |
| `0a875278d337f59776df21525053ce5e9340332cdd5c67f09fa8f8e004c2e6fa` | `runs/add-c04-04.stdout` |
| `31a36b71ce386e3fd962b4a0fd6d823cf6433d5e3412dc39aea5962619d5d45e` | `runs/add-c04-05.json` |
| `2953d7dfa4126b99236d62b9a61566906a174f7c1a6a7ef941016d026ece9e55` | `runs/add-c04-05.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-c04-05.stderr` |
| `31a36b71ce386e3fd962b4a0fd6d823cf6433d5e3412dc39aea5962619d5d45e` | `runs/add-c04-05.stdout` |
| `5817190ffebafdf6fb26f4d5659fdd98ec92e1385a28d246b2c6317b9a658b60` | `runs/add-c08-01.json` |
| `45f8371eafb93078e7ff20005f250fc64f47382224bab68dc2db9873b7ad3f73` | `runs/add-c08-01.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-c08-01.stderr` |
| `5817190ffebafdf6fb26f4d5659fdd98ec92e1385a28d246b2c6317b9a658b60` | `runs/add-c08-01.stdout` |
| `82bd4b5b2d6b40f2f40bb328e59705ac4d9026e27e79ddc2c30ac2eb6ed32592` | `runs/add-c08-02.json` |
| `c974f1eeb24885136f6694934755dcf57c61dfae3cd80d2381362176024096b0` | `runs/add-c08-02.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-c08-02.stderr` |
| `82bd4b5b2d6b40f2f40bb328e59705ac4d9026e27e79ddc2c30ac2eb6ed32592` | `runs/add-c08-02.stdout` |
| `cda6827c8bcac48e23e916e2ad8cc604e4c30733aae3af50f6d402afc0f8d7f5` | `runs/add-c08-03.json` |
| `a192a442cfff287941a181dc00f9c8398ee0adc264bd84c4417b51538c4ca912` | `runs/add-c08-03.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-c08-03.stderr` |
| `cda6827c8bcac48e23e916e2ad8cc604e4c30733aae3af50f6d402afc0f8d7f5` | `runs/add-c08-03.stdout` |
| `b956f4ca8a75a547446324e02a262ac22c0bdbdb58cb7b566eb87196fea6f439` | `runs/add-c08-04.json` |
| `656ff61b72c3ed296290523150fe753a4c40f49fdf66c72876f0b78dfe0f8a5a` | `runs/add-c08-04.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-c08-04.stderr` |
| `b956f4ca8a75a547446324e02a262ac22c0bdbdb58cb7b566eb87196fea6f439` | `runs/add-c08-04.stdout` |
| `690ccfa8215db3b91f4e85dba5f5540d5308f662bb840a310aae34819478334b` | `runs/add-c08-05.json` |
| `a09564d2d4f53e66a7a640245238c4d5862fd0870d127b9acd6d1f5fb28b4fb3` | `runs/add-c08-05.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/add-c08-05.stderr` |
| `690ccfa8215db3b91f4e85dba5f5540d5308f662bb840a310aae34819478334b` | `runs/add-c08-05.stdout` |
| `f1f9b05559b80974b2e462ea681cb4ee21a5103156a904ed8b0eb2457d0ad9f6` | `runs/mnist-c01-01.json` |
| `3ef01fca604006bdf9a5cac0024427ffcd655cbf7506de3ea5bb365ed1de038d` | `runs/mnist-c01-01.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-c01-01.stderr` |
| `f1f9b05559b80974b2e462ea681cb4ee21a5103156a904ed8b0eb2457d0ad9f6` | `runs/mnist-c01-01.stdout` |
| `9d798caed9f233ab173b5ca2c7d46e84f22c7096a844084abb92cd5d1eb289b6` | `runs/mnist-c01-02.json` |
| `08c091fb87adf627dcd717915be0dcbf3f276169fa5d1a128d3a4aed6288ee38` | `runs/mnist-c01-02.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-c01-02.stderr` |
| `9d798caed9f233ab173b5ca2c7d46e84f22c7096a844084abb92cd5d1eb289b6` | `runs/mnist-c01-02.stdout` |
| `ac03481890a670e22db396906595c7e3a693bf9ba30af1b9bd57ceb85376ac6a` | `runs/mnist-c01-03.json` |
| `0157638ad798b0785d26ae0f48ee3e631505a4f597ae3cee5ca2598c37344ba2` | `runs/mnist-c01-03.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-c01-03.stderr` |
| `ac03481890a670e22db396906595c7e3a693bf9ba30af1b9bd57ceb85376ac6a` | `runs/mnist-c01-03.stdout` |
| `821f04d48c4fc0dc14500d338c812fad883b34c0bdad61acb1b5c921aca13fd9` | `runs/mnist-c01-04.json` |
| `4bce0951d0fd86ff555616589a27dec6f5023785005fab62a7172d70d0f231dc` | `runs/mnist-c01-04.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-c01-04.stderr` |
| `821f04d48c4fc0dc14500d338c812fad883b34c0bdad61acb1b5c921aca13fd9` | `runs/mnist-c01-04.stdout` |
| `9fc0527c0d6d50fa6cb30885aece4c551143e20a2a31b45639a34b0e7ddf65b9` | `runs/mnist-c01-05.json` |
| `e9be1955811d43a19c6f14673dd3d069cbfe1cde8adfe409601afdf2a22d8ee1` | `runs/mnist-c01-05.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-c01-05.stderr` |
| `9fc0527c0d6d50fa6cb30885aece4c551143e20a2a31b45639a34b0e7ddf65b9` | `runs/mnist-c01-05.stdout` |
| `6088c052333d221f9139d071f59e23aa714d6b37da2d350020a870f95cf9f974` | `runs/mnist-c02-01.json` |
| `d48e0a71a62f1a505fadc66a6f423d80b1ab368f1a426fac3eb700dc91e0cc3b` | `runs/mnist-c02-01.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-c02-01.stderr` |
| `6088c052333d221f9139d071f59e23aa714d6b37da2d350020a870f95cf9f974` | `runs/mnist-c02-01.stdout` |
| `269e9befb77ef35f0b70be0a7214f5aaf2e9f5eaa07ccc8d17e4c5dc0f743d7a` | `runs/mnist-c02-02.json` |
| `85f2f8d2d4580d503984a2e92a183df8ec867ea556d15ea66c6dd962fe0d23c5` | `runs/mnist-c02-02.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-c02-02.stderr` |
| `269e9befb77ef35f0b70be0a7214f5aaf2e9f5eaa07ccc8d17e4c5dc0f743d7a` | `runs/mnist-c02-02.stdout` |
| `ac472b6572e347621828be2a964c903c7a27e79acaaa180a880ccf8f47bbef12` | `runs/mnist-c02-03.json` |
| `9392f22d5c49dcc981c6d64193742e628195f8c985250a6b65e9d827ca94aad2` | `runs/mnist-c02-03.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-c02-03.stderr` |
| `ac472b6572e347621828be2a964c903c7a27e79acaaa180a880ccf8f47bbef12` | `runs/mnist-c02-03.stdout` |
| `33a4d20228e94e8861a55c87b22e8e3c8dc360ee70846253dbec7d63e317a550` | `runs/mnist-c02-04.json` |
| `6d7a1513ac7446c3507ba18cd377319486184f75ba9762c94b1d4924d6c7c248` | `runs/mnist-c02-04.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-c02-04.stderr` |
| `33a4d20228e94e8861a55c87b22e8e3c8dc360ee70846253dbec7d63e317a550` | `runs/mnist-c02-04.stdout` |
| `cd49c292d4bb635e03d2f192ca539a3a592a478951956ab50f3595917cc38f1f` | `runs/mnist-c02-05.json` |
| `104fab3e441d9e600c511adad3d373366475390d5d54eff8c13bb64ea414fa1b` | `runs/mnist-c02-05.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-c02-05.stderr` |
| `cd49c292d4bb635e03d2f192ca539a3a592a478951956ab50f3595917cc38f1f` | `runs/mnist-c02-05.stdout` |
| `98484fcd56980102e5dfe16ebe44ab03cf71749afa4ca45ae9bbec46fb408689` | `runs/mnist-c04-01.json` |
| `cd85ce167042fcb6f9653452f7619fcf514f903575c3aadd52379eb2a26a4d69` | `runs/mnist-c04-01.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-c04-01.stderr` |
| `98484fcd56980102e5dfe16ebe44ab03cf71749afa4ca45ae9bbec46fb408689` | `runs/mnist-c04-01.stdout` |
| `e7bed3e3dbf42c830a3bd3f28e8e96727d71ecaf9c7e50c4ca05f516470195b0` | `runs/mnist-c04-02.json` |
| `cebb2c4bfe646532e15e24948611bc5835abca02f4b9df492ec7689abed7d3c5` | `runs/mnist-c04-02.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-c04-02.stderr` |
| `e7bed3e3dbf42c830a3bd3f28e8e96727d71ecaf9c7e50c4ca05f516470195b0` | `runs/mnist-c04-02.stdout` |
| `199668ed9c979c91477768bc5fb4f05850f6ea6a8def33c6406a0acd33462cdc` | `runs/mnist-c04-03.json` |
| `1ca9c4bc27755c4af8379da58d0a09cd75110db3aa40be5c815beb9dbe0771b8` | `runs/mnist-c04-03.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-c04-03.stderr` |
| `199668ed9c979c91477768bc5fb4f05850f6ea6a8def33c6406a0acd33462cdc` | `runs/mnist-c04-03.stdout` |
| `512f1f9a9e8d799642ed98a07caf9ebf251771588c647ac618d59b1aeef4e48f` | `runs/mnist-c04-04.json` |
| `b971c3b877428713120b45be4b7247a38f0e492d674d5c8eb05e2207ea8fb928` | `runs/mnist-c04-04.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-c04-04.stderr` |
| `512f1f9a9e8d799642ed98a07caf9ebf251771588c647ac618d59b1aeef4e48f` | `runs/mnist-c04-04.stdout` |
| `fb59a1d229cfb97cf560bcb721b22b60072e3dd6a261ef3bdd138f78b88a3235` | `runs/mnist-c04-05.json` |
| `cff2d83386e7569ef8732576046e98d899ccae8e00f15035565f8f153295b9b4` | `runs/mnist-c04-05.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-c04-05.stderr` |
| `fb59a1d229cfb97cf560bcb721b22b60072e3dd6a261ef3bdd138f78b88a3235` | `runs/mnist-c04-05.stdout` |
| `90457eefd8be85dc653ced2889f496a329115dfa1b09758780d10179b23dd182` | `runs/mnist-c08-01.json` |
| `0734381f7cf7fc47c9e93d3f11b90dc3bac21469ef23033b199b5e82d516c7e5` | `runs/mnist-c08-01.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-c08-01.stderr` |
| `90457eefd8be85dc653ced2889f496a329115dfa1b09758780d10179b23dd182` | `runs/mnist-c08-01.stdout` |
| `a39d60542f704acd051ef781cdb0e5475c9a35a2d5e5cb331698b36df9db1525` | `runs/mnist-c08-02.json` |
| `7e2079626d50af9d3d1eff0b78b15b593327d7d4530d0be333604a9dc9370bbb` | `runs/mnist-c08-02.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-c08-02.stderr` |
| `a39d60542f704acd051ef781cdb0e5475c9a35a2d5e5cb331698b36df9db1525` | `runs/mnist-c08-02.stdout` |
| `fe237af77031b94ce40b474cd7f282c8f0577a695c51a9036e4a9cb9106f535f` | `runs/mnist-c08-03.json` |
| `646b8142f267301a85da69362ae9e6b1e2b3e47cb7719ed6795348a8bfefc2e7` | `runs/mnist-c08-03.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-c08-03.stderr` |
| `fe237af77031b94ce40b474cd7f282c8f0577a695c51a9036e4a9cb9106f535f` | `runs/mnist-c08-03.stdout` |
| `757f6da51087eea57f870a5965817bec046b6bf17bbd94ffaa234cf3eaf19d04` | `runs/mnist-c08-04.json` |
| `3b9ad54e4bff01344ead0597b8bc42cd59f1b14922f502ea4faa7eb3fd512177` | `runs/mnist-c08-04.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-c08-04.stderr` |
| `757f6da51087eea57f870a5965817bec046b6bf17bbd94ffaa234cf3eaf19d04` | `runs/mnist-c08-04.stdout` |
| `cb85d2c7c7712042a19dfea6a05aea7dd71422f9f56c7a1fb42d9acbecc88b2a` | `runs/mnist-c08-05.json` |
| `e2be76fef6b9d328411baf7a9b943df1edd1a75a567e1cefb6123754c18ce42b` | `runs/mnist-c08-05.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/mnist-c08-05.stderr` |
| `cb85d2c7c7712042a19dfea6a05aea7dd71422f9f56c7a1fb42d9acbecc88b2a` | `runs/mnist-c08-05.stdout` |
| `fe6229af9e1374972b8d9177e625ec86ea9eb1f188ff9195c95bef1936ccb47d` | `runs/pilot-add-c01.json` |
| `e92016c7816418208df2e70830f15e6623708fcee084c7c786f677574e914344` | `runs/pilot-add-c01.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/pilot-add-c01.stderr` |
| `fe6229af9e1374972b8d9177e625ec86ea9eb1f188ff9195c95bef1936ccb47d` | `runs/pilot-add-c01.stdout` |
| `c2c7f7604075b73d6ccc46f05a428159df930aa7918571c87ab1427d385b60aa` | `runs/pilot-add-c02.json` |
| `8b4c50fc343581a9a33eda5ee27380e721f3cc91c3b65126553e0bc156b8856b` | `runs/pilot-add-c02.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/pilot-add-c02.stderr` |
| `c2c7f7604075b73d6ccc46f05a428159df930aa7918571c87ab1427d385b60aa` | `runs/pilot-add-c02.stdout` |
| `3ad889244d657cc019cdaee53bd70f7643a0607d080527f6d2f278def4bc48fe` | `runs/pilot-add-c04.json` |
| `a9112cb7e01b0b342794df4538727ba5ddf9ea0395a9f27922f2d40d22b40838` | `runs/pilot-add-c04.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/pilot-add-c04.stderr` |
| `3ad889244d657cc019cdaee53bd70f7643a0607d080527f6d2f278def4bc48fe` | `runs/pilot-add-c04.stdout` |
| `e781202ab1718ba30e2b0a72ecb214d025ddd2ab77da2666aa587c552d135644` | `runs/pilot-add-c08.json` |
| `551e8df932cdd012f1f753d002b1ce91e50fe3c6e3cae3cfcf942046fd73ddda` | `runs/pilot-add-c08.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/pilot-add-c08.stderr` |
| `e781202ab1718ba30e2b0a72ecb214d025ddd2ab77da2666aa587c552d135644` | `runs/pilot-add-c08.stdout` |
| `7994af87401bf180623159390f230144978776de8ab5f7df91a106fe63b1e69f` | `runs/pilot-mnist-c01.json` |
| `ed61ef7e4693764c4464cd24923cddc97381c41daacaf7c9b8cc92091bcc0bdf` | `runs/pilot-mnist-c01.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/pilot-mnist-c01.stderr` |
| `7994af87401bf180623159390f230144978776de8ab5f7df91a106fe63b1e69f` | `runs/pilot-mnist-c01.stdout` |
| `c669dd195a51d22587ad5fe385e3c9d2619498c7e8ad1a4a84662d8674ce5eb8` | `runs/pilot-mnist-c02.json` |
| `e4d2e7dc0d27403501e0cf49fe123739d72b7b22b545379da45dcd80c88b5c72` | `runs/pilot-mnist-c02.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/pilot-mnist-c02.stderr` |
| `c669dd195a51d22587ad5fe385e3c9d2619498c7e8ad1a4a84662d8674ce5eb8` | `runs/pilot-mnist-c02.stdout` |
| `3728198b5bd14c61a3c38c174d06b6e7f1ad9e8a467635b96f25b6f23cd616cb` | `runs/pilot-mnist-c04.json` |
| `a2332d8abe7d17296bc44c41b878d77fb32cd224b197cd6790b4ffe46c810b39` | `runs/pilot-mnist-c04.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/pilot-mnist-c04.stderr` |
| `3728198b5bd14c61a3c38c174d06b6e7f1ad9e8a467635b96f25b6f23cd616cb` | `runs/pilot-mnist-c04.stdout` |
| `96ad2ef6786878df3091b43f17042752cf79f59f5a1b6513a7bb8fa15f573352` | `runs/pilot-mnist-c08.json` |
| `62a47375e8f93463e281f3efa992565846640818cc47096f9a6c2361a9d384ca` | `runs/pilot-mnist-c08.meta.json` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `runs/pilot-mnist-c08.stderr` |
| `96ad2ef6786878df3091b43f17042752cf79f59f5a1b6513a7bb8fa15f573352` | `runs/pilot-mnist-c08.stdout` |
