# MNIST data, training, and evaluation

This directory prepares official MNIST, trains a small MLP, scores one promoted checkpoint on the official test set, and exports that checkpoint to ONNX. The C++ build does not install these packages.

## Environment

Python **3.12.14**. The virtual environment is `.venv-mnist` at the repository root. It is separate from `.venv`, which only regenerates the hand-written ONNX fixtures.

```bash
python3.12 -m venv .venv-mnist
.venv-mnist/bin/pip install -r models/mnist/requirements.txt
```

Pins verified together on macOS arm64:

| Package | Version |
| --- | --- |
| numpy | 2.5.4 |
| torch | 2.14.1 |
| torchvision | 0.29.1 |
| onnx | 1.22.0 |
| onnxruntime | 1.31.0 |

`onnxruntime` matches the C++ runtime. These wheels were checked on this Mac. Another operating system still needs a wheel for the same versions.

## Dataset

`torchvision.datasets.MNIST` downloads the official files. The class documents MNIST at `http://yann.lecun.com/exdb/mnist/` and tries these mirrors, in order:

1. `https://ossci-datasets.s3.amazonaws.com/mnist/`
2. `http://yann.lecun.com/exdb/mnist/`

Each gzip is checked against the MD5 stored on `MNIST.resources` in torchvision 0.29.1. The cached files matched:

| File | MD5 |
| --- | --- |
| `train-images-idx3-ubyte.gz` | `f68b3c2dcbeaaa9fbdd348bbdeb94873` |
| `train-labels-idx1-ubyte.gz` | `d53e105ee54ea40749a09fcbcd1e9432` |
| `t10k-images-idx3-ubyte.gz` | `9fb629c4189551a2d022fa330f9573f3` |
| `t10k-labels-idx1-ubyte.gz` | `ec29112dd5afa0611ce80d1b7f02629c` |

The cache is `models/mnist/.cache/MNIST/raw/`. That directory is gitignored. Do not commit the raw images or labels.

## Arrays

Images stay in IDX order: grayscale `uint8` shaped `[28, 28]`. `preprocess_image` divides by 255 and flattens row-major to `float32[784]` in `[0, 1]`. There is no mean/std normalization and no augmentation. Labels are `int64` class indices from 0 through 9.

A scan of the prepared arrays measured pixel minimum `0.0` and maximum `1.0` on the training, validation, and test parts. The first official training image is label 5.

## Split

The official training set has 60,000 images. The official test set has 10,000 and is not used to choose the split.

`numpy.random.Generator(np.random.PCG64(0)).permutation(60000)` orders the training indices. The first 54,000 are the training indices. The last 6,000 are the validation indices. With NumPy 2.5.4 those prefixes start at `[4013, 23840, 29603, 43011, 58703]` and `[57177, 43545, 14004, 32443, 35364]`.

The index sets are disjoint and cover `0 .. 59999` once. Repeating the call returns the same arrays. A different NumPy build can change `PCG64` output, so the pin matters.

## Tests

From `models/mnist`:

```bash
../../.venv-mnist/bin/python -m unittest test_data.py -v
```

The tests check synthetic pixels, the real files, the split, and the checksums. `test_model.py` checks the MLP shape and a state-dict round trip. `test_evaluation.py` checks checkpoint loading and confusion-matrix arithmetic on synthetic labels. It does not score the 10,000-image test set. They are not part of CTest.

```bash
../../.venv-mnist/bin/python -m unittest discover -s . -p 'test_*.py' -v
```

## Training

`MnistMlp` is `Linear(784, 128)`, ReLU, `Linear(128, 64)`, ReLU, `Linear(64, 10)`. It returns 10 raw logits and has 109,386 parameters. There is no dropout, batch normalization, or softmax. The predicted class is the index of the largest logit.

```bash
.venv-mnist/bin/python models/mnist/train_mnist_mlp.py \
  --checkpoint models/experiments/new_run.pt \
  --history models/experiments/new_run_history.json
```

The checkpoint path and the history path are required. There is no default destination. If either file already exists, the script stops before training unless `--overwrite` is also passed.

The selected model did not come from a fresh five-epoch file. A separate CPU run used the same architecture, seed 0, Adam learning rate 0.001, `CrossEntropyLoss`, batch size 64, and `DataLoader` workers 0, and stopped after ten epochs. Training started from the seeded initialization. It did not resume the five-epoch weights. The saved epoch is the one with the highest validation accuracy. A tie keeps the earlier epoch.

On the 6,000-image validation split, epoch 10 reached accuracy 0.9773333333333334 (97.73%) and loss 0.0824415915440768. The loop took 4.195 seconds. Those `state_dict` bytes were copied to `models/mnist_mlp.pt` without another training run. SHA-256 `b53a41676adfa020e3808617539f7966776fdd20bd7a247f34656e230459f22a`. `models/mnist/training_history.json` is that ten-epoch record.

The preserved five-epoch checkpoint is `models/experiments/mnist_mlp_5epochs.pt` (SHA-256 `1717f76de3d223ccf6e219c8d9a476af41cd7f8cf3097f701ee5c5e1d1a0cd79`). Its history is `models/experiments/mnist_mlp_5epochs_history.json`. That run's best validation accuracy was 0.971 at epoch 3. The ten-epoch experiment files remain `models/experiments/mnist_mlp_10epochs.pt` and `models/experiments/mnist_mlp_10epochs_history.json`.

## Evaluation

```bash
.venv-mnist/bin/python models/mnist/evaluate_mnist_mlp.py
```

The script loads `models/mnist_mlp.pt` with `torch.load(..., weights_only=True)`, runs it on CPU in eval mode, and scores the official 10,000-image test set once. Pixels stay `float32[784]` after dividing by 255. The report is `models/mnist/evaluation_report.json`.

That run counted 9,790 correct predictions and 210 incorrect predictions. Accuracy is 0.979 (97.90%). Sample-weighted cross-entropy loss is 0.08261763916015626. The test set was not used to choose the epoch, the architecture, or the preprocessing. These measurements were not used to train again.

Rows of the confusion matrix are true labels. Columns are predicted labels. Digit 9 has the lowest per-class accuracy in this report, 971 of 1,009. The example list is deterministic: the first official test index of each digit 0 through 9, then the first three incorrect predictions that were not already listed. Those ten first examples were correct. The incorrect examples are test indices 151 (true 9, predicted 8), 217 (true 6, predicted 5), and 241 (true 9, predicted 8).

## ONNX export

```bash
.venv-mnist/bin/python models/mnist/export_mnist_mlp.py
```

The script loads `models/mnist_mlp.pt` with `torch.load(..., weights_only=True)` and exports `models/mnist_mlp.onnx`. The call is `torch.onnx.export(..., dynamo=False, opset_version=13, external_data=False)` with input name `image` and output name `logits`. `dynamo=True` is the PyTorch 2.14.1 default and needs `onnxscript`, which is not installed. The TorchScript exporter is the installed path that accepts opset 13.

The traced input is `float32[784]`, with value `k / 784` at index `k`. The exported input is `image`, float32, shape `[784]`. The exported output is `logits`, float32, shape `[10]`. Those are raw logits. The graph applies no softmax. Node order is `MatMul`, `Add`, `Relu`, `MatMul`, `Add`, `Relu`, `MatMul`, `Add`. The requested opset is 13 and the file imports opset 13. The exporter first wrote IR 7. The saved file is IR 10 after `onnx.checker.check_model` accepted that version. The file is 438,304 bytes. SHA-256 `3473a4b2673fe4353c2b3a04a7d432717a4dfcdc42fc217565c99743542d8cce`.

Python ONNX Runtime 1.31.0 loaded it with `CPUExecutionProvider` and ran the same synthetic input. PyTorch and that session were later compared on 14 inputs. The C++ engine now loads this same file.

## Native C++ inference

`InferenceEngine` loads `models/mnist_mlp.onnx` with the existing CPU session. The public metadata is one input, `image`, `float32`, shape `[784]`, and one output, `logits`, `float32`, shape `[10]`. The tests call `run()` and do not use `Ort::Session` directly. No softmax is applied. The predicted class is the index of the largest logit.

The reference files are the 14 little-endian float32 arrays under `models/fixtures/mnist/`. `models/fixtures/mnist/manifest.json` is the Python checksum record. The C++ test names those fixture ids and their expected classes; it reads the ONNX reference logits from disk and does not embed them. For each logit the test requires:

```text
abs(actual - reference) <= 1e-5 + 1e-4 * abs(reference)
```

On this machine the 14 C++ outputs agreed with all 14 expected classes. The maximum absolute difference from the ONNX reference was 7.62939453125e-06. The mean absolute difference was 1.5224729265485492e-06.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

CTest sets `ORT_DISABLE_TELEMETRY=1`. The MNIST cases are part of `inferedge_onnx_smoke`. Configuration checks the ONNX SHA-256 and does not run Python.

The current CLI already accepts one fixed rank-1 float32 input. `image` with 784 values matches that rule, and 784 is within the existing value limit. The CLI prints the ten logits. It does not format a digit.

## CLI demonstration

Build the native executable, then run one fixture through it:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
.venv-mnist/bin/python models/mnist/demo_inferedge_cli.py test_00003
```

The default fixture is `test_00003`. Pass another manifest name, such as `test_00000`, `test_00151`, or `synthetic`. The script reads that case's little-endian `float32[784]` file, checks its SHA-256, and launches `build/inferedge` with one `--input image=...` argument. `subprocess` uses an argument list. Python does not import PyTorch or ONNX Runtime and does not calculate the network output.

`inferedge` prints ten raw logits. The largest value's index is the predicted digit. There is no softmax, so the numbers are not probabilities. `test_00003` is a true 0 and the manifest expects class 0. `test_00000` is a true 7. `test_00151` is a true 9 that this model classifies as 8. `synthetic` has no digit label and the manifest expects class 5. The script checks that argmax and compares the printed logits with the saved ONNX reference at `rtol=1e-4` and `atol=1e-5`.

The adapter exists because the CLI takes comma-separated text rather than a binary fixture. It is not a second inference engine.

## Limitations

The checkpoint is a PyTorch `state_dict`, and the ONNX file is a separate export of those weights. The ONNX input is rank-1 `[784]`, so a batched `[1, 784]` tensor does not match this graph. The native comparison covers the 14 fixtures, not the full 10,000-image test set, and it is a tolerance check rather than a bitwise identity check. The CLI demonstration prints those logits through `std::to_chars`, so the text can differ from the binary reference by less than the same tolerance. The C++ session disables graph optimizations. The TorchScript exporter is deprecated in PyTorch 2.14.1 and prints that warning. The same training seed can still produce different weights on another CPU or PyTorch build. The preserved five-epoch history names `models/mnist_mlp.pt` because that was the path when those weights were saved; the bytes now live in `models/experiments/mnist_mlp_5epochs.pt`.
