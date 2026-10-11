# MNIST data pipeline

This directory prepares official MNIST for a later training run. It does not train a model. The C++ build does not install these packages.

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

The tests check synthetic pixels, the real files, the split, and the checksums. They are not part of CTest.
