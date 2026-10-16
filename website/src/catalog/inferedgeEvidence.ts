export const GITHUB_REPO = "https://github.com/KasaVarun/inferedge";
export const GITHUB_DOCS = `${GITHUB_REPO}/blob/main/docs/benchmarks`;
export const CPP_CI_RUN_URL = "https://github.com/KasaVarun/inferedge/actions/runs/38840295859";
export const WEBSITE_CI_RUN_URL = "https://github.com/KasaVarun/inferedge/actions/runs/38840349770";
export const PRIMARY_MARKDOWN = "arrival-aware-sparse-gemm.md";
export const PRIMARY_JSON = "arrival-aware-sparse-gemm-results.json";

export const GEMM_ONNX_SHA256 =
  "a6203020b870eb04f9fd8e836cf4ccd930a57156a026d2a50163121d83debf32";
export const RESNET18_ONNX_SHA256 =
  "cfc843f2939dc6065d95d911ec778f57ca5f0f25faa4988a5b5c61011bebd935";

export const GEMM_MODEL = {
  id: "gemm_f32",
  title: "Neural GEMM",
  purpose: "Neural GEMM inference workload on ONNX Runtime CPU.",
  inputName: "features",
  inputShape: "[N, 512]",
  outputName: "product",
  outputShape: "[N, 512]",
  provider: "ONNX Runtime 1.31.0 CPU",
  onnxSha256: GEMM_ONNX_SHA256,
  correctness: "Independent C++ GEMM reference with atol = rtol = 1e-4.",
  publishedStudy: "G2 versus G4 producer-observed P50 at 100 RPS. Measured figures live on the Performance page.",
} as const;

export const RESNET18_MODEL = {
  id: "resnet18_f32",
  title: "Synthetic ResNet-18",
  architecture: "Synthetic ResNet-v1 BasicBlock [2, 2, 2, 2].",
  inputName: "images",
  inputShape: "[N, 3, 224, 224]",
  outputName: "logits",
  outputShape: "[N, 1000]",
  provider: "ONNX Runtime 1.31.0 CPU",
  onnxSha256: RESNET18_ONNX_SHA256,
  fixtureIdMin: 1000,
  fixtureIdMax: 1007,
  atol: "1e-5",
  rtol: "1e-5",
  correctness:
    "Committed NumPy-generated known-answer fixtures 1000–1007. Scheduler A/G2/G3/G4 smokes are correctness-only.",
} as const;

export const ADDITIONAL_MODELS = [
  {
    id: "add_f32",
    tensors: "A, B → C; rank-1 float32",
    check: "Exact float equality",
    publishedLatency: false,
  },
  {
    id: "batch_add_f32",
    tensors: "A, B → C; symbolic batch axis",
    check: "Exact float equality",
    publishedLatency: false,
  },
  {
    id: "mlp_small_f32",
    tensors: "features [N, 32] → logits [N, 8]",
    check: "Independent MLP reference, atol = rtol = 2e-4",
    publishedLatency: false,
  },
  {
    id: "mlp_medium_f32",
    tensors: "features [N, 256] → logits [N, 16]",
    check: "Independent MLP reference, atol = rtol = 2e-4",
    publishedLatency: false,
  },
  {
    id: "mnist_mlp",
    tensors: "image [784] → logits [10]",
    check: "Fixture-backed MNIST MLP inference",
    publishedLatency: false,
  },
  {
    id: "int64_echo",
    tensors: "int64 identity",
    check: "Exact int64 echo",
    publishedLatency: false,
  },
  {
    id: "rank2_f32",
    tensors: "rank-2 float32",
    check: "Fixture-backed shape/dtype check",
    publishedLatency: false,
  },
] as const;

export const GEMM_STUDY = {
  contrastId: "gemm_f32_G2_vs_G4_sparse_rps100",
  modelId: "gemm_f32",
  offeredRps: 100,
  durationMs: 5000,
  warmupRequests: 100,
  workers: 1,
  uncountedPilots: 2,
  countedBlocks: 20,
  bootstrapMethod: "paired_block_bootstrap_mean",
  replicates: 10000,
  seed: 20261016,
  interval: "95%",
  endpoint: "producer-observed P50",
  host: "one Darwin arm64 host",
  runtime: "ONNX Runtime 1.31.0 CPU",
  coarseLagMisses: 3,
  observationQuantumMs: 1,
} as const;

export const CPP_CI = {
  ctest: "800/800",
  benchmarkPython: 537,
  dynamicOnnx: 30,
  url: CPP_CI_RUN_URL,
  sha: "73c7c24e6c5f170b2d1c62b22457fb21733ff986",
} as const;

export const WEBSITE_CI = {
  frontendTests: 52,
  url: WEBSITE_CI_RUN_URL,
  sha: "73c7c24e6c5f170b2d1c62b22457fb21733ff986",
} as const;

export function githubDocUrl(name: string): string {
  return `${GITHUB_DOCS}/${name}`;
}
