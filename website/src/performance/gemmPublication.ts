export const PRIMARY_JSON = "arrival-aware-sparse-gemm-results.json";
export const PRIMARY_MARKDOWN = "arrival-aware-sparse-gemm.md";
export const P50_FIGURE = "arrival-aware-sparse-gemm-p50.png";
export const DELTA_FIGURE = "arrival-aware-sparse-gemm-paired-deltas.png";
export const GITHUB_REPO = "https://github.com/KasaVarun/inferedge";
export const GITHUB_DOCS = `${GITHUB_REPO}/blob/main/docs/benchmarks`;

export const EXPECTED_CONTRAST = "gemm_f32_G2_vs_G4_sparse_rps100";
export const EXPECTED_G2_P50 = 3792.90625;
export const EXPECTED_G4_P50 = 1280.4875;
export const EXPECTED_PAIRED_DIFF = -2512.41875;
export const EXPECTED_CI_LOW = -2513.64165;
export const EXPECTED_CI_HIGH = -2511.3562987499995;
export const EXPECTED_BLOCKS = 20;
export const EXPECTED_RPS = 100;
export const EXPECTED_SEED = 20261016;
export const EXPECTED_REPLICATES = 10000;
export const EXPECTED_WORKERS = 1;
export const EXPECTED_LAG_MISSES = 3;

export const EXPECTED_RELATED = [
  {
    contrastId: "gemm_f32_A_vs_G2_delay2ms_rps1000",
    markdown: "neural-batching-gemm.md",
    resultsJson: "neural-batching-gemm-results.json",
    relationship: "prior_two_arm_configured_batching_study_at_1000_rps",
  },
  {
    contrastId: "gemm_f32_A_G2_G3_adaptive_rps1000",
    markdown: "adaptive-neural-batching-gemm.md",
    resultsJson: "adaptive-neural-batching-gemm-results.json",
    relationship: "prior_three_arm_adaptive_batching_study_at_1000_rps",
  },
] as const;

export const EXPECTED_LIMITATIONS = [
  "Arrival fidelity was not certified; phase_boundary_fidelity remains unavailable.",
  "No historical trials served as controls. Prior A/G2 and A/G2/G3 publications are not combined into this paired estimate.",
  "Completion timing and batch-decision instrumentation were OFF. Counted trials used schema 3; schema-4 completion timing was not enabled.",
  "The 5,000,000 nanosecond maximum generator-lag screen is a coarse non-gating lateness bound, not precise scheduling certification.",
  "Three counted trials missed that coarse screen: G2 block 14, G4 block 16, and G2 block 18.",
  "Producer-observed latency includes the 1 ms open-loop completion-observation quantum.",
  "Queue waiting was not separately instrumented in schema 3, so isolated queue-wait reduction is not claimed.",
  "Sparse traffic produced almost exclusively size-one batches; this study does not demonstrate larger G4 batches.",
  "G4 seal reasons cannot be inferred from schema-3 batch histograms.",
  "Prior instrumented calibration traces are supporting mechanism context only and are not contemporaneous paired evidence from this campaign.",
  "The study used one Darwin arm64 Apple Silicon host.",
  "The model was CPU GEMM (gemm_f32) on ONNX Runtime 1.31.0 CPU, not a GPU or production LLM workload.",
  "Successful throughput was similar because the offered rate was frozen; this is not a throughput gain.",
  "Results should not be generalized to all workloads, hosts, saturation regimes, or scheduler configurations.",
  "The untracked local raw run set is not automatically available to external readers and has not been published or archived as public evidence.",
  "Bootstrap bit-identity is not promised across all software environments.",
] as const;

const FORBIDDEN_CLAIM_FLAGS = [
  "winner_declared",
  "superiority_claimed",
  "universal_scheduler_superiority_claimed",
  "guaranteed_speedup_claimed",
  "gpu_optimization_claimed",
  "llm_speedup_claimed",
  "throughput_increase_claimed",
  "certified_arrival_fidelity_claimed",
  "isolated_queue_wait_reduction_claimed",
  "generic_3x_speedup_headline_claimed",
  "larger_g4_batches_demonstrated",
] as const;

export type RelatedStudy = {
  contrastId: string;
  markdown: string;
  resultsJson: string;
  relationship: string;
  combinable: false;
};

export type GemmG2G4Publication = {
  contrastId: string;
  g2P50: number;
  g4P50: number;
  meanPairedDiff: number;
  ciLow: number;
  ciHigh: number;
  pairedBlocks: number;
  offeredRps: number;
  workerCount: number;
  modelId: string;
  onnxRuntime: string;
  executionProvider: string;
  bootstrapMethod: string;
  bootstrapReplicates: number;
  bootstrapSeed: number;
  experimentalUnit: string;
  primaryEndpoint: string;
  reportSchemaVersion: number;
  completionTiming: string;
  batchDecisionInstrumentation: string;
  hostOs: string;
  hostArch: string;
  arrivalStatus: string;
  coarseLagMissCount: number;
  arrivalScreenGating: boolean;
  batchingNote: string;
  limitations: readonly string[];
  rawEvidencePublished: boolean;
  winnerDeclared: boolean;
  superiorityClaimed: boolean;
  related: readonly RelatedStudy[];
};

export class PublicationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PublicationError";
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireRecord(value: unknown, label: string): Record<string, unknown> {
  if (!isRecord(value)) {
    throw new PublicationError(`${label} is missing or not an object`);
  }
  return value;
}

function requireString(record: Record<string, unknown>, key: string, label: string): string {
  const value = record[key];
  if (typeof value !== "string" || value.length === 0) {
    throw new PublicationError(`${label} is missing`);
  }
  return value;
}

function requireNumber(record: Record<string, unknown>, key: string, label: string): number {
  const value = record[key];
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new PublicationError(`${label} is missing`);
  }
  return value;
}

function requireBoolean(record: Record<string, unknown>, key: string, label: string): boolean {
  const value = record[key];
  if (typeof value !== "boolean") {
    throw new PublicationError(`${label} is missing`);
  }
  return value;
}

function requireEqual(actual: number, expected: number, label: string): void {
  if (actual !== expected) {
    throw new PublicationError(`${label} does not match the published extract`);
  }
}

function requireText(actual: string, expected: string, label: string): void {
  if (actual !== expected) {
    throw new PublicationError(`${label} does not match the published extract`);
  }
}

function requireFalse(record: Record<string, unknown>, key: string, label: string): void {
  if (requireBoolean(record, key, label)) {
    throw new PublicationError(`${label} must remain false`);
  }
}

function parseRelated(value: unknown): readonly RelatedStudy[] {
  if (!Array.isArray(value)) {
    throw new PublicationError("related_publications is missing");
  }
  if (value.length !== EXPECTED_RELATED.length) {
    throw new PublicationError("Unexpected related publication set");
  }
  return value.map((item, index) => {
    const record = requireRecord(item, `related_publications[${index}]`);
    const expected = EXPECTED_RELATED[index];
    if (expected === undefined) {
      throw new PublicationError("Unexpected related publication set");
    }
    if (record.combinable_into_paired_estimate !== false) {
      throw new PublicationError(`related_publications[${index}] must not be combinable`);
    }
    const contrastId = requireString(record, "contrast_id", `related contrast ${index}`);
    const markdown = requireString(record, "markdown", `related markdown ${index}`);
    const resultsJson = requireString(record, "results_json", `related JSON ${index}`);
    const relationship = requireString(record, "relationship", `related relationship ${index}`);
    requireText(contrastId, expected.contrastId, `related contrast ${index}`);
    requireText(markdown, expected.markdown, `related markdown ${index}`);
    requireText(resultsJson, expected.resultsJson, `related JSON ${index}`);
    requireText(relationship, expected.relationship, `related relationship ${index}`);
    return { contrastId, markdown, resultsJson, relationship, combinable: false };
  });
}

function parseLimitations(value: unknown): readonly string[] {
  if (!Array.isArray(value)) {
    throw new PublicationError("limitations is missing");
  }
  const limitations = value.map((item, index) => {
    if (typeof item !== "string" || item.length === 0) {
      throw new PublicationError(`limitations[${index}] is missing`);
    }
    return item;
  });
  if (limitations.length !== EXPECTED_LIMITATIONS.length) {
    throw new PublicationError("limitations does not match the published extract");
  }
  EXPECTED_LIMITATIONS.forEach((expected, index) => {
    if (limitations[index] !== expected) {
      throw new PublicationError(`limitations[${index}] does not match the published extract`);
    }
  });
  return limitations;
}

function requireClaimPolicy(root: Record<string, unknown>, provenance: Record<string, unknown>): void {
  requireFalse(root, "winner_declared", "winner_declared");
  requireFalse(root, "superiority_claimed", "superiority_claimed");
  requireFalse(root, "raw_evidence_published_on_github", "raw evidence flag");
  requireFalse(provenance, "winner_declared", "provenance.winner_declared");
  requireFalse(provenance, "superiority_claimed", "provenance.superiority_claimed");

  const claims = requireRecord(root.claim_policy, "claim_policy");
  for (const flag of FORBIDDEN_CLAIM_FLAGS) {
    requireFalse(claims, flag, `claim_policy.${flag}`);
  }
  if (claims.winner_declared !== root.winner_declared || claims.superiority_claimed !== root.superiority_claimed) {
    throw new PublicationError("claim_policy flags do not agree with the publication root");
  }
  if (
    provenance.winner_declared !== root.winner_declared ||
    provenance.superiority_claimed !== root.superiority_claimed
  ) {
    throw new PublicationError("provenance claim flags do not agree with the publication root");
  }
}

export function parseGemmG2G4Publication(value: unknown): GemmG2G4Publication {
  const root = requireRecord(value, "publication");
  const contrastId = requireString(root, "contrast_id", "contrast_id");
  requireText(contrastId, EXPECTED_CONTRAST, "contrast_id");
  requireText(requireString(root, "evidence_class", "evidence_class"), "confirmatory", "evidence_class");
  requireText(requireString(root, "collection_mode", "collection_mode"), "reportable", "collection_mode");
  requireText(
    requireString(root, "contrast_sign_convention", "contrast_sign_convention"),
    "G4_minus_G2",
    "contrast_sign_convention",
  );
  requireText(
    requireString(root, "primary_endpoint", "primary_endpoint"),
    "successful_e2e_latency.p50_microseconds",
    "primary_endpoint",
  );

  const model = requireRecord(root.model, "model");
  const workload = requireRecord(root.workload, "workload");
  const provenance = requireRecord(root.provenance, "provenance");
  const environment = requireRecord(root.environment, "environment");
  const design = requireRecord(root.experimental_design, "experimental_design");
  const arrival = requireRecord(root.arrival_policy, "arrival_policy");
  const primary = requireRecord(root.primary_result, "primary_result");
  const ci = requireRecord(primary.confidence_interval_95_percent, "confidence_interval_95_percent");
  const bootstrap = requireRecord(primary.bootstrap, "bootstrap");
  const batching = requireRecord(root.actual_batching, "actual_batching");

  const g2P50 = requireNumber(primary, "G2_p50_mean_across_blocks_microseconds", "G2 P50");
  const g4P50 = requireNumber(primary, "G4_p50_mean_across_blocks_microseconds", "G4 P50");
  const meanPairedDiff = requireNumber(primary, "mean_paired_effect_microseconds", "paired difference");
  const ciLow = requireNumber(ci, "low_microseconds", "CI low");
  const ciHigh = requireNumber(ci, "high_microseconds", "CI high");
  const pairedBlocks = requireNumber(design, "counted_paired_blocks", "paired blocks");
  const offeredRps = requireNumber(workload, "offered_rate_requests_per_second", "offered rate");
  const workerCount = requireNumber(workload, "worker_count", "worker count");
  const bootstrapSeed = requireNumber(provenance, "bootstrap_seed", "bootstrap seed");
  const bootstrapReplicates = requireNumber(provenance, "bootstrap_replicates", "bootstrap replicates");
  const reportSchemaVersion = requireNumber(provenance, "report_schema_version", "report schema version");
  const coarseLagMissCount = requireNumber(
    arrival,
    "counted_coarse_screen_miss_count",
    "coarse lag miss count",
  );

  requireEqual(g2P50, EXPECTED_G2_P50, "G2 P50");
  requireEqual(g4P50, EXPECTED_G4_P50, "G4 P50");
  requireEqual(meanPairedDiff, EXPECTED_PAIRED_DIFF, "paired difference");
  requireEqual(ciLow, EXPECTED_CI_LOW, "CI low");
  requireEqual(ciHigh, EXPECTED_CI_HIGH, "CI high");
  requireEqual(pairedBlocks, EXPECTED_BLOCKS, "paired blocks");
  requireEqual(offeredRps, EXPECTED_RPS, "offered rate");
  requireEqual(workerCount, EXPECTED_WORKERS, "worker count");
  requireEqual(bootstrapSeed, EXPECTED_SEED, "bootstrap seed");
  requireEqual(bootstrapReplicates, EXPECTED_REPLICATES, "bootstrap replicates");
  requireEqual(requireNumber(bootstrap, "seed", "primary bootstrap seed"), EXPECTED_SEED, "primary bootstrap seed");
  requireEqual(
    requireNumber(bootstrap, "replicates", "primary bootstrap replicates"),
    EXPECTED_REPLICATES,
    "primary bootstrap replicates",
  );
  requireEqual(reportSchemaVersion, 3, "report schema version");
  requireEqual(coarseLagMissCount, EXPECTED_LAG_MISSES, "coarse lag miss count");

  requireText(requireString(design, "baseline_cell", "baseline_cell"), "G2", "baseline_cell");
  requireText(requireString(design, "alternative_cell", "alternative_cell"), "G4", "alternative_cell");
  requireText(
    requireString(primary, "effect_sign_convention", "primary effect sign"),
    "G4_minus_G2",
    "primary effect sign",
  );

  const modelId = requireString(model, "model_id", "model_id");
  requireText(modelId, "gemm_f32", "model_id");

  const endpoint = requireString(primary, "metric", "primary metric");
  requireText(endpoint, "producer_observed_p50_successful_e2e_latency", "primary metric");

  const bootstrapMethod = requireString(bootstrap, "method", "bootstrap method");
  requireText(bootstrapMethod, "paired_block_bootstrap_mean", "bootstrap method");
  requireText(
    requireString(provenance, "bootstrap_method", "provenance bootstrap method"),
    "paired_block_bootstrap_mean",
    "provenance bootstrap method",
  );

  const experimentalUnit = requireString(bootstrap, "experimental_unit", "experimental unit");
  requireText(experimentalUnit, "paired_block", "experimental unit");
  requireText(
    requireString(provenance, "experimental_unit", "provenance experimental unit"),
    "paired_block",
    "provenance experimental unit",
  );

  const executionProvider = requireString(provenance, "execution_provider", "execution provider");
  requireText(executionProvider, "CPU", "execution provider");

  const completionTiming = requireString(provenance, "completion_timing", "completion timing");
  requireText(completionTiming, "off", "completion timing");
  const batchDecisionInstrumentation = requireString(
    provenance,
    "batch_decision_instrumentation",
    "batch-decision instrumentation",
  );
  requireText(batchDecisionInstrumentation, "off", "batch-decision instrumentation");

  const arrivalStatus = requireString(arrival, "arrival_fidelity_status", "arrival fidelity");
  requireText(arrivalStatus, "not_certified", "arrival fidelity");
  const arrivalScreenGating = requireBoolean(arrival, "arrival_screen_gating", "arrival screen gating");
  if (arrivalScreenGating) {
    throw new PublicationError("arrival screen must remain non-gating");
  }

  const batchingNote = requireString(batching, "note", "actual_batching.note");
  requireText(
    batchingNote,
    "Sparse 100 RPS traffic produced almost exclusively size-one batches. This study does not demonstrate larger G4 batches.",
    "actual_batching.note",
  );

  requireClaimPolicy(root, provenance);
  const limitations = parseLimitations(root.limitations);

  return {
    contrastId,
    g2P50,
    g4P50,
    meanPairedDiff,
    ciLow,
    ciHigh,
    pairedBlocks,
    offeredRps,
    workerCount,
    modelId,
    onnxRuntime: requireString(provenance, "onnxruntime_version", "ONNX Runtime version"),
    executionProvider,
    bootstrapMethod,
    bootstrapReplicates,
    bootstrapSeed,
    experimentalUnit,
    primaryEndpoint: endpoint,
    reportSchemaVersion,
    completionTiming,
    batchDecisionInstrumentation,
    hostOs: requireString(environment, "operating_system", "operating system"),
    hostArch: requireString(environment, "cpu_architecture", "CPU architecture"),
    arrivalStatus,
    coarseLagMissCount,
    arrivalScreenGating,
    batchingNote,
    limitations,
    rawEvidencePublished: false,
    winnerDeclared: false,
    superiorityClaimed: false,
    related: parseRelated(root.related_publications),
  };
}

export function formatMicros(value: number): string {
  if (value === EXPECTED_CI_HIGH) {
    return "−2511.35629875";
  }
  const sign = value < 0 ? "−" : "";
  return `${sign}${Math.abs(value)}`;
}

export function formatCi(low: number, high: number): string {
  return `[${formatMicros(low)}, ${formatMicros(high)}]`;
}

export function publicationUrl(baseUrl: string, destDir: "data" | "figures", name: string): string {
  const root = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  return `${root}${destDir}/${name}`;
}

export function githubDocUrl(name: string): string {
  return `${GITHUB_DOCS}/${name}`;
}
