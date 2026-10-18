export const GENERATIVE_MARKDOWN = "generative-overhead-qwen3-0.6b.md";
export const GENERATIVE_JSON = "generative-overhead-qwen3-0.6b-results.json";
export const GENERATIVE_PUBLICATION_TYPE = "generative_overhead_qwen3_0_6b_c01" as const;

export const EXPECTED_STUDY_ID = "generative_overhead_qwen3_0_6b_c01";
export const EXPECTED_CAMPAIGN_ID = "phase-9-b3d-campaign-001";
export const EXPECTED_SOURCE_COMMIT = "f91cb463fd4026a3dd05af4b0ac7c6f9b3182955";
export const EXPECTED_CONTRACT = "generative-overhead-r7-inventory-v1";
export const EXPECTED_MANIFEST_SHA256 = "273fc869d45c26118297c36bdc5a37b7e21ff4845b231845af22d43f0cdef371";
export const EXPECTED_AGGREGATE_SHA256 = "93da8a1fa8c97fc12e7c20079210f5484c1a151a8532fdd8450c3cfaa1ed9887";
export const EXPECTED_SUMMARY_SHA256 = "a4d020038e79b13ae4c3aa7cf0463cec0602d19c0010083fed4d850bab0eac48";
export const EXPECTED_PUBLICATION_STATEMENT =
  "In one protocol-valid 20-pair C01 campaign on the frozen Qwen3-0.6B package and tested Apple host, neither implementation satisfied all prespecified practical-threshold and strict 95% bootstrap-interval criteria for a directional winner.";

export const EXPECTED_SUM_IE_NS = 1490295957;
export const EXPECTED_SUM_REF_NS = 1493768458;
export const EXPECTED_SUM_D_NS = -3472501;
export const EXPECTED_INFEREDGE_MEAN_NS = 74514797.85;
export const EXPECTED_REFERENCE_MEAN_NS = 74688422.9;
export const EXPECTED_MEAN_DIFFERENCE_NS = -173625.05;
export const EXPECTED_CI_LOW_NS = -1060655.6224999994;
export const EXPECTED_CI_HIGH_NS = 730089.1224999995;
export const EXPECTED_BOOTSTRAP_REPLICATES = 10000;
export const EXPECTED_BOOTSTRAP_SEED = 20261018;
export const EXPECTED_CONFIDENCE_LEVEL = 0.95;
export const EXPECTED_PAIR_COUNT = 20;

const EXPECTED_PAIRS = [
  ["00", "ie_then_ref", 75366500, 74909250, 457250],
  ["01", "ref_then_ie", 75062708, 75750333, -687625],
  ["02", "ie_then_ref", 75077459, 75553250, -475791],
  ["03", "ref_then_ie", 73028750, 75614417, -2585667],
  ["04", "ie_then_ref", 74766208, 76357209, -1591001],
  ["05", "ref_then_ie", 74979541, 74404208, 575333],
  ["06", "ie_then_ref", 71440458, 75742834, -4302376],
  ["07", "ref_then_ie", 75316291, 75732625, -416334],
  ["08", "ie_then_ref", 74644417, 71888916, 2755501],
  ["09", "ref_then_ie", 74682583, 74466208, 216375],
  ["10", "ie_then_ref", 74888375, 75001666, -113291],
  ["11", "ref_then_ie", 75210917, 72143959, 3066958],
  ["12", "ie_then_ref", 74106458, 75361625, -1255167],
  ["13", "ref_then_ie", 75050542, 75458250, -407708],
  ["14", "ie_then_ref", 75767083, 72174708, 3592375],
  ["15", "ref_then_ie", 75470125, 75271250, 198875],
  ["16", "ie_then_ref", 72016292, 74339000, -2322708],
  ["17", "ref_then_ie", 71963667, 75393834, -3430167],
  ["18", "ie_then_ref", 75715250, 74881041, 834209],
  ["19", "ref_then_ie", 75742333, 73323875, 2418458],
] as const;

const DISPLAY_MILLISECONDS = new Map<number, string>([
  [EXPECTED_INFEREDGE_MEAN_NS, "74.51479785"],
  [EXPECTED_REFERENCE_MEAN_NS, "74.6884229"],
  [EXPECTED_MEAN_DIFFERENCE_NS, "-0.17362505"],
  [EXPECTED_CI_LOW_NS, "-1.0606556224999994"],
  [EXPECTED_CI_HIGH_NS, "0.7300891224999995"],
]);

const TOP_LEVEL_KEYS = [
  "publication_format_version",
  "study_id",
  "title",
  "evidence_class",
  "campaign",
  "scope",
  "provenance",
  "design",
  "primary_c01",
  "secondary_cells_policy",
  "claim_policy",
  "limitations",
  "raw_evidence_published_on_github",
  "publication_statement",
] as const;

const CAMPAIGN_KEYS = [
  "campaign_class",
  "campaign_id",
  "contract",
  "source_commit",
  "start_utc",
  "end_utc",
  "complete_matrix",
  "rejection_codes",
] as const;

const PROVENANCE_KEYS = [
  "campaign_class",
  "campaign_id",
  "source_commit",
  "protocol_document_sha256",
  "collector_script_sha256",
  "analyzer_script_sha256",
  "fixtures_sha256",
  "gguf_sha256",
  "gguf_size_bytes",
  "inferedge_worker_sha256",
  "reference_worker_sha256",
  "campaign_manifest_sha256",
  "aggregate_sha256",
  "summary_sha256",
  "schema_family",
  "schema_version",
  "contract",
  "compiler_identity",
  "build_type",
  "model_id",
  "gguf_filename",
  "quantization",
  "llama_tag",
  "llama_commit",
  "campaign_start_utc",
  "campaign_end_utc",
  "collector_process_start_utc",
  "collector_process_end_utc",
  "analyzer_process_start_utc",
  "aggregate_summary_deterministic_reproduction",
  "source_ci_run_url",
  "ci_run_url",
] as const;

const PRIMARY_KEYS = [
  "cell_id",
  "role",
  "pair_count",
  "sum_ie_ns",
  "sum_ref_ns",
  "sum_d_ns",
  "inferedge_mean_ns",
  "reference_mean_ns",
  "mean_difference_ns",
  "signed_direction_label",
  "signed_direction_is_winner",
  "bootstrap_replicates",
  "bootstrap_seed",
  "confidence_level",
  "ci_low_ns",
  "ci_high_ns",
  "absolute_threshold_per_pair_ns",
  "absolute_threshold_20_pair_ns",
  "absolute_threshold_met",
  "relative_threshold_numerator",
  "relative_threshold_denominator",
  "relative_threshold_met",
  "ci_excludes_zero_strict",
  "winner_declared",
  "superiority_claimed",
  "practical_equivalence_claimed",
  "claim_status",
  "complete_matrix",
  "rejection_codes",
  "descriptive_statistics",
  "pairs",
] as const;

const CLAIM_KEYS = [
  "primary_cell",
  "estimator",
  "pair_count",
  "bootstrap_replicates",
  "bootstrap_seed",
  "confidence_level",
  "absolute_threshold_ns",
  "relative_threshold_numerator",
  "relative_threshold_denominator",
  "both_thresholds_required",
  "strict_directional_ci_exclusion_required",
  "secondary_cells_cannot_rescue_c01",
  "equivalence_inference_allowed",
  "winner_declared",
  "superiority_claimed",
  "practical_equivalence_claimed",
  "claim_status",
  "complete_matrix",
  "rejection_codes",
] as const;

const LIMITATIONS = [
  "one_host",
  "one_official_campaign",
  "one_frozen_qwen3_0_6b_q4_k_m_package",
  "one_primary_prompt_token_thread_configuration",
  "twenty_paired_c01_observations",
  "cpu_native_backend_as_recorded",
  "no_gpu_conclusion",
  "no_concurrency_conclusion",
  "no_energy_conclusion",
  "no_model_quality_conclusion",
  "no_training_or_continual_learning_conclusion",
  "no_generalization_to_other_qwen_or_llama_models",
  "no_production_wide_latency_conclusion",
  "no_practical_equivalence_conclusion",
  "raw_evidence_local_unpublished",
] as const;

export type GenerativeOverheadPublication = {
  publicationType: typeof GENERATIVE_PUBLICATION_TYPE;
  studyId: string;
  title: string;
  modelId: string;
  quantization: string;
  ggufFilename: string;
  hostClass: string;
  hardwareModel: string;
  pairCount: number;
  inferedgeMeanNs: number;
  referenceMeanNs: number;
  meanDifferenceNs: number;
  ciLowNs: number;
  ciHighNs: number;
  absoluteThresholdMet: false;
  relativeThresholdMet: false;
  ciExcludesZeroStrict: false;
  winnerDeclared: false;
  superiorityClaimed: false;
  practicalEquivalenceClaimed: false;
  claimStatus: "no_winner_below_threshold_or_ci";
  signedDirectionLabel: "inferedge_faster";
  signedDirectionIsWinner: false;
  publicationStatement: string;
  rawEvidencePublished: false;
};

export class GenerativePublicationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GenerativePublicationError";
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireRecord(value: unknown, label: string): Record<string, unknown> {
  if (!isRecord(value)) {
    throw new GenerativePublicationError(`${label} is missing or not an object`);
  }
  return value;
}

function requireExactKeys(record: Record<string, unknown>, keys: readonly string[], label: string): void {
  const actual = Object.keys(record);
  if (actual.length !== keys.length || keys.some((key) => !Object.prototype.hasOwnProperty.call(record, key))) {
    throw new GenerativePublicationError(`${label} keys do not match the published extract`);
  }
}

function requireString(record: Record<string, unknown>, key: string, label: string): string {
  const value = record[key];
  if (typeof value !== "string" || value.length === 0) {
    throw new GenerativePublicationError(`${label} is missing`);
  }
  return value;
}

function requireText(actual: string, expected: string, label: string): void {
  if (actual !== expected) {
    throw new GenerativePublicationError(`${label} does not match the published extract`);
  }
}

function requireFiniteNumber(record: Record<string, unknown>, key: string, label: string): number {
  const value = record[key];
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new GenerativePublicationError(`${label} is missing or non-finite`);
  }
  return value;
}

function requireInteger(record: Record<string, unknown>, key: string, label: string): number {
  const value = requireFiniteNumber(record, key, label);
  if (!Number.isInteger(value)) {
    throw new GenerativePublicationError(`${label} must be an integer`);
  }
  return value;
}

function requireEqual(actual: number, expected: number, label: string): void {
  if (actual !== expected) {
    throw new GenerativePublicationError(`${label} does not match the published extract`);
  }
}

function requireFalse(value: unknown, label: string): false {
  if (value !== false) {
    throw new GenerativePublicationError(`${label} must remain false`);
  }
  return false;
}

function requireTrue(value: unknown, label: string): true {
  if (value !== true) {
    throw new GenerativePublicationError(`${label} must remain true`);
  }
  return true;
}

function requireEmptyCodes(value: unknown, label: string): void {
  if (!Array.isArray(value) || value.length !== 0) {
    throw new GenerativePublicationError(`${label} must be empty`);
  }
}

function requireLimitations(value: unknown): void {
  if (!Array.isArray(value) || value.length !== LIMITATIONS.length) {
    throw new GenerativePublicationError("limitations do not match the published extract");
  }
  LIMITATIONS.forEach((expected, index) => {
    if (value[index] !== expected) {
      throw new GenerativePublicationError(`limitations[${index}] does not match the published extract`);
    }
  });
}

function requirePairs(value: unknown): void {
  if (!Array.isArray(value) || value.length !== EXPECTED_PAIR_COUNT) {
    throw new GenerativePublicationError("pair count does not match the published extract");
  }
  const seen = new Set<string>();
  value.forEach((item, index) => {
    const pair = requireRecord(item, `pairs[${index}]`);
    requireExactKeys(
      pair,
      ["pair_id", "arm_order", "inferedge_duration_ns", "reference_duration_ns", "difference_ns"],
      `pairs[${index}]`,
    );
    const expected = EXPECTED_PAIRS[index];
    if (expected === undefined) {
      throw new GenerativePublicationError("pair count does not match the published extract");
    }
    const pairId = requireString(pair, "pair_id", `pairs[${index}].pair_id`);
    const armOrder = requireString(pair, "arm_order", `pairs[${index}].arm_order`);
    const inferedge = requireInteger(pair, "inferedge_duration_ns", `pairs[${index}] InferEdge duration`);
    const reference = requireInteger(pair, "reference_duration_ns", `pairs[${index}] reference duration`);
    const difference = requireInteger(pair, "difference_ns", `pairs[${index}] difference`);
    if (seen.has(pairId)) {
      throw new GenerativePublicationError(`duplicate pair id ${pairId}`);
    }
    seen.add(pairId);
    requireText(pairId, expected[0], `pairs[${index}].pair_id`);
    requireText(armOrder, expected[1], `pairs[${index}].arm_order`);
    requireEqual(inferedge, expected[2], `pairs[${index}] InferEdge duration`);
    requireEqual(reference, expected[3], `pairs[${index}] reference duration`);
    if (difference !== inferedge - reference) {
      throw new GenerativePublicationError(`pairs[${index}] difference is not InferEdge minus reference`);
    }
    requireEqual(difference, expected[4], `pairs[${index}] difference`);
  });
}

export function parseGenerativeOverheadPublication(value: unknown): GenerativeOverheadPublication {
  const root = requireRecord(value, "publication");
  requireExactKeys(root, TOP_LEVEL_KEYS, "publication");
  requireEqual(requireInteger(root, "publication_format_version", "publication format version"), 1, "publication format version");
  requireText(requireString(root, "study_id", "study id"), EXPECTED_STUDY_ID, "study id");
  requireText(
    requireString(root, "title", "title"),
    "Qwen3-0.6B generative-overhead benchmark: no directional winner under the frozen C01 protocol",
    "title",
  );
  requireText(requireString(root, "evidence_class", "evidence class"), "official_campaign", "evidence class");
  requireFalse(root.raw_evidence_published_on_github, "raw evidence flag");
  requireText(
    requireString(root, "publication_statement", "publication statement"),
    EXPECTED_PUBLICATION_STATEMENT,
    "publication statement",
  );
  requireLimitations(root.limitations);

  const campaign = requireRecord(root.campaign, "campaign");
  requireExactKeys(campaign, CAMPAIGN_KEYS, "campaign");
  requireText(requireString(campaign, "campaign_class", "campaign class"), "campaign", "campaign class");
  requireText(requireString(campaign, "campaign_id", "campaign id"), EXPECTED_CAMPAIGN_ID, "campaign id");
  requireText(requireString(campaign, "contract", "contract"), EXPECTED_CONTRACT, "contract");
  requireText(requireString(campaign, "source_commit", "source commit"), EXPECTED_SOURCE_COMMIT, "source commit");
  requireText(requireString(campaign, "start_utc", "campaign start"), "2026-10-18T16:35:52.665569Z", "campaign start");
  requireText(requireString(campaign, "end_utc", "campaign end"), "2026-10-18T17:22:57.891498Z", "campaign end");
  requireTrue(campaign.complete_matrix, "complete matrix");
  requireEmptyCodes(campaign.rejection_codes, "rejection codes");

  const scope = requireRecord(root.scope, "scope");
  requireExactKeys(
    scope,
    ["model_id", "gguf_filename", "quantization", "host_class", "host", "primary_cell", "pair_count", "comparison", "backend"],
    "scope",
  );
  const modelId = requireString(scope, "model_id", "model id");
  requireText(modelId, "qwen3-0.6b", "model id");
  const ggufFilename = requireString(scope, "gguf_filename", "gguf filename");
  requireText(ggufFilename, "qwen3-0.6b-Q4_K_M.gguf", "gguf filename");
  const quantization = requireString(scope, "quantization", "quantization");
  requireText(quantization, "Q4_K_M", "quantization");
  const hostClass = requireString(scope, "host_class", "host class");
  requireText(hostClass, "one_tested_apple_silicon_host", "host class");
  requireText(requireString(scope, "primary_cell", "scope primary cell"), "C01", "scope primary cell");
  requireEqual(requireInteger(scope, "pair_count", "scope pair count"), EXPECTED_PAIR_COUNT, "scope pair count");
  requireText(
    requireString(scope, "comparison", "comparison"),
    "inferedge_integration_versus_pinned_direct_llamacpp",
    "comparison",
  );
  requireText(requireString(scope, "backend", "backend"), "cpu_native_as_recorded", "backend");
  const host = requireRecord(scope.host, "host");
  requireExactKeys(
    host,
    ["operating_system", "cpu_architecture", "hardware_model", "os_product_name", "os_product_version"],
    "host",
  );
  requireText(requireString(host, "operating_system", "operating system"), "Darwin", "operating system");
  requireText(requireString(host, "cpu_architecture", "cpu architecture"), "arm64", "cpu architecture");
  const hardwareModel = requireString(host, "hardware_model", "hardware model");
  requireText(hardwareModel, "Mac17,8", "hardware model");
  requireText(requireString(host, "os_product_name", "os product name"), "macOS", "os product name");
  requireText(requireString(host, "os_product_version", "os product version"), "27.0.1", "os product version");

  const provenance = requireRecord(root.provenance, "provenance");
  requireExactKeys(provenance, PROVENANCE_KEYS, "provenance");
  requireText(requireString(provenance, "campaign_class", "provenance campaign class"), "campaign", "provenance campaign class");
  requireText(requireString(provenance, "campaign_id", "provenance campaign id"), EXPECTED_CAMPAIGN_ID, "provenance campaign id");
  requireText(requireString(provenance, "source_commit", "provenance source commit"), EXPECTED_SOURCE_COMMIT, "provenance source commit");
  requireText(requireString(provenance, "contract", "provenance contract"), EXPECTED_CONTRACT, "provenance contract");
  requireText(
    requireString(provenance, "campaign_manifest_sha256", "manifest hash"),
    EXPECTED_MANIFEST_SHA256,
    "manifest hash",
  );
  requireText(requireString(provenance, "aggregate_sha256", "aggregate hash"), EXPECTED_AGGREGATE_SHA256, "aggregate hash");
  requireText(requireString(provenance, "summary_sha256", "summary hash"), EXPECTED_SUMMARY_SHA256, "summary hash");
  requireEqual(requireInteger(provenance, "schema_version", "schema version"), 1, "schema version");
  requireEqual(requireInteger(provenance, "gguf_size_bytes", "gguf size"), 484220192, "gguf size");
  requireText(requireString(provenance, "schema_family", "schema family"), "generative_overhead", "schema family");
  requireText(requireString(provenance, "model_id", "provenance model id"), "qwen3-0.6b", "provenance model id");
  requireText(requireString(provenance, "campaign_start_utc", "provenance campaign start"), "2026-10-18T16:35:52.665569Z", "provenance campaign start");
  requireText(requireString(provenance, "campaign_end_utc", "provenance campaign end"), "2026-10-18T17:22:57.891498Z", "provenance campaign end");
  requireText(
    requireString(provenance, "collector_process_start_utc", "collector process start"),
    "2026-10-18T16:35:50.535790Z",
    "collector process start",
  );
  requireText(
    requireString(provenance, "collector_process_end_utc", "collector process end"),
    "2026-10-18T17:22:58.079740Z",
    "collector process end",
  );
  requireText(
    requireString(provenance, "analyzer_process_start_utc", "analyzer process start"),
    "2026-10-18T17:23:52.621731Z",
    "analyzer process start",
  );
  for (const key of [
    "protocol_document_sha256",
    "collector_script_sha256",
    "analyzer_script_sha256",
    "fixtures_sha256",
    "gguf_sha256",
    "inferedge_worker_sha256",
    "reference_worker_sha256",
    "compiler_identity",
    "build_type",
    "gguf_filename",
    "quantization",
    "llama_tag",
    "llama_commit",
    "aggregate_summary_deterministic_reproduction",
    "source_ci_run_url",
    "ci_run_url",
  ] as const) {
    requireString(provenance, key, key);
  }
  requireText(requireString(provenance, "gguf_sha256", "gguf hash"), "3f384c3796ca2547c5547be6bca3d487a18807cc1a2c259de6e0e1fa433bb76d", "gguf hash");
  requireText(
    requireString(provenance, "source_ci_run_url", "source ci url"),
    "https://github.com/KasaVarun/inferedge/actions/runs/39143884593",
    "source ci url",
  );
  requireText(requireString(provenance, "ci_run_url", "ci url"), "https://github.com/KasaVarun/inferedge/actions/runs/39143884593", "ci url");

  const design = requireRecord(root.design, "design");
  requireExactKeys(
    design,
    [
      "primary_cell",
      "fixture_id",
      "prompt_token_count",
      "max_new_tokens",
      "threads",
      "measurement_operation",
      "pair_count",
      "pair_order_rule",
      "n_ctx",
      "n_batch",
      "n_ubatch",
      "n_seq_max",
      "warmup_count_per_worker",
      "difference_definition",
    ],
    "design",
  );
  requireText(requireString(design, "primary_cell", "design primary cell"), "C01", "design primary cell");
  requireText(requireString(design, "fixture_id", "fixture"), "P_tiny", "fixture");
  requireEqual(requireInteger(design, "prompt_token_count", "prompt token count"), 13, "prompt token count");
  requireEqual(requireInteger(design, "max_new_tokens", "max new tokens"), 1, "max new tokens");
  requireEqual(requireInteger(design, "threads", "threads"), 1, "threads");
  requireText(requireString(design, "measurement_operation", "measurement operation"), "warm_request", "measurement operation");
  requireEqual(requireInteger(design, "pair_count", "design pair count"), EXPECTED_PAIR_COUNT, "design pair count");
  requireText(
    requireString(design, "pair_order_rule", "pair order rule"),
    "even ie_then_ref, odd ref_then_ie",
    "pair order rule",
  );
  requireEqual(requireInteger(design, "n_ctx", "n_ctx"), 4096, "n_ctx");
  requireEqual(requireInteger(design, "n_batch", "n_batch"), 4096, "n_batch");
  requireEqual(requireInteger(design, "n_ubatch", "n_ubatch"), 512, "n_ubatch");
  requireEqual(requireInteger(design, "n_seq_max", "n_seq_max"), 1, "n_seq_max");
  requireEqual(requireInteger(design, "warmup_count_per_worker", "warmup count"), 2, "warmup count");
  requireText(
    requireString(design, "difference_definition", "difference definition"),
    "inferedge_duration_ns_minus_reference_duration_ns",
    "difference definition",
  );

  const primary = requireRecord(root.primary_c01, "primary_c01");
  requireExactKeys(primary, PRIMARY_KEYS, "primary_c01");
  requireText(requireString(primary, "cell_id", "primary cell"), "C01", "primary cell");
  requireText(requireString(primary, "role", "primary role"), "primary", "primary role");
  requireEqual(requireInteger(primary, "pair_count", "primary pair count"), EXPECTED_PAIR_COUNT, "primary pair count");
  requireEqual(requireInteger(primary, "sum_ie_ns", "InferEdge sum"), EXPECTED_SUM_IE_NS, "InferEdge sum");
  requireEqual(requireInteger(primary, "sum_ref_ns", "reference sum"), EXPECTED_SUM_REF_NS, "reference sum");
  requireEqual(requireInteger(primary, "sum_d_ns", "paired sum"), EXPECTED_SUM_D_NS, "paired sum");
  requireEqual(requireFiniteNumber(primary, "inferedge_mean_ns", "InferEdge mean"), EXPECTED_INFEREDGE_MEAN_NS, "InferEdge mean");
  requireEqual(requireFiniteNumber(primary, "reference_mean_ns", "reference mean"), EXPECTED_REFERENCE_MEAN_NS, "reference mean");
  requireEqual(requireFiniteNumber(primary, "mean_difference_ns", "mean difference"), EXPECTED_MEAN_DIFFERENCE_NS, "mean difference");
  requireText(requireString(primary, "signed_direction_label", "signed direction"), "inferedge_faster", "signed direction");
  requireFalse(primary.signed_direction_is_winner, "signed direction winner flag");
  requireEqual(requireInteger(primary, "bootstrap_replicates", "bootstrap replicates"), EXPECTED_BOOTSTRAP_REPLICATES, "bootstrap replicates");
  requireEqual(requireInteger(primary, "bootstrap_seed", "bootstrap seed"), EXPECTED_BOOTSTRAP_SEED, "bootstrap seed");
  requireEqual(requireFiniteNumber(primary, "confidence_level", "confidence level"), EXPECTED_CONFIDENCE_LEVEL, "confidence level");
  requireEqual(requireFiniteNumber(primary, "ci_low_ns", "CI low"), EXPECTED_CI_LOW_NS, "CI low");
  requireEqual(requireFiniteNumber(primary, "ci_high_ns", "CI high"), EXPECTED_CI_HIGH_NS, "CI high");
  requireEqual(requireInteger(primary, "absolute_threshold_per_pair_ns", "per-pair absolute threshold"), 2000000, "per-pair absolute threshold");
  requireEqual(requireInteger(primary, "absolute_threshold_20_pair_ns", "20-pair absolute threshold"), 40000000, "20-pair absolute threshold");
  requireFalse(primary.absolute_threshold_met, "absolute threshold");
  requireEqual(requireInteger(primary, "relative_threshold_numerator", "relative numerator"), 5, "relative numerator");
  requireEqual(requireInteger(primary, "relative_threshold_denominator", "relative denominator"), 100, "relative denominator");
  requireFalse(primary.relative_threshold_met, "relative threshold");
  requireFalse(primary.ci_excludes_zero_strict, "strict CI exclusion");
  requireFalse(primary.winner_declared, "winner");
  requireFalse(primary.superiority_claimed, "superiority");
  requireFalse(primary.practical_equivalence_claimed, "practical equivalence");
  requireText(requireString(primary, "claim_status", "claim status"), "no_winner_below_threshold_or_ci", "claim status");
  requireTrue(primary.complete_matrix, "primary complete matrix");
  requireEmptyCodes(primary.rejection_codes, "primary rejection codes");

  const stats = requireRecord(primary.descriptive_statistics, "descriptive statistics");
  requireExactKeys(
    stats,
    [
      "affects_winner_decision",
      "p25_difference_ns",
      "p50_difference_ns",
      "median_difference_ns",
      "p75_difference_ns",
      "p95_difference_ns",
      "iqr_difference_ns",
      "mad_difference_ns",
    ],
    "descriptive statistics",
  );
  requireFalse(stats.affects_winner_decision, "descriptive winner effect");
  requireEqual(requireInteger(stats, "p25_difference_ns", "P25"), -1591001, "P25");
  requireEqual(requireInteger(stats, "p50_difference_ns", "P50"), -407708, "P50");
  requireEqual(requireInteger(stats, "median_difference_ns", "median"), -407708, "median");
  requireEqual(requireInteger(stats, "p75_difference_ns", "P75"), 575333, "P75");
  requireEqual(requireInteger(stats, "p95_difference_ns", "P95"), 3066958, "P95");
  requireEqual(requireInteger(stats, "iqr_difference_ns", "IQR"), 2166334, "IQR");
  requireEqual(requireInteger(stats, "mad_difference_ns", "MAD"), 983041, "MAD");
  requirePairs(primary.pairs);

  const secondary = requireRecord(root.secondary_cells_policy, "secondary cell policy");
  requireExactKeys(
    secondary,
    ["C02_through_C11", "L01", "enter_primary_c01_claim", "may_rescue_or_override_c01", "ranking_published", "descriptive_only"],
    "secondary cell policy",
  );
  requireText(requireString(secondary, "C02_through_C11", "C02 through C11"), "descriptive_only", "C02 through C11");
  requireText(requireString(secondary, "L01", "L01"), "descriptive_filesystem_cache_warm_load", "L01");
  requireFalse(secondary.enter_primary_c01_claim, "secondary entry into C01");
  requireFalse(secondary.may_rescue_or_override_c01, "secondary rescue");
  requireFalse(secondary.ranking_published, "secondary ranking");
  requireTrue(secondary.descriptive_only, "secondary descriptive policy");

  const claims = requireRecord(root.claim_policy, "claim policy");
  requireExactKeys(claims, CLAIM_KEYS, "claim policy");
  requireText(requireString(claims, "primary_cell", "claim primary cell"), "C01", "claim primary cell");
  requireText(requireString(claims, "estimator", "estimator"), "mean_paired_difference", "estimator");
  requireEqual(requireInteger(claims, "pair_count", "claim pair count"), EXPECTED_PAIR_COUNT, "claim pair count");
  requireEqual(requireInteger(claims, "bootstrap_replicates", "claim bootstrap replicates"), EXPECTED_BOOTSTRAP_REPLICATES, "claim bootstrap replicates");
  requireEqual(requireInteger(claims, "bootstrap_seed", "claim bootstrap seed"), EXPECTED_BOOTSTRAP_SEED, "claim bootstrap seed");
  requireEqual(requireFiniteNumber(claims, "confidence_level", "claim confidence level"), EXPECTED_CONFIDENCE_LEVEL, "claim confidence level");
  requireEqual(requireInteger(claims, "absolute_threshold_ns", "claim absolute threshold"), 2000000, "claim absolute threshold");
  requireEqual(requireInteger(claims, "relative_threshold_numerator", "claim relative numerator"), 5, "claim relative numerator");
  requireEqual(requireInteger(claims, "relative_threshold_denominator", "claim relative denominator"), 100, "claim relative denominator");
  requireTrue(claims.both_thresholds_required, "both thresholds required");
  requireTrue(claims.strict_directional_ci_exclusion_required, "strict CI exclusion required");
  requireTrue(claims.secondary_cells_cannot_rescue_c01, "secondary cells cannot rescue C01");
  requireFalse(claims.equivalence_inference_allowed, "equivalence inference");
  requireFalse(claims.winner_declared, "claim winner");
  requireFalse(claims.superiority_claimed, "claim superiority");
  requireFalse(claims.practical_equivalence_claimed, "claim practical equivalence");
  requireText(requireString(claims, "claim_status", "claim policy status"), "no_winner_below_threshold_or_ci", "claim policy status");
  requireTrue(claims.complete_matrix, "claim complete matrix");
  requireEmptyCodes(claims.rejection_codes, "claim rejection codes");

  return {
    publicationType: GENERATIVE_PUBLICATION_TYPE,
    studyId: EXPECTED_STUDY_ID,
    title: "Qwen3-0.6B generative-overhead benchmark: no directional winner under the frozen C01 protocol",
    modelId,
    quantization,
    ggufFilename,
    hostClass,
    hardwareModel,
    pairCount: EXPECTED_PAIR_COUNT,
    inferedgeMeanNs: EXPECTED_INFEREDGE_MEAN_NS,
    referenceMeanNs: EXPECTED_REFERENCE_MEAN_NS,
    meanDifferenceNs: EXPECTED_MEAN_DIFFERENCE_NS,
    ciLowNs: EXPECTED_CI_LOW_NS,
    ciHighNs: EXPECTED_CI_HIGH_NS,
    absoluteThresholdMet: false,
    relativeThresholdMet: false,
    ciExcludesZeroStrict: false,
    winnerDeclared: false,
    superiorityClaimed: false,
    practicalEquivalenceClaimed: false,
    claimStatus: "no_winner_below_threshold_or_ci",
    signedDirectionLabel: "inferedge_faster",
    signedDirectionIsWinner: false,
    publicationStatement: EXPECTED_PUBLICATION_STATEMENT,
    rawEvidencePublished: false,
  };
}

export function formatMilliseconds(nanoseconds: number): string {
  const text = DISPLAY_MILLISECONDS.get(nanoseconds);
  if (text === undefined) {
    throw new GenerativePublicationError("millisecond display is only defined for accepted publication values");
  }
  return text;
}

export function formatMillisecondInterval(lowNs: number, highNs: number): string {
  return `[${formatMilliseconds(lowNs)}, ${formatMilliseconds(highNs)}]`;
}

export function publicationUrl(baseUrl: string, name: string): string {
  const root = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  return `${root}data/${name}`;
}
