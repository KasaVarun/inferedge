import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import {
  EXPECTED_CONTRAST,
  formatCi,
  parseGemmG2G4Publication,
  publicationUrl,
} from "../src/performance/gemmPublication.ts";

const websiteRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const published = JSON.parse(
  readFileSync(join(websiteRoot, "../docs/benchmarks/arrival-aware-sparse-gemm-results.json"), "utf8"),
) as unknown;

function clonePublished(): Record<string, unknown> {
  return structuredClone(published) as Record<string, unknown>;
}

function record(value: unknown, label: string): Record<string, unknown> {
  assert.equal(typeof value, "object");
  assert.ok(value !== null);
  return value as Record<string, unknown>;
}

function reject(mutator: (clone: Record<string, unknown>) => void, pattern: RegExp): void {
  const clone = clonePublished();
  mutator(clone);
  assert.throws(() => parseGemmG2G4Publication(clone), pattern);
}

test("parses the published G2 versus G4 extract", () => {
  const result = parseGemmG2G4Publication(published);
  assert.equal(result.contrastId, EXPECTED_CONTRAST);
  assert.equal(result.g2P50, 3792.90625);
  assert.equal(result.g4P50, 1280.4875);
  assert.equal(result.meanPairedDiff, -2512.41875);
  assert.equal(result.ciLow, -2513.64165);
  assert.equal(result.pairedBlocks, 20);
  assert.equal(result.offeredRps, 100);
  assert.equal(result.workerCount, 1);
  assert.equal(result.bootstrapSeed, 20261016);
  assert.equal(result.bootstrapReplicates, 10000);
  assert.equal(result.experimentalUnit, "paired_block");
  assert.equal(result.reportSchemaVersion, 3);
  assert.equal(result.completionTiming, "off");
  assert.equal(result.batchDecisionInstrumentation, "off");
  assert.equal(result.arrivalStatus, "not_certified");
  assert.equal(result.coarseLagMissCount, 3);
  assert.equal(result.arrivalScreenGating, false);
  assert.equal(result.rawEvidencePublished, false);
  assert.equal(result.winnerDeclared, false);
  assert.equal(result.superiorityClaimed, false);
  assert.equal(result.limitations.length, 16);
  assert.equal(result.related.length, 2);
  assert.equal(result.related[0]?.contrastId, "gemm_f32_A_vs_G2_delay2ms_rps1000");
  assert.equal(result.related[1]?.contrastId, "gemm_f32_A_G2_G3_adaptive_rps1000");
  assert.equal(result.related[0]?.combinable, false);
  assert.equal(result.related[1]?.combinable, false);
});

test("formats the published confidence interval without inventing a new value", () => {
  const result = parseGemmG2G4Publication(published);
  assert.equal(formatCi(result.ciLow, result.ciHigh), "[−2513.64165, −2511.35629875]");
});

test("rejects a missing primary result instead of inventing statistics", () => {
  reject((clone) => {
    delete clone.primary_result;
  }, /primary_result/);
});

test("rejects an altered paired difference", () => {
  reject((clone) => {
    record(clone.primary_result, "primary").mean_paired_effect_microseconds = -1;
  }, /paired difference/);
});

test("rejects a wrong contrast identity", () => {
  reject((clone) => {
    clone.contrast_id = "gemm_f32_A_vs_G2_delay2ms_rps1000";
  }, /contrast_id/);
});

test("rejects a wrong baseline cell", () => {
  reject((clone) => {
    record(clone.experimental_design, "design").baseline_cell = "A";
  }, /baseline_cell/);
});

test("rejects a wrong alternative cell", () => {
  reject((clone) => {
    record(clone.experimental_design, "design").alternative_cell = "G3";
  }, /alternative_cell/);
});

test("rejects a wrong effect sign", () => {
  reject((clone) => {
    clone.contrast_sign_convention = "G2_minus_G4";
  }, /contrast_sign_convention/);
});

test("rejects a malformed numeric field", () => {
  reject((clone) => {
    record(clone.primary_result, "primary").G2_p50_mean_across_blocks_microseconds = "3792.90625";
  }, /G2 P50/);
});

test("rejects a non-finite numeric field", () => {
  reject((clone) => {
    record(clone.primary_result, "primary").G4_p50_mean_across_blocks_microseconds = Number.NaN;
  }, /G4 P50/);
});

test("rejects an infinite numeric field", () => {
  reject((clone) => {
    record(clone.primary_result, "primary").mean_paired_effect_microseconds = Number.POSITIVE_INFINITY;
  }, /paired difference/);
});

test("rejects a wrong confidence interval bound", () => {
  reject((clone) => {
    record(
      record(clone.primary_result, "primary").confidence_interval_95_percent,
      "ci",
    ).high_microseconds = -2511.3;
  }, /CI high/);
});

test("rejects a wrong paired-block count", () => {
  reject((clone) => {
    record(clone.experimental_design, "design").counted_paired_blocks = 19;
  }, /paired blocks/);
});

test("rejects a wrong worker count", () => {
  reject((clone) => {
    record(clone.workload, "workload").worker_count = 2;
  }, /worker count/);
});

test("rejects a wrong evidence classification", () => {
  reject((clone) => {
    clone.evidence_class = "exploratory";
  }, /evidence_class/);
});

test("rejects flipped root claim flags", () => {
  reject((clone) => {
    clone.winner_declared = true;
  }, /winner_declared/);
});

test("rejects flipped nested claim flags", () => {
  reject((clone) => {
    record(clone.claim_policy, "claims").throughput_increase_claimed = true;
  }, /claim_policy.throughput_increase_claimed/);
});

test("rejects raw evidence marked public", () => {
  reject((clone) => {
    clone.raw_evidence_published_on_github = true;
  }, /raw evidence flag/);
});

test("rejects an unexpected related publication", () => {
  reject((clone) => {
    const related = clone.related_publications as Array<Record<string, unknown>>;
    const first = related[0];
    assert.ok(first);
    first.markdown = "unpublished-raw-runs.md";
  }, /related markdown/);
});

test("rejects related studies marked combinable", () => {
  reject((clone) => {
    const related = clone.related_publications as Array<Record<string, unknown>>;
    const first = related[0];
    assert.ok(first);
    first.combinable_into_paired_estimate = true;
  }, /combinable/);
});

test("rejects a missing required limitation", () => {
  reject((clone) => {
    const limitations = clone.limitations as string[];
    limitations.splice(7, 1);
  }, /limitations/);
});

test("builds public artifact URLs under the Pages base", () => {
  assert.equal(
    publicationUrl("/inferedge/", "data", "arrival-aware-sparse-gemm-results.json"),
    "/inferedge/data/arrival-aware-sparse-gemm-results.json",
  );
});
