import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import {
  EXPECTED_PUBLICATION_STATEMENT,
  GENERATIVE_PUBLICATION_TYPE,
  formatMillisecondInterval,
  parseGenerativeOverheadPublication,
} from "../src/performance/generativePublication.ts";

const websiteRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const published = JSON.parse(
  readFileSync(join(websiteRoot, "../docs/benchmarks/generative-overhead-qwen3-0.6b-results.json"), "utf8"),
) as unknown;

function clonePublished(): Record<string, unknown> {
  return structuredClone(published) as Record<string, unknown>;
}

function record(value: unknown): Record<string, unknown> {
  assert.equal(typeof value, "object");
  assert.ok(value !== null);
  assert.ok(!Array.isArray(value));
  return value as Record<string, unknown>;
}

function reject(mutator: (clone: Record<string, unknown>) => void, pattern: RegExp): void {
  const clone = clonePublished();
  mutator(clone);
  assert.throws(() => parseGenerativeOverheadPublication(clone), pattern);
}

test("parses the accepted generative-overhead publication", () => {
  const result = parseGenerativeOverheadPublication(published);
  assert.equal(result.publicationType, GENERATIVE_PUBLICATION_TYPE);
  assert.equal(result.studyId, "generative_overhead_qwen3_0_6b_c01");
  assert.equal(result.pairCount, 20);
  assert.equal(result.inferedgeMeanNs, 74514797.85);
  assert.equal(result.referenceMeanNs, 74688422.9);
  assert.equal(result.meanDifferenceNs, -173625.05);
  assert.equal(result.ciLowNs, -1060655.6224999994);
  assert.equal(result.ciHighNs, 730089.1224999995);
  assert.equal(result.signedDirectionLabel, "inferedge_faster");
  assert.equal(result.signedDirectionIsWinner, false);
  assert.equal(result.winnerDeclared, false);
  assert.equal(result.superiorityClaimed, false);
  assert.equal(result.practicalEquivalenceClaimed, false);
  assert.equal(result.claimStatus, "no_winner_below_threshold_or_ci");
  assert.equal(result.rawEvidencePublished, false);
  assert.equal(result.publicationStatement, EXPECTED_PUBLICATION_STATEMENT);
  assert.equal(
    formatMillisecondInterval(result.ciLowNs, result.ciHighNs),
    "[-1.0606556224999994, 0.7300891224999995]",
  );
});

test("rejects a wrong study id", () => {
  reject((clone) => {
    clone.study_id = "other";
  }, /study id/);
});

test("rejects a wrong evidence class", () => {
  reject((clone) => {
    clone.evidence_class = "confirmatory";
  }, /evidence class/);
});

test("rejects a wrong campaign class", () => {
  reject((clone) => {
    record(clone.campaign).campaign_class = "dry_run";
  }, /campaign class/);
});

test("rejects a wrong campaign id", () => {
  reject((clone) => {
    record(clone.campaign).campaign_id = "phase-9-b3d-campaign-002";
  }, /campaign id/);
});

test("rejects a wrong source commit", () => {
  reject((clone) => {
    record(clone.campaign).source_commit = "0000000000000000000000000000000000000000";
  }, /source commit/);
});

test("rejects a wrong manifest hash", () => {
  reject((clone) => {
    record(clone.provenance).campaign_manifest_sha256 = "0".repeat(64);
  }, /manifest hash/);
});

test("rejects a wrong aggregate hash", () => {
  reject((clone) => {
    record(clone.provenance).aggregate_sha256 = "1".repeat(64);
  }, /aggregate hash/);
});

test("rejects a wrong summary hash", () => {
  reject((clone) => {
    record(clone.provenance).summary_sha256 = "2".repeat(64);
  }, /summary hash/);
});

test("rejects raw evidence marked public", () => {
  reject((clone) => {
    clone.raw_evidence_published_on_github = true;
  }, /raw evidence flag/);
});

test("rejects an incomplete matrix", () => {
  reject((clone) => {
    record(clone.campaign).complete_matrix = false;
  }, /complete matrix/);
});

test("rejects nonempty rejection codes", () => {
  reject((clone) => {
    record(clone.campaign).rejection_codes = ["sample_count_invalid"];
  }, /rejection codes/);
});

test("rejects a winner flag", () => {
  reject((clone) => {
    record(clone.primary_c01).winner_declared = true;
    record(clone.claim_policy).winner_declared = true;
  }, /winner/);
});

test("rejects a superiority flag", () => {
  reject((clone) => {
    record(clone.primary_c01).superiority_claimed = true;
    record(clone.claim_policy).superiority_claimed = true;
  }, /superiority/);
});

test("rejects a practical-equivalence flag", () => {
  reject((clone) => {
    record(clone.primary_c01).practical_equivalence_claimed = true;
    record(clone.claim_policy).practical_equivalence_claimed = true;
  }, /practical equivalence/);
});

test("rejects a wrong claim status", () => {
  reject((clone) => {
    record(clone.primary_c01).claim_status = "inferedge_faster";
    record(clone.claim_policy).claim_status = "inferedge_faster";
  }, /claim status/);
});

test("rejects a signed direction marked as a winner", () => {
  reject((clone) => {
    record(clone.primary_c01).signed_direction_is_winner = true;
  }, /signed direction winner flag/);
});

test("rejects an absolute threshold marked met", () => {
  reject((clone) => {
    record(clone.primary_c01).absolute_threshold_met = true;
  }, /absolute threshold/);
});

test("rejects a relative threshold marked met", () => {
  reject((clone) => {
    record(clone.primary_c01).relative_threshold_met = true;
  }, /relative threshold/);
});

test("rejects strict CI exclusion marked true", () => {
  reject((clone) => {
    record(clone.primary_c01).ci_excludes_zero_strict = true;
  }, /strict CI exclusion/);
});

test("rejects secondary rescue of C01", () => {
  reject((clone) => {
    record(clone.secondary_cells_policy).may_rescue_or_override_c01 = true;
  }, /secondary rescue/);
});

test("rejects a published secondary ranking", () => {
  reject((clone) => {
    record(clone.secondary_cells_policy).ranking_published = true;
  }, /secondary ranking/);
});

test("rejects a wrong pair count", () => {
  reject((clone) => {
    record(clone.primary_c01).pair_count = 19;
  }, /pair count/);
});

test("rejects a missing pair", () => {
  reject((clone) => {
    const pairs = record(clone.primary_c01).pairs;
    assert.ok(Array.isArray(pairs));
    pairs.pop();
  }, /pair count/);
});

test("rejects a duplicate pair id", () => {
  reject((clone) => {
    const pairs = record(clone.primary_c01).pairs;
    assert.ok(Array.isArray(pairs));
    record(pairs[1]).pair_id = "00";
  }, /duplicate pair id/);
});

test("rejects a wrong pair order", () => {
  reject((clone) => {
    const pairs = record(clone.primary_c01).pairs;
    assert.ok(Array.isArray(pairs));
    record(pairs[0]).arm_order = "ref_then_ie";
  }, /arm_order/);
});

test("rejects an altered duration", () => {
  reject((clone) => {
    const pairs = record(clone.primary_c01).pairs;
    assert.ok(Array.isArray(pairs));
    record(pairs[0]).inferedge_duration_ns = 1;
  }, /InferEdge duration/);
});

test("rejects an altered difference", () => {
  reject((clone) => {
    const pairs = record(clone.primary_c01).pairs;
    assert.ok(Array.isArray(pairs));
    record(pairs[0]).difference_ns = 1;
  }, /difference/);
});

test("rejects an altered sum", () => {
  reject((clone) => {
    record(clone.primary_c01).sum_d_ns = 0;
  }, /paired sum/);
});

test("rejects an altered mean", () => {
  reject((clone) => {
    record(clone.primary_c01).inferedge_mean_ns = 1;
  }, /InferEdge mean/);
});

test("rejects an altered confidence interval", () => {
  reject((clone) => {
    record(clone.primary_c01).ci_low_ns = 0;
  }, /CI low/);
});

test("rejects an altered descriptive statistic", () => {
  reject((clone) => {
    record(record(clone.primary_c01).descriptive_statistics).p50_difference_ns = 0;
  }, /P50/);
});

test("rejects a wrong bootstrap seed", () => {
  reject((clone) => {
    record(clone.primary_c01).bootstrap_seed = 1;
    record(clone.claim_policy).bootstrap_seed = 1;
  }, /bootstrap seed/);
});

test("rejects a wrong bootstrap replicate count", () => {
  reject((clone) => {
    record(clone.primary_c01).bootstrap_replicates = 100;
    record(clone.claim_policy).bootstrap_replicates = 100;
  }, /bootstrap replicates/);
});

test("rejects a wrong confidence level", () => {
  reject((clone) => {
    record(clone.primary_c01).confidence_level = 0.9;
    record(clone.claim_policy).confidence_level = 0.9;
  }, /confidence level/);
});

test("rejects an altered publication statement", () => {
  reject((clone) => {
    clone.publication_statement = "InferEdge is faster.";
  }, /publication statement/);
});

test("rejects a non-finite statistic", () => {
  reject((clone) => {
    record(clone.primary_c01).ci_high_ns = Number.NaN;
  }, /non-finite/);
});

test("rejects an infinite statistic", () => {
  reject((clone) => {
    record(clone.primary_c01).mean_difference_ns = Number.POSITIVE_INFINITY;
  }, /non-finite/);
});

test("rejects a changed raw-evidence limitation", () => {
  reject((clone) => {
    const limitations = clone.limitations;
    assert.ok(Array.isArray(limitations));
    limitations[limitations.length - 1] = "raw_evidence_published";
  }, /limitations/);
});
