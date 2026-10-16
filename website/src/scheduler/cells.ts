export const CELL_IDS = ["A", "G2", "G3", "G4"] as const;
export type CellId = (typeof CELL_IDS)[number];

export const TRAFFIC_PATTERNS = ["sparse", "burst", "steady"] as const;
export type TrafficPattern = (typeof TRAFFIC_PATTERNS)[number];

export const ADAPTIVE_MIN_TARGET = 2;
export const MIN_BATCH_SIZE = 1;
export const MAX_BATCH_SIZE = 4;
export const MIN_DELAY_MS = 1;
export const MAX_DELAY_MS = 8;
export const MAX_REQUESTS = 8;
export const MAX_EVENTS = 256;
export const NS_PER_MS = 1_000_000;
export const ILLUSTRATIVE_RUN_NS_PER_REQUEST = 3 * NS_PER_MS;
export const EWMA_SHIFT = 3;
export const IDLE_THRESHOLD_FLOOR_NS = 50 * NS_PER_MS;

export const CELL_SUMMARIES: Record<
  CellId,
  { title: string; modeled: string; simplified: string }
> = {
  A: {
    title: "Unbatched",
    modeled: "Each request is executed alone. The explorer never groups Cell A work.",
    simplified: "Admission rejection, shutdown, and multi-worker interleaving are not modeled.",
  },
  G2: {
    title: "Fixed-delay batching",
    modeled:
      "A forming batch seals at configured capacity or when the open-batch deadline expires.",
    simplified: "One worker and an unbounded educational queue. No mutex, wait, or closed-queue seal.",
  },
  G3: {
    title: "Adaptive minimum-target batching",
    modeled:
      "When the queue is empty and the worker already owns at least two validated requests, the batch seals without waiting out the remaining delay. This matches kAdaptiveMinTargetBatch = 2. G3 does not estimate arrival rate.",
    simplified: "Invalid requests, drain/cancel shutdown, and closed-queue seals are omitted.",
  },
  G4: {
    title: "Arrival-aware batching (educational approximation)",
    modeled:
      "Accepted-enqueue interarrival uses the documented integer EWMA (alpha 1/8). After an idle gap larger than max(50 ms, 8 × delay) the estimate resets. On an empty queue, if the estimate is initialized and exceeds remaining deadline nanoseconds, the batch seals early.",
    simplified:
      "This is a discrete-time approximation of the C++ decision points, not a bit-identical EWMA or mutex-ordered port. Uninitialized EWMA keeps fixed-delay waiting.",
  },
};

export function isCellId(value: string): value is CellId {
  return (CELL_IDS as readonly string[]).includes(value);
}

export function isTrafficPattern(value: string): value is TrafficPattern {
  return (TRAFFIC_PATTERNS as readonly string[]).includes(value);
}

export function arrivalIdleThresholdNs(maxBatchDelayMs: number): number {
  const scaled = maxBatchDelayMs * 8 * NS_PER_MS;
  return scaled > IDLE_THRESHOLD_FLOOR_NS ? scaled : IDLE_THRESHOLD_FLOOR_NS;
}

export function minimumBatchSizeForCell(cell: CellId): number {
  if (cell === "G3" || cell === "G4") {
    return ADAPTIVE_MIN_TARGET;
  }
  return MIN_BATCH_SIZE;
}
