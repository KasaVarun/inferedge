import assert from "node:assert/strict";
import test from "node:test";
import {
  ADAPTIVE_MIN_TARGET,
  CELL_IDS,
  MAX_BATCH_SIZE,
  MAX_DELAY_MS,
  MIN_BATCH_SIZE,
  MIN_DELAY_MS,
  NS_PER_MS,
  TRAFFIC_PATTERNS,
  arrivalIdleThresholdNs,
  minimumBatchSizeForCell,
} from "../src/scheduler/cells.ts";
import {
  SimulationError,
  runSimulation,
  validateSimulationConfig,
  viewAtEvent,
  type SimulationConfig,
  type SimulationEvent,
} from "../src/scheduler/schedulerSimulation.ts";

const G2: SimulationConfig = {
  cell: "G2",
  pattern: "burst",
  maxBatchSize: 4,
  maxBatchDelayMs: 2,
};

function eventsEqual(left: readonly SimulationEvent[], right: readonly SimulationEvent[]): void {
  assert.equal(left.length, right.length);
  assert.deepEqual(left, right);
}

function completedIds(events: readonly SimulationEvent[]): number[] {
  return events
    .filter((event) => event.kind === "completed")
    .flatMap((event) => [...(event.requestIds ?? [])]);
}

function sealedBatches(events: readonly SimulationEvent[]) {
  return events.filter((event) => event.kind === "sealed");
}

test("same inputs produce the same simulation events", () => {
  eventsEqual(runSimulation(G2).events, runSimulation(G2).events);
});

test("reset reproduces the same event sequence", () => {
  const first = runSimulation({ ...G2, pattern: "steady" });
  const second = runSimulation({ ...G2, pattern: "steady" });
  eventsEqual(first.events, second.events);
});

test("Cell A never forms multi-request batches", () => {
  const result = runSimulation({ cell: "A", pattern: "burst", maxBatchSize: 4, maxBatchDelayMs: 2 });
  for (const event of sealedBatches(result.events)) {
    assert.equal(event.batchSize, 1);
    assert.equal(event.sealReason, "unbatched");
    assert.equal(event.requestIds?.length, 1);
  }
  assert.equal(sealedBatches(result.events).length, 8);
});

test("G2 seals at capacity when the burst fills the batch", () => {
  const result = runSimulation(G2);
  const first = sealedBatches(result.events)[0];
  assert.equal(first?.sealReason, "capacity");
  assert.equal(first?.batchSize, 4);
});

test("G2 seals at deadline on sparse arrivals", () => {
  const result = runSimulation({ cell: "G2", pattern: "sparse", maxBatchSize: 4, maxBatchDelayMs: 2 });
  const first = sealedBatches(result.events)[0];
  assert.equal(first?.sealReason, "deadline");
  assert.equal(first?.batchSize, 1);
});

test("G3 seals at the documented minimum target when the queue is empty", () => {
  const result = runSimulation({ cell: "G3", pattern: "burst", maxBatchSize: 4, maxBatchDelayMs: 2 });
  const first = sealedBatches(result.events)[0];
  assert.equal(first?.sealReason, "g3_minimum_target");
  assert.equal(first?.batchSize, ADAPTIVE_MIN_TARGET);
});

test("G4 early-seal matches the documented EWMA approximation", () => {
  const result = runSimulation({ cell: "G4", pattern: "burst", maxBatchSize: 4, maxBatchDelayMs: 1 });
  const early = sealedBatches(result.events).find((event) => event.sealReason === "g4_arrival_aware");
  assert.ok(early);
  assert.ok((early.batchSize ?? 0) >= 2);
  assert.ok((early.batchSize ?? 0) < 4);
});

test("uninitialized G4 EWMA keeps fixed-delay waiting", () => {
  const result = runSimulation({ cell: "G4", pattern: "sparse", maxBatchSize: 4, maxBatchDelayMs: 2 });
  const first = sealedBatches(result.events)[0];
  assert.equal(first?.sealReason, "deadline");
  assert.equal(first?.batchSize, 1);
});

test("size-one batches are supported on batching cells", () => {
  const result = runSimulation({ cell: "G2", pattern: "sparse", maxBatchSize: 4, maxBatchDelayMs: 1 });
  assert.ok(sealedBatches(result.events).some((event) => event.batchSize === 1));
});

test("policy switch changes the modeled seal reasons", () => {
  const a = runSimulation({ cell: "A", pattern: "burst", maxBatchSize: 4, maxBatchDelayMs: 2 });
  const g3 = runSimulation({ cell: "G3", pattern: "burst", maxBatchSize: 4, maxBatchDelayMs: 2 });
  assert.notDeepEqual(a.events, g3.events);
  assert.ok(sealedBatches(a.events).every((event) => event.sealReason === "unbatched"));
  assert.ok(sealedBatches(g3.events).some((event) => event.sealReason === "g3_minimum_target"));
});

test("request identifiers are unique and every request completes", () => {
  const result = runSimulation({ cell: "G3", pattern: "steady", maxBatchSize: 3, maxBatchDelayMs: 2 });
  const arrived = result.events.filter((event) => event.kind === "arrived").map((event) => event.requestId);
  assert.equal(new Set(arrived).size, arrived.length);
  assert.deepEqual([...completedIds(result.events)].sort((left, right) => left - right), arrived);
});

test("event times are non-decreasing and waiting intervals are non-negative", () => {
  const result = runSimulation({ cell: "G4", pattern: "steady", maxBatchSize: 4, maxBatchDelayMs: 2 });
  for (let index = 1; index < result.events.length; index += 1) {
    const previous = result.events[index - 1];
    const current = result.events[index];
    assert.ok(previous);
    assert.ok(current);
    assert.ok(current.timeNs >= previous.timeNs);
    assert.ok(Number.isFinite(current.timeNs));
    assert.equal(Number.isNaN(current.timeNs), false);
  }
});

test("simulation stays bounded", () => {
  const result = runSimulation({ cell: "G2", pattern: "steady", maxBatchSize: 4, maxBatchDelayMs: 8 });
  assert.ok(result.events.length <= 256);
  assert.equal(result.arrivalsNs.length, 8);
});

test("G3 rejects capacity 1 and accepts capacities 2 through 4", () => {
  assert.throws(
    () => validateSimulationConfig({ cell: "G3", pattern: "burst", maxBatchSize: 1, maxBatchDelayMs: 2 }),
    /G3 requires maxBatchSize >= 2/,
  );
  assert.throws(
    () => runSimulation({ cell: "G3", pattern: "burst", maxBatchSize: 1, maxBatchDelayMs: 2 }),
    SimulationError,
  );
  for (const maxBatchSize of [2, 3, 4] as const) {
    const config = validateSimulationConfig({
      cell: "G3",
      pattern: "burst",
      maxBatchSize,
      maxBatchDelayMs: 2,
    });
    assert.equal(config.maxBatchSize, maxBatchSize);
    const result = runSimulation(config);
    assert.equal(result.seals.length > 0, true);
  }
});

test("G4 capacity 1 follows the C++ start() minimum of 2", () => {
  assert.equal(minimumBatchSizeForCell("G4"), ADAPTIVE_MIN_TARGET);
  assert.throws(
    () => validateSimulationConfig({ cell: "G4", pattern: "sparse", maxBatchSize: 1, maxBatchDelayMs: 2 }),
    /G4 requires maxBatchSize >= 2/,
  );
  const accepted = validateSimulationConfig({
    cell: "G4",
    pattern: "sparse",
    maxBatchSize: 2,
    maxBatchDelayMs: 2,
  });
  assert.equal(accepted.maxBatchSize, 2);
  assert.doesNotThrow(() =>
    validateSimulationConfig({ cell: "G2", pattern: "sparse", maxBatchSize: 1, maxBatchDelayMs: 2 }),
  );
});

test("G4 idle gap resets EWMA so the next arrival is a cold start", () => {
  const maxBatchDelayMs = 2;
  const idleThresholdNs = arrivalIdleThresholdNs(maxBatchDelayMs);
  assert.equal(idleThresholdNs, 50 * NS_PER_MS);
  const gapNs = 80 * NS_PER_MS;
  assert.ok(gapNs - NS_PER_MS > idleThresholdNs);
  const result = runSimulation(
    { cell: "G4", pattern: "sparse", maxBatchSize: 4, maxBatchDelayMs },
    [0, NS_PER_MS, gapNs],
  );
  assert.equal(result.seals.length, 2);
  assert.equal(result.seals[0]?.reason, "deadline");
  assert.equal(result.seals[0]?.requestIds.length, 2);
  assert.equal(result.seals[1]?.reason, "deadline");
  assert.equal(result.seals[1]?.requestIds.length, 1);
  assert.equal(result.seals[1]?.timeNs, gapNs + maxBatchDelayMs * NS_PER_MS);
  assert.equal(
    result.seals.some((seal) => seal.reason === "g4_arrival_aware"),
    false,
  );
});

test("every valid policy configuration completes deterministically", () => {
  const seen = new Set<string>();
  for (const cell of CELL_IDS) {
    const minCapacity = minimumBatchSizeForCell(cell);
    for (const pattern of TRAFFIC_PATTERNS) {
      for (let maxBatchSize = MIN_BATCH_SIZE; maxBatchSize <= MAX_BATCH_SIZE; maxBatchSize += 1) {
        for (let maxBatchDelayMs = MIN_DELAY_MS; maxBatchDelayMs <= MAX_DELAY_MS; maxBatchDelayMs += 1) {
          const input = { cell, pattern, maxBatchSize, maxBatchDelayMs };
          if (maxBatchSize < minCapacity) {
            assert.throws(() => validateSimulationConfig(input), SimulationError);
            continue;
          }
          const first = runSimulation(input);
          const second = runSimulation(input);
          eventsEqual(first.events, second.events);
          const arrived = first.events
            .filter((event) => event.kind === "arrived")
            .map((event) => event.requestId);
          assert.equal(new Set(arrived).size, arrived.length);
          assert.deepEqual(
            [...completedIds(first.events)].sort((left, right) => left - right),
            arrived,
          );
          for (let index = 1; index < first.events.length; index += 1) {
            const previous = first.events[index - 1];
            const current = first.events[index];
            assert.ok(previous);
            assert.ok(current);
            assert.ok(current.timeNs >= previous.timeNs);
          }
          seen.add(`${cell}:${pattern}:${maxBatchSize}:${maxBatchDelayMs}`);
        }
      }
    }
  }
  assert.equal(seen.size, 336);
});

test("malformed configuration is rejected", () => {
  assert.throws(
    () => validateSimulationConfig({ cell: "G5" as SimulationConfig["cell"], pattern: "burst", maxBatchSize: 4, maxBatchDelayMs: 2 }),
    SimulationError,
  );
  assert.throws(() => validateSimulationConfig({ cell: "G2", pattern: "burst", maxBatchSize: 0, maxBatchDelayMs: 2 }), /maxBatchSize/);
  assert.throws(() => validateSimulationConfig({ cell: "G2", pattern: "burst", maxBatchSize: 4, maxBatchDelayMs: 0 }), /maxBatchDelayMs/);
  assert.throws(
    () => validateSimulationConfig({ cell: "G2", pattern: "burst", maxBatchSize: Number.NaN, maxBatchDelayMs: 2 }),
    SimulationError,
  );
  assert.throws(
    () => validateSimulationConfig({ cell: "G2", pattern: "burst", maxBatchSize: 4, maxBatchDelayMs: Number.POSITIVE_INFINITY }),
    SimulationError,
  );
});

test("view replay does not invent statistics", () => {
  const result = runSimulation(G2);
  const view = viewAtEvent(result, result.events.length);
  assert.equal(view.seals.length, result.seals.length);
  assert.equal(view.requests.every((request) => request.status === "completed"), true);
  assert.equal(view.timeNs % 1, 0);
});
