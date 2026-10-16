import {
  ADAPTIVE_MIN_TARGET,
  CELL_IDS,
  EWMA_SHIFT,
  ILLUSTRATIVE_RUN_NS_PER_REQUEST,
  MAX_BATCH_SIZE,
  MAX_DELAY_MS,
  MAX_EVENTS,
  MAX_REQUESTS,
  MIN_BATCH_SIZE,
  MIN_DELAY_MS,
  NS_PER_MS,
  TRAFFIC_PATTERNS,
  arrivalIdleThresholdNs,
  minimumBatchSizeForCell,
  type CellId,
  type TrafficPattern,
} from "./cells.ts";

export type SealReason =
  | "unbatched"
  | "capacity"
  | "deadline"
  | "g3_minimum_target"
  | "g4_arrival_aware";

export type RequestStatus = "arrived" | "queued" | "sealed" | "executing" | "completed";

export type SimulationConfig = {
  cell: CellId;
  pattern: TrafficPattern;
  maxBatchSize: number;
  maxBatchDelayMs: number;
};

export type SimulationEvent = {
  seq: number;
  timeNs: number;
  kind:
    | "arrived"
    | "queued"
    | "batch_opened"
    | "sealed"
    | "executing"
    | "completed"
    | "batch_closed";
  requestId?: number;
  requestIds?: readonly number[];
  batchId?: number;
  sealReason?: SealReason;
  batchSize?: number;
};

export type RequestView = {
  id: number;
  status: RequestStatus;
  batchId: number | null;
};

export type FormingBatch = {
  batchId: number;
  requestIds: readonly number[];
  openedNs: number;
  deadlineNs: number;
};

export type SealRecord = {
  batchId: number;
  timeNs: number;
  requestIds: readonly number[];
  reason: SealReason;
};

export type SimulationResult = {
  config: SimulationConfig;
  arrivalsNs: readonly number[];
  events: readonly SimulationEvent[];
  seals: readonly SealRecord[];
};

export type SimulationView = {
  timeNs: number;
  eventIndex: number;
  requests: readonly RequestView[];
  forming: FormingBatch | null;
  seals: readonly SealRecord[];
  lastEvent: SimulationEvent | null;
};

export class SimulationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SimulationError";
  }
}

function requireFiniteInteger(value: number, label: string, min: number, max: number): number {
  if (typeof value !== "number" || !Number.isInteger(value) || !Number.isFinite(value)) {
    throw new SimulationError(`${label} must be a finite integer`);
  }
  if (value < min || value > max) {
    throw new SimulationError(`${label} is out of range`);
  }
  return value;
}

export function validateSimulationConfig(config: SimulationConfig): SimulationConfig {
  if (!CELL_IDS.includes(config.cell)) {
    throw new SimulationError("Unknown cell");
  }
  if (!TRAFFIC_PATTERNS.includes(config.pattern)) {
    throw new SimulationError("Unknown traffic pattern");
  }
  const maxBatchSize = requireFiniteInteger(config.maxBatchSize, "maxBatchSize", MIN_BATCH_SIZE, MAX_BATCH_SIZE);
  const maxBatchDelayMs = requireFiniteInteger(
    config.maxBatchDelayMs,
    "maxBatchDelayMs",
    MIN_DELAY_MS,
    MAX_DELAY_MS,
  );
  const requiredCapacity = minimumBatchSizeForCell(config.cell);
  if (maxBatchSize < requiredCapacity) {
    throw new SimulationError(
      `${config.cell} requires maxBatchSize >= ${requiredCapacity}`,
    );
  }
  return { cell: config.cell, pattern: config.pattern, maxBatchSize, maxBatchDelayMs };
}

export function arrivalScheduleNs(pattern: TrafficPattern): readonly number[] {
  if (pattern === "sparse") {
    return Array.from({ length: MAX_REQUESTS }, (_, index) => index * 8 * NS_PER_MS);
  }
  if (pattern === "steady") {
    return Array.from({ length: MAX_REQUESTS }, (_, index) => index * NS_PER_MS);
  }
  return [0, 400_000, 800_000, 1_200_000, 20 * NS_PER_MS, 20_400_000, 20_800_000, 21_200_000];
}

function updateEwma(
  lastEnqueueNs: number | null,
  ewmaNs: number | null,
  nowNs: number,
  idleThresholdNs: number,
): { lastEnqueueNs: number; ewmaNs: number | null } {
  if (lastEnqueueNs === null) {
    return { lastEnqueueNs: nowNs, ewmaNs: null };
  }
  const delta = nowNs - lastEnqueueNs;
  if (delta <= 0) {
    return { lastEnqueueNs: nowNs, ewmaNs };
  }
  if (delta > idleThresholdNs) {
    return { lastEnqueueNs: nowNs, ewmaNs: null };
  }
  if (ewmaNs === null) {
    return { lastEnqueueNs: nowNs, ewmaNs: delta };
  }
  return {
    lastEnqueueNs: nowNs,
    ewmaNs: ewmaNs - (ewmaNs >> EWMA_SHIFT) + (delta >> EWMA_SHIFT),
  };
}

type Request = {
  id: number;
  arrivedNs: number;
};

function assertSafeNumber(value: number, label: string): void {
  if (!Number.isFinite(value) || Number.isNaN(value)) {
    throw new SimulationError(`${label} is not a finite number`);
  }
  if (value < 0) {
    throw new SimulationError(`${label} is negative`);
  }
}

export function runSimulation(
  input: SimulationConfig,
  arrivalsOverride?: readonly number[],
): SimulationResult {
  const config = validateSimulationConfig(input);
  const arrivalsNs =
    arrivalsOverride === undefined ? arrivalScheduleNs(config.pattern) : [...arrivalsOverride];
  if (arrivalsNs.length === 0 || arrivalsNs.length > MAX_REQUESTS) {
    throw new SimulationError("Arrival schedule is out of bounds");
  }
  for (let index = 0; index < arrivalsNs.length; index += 1) {
    const time = arrivalsNs[index];
    if (time === undefined) {
      throw new SimulationError("Arrival timestamp is missing");
    }
    assertSafeNumber(time, "arrival time");
    if (index > 0) {
      const previous = arrivalsNs[index - 1];
      if (previous !== undefined && time < previous) {
        throw new SimulationError("Arrival times are not ordered");
      }
    }
  }

  const events: SimulationEvent[] = [];
  const seals: SealRecord[] = [];
  const queue: Request[] = [];
  let nextEvent = 0;
  let nextRequestId = 1;
  let nextBatchId = 1;
  let lastTime = 0;
  let arrivalCursor = 0;
  let lastEnqueueNs: number | null = null;
  let ewmaNs: number | null = null;
  const idleThresholdNs = arrivalIdleThresholdNs(config.maxBatchDelayMs);

  const pushEvent = (event: Omit<SimulationEvent, "seq">): void => {
    if (events.length >= MAX_EVENTS) {
      throw new SimulationError("Simulation exceeded the event bound");
    }
    assertSafeNumber(event.timeNs, "event time");
    if (event.timeNs < lastTime) {
      throw new SimulationError("Event order violation");
    }
    lastTime = event.timeNs;
    events.push({ ...event, seq: nextEvent });
    nextEvent += 1;
  };

  const enqueueArrivalsThrough = (timeNs: number): void => {
    while (arrivalCursor < arrivalsNs.length) {
      const arrivedNs = arrivalsNs[arrivalCursor];
      if (arrivedNs === undefined || arrivedNs > timeNs) {
        break;
      }
      const request = { id: nextRequestId, arrivedNs };
      nextRequestId += 1;
      arrivalCursor += 1;
      pushEvent({ timeNs: arrivedNs, kind: "arrived", requestId: request.id });
      queue.push(request);
      pushEvent({ timeNs: arrivedNs, kind: "queued", requestId: request.id });
      if (config.cell === "G4") {
        const updated = updateEwma(lastEnqueueNs, ewmaNs, arrivedNs, idleThresholdNs);
        lastEnqueueNs = updated.lastEnqueueNs;
        ewmaNs = updated.ewmaNs;
      }
    }
  };

  const sealBatch = (
    timeNs: number,
    owned: Request[],
    batchId: number,
    reason: SealReason,
  ): number => {
    if (owned.length === 0) {
      throw new SimulationError("Cannot seal an empty batch");
    }
    if (config.cell === "A" && owned.length !== 1) {
      throw new SimulationError("Cell A cannot form a multi-request batch");
    }
    if (config.cell === "A" && reason !== "unbatched") {
      throw new SimulationError("Cell A only seals as unbatched");
    }
    const requestIds = owned.map((request) => request.id);
    pushEvent({
      timeNs,
      kind: "sealed",
      batchId,
      requestIds,
      sealReason: reason,
      batchSize: owned.length,
    });
    seals.push({ batchId, timeNs, requestIds, reason });
    pushEvent({ timeNs, kind: "executing", batchId, requestIds });
    const doneNs = timeNs + owned.length * ILLUSTRATIVE_RUN_NS_PER_REQUEST;
    assertSafeNumber(doneNs, "completion time");
    enqueueArrivalsThrough(doneNs);
    pushEvent({ timeNs: doneNs, kind: "completed", batchId, requestIds });
    pushEvent({
      timeNs: doneNs,
      kind: "batch_closed",
      batchId,
      requestIds,
      sealReason: reason,
      batchSize: owned.length,
    });
    return doneNs;
  };

  const decideSeal = (
    timeNs: number,
    owned: Request[],
    deadlineNs: number,
  ): SealReason | null => {
    if (config.cell === "A") {
      return "unbatched";
    }
    if (owned.length >= config.maxBatchSize) {
      return "capacity";
    }
    if (config.cell === "G3" && owned.length >= ADAPTIVE_MIN_TARGET && queue.length === 0) {
      return "g3_minimum_target";
    }
    if (config.cell === "G4" && queue.length === 0 && ewmaNs !== null) {
      const remaining = deadlineNs - timeNs;
      if (remaining <= 0) {
        return "deadline";
      }
      if (ewmaNs > remaining) {
        return "g4_arrival_aware";
      }
    }
    if (timeNs >= deadlineNs) {
      return "deadline";
    }
    return null;
  };

  let nowNs = 0;
  let steps = 0;
  while (arrivalCursor < arrivalsNs.length || queue.length > 0) {
    steps += 1;
    if (steps > MAX_EVENTS) {
      throw new SimulationError("Simulation exceeded the step bound");
    }
    enqueueArrivalsThrough(nowNs);
    if (queue.length === 0) {
      const nextArrival = arrivalsNs[arrivalCursor];
      if (nextArrival === undefined) {
        break;
      }
      nowNs = nextArrival;
      continue;
    }

    const first = queue.shift();
    if (first === undefined) {
      throw new SimulationError("Queue underflow");
    }
    const owned = [first];
    const batchId = nextBatchId;
    nextBatchId += 1;
    const openedNs = Math.max(nowNs, first.arrivedNs);
    nowNs = openedNs;
    const deadlineNs =
      config.cell === "A" ? openedNs : openedNs + config.maxBatchDelayMs * NS_PER_MS;
    pushEvent({
      timeNs: openedNs,
      kind: "batch_opened",
      batchId,
      requestIds: [first.id],
    });

    while (config.cell !== "A" && owned.length < config.maxBatchSize) {
      enqueueArrivalsThrough(nowNs);
      while (owned.length < config.maxBatchSize && queue.length > 0) {
        const next = queue.shift();
        if (next === undefined) {
          throw new SimulationError("Queue underflow");
        }
        owned.push(next);
      }
      const reason = decideSeal(nowNs, owned, deadlineNs);
      if (reason !== null) {
        nowNs = sealBatch(nowNs, owned, batchId, reason);
        break;
      }
      const nextArrival = arrivalsNs[arrivalCursor];
      const wakeNs =
        nextArrival !== undefined && nextArrival < deadlineNs ? nextArrival : deadlineNs;
      if (wakeNs < nowNs) {
        throw new SimulationError("Negative waiting interval");
      }
      nowNs = wakeNs;
      if (nowNs >= deadlineNs && decideSeal(nowNs, owned, deadlineNs) === null) {
        nowNs = sealBatch(nowNs, owned, batchId, "deadline");
        break;
      }
    }
    if (owned.length > 0 && events.every((event) => event.batchId !== batchId || event.kind !== "sealed")) {
      const reason = decideSeal(nowNs, owned, deadlineNs) ?? (config.cell === "A" ? "unbatched" : "capacity");
      nowNs = sealBatch(nowNs, owned, batchId, reason);
    }
  }

  const seen = new Set<number>();
  for (const event of events) {
    if (event.kind === "arrived" && event.requestId !== undefined) {
      if (seen.has(event.requestId)) {
        throw new SimulationError("Duplicate request id");
      }
      seen.add(event.requestId);
    }
  }
  for (const id of seen) {
    const completed = events.some(
      (event) => event.kind === "completed" && event.requestIds?.includes(id) === true,
    );
    if (!completed) {
      throw new SimulationError(`Request ${id} has no terminal completion`);
    }
  }

  return { config, arrivalsNs, events, seals };
}

export function viewAtEvent(result: SimulationResult, eventIndex: number): SimulationView {
  const clamped = Math.min(Math.max(eventIndex, 0), result.events.length);
  const requests = new Map<number, RequestView>();
  let forming: FormingBatch | null = null;
  const seals: SealRecord[] = [];
  let timeNs = 0;
  let lastEvent: SimulationEvent | null = null;

  for (let index = 0; index < clamped; index += 1) {
    const event = result.events[index];
    if (event === undefined) {
      break;
    }
    lastEvent = event;
    timeNs = event.timeNs;
    const touch = (id: number, status: RequestStatus, batchId: number | null): void => {
      requests.set(id, { id, status, batchId });
    };
    if (event.kind === "arrived" && event.requestId !== undefined) {
      touch(event.requestId, "arrived", null);
    }
    if (event.kind === "queued" && event.requestId !== undefined) {
      touch(event.requestId, "queued", null);
    }
    if (event.kind === "batch_opened" && event.batchId !== undefined && event.requestIds !== undefined) {
      forming = {
        batchId: event.batchId,
        requestIds: event.requestIds,
        openedNs: event.timeNs,
        deadlineNs:
          result.config.cell === "A"
            ? event.timeNs
            : event.timeNs + result.config.maxBatchDelayMs * NS_PER_MS,
      };
      for (const id of event.requestIds) {
        touch(id, "queued", event.batchId);
      }
    }
    if (event.kind === "sealed" && event.batchId !== undefined && event.requestIds !== undefined) {
      if (forming !== null && forming.batchId === event.batchId) {
        forming = { ...forming, requestIds: event.requestIds };
      }
      for (const id of event.requestIds) {
        touch(id, "sealed", event.batchId);
      }
      if (event.sealReason !== undefined) {
        seals.push({
          batchId: event.batchId,
          timeNs: event.timeNs,
          requestIds: event.requestIds,
          reason: event.sealReason,
        });
      }
      forming = null;
    }
    if (event.kind === "executing" && event.requestIds !== undefined) {
      for (const id of event.requestIds) {
        touch(id, "executing", event.batchId ?? null);
      }
    }
    if (event.kind === "completed" && event.requestIds !== undefined) {
      for (const id of event.requestIds) {
        touch(id, "completed", event.batchId ?? null);
      }
    }
  }

  if (forming !== null) {
    const owned = [...forming.requestIds];
    for (const request of requests.values()) {
      if (request.status === "queued" && !owned.includes(request.id)) {
        owned.push(request.id);
      }
    }
    forming = { ...forming, requestIds: owned };
  }

  return {
    timeNs,
    eventIndex: clamped,
    requests: [...requests.values()].sort((left, right) => left.id - right.id),
    forming,
    seals,
    lastEvent,
  };
}

export function formatSimTime(timeNs: number): string {
  return `${(timeNs / NS_PER_MS).toFixed(1)} ms`;
}

export function sealReasonLabel(reason: SealReason): string {
  switch (reason) {
    case "unbatched":
      return "Unbatched execute";
    case "capacity":
      return "Capacity reached";
    case "deadline":
      return "Deadline expired";
    case "g3_minimum_target":
      return "G3 minimum target";
    case "g4_arrival_aware":
      return "G4 arrival-aware (approx.)";
  }
}
