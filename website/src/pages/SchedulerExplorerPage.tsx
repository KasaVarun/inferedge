import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { PageLayout } from "../components/PageLayout";
import {
  CELL_IDS,
  CELL_SUMMARIES,
  MAX_BATCH_SIZE,
  MAX_DELAY_MS,
  MIN_DELAY_MS,
  TRAFFIC_PATTERNS,
  minimumBatchSizeForCell,
  type CellId,
  type TrafficPattern,
} from "../scheduler/cells";
import {
  formatSimTime,
  runSimulation,
  sealReasonLabel,
  viewAtEvent,
  type SimulationConfig,
} from "../scheduler/schedulerSimulation";
import styles from "./SchedulerExplorerPage.module.css";

const TRAFFIC_LABELS: Record<TrafficPattern, string> = {
  sparse: "Sparse — 8 ms gaps",
  burst: "Burst — tight pairs",
  steady: "Steady — 1 ms gaps",
};

function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export function SchedulerExplorerPage() {
  const [cell, setCell] = useState<CellId>("G2");
  const [pattern, setPattern] = useState<TrafficPattern>("burst");
  const [maxBatchSize, setMaxBatchSize] = useState(4);
  const [maxBatchDelayMs, setMaxBatchDelayMs] = useState(2);
  const [eventIndex, setEventIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  const capacityMin = minimumBatchSizeForCell(cell);
  const effectiveCapacity = Math.max(maxBatchSize, capacityMin);
  const config = useMemo<SimulationConfig>(
    () => ({ cell, pattern, maxBatchSize: effectiveCapacity, maxBatchDelayMs }),
    [cell, pattern, effectiveCapacity, maxBatchDelayMs],
  );
  const result = useMemo(() => runSimulation(config), [config]);
  const view = useMemo(() => viewAtEvent(result, eventIndex), [result, eventIndex]);
  const summary = CELL_SUMMARIES[cell];
  const batching = cell !== "A";

  function selectCell(next: CellId): void {
    setCell(next);
    if (maxBatchSize < minimumBatchSizeForCell(next)) {
      setMaxBatchSize(minimumBatchSizeForCell(next));
    }
  }

  useEffect(() => {
    setEventIndex(0);
    setPlaying(false);
  }, [config]);

  useEffect(() => {
    if (!playing) {
      return;
    }
    if (prefersReducedMotion()) {
      setEventIndex(result.events.length);
      setPlaying(false);
      return;
    }
    const intervalMs = speed === 2 ? 140 : speed === 4 ? 70 : 280;
    const timer = window.setInterval(() => {
      setEventIndex((current) => {
        if (current >= result.events.length) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [playing, result.events.length, speed]);

  return (
    <PageLayout kicker="Scheduler Explorer" title="Illustrative scheduling cells">
      <p className={styles.disclaimer} role="note">
        Illustrative simulation — not measured InferEdge latency. This page does not run the C++
        scheduler, does not execute ONNX inference, and does not replay the audited GEMM extract.
      </p>
      <p className={styles.lede}>
        Compare unbatched Cell A with G2 fixed-delay, G3 adaptive, and G4 arrival-aware batching
        using a deterministic, bounded event model. For measured producer-observed P50, open the{" "}
        <Link to="/performance">Performance page</Link>.
      </p>

      <section className={`${styles.section} ${styles.controlsAnchor}`} aria-labelledby="scheduler-controls">
        <h2 id="scheduler-controls">Configuration</h2>
        <fieldset className={styles.fieldset}>
          <legend>Cell</legend>
          <div className={styles.pills}>
            {CELL_IDS.map((id) => (
              <label key={id} className={styles.pill}>
                <input
                  type="radio"
                  name="cell"
                  value={id}
                  checked={cell === id}
                  onChange={() => selectCell(id)}
                />
                <span>
                  {id} · {CELL_SUMMARIES[id].title}
                </span>
              </label>
            ))}
          </div>
        </fieldset>
        <div className={styles.controls}>
          <label>
            Traffic pattern
            <select
              value={pattern}
              onChange={(event) => setPattern(event.target.value as TrafficPattern)}
            >
              {TRAFFIC_PATTERNS.map((id) => (
                <option key={id} value={id}>
                  {TRAFFIC_LABELS[id]}
                </option>
              ))}
            </select>
          </label>
          <label>
            Batch capacity
            <input
              type="range"
              min={capacityMin}
              max={MAX_BATCH_SIZE}
              value={effectiveCapacity}
              disabled={!batching}
              onChange={(event) => setMaxBatchSize(Number(event.target.value))}
            />
            <span>
              {batching
                ? `${effectiveCapacity}${cell === "G3" || cell === "G4" ? " (minimum 2)" : ""}`
                : "unused on Cell A"}
            </span>
          </label>
          <label>
            Maximum batch delay
            <input
              type="range"
              min={MIN_DELAY_MS}
              max={MAX_DELAY_MS}
              value={maxBatchDelayMs}
              disabled={!batching}
              onChange={(event) => setMaxBatchDelayMs(Number(event.target.value))}
            />
            <span>{batching ? `${maxBatchDelayMs} ms` : "unused on Cell A"}</span>
          </label>
        </div>
        <div className={styles.transport} role="group" aria-label="Simulation transport">
          <button type="button" className={styles.primary} onClick={() => setPlaying(true)} disabled={playing}>
            Play
          </button>
          <button type="button" className={styles.secondary} onClick={() => setPlaying(false)}>
            Pause
          </button>
          <button
            type="button"
            className={styles.secondary}
            onClick={() => {
              setPlaying(false);
              setEventIndex(0);
            }}
          >
            Reset
          </button>
          <button
            type="button"
            className={styles.secondary}
            onClick={() => {
              setPlaying(false);
              setEventIndex((current) => Math.min(current + 1, result.events.length));
            }}
          >
            Step
          </button>
          <label>
            Speed
            <select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}>
              <option value={1}>1× event steps</option>
              <option value={2}>2× event steps</option>
              <option value={4}>4× event steps</option>
            </select>
          </label>
        </div>
        <p className={styles.clock}>
          Simulated clock {formatSimTime(view.timeNs)} · event {view.eventIndex} of{" "}
          {result.events.length}. Play advances one recorded event at a time. Reduced-motion
          settings jump to the final event.
        </p>
      </section>

      <section className={styles.section} aria-labelledby="scheduler-timeline">
        <h2 id="scheduler-timeline">Request timeline</h2>
        <p>Text labels name each state. Color is secondary.</p>
        <ol className={styles.timeline}>
          {view.requests.map((request) => (
            <li key={request.id} className={styles[request.status]}>
              <span className={styles.badge}>{request.status}</span>
              Request {request.id}
              {request.batchId !== null ? ` · batch ${request.batchId}` : ""}
            </li>
          ))}
        </ol>
        {view.requests.length === 0 ? <p>No events yet. Press Play or Step.</p> : null}
      </section>

      <section className={styles.section} aria-labelledby="scheduler-forming">
        <h2 id="scheduler-forming">Current forming batch</h2>
        {view.forming !== null ? (
          <p>
            Batch {view.forming.batchId} holds requests {view.forming.requestIds.join(", ")}. Opened
            at {formatSimTime(view.forming.openedNs)}
            {batching ? `; deadline ${formatSimTime(view.forming.deadlineNs)}` : ""}.
          </p>
        ) : (
          <p>No batch is forming. The worker is idle or executing a sealed group.</p>
        )}
      </section>

      <section className={styles.section} aria-labelledby="scheduler-seals">
        <h2 id="scheduler-seals">Seal-event history</h2>
        <ol className={styles.seals}>
          {view.seals.map((seal) => (
            <li key={seal.batchId}>
              {formatSimTime(seal.timeNs)} · batch {seal.batchId} · size {seal.requestIds.length} ·{" "}
              {sealReasonLabel(seal.reason)} · requests {seal.requestIds.join(", ")}
            </li>
          ))}
        </ol>
        {view.seals.length === 0 ? <p>No seals yet.</p> : null}
      </section>

      <section className={styles.section} aria-labelledby="scheduler-policy">
        <h2 id="scheduler-policy">Policy explanation</h2>
        <p>
          <strong>
            {cell} — {summary.title}.
          </strong>{" "}
          {summary.modeled}
        </p>
        <p>
          <strong>Simplified:</strong> {summary.simplified}
        </p>
      </section>

      <section className={styles.section} aria-labelledby="scheduler-limits">
        <h2 id="scheduler-limits">Limitations</h2>
        <ul>
          <li>This is not a live InferEdge scheduler and not a performance benchmark.</li>
          <li>Cell A never groups requests. Batching-enabled cells may still seal size-one batches.</li>
          <li>G4 uses the documented integer EWMA approximation, not a bit-identical C++ port.</li>
          <li>
            Execution bars are a fixed illustrative duration of 3 simulated milliseconds per request.
            They are not measured ONNX Runtime times.
          </li>
          <li>One educational worker and no admission rejection, shutdown, or metrics path.</li>
          <li>Do not read these seals as the audited GEMM P50 study.</li>
        </ul>
      </section>
    </PageLayout>
  );
}
