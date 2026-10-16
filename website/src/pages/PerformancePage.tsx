import { useEffect, useState } from "react";
import { PageLayout } from "../components/PageLayout";
import {
  DELTA_FIGURE,
  GITHUB_REPO,
  P50_FIGURE,
  PRIMARY_JSON,
  PRIMARY_MARKDOWN,
  formatCi,
  formatMicros,
  githubDocUrl,
  parseGemmG2G4Publication,
  publicationUrl,
  type GemmG2G4Publication,
} from "../performance/gemmPublication";
import styles from "./PerformancePage.module.css";

type LoadState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; publication: GemmG2G4Publication };

export function PerformancePage() {
  const [state, setState] = useState<LoadState>({ status: "loading" });
  const baseUrl = import.meta.env.BASE_URL;

  useEffect(() => {
    const controller = new AbortController();
    const url = publicationUrl(baseUrl, "data", PRIMARY_JSON);

    void fetch(url, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error("Published results JSON is not available");
        }
        return (await response.json()) as unknown;
      })
      .then((payload) => {
        setState({ status: "ready", publication: parseGemmG2G4Publication(payload) });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        const message = error instanceof Error ? error.message : "Publication validation failed";
        setState({ status: "error", message });
      });

    return () => controller.abort();
  }, [baseUrl]);

  return (
    <PageLayout kicker="Performance" title="Audited GEMM scheduler extract">
      <p className={styles.lede}>
        This page presents one reviewed publication: producer-observed P50 for G2 versus G4 on a
        frozen 100 requests/s <code>gemm_f32</code> ONNX Runtime CPU workload. Numbers come from the
        committed JSON. Missing or altered fields are not replaced with invented statistics.
      </p>
      <p className={styles.actions}>
        <a className={styles.primary} href={githubDocUrl(PRIMARY_MARKDOWN)} rel="noreferrer">
          Published Markdown
        </a>
        <a className={styles.secondary} href={githubDocUrl(PRIMARY_JSON)} rel="noreferrer">
          Machine-readable JSON
        </a>
        <a className={styles.secondary} href={GITHUB_REPO} rel="noreferrer">
          GitHub repository
        </a>
      </p>
      {state.status === "loading" ? <p>Loading the published extract.</p> : null}
      {state.status === "error" ? (
        <p className={styles.error} role="alert">
          The dashboard cannot display statistics: {state.message}. No substitute figures are
          shown.
        </p>
      ) : null}
      {state.status === "ready" ? <ReadyPublication publication={state.publication} /> : null}
    </PageLayout>
  );
}

function ReadyPublication({ publication }: { publication: GemmG2G4Publication }) {
  const baseUrl = import.meta.env.BASE_URL;
  const p50Src = publicationUrl(baseUrl, "figures", P50_FIGURE);
  const deltaSrc = publicationUrl(baseUrl, "figures", DELTA_FIGURE);

  return (
    <>
      <section className={styles.section} aria-labelledby="performance-headline">
        <h2 id="performance-headline">Primary contrast</h2>
        <p>
          Contrast <code>{publication.contrastId}</code> on one {publication.hostOs}{" "}
          {publication.hostArch} host. Endpoint: producer-observed P50. Sign convention is G4 minus
          G2. This is not a live inference service.
        </p>
        <dl className={styles.stats}>
          <div>
            <dt>G2 mean P50</dt>
            <dd>{formatMicros(publication.g2P50)} µs</dd>
          </div>
          <div>
            <dt>G4 mean P50</dt>
            <dd>{formatMicros(publication.g4P50)} µs</dd>
          </div>
          <div>
            <dt>Mean paired difference</dt>
            <dd>{formatMicros(publication.meanPairedDiff)} µs</dd>
          </div>
          <div>
            <dt>95% paired-block bootstrap CI</dt>
            <dd>{formatCi(publication.ciLow, publication.ciHigh)} µs</dd>
          </div>
          <div>
            <dt>Paired blocks</dt>
            <dd>{publication.pairedBlocks}</dd>
          </div>
          <div>
            <dt>Workload</dt>
            <dd>
              {publication.offeredRps} RPS · {publication.workerCount} worker · {publication.modelId}{" "}
              · ORT {publication.onnxRuntime} {publication.executionProvider}
            </dd>
          </div>
        </dl>
      </section>

      <section className={styles.section} aria-labelledby="performance-charts">
        <h2 id="performance-charts">Published figures</h2>
        <p>These PNGs are the audited copies synced from docs/benchmarks. They are not new trials.</p>
        <figure className={styles.figure}>
          <img
            src={p50Src}
            alt="Published producer-observed P50 chart for G2 and G4 on the gemm_f32 100 RPS extract."
          />
          <figcaption>Producer-observed P50 by arm. Source: {P50_FIGURE}.</figcaption>
        </figure>
        <figure className={styles.figure}>
          <img
            src={deltaSrc}
            alt="Published paired-block producer-observed P50 differences for G4 minus G2."
          />
          <figcaption>Paired-block P50 differences. Source: {DELTA_FIGURE}.</figcaption>
        </figure>
      </section>

      <section className={styles.section} aria-labelledby="performance-methods">
        <h2 id="performance-methods">Statistical methodology</h2>
        <ul>
          <li>
            Estimator: <code>{publication.bootstrapMethod}</code>
          </li>
          <li>Replicates: {publication.bootstrapReplicates.toLocaleString("en-US")}</li>
          <li>Seed: {publication.bootstrapSeed}</li>
          <li>
            Experimental unit: <code>{publication.experimentalUnit}</code>, not independent requests
          </li>
          <li>
            Primary metric: <code>{publication.primaryEndpoint}</code>
          </li>
          <li>Runner report schema: {publication.reportSchemaVersion}</li>
          <li>
            Completion timing: <code>{publication.completionTiming}</code>
          </li>
          <li>
            Batch-decision instrumentation: <code>{publication.batchDecisionInstrumentation}</code>
          </li>
        </ul>
        <p>
          Producer-observed latency starts immediately before <code>submit()</code> and ends when
          the collector observes the future ready. It includes scheduler processing and observation
          delay. It is not pure ONNX execution time.
        </p>
      </section>

      <section className={styles.section} aria-labelledby="performance-limits">
        <h2 id="performance-limits">Limitations</h2>
        <ul>
          <li>
            The scheduler used {publication.workerCount} worker on one {publication.hostOs}{" "}
            {publication.hostArch} host.
          </li>
          {publication.limitations.map((item) => (
            <li key={item}>{item}</li>
          ))}
          <li>
            Arrival fidelity status is <code>{publication.arrivalStatus}</code>. The coarse generator
            lag screen recorded {publication.coarseLagMissCount} counted misses and remained
            non-gating.
          </li>
          <li>{publication.batchingNote}</li>
        </ul>
      </section>

      <section className={styles.section} aria-labelledby="performance-related">
        <h2 id="performance-related">Earlier GEMM studies</h2>
        <p>
          The following publications are listed only as separate prior work. They are not combined
          into the paired G2 versus G4 estimate above.
        </p>
        <ul className={styles.related}>
          {publication.related.map((study) => (
            <li key={study.contrastId} className={styles.card}>
              <h3>
                <code>{study.contrastId}</code>
              </h3>
              <p>{study.relationship.replaceAll("_", " ")}</p>
              <p>Combinable into this paired estimate: no.</p>
              <p>
                <a href={githubDocUrl(study.markdown)} rel="noreferrer">
                  {study.markdown}
                </a>
                {" · "}
                <a href={githubDocUrl(study.resultsJson)} rel="noreferrer">
                  {study.resultsJson}
                </a>
              </p>
            </li>
          ))}
        </ul>
      </section>
    </>
  );
}
