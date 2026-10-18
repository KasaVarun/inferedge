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
import {
  GENERATIVE_JSON,
  GENERATIVE_MARKDOWN,
  formatMillisecondInterval,
  formatMilliseconds,
  parseGenerativeOverheadPublication,
  publicationUrl as generativePublicationUrl,
  type GenerativeOverheadPublication,
} from "../performance/generativePublication";
import styles from "./PerformancePage.module.css";

type LoadState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; publication: GemmG2G4Publication };

type GenerativeLoadState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; publication: GenerativeOverheadPublication };

export function PerformancePage() {
  const [state, setState] = useState<LoadState>({ status: "loading" });
  const [generativeState, setGenerativeState] = useState<GenerativeLoadState>({ status: "loading" });
  const baseUrl = import.meta.env.BASE_URL;

  useEffect(() => {
    const controller = new AbortController();
    const url = publicationUrl(baseUrl, "data", PRIMARY_JSON);

    void fetch(url, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error("Published GEMM results JSON is not available");
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
        const message = error instanceof Error ? error.message : "GEMM publication validation failed";
        setState({ status: "error", message });
      });

    return () => controller.abort();
  }, [baseUrl]);

  useEffect(() => {
    const controller = new AbortController();
    const url = generativePublicationUrl(baseUrl, GENERATIVE_JSON);

    void fetch(url, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error("Published generative-overhead JSON is not available");
        }
        return (await response.json()) as unknown;
      })
      .then((payload) => {
        setGenerativeState({
          status: "ready",
          publication: parseGenerativeOverheadPublication(payload),
        });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        const message =
          error instanceof Error ? error.message : "Generative publication validation failed";
        setGenerativeState({ status: "error", message });
      });

    return () => controller.abort();
  }, [baseUrl]);

  return (
    <PageLayout kicker="Performance" title="Audited performance publications">
      <p className={styles.lede}>
        This page presents two reviewed publications that stay separate: a GEMM scheduler extract
        and one official Qwen3-0.6B generative-overhead campaign. Each JSON is loaded and validated
        on its own. Missing or altered fields are not replaced with invented statistics, and one
        study is not substituted for the other.
      </p>
      <p className={styles.actions}>
        <a className={styles.secondary} href={GITHUB_REPO} rel="noreferrer">
          GitHub repository
        </a>
      </p>
      <section className={styles.section} aria-labelledby="gemm-study">
        <h2 id="gemm-study">Audited GEMM scheduler extract</h2>
        <p>
          Producer-observed P50 for G2 versus G4 on a frozen 100 requests/s <code>gemm_f32</code>{" "}
          ONNX Runtime CPU workload. This extract is not combined with the generative-overhead
          campaign.
        </p>
        <p className={styles.actions}>
          <a className={styles.primary} href={githubDocUrl(PRIMARY_MARKDOWN)} rel="noreferrer">
            GEMM Markdown
          </a>
          <a className={styles.secondary} href={githubDocUrl(PRIMARY_JSON)} rel="noreferrer">
            GEMM JSON
          </a>
        </p>
        {state.status === "loading" ? <p>Loading the GEMM scheduler extract.</p> : null}
        {state.status === "error" ? (
          <p className={styles.error} role="alert">
            The GEMM scheduler extract cannot be displayed: {state.message}. No substitute figures
            are shown.
          </p>
        ) : null}
        {state.status === "ready" ? <ReadyPublication publication={state.publication} /> : null}
      </section>
      <GenerativeSection state={generativeState} />
    </PageLayout>
  );
}

function GenerativeSection({ state }: { state: GenerativeLoadState }) {
  return (
    <section className={styles.section} aria-labelledby="generative-study">
      <h2 id="generative-study">Qwen3-0.6B generative overhead: no directional winner</h2>
      <p className={styles.actions}>
        <a className={styles.primary} href={githubDocUrl(GENERATIVE_MARKDOWN)} rel="noreferrer">
          Generative-overhead Markdown
        </a>
        <a className={styles.secondary} href={githubDocUrl(GENERATIVE_JSON)} rel="noreferrer">
          Generative-overhead JSON
        </a>
      </p>
      {state.status === "loading" ? <p>Loading the generative-overhead publication.</p> : null}
      {state.status === "error" ? (
        <p className={styles.error} role="alert">
          The generative-overhead publication cannot be displayed: {state.message}. No substitute
          figures are shown.
        </p>
      ) : null}
      {state.status === "ready" ? <ReadyGenerative publication={state.publication} /> : null}
    </section>
  );
}

function ReadyGenerative({ publication }: { publication: GenerativeOverheadPublication }) {
  return (
    <>
      <p>{publication.publicationStatement}</p>
      <dl className={styles.stats}>
        <div>
          <dt>Model package</dt>
          <dd>
            <code>{publication.modelId}</code> · <code>{publication.quantization}</code> ·{" "}
            <code>{publication.ggufFilename}</code>
          </dd>
        </div>
        <div>
          <dt>Host</dt>
          <dd>One tested Apple host ({publication.hardwareModel})</dd>
        </div>
        <div>
          <dt>C01 pair count</dt>
          <dd>{publication.pairCount}</dd>
        </div>
        <div>
          <dt>InferEdge mean</dt>
          <dd>{formatMilliseconds(publication.inferedgeMeanNs)} ms</dd>
        </div>
        <div>
          <dt>Reference mean</dt>
          <dd>{formatMilliseconds(publication.referenceMeanNs)} ms</dd>
        </div>
        <div>
          <dt>Mean paired difference</dt>
          <dd>{formatMilliseconds(publication.meanDifferenceNs)} ms</dd>
        </div>
        <div>
          <dt>95% bootstrap interval</dt>
          <dd>{formatMillisecondInterval(publication.ciLowNs, publication.ciHighNs)} ms</dd>
        </div>
        <div>
          <dt>Absolute threshold met</dt>
          <dd>no</dd>
        </div>
        <div>
          <dt>Relative threshold met</dt>
          <dd>no</dd>
        </div>
        <div>
          <dt>Strict directional CI exclusion</dt>
          <dd>no</dd>
        </div>
        <div>
          <dt>Winner declared</dt>
          <dd>no</dd>
        </div>
        <div>
          <dt>Practical equivalence claimed</dt>
          <dd>no</dd>
        </div>
        <div>
          <dt>Claim status</dt>
          <dd>
            <code>{publication.claimStatus}</code>
          </dd>
        </div>
      </dl>
      <p>
        The protocol label <code>{publication.signedDirectionLabel}</code> records only the sign of
        the paired sum. It is not a winner. Winner declared remains no, and practical equivalence is
        not claimed.
      </p>
    </>
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
