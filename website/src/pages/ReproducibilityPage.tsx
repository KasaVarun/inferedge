import { Link } from "react-router-dom";
import {
  CPP_CI,
  GEMM_ONNX_SHA256,
  GEMM_STUDY,
  GITHUB_REPO,
  PRIMARY_JSON,
  PRIMARY_MARKDOWN,
  RESNET18_ONNX_SHA256,
  WEBSITE_CI,
  githubDocUrl,
} from "../catalog/inferedgeEvidence";
import { PageLayout } from "../components/PageLayout";
import styles from "./ReproducibilityPage.module.css";

export function ReproducibilityPage() {
  return (
    <PageLayout kicker="Reproducibility" title="Source, methodology, and CI">
      <p className={styles.disclaimer} role="note">
        This page cites committed source, hash-pinned models, published methodology, and specific
        successful CI runs. It does not host raw trial directories and does not claim that readers
        can recreate the original 20-block dataset without those unpublished records.
      </p>
      <p className={styles.lede}>
        InferEdge is a static dashboard over public artifacts. Educational Scheduler Explorer
        events are not measured latency.
      </p>

      <section className={styles.section} aria-labelledby="repro-source">
        <h2 id="repro-source">A. Source repository</h2>
        <p>
          Public Git repository:{" "}
          <a href={GITHUB_REPO} rel="noreferrer">
            {GITHUB_REPO}
          </a>
          . C++20 runtime, CMake/Ninja build, website under <code>website/</code>.
        </p>
      </section>

      <section className={styles.section} aria-labelledby="repro-toolchain">
        <h2 id="repro-toolchain">B. Build and test toolchain</h2>
        <ul>
          <li>C++20, CMake, Ninja, project warnings as errors</li>
          <li>GoogleTest fetched by CMake</li>
          <li>ONNX Runtime 1.31.0 CPU (pinned configure-time archives)</li>
          <li>clang-format 23.1.3</li>
          <li>
            Website: Node 22 in CI, React 18.3.1, TypeScript 5.7.3, Vite 5.4.21
          </li>
        </ul>
      </section>

      <section className={styles.section} aria-labelledby="repro-models">
        <h2 id="repro-models">C. Model artifact provenance</h2>
        <ul>
          <li>
            <code>gemm_f32</code> ONNX SHA-256 <code className={styles.hash}>{GEMM_ONNX_SHA256}</code>
          </li>
          <li>
            <code>resnet18_f32</code> ONNX SHA-256{" "}
            <code className={styles.hash}>{RESNET18_ONNX_SHA256}</code>
          </li>
        </ul>
        <p>
          Pins are checked at CMake configure time. See the{" "}
          <Link to="/models">Models page</Link> for shapes and fixture identities.
        </p>
      </section>

      <section className={styles.section} aria-labelledby="repro-protocol">
        <h2 id="repro-protocol">D. Published benchmark protocol</h2>
        <p>
          One reviewed extract: contrast <code>{GEMM_STUDY.contrastId}</code>. Documentation:{" "}
          <a href={githubDocUrl(PRIMARY_MARKDOWN)} rel="noreferrer">
            {PRIMARY_MARKDOWN}
          </a>{" "}
          and{" "}
          <a href={githubDocUrl(PRIMARY_JSON)} rel="noreferrer">
            {PRIMARY_JSON}
          </a>
          . Figures stay on the <Link to="/performance">Performance page</Link>. This page does not
          recompute statistics.
        </p>
        <dl className={styles.meta}>
          <div>
            <dt>Model</dt>
            <dd>
              <code>{GEMM_STUDY.modelId}</code>
            </dd>
          </div>
          <div>
            <dt>Offered rate</dt>
            <dd>{GEMM_STUDY.offeredRps} requests/s</dd>
          </div>
          <div>
            <dt>Trial duration</dt>
            <dd>{GEMM_STUDY.durationMs} ms plus {GEMM_STUDY.warmupRequests} warmup requests</dd>
          </div>
          <div>
            <dt>Workers</dt>
            <dd>{GEMM_STUDY.workers}</dd>
          </div>
          <div>
            <dt>Design</dt>
            <dd>
              {GEMM_STUDY.uncountedPilots} uncounted pilots, then {GEMM_STUDY.countedBlocks} counted
              paired blocks with odd/even arm rotation
            </dd>
          </div>
        </dl>
      </section>

      <section className={styles.section} aria-labelledby="repro-stats">
        <h2 id="repro-stats">E. Statistical methodology</h2>
        <ul>
          <li>Primary endpoint: {GEMM_STUDY.endpoint} (G4 minus G2)</li>
          <li>
            Estimator: <code>{GEMM_STUDY.bootstrapMethod}</code>
          </li>
          <li>{GEMM_STUDY.replicates.toLocaleString("en-US")} replicates</li>
          <li>Seed {GEMM_STUDY.seed}</li>
          <li>{GEMM_STUDY.interval} percentile interval on the paired-block mean</li>
          <li>Experimental unit is the paired block, not independent requests</li>
        </ul>
        <p>
          Winner and superiority flags stay false in the published extract. A CI that excludes zero
          is still not a universal scheduler ranking.
        </p>
      </section>

      <section className={styles.section} aria-labelledby="repro-ci">
        <h2 id="repro-ci">F. Linux CI evidence</h2>
        <p>
          The following counts come from two successful Actions runs on commit{" "}
          <code>{CPP_CI.sha.slice(0, 7)}</code>. They do not describe every future commit.
        </p>
        <ul>
          <li>
            C++ CI: CTest {CPP_CI.ctest}, benchmark Python {CPP_CI.benchmarkPython}, dynamic ONNX{" "}
            {CPP_CI.dynamicOnnx}.{" "}
            <a href={CPP_CI.url} rel="noreferrer">
              Actions run 38840295859
            </a>
          </li>
          <li>
            Website CI: {WEBSITE_CI.frontendTests} frontend tests.{" "}
            <a href={WEBSITE_CI.url} rel="noreferrer">
              Actions run 38840349770
            </a>
          </li>
        </ul>
      </section>

      <section className={styles.section} aria-labelledby="repro-limits">
        <h2 id="repro-limits">G. Known limitations</h2>
        <ul>
          <li>Measured on {GEMM_STUDY.host} with {GEMM_STUDY.runtime}.</li>
          <li>Arrival fidelity is not certified.</li>
          <li>
            {GEMM_STUDY.coarseLagMisses} non-gating coarse generator-lag misses (G2 block 14, G4
            block 16, G2 block 18).
          </li>
          <li>
            {GEMM_STUDY.observationQuantumMs} ms open-loop completion observation quantum is included
            in producer-observed latency.
          </li>
          <li>Sparse traffic produced almost exclusively size-one batches.</li>
          <li>No throughput improvement claim. Offered rate was frozen.</li>
          <li>No GPU or LLM performance claim.</li>
          <li>Related published A/G2 and A/G2/G3 studies are not combinable with this extract.</li>
          <li>Raw trial stdout, stderr, and unpublished run directories are not distributed here.</li>
        </ul>
      </section>

      <section className={styles.section} aria-labelledby="repro-public">
        <h2 id="repro-public">H. What is and is not public</h2>
        <ul>
          <li>Source code is publicly available.</li>
          <li>Model artifacts are committed and hash-pinned.</li>
          <li>Benchmark methodology and aggregate results are published under docs/benchmarks.</li>
          <li>Historical raw trial records are not publicly distributed.</li>
          <li>
            Independent readers cannot reproduce the original 20-block dataset without those raw
            records. This site does not link unpublished <code>scheduler_runs</code> trees.
          </li>
        </ul>
        <p>
          Published technical documentation:{" "}
          <a href={githubDocUrl(PRIMARY_MARKDOWN)} rel="noreferrer">
            arrival-aware sparse GEMM
          </a>
          ,{" "}
          <a href={githubDocUrl("neural-batching-gemm.md")} rel="noreferrer">
            prior A-versus-G2 GEMM
          </a>
          , and{" "}
          <a href={githubDocUrl("adaptive-neural-batching-gemm.md")} rel="noreferrer">
            prior A/G2/G3 GEMM
          </a>
          . Those related studies remain separate evidence.
        </p>
      </section>
    </PageLayout>
  );
}
