import { Link } from "react-router-dom";
import { PageLayout } from "../components/PageLayout";
import styles from "./OverviewPage.module.css";

const GITHUB_URL = "https://github.com/KasaVarun/inferedge";
const CI_RUN_URL = "https://github.com/KasaVarun/inferedge/actions/runs/38820260794";

const PIPELINE = [
  { title: "Request producer", detail: "Open- or closed-loop clients submit work." },
  { title: "Admission / submit", detail: "submit() accepts a request or records rejection." },
  { title: "Bounded FIFO queue", detail: "Waiting work is BoundedFifoQueue, not a shared session pool." },
  {
    title: "Scheduler policy",
    detail:
      "One model path. Cell A stays unbatched. G2 uses fixed-delay batching, G3 adaptive batching, and G4 arrival-aware batching. Batching-enabled cells may still seal size-one batches.",
  },
  { title: "ONNX Runtime CPU workers", detail: "One InferenceEngine per worker; each run is single-threaded." },
  { title: "Inference results", detail: "Outputs stay with the completing request." },
  { title: "Completion observation", detail: "The producer observes the future after publication." },
] as const;

const CAPABILITIES = [
  {
    title: "C++20 inference execution",
    body: "InferenceEngine loads a pinned ONNX model and runs float32 inference on the calling worker thread. Concurrent workers do not share one engine.",
  },
  {
    title: "ONNX Runtime CPU",
    body: "ONNX Runtime 1.31.0 CPU is the pinned execution provider. CUDA, TensorRT, and CoreML providers are not configured.",
  },
  {
    title: "Dynamic batching",
    body: "G2, G3, and G4 enable configured batching on a single model path. Cell A does not batch. The scheduler does not regroup requests across models.",
  },
  {
    title: "Request scheduling",
    body: "InferenceScheduler admits requests, drains or cancels pending work on shutdown, and completes waiting queue items without stealing an in-flight run().",
  },
  {
    title: "A / G2 / G3 / G4 policies",
    body: "Cell A is unbatched. G2 is fixed-delay batching, G3 is adaptive batching, and G4 is arrival-aware batching. Comparative latency figures apply only to the published GEMM extract, not as a ranking of every workload.",
  },
  {
    title: "Numerical correctness",
    body: "Committed fixtures check Add, GEMM, MLP, and synthetic ResNet-18 outputs independently of ONNX Runtime goldens used at generation time.",
  },
  {
    title: "Reproducible benchmarking",
    body: "Published extracts keep schema, protocol, and bootstrap settings. This dashboard presents those artifacts. It does not run new trials.",
  },
] as const;

const TECHNOLOGIES = [
  "C++20",
  "CMake",
  "GoogleTest",
  "ONNX Runtime 1.31.0 CPU",
  "Python benchmark tooling",
  "React",
  "TypeScript",
  "Vite",
  "GitHub Actions",
] as const;

export function OverviewPage() {
  return (
    <PageLayout kicker="Overview" title="InferEdge">
      <p className={styles.lede}>
        A C++20 inference runtime on ONNX Runtime 1.31.0 CPU. InferEdge admits work onto a bounded
        FIFO queue and applies a scheduler policy: unbatched Cell A, G2 fixed-delay batching, G3
        adaptive batching, or G4 arrival-aware batching. Each worker runs one InferenceEngine. This
        site is a static presentation of that engineering. It does not host a live inference API.
      </p>
      <p className={styles.actions}>
        <a className={styles.primary} href={GITHUB_URL} rel="noreferrer">
          View on GitHub
        </a>
        <Link className={styles.secondary} to="/performance">
          Explore benchmark results
        </Link>
      </p>

      <section className={styles.section} aria-labelledby="overview-architecture">
        <h2 id="overview-architecture">Architecture</h2>
        <p>
          The diagram is an architectural illustration of the implemented scheduler path. It is
          not a runtime trace, profiler capture, or measurement of any specific trial.
        </p>
        <figure className={styles.diagram}>
          <figcaption className={styles.caption}>
            Shared admission and queue, then a scheduler policy fork: Cell A stays unbatched; G2,
            G3, and G4 enable batching. Both paths converge at ONNX Runtime CPU, then results and
            future observation.
          </figcaption>
          <svg
            className={styles.flowSvg}
            viewBox="0 0 1100 220"
            role="img"
            aria-hidden="true"
            focusable="false"
          >
            <defs>
              <marker
                id="overview-arrow"
                markerWidth="8"
                markerHeight="8"
                refX="6"
                refY="4"
                orient="auto"
              >
                <path d="M0 0 L8 4 L0 8 Z" fill="currentColor" />
              </marker>
            </defs>
            <g>
              <rect x="18" y="88" width="108" height="40" rx="6" fill="none" stroke="currentColor" />
              <text x="72" y="113" textAnchor="middle">
                Producer
              </text>
              <line x1="126" y1="108" x2="154" y2="108" stroke="currentColor" markerEnd="url(#overview-arrow)" />
              <rect x="160" y="88" width="108" height="40" rx="6" fill="none" stroke="currentColor" />
              <text x="214" y="113" textAnchor="middle">
                Submit
              </text>
              <line x1="268" y1="108" x2="296" y2="108" stroke="currentColor" markerEnd="url(#overview-arrow)" />
              <rect x="302" y="88" width="108" height="40" rx="6" fill="none" stroke="currentColor" />
              <text x="356" y="113" textAnchor="middle">
                FIFO
              </text>
              <line x1="410" y1="108" x2="438" y2="108" stroke="currentColor" markerEnd="url(#overview-arrow)" />
              <rect x="444" y="88" width="108" height="40" rx="6" fill="none" stroke="currentColor" />
              <text x="498" y="113" textAnchor="middle">
                Policy
              </text>
              <path
                d="M552 100 C 572 100, 572 36, 592 36"
                fill="none"
                stroke="currentColor"
                markerEnd="url(#overview-arrow)"
              />
              <path
                d="M552 116 C 572 116, 572 180, 592 180"
                fill="none"
                stroke="currentColor"
                markerEnd="url(#overview-arrow)"
              />
              <rect x="598" y="16" width="168" height="40" rx="6" fill="none" stroke="currentColor" />
              <text x="682" y="41" textAnchor="middle">
                A: Unbatched
              </text>
              <rect x="598" y="160" width="168" height="40" rx="6" fill="none" stroke="currentColor" />
              <text x="682" y="185" textAnchor="middle">
                G2/G3/G4: Batching
              </text>
              <path
                d="M766 36 C 786 36, 786 100, 806 100"
                fill="none"
                stroke="currentColor"
                markerEnd="url(#overview-arrow)"
              />
              <path
                d="M766 180 C 786 180, 786 116, 806 116"
                fill="none"
                stroke="currentColor"
                markerEnd="url(#overview-arrow)"
              />
              <rect x="812" y="88" width="80" height="40" rx="6" fill="none" stroke="currentColor" />
              <text x="852" y="113" textAnchor="middle">
                ORT CPU
              </text>
              <line x1="892" y1="108" x2="916" y2="108" stroke="currentColor" markerEnd="url(#overview-arrow)" />
              <rect x="922" y="88" width="68" height="40" rx="6" fill="none" stroke="currentColor" />
              <text x="956" y="113" textAnchor="middle">
                Results
              </text>
              <line x1="990" y1="108" x2="1008" y2="108" stroke="currentColor" markerEnd="url(#overview-arrow)" />
              <rect x="1014" y="88" width="72" height="40" rx="6" fill="none" stroke="currentColor" />
              <text x="1050" y="113" textAnchor="middle">
                Observe
              </text>
            </g>
          </svg>
          <ol className={styles.pipeline}>
            {PIPELINE.map((stage) => (
              <li key={stage.title} className={styles.stage}>
                <h3>{stage.title}</h3>
                <p>{stage.detail}</p>
              </li>
            ))}
          </ol>
        </figure>
      </section>

      <section className={styles.section} aria-labelledby="overview-capabilities">
        <h2 id="overview-capabilities">Engineering capabilities</h2>
        <p>Only capabilities that exist in the repository are listed.</p>
        <ul className={styles.cards}>
          {CAPABILITIES.map((item) => (
            <li key={item.title} className={styles.card}>
              <h3>{item.title}</h3>
              <p>{item.body}</p>
            </li>
          ))}
        </ul>
      </section>

      <section className={styles.section} aria-labelledby="overview-evidence">
        <h2 id="overview-evidence">Evidence</h2>
        <p>
          Measured latency, fixture-backed correctness, and CI are separate kinds of evidence.
          None of the items below is a production-scale or GPU result.
        </p>
        <ul className={styles.evidence}>
          <li className={styles.card}>
            <h3>Audited GEMM benchmark</h3>
            <p>
              Under the frozen 100 requests/s <code>gemm_f32</code> CPU workload on one Darwin
              arm64 host, G4 showed lower producer-observed P50 than G2. That contrast is not a
              universal superiority claim and is not a throughput gain.
            </p>
            <dl className={styles.stats}>
              <div>
                <dt>Contrast</dt>
                <dd>
                  <code>gemm_f32_G2_vs_G4_sparse_rps100</code>
                </dd>
              </div>
              <div>
                <dt>Mean paired difference</dt>
                <dd>−2512.41875 µs</dd>
              </div>
              <div>
                <dt>95% paired-block bootstrap CI</dt>
                <dd>[−2513.64165, −2511.35629875] µs</dd>
              </div>
              <div>
                <dt>Paired blocks</dt>
                <dd>20</dd>
              </div>
            </dl>
          </li>
          <li className={styles.card}>
            <h3>ResNet-18 integration</h3>
            <p>
              Architecture-faithful synthetic ResNet-18: input <code>[N, 3, 224, 224]</code>,
              output <code>[N, 1000]</code>. Fixture-backed numerical checks cover inference.
              Scheduler smokes for A/G2/G3/G4 are correctness-only.
            </p>
            <p className={styles.limit}>
              No pretrained weights, ImageNet accuracy, or ResNet latency publication.
            </p>
          </li>
          <li className={styles.card}>
            <h3>Linux CI</h3>
            <p>
              The dashboard-foundation commit passed Linux CI: C++ tests 800/800, benchmark Python
              537, dynamic ONNX 30.
            </p>
            <p>
              <a href={CI_RUN_URL} rel="noreferrer">
                Successful GitHub Actions run for commit da483d5
              </a>
            </p>
          </li>
        </ul>
      </section>

      <section className={styles.section} aria-labelledby="overview-technology">
        <h2 id="overview-technology">Technology</h2>
        <ul className={styles.tech}>
          {TECHNOLOGIES.map((label) => (
            <li key={label}>{label}</li>
          ))}
        </ul>
      </section>

      <section className={styles.section} aria-labelledby="overview-boundaries">
        <h2 id="overview-boundaries">Claim boundaries</h2>
        <p>
          InferEdge does not claim universal G4 superiority, guaranteed speedups, 3× throughput,
          GPU or LLM acceleration, ImageNet accuracy, pretrained ResNet weights, certified arrival
          fidelity, a live inference API, or production-scale deployment. Architecture diagrams
          illustrate structure. GEMM numbers are a measured extract on one Darwin arm64 host.
          ResNet work is correctness verification. Cell A does not batch.
        </p>
      </section>
    </PageLayout>
  );
}
