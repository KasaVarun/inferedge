import { PageLayout } from "../components/PageLayout";
import styles from "../components/PageLayout.module.css";

export function OverviewPage() {
  return (
    <PageLayout kicker="Overview" title="A C++20 inference runtime and scheduler">
      <p>
        InferEdge loads ONNX models with ONNX Runtime 1.31.0 CPU, runs float32 inference through
        InferenceEngine, and schedules requests with one engine per worker. This website is a
        static presentation of that work. It does not execute models in the browser.
      </p>
      <section className={styles.card} aria-labelledby="overview-status">
        <h2 id="overview-status">Project status</h2>
        <p className={styles.note}>
          The repository currently includes a CLI, sequential and concurrent benchmark tools, a
          request scheduler with configured and arrival-aware batching, fixture-backed GEMM and
          ResNet-18 correctness paths, and Linux CI. There is no HTTP API, GPU provider, or
          production server on this site.
        </p>
      </section>
    </PageLayout>
  );
}
