import { PageLayout } from "../components/PageLayout";
import styles from "../components/PageLayout.module.css";

export function ModelsPage() {
  return (
    <PageLayout kicker="Models" title="Committed workloads and correctness">
      <p>
        This section is reserved for model identities, tensor shapes, SHA-256 pins, and
        fixture-backed checks. GEMM and ResNet-18 will be documented here without ImageNet
        accuracy or ResNet speed claims.
      </p>
      <section className={styles.card} aria-labelledby="models-placeholder">
        <h2 id="models-placeholder">Correctness, not leaderboard scores</h2>
        <p className={styles.note}>
          ResNet-18 support in the runtime is fixture-backed numerical checking. This dashboard
          will not present ImageNet top-1 figures or unpublished ResNet latency.
        </p>
      </section>
    </PageLayout>
  );
}
