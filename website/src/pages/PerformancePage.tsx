import { PageLayout } from "../components/PageLayout";
import styles from "../components/PageLayout.module.css";

export function PerformancePage() {
  return (
    <PageLayout kicker="Performance" title="Audited benchmark publications">
      <p>
        This section is reserved for reviewed GEMM scheduler results that already exist under
        docs/benchmarks. Phase 8.1 does not display numbers, charts, or winner claims.
      </p>
      <section className={styles.card} aria-labelledby="performance-scope">
        <h2 id="performance-scope">Scope</h2>
        <p className={styles.note}>
          Later work may present the authorized G2 versus G4 sparse GEMM extract, including
          producer-observed P50 and the paired-block bootstrap interval, with the published
          limitations intact. This page does not invent statistics.
        </p>
      </section>
    </PageLayout>
  );
}
