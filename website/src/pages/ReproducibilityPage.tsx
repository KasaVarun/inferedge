import { PageLayout } from "../components/PageLayout";
import styles from "../components/PageLayout.module.css";

export function ReproducibilityPage() {
  return (
    <PageLayout kicker="Reproducibility" title="Source, methodology, and CI">
      <p>
        This section is reserved for repository provenance, model hash pins, published benchmark
        methodology, Linux CI evidence, and known limitations. Raw unpublished run directories
        are not hosted here.
      </p>
      <section className={styles.card} aria-labelledby="repro-placeholder">
        <h2 id="repro-placeholder">What this site will cite</h2>
        <p className={styles.note}>
          Later pages may link the GitHub repository, authorized docs/benchmarks extracts, and
          GitHub Actions results. They will not copy unpublished scheduler_runs, baselines, or
          experiment trees.
        </p>
      </section>
    </PageLayout>
  );
}
