import { PageLayout } from "../components/PageLayout";
import styles from "../components/PageLayout.module.css";

export function SchedulerExplorerPage() {
  return (
    <PageLayout kicker="Scheduler Explorer" title="Illustrative scheduling cells">
      <p>
        This section is reserved for an educational visualization of unbatched A, fixed-delay G2,
        adaptive G3, and arrival-aware G4. Any future animation will be labeled as an illustrative
        simulation, not a live InferEdge scheduler.
      </p>
      <section className={styles.card} aria-labelledby="scheduler-placeholder">
        <h2 id="scheduler-placeholder">Not a runtime</h2>
        <p className={styles.note}>
          There is no browser backend, no request submission, and no measured latency on this
          page. Cells A, G2, G3, and G4 will be documented here in a later phase.
        </p>
      </section>
    </PageLayout>
  );
}
