import styles from "./SiteFooter.module.css";

export function SiteFooter() {
  return (
    <footer className={styles.footer}>
      <div className={styles.inner}>
        <p>InferEdge is MIT licensed. This dashboard is a static, read-only presentation.</p>
        <p>
          It does not run inference, does not expose a scheduler API, and does not declare a
          performance winner.
        </p>
      </div>
    </footer>
  );
}
