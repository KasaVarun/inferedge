import type { ReactNode } from "react";
import styles from "./PageLayout.module.css";

type PageLayoutProps = {
  kicker: string;
  title: string;
  children: ReactNode;
};

export function PageLayout({ kicker, title, children }: PageLayoutProps) {
  return (
    <main id="main" className={styles.main} tabIndex={-1}>
      <article className={styles.article}>
        <p className={styles.kicker}>{kicker}</p>
        <h1>{title}</h1>
        {children}
      </article>
    </main>
  );
}
