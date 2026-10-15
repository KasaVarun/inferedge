import { useEffect, useId, useState } from "react";
import { NavLink } from "react-router-dom";
import styles from "./SiteHeader.module.css";

const GITHUB_URL = "https://github.com/KasaVarun/inferedge";

const ROUTES = [
  { to: "/", label: "Overview", end: true },
  { to: "/performance", label: "Performance", end: false },
  { to: "/scheduler", label: "Scheduler Explorer", end: false },
  { to: "/models", label: "Models", end: false },
  { to: "/reproducibility", label: "Reproducibility", end: false },
] as const;

export function SiteHeader() {
  const [open, setOpen] = useState(false);
  const menuId = useId();

  useEffect(() => {
    if (!open) {
      return undefined;
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        <NavLink className={styles.brand ?? ""} to="/" onClick={() => setOpen(false)}>
          <span className={styles.wordmark}>InferEdge</span>
          <span className={styles.meta}>v0.1.0 · CPU · C++20</span>
        </NavLink>
        <button
          type="button"
          className={styles.menuButton}
          aria-expanded={open}
          aria-controls={menuId}
          onClick={() => setOpen((value) => !value)}
        >
          {open ? "Close menu" : "Open menu"}
        </button>
        <nav id={menuId} className={`${styles.nav} ${open ? styles.navOpen : ""}`} aria-label="Site">
          {ROUTES.map((route) => (
            <NavLink
              key={route.to}
              to={route.to}
              end={route.end}
              className={({ isActive }) => `${styles.link} ${isActive ? styles.active : ""}`}
              onClick={() => setOpen(false)}
            >
              {route.label}
            </NavLink>
          ))}
          <a className={styles.github} href={GITHUB_URL} rel="noreferrer">
            GitHub repository
          </a>
        </nav>
      </div>
    </header>
  );
}
