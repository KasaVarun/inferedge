import type { MouseEvent } from "react";
import { useEffect } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { SiteFooter } from "./components/SiteFooter";
import { SiteHeader } from "./components/SiteHeader";
import { ModelsPage } from "./pages/ModelsPage";
import { OverviewPage } from "./pages/OverviewPage";
import { PerformancePage } from "./pages/PerformancePage";
import { ReproducibilityPage } from "./pages/ReproducibilityPage";
import { SchedulerExplorerPage } from "./pages/SchedulerExplorerPage";
import { skipToMainContent } from "./skipToMain";
import styles from "./App.module.css";

const TITLES: Record<string, string> = {
  "/": "Overview · InferEdge",
  "/performance": "Performance · InferEdge",
  "/scheduler": "Scheduler Explorer · InferEdge",
  "/models": "Models · InferEdge",
  "/reproducibility": "Reproducibility · InferEdge",
};

function DocumentTitle() {
  const location = useLocation();
  useEffect(() => {
    document.title = TITLES[location.pathname] ?? "InferEdge";
  }, [location.pathname]);
  return null;
}

function onSkipToMain(event: MouseEvent<HTMLAnchorElement>) {
  skipToMainContent(event, (id) => document.getElementById(id));
}

export function App() {
  return (
    <div className={styles.shell}>
      <a className={styles.skip} href={import.meta.env.BASE_URL} onClick={onSkipToMain}>
        Skip to main content
      </a>
      <DocumentTitle />
      <SiteHeader />
      <Routes>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/performance" element={<PerformancePage />} />
        <Route path="/scheduler" element={<SchedulerExplorerPage />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/reproducibility" element={<ReproducibilityPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <SiteFooter />
    </div>
  );
}
