import { useState } from "react";
import { Link } from "react-router-dom";
import {
  ADDITIONAL_MODELS,
  GEMM_MODEL,
  GITHUB_REPO,
  RESNET18_MODEL,
} from "../catalog/inferedgeEvidence";
import styles from "./ModelsPage.module.css";
import { PageLayout } from "../components/PageLayout";

function HashValue({ label, value }: { label: string; value: string }) {
  const [copied, setCopied] = useState(false);

  return (
    <div className={styles.hashRow}>
      <p>
        <span className={styles.hashLabel}>{label}</span>
        <code className={styles.hash}>{value}</code>
      </p>
      <button
        type="button"
        className={styles.copy}
        onClick={() => {
          void navigator.clipboard?.writeText(value)
            .then(() => {
              setCopied(true);
              window.setTimeout(() => setCopied(false), 1500);
            })
            .catch(() => {
              setCopied(false);
            });
        }}
      >
        {copied ? "Copied" : "Copy hash"}
      </button>
    </div>
  );
}

export function ModelsPage() {
  return (
    <PageLayout kicker="Models" title="Committed workloads and correctness">
      <p className={styles.disclaimer} role="note">
        These cards describe committed ONNX artifacts and fixture-backed checks. They are not
        accuracy leaderboards and not a latency catalog. Measured scheduler latency exists only for
        the published <code>{GEMM_MODEL.id}</code> G2-versus-G4 extract.
      </p>
      <p className={styles.lede}>
        InferEdge loads pinned ONNX Runtime 1.31.0 CPU models. Hashes below match the CMake
        configure-time pins. CUDA, TensorRT, and CoreML providers are not configured.
      </p>

      <section className={styles.section} aria-labelledby="models-featured">
        <h2 id="models-featured">Featured models</h2>
        <ul className={styles.cards}>
          <li className={styles.card}>
            <h3>
              <code>{GEMM_MODEL.id}</code> · {GEMM_MODEL.title}
            </h3>
            <p>{GEMM_MODEL.purpose}</p>
            <dl className={styles.meta}>
              <div>
                <dt>Input</dt>
                <dd>
                  <code>{GEMM_MODEL.inputName}</code> {GEMM_MODEL.inputShape}
                </dd>
              </div>
              <div>
                <dt>Output</dt>
                <dd>
                  <code>{GEMM_MODEL.outputName}</code> {GEMM_MODEL.outputShape}
                </dd>
              </div>
              <div>
                <dt>Runtime</dt>
                <dd>{GEMM_MODEL.provider}</dd>
              </div>
              <div>
                <dt>Correctness</dt>
                <dd>{GEMM_MODEL.correctness}</dd>
              </div>
            </dl>
            <HashValue label="ONNX SHA-256" value={GEMM_MODEL.onnxSha256} />
            <p>{GEMM_MODEL.publishedStudy}</p>
            <p>
              <Link className={styles.linkButton} to="/performance">
                Open Performance page
              </Link>
            </p>
            <details>
              <summary>Limitations</summary>
              <ul>
                <li>No GPU or LLM acceleration claim.</li>
                <li>The published contrast is one sparse 100 RPS extract, not a general ranking.</li>
                <li>Catalog membership is not a performance result.</li>
              </ul>
            </details>
          </li>
          <li className={styles.card}>
            <h3>
              <code>{RESNET18_MODEL.id}</code> · {RESNET18_MODEL.title}
            </h3>
            <p>{RESNET18_MODEL.architecture}</p>
            <dl className={styles.meta}>
              <div>
                <dt>Input</dt>
                <dd>
                  <code>{RESNET18_MODEL.inputName}</code> {RESNET18_MODEL.inputShape}
                </dd>
              </div>
              <div>
                <dt>Output</dt>
                <dd>
                  <code>{RESNET18_MODEL.outputName}</code> {RESNET18_MODEL.outputShape}
                </dd>
              </div>
              <div>
                <dt>Runtime</dt>
                <dd>{RESNET18_MODEL.provider}</dd>
              </div>
              <div>
                <dt>Fixtures</dt>
                <dd>
                  IDs {RESNET18_MODEL.fixtureIdMin}–{RESNET18_MODEL.fixtureIdMax}. atol ={" "}
                  {RESNET18_MODEL.atol}, rtol = {RESNET18_MODEL.rtol}.
                </dd>
              </div>
            </dl>
            <HashValue label="ONNX SHA-256" value={RESNET18_MODEL.onnxSha256} />
            <p>{RESNET18_MODEL.correctness}</p>
            <details>
              <summary>Limitations</summary>
              <ul>
                <li>Not ImageNet-pretrained. No ImageNet accuracy claim.</li>
                <li>No published ResNet performance contrast or latency improvement.</li>
                <li>Scheduler A/G2/G3/G4 coverage is correctness smoke only.</li>
              </ul>
            </details>
          </li>
        </ul>
      </section>

      <section className={styles.section} aria-labelledby="models-catalog">
        <h2 id="models-catalog">Other verified fixtures</h2>
        <p>
          These identities are CMake-pinned and exercised by repository tests. None of them has a
          published scheduler latency contrast on this dashboard.
        </p>
        <div className={styles.tableWrap}>
          <table>
            <caption>Committed correctness fixtures that are not featured latency studies</caption>
            <thead>
              <tr>
                <th scope="col">Model id</th>
                <th scope="col">Tensors</th>
                <th scope="col">Correctness approach</th>
                <th scope="col">Published latency extract</th>
              </tr>
            </thead>
            <tbody>
              {ADDITIONAL_MODELS.map((model) => (
                <tr key={model.id}>
                  <th scope="row">
                    <code>{model.id}</code>
                  </th>
                  <td>{model.tensors}</td>
                  <td>{model.check}</td>
                  <td>{model.publishedLatency ? "Yes" : "No"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className={styles.section} aria-labelledby="models-provenance">
        <h2 id="models-provenance">Artifact provenance</h2>
        <p>
          Configure-time SHA-256 pins live in{" "}
          <a href={`${GITHUB_REPO}/blob/main/CMakeLists.txt`} rel="noreferrer">
            CMakeLists.txt
          </a>
          . ResNet fixture identities 1000–1007 are declared in{" "}
          <code>include/inferedge/resnet18_fixtures.hpp</code>. Normal CI does not regenerate ONNX
          files.
        </p>
        <p>
          <Link to="/reproducibility">How to cite source, hashes, and CI</Link>
        </p>
      </section>
    </PageLayout>
  );
}
