# InferEdge website

Static, read-only dashboard for InferEdge. It does not run inference, does not expose a
scheduler, and does not deploy itself.

## Commands

```bash
cd website
npm ci
npm run sync:publications
npm run typecheck
npm run dev
npm run build
npm run preview
```

`npm run build` copies allowlisted files from `docs/benchmarks/` into `website/public/`
and then emits `dist/` for a GitHub Pages project site at `/inferedge/`. Navigation uses
hash routes so reloads work without a server rewrite.

Generated copies under `public/data/` and `public/figures/` are gitignored. Do not edit
the source publications.
