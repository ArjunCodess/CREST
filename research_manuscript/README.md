# Research manuscript package (separate from `paper/main.tex`)

Competition paper under `crest/paper/` is untouched.

## Regenerate artifacts

```powershell
cd crest
python research_manuscript/scripts/verify_immutable_inputs.py
python research_manuscript/scripts/generate_manuscript_tables.py
python research_manuscript/scripts/make_manuscript_figures.py
```

## Provenance chain

1. Frozen inference hash `c15d37e54fe38d63` — scenario rows in `results/research_final_v4/pilot_results.json`.
2. Post-hoc analysis `8ac2ca5f…` — `results/research_final_v4/final_analysis.json`.
3. Stale pointer in `results/research/build_verification.json` (`e00d4da5…`) — **do not rewrite**; see `packaging_analysis_receipt.json`.

## LaTeX preprint style

`main.tex` uses George Kour's MIT-licensed [arxiv-style](https://github.com/kourgeorge/arxiv-style) (`arxiv.sty`), as published on [Overleaf](https://www.overleaf.com/latex/templates/style-and-template-for-preprints-arxiv-bio-arxiv/fxsnsrzpnvwc). See `TEMPLATE_PROVENANCE.md` and `third_party_arxiv_style_LICENSE.txt`.

```powershell
cd crest/research_manuscript
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Primary sources

- `manuscript.md` — Markdown draft
- `main.tex` + `references.bib` — LaTeX sources
- `CONTRIBUTION_EVIDENCE_MAP.md`, `SUBMISSION_READINESS_GAPS.md`, `CITATION_VERIFICATION.md`
