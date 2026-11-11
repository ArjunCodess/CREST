# PDF compile QA

| Check | Result |
|-------|--------|
| Style | George Kour `arxiv.sty` (see `TEMPLATE_PROVENANCE.md`) |
| Author | Arjun Vijay Prakash, Independent Researcher, `arjunv.prakash12345@gmail.com` |
| Compiler | MiKTeX `pdflatex` + `bibtex main` + two `pdflatex` passes |
| Output | `research_manuscript/main.pdf` (5 pages after arxiv-style integration, 2026-11-11) |
| Known warnings | Overfull hboxes from long `\texttt` paths (cosmetic) |
| PDF metadata | `hyperref` `hypersetup` with `pdftitle`, `pdfauthor`, `pdfsubject`, `pdfkeywords` |

Regenerate figures/tables before compile if `final_analysis.json` changes.
