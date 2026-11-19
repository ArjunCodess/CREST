# PDF compile QA

| Check | Result |
|-------|--------|
| Style | George Kour `arxiv.sty` (see `TEMPLATE_PROVENANCE.md`) |
| Author | Arjun Vijay Prakash, Independent Researcher, `arjunv.prakash12345@gmail.com` |
| Compiler | MiKTeX 25.12 via `latexmk -norc -pdf` |
| Output | `research_manuscript/main.pdf` (6 pages, 2026-11-20) |
| Checks | All pages contain extractable text inside the page bounds. No unresolved citations, unresolved references, or overfull boxes. The changed page was rendered and inspected. |
| PDF metadata | `hyperref` `hypersetup` with `pdftitle`, `pdfauthor`, `pdfsubject`, `pdfkeywords` |

Regenerate the v4 figures and tables if the archived analysis changes. Regenerate the development table from matching roadmap receipts. [Build commands](README.md)
