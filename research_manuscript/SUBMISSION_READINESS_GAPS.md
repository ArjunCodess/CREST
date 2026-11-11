# Submission readiness gaps

## Scientific

1. No comparison to contemporary graph or physics-informed estimators cited in literature.
2. Single synthetic geometry (chain) and one final matrix.
3. Bootstrap clusters = seeds (n=10); precision limited.
4. v4 protocol amended after v3 failure — disclose prominently in cover letter.
5. OD NNLS pathway needs code fix before any OD benchmark claims.

## Provenance

6. Historical `build_verification.json` analysis hash is stale; cite `packaging_analysis_receipt.json` in supplemental provenance.
7. No per-file digests for original v3 monoliths — do not claim stronger v3 archival than exists.

## Authorship and compliance

8. All declaration blocks in `AUTHOR_VERIFICATION_REQUIRED.md` unfinished.
9. No target venue selected; calls for workshops not verified in this pass.

## Venue families (deadlines unverified)

| Family | Fit | Notes |
|--------|-----|-------|
| IEEE ITSC workshops on traffic estimation / data fusion | Moderate | Negative + reproducibility angle |
| TRB poster / committee on traffic data | Moderate | Methods + transparency |
| NeurIPS Datasets & Benchmarks | Weak | Small synthetic matrix; no learned baseline grid |
| Transportation Research Part C short communication | Weak without external validation | Would need real data or stronger baselines |

**Action:** Human authors must verify current CFP deadlines before submission.

## PDF artifact

`pdflatex` was not available in the packaging environment PATH. Compile locally:

```powershell
cd crest/research_manuscript
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
