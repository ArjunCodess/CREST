# Contribution and evidence map

| Claim | Supported? | Evidence |
|-------|------------|----------|
| Discrete conservation polytope estimator implemented | Yes | `src/research/boundary_conservation.py`, tests |
| Partial-ID LP extrema on path flows | Yes | `src/research/partial_identification.py` |
| Single completed final holdout v4 (100 scenarios) | Yes | `final_receipt.json`, registry |
| Bounded deployable **worse** than masked mean on missing RMSE (n=81) | Yes | `final_analysis.json` paired block |
| 19 infeasible scenarios with documented breakdown | Yes | `denominator_reconciliation.totals` |
| LP data coverage 79/81 on bounded-valid rows | Yes | Same JSON |
| Boundary interval coverage 80/100 matrix-wide | Yes | Same JSON |
| Post-hoc bootstrap corrected (not frozen) | Yes | `analysis_receipt.json`, erratum |
| OD NNLS comparative ranking | **Withdrawn** | Erratum + `od_nnls_benchmark_status` |
| Forward OD comparator uses observed rows consistently | Yes | `src/research/baselines.py`, missing-versus-zero regression |
| Cross-site development MAE on one US-101 recording | Yes | `generated/roadmap_development.json`, speed and flow scored on shared masks |
| Incremental detector replay reproduces full recomputation | Yes | Development receipt, both measured corridors |
| Vehicle proxy results use an issuance-time path projection | Yes | Development receipt and trajectory replay scorer |
| Beat GNN / PIDL / SOTA | **No evidence** | Baselines not run |
| Real-network generalization | **No evidence** | Measured pilot development-only |
| Algorithm novelty / first partial-ID traffic estimator | **Not claimed** | Literature prior art |
| Blind v3 preregistration repeat | **No** | v4 amended after generative failure |
