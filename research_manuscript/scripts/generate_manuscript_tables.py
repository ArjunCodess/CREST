"""Emit LaTeX tables from corrected final_analysis.json (no scenario rerun)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "results/research_final_v4/final_analysis.json"
OUT_DIR = REPO / "research_manuscript/generated"


def _fmt(x: float | None, nd: int = 2) -> str:
    if x is None:
        return "---"
    return f"{x:.{nd}f}"


def table_primary_comparison(data: dict) -> str:
    shared = data["paired_conditional_comparisons_shared_valid"]
    full = data["paired_full_matrix_baselines"]
    diffs = data["paired_diff_bounded_minus_baseline_shared_valid"]
    boot = data["uncertainty_bootstrap"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{State RMSE on missing links (vehicles). Shared-valid denominator "
        r"$n{=}81$ rows with optimal bounded LP point estimate; full-matrix column uses "
        r"all $n{=}100$ scenarios where the metric is finite. Bootstrap: 10 seed clusters, "
        r"10{,}000 replicates (\texttt{resampling\_seed=424242}); post-hoc, not frozen at preregistration.}",
        r"\label{tab:primary-rmse}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & \multicolumn{2}{c}{Shared valid ($n{=}81$)} & \multicolumn{2}{c}{Full matrix ($n{=}100$)} \\",
        r" & Mean & $\Delta$ bounded$-$base & Mean & $n$ finite \\",
        r"\midrule",
    ]
    order = [
        ("bounded_deployable", "Bounded deployable LP"),
        ("masked_mean", "Masked mean"),
        ("forward_fill", "Forward fill"),
        ("interval_midpoint", "Interval midpoint"),
        ("oracle_diagnostic", "Oracle (diagnostic)"),
    ]
    for key, label in order:
        s = shared[key]
        f = full[key]
        delta = ""
        if key != "bounded_deployable" and key in diffs:
            delta = _fmt(diffs[key]["mean_bounded_minus_baseline"], 2)
        elif key == "bounded_deployable":
            delta = "---"
        lines.append(
            f"{label} & {_fmt(s['mean'])} & {delta} & {_fmt(f['mean'])} & {int(f['n_finite'])} \\\\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Bootstrap 95\% CI for bounded missing RMSE:} "
        + f"[{_fmt(boot['primary_ci95_low'])}, {_fmt(boot['primary_ci95_high'])}]"
        + r"} \\",
        r"\multicolumn{5}{l}{\textit{Paired diff bounded$-$masked mean:} "
        + f"mean {_fmt(boot['paired_diff_masked_mean']['mean_of_replicate_paired_means'])} "
        + f"[{_fmt(boot['paired_diff_masked_mean']['ci95_low'])}, "
        + f"{_fmt(boot['paired_diff_masked_mean']['ci95_high'])}]"
        + r"} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def table_rmse_slices(data: dict) -> str:
    sl = data["rmse_slice_comparison_shared_valid"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{RMSE slices on shared-valid rows ($n{=}81$): missing, observed, and all links.}",
        r"\label{tab:rmse-slices}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Missing & Observed & All \\",
        r"\midrule",
    ]
    for key, label in [
        ("bounded", "Bounded deployable"),
        ("masked_mean", "Masked mean"),
        ("forward_fill", "Forward fill"),
    ]:
        lines.append(
            f"{label} & "
            f"{_fmt(sl['missing'][key]['mean'])} & "
            f"{_fmt(sl['observed'][key]['mean'])} & "
            f"{_fmt(sl['all'][key]['mean'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def table_matrix_outcomes(data: dict) -> str:
    t = data["denominator_reconciliation"]["totals"]
    br = t["expected_infeasible_breakdown"]
    cov = t
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Final holdout v4 matrix outcomes ($10$ stress variants $\times$ $10$ seeds).}",
        r"\label{tab:matrix-outcomes}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Quantity & Count \\",
        r"\midrule",
        f"Scenarios completed & {t['n_rows_matrix']} \\\\",
        f"Bounded point estimate valid & {t['n_bounded_point_valid']} \\\\",
        f"Infeasible conservation polytope & {t['n_infeasible_polytope']} \\\\",
        f"Infeasible: truth\\_outside\\_interval & {br['truth_outside_interval']} \\\\",
        f"Infeasible: intentional\\_boundary\\_inconsistency & {br['intentional_boundary_inconsistency']} \\\\",
        f"Infeasible: gaussian\\_misspecification & {br['gaussian_misspecification']} \\\\",
        f"Rows with generative boundary truth in stated box & {t['n_boundary_truth_in_interval_rows']} \\\\",
        r"\midrule",
        r"Boundary interval covers truth (net vehicles), matrix & "
        f"{cov['boundary_interval_covers_truth_net_vehicles']['unconditional_matrix']['n_covered']}/"
        f"{cov['boundary_interval_covers_truth_net_vehicles']['unconditional_matrix']['n_eligible']} \\\\",
        r"LP data covers truth (bounded-valid rows only) & "
        f"{cov['lp_data_covers_truth_net_vehicles']['conditional_on_bounded_point_valid']['n_covered']}/"
        f"{cov['lp_data_covers_truth_net_vehicles']['conditional_on_bounded_point_valid']['n_eligible']} \\\\",
        r"LP box covers truth (bounded-valid rows) & "
        f"{cov['lp_box_covers_truth_net_vehicles']['conditional_on_bounded_point_valid']['n_covered']}/"
        f"{cov['lp_box_covers_truth_net_vehicles']['conditional_on_bounded_point_valid']['n_eligible']} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def table_variant_primary(data: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-variant deployable primary metric "
        r"\texttt{bounded\_state\_rmse\_missing} (conditional mean, valid bounded rows only).}",
        r"\label{tab:variant-primary}",
        r"\footnotesize",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Variant & $n$ valid & $n$ infeasible & Mean RMSE & Std \\",
        r"\midrule",
    ]
    for row in data["variant_denominators"]:
        lines.append(
            f"{row['stress_variant'].replace('_', r'\_')} & "
            f"{row['n_bounded_point_valid']} & {row['n_infeasible_polytope']} & "
            f"{_fmt(row['primary_mean_conditional'])} & {_fmt(row['primary_std_conditional'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    data = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "table_primary_comparison.tex").write_text(table_primary_comparison(data), encoding="utf-8")
    (OUT_DIR / "table_rmse_slices.tex").write_text(table_rmse_slices(data), encoding="utf-8")
    (OUT_DIR / "table_matrix_outcomes.tex").write_text(table_matrix_outcomes(data), encoding="utf-8")
    (OUT_DIR / "table_variant_primary.tex").write_text(table_variant_primary(data), encoding="utf-8")
    manifest = {
        "source": str(ANALYSIS.relative_to(REPO)),
        "analysis_script_sha256_prefix": data.get("analysis_script_sha256_prefix"),
        "tables": [
            "table_primary_comparison.tex",
            "table_rmse_slices.tex",
            "table_matrix_outcomes.tex",
            "table_variant_primary.tex",
        ],
    }
    (OUT_DIR / "tables_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Wrote tables to", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
