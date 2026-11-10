"""Regenerate LaTeX result tables from official metric files (read-only artifacts)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

PAPER = Path(__file__).resolve().parent
FIGURES = PAPER / "figures"


def resolve_artifacts() -> Path:
    candidates = [
        os.environ.get("TRAFFICFLOW_ARTIFACTS_OFFICIAL"),
        PAPER.parent / "artifacts" / "official",
    ]
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw)
        if (path / "temporal_cv.csv").exists():
            return path
    raise FileNotFoundError("Could not locate official artifacts (need temporal_cv.csv)")


def resolve_contender() -> Path | None:
    candidates = [
        os.environ.get("TRAFFICFLOW_ARTIFACTS_CONTENDER"),
        PAPER.parent / "artifacts" / "contender",
    ]
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw)
        if (path / "refinement_cv.csv").exists():
            return path
    return None


def escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def deployed_panels(run: Path) -> dict[str, str] | None:
    audit_path = run / "final_audit.json"
    if not audit_path.exists():
        return None
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    choices = audit.get("model_choices") or []
    if not choices:
        return None
    return {row["panel"]: row["state_model"] for row in choices}


def merge_cv(run: Path, contender: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv = pd.read_csv(run / "temporal_cv.csv")
    selected = cv[cv.selected.fillna(True).astype(bool)].copy() if "selected" in cv.columns else cv.copy()
    refined = pd.DataFrame()
    if contender is not None and (contender / "refinement_cv.csv").exists():
        refined = pd.read_csv(contender / "refinement_cv.csv")
        available = refined.panel.map(lambda p: (contender / f"{p}_refined.pkl").exists())
        refined = refined[~refined.accepted | available].copy()
        cv = pd.concat([cv, refined], ignore_index=True)
    deploy = deployed_panels(run)
    if deploy:
        pieces = []
        for panel, state_model in deploy.items():
            if state_model == "residual" and not refined.empty:
                block = refined[
                    (refined.panel == panel) & (refined.model == "residual") & (refined.projection == "fd")
                ].copy()
            else:
                block = cv[
                    (cv.panel == panel)
                    & (cv.model == "tree_only")
                    & (cv.projection == "fd")
                    & cv.selected.fillna(False).astype(bool)
                ].copy()
            if block.empty:
                block = selected[selected.panel == panel].copy()
            block["selected"] = True
            pieces.append(block)
        selected = pd.concat(pieces, ignore_index=True)
    elif not refined.empty:
        for panel in refined.loc[refined.accepted, "panel"].unique():
            if not (contender / f"{panel}_refined.pkl").exists():
                continue
            replacement = refined[
                (refined.panel == panel) & (refined.model == "residual") & (refined.projection == "fd")
            ].copy()
            replacement["selected"] = True
            selected = pd.concat([selected[selected.panel != panel], replacement], ignore_index=True)
    return cv, selected


def write_ablations(cv: pd.DataFrame, out: Path) -> None:
    comparable = cv.dropna(subset=["model", "projection"]).copy()
    panel_means = comparable.groupby(["model", "projection", "panel"], as_index=False).S_state.mean()
    panel_means["family"] = panel_means.panel.str.rsplit("_", n=1).str[0]
    ablations = (
        panel_means.groupby(["model", "projection", "family"], as_index=False)
        .S_state.mean()
        .groupby(["model", "projection"], as_index=False)
        .S_state.mean()
    )
    counts = panel_means.groupby(["model", "projection"], as_index=False).panel.nunique().rename(
        columns={"panel": "n_panels"}
    )
    ablations = ablations.merge(counts, on=["model", "projection"])
    ablations.to_csv(out / "ablations.csv", index=False)
    lines = [
        r"\begin{table}[t]",
        r"\caption{Family-macro artificial-mask holdout scores. Panel coverage varies, so partial-panel scores are not directly comparable with ten-panel scores. Model selection uses this same validation set.}",
        r"\centering\begin{tabular}{llrr}",
        r"\hline Model & FD map & Panels & $S_{state}$ \\",
        r"\hline",
    ]
    for row in ablations.itertuples():
        lines.append(f"{escape(row.model)} & {row.projection} & {row.n_panels} & {row.S_state:.4f} \\\\")
    lines.extend([r"\hline\end{tabular}", r"\end{table}"])
    (out / "ablations_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_results(selected: pd.DataFrame, out: Path) -> float | None:
    if selected.empty:
        (out / "results_table.tex").write_text("", encoding="utf-8")
        return None
    by_panel = selected.groupby("panel", as_index=False).S_state.mean()
    by_panel["family"] = by_panel.panel.str.rsplit("_", n=1).str[0]
    score = float(by_panel.groupby("family").S_state.mean().mean())
    rows = [
        f"{escape(r.panel)} & {r.regime} & {r.rmse_speed_kmh:.2f} & {r.rmse_flow_vph_per_lane:.2f} & {r.S_state:.3f} \\\\"
        for r in selected.itertuples()
    ]
    table = "\n".join(
        [
            r"\begin{table}[t]",
            r"\caption{Public training holdout: artificial-mask temporal holdout. These are not leaderboard scores.}",
            r"\centering\scriptsize",
            r"\begin{tabular}{llrrr}",
            r"\hline Panel & Regime & Speed RMSE & Flow/lane & $S_{state}$ \\",
            r"\hline",
            *rows,
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    (out / "results_table.tex").write_text(table, encoding="utf-8")
    return score


def write_od_link(run: Path, out: Path) -> None:
    val = pd.read_csv(run / "official_link_score_validation.csv")
    priv = pd.read_csv(run / "official_link_score_private.csv")
    panels = val[val.level == "panel"][["panel", "S_link"]].rename(columns={"S_link": "S_link_val"})
    panels = panels.merge(
        priv[priv.level == "panel"][["panel", "S_link"]].rename(columns={"S_link": "S_link_priv"}),
        on="panel",
        how="outer",
    )
    fam_val = val[val.level == "family"][["family_id", "S_link"]].rename(columns={"S_link": "S_link_val"})
    fam_priv = priv[priv.level == "family"][["family_id", "S_link"]].rename(columns={"S_link": "S_link_priv"})
    families = fam_val.merge(fam_priv, on="family_id", how="outer")
    lines = [
        r"\begin{table}[t]",
        r"\caption{Public $S_{\mathrm{link}}$ from organizer link-fit scoring (validation and private months).}",
        r"\centering\scriptsize",
        r"\begin{tabular}{lrr}",
        r"\hline Panel & Val. $S_{\mathrm{link}}$ & Priv. $S_{\mathrm{link}}$ \\",
        r"\hline",
    ]
    for row in panels.itertuples():
        lines.append(
            f"{escape(row.panel)} & {row.S_link_val:.6f} & {row.S_link_priv:.6f} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\multicolumn{3}{l}{\textit{Family macro-average}} \\")
    for row in families.itertuples():
        lines.append(
            f"{escape(row.family_id)} & {row.S_link_val:.6f} & {row.S_link_priv:.6f} \\\\"
        )
    overall_val = float(val.loc[val.level == "overall", "S_link"].iloc[0])
    overall_priv = float(priv.loc[priv.level == "overall", "S_link"].iloc[0])
    lines.extend(
        [
            r"\hline",
            f"Overall & {overall_val:.6f} & {overall_priv:.6f} \\\\",
            r"\hline\end{tabular}",
            r"\end{table}",
        ]
    )
    (out / "od_link_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_spatial(run: Path, out: Path) -> float | None:
    spatial_path = run / "spatial_cv.csv"
    if not spatial_path.exists():
        (out / "spatial_summary.json").write_text(json.dumps({"S_state": None}))
        return None
    spatial = pd.read_csv(spatial_path)
    spatial["family"] = spatial.panel.str.rsplit("_", n=1).str[0]
    score = float(spatial.groupby("family").S_state.mean().mean())
    (out / "spatial_summary.json").write_text(json.dumps({"S_state": score}, indent=2))
    return score


def main() -> None:
    run = resolve_artifacts()
    contender = resolve_contender()
    FIGURES.mkdir(parents=True, exist_ok=True)
    cv, selected = merge_cv(run, contender)
    cv.to_csv(FIGURES / "all_validation_results.csv", index=False)
    selected.to_csv(FIGURES / "temporal_cv.csv", index=False)
    selected.to_csv(FIGURES / "baseline_validation.csv", index=False)
    write_ablations(cv, FIGURES)
    score = write_results(selected, FIGURES)
    write_od_link(run, FIGURES)
    spatial_score = write_spatial(run, FIGURES)
    audit = json.loads((run / "final_audit.json").read_text(encoding="utf-8")) if (run / "final_audit.json").exists() else {}
    ablation = pd.read_csv(FIGURES / "ablations.csv")
    historical = float(ablation[(ablation.model == "historical_mean") & (ablation.projection == "raw")].S_state.iloc[0])
    neural = ablation[(ablation.model == "neural_only") & (ablation.projection == "raw")]
    tree = ablation[(ablation.model == "tree_only") & (ablation.projection == "raw")]
    deploy = deployed_panels(run)
    refined_count = (
        sum(1 for m in deploy.values() if m == "residual")
        if deploy
        else int(selected[selected.model == "residual"].panel.nunique()) if "model" in selected.columns else 0
    )
    summary = {
        "S_state": score if score is not None else audit.get("holdout", {}).get("S_state"),
        "historical_mean_S_state": historical,
        "neural_only_S_state": float(neural.S_state.iloc[0]) if len(neural) else None,
        "tree_only_S_state": float(tree.S_state.iloc[0]) if len(tree) else None,
        "neural_panels": int(neural.n_panels.iloc[0]) if len(neural) else None,
        "tree_panels": int(tree.n_panels.iloc[0]) if len(tree) else None,
        "refined_panels": refined_count,
        "S_link": audit.get("holdout", {}).get("S_link"),
        "spatial_S_state": spatial_score,
        "official_overall_score": audit.get("official_overall_score"),
        "artifacts_root": str(run),
    }
    (PAPER / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    s_state = summary["S_state"]
    hist = summary["historical_mean_S_state"]
    spatial = summary["spatial_S_state"]
    link = summary["S_link"]
    refined = summary["refined_panels"]
    macros = [
        f"\\providecommand{{\\MacroSstate}}{{{s_state:.4f}}}",
        f"\\providecommand{{\\MacroSstateHist}}{{{hist:.4f}}}",
        f"\\providecommand{{\\MacroSpatialSstate}}{{{spatial:.4f}}}" if spatial is not None else "",
        f"\\providecommand{{\\MacroSlink}}{{{link:.6f}}}" if link is not None else "",
        f"\\providecommand{{\\MacroRefinedPanels}}{{{refined}}}",
    ]
    (PAPER / "generated_macros.tex").write_text("\n".join(m for m in macros if m) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
