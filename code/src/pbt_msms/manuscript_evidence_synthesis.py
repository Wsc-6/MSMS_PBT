"""Notebook 37: freeze the manuscript-level PBT-MS/MS evidence synthesis.

The synthesis keeps three evidence layers separate:

1. blinded ChemVista external performance (quantitative generalization),
2. Portugal wastewater DDA application (known-pollutant recovery plus a small
   truth audit), and
3. Bangladesh DIA application (identity-blind prioritization followed by
   evidence-tiered tentative annotation).

It does not refit a model, select a threshold, revise an identity, or pool
development-overlap compounds into the primary external test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VERSION = "PBT_MSMS_MANUSCRIPT_EVIDENCE_SYNTHESIS_37_V1"
FINAL_STATUS = "MANUSCRIPT_EVIDENCE_SYNTHESIS_FROZEN"

TASK_ORDER = [
    "P_screen_not_ready_binary",
    "B_LOGBCF_REGRESSION",
    "T_EC50_MG_L_REGRESSION",
    "T_LC50_MG_L_REGRESSION",
    "T_NOEC_MG_L_REGRESSION",
]
TASK_SHORT = {
    "P_screen_not_ready_binary": "P",
    "B_LOGBCF_REGRESSION": "B logBCF",
    "T_EC50_MG_L_REGRESSION": "T EC50",
    "T_LC50_MG_L_REGRESSION": "T LC50",
    "T_NOEC_MG_L_REGRESSION": "T NOEC",
}


def locate_root(root: str | Path | None = None) -> Path:
    if root is not None:
        result = Path(root).resolve()
    else:
        cwd = Path.cwd().resolve()
        result = next((p for p in [cwd, *cwd.parents] if (p / "outputs/pbt_msms_v1").exists()), cwd)
    if not (result / "outputs/pbt_msms_v1").exists():
        raise FileNotFoundError(f"Cannot locate project root from {result}")
    return result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def project_paths(root: Path) -> dict[str, Path]:
    cv = root / "outputs/pbt_msms_v1/chemvista_external_20_v1"
    pt = root / "external_data/portugal_wastewater_5830725"
    bd = root / "external_data/bangladesh_surface_water/06_revealed_evaluation/35_final_evidence_v1"
    out = root / "outputs/pbt_msms_v1/manuscript_synthesis_37_v1"
    return {
        "root": root,
        "output": out,
        "figures": out / "figures",
        "cv_manifest": cv / "20_run_manifest.json",
        "cv_prepare": cv / "20_preparation_manifest.json",
        "cv_metrics": cv / "20_external_metrics.csv",
        "cv_qc": cv / "20_qc_checks.csv",
        "pt26_manifest": pt / "manifests/26_case_interpretation_run_manifest.json",
        "pt36_manifest": pt / "08_external_validation/36_portugal_pbt_truth_validation_v1/36_run_manifest.json",
        "pt36_metrics": pt / "08_external_validation/36_portugal_pbt_truth_validation_v1/36_endpoint_performance.csv",
        "pt36_qc": pt / "08_external_validation/36_portugal_pbt_truth_validation_v1/36_validation_qc_checks.csv",
        "bd35_manifest": bd / "35_run_manifest.json",
        "bd35_clusters": bd / "35_cluster_final_evidence_table.csv",
        "bd35_top": bd / "35_top_priority_verification_table.csv",
    }


def verify_inputs(p: dict[str, Path]) -> tuple[dict[str, Any], ...]:
    names = [
        "cv_manifest", "cv_prepare", "cv_metrics", "cv_qc", "pt26_manifest", "pt36_manifest",
        "pt36_metrics", "pt36_qc", "bd35_manifest", "bd35_clusters", "bd35_top",
    ]
    missing = [str(p[name]) for name in names if not p[name].exists()]
    if missing:
        raise FileNotFoundError("Missing Notebook 37 input(s):\n" + "\n".join(missing))

    cv_manifest = read_json(p["cv_manifest"])
    cv_prepare = read_json(p["cv_prepare"])
    pt26 = read_json(p["pt26_manifest"])
    pt36 = read_json(p["pt36_manifest"])
    bd35 = read_json(p["bd35_manifest"])
    expected_statuses = {
        "ChemVista": (cv_manifest.get("status"), "READY_FOR_NOTEBOOK_21"),
        "ChemVista preparation": (cv_prepare.get("status"), "PASS"),
        "Portugal case": (pt26.get("status"), "PORTUGAL_CASE_COMPLETE_READY_FOR_MANUSCRIPT_INTEGRATION"),
        "Portugal truth audit": (pt36.get("status"), "PORTUGAL_REAL_MATRIX_VALIDATION_COMPLETE_UNDERPOWERED"),
        "Bangladesh evidence": (bd35.get("status"), "FINAL_EVIDENCE_TABLE_FROZEN"),
    }
    errors = [f"{name}: {actual} != {expected}" for name, (actual, expected) in expected_statuses.items() if actual != expected]
    if errors:
        raise RuntimeError("Unexpected upstream status:\n" + "\n".join(errors))
    if not bool(cv_manifest.get("qc_all_pass")) or not bool(pt26.get("qc_all_required_pass")) or not bool(pt36.get("qc_all_required_pass")):
        raise RuntimeError("An upstream QC manifest is not fully passing")

    frozen_hashes = [
        (p["cv_metrics"], cv_manifest["output_hashes"]["20_external_metrics.csv"]),
        (p["pt36_metrics"], pt36["output_hashes"]["36_endpoint_performance.csv"]),
        (p["bd35_clusters"], bd35["artifacts"]["35_cluster_final_evidence_table.csv"]["sha256"]),
        (p["bd35_top"], bd35["artifacts"]["35_top_priority_verification_table.csv"]["sha256"]),
    ]
    for path, expected in frozen_hashes:
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Frozen upstream artifact changed: {path.name}")
    hashes = {name: sha256_file(p[name]) for name in names}
    return cv_manifest, cv_prepare, pt26, pt36, bd35, hashes


def external_validation_table(metrics_path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    retained_false = metrics["retained_only"].astype(str).str.lower().eq("false")
    main = metrics.loc[metrics["cohort"].eq("novel_all") & retained_false].copy()
    main["task"] = pd.Categorical(main["task"], TASK_ORDER, ordered=True)
    main = main.sort_values("task").reset_index(drop=True)
    if len(main) != 5 or main["task"].nunique() != 5:
        raise RuntimeError("Expected five full-coverage novel_all external rows")
    rows: list[dict[str, Any]] = []
    for _, row in main.iterrows():
        task = str(row["task"])
        classification = row["kind"] == "classification"
        if classification:
            primary_metric, primary_value = "PR-AUC", float(row["pr_auc"])
            supporting_metric, supporting_value = "ROC-AUC", float(row["roc_auc"])
            interpretation = "Endpoint signal present, but ROC discrimination is limited"
        else:
            primary_metric, primary_value = "MAE", float(row["mae"])
            supporting_metric, supporting_value = "Spearman rho", float(row["spearman"])
            rho = supporting_value
            interpretation = (
                "Moderate rank agreement" if rho >= 0.4 else
                "Weak rank agreement" if rho >= 0.2 else
                "Limited rank agreement"
            )
        rows.append({
            "task": task,
            "endpoint": TASK_SHORT[task],
            "selected_pipeline": row["selected_pipeline"],
            "external_cohort": "novel_all",
            "compound_n": int(row["n"]),
            "coverage": float(row["realized_coverage"]),
            "primary_metric": primary_metric,
            "primary_value": primary_value,
            "primary_ci95_low": float(row["primary_ci_low"]),
            "primary_ci95_high": float(row["primary_ci_high"]),
            "supporting_metric": supporting_metric,
            "supporting_value": supporting_value,
            "rmse": float(row["rmse"]) if pd.notna(row["rmse"]) else np.nan,
            "r2": float(row["r2"]) if pd.notna(row["r2"]) else np.nan,
            "interpretation": interpretation,
        })
    return pd.DataFrame(rows)


def portugal_truth_table(metrics_path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    part = metrics.loc[metrics["evaluation_stratum"].eq("PRIMARY_UNSEEN_COMPOUND")].copy()
    rows: list[dict[str, Any]] = []
    for task in TASK_ORDER:
        q = part.loc[part["task"].eq(task)]
        if q.empty:
            raise RuntimeError(f"Portugal primary truth audit missing endpoint: {task}")
        preferred = "average_precision" if task == "P_screen_not_ready_binary" else "spearman_rho"
        selected = q.loc[q["metric"].eq(preferred)]
        if selected.empty:
            preferred = "mae"
            selected = q.loc[q["metric"].eq(preferred)]
        row = selected.iloc[0]
        rows.append({
            "task": task,
            "endpoint": TASK_SHORT[task],
            "unseen_truth_n": int(row["n_compounds"]),
            "display_metric": preferred,
            "display_value": float(row["value"]) if pd.notna(row["value"]) else np.nan,
            "interpretation_status": row["interpretation_status"],
            "allowed_role": "Supportive real-matrix truth check only",
        })
    return pd.DataFrame(rows)


def case_study_table(
    cv_prepare: dict[str, Any], pt26: dict[str, Any], pt36: dict[str, Any], bd35: dict[str, Any]
) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "evidence_layer": "Blinded external validation",
            "dataset": "ChemVista",
            "acquisition_context": "Independent measured MS/MS library",
            "scale": f"{cv_prepare['n_compounds']} compounds; 1,873 spectra",
            "primary_role": "Quantitative unseen-compound generalization",
            "claim_ceiling": "Endpoint-dependent predictive performance; no universal high-accuracy claim",
        },
        {
            "evidence_layer": "Known-pollutant real-matrix case",
            "dataset": "Portugal wastewater",
            "acquisition_context": "Real wastewater DDA-MS/MS",
            "scale": (
                f"{pt26['counts']['model_scored_features']:,} scored features; "
                f"{pt26['counts']['common_evaluation_groups']:,} groups; "
                f"{pt26['counts']['published_cecs_recovered']} published CECs recovered"
            ),
            "primary_role": "Workflow feasibility, known-pollutant recovery, and prioritization",
            "claim_ceiling": (
                f"Truth audit is supportive and underpowered: {pt36['counts']['primary_unseen_compounds']} unseen compounds"
            ),
        },
        {
            "evidence_layer": "Unknown-candidate real-matrix case",
            "dataset": "Bangladesh surface water",
            "acquisition_context": "DIA-derived pseudo-MS/MS with post-freeze SIRIUS annotation",
            "scale": (
                f"{bd35['n_features']} features; {bd35['n_clusters']} cross-polarity clusters; "
                f"{bd35['n_top_priority_clusters']} top-priority clusters"
            ),
            "primary_role": "Unknown-feature PBT prioritization and follow-up triage",
            "claim_ceiling": "Tentative Level 3 structures; exploratory PBT scores, not regulatory determinations",
        },
    ])


def evidence_claim_matrix() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "topic": "MS/MS-to-PBT model generalization",
            "primary_evidence": "ChemVista blinded novel_all external cohort",
            "allowed_statement": "The frozen framework showed endpoint-dependent generalization to unseen compounds.",
            "prohibited_overclaim": "The model predicts all P/B/T endpoints with uniformly high accuracy.",
        },
        {
            "topic": "Real wastewater feasibility",
            "primary_evidence": "Portugal DDA-MS/MS case",
            "allowed_statement": "The frozen workflow scored, ranked, and retrospectively recovered known wastewater contaminants.",
            "prohibited_overclaim": "Portugal independently re-estimated population-level predictive accuracy.",
        },
        {
            "topic": "DIA domain application",
            "primary_evidence": "Bangladesh DIA pseudo-MS/MS case",
            "allowed_statement": "The workflow prioritized unknown features and generated evidence-tiered candidates for confirmation.",
            "prohibited_overclaim": "DIA was converted into true DDA or tentative structures were confirmed identities.",
        },
        {
            "topic": "Chemical identification",
            "primary_evidence": "Public library matching plus SIRIUS/CSI:FingerID",
            "allowed_statement": "Candidates were assigned explicit evidence tiers and orthogonal follow-up actions.",
            "prohibited_overclaim": "In-silico structures constitute Schymanski Level 1 identification.",
        },
        {
            "topic": "Regulatory interpretation",
            "primary_evidence": "Frozen model outputs and case-study rankings",
            "allowed_statement": "Scores support screening and prioritization.",
            "prohibited_overclaim": "Scores replace guideline PBT testing or establish regulatory PBT status.",
        },
    ])


def limitations_register(pt36: dict[str, Any], bd35: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "limitation_id": "L37_01",
            "scope": "External model validation",
            "limitation": "External performance is endpoint-dependent; P and NOEC generalization are weaker than the strongest endpoints.",
            "handling": "Report every endpoint separately with sample size and confidence interval.",
        },
        {
            "limitation_id": "L37_02",
            "scope": "Portugal truth audit",
            "limitation": (
                "Only " + str(pt36["counts"]["primary_unseen_compounds"]) +
                " recovered unseen compounds have at least one frozen truth endpoint, with 1-7 truths per endpoint."
            ),
            "handling": "Use as supportive real-matrix evidence, not the primary performance estimate.",
        },
        {
            "limitation_id": "L37_03",
            "scope": "Portugal occurrence",
            "limitation": "Detection frequency and peak area are not measured concentrations; explicit blank/QC subtraction is unavailable.",
            "handling": "Describe hazard prioritization and occurrence patterns, not quantitative environmental risk.",
        },
        {
            "limitation_id": "L37_04",
            "scope": "Bangladesh identity",
            "limitation": (
                f"Only {bd35['cluster_evidence_tier_counts']['STRUCTURE_HIGHER_SUPPORT']} clusters reached the highest internal "
                "structure-support band, and candidates remain tentative."
            ),
            "handling": "Preserve evidence tiers; require authentic-standard RT/MS/MS for confirmation.",
        },
        {
            "limitation_id": "L37_05",
            "scope": "Bangladesh annotation conflicts",
            "limitation": f"{bd35['n_conflict_clusters']} clusters contain cross-member annotation conflicts.",
            "handling": "Keep conflicts unresolved and route them to manual cross-polarity review.",
        },
        {
            "limitation_id": "L37_06",
            "scope": "PBT interpretation",
            "limitation": "Predicted endpoints are screening estimates rather than guideline-study determinations.",
            "handling": "Use prioritization language and avoid regulatory classification claims.",
        },
    ])


def bangladesh_tier_table(bd35: dict[str, Any]) -> pd.DataFrame:
    preferred_order = [
        "STRUCTURE_HIGHER_SUPPORT", "STRUCTURE_MODERATE_SUPPORT", "STRUCTURE_LOW_SUPPORT",
        "FORMULA_ONLY", "EXACT_MASS_ONLY", "CONFLICTING_MEMBER_ANNOTATIONS",
    ]
    counts = bd35["cluster_evidence_tier_counts"]
    return pd.DataFrame([
        {"evidence_tier": tier, "cluster_n": int(counts.get(tier, 0)), "fraction": float(counts.get(tier, 0) / bd35["n_clusters"])}
        for tier in preferred_order
    ])


def manuscript_results_markdown(
    external: pd.DataFrame, pt26: dict[str, Any], pt36: dict[str, Any], bd35: dict[str, Any]
) -> str:
    e = external.set_index("task")
    return f"""# Notebook 37 manuscript-ready result statements

## English results draft

The frozen MS/MS-to-PBT framework was evaluated in a one-shot blinded ChemVista external test without external recalibration or hyperparameter selection. In the unseen-compound (`novel_all`) cohort, the P screening model achieved PR-AUC {e.loc['P_screen_not_ready_binary', 'primary_value']:.3f} and ROC-AUC {e.loc['P_screen_not_ready_binary', 'supporting_value']:.3f} (n={int(e.loc['P_screen_not_ready_binary', 'compound_n'])}). Regression performance was endpoint-dependent: Spearman correlations were {e.loc['B_LOGBCF_REGRESSION', 'supporting_value']:.3f} for B logBCF (n={int(e.loc['B_LOGBCF_REGRESSION', 'compound_n'])}), {e.loc['T_EC50_MG_L_REGRESSION', 'supporting_value']:.3f} for EC50 (n={int(e.loc['T_EC50_MG_L_REGRESSION', 'compound_n'])}), {e.loc['T_LC50_MG_L_REGRESSION', 'supporting_value']:.3f} for LC50 (n={int(e.loc['T_LC50_MG_L_REGRESSION', 'compound_n'])}), and {e.loc['T_NOEC_MG_L_REGRESSION', 'supporting_value']:.3f} for NOEC (n={int(e.loc['T_NOEC_MG_L_REGRESSION', 'compound_n'])}). These results support endpoint-dependent, rather than uniformly high, unseen-compound generalization.

Application to the Portugal wastewater DDA-MS/MS dataset produced frozen predictions for {pt26['counts']['model_scored_features']:,} features grouped into {pt26['counts']['common_evaluation_groups']:,} cross-sample feature groups. Post-freeze library matching recovered {pt26['counts']['published_cecs_recovered']} published contaminants, including all {pt26['counts']['watchlist_cecs_recovered']} article-defined Watch-List compounds. A leakage-controlled truth audit contained only {pt36['counts']['primary_unseen_compounds']} unseen compounds and 1-7 truth-labelled compounds per endpoint; it was therefore treated as supportive real-matrix evidence rather than a second definitive performance estimate.

In the Bangladesh surface-water DIA case, {bd35['n_features']} frozen feature predictions were consolidated into {bd35['n_clusters']} cross-polarity clusters, including {bd35['n_top_priority_clusters']} top-priority clusters. Post-freeze SIRIUS processing returned formulas for {bd35['feature_formula_success']} features and structure candidates for {bd35['feature_structure_success']} features. Evidence adjudication retained {bd35['cluster_evidence_tier_counts']['STRUCTURE_HIGHER_SUPPORT']} higher-support and {bd35['cluster_evidence_tier_counts']['STRUCTURE_MODERATE_SUPPORT']} moderate-support structure clusters, while preserving {bd35['n_conflict_clusters']} conflicting clusters for manual review. All computational structures remain tentative until authentic-standard confirmation.

## 中文结果草稿

冻结的 MS/MS–PBT 框架首先通过 ChemVista 盲法外部数据进行一次性评价，未利用外部标签重新校准或选择模型。未见化合物队列中，P 筛查的 PR-AUC 为 {e.loc['P_screen_not_ready_binary', 'primary_value']:.3f}、ROC-AUC 为 {e.loc['P_screen_not_ready_binary', 'supporting_value']:.3f}（n={int(e.loc['P_screen_not_ready_binary', 'compound_n'])}）。B、EC50、LC50 和 NOEC 的 Spearman 相关系数分别为 {e.loc['B_LOGBCF_REGRESSION', 'supporting_value']:.3f}、{e.loc['T_EC50_MG_L_REGRESSION', 'supporting_value']:.3f}、{e.loc['T_LC50_MG_L_REGRESSION', 'supporting_value']:.3f} 和 {e.loc['T_NOEC_MG_L_REGRESSION', 'supporting_value']:.3f}，表明模型具备端点依赖的跨化合物泛化能力，但并非所有端点均达到高性能。

葡萄牙污水 DDA-MS/MS 案例共获得 {pt26['counts']['model_scored_features']:,} 个特征的冻结预测，并整合为 {pt26['counts']['common_evaluation_groups']:,} 个跨样品特征组；揭盲后的谱库匹配恢复了 {pt26['counts']['published_cecs_recovered']} 种文献污染物，包括全部 {pt26['counts']['watchlist_cecs_recovered']} 种文章 Watch-List 化合物。由于独立真值层仅包含 {pt36['counts']['primary_unseen_compounds']} 个训练未见化合物，各端点仅有 1–7 个真值，因此该部分仅作为真实基质支持性证据，不作为第二套决定性性能估计。

孟加拉国地表水 DIA 案例将 {bd35['n_features']} 个冻结特征预测整合为 {bd35['n_clusters']} 个跨极性候选簇，并冻结其中 {bd35['n_top_priority_clusters']} 个最高优先级候选。揭盲后的 SIRIUS 分析为 {bd35['feature_formula_success']} 个特征提供分子式候选、为 {bd35['feature_structure_success']} 个特征提供结构候选；最终仅 {bd35['cluster_evidence_tier_counts']['STRUCTURE_HIGHER_SUPPORT']} 个簇进入较高支持结构层、{bd35['cluster_evidence_tier_counts']['STRUCTURE_MODERATE_SUPPORT']} 个进入中等支持结构层，同时保留 {bd35['n_conflict_clusters']} 个冲突簇供人工复核。所有计算结构均为待标准品确认的暂定候选。
"""


def make_figure(
    external: pd.DataFrame, pt26: dict[str, Any], pt36: dict[str, Any], tiers: pd.DataFrame, output: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))

    ax = axes[0]
    y = np.arange(len(external))
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#B279A2"]
    ax.barh(y, external["compound_n"], color=colors)
    ax.set_yticks(y, external["endpoint"])
    ax.invert_yaxis()
    ax.set_xlabel("Unseen compounds")
    ax.set_title("A  Blinded external validation")
    for i, row in external.iterrows():
        if row["task"] == "P_screen_not_ready_binary":
            label = f"PR-AUC {row['primary_value']:.2f}; ROC {row['supporting_value']:.2f}"
        else:
            label = f"rho {row['supporting_value']:.2f}; MAE {row['primary_value']:.2f}"
        ax.text(row["compound_n"] + 3, i, label, va="center", fontsize=9)
    ax.set_xlim(0, max(external["compound_n"]) * 1.75)

    ax = axes[1]
    pt_labels = ["Scored features", "Feature groups", "Recovered CECs", "Watch-List CECs", "Unseen truth compounds"]
    pt_values = [
        pt26["counts"]["model_scored_features"], pt26["counts"]["common_evaluation_groups"],
        pt26["counts"]["published_cecs_recovered"], pt26["counts"]["watchlist_cecs_recovered"],
        pt36["counts"]["primary_unseen_compounds"],
    ]
    y2 = np.arange(len(pt_labels))
    ax.barh(y2, pt_values, color="#76B7B2")
    ax.set_yticks(y2, pt_labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Count (log scale)")
    ax.set_title("B  Portugal wastewater DDA case")
    for i, value in enumerate(pt_values):
        ax.text(value * 1.12, i, f"{value:,}", va="center", fontsize=9)
    ax.set_xlim(1, max(pt_values) * 3)

    ax = axes[2]
    tier_labels = [
        "Higher structure", "Moderate structure", "Low structure",
        "Formula only", "Exact mass only", "Conflicting",
    ]
    tier_colors = ["#2E8B57", "#7CB342", "#FBC02D", "#FF9800", "#9E9E9E", "#D32F2F"]
    y3 = np.arange(len(tiers))
    ax.barh(y3, tiers["cluster_n"], color=tier_colors)
    ax.set_yticks(y3, tier_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Cross-polarity clusters")
    ax.set_title("C  Bangladesh DIA evidence tiers")
    for i, value in enumerate(tiers["cluster_n"]):
        ax.text(value + 2, i, str(int(value)), va="center", fontsize=9)
    ax.set_xlim(0, max(tiers["cluster_n"]) * 1.25)

    fig.suptitle("Frozen evidence chain for MS/MS-based PBT screening", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output / "37_integrated_evidence_chain.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "37_integrated_evidence_chain.pdf", bbox_inches="tight")
    plt.close(fig)


def run(root: str | Path | None = None) -> dict[str, Any]:
    root_path = locate_root(root)
    p = project_paths(root_path)
    p["output"].mkdir(parents=True, exist_ok=True)
    p["figures"].mkdir(parents=True, exist_ok=True)
    cv_manifest, cv_prepare, pt26, pt36, bd35, input_hashes = verify_inputs(p)

    external = external_validation_table(p["cv_metrics"])
    portugal_truth = portugal_truth_table(p["pt36_metrics"])
    case_studies = case_study_table(cv_prepare, pt26, pt36, bd35)
    claims = evidence_claim_matrix()
    limitations = limitations_register(pt36, bd35)
    tiers = bangladesh_tier_table(bd35)
    manuscript_text = manuscript_results_markdown(external, pt26, pt36, bd35)

    cv_qc = pd.read_csv(p["cv_qc"])
    pt_qc = pd.read_csv(p["pt36_qc"])
    qc_rows = [
        ("Q37_001", "ChemVista one-shot external evaluation status verified", cv_manifest["status"] == "READY_FOR_NOTEBOOK_21"),
        ("Q37_002", "All ChemVista QC checks pass", cv_qc["passed"].astype(str).str.lower().eq("true").all()),
        ("Q37_003", "External main table uses novel_all only", external["external_cohort"].eq("novel_all").all()),
        ("Q37_004", "Five endpoint-specific external rows retained", len(external) == 5 and external["task"].nunique() == 5),
        ("Q37_005", "No Portugal development-overlap compounds pooled into external table", not external["external_cohort"].eq("transfer").any()),
        ("Q37_006", "Portugal truth audit remains explicitly underpowered", pt36["status"].endswith("UNDERPOWERED")),
        ("Q37_007", "All Portugal Notebook 36 QC checks pass", pt_qc["passed"].astype(str).str.lower().eq("true").all()),
        ("Q37_008", "Bangladesh evidence tiers sum to 400 clusters", int(tiers["cluster_n"].sum()) == int(bd35["n_clusters"])),
        ("Q37_009", "Bangladesh identity ceiling remains tentative Level 3", "Level 3" in bd35["identity_claim_ceiling"]),
        ("Q37_010", "Regulatory PBT determinations are explicitly prohibited", claims["prohibited_overclaim"].str.contains("regulatory|guideline", case=False, regex=True).any()),
        ("Q37_011", "All frozen upstream hashes verified", True),
    ]
    qc = pd.DataFrame(qc_rows, columns=["check_id", "description", "passed"])
    qc["status"] = np.where(qc["passed"], "PASS", "FAIL")
    if not qc["passed"].all():
        raise RuntimeError("Notebook 37 QC failed:\n" + qc.loc[~qc["passed"]].to_string(index=False))

    outputs = {
        "37_external_validation_main.csv": external,
        "37_portugal_supportive_truth_audit.csv": portugal_truth,
        "37_case_study_summary.csv": case_studies,
        "37_evidence_claim_matrix.csv": claims,
        "37_limitations_register.csv": limitations,
        "37_bangladesh_evidence_tiers.csv": tiers,
        "37_synthesis_qc_checks.csv": qc,
    }
    for name, frame in outputs.items():
        atomic_csv(frame, p["output"] / name)
    atomic_text(p["output"] / "37_manuscript_results_draft.md", manuscript_text)
    make_figure(external, pt26, pt36, tiers, p["figures"])

    summary = {
        "version": VERSION,
        "status": FINAL_STATUS,
        "generated_at_utc": utc_now(),
        "evidence_chain": {
            "quantitative_external_validation": "ChemVista novel_all",
            "known_pollutant_real_matrix_case": "Portugal wastewater DDA-MS/MS",
            "unknown_candidate_real_matrix_case": "Bangladesh surface-water DIA pseudo-MS/MS",
        },
        "external_endpoint_n": dict(zip(external["task"], external["compound_n"].astype(int))),
        "portugal": pt26["counts"],
        "portugal_truth_audit": pt36["counts"],
        "bangladesh": {
            "features": bd35["n_features"],
            "clusters": bd35["n_clusters"],
            "top_priority_clusters": bd35["n_top_priority_clusters"],
            "evidence_tiers": bd35["cluster_evidence_tier_counts"],
        },
        "overall_conclusion": (
            "The evidence supports endpoint-dependent unseen-compound generalization and practical PBT screening/prioritization "
            "in DDA and DIA environmental workflows; it does not support uniformly high endpoint accuracy, confirmed identities, "
            "or regulatory PBT determinations."
        ),
        "qc_all_required_pass": True,
    }
    atomic_json(p["output"] / "37_synthesis_summary.json", summary)

    output_files = [p["output"] / name for name in outputs]
    output_files += [
        p["output"] / "37_manuscript_results_draft.md",
        p["output"] / "37_synthesis_summary.json",
        p["figures"] / "37_integrated_evidence_chain.png",
        p["figures"] / "37_integrated_evidence_chain.pdf",
    ]
    code_path = root_path / "src/pbt_msms/manuscript_evidence_synthesis.py"
    manifest = {
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "status": FINAL_STATUS,
        "qc_all_required_pass": True,
        "upstream_status": {
            "chemvista_20": cv_manifest["status"],
            "portugal_26": pt26["status"],
            "portugal_36": pt36["status"],
            "bangladesh_35": bd35["status"],
        },
        "input_hashes": input_hashes,
        "analysis_code": {"path": str(code_path), "sha256": sha256_file(code_path)},
        "summary": summary,
        "output_hashes": {path.relative_to(p["output"]).as_posix(): sha256_file(path) for path in output_files},
        "software": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "next_step": "Draft the manuscript using the frozen claim matrix, limitations register, tables, and integrated figure.",
    }
    atomic_json(p["output"] / "37_run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="Project root")
    args = parser.parse_args()
    print(json.dumps(run(args.root), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
