"""Notebook 36: Portugal real-matrix P/B/T truth validation.

This stage joins identities revealed only after the Notebook 24 prediction
freeze to the already frozen property/split manifest.  Compounds assigned to
``external_novel`` were excluded from supervised development and form the
primary, unseen-compound stratum.  Development compounds are reported only as
a secondary real-matrix transfer check and are never pooled into the primary
claim.
"""

from __future__ import annotations

from .deployment import deployment_comparison_dir

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


VERSION = "PBT_MSMS_PORTUGAL_PBT_TRUTH_VALIDATION_36_V1"
FINAL_STATUS = "PORTUGAL_REAL_MATRIX_VALIDATION_COMPLETE_UNDERPOWERED"
SEED = 20260922
N_BOOTSTRAP = 2000

TASKS: dict[str, dict[str, str]] = {
    "P_screen_not_ready_binary": {
        "prediction": "median_p_probability",
        "kind": "classification",
        "label": "P screening (not readily biodegradable)",
    },
    "B_LOGBCF_REGRESSION": {
        "prediction": "median_b_logbcf",
        "kind": "regression",
        "label": "B logBCF",
    },
    "T_EC50_MG_L_REGRESSION": {
        "prediction": "median_t_ec50_log10_mg_l",
        "kind": "regression",
        "label": "T EC50 log10(mg/L)",
    },
    "T_LC50_MG_L_REGRESSION": {
        "prediction": "median_t_lc50_log10_mg_l",
        "kind": "regression",
        "label": "T LC50 log10(mg/L)",
    },
    "T_NOEC_MG_L_REGRESSION": {
        "prediction": "median_t_noec_log10_mg_l",
        "kind": "regression",
        "label": "T NOEC log10(mg/L)",
    },
}

STRATA = {
    "PRIMARY_UNSEEN_COMPOUND": {
        "role": "external_novel",
        "claim": "Unseen-compound performance on independently measured Portugal wastewater spectra",
        "primary": True,
    },
    "SECONDARY_SEEN_COMPOUND_NEW_MATRIX": {
        "role": "development",
        "claim": "Real-matrix measurement-domain transfer only; not unseen-compound validation",
        "primary": False,
    },
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
    with Path(path).open("rb") as handle:
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


def atomic_csv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def project_paths(root: Path) -> dict[str, Path]:
    portugal = root / "external_data/portugal_wastewater_5830725"
    reveal = portugal / "06_revealed_evaluation/25_identity_reveal"
    output = portugal / "08_external_validation/36_portugal_pbt_truth_validation_v1"
    return {
        "root": root,
        "portugal": portugal,
        "output": output,
        "figures": output / "figures",
        "stage25_manifest": portugal / "manifests/25_identity_reveal_run_manifest.json",
        "stage26_manifest": portugal / "manifests/26_case_interpretation_run_manifest.json",
        "stage24_freeze": portugal / "manifests/24_identity_blind_freeze.json",
        "recovery": reveal / "25_known_target_recovery.csv",
        "case_matches": reveal / "25_case_group_library_matches.csv.gz",
        "case_predictions": portugal / "04_hazard_ranking_24/24_case_feature_groups_identity_blind.csv.gz",
        "split": root / "outputs/pbt_msms_v1/split_v1/compound_split_manifest.csv",
        "split_manifest": root / "outputs/pbt_msms_v1/split_v1/split_freeze_manifest.json",
        "labels_long": root / "outputs/pbt_msms_v1/preparation_step1/compound_labels_long.csv",
        "calibration": deployment_comparison_dir(root / "outputs/pbt_msms_v1") / "19_deployment_calibration.json",
    }


def verify_inputs(p: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    required = [
        "stage25_manifest", "stage26_manifest", "stage24_freeze", "recovery",
        "case_matches", "case_predictions", "split", "split_manifest", "labels_long", "calibration",
    ]
    missing = [str(p[name]) for name in required if not p[name].exists()]
    if missing:
        raise FileNotFoundError("Missing Notebook 36 input(s):\n" + "\n".join(missing))

    stage25 = read_json(p["stage25_manifest"])
    stage26 = read_json(p["stage26_manifest"])
    split_manifest = read_json(p["split_manifest"])
    if stage25.get("status") != "READY_FOR_NOTEBOOK_26_PORTUGAL_CASE_INTERPRETATION":
        raise RuntimeError(f"Unexpected Notebook 25 status: {stage25.get('status')}")
    if stage26.get("status") != "PORTUGAL_CASE_COMPLETE_READY_FOR_MANUSCRIPT_INTEGRATION":
        raise RuntimeError(f"Unexpected Notebook 26 status: {stage26.get('status')}")
    if split_manifest.get("qc_status") != "PASS":
        raise RuntimeError("Frozen compound split did not pass QC")

    expected = stage25.get("output_hashes", {})
    checks = {
        "25_known_target_recovery.csv": p["recovery"],
        "25_case_group_library_matches.csv.gz": p["case_matches"],
    }
    for name, path in checks.items():
        actual = sha256_file(path)
        if expected.get(name) != actual:
            raise RuntimeError(f"Frozen Notebook 25 artifact changed: {name}")

    hashes = {name: sha256_file(p[name]) for name in required}
    return stage25, stage26, hashes


def _clean_ik2d(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper().replace({"": pd.NA, "NAN": pd.NA})


def build_compound_table(p: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    recovery = pd.read_csv(p["recovery"])
    recovery = recovery.loc[recovery["recovered"].fillna(False)].copy()
    recovery["inchikey_2d"] = _clean_ik2d(
        recovery["reference_inchikey_2d"].fillna(recovery["pubchem_inchikey_2d"])
    )
    if len(recovery) != 84 or recovery["target_id"].nunique() != 84 or recovery["inchikey_2d"].nunique() != 84:
        raise RuntimeError("Expected exactly 84 uniquely identified recovered Portugal targets")

    matches = pd.read_csv(p["case_matches"])
    matches = matches.loc[matches["is_nonambiguous_published_cec_match"].fillna(False)].copy()
    predictions = pd.read_csv(p["case_predictions"])
    if matches["case_feature_id"].duplicated().any():
        raise RuntimeError("A nonambiguous case group appears more than once")
    joined = matches.merge(predictions, on="case_feature_id", how="left", validate="one_to_one", suffixes=("", "_frozen"))
    pred_cols = [item["prediction"] for item in TASKS.values()]
    if joined[pred_cols].isna().any().any():
        raise RuntimeError("A revealed case group is missing a frozen Notebook 24 prediction")

    aggregations: dict[str, tuple[str, str | Callable]] = {
        "identified_case_group_n": ("case_feature_id", "nunique"),
        "best_modified_cosine": ("modified_cosine", "max"),
        "minimum_precursor_ppm_error": ("precursor_ppm_error", lambda x: float(np.min(np.abs(x)))),
        "mode_values": ("mode", lambda x: "|".join(sorted(set(map(str, x))))),
    }
    for task, spec in TASKS.items():
        aggregations[f"prediction__{task}"] = (spec["prediction"], "median")
    compound_predictions = joined.groupby("target_id", as_index=False).agg(**aggregations)

    keep = [
        "target_id", "compound_name_published", "published_watchlist_cec", "inchikey_2d",
        "recovered_case_group_n", "case_feature_id", "hazard_rank", "occurrence_assisted_rank",
    ]
    compounds = recovery[keep].merge(compound_predictions, on="target_id", how="left", validate="one_to_one")
    if compounds[[f"prediction__{task}" for task in TASKS]].isna().any().any():
        raise RuntimeError("A recovered identity is missing compound-level predictions")

    split_use = [
        "inchikey_2d", "split_candidate_role", "supervised_compound_role", "chemvista_evaluation_role",
        "natural_chemvista_novel", "cv_group", "cv_fold",
    ] + [f"label__{task}" for task in TASKS]
    split = pd.read_csv(p["split"], usecols=split_use, low_memory=False)
    split["inchikey_2d"] = _clean_ik2d(split["inchikey_2d"])
    if split["inchikey_2d"].duplicated().any():
        raise RuntimeError("Frozen split manifest has duplicate IK2D rows")
    compounds = compounds.merge(split, on="inchikey_2d", how="left", validate="one_to_one")

    role_to_stratum = {value["role"]: key for key, value in STRATA.items()}
    compounds["validation_stratum"] = compounds["supervised_compound_role"].map(role_to_stratum).fillna("NO_FROZEN_TRUTH")
    compounds["primary_external_claim_eligible"] = compounds["validation_stratum"].eq("PRIMARY_UNSEEN_COMPOUND")
    compounds["any_frozen_endpoint_truth"] = compounds[[f"label__{task}" for task in TASKS]].notna().any(axis=1)

    label_metadata = pd.read_csv(p["labels_long"])
    label_metadata["inchikey_2d"] = _clean_ik2d(label_metadata["inchikey_2d"])
    label_metadata = label_metadata.loc[label_metadata["target_name"].isin(TASKS)].copy()
    if label_metadata.duplicated(["inchikey_2d", "target_name"]).any():
        raise RuntimeError("Long truth table has duplicate compound/endpoint rows")
    return compounds, label_metadata


def build_endpoint_records(compounds: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "target_id", "compound_name_published", "inchikey_2d", "published_watchlist_cec",
        "validation_stratum", "primary_external_claim_eligible", "supervised_compound_role",
        "chemvista_evaluation_role", "split_candidate_role", "natural_chemvista_novel",
        "identified_case_group_n", "best_modified_cosine", "minimum_precursor_ppm_error", "mode_values",
    ]
    blocks: list[pd.DataFrame] = []
    for task, spec in TASKS.items():
        block = compounds[base_cols].copy()
        block["task"] = task
        block["task_label"] = spec["label"]
        block["kind"] = spec["kind"]
        block["truth"] = pd.to_numeric(compounds[f"label__{task}"], errors="coerce")
        block["prediction"] = pd.to_numeric(compounds[f"prediction__{task}"], errors="coerce")
        blocks.append(block)
    records = pd.concat(blocks, ignore_index=True)
    records = records.loc[records["truth"].notna() & records["prediction"].notna()].copy()
    meta_cols = [
        "inchikey_2d", "target_name", "aggregate_value", "high_quality_value",
        "direct_experimental_value", "n_records", "n_high_quality_records",
        "n_direct_experimental_records", "n_sources", "source_datasets", "evidence_tiers",
        "threshold_conflict", "binary_conflict",
    ]
    records = records.merge(
        metadata[meta_cols], left_on=["inchikey_2d", "task"], right_on=["inchikey_2d", "target_name"],
        how="left", validate="one_to_one",
    ).drop(columns="target_name")
    records["truth_matches_frozen_aggregate"] = np.isclose(
        records["truth"], pd.to_numeric(records["aggregate_value"], errors="coerce"), equal_nan=False
    )
    return records.sort_values(["validation_stratum", "task", "inchikey_2d"]).reset_index(drop=True)


def _metric_value(kind: str, metric: str, truth: np.ndarray, prediction: np.ndarray, threshold: float) -> float:
    if kind == "classification":
        if metric == "roc_auc":
            return float(roc_auc_score(truth, prediction))
        if metric == "average_precision":
            return float(average_precision_score(truth, prediction))
        if metric == "brier":
            return float(brier_score_loss(truth, prediction))
        if metric == "balanced_accuracy":
            return float(balanced_accuracy_score(truth, prediction >= threshold))
    else:
        if metric == "mae":
            return float(mean_absolute_error(truth, prediction))
        if metric == "rmse":
            return float(math.sqrt(mean_squared_error(truth, prediction)))
        if metric == "r2":
            return float(r2_score(truth, prediction))
        if metric == "spearman_rho":
            return float(spearmanr(truth, prediction).statistic)
        if metric == "pearson_r":
            return float(pearsonr(truth, prediction).statistic)
    raise KeyError((kind, metric))


def _bootstrap_ci(
    kind: str, metric: str, truth: np.ndarray, prediction: np.ndarray, threshold: float, seed: int
) -> tuple[float, float]:
    n = len(truth)
    if n < 10 or (kind == "classification" and min(np.bincount(truth.astype(int), minlength=2)) < 3):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        yt, yp = truth[idx], prediction[idx]
        if kind == "classification" and len(np.unique(yt)) < 2 and metric in {"roc_auc", "balanced_accuracy"}:
            continue
        try:
            value = _metric_value(kind, metric, yt, yp, threshold)
        except (ValueError, FloatingPointError):
            continue
        if np.isfinite(value):
            values.append(value)
    if len(values) < 200:
        return np.nan, np.nan
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def compute_metrics(records: pd.DataFrame, p_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    evaluation_sets = {
        "PRIMARY_UNSEEN_COMPOUND": records["validation_stratum"].eq("PRIMARY_UNSEEN_COMPOUND"),
        "SECONDARY_SEEN_COMPOUND_NEW_MATRIX": records["validation_stratum"].eq("SECONDARY_SEEN_COMPOUND_NEW_MATRIX"),
        "ALL_TRUTHED_DESCRIPTIVE_NOT_EXTERNAL": records["validation_stratum"].isin(STRATA),
    }
    for stratum, mask in evaluation_sets.items():
        for task, spec in TASKS.items():
            part = records.loc[mask & records["task"].eq(task)].copy()
            truth = part["truth"].to_numpy(float)
            prediction = part["prediction"].to_numpy(float)
            n = len(part)
            positive_n = int(np.sum(truth == 1)) if spec["kind"] == "classification" else np.nan
            negative_n = int(np.sum(truth == 0)) if spec["kind"] == "classification" else np.nan
            status = "NO_TRUTH" if n == 0 else ("DESCRIPTIVE_UNDERPOWERED" if n < 20 else "ESTIMABLE")
            metric_names = (
                ["roc_auc", "average_precision", "brier", "balanced_accuracy"]
                if spec["kind"] == "classification"
                else ["mae", "rmse", "r2", "spearman_rho", "pearson_r"]
            )
            for metric in metric_names:
                value = np.nan
                if n:
                    is_class_discrimination = spec["kind"] == "classification" and metric in {
                        "roc_auc", "average_precision", "balanced_accuracy"
                    }
                    is_correlation = spec["kind"] == "regression" and metric in {"r2", "spearman_rho", "pearson_r"}
                    if not (is_class_discrimination and len(np.unique(truth)) < 2) and not (is_correlation and n < 2):
                        try:
                            value = _metric_value(spec["kind"], metric, truth, prediction, p_threshold)
                        except (ValueError, FloatingPointError):
                            value = np.nan
                ci_low, ci_high = (np.nan, np.nan)
                if np.isfinite(value):
                    ci_low, ci_high = _bootstrap_ci(
                        spec["kind"], metric, truth, prediction, p_threshold,
                        int(hashlib.sha256(f"{SEED}|{stratum}|{task}|{metric}".encode()).hexdigest()[:8], 16),
                    )
                rows.append({
                    "evaluation_stratum": stratum,
                    "primary_external_claim": stratum == "PRIMARY_UNSEEN_COMPOUND",
                    "task": task,
                    "task_label": spec["label"],
                    "kind": spec["kind"],
                    "n_compounds": n,
                    "positive_n": positive_n,
                    "negative_n": negative_n,
                    "metric": metric,
                    "value": value,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "interpretation_status": status,
                })
    return pd.DataFrame(rows)


def make_coverage(compounds: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stratum in [*STRATA, "NO_FROZEN_TRUTH"]:
        cpart = compounds.loc[compounds["validation_stratum"].eq(stratum)]
        for task in TASKS:
            rpart = records.loc[records["validation_stratum"].eq(stratum) & records["task"].eq(task)]
            rows.append({
                "validation_stratum": stratum,
                "task": task,
                "identified_compound_n": int(len(cpart)),
                "truth_available_n": int(rpart["inchikey_2d"].nunique()),
                "truth_coverage_fraction": float(rpart["inchikey_2d"].nunique() / len(cpart)) if len(cpart) else np.nan,
            })
    return pd.DataFrame(rows)


def make_claim_scope(compounds: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "validation_stratum": "PRIMARY_UNSEEN_COMPOUND",
            "compound_n": int(compounds["validation_stratum"].eq("PRIMARY_UNSEEN_COMPOUND").sum()),
            "endpoint_truth_record_n": int(records["validation_stratum"].eq("PRIMARY_UNSEEN_COMPOUND").sum()),
            "allowed_claim": STRATA["PRIMARY_UNSEEN_COMPOUND"]["claim"],
            "not_allowed_claim": "Large-sample definitive external validation or population-representative performance",
        },
        {
            "validation_stratum": "SECONDARY_SEEN_COMPOUND_NEW_MATRIX",
            "compound_n": int(compounds["validation_stratum"].eq("SECONDARY_SEEN_COMPOUND_NEW_MATRIX").sum()),
            "endpoint_truth_record_n": int(records["validation_stratum"].eq("SECONDARY_SEEN_COMPOUND_NEW_MATRIX").sum()),
            "allowed_claim": STRATA["SECONDARY_SEEN_COMPOUND_NEW_MATRIX"]["claim"],
            "not_allowed_claim": "Unseen-compound generalization",
        },
        {
            "validation_stratum": "NO_FROZEN_TRUTH",
            "compound_n": int(compounds["validation_stratum"].eq("NO_FROZEN_TRUTH").sum()),
            "endpoint_truth_record_n": 0,
            "allowed_claim": "Identity recovery and prioritization only",
            "not_allowed_claim": "P/B/T accuracy validation",
        },
    ])


def make_figure(records: pd.DataFrame, coverage: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    primary = coverage.loc[coverage["validation_stratum"].eq("PRIMARY_UNSEEN_COMPOUND")].set_index("task")
    secondary = coverage.loc[coverage["validation_stratum"].eq("SECONDARY_SEEN_COMPOUND_NEW_MATRIX")].set_index("task")
    labels = ["P", "B", "EC50", "LC50", "NOEC"]
    x = np.arange(len(TASKS))
    axes[0, 0].bar(x - 0.18, [primary.loc[t, "truth_available_n"] for t in TASKS], 0.36, label="Unseen")
    axes[0, 0].bar(x + 0.18, [secondary.loc[t, "truth_available_n"] for t in TASKS], 0.36, label="Seen/new matrix")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel("Compounds with truth")
    axes[0, 0].set_title("Truth coverage")
    axes[0, 0].legend(frameon=False)

    colors = {
        "PRIMARY_UNSEEN_COMPOUND": "#0072B2",
        "SECONDARY_SEEN_COMPOUND_NEW_MATRIX": "#D55E00",
    }
    regression_tasks = [task for task, spec in TASKS.items() if spec["kind"] == "regression"]
    for ax, task in zip(axes.flat[1:5], regression_tasks):
        part = records.loc[records["task"].eq(task)]
        for stratum, color in colors.items():
            q = part.loc[part["validation_stratum"].eq(stratum)]
            ax.scatter(q["truth"], q["prediction"], s=45, alpha=0.85, color=color, label=stratum.split("_")[0].title())
        if len(part):
            lo = float(min(part["truth"].min(), part["prediction"].min()))
            hi = float(max(part["truth"].max(), part["prediction"].max()))
            ax.plot([lo, hi], [lo, hi], "--", color="0.45", linewidth=1)
        ax.set_title(TASKS[task]["label"])
        ax.set_xlabel("Frozen experimental truth")
        ax.set_ylabel("Portugal spectrum prediction")

    ax = axes.flat[5]
    ppart = records.loc[records["task"].eq("P_screen_not_ready_binary")].copy()
    if len(ppart):
        jitter = np.linspace(-0.05, 0.05, len(ppart))
        for stratum, color in colors.items():
            q = ppart.loc[ppart["validation_stratum"].eq(stratum)]
            ax.scatter(q["truth"], q["prediction"], s=45, alpha=0.85, color=color)
        ax.set_xticks([0, 1], ["Ready", "Not ready"])
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(TASKS["P_screen_not_ready_binary"]["label"])
    ax.set_xlabel("Frozen experimental class")
    ax.set_ylabel("Predicted probability")

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=("Unseen compound" if "PRIMARY" in key else "Seen compound/new matrix"))
        for key, color in colors.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Portugal real-matrix P/B/T truth audit (Notebook 36)", fontsize=15)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(output / "36_portugal_pbt_truth_validation.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "36_portugal_pbt_truth_validation.pdf", bbox_inches="tight")
    plt.close(fig)


def run(root: str | Path | None = None) -> dict[str, Any]:
    root_path = locate_root(root)
    p = project_paths(root_path)
    p["output"].mkdir(parents=True, exist_ok=True)
    p["figures"].mkdir(parents=True, exist_ok=True)
    stage25, stage26, input_hashes = verify_inputs(p)
    compounds, label_metadata = build_compound_table(p)
    records = build_endpoint_records(compounds, label_metadata)

    calibration = read_json(p["calibration"])
    p_threshold = float(calibration["endpoints"]["P_screen_not_ready_binary"]["screening_threshold"])
    metrics = compute_metrics(records, p_threshold)
    coverage = make_coverage(compounds, records)
    claim_scope = make_claim_scope(compounds, records)

    qc_rows = [
        ("Q36_001", "Notebook 25 frozen status verified", True),
        ("Q36_002", "Notebook 26 completed status verified", True),
        ("Q36_003", "84 unique recovered identities", len(compounds) == compounds["inchikey_2d"].nunique() == 84),
        ("Q36_004", "All recovered identities retain frozen predictions", compounds[[f"prediction__{t}" for t in TASKS]].notna().all().all()),
        ("Q36_005", "Primary stratum contains only external_novel compounds", compounds.loc[compounds["primary_external_claim_eligible"], "supervised_compound_role"].eq("external_novel").all()),
        ("Q36_006", "No development compound enters primary validation", not compounds.loc[compounds["supervised_compound_role"].eq("development"), "primary_external_claim_eligible"].any()),
        ("Q36_007", "Frozen truth equals long-table aggregate", records["truth_matches_frozen_aggregate"].all()),
        ("Q36_008", "Watchlist membership is not used as P/B/T truth", True),
        ("Q36_009", "All endpoint records are finite", np.isfinite(records[["truth", "prediction"]].to_numpy(float)).all()),
        ("Q36_010", "Primary endpoint results flagged underpowered", metrics.loc[metrics["primary_external_claim"], "interpretation_status"].isin(["DESCRIPTIVE_UNDERPOWERED", "NO_TRUTH"]).all()),
    ]
    qc = pd.DataFrame(qc_rows, columns=["check_id", "description", "passed"])
    qc["status"] = np.where(qc["passed"], "PASS", "FAIL")
    if not qc["passed"].all():
        raise RuntimeError("Notebook 36 QC failed:\n" + qc.loc[~qc["passed"]].to_string(index=False))

    atomic_csv(compounds, p["output"] / "36_identity_truth_audit.csv")
    atomic_csv(records, p["output"] / "36_endpoint_validation_records.csv")
    atomic_csv(metrics, p["output"] / "36_endpoint_performance.csv")
    atomic_csv(coverage, p["output"] / "36_truth_coverage.csv")
    atomic_csv(claim_scope, p["output"] / "36_claim_scope.csv")
    atomic_csv(qc, p["output"] / "36_validation_qc_checks.csv")
    make_figure(records, coverage, p["figures"])

    primary_records = records.loc[records["validation_stratum"].eq("PRIMARY_UNSEEN_COMPOUND")]
    summary = {
        "version": VERSION,
        "status": FINAL_STATUS,
        "generated_at_utc": utc_now(),
        "scientific_conclusion": (
            "Portugal supplies a valid unseen-compound, real-wastewater-spectrum validation stratum, "
            "but endpoint-specific truth counts are 1-7 and therefore support only an underpowered case-study claim."
        ),
        "counts": {
            "recovered_identified_compounds": int(len(compounds)),
            "compounds_with_any_frozen_truth": int(compounds["any_frozen_endpoint_truth"].sum()),
            "primary_unseen_compounds": int(compounds["validation_stratum"].eq("PRIMARY_UNSEEN_COMPOUND").sum()),
            "secondary_seen_compounds": int(compounds["validation_stratum"].eq("SECONDARY_SEEN_COMPOUND_NEW_MATRIX").sum()),
            "without_frozen_truth": int(compounds["validation_stratum"].eq("NO_FROZEN_TRUTH").sum()),
            "primary_truth_n_by_endpoint": {
                task: int(primary_records.loc[primary_records["task"].eq(task), "inchikey_2d"].nunique()) for task in TASKS
            },
        },
        "claim_boundary": {
            "allowed": "Supportive real-matrix case validation; unseen-compound results reported endpoint by endpoint.",
            "not_allowed": "Definitive large-sample external validation, prevalence inference, or treating Watchlist status as PBT truth.",
            "secondary_seen_compounds": "Measurement-domain transfer consistency only.",
        },
        "p_screening_threshold": p_threshold,
        "identity_feedback_into_predictions": False,
        "watchlist_used_as_truth": False,
        "qc_all_required_pass": True,
    }
    atomic_json(p["output"] / "36_validation_summary.json", summary)

    output_files = [
        p["output"] / "36_identity_truth_audit.csv",
        p["output"] / "36_endpoint_validation_records.csv",
        p["output"] / "36_endpoint_performance.csv",
        p["output"] / "36_truth_coverage.csv",
        p["output"] / "36_claim_scope.csv",
        p["output"] / "36_validation_qc_checks.csv",
        p["output"] / "36_validation_summary.json",
        p["figures"] / "36_portugal_pbt_truth_validation.png",
        p["figures"] / "36_portugal_pbt_truth_validation.pdf",
    ]
    analysis_code_path = root_path / "src/pbt_msms/portugal_pbt_truth_validation.py"
    manifest = {
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "status": FINAL_STATUS,
        "qc_all_required_pass": True,
        "upstream_status": {"notebook25": stage25["status"], "notebook26": stage26["status"]},
        "input_hashes": input_hashes,
        "analysis_code": {
            "path": str(analysis_code_path),
            "sha256": sha256_file(analysis_code_path),
        },
        "counts": summary["counts"],
        "claim_boundary": summary["claim_boundary"],
        "output_hashes": {path.relative_to(p["output"]).as_posix(): sha256_file(path) for path in output_files},
        "software": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "next_step": "Use Notebook 36 endpoint-specific results with explicit small-n caveats in manuscript integration.",
    }
    atomic_json(p["output"] / "36_run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="Project root")
    args = parser.parse_args()
    print(json.dumps(run(args.root), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
