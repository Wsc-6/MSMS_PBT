"""Notebook 38: external baselines and MS/MS information-gain audit.

This analysis is deliberately post-freeze.  It does not select a new model or
change Notebook 19/20 decisions.  Precursor-only and metadata-only baselines
are fitted on Development with the fixed Notebook 5 specification.  ChemVista
endpoint labels are opened only after all new baseline predictions have been
written to an unlabeled, hashed artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from pbt_msms.chemvista_external import (
    N_FOLDS,
    TASKS,
    aggregate_spectra,
    load_binned_external,
    locate_root,
    sample_development_rows,
    stable_seed,
)


VERSION = "PBT_MSMS_CHEMVISTA_INFORMATION_GAIN_38_V1"
PIPELINES = [
    "Development null",
    "Precursor m/z only",
    "Metadata only",
    "Binned MS/MS",
    "Frozen final model",
]
COHORTS = ["selected_novel", "natural_novel", "novel_all", "transfer"]
TASK_LABELS = {
    "P_screen_not_ready_binary": "P screen",
    "B_LOGBCF_REGRESSION": "B (log BCF)",
    "T_EC50_MG_L_REGRESSION": "T EC50",
    "T_LC50_MG_L_REGRESSION": "T LC50",
    "T_NOEC_MG_L_REGRESSION": "T NOEC",
}
COMPARISONS = [
    ("Precursor m/z only", "Development null"),
    ("Metadata only", "Precursor m/z only"),
    ("Binned MS/MS", "Metadata only"),
    ("Frozen final model", "Metadata only"),
    ("Frozen final model", "Development null"),
]


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
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    compression = "gzip" if path.name.endswith(".gz") else None
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def analysis_paths(root: Path) -> dict[str, Path]:
    base = root / "outputs/pbt_msms_v1"
    ext = base / "chemvista_external_20_v1"
    simple = base / "simple_spectrum_baselines_v1"
    out = base / "chemvista_information_gain_38_v1"
    return {
        "base": base,
        "ext": ext,
        "simple": simple,
        "out": out,
        "figures": out / "figures",
        "development": base / "split_v1/development_compounds.csv",
        "dev_index": simple / "spectrum_feature_row_index.csv",
        "dev_metadata": simple / "spectrum_metadata.npy",
        "simple_manifest": simple / "simple_spectrum_run_manifest.json",
        "ext_index": ext / "20_chemvista_spectrum_index.csv.gz",
        "ext_roster": ext / "20_chemvista_compound_roster.csv.gz",
        "ext_binned": ext / "20_chemvista_binned_features.npz",
        "candidate": ext / "20_all_candidate_predictions.csv.gz",
        "final": ext / "20_final_selected_model_predictions.csv.gz",
        "ext_metrics": ext / "20_external_metrics.csv",
        "ext_manifest": ext / "20_run_manifest.json",
        "new_fold": out / "38_new_baseline_fold_predictions_unlabeled.csv.gz",
        "new_predictions": out / "38_new_baseline_predictions_unlabeled.csv.gz",
        "prediction_freeze": out / "38_prediction_freeze_before_label_access.json",
        "label_audit": out / "38_terminal_label_access_audit.json",
        "predictions": out / "38_external_baseline_predictions.csv.gz",
        "metrics": out / "38_external_baseline_metrics.csv",
        "gain": out / "38_paired_information_gain.csv",
        "calibration": out / "38_p_calibration_table.csv",
        "qc": out / "38_qc_checks.csv",
        "summary": out / "38_manuscript_result_summary.md",
        "manifest": out / "38_run_manifest.json",
        "figure_png": out / "figures/38_chemvista_msms_information_gain.png",
        "figure_pdf": out / "figures/38_chemvista_msms_information_gain.pdf",
    }


def verify_upstream(p: dict[str, Path]) -> dict[str, str]:
    required = [
        "development", "dev_index", "dev_metadata", "simple_manifest",
        "ext_index", "ext_roster", "ext_binned", "candidate", "final",
        "ext_metrics", "ext_manifest",
    ]
    missing = [name for name in required if not p[name].exists()]
    if missing:
        raise FileNotFoundError(f"Missing required upstream artifacts: {missing}")

    simple_manifest = read_json(p["simple_manifest"])
    external_manifest = read_json(p["ext_manifest"])
    if simple_manifest.get("qc_status") != "PASS":
        raise RuntimeError("Notebook 5 simple-baseline manifest is not PASS")
    if simple_manifest.get("external_test_unlocked") is not False:
        raise RuntimeError("Notebook 5 baselines were not frozen before external unlock")
    if external_manifest.get("status") != "READY_FOR_NOTEBOOK_21" or not external_manifest.get("qc_all_pass"):
        raise RuntimeError("Notebook 20 external evaluation is not frozen PASS")

    for filename, expected in external_manifest.get("output_hashes", {}).items():
        path = p["ext"] / filename
        if path.exists() and sha256_file(path) != expected:
            raise RuntimeError(f"Notebook 20 hash mismatch: {filename}")

    return {name: sha256_file(p[name]) for name in required}


def fit_new_unlabeled_baselines(p: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit fixed Development-only baselines and predict ChemVista without labels."""
    development = pd.read_csv(p["development"], low_memory=False)
    dev_index = pd.read_csv(p["dev_index"], low_memory=False)
    ext_index = pd.read_csv(p["ext_index"], low_memory=False)
    roster = pd.read_csv(p["ext_roster"], low_memory=False)
    dev_metadata = np.load(p["dev_metadata"], allow_pickle=False)
    _, _, ext_metadata = load_binned_external(p["ext_binned"])

    if dev_metadata.shape[1] != 6 or ext_metadata.shape[1] != 6:
        raise RuntimeError("Metadata schema differs from frozen Notebook 5 six-column definition")
    if not np.array_equal(ext_index["feature_row"].to_numpy(int), np.arange(len(ext_index))):
        raise RuntimeError("ChemVista feature_row is not aligned to the frozen feature matrix")

    label_frame = development.set_index("inchikey_2d")
    spectrum_keys = set(dev_index["inchikey_2d"].astype(str))
    external_keys = sorted(roster["inchikey_2d"].astype(str).unique())
    fold_rows: list[dict[str, Any]] = []

    for task, spec in TASKS.items():
        available = development.loc[
            pd.to_numeric(development[spec["column"]], errors="coerce").notna()
            & development["inchikey_2d"].astype(str).isin(spectrum_keys)
        ].copy()
        for fold in range(N_FOLDS):
            train_compounds = available.loc[
                available["cv_fold"].astype(int).ne(fold), "inchikey_2d"
            ].astype(str).tolist()
            train_rows = sample_development_rows(dev_index, train_compounds, task, fold)
            train_keys = dev_index.iloc[train_rows]["inchikey_2d"].astype(str).tolist()
            y_train = np.asarray([label_frame.at[key, spec["column"]] for key in train_keys], float)
            unique_y = available.loc[
                available["cv_fold"].astype(int).ne(fold), spec["column"]
            ].to_numpy(float)

            null_value = float(np.mean(unique_y)) if spec["kind"] == "classification" else float(np.median(unique_y))
            for key in external_keys:
                fold_rows.append({
                    "pipeline": "Development null", "task": task, "kind": spec["kind"],
                    "outer_fold": fold, "inchikey_2d": key, "y_score": null_value,
                })

            for route, columns in [("Precursor m/z only", [0]), ("Metadata only", list(range(6)))]:
                scaler = StandardScaler().fit(dev_metadata[train_rows][:, columns])
                x_train = scaler.transform(dev_metadata[train_rows][:, columns])
                x_external = scaler.transform(ext_metadata[:, columns])
                if spec["kind"] == "classification":
                    model = LogisticRegression(
                        C=1.0,
                        solver="liblinear",
                        max_iter=4000,
                        random_state=stable_seed(42, route, task, fold),
                    )
                    model.fit(x_train, y_train.astype(int))
                    spectrum_values = model.predict_proba(x_external)[:, 1]
                else:
                    model = Ridge(alpha=10.0, solver="lsqr", tol=1e-4, max_iter=3000)
                    model.fit(x_train, y_train)
                    spectrum_values = model.predict(x_external)
                compound_values = aggregate_spectra(spectrum_values, ext_index, "median")
                fold_rows.extend(
                    {
                        "pipeline": route, "task": task, "kind": spec["kind"],
                        "outer_fold": fold, "inchikey_2d": key, "y_score": float(value),
                    }
                    for key, value in compound_values.items()
                )

    fold_frame = pd.DataFrame(fold_rows).sort_values(
        ["pipeline", "task", "outer_fold", "inchikey_2d"]
    ).reset_index(drop=True)
    predictions = (
        fold_frame.groupby(["pipeline", "task", "kind", "inchikey_2d"], as_index=False)["y_score"]
        .mean()
        .sort_values(["pipeline", "task", "inchikey_2d"])
        .reset_index(drop=True)
    )
    atomic_csv(fold_frame, p["new_fold"])
    atomic_csv(predictions, p["new_predictions"])
    atomic_json(p["prediction_freeze"], {
        "version": VERSION,
        "frozen_at_utc": utc_now(),
        "chemvista_endpoint_labels_opened": False,
        "external_labels_used_for_fitting_or_selection": False,
        "model_specification": {
            "source": "Notebook 5 frozen simple-baseline specification",
            "classification": "StandardScaler + LogisticRegression(C=1, liblinear)",
            "regression": "StandardScaler + Ridge(alpha=10, lsqr)",
            "training": "Development only; five frozen outer-fold deployment models",
            "compound_weighting": "one spectrum per compound per repeat; five deterministic repeats",
            "external_spectrum_aggregation": "median, then mean across five folds",
        },
        "artifacts": {
            p["new_fold"].name: sha256_file(p["new_fold"]),
            p["new_predictions"].name: sha256_file(p["new_predictions"]),
        },
    })
    return fold_frame, predictions


def load_terminal_labels(p: dict[str, Path], roster: pd.DataFrame) -> pd.DataFrame:
    parts = []
    hashes = {}
    for role, filename in {
        "transfer": "chemvista_transfer.csv",
        "selected_novel": "chemvista_selected_novel.csv",
        "natural_novel": "chemvista_natural_novel.csv",
    }.items():
        path = p["base"] / "split_v1" / filename
        keep = ["inchikey_2d", *[spec["column"] for spec in TASKS.values()]]
        part = pd.read_csv(path, low_memory=False)[keep].copy()
        part["evaluation_role"] = role
        parts.append(part)
        hashes[filename] = sha256_file(path)
    labels = pd.concat(parts, ignore_index=True)
    observed = labels[["inchikey_2d", "evaluation_role"]].sort_values("inchikey_2d").reset_index(drop=True)
    expected = roster[["inchikey_2d", "evaluation_role"]].sort_values("inchikey_2d").reset_index(drop=True)
    if labels["inchikey_2d"].duplicated().any() or not observed.equals(expected):
        raise RuntimeError("Terminal labels do not match the frozen ChemVista roster")
    atomic_json(p["label_audit"], {
        "version": VERSION,
        "opened_at_utc": utc_now(),
        "opened_after_new_predictions_frozen": True,
        "prediction_freeze_sha256": sha256_file(p["prediction_freeze"]),
        "external_labels_used_for_fitting_selection_or_calibration": False,
        "label_file_hashes": hashes,
    })
    return labels


def combine_predictions(p: dict[str, Path], new_predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    candidate = pd.read_csv(p["candidate"], low_memory=False)
    binned = candidate.loc[candidate["pipeline"].eq("Binned MS/MS")].copy()
    final = pd.read_csv(p["final"], low_memory=False)
    final = final[["task", "kind", "inchikey_2d", "y_score_calibrated"]].rename(
        columns={"y_score_calibrated": "y_score"}
    )
    final["pipeline"] = "Frozen final model"
    combined = pd.concat(
        [new_predictions, binned[["pipeline", "task", "kind", "inchikey_2d", "y_score"]], final],
        ignore_index=True,
    )
    label_long = []
    for task, spec in TASKS.items():
        part = labels[["inchikey_2d", "evaluation_role", spec["column"]]].copy()
        part["task"] = task
        part = part.rename(columns={spec["column"]: "y_true"})
        label_long.append(part)
    label_long_frame = pd.concat(label_long, ignore_index=True)
    combined = combined.merge(label_long_frame, on=["inchikey_2d", "task"], how="left", validate="many_to_one")
    combined = combined.sort_values(["pipeline", "task", "inchikey_2d"]).reset_index(drop=True)
    return combined


def cohort_mask(frame: pd.DataFrame, cohort: str) -> pd.Series:
    if cohort == "novel_all":
        return frame["evaluation_role"].isin(["selected_novel", "natural_novel"])
    return frame["evaluation_role"].eq(cohort)


def expected_calibration_error(y: np.ndarray, score: np.ndarray, n_bins: int = 5) -> float:
    if len(y) == 0:
        return np.nan
    unique_scores = np.unique(score).size
    if unique_scores <= 1:
        bins = np.zeros(len(score), dtype=int)
    else:
        bins = np.asarray(pd.qcut(
            score,
            q=min(n_bins, unique_scores),
            labels=False,
            duplicates="drop",
        ), dtype=int)
    return float(sum(np.mean(bins == i) * abs(np.mean(y[bins == i]) - np.mean(score[bins == i]))
                     for i in np.unique(bins)))


def calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in COHORTS:
        for pipeline in PIPELINES:
            for task, spec in TASKS.items():
                all_labels = predictions.loc[
                    cohort_mask(predictions, cohort) & predictions["task"].eq(task),
                    ["inchikey_2d", "y_true"],
                ].drop_duplicates("inchikey_2d").dropna(subset=["y_true"])
                frame = predictions.loc[
                    cohort_mask(predictions, cohort)
                    & predictions["pipeline"].eq(pipeline)
                    & predictions["task"].eq(task)
                ].dropna(subset=["y_true", "y_score"])
                y = frame["y_true"].to_numpy(float)
                score = frame["y_score"].to_numpy(float)
                common = {
                    "cohort": cohort, "pipeline": pipeline, "task": task,
                    "kind": spec["kind"], "n": len(frame),
                    "n_labelled": len(all_labels),
                    "coverage": len(frame) / len(all_labels) if len(all_labels) else np.nan,
                }
                if spec["kind"] == "classification":
                    both = np.unique(y).size == 2
                    values = {
                        "primary_metric": "pr_auc",
                        "primary_value": float(average_precision_score(y, score)) if both else np.nan,
                        "pr_auc": float(average_precision_score(y, score)) if both else np.nan,
                        "roc_auc": float(roc_auc_score(y, score)) if both else np.nan,
                        "brier": float(brier_score_loss(y, score)),
                        "ece_5bin": expected_calibration_error(y, score),
                        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, score >= 0.5)) if both else np.nan,
                        "mae": np.nan, "rmse": np.nan, "r2": np.nan, "spearman": np.nan,
                    }
                else:
                    rho = spearmanr(y, score).statistic if len(y) >= 3 and np.unique(score).size > 1 else np.nan
                    values = {
                        "primary_metric": "mae",
                        "primary_value": float(mean_absolute_error(y, score)),
                        "pr_auc": np.nan, "roc_auc": np.nan, "brier": np.nan,
                        "ece_5bin": np.nan, "balanced_accuracy_at_0_5": np.nan,
                        "mae": float(mean_absolute_error(y, score)),
                        "rmse": float(np.sqrt(mean_squared_error(y, score))),
                        "r2": float(r2_score(y, score)) if len(y) >= 2 else np.nan,
                        "spearman": float(rho),
                    }
                rows.append({**common, **values})
    return pd.DataFrame(rows)


def metric_improvement(y: np.ndarray, candidate: np.ndarray, reference: np.ndarray, metric: str) -> float:
    if metric == "pr_auc":
        if np.unique(y).size < 2:
            return np.nan
        return float(average_precision_score(y, candidate) - average_precision_score(y, reference))
    if metric == "brier_improvement":
        return float(brier_score_loss(y, reference) - brier_score_loss(y, candidate))
    if metric == "mae_improvement":
        return float(mean_absolute_error(y, reference) - mean_absolute_error(y, candidate))
    raise KeyError(metric)


def paired_information_gain(predictions: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows = []
    for cohort in COHORTS:
        for task, spec in TASKS.items():
            metrics = ["pr_auc", "brier_improvement"] if spec["kind"] == "classification" else ["mae_improvement"]
            base = predictions.loc[
                cohort_mask(predictions, cohort) & predictions["task"].eq(task),
                ["inchikey_2d", "pipeline", "y_true", "y_score"],
            ].dropna(subset=["y_true", "y_score"])
            wide = base.pivot(index=["inchikey_2d", "y_true"], columns="pipeline", values="y_score").reset_index()
            for candidate, reference in COMPARISONS:
                paired = wide.dropna(subset=[candidate, reference])
                y = paired["y_true"].to_numpy(float)
                candidate_score = paired[candidate].to_numpy(float)
                reference_score = paired[reference].to_numpy(float)
                for metric in metrics:
                    observed = metric_improvement(y, candidate_score, reference_score, metric)
                    rng = np.random.default_rng(stable_seed(VERSION, cohort, task, candidate, reference, metric))
                    boot = []
                    for _ in range(n_bootstrap):
                        idx = rng.integers(0, len(paired), len(paired))
                        value = metric_improvement(y[idx], candidate_score[idx], reference_score[idx], metric)
                        if np.isfinite(value):
                            boot.append(value)
                    array = np.asarray(boot, float)
                    rows.append({
                        "cohort": cohort, "task": task, "kind": spec["kind"],
                        "candidate": candidate, "reference": reference, "metric": metric,
                        "direction": "positive means candidate is better", "n": len(paired),
                        "observed_improvement": observed,
                        "ci_low": float(np.quantile(array, 0.025)) if len(array) else np.nan,
                        "ci_high": float(np.quantile(array, 0.975)) if len(array) else np.nan,
                        "bootstrap_probability_improvement": float(np.mean(array > 0)) if len(array) else np.nan,
                        "one_sided_p": float((1 + np.sum(array <= 0)) / (1 + len(array))) if len(array) else np.nan,
                        "valid_bootstrap": len(array),
                    })
    return pd.DataFrame(rows)


def calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    task = "P_screen_not_ready_binary"
    rows = []
    for pipeline in PIPELINES:
        frame = predictions.loc[
            cohort_mask(predictions, "novel_all")
            & predictions["task"].eq(task)
            & predictions["pipeline"].eq(pipeline)
        ].dropna(subset=["y_true", "y_score"]).copy()
        unique_scores = frame["y_score"].nunique()
        if unique_scores <= 1:
            frame["calibration_bin"] = 0
        else:
            frame["calibration_bin"] = pd.qcut(
                frame["y_score"], q=min(5, unique_scores), labels=False, duplicates="drop"
            )
        grouped = frame.groupby("calibration_bin", observed=True)
        for bin_id, part in grouped:
            rows.append({
                "cohort": "novel_all", "pipeline": pipeline, "calibration_bin": int(bin_id),
                "n": len(part), "mean_predicted_probability": part["y_score"].mean(),
                "observed_positive_fraction": part["y_true"].mean(),
                "min_probability": part["y_score"].min(), "max_probability": part["y_score"].max(),
            })
    return pd.DataFrame(rows)


def make_figure(metrics: pd.DataFrame, gain: pd.DataFrame, calibration: pd.DataFrame, p: dict[str, Path]) -> None:
    colors = {
        "Precursor m/z only": "#9c9c9c", "Metadata only": "#d68a34",
        "Binned MS/MS": "#2f7d32", "Frozen final model": "#315da8",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    ax = axes[0]
    novel = metrics.loc[metrics["cohort"].eq("novel_all")]
    null_map = novel.loc[novel["pipeline"].eq("Development null")].set_index("task")["primary_value"]
    x = np.arange(len(TASKS))
    width = 0.19
    routes = ["Precursor m/z only", "Metadata only", "Binned MS/MS", "Frozen final model"]
    for offset, route in enumerate(routes):
        values = []
        for task, spec in TASKS.items():
            value = novel.loc[(novel["pipeline"] == route) & (novel["task"] == task), "primary_value"].iloc[0]
            null = null_map[task]
            values.append(value - null if spec["kind"] == "classification" else null - value)
        ax.bar(x + (offset - 1.5) * width, values, width, label=route, color=colors[route])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [TASK_LABELS[t] for t in TASKS], rotation=35, ha="right")
    ax.set_ylabel("Improvement over Development null\n(PR-AUC gain or MAE reduction)")
    ax.set_title("A  Absolute external utility")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    subset = gain.loc[
        gain["cohort"].eq("novel_all")
        & gain["reference"].eq("Metadata only")
        & gain["candidate"].isin(["Binned MS/MS", "Frozen final model"])
        & gain["metric"].isin(["pr_auc", "mae_improvement"])
    ].copy()
    for j, route in enumerate(["Binned MS/MS", "Frozen final model"]):
        part = subset.loc[subset["candidate"].eq(route)].set_index("task").reindex(TASKS)
        y = np.arange(len(TASKS)) + (j - 0.5) * 0.18
        center = part["observed_improvement"].to_numpy(float)
        low = part["ci_low"].to_numpy(float)
        high = part["ci_high"].to_numpy(float)
        ax.errorbar(center, y, xerr=np.vstack([center - low, high - center]), fmt="o", capsize=3,
                    color=colors[route], label=route)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(np.arange(len(TASKS)), [TASK_LABELS[t] for t in TASKS])
    ax.invert_yaxis()
    ax.set_xlabel("Paired improvement over metadata-only\n(positive favors MS/MS)")
    ax.set_title("B  MS/MS information gain")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
    for route in PIPELINES:
        part = calibration.loc[calibration["pipeline"].eq(route)]
        color = "#bbbbbb" if route == "Development null" else colors.get(route, "black")
        ax.plot(part["mean_predicted_probability"], part["observed_positive_fraction"], "o-",
                color=color, label=route, ms=4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed P-screen fraction")
    ax.set_title("C  External P calibration")
    ax.legend(frameon=False, fontsize=7)

    fig.suptitle("ChemVista novel compounds: frozen external baselines and MS/MS information gain", fontsize=13)
    fig.savefig(p["figure_png"], dpi=300, bbox_inches="tight")
    fig.savefig(p["figure_pdf"], bbox_inches="tight")
    plt.close(fig)


def write_summary(metrics: pd.DataFrame, gain: pd.DataFrame, p: dict[str, Path]) -> None:
    novel = metrics.loc[metrics["cohort"].eq("novel_all")]
    key_gain = gain.loc[
        gain["cohort"].eq("novel_all")
        & gain["reference"].eq("Metadata only")
        & gain["candidate"].isin(["Binned MS/MS", "Frozen final model"])
        & gain["metric"].isin(["pr_auc", "mae_improvement"])
    ]
    lines = [
        "# Notebook 38: ChemVista external baselines and MS/MS information gain",
        "",
        "Primary population: `novel_all` (selected-novel + natural-novel compounds). Positive paired deltas always favor the candidate route.",
        "",
        "## Absolute external performance",
        "",
        "| Endpoint | Route | n | Primary metric | Value | Coverage |",
        "|---|---|---:|---|---:|---:|",
    ]
    for task in TASKS:
        for route in PIPELINES:
            row = novel.loc[(novel["task"] == task) & (novel["pipeline"] == route)].iloc[0]
            lines.append(
                f"| {TASK_LABELS[task]} | {route} | {int(row['n'])} | {row['primary_metric']} | "
                f"{row['primary_value']:.3f} | {row['coverage']:.3f} |"
            )
    lines.extend([
        "", "## Paired MS/MS gain over metadata-only", "",
        "| Endpoint | Candidate | Improvement | 95% paired bootstrap CI | Interpretation |",
        "|---|---|---:|---:|---|",
    ])
    for task in TASKS:
        for route in ["Binned MS/MS", "Frozen final model"]:
            row = key_gain.loc[(key_gain["task"] == task) & (key_gain["candidate"] == route)].iloc[0]
            low, high = row["ci_low"], row["ci_high"]
            if low > 0:
                interpretation = "supported positive gain"
            elif high < 0:
                interpretation = "candidate underperformed metadata-only"
            else:
                interpretation = "uncertain; CI crosses zero"
            lines.append(
                f"| {TASK_LABELS[task]} | {route} | {row['observed_improvement']:.3f} | "
                f"[{low:.3f}, {high:.3f}] | {interpretation} |"
            )
    lines.extend([
        "", "## P-endpoint probability calibration", "",
        "| Route | Brier score | ECE (5 bins) |",
        "|---|---:|---:|",
    ])
    for route in PIPELINES:
        row = novel.loc[
            (novel["task"] == "P_screen_not_ready_binary") & (novel["pipeline"] == route)
        ].iloc[0]
        lines.append(f"| {route} | {row['brier']:.3f} | {row['ece_5bin']:.3f} |")
    p_calibration_gain = gain.loc[
        gain["cohort"].eq("novel_all")
        & gain["task"].eq("P_screen_not_ready_binary")
        & gain["reference"].eq("Metadata only")
        & gain["candidate"].isin(["Binned MS/MS", "Frozen final model"])
        & gain["metric"].eq("brier_improvement")
    ]
    lines.extend(["", "Paired Brier improvement over metadata-only (positive is better):", ""])
    for route in ["Binned MS/MS", "Frozen final model"]:
        row = p_calibration_gain.loc[p_calibration_gain["candidate"].eq(route)].iloc[0]
        lines.append(
            f"- {route}: {row['observed_improvement']:.3f} "
            f"(95% CI {row['ci_low']:.3f} to {row['ci_high']:.3f})."
        )
    lines.extend([
        "",
        "## Claim boundary",
        "",
        "This analysis tests whether fragment-level MS/MS carries predictive information beyond precursor mass and acquisition metadata. "
        "It does not claim universal superiority over every structure model, nor does it identify an unknown compound. "
        "Endpoint-specific gains with paired uncertainty should be reported exactly as observed.",
        "",
        "ChemVista labels were not used to fit, select, calibrate, or tune any route in Notebook 38.",
    ])
    p["summary"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_qc(
    p: dict[str, Path], predictions: pd.DataFrame, metrics: pd.DataFrame,
    gain: pd.DataFrame, upstream_hashes: dict[str, str],
) -> pd.DataFrame:
    expected_rows = len(PIPELINES) * len(TASKS) * predictions["inchikey_2d"].nunique()
    checks = [
        ("prediction_row_count", len(predictions) == expected_rows, f"{len(predictions)} / {expected_rows}"),
        ("pipeline_set_exact", set(predictions["pipeline"]) == set(PIPELINES), ", ".join(sorted(predictions["pipeline"].unique()))),
        ("no_duplicate_prediction_keys", not predictions.duplicated(["pipeline", "task", "inchikey_2d"]).any(), "pipeline-task-compound"),
        ("all_scores_finite", np.isfinite(predictions["y_score"].to_numpy(float)).all(), "all routes"),
        ("classification_scores_in_unit_interval",
         predictions.loc[predictions["kind"].eq("classification"), "y_score"].between(0, 1).all(), "P route scores"),
        ("new_prediction_freeze_precedes_label_access", p["prediction_freeze"].exists() and p["label_audit"].exists(), "audited artifacts"),
        ("new_unlabeled_file_contains_no_labels",
         not any(c.startswith("label__") or c == "y_true" for c in pd.read_csv(p["new_predictions"], nrows=1).columns), "no endpoint labels"),
        ("complete_prediction_coverage", bool(np.isclose(metrics["coverage"].dropna(), 1.0).all()), "all cohorts/routes/tasks"),
        ("paired_bootstrap_generated", len(gain) > 0 and gain["valid_bootstrap"].gt(0).all(), f"rows={len(gain)}"),
        ("upstream_hashes_unchanged", all(sha256_file(p[name]) == value for name, value in upstream_hashes.items()), "Notebook 5/20 inputs"),
    ]
    frame = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    atomic_csv(frame, p["qc"])
    if not frame["passed"].all():
        failures = frame.loc[~frame["passed"], "check"].tolist()
        raise RuntimeError(f"Notebook 38 QC failed: {failures}")
    return frame


def run_analysis(root: str | Path | None = None, n_bootstrap: int = 1000) -> dict[str, Any]:
    root_path = locate_root(root)
    p = analysis_paths(root_path)
    p["out"].mkdir(parents=True, exist_ok=True)
    p["figures"].mkdir(parents=True, exist_ok=True)
    upstream_hashes = verify_upstream(p)

    _, new_predictions = fit_new_unlabeled_baselines(p)
    roster = pd.read_csv(p["ext_roster"], low_memory=False)
    labels = load_terminal_labels(p, roster)
    predictions = combine_predictions(p, new_predictions, labels)
    metrics = calculate_metrics(predictions)
    gain = paired_information_gain(predictions, n_bootstrap=n_bootstrap)
    calibration = calibration_table(predictions)

    atomic_csv(predictions, p["predictions"])
    atomic_csv(metrics, p["metrics"])
    atomic_csv(gain, p["gain"])
    atomic_csv(calibration, p["calibration"])
    make_figure(metrics, gain, calibration, p)
    write_summary(metrics, gain, p)
    qc = run_qc(p, predictions, metrics, gain, upstream_hashes)

    outputs = [
        p["new_fold"], p["new_predictions"], p["predictions"], p["metrics"], p["gain"], p["calibration"], p["qc"],
        p["summary"], p["figure_png"], p["figure_pdf"], p["prediction_freeze"], p["label_audit"],
    ]
    manifest = {
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "scientific_question": "Does fragment-level MS/MS add external predictive information beyond precursor m/z and acquisition metadata?",
        "primary_cohort": "novel_all",
        "n_bootstrap": n_bootstrap,
        "external_labels_used_for_fitting_selection_or_calibration": False,
        "new_models": ["Development null", "Precursor m/z only", "Metadata only"],
        "reused_frozen_routes": ["Binned MS/MS", "Frozen final model"],
        "qc_all_pass": bool(qc["passed"].all()),
        "upstream_hashes": upstream_hashes,
        "output_hashes": {path.name: sha256_file(path) for path in outputs},
        "software": {"python": sys.version, "python_executable": sys.executable},
    }
    atomic_json(p["manifest"], manifest)
    return {
        "root": root_path,
        "output_dir": p["out"],
        "manifest": manifest,
        "metrics": metrics,
        "gain": gain,
        "calibration": calibration,
        "qc": qc,
        "figure": p["figure_png"],
        "summary": p["summary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    result = run_analysis(args.root, n_bootstrap=args.bootstrap)
    print(json.dumps({
        "status": result["manifest"]["status"],
        "output_dir": str(result["output_dir"]),
        "figure": str(result["figure"]),
        "summary": str(result["summary"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
