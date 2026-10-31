"""Presubmission matched-control and generalization robustness audits.

This module is deliberately downstream of the frozen initial-submission
pipeline.  It never changes model selection, folds, labels, or any artifact
from Notebooks 01--39.  New models are non-fragment controls trained only on
Development compounds.  External labels are joined only after the new
external predictions have been written to an unlabeled freeze artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pbt_msms.chemvista_external import (
    N_FOLDS,
    TASKS,
    aggregate_spectra,
    locate_root,
    sample_development_rows,
    stable_seed,
)


VERSION = "PBT_MSMS_PRESUBMISSION_MODEL_AUDIT_40_V1_2"
SEED_NAMESPACE = "PBT_MSMS_PRESUBMISSION_MODEL_AUDIT_40_V1_2"
OUTPUT_SUBDIR = Path("outputs/pbt_msms_v1/presubmission_robustness_v1/model_confounding_audit_40")
PIPELINE_FINAL = "Frozen final model"
PIPELINE_METADATA = "Original six-field metadata"
ROUTES = [
    "Numeric nonfragment HGB",
    "Numeric + adduct/instrument HGB",
    "Matched nonfragment HGB",
    "Provenance-only HGB",
    "Matched nonfragment linear",
]
PRIMARY_MATCHED_ROUTE = "Matched nonfragment HGB"
TASK_LABELS = {
    "P_screen_not_ready_binary": "P screen",
    "B_LOGBCF_REGRESSION": "B (log BCF)",
    "T_EC50_MG_L_REGRESSION": "EC50",
    "T_LC50_MG_L_REGRESSION": "LC50",
    "T_NOEC_MG_L_REGRESSION": "NOEC",
}
TOXICITY_TASKS = {
    "T_EC50_MG_L_REGRESSION",
    "T_LC50_MG_L_REGRESSION",
    "T_NOEC_MG_L_REGRESSION",
}
SIMILARITY_BINS = [-np.inf, 0.30, 0.50, 0.70, np.inf]
SIMILARITY_LABELS = ["<0.30", "0.30-<0.50", "0.50-<0.70", ">=0.70"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    compression = "gzip" if path.name.endswith(".gz") else None
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def paths(root: Path) -> dict[str, Path]:
    base = root / "outputs/pbt_msms_v1"
    out = root / OUTPUT_SUBDIR
    return {
        "root": root,
        "base": base,
        "out": out,
        "figures": out / "figures",
        "development": base / "split_v1/development_compounds.csv",
        "development_applicability": base / "split_v1/development_cv_applicability.csv",
        "novel_applicability": base / "split_v1/novel_structure_applicability.csv",
        "development_index": base / "transformer_data_v1/real_development_spectrum_index.csv.gz",
        "development_tokens": base / "transformer_data_v1/token_caches/real_development_tokens.npz",
        "vocab": base / "transformer_data_v1/transformer_categorical_vocab.json",
        "selected_oof": base / "comparison_19_v1/19_selected_model_oof_calibrated.csv.gz",
        "all_oof": base / "comparison_19_v1/19_all_pipeline_oof_long.csv.gz",
        "direct_common_oof": base / "comparison_19_v1/19_direct_candidate_common_oof.csv.gz",
        "registry": base / "comparison_19_v1/19_final_model_registry.csv",
        "calibration": base / "comparison_19_v1/19_deployment_calibration.json",
        "external_index": base / "chemvista_external_20_v1/20_chemvista_spectrum_index.csv.gz",
        "external_tokens": base / "chemvista_external_20_v1/20_chemvista_transformer_tokens.npz",
        "external_final": base / "chemvista_external_20_v1/20_final_selected_model_predictions.csv.gz",
        "external_38_predictions": base / "chemvista_information_gain_38_v1/38_external_baseline_predictions.csv.gz",
        "strict_records": root / "性质数据/Cleaned_datasets/01_integrated_properties/07b_PBT_strict_MSMS_modeling_candidates.csv",
        "evidence_records": root / "性质数据/Cleaned_datasets/01_integrated_properties/04_PBT_evidence_records_long.csv",
        "label_reliability": root / "论文绘图框架_最终版/tables/SI/Figure2_InformationBoundary/TableS15_label_reliability.csv",
        "analysis_plan": out / "40_analysis_plan.json",
        "unlabeled_external": out / "40_new_external_predictions_unlabeled.csv.gz",
        "prediction_freeze": out / "40_prediction_freeze_before_label_join.json",
        "predictions": out / "40_predictions_long.csv.gz",
        "metrics": out / "40_metrics.csv",
        "paired": out / "40_paired_information_gain.csv",
        "source_strata": out / "40_source_stratified_gain.csv",
        "similarity_strata": out / "40_similarity_stratified_gain.csv",
        "toxicity_strata": out / "40_toxicity_species_duration_sensitivity.csv",
        "toxicity_counts": out / "40_toxicity_stratum_counts.csv",
        "utility": out / "40_screening_utility.csv",
        "effect_context": out / "40_effect_size_context.csv",
        "selection_stability": out / "40_pipeline_selection_stability.csv",
        "qc": out / "40_qc_checks.csv",
        "summary": out / "40_manuscript_result_summary.md",
        "manifest": out / "40_run_manifest.json",
        "figure_png": out / "figures/40_matched_control_audit.png",
        "figure_pdf": out / "figures/40_matched_control_audit.pdf",
    }


def analysis_plan() -> dict[str, Any]:
    return {
        "version": VERSION,
        "status": "FIXED_BEFORE_EXECUTION",
        "revision_note": (
            "V1.1 corrects numerical pairing to use compound identifiers rather than "
            "float-serialized labels and reads label-IQR context from frozen Table S15."
        ),
        "purpose": "Presubmission post-freeze sensitivity analysis; not model reselection.",
        "frozen_folds": 5,
        "primary_matched_control": PRIMARY_MATCHED_ROUTE,
        "nonfragment_inputs": {
            "numeric": [
                "precursor_mz_norm", "polarity", "collision_energy_norm",
                "collision_energy_missing", "log_peak_count_norm", "is_synthetic",
                "sparse_lt3_peaks", "instrument_missing",
            ],
            "categorical": ["adduct", "instrument_category", "source_library"],
        },
        "model_policy": {
            "training_data": "Development only",
            "folds": "unchanged structure-cluster outer folds",
            "compound_weighting": "five deterministic one-spectrum-per-compound samples",
            "hyperparameter_tuning": "none; fixed in code",
            "external_use": "prediction only; no external-label fitting, tuning, or selection",
        },
        "primary_comparison": "Frozen final model versus Matched nonfragment HGB",
        "secondary_comparison": "Frozen final model versus original six-field metadata",
        "multiplicity_families": "five endpoints within each cohort and model comparison",
        "similarity_bins": SIMILARITY_LABELS,
        "toxicity_strata": [
            "EC50 crustacean 48 h", "EC50 algae 72 h", "EC50 algae 96 h",
            "LC50 fish 96 h", "NOEC fish 21 d", "NOEC crustacean 21 d",
        ],
        "claim_boundary": (
            "Matched controls test whether the selected spectrum pipeline adds predictive information "
            "beyond all non-fragment fields available to the Transformer. They do not prove a causal "
            "molecular mechanism."
        ),
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs(root: str | Path | None = None) -> pd.DataFrame:
    root_path = locate_root(root)
    p = paths(root_path)
    required = [
        "development", "development_applicability", "novel_applicability",
        "development_index", "development_tokens", "vocab", "selected_oof",
        "all_oof", "direct_common_oof", "registry", "calibration",
        "external_index", "external_tokens", "external_final",
        "external_38_predictions", "strict_records", "evidence_records", "label_reliability",
    ]
    rows = []
    for name in required:
        path = p[name]
        rows.append({"input": name, "exists": path.exists(), "path": str(path)})
    frame = pd.DataFrame(rows)
    if not frame["exists"].all():
        missing = frame.loc[~frame["exists"], "path"].tolist()
        raise FileNotFoundError("Missing upstream artifacts:\n" + "\n".join(missing))

    dev_index = pd.read_csv(p["development_index"], nrows=10, low_memory=False)
    ext_index = pd.read_csv(p["external_index"], nrows=10, low_memory=False)
    required_dev = {"cache_row", "spectrum_id", "inchikey_2d", "source_library", "cv_fold"}
    required_ext = {"feature_row", "spectrum_id", "inchikey_2d", "evaluation_role"}
    if not required_dev.issubset(dev_index.columns):
        raise RuntimeError(f"Development index schema mismatch: {required_dev - set(dev_index.columns)}")
    if not required_ext.issubset(ext_index.columns):
        raise RuntimeError(f"External index schema mismatch: {required_ext - set(ext_index.columns)}")

    with np.load(p["development_tokens"], allow_pickle=False) as z:
        dev_shapes = {name: z[name].shape for name in ["metadata_numeric", "adduct_code", "instrument_code", "source_code"]}
    with np.load(p["external_tokens"], allow_pickle=False) as z:
        ext_shapes = {name: z[name].shape for name in ["metadata_numeric", "adduct_code", "instrument_code", "source_code"]}
    if dev_shapes["metadata_numeric"][1] != 8 or ext_shapes["metadata_numeric"][1] != 8:
        raise RuntimeError("Expected the frozen eight-column Transformer numeric metadata schema")
    frame.attrs["development_token_shapes"] = dev_shapes
    frame.attrs["external_token_shapes"] = ext_shapes
    return frame


def _load_token_inputs(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {
            "numeric": np.asarray(z["metadata_numeric"], dtype=np.float32),
            "adduct": np.asarray(z["adduct_code"], dtype=np.int16),
            "instrument": np.asarray(z["instrument_code"], dtype=np.int16),
            "source": np.asarray(z["source_code"], dtype=np.int16),
        }


def _route_matrix(inputs: dict[str, np.ndarray], rows: np.ndarray, route: str) -> tuple[np.ndarray, list[bool]]:
    numeric = inputs["numeric"][rows]
    adduct = inputs["adduct"][rows, None].astype(float)
    instrument = inputs["instrument"][rows, None].astype(float)
    source = inputs["source"][rows, None].astype(float)
    if route == "Numeric nonfragment HGB":
        return numeric, [False] * numeric.shape[1]
    if route == "Numeric + adduct/instrument HGB":
        return np.hstack([numeric, adduct, instrument]), [False] * 8 + [True, True]
    if route == "Matched nonfragment HGB":
        return np.hstack([numeric, adduct, instrument, source]), [False] * 8 + [True, True, True]
    if route == "Provenance-only HGB":
        return np.hstack([instrument, source]), [True, True]
    raise KeyError(route)


def _fit_hgb(kind: str, categorical: list[bool], seed: int):
    common = dict(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        categorical_features=categorical,
        random_state=seed,
    )
    if kind == "classification":
        return HistGradientBoostingClassifier(loss="log_loss", **common)
    return HistGradientBoostingRegressor(loss="squared_error", **common)


def _fit_predict_linear(
    kind: str,
    train_inputs: dict[str, np.ndarray],
    train_rows: np.ndarray,
    y_train: np.ndarray,
    prediction_inputs: dict[str, np.ndarray],
    prediction_rows: np.ndarray,
    vocab_sizes: tuple[int, int, int],
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler().fit(train_inputs["numeric"][train_rows])
    train_numeric = sparse.csr_matrix(scaler.transform(train_inputs["numeric"][train_rows]))
    pred_numeric = sparse.csr_matrix(scaler.transform(prediction_inputs["numeric"][prediction_rows]))
    categories = [np.arange(size, dtype=int) for size in vocab_sizes]
    encoder = OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    train_cat_raw = np.column_stack([
        train_inputs["adduct"][train_rows],
        train_inputs["instrument"][train_rows],
        train_inputs["source"][train_rows],
    ])
    pred_cat_raw = np.column_stack([
        prediction_inputs["adduct"][prediction_rows],
        prediction_inputs["instrument"][prediction_rows],
        prediction_inputs["source"][prediction_rows],
    ])
    train_cat = encoder.fit_transform(train_cat_raw)
    pred_cat = encoder.transform(pred_cat_raw)
    x_train = sparse.hstack([train_numeric, train_cat], format="csr")
    x_pred = sparse.hstack([pred_numeric, pred_cat], format="csr")
    if kind == "classification":
        model = LogisticRegression(C=1.0, solver="liblinear", max_iter=4000, random_state=seed)
        model.fit(x_train, y_train.astype(int))
        return model.predict_proba(x_pred)[:, 1]
    model = Ridge(alpha=10.0, solver="lsqr", tol=1e-4, max_iter=3000)
    model.fit(x_train, y_train)
    return model.predict(x_pred)


def _predict_hgb(
    route: str,
    kind: str,
    train_inputs: dict[str, np.ndarray],
    train_rows: np.ndarray,
    y_train: np.ndarray,
    prediction_inputs: dict[str, np.ndarray],
    prediction_rows: np.ndarray,
    seed: int,
) -> np.ndarray:
    x_train, categorical = _route_matrix(train_inputs, train_rows, route)
    x_pred, _ = _route_matrix(prediction_inputs, prediction_rows, route)
    model = _fit_hgb(kind, categorical, seed)
    model.fit(x_train, y_train.astype(int) if kind == "classification" else y_train)
    if kind == "classification":
        return model.predict_proba(x_pred)[:, 1]
    return model.predict(x_pred)


def fit_new_controls(p: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    development = pd.read_csv(p["development"], low_memory=False)
    dev_index = pd.read_csv(p["development_index"], low_memory=False).copy()
    ext_index = pd.read_csv(p["external_index"], low_memory=False).copy()
    dev_index["feature_row"] = dev_index["cache_row"].astype(int)
    if not np.array_equal(dev_index["feature_row"].to_numpy(), np.arange(len(dev_index))):
        raise RuntimeError("Development token rows are not aligned with the frozen index")
    if not np.array_equal(ext_index["feature_row"].astype(int).to_numpy(), np.arange(len(ext_index))):
        raise RuntimeError("External token rows are not aligned with the frozen index")
    dev_inputs = _load_token_inputs(p["development_tokens"])
    ext_inputs = _load_token_inputs(p["external_tokens"])
    if len(dev_inputs["numeric"]) != len(dev_index) or len(ext_inputs["numeric"]) != len(ext_index):
        raise RuntimeError("Token cache/index row-count mismatch")
    vocab = _read_json(p["vocab"])
    vocab_sizes = (len(vocab["adduct"]), len(vocab["instrument_category"]), len(vocab["source_library"]))
    label_frame = development.set_index("inchikey_2d")
    spectrum_keys = set(dev_index["inchikey_2d"].astype(str))
    internal_rows: list[dict[str, Any]] = []
    external_fold_rows: list[dict[str, Any]] = []
    ext_all_rows = np.arange(len(ext_index), dtype=int)

    for task, spec in TASKS.items():
        available = development.loc[
            pd.to_numeric(development[spec["column"]], errors="coerce").notna()
            & development["inchikey_2d"].astype(str).isin(spectrum_keys)
        ].copy()
        for fold in range(N_FOLDS):
            train_keys = available.loc[available["cv_fold"].astype(int).ne(fold), "inchikey_2d"].astype(str).tolist()
            test_keys = set(available.loc[available["cv_fold"].astype(int).eq(fold), "inchikey_2d"].astype(str))
            train_rows = sample_development_rows(dev_index, train_keys, task, fold)
            sampled_keys = dev_index.iloc[train_rows]["inchikey_2d"].astype(str).tolist()
            y_train = np.asarray([label_frame.at[key, spec["column"]] for key in sampled_keys], dtype=float)
            test_mask = dev_index["inchikey_2d"].astype(str).isin(test_keys).to_numpy()
            test_rows = np.flatnonzero(test_mask)
            test_index = dev_index.iloc[test_rows].reset_index(drop=True)

            for route in ROUTES:
                seed = stable_seed(SEED_NAMESPACE, route, task, fold)
                if route == "Matched nonfragment linear":
                    test_values = _fit_predict_linear(
                        spec["kind"], dev_inputs, train_rows, y_train, dev_inputs, test_rows, vocab_sizes, seed
                    )
                    ext_values = _fit_predict_linear(
                        spec["kind"], dev_inputs, train_rows, y_train, ext_inputs, ext_all_rows, vocab_sizes, seed
                    )
                else:
                    test_values = _predict_hgb(
                        route, spec["kind"], dev_inputs, train_rows, y_train, dev_inputs, test_rows, seed
                    )
                    ext_values = _predict_hgb(
                        route, spec["kind"], dev_inputs, train_rows, y_train, ext_inputs, ext_all_rows, seed
                    )
                test_compound = aggregate_spectra(test_values, test_index, "median")
                ext_compound = aggregate_spectra(ext_values, ext_index, "median")
                internal_rows.extend(
                    {
                        "cohort": "development_oof", "pipeline": route, "task": task,
                        "kind": spec["kind"], "outer_fold": fold, "inchikey_2d": key,
                        "y_true": float(label_frame.at[key, spec["column"]]), "y_score": float(value),
                    }
                    for key, value in test_compound.items()
                )
                external_fold_rows.extend(
                    {
                        "pipeline": route, "task": task, "kind": spec["kind"],
                        "outer_fold": fold, "inchikey_2d": key, "y_score": float(value),
                    }
                    for key, value in ext_compound.items()
                )

    internal = pd.DataFrame(internal_rows).sort_values(["pipeline", "task", "inchikey_2d"]).reset_index(drop=True)
    external_fold = pd.DataFrame(external_fold_rows)
    external = (
        external_fold.groupby(["pipeline", "task", "kind", "inchikey_2d"], as_index=False)["y_score"]
        .mean()
        .sort_values(["pipeline", "task", "inchikey_2d"])
        .reset_index(drop=True)
    )
    atomic_csv(external, p["unlabeled_external"])
    atomic_json(p["prediction_freeze"], {
        "version": VERSION,
        "frozen_at_utc": utc_now(),
        "external_labels_joined": False,
        "external_labels_used_for_training_tuning_or_selection": False,
        "routes": ROUTES,
        "artifact": {p["unlabeled_external"].name: sha256_file(p["unlabeled_external"])},
    })
    return internal, external


def _cohort_mask(frame: pd.DataFrame, cohort: str) -> pd.Series:
    if cohort == "novel_all":
        return frame["cohort"].isin(["selected_novel", "natural_novel"])
    return frame["cohort"].eq(cohort)


def combine_predictions(p: dict[str, Path], internal: pd.DataFrame, new_external: pd.DataFrame) -> pd.DataFrame:
    selected = pd.read_csv(p["selected_oof"], low_memory=False)
    selected_internal = selected[["inchikey_2d", "outer_fold", "task", "kind", "y_true", "y_score_calibrated"]].rename(
        columns={"y_score_calibrated": "y_score"}
    )
    selected_internal["pipeline"] = PIPELINE_FINAL
    selected_internal["cohort"] = "development_oof"

    all_oof = pd.read_csv(p["all_oof"], low_memory=False)
    metadata_internal = all_oof.loc[all_oof["pipeline"].eq("Metadata")].copy()
    metadata_internal["pipeline"] = PIPELINE_METADATA
    metadata_internal["cohort"] = "development_oof"
    metadata_internal = metadata_internal[["cohort", "pipeline", "task", "kind", "outer_fold", "inchikey_2d", "y_true", "y_score"]]

    final_external_raw = pd.read_csv(p["external_final"], low_memory=False)
    final_external = final_external_raw[["task", "kind", "inchikey_2d", "evaluation_role", "y_true", "y_score_calibrated"]].rename(
        columns={"evaluation_role": "cohort", "y_score_calibrated": "y_score"}
    )
    final_external["pipeline"] = PIPELINE_FINAL
    final_external["outer_fold"] = np.nan

    labels = final_external[["task", "kind", "inchikey_2d", "cohort", "y_true"]].drop_duplicates()
    new_external_labeled = new_external.merge(labels, on=["task", "kind", "inchikey_2d"], how="left", validate="many_to_one")
    new_external_labeled["outer_fold"] = np.nan

    external_38 = pd.read_csv(p["external_38_predictions"], low_memory=False)
    metadata_external = external_38.loc[external_38["pipeline"].eq("Metadata only")].copy()
    metadata_external["pipeline"] = PIPELINE_METADATA
    metadata_external = metadata_external.rename(columns={"evaluation_role": "cohort"})
    metadata_external["outer_fold"] = np.nan
    metadata_external = metadata_external[["cohort", "pipeline", "task", "kind", "outer_fold", "inchikey_2d", "y_true", "y_score"]]

    columns = ["cohort", "pipeline", "task", "kind", "outer_fold", "inchikey_2d", "y_true", "y_score"]
    combined = pd.concat([
        internal[columns], selected_internal[columns], metadata_internal[columns],
        new_external_labeled[columns], final_external[columns], metadata_external[columns],
    ], ignore_index=True)
    combined = combined.drop_duplicates(["cohort", "pipeline", "task", "inchikey_2d"], keep="first")
    return combined.sort_values(["cohort", "pipeline", "task", "inchikey_2d"]).reset_index(drop=True)


def _primary_value(y: np.ndarray, score: np.ndarray, kind: str) -> float:
    if kind == "classification":
        return float(average_precision_score(y, score)) if np.unique(y).size == 2 else np.nan
    return float(mean_absolute_error(y, score))


def calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    cohorts = ["development_oof", "selected_novel", "natural_novel", "novel_all", "transfer"]
    rows = []
    for cohort in cohorts:
        for pipeline in [PIPELINE_METADATA, *ROUTES, PIPELINE_FINAL]:
            for task, spec in TASKS.items():
                frame = predictions.loc[
                    _cohort_mask(predictions, cohort)
                    & predictions["pipeline"].eq(pipeline)
                    & predictions["task"].eq(task)
                ].dropna(subset=["y_true", "y_score"])
                y = frame["y_true"].to_numpy(float)
                score = frame["y_score"].to_numpy(float)
                row = {
                    "cohort": cohort, "pipeline": pipeline, "task": task, "kind": spec["kind"],
                    "n": len(frame), "primary_metric": "pr_auc" if spec["kind"] == "classification" else "mae",
                    "primary_value": _primary_value(y, score, spec["kind"]) if len(frame) else np.nan,
                    "pr_auc": np.nan, "roc_auc": np.nan, "brier": np.nan, "mae": np.nan,
                }
                if len(frame) and spec["kind"] == "classification":
                    row["pr_auc"] = row["primary_value"]
                    row["roc_auc"] = float(roc_auc_score(y, score)) if np.unique(y).size == 2 else np.nan
                    row["brier"] = float(brier_score_loss(y, score))
                elif len(frame):
                    row["mae"] = row["primary_value"]
                rows.append(row)
    return pd.DataFrame(rows)


def metric_improvement(y: np.ndarray, candidate: np.ndarray, reference: np.ndarray, kind: str) -> float:
    if kind == "classification":
        if np.unique(y).size < 2:
            return np.nan
        return float(average_precision_score(y, candidate) - average_precision_score(y, reference))
    return float(mean_absolute_error(y, reference) - mean_absolute_error(y, candidate))


def bootstrap_improvement(
    y: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    kind: str,
    seed: int,
    n_bootstrap: int,
) -> dict[str, float]:
    observed = metric_improvement(y, candidate, reference, kind)
    if len(y) == 0:
        return {"observed_improvement": np.nan, "ci_low": np.nan, "ci_high": np.nan, "one_sided_p": np.nan, "valid_bootstrap": 0}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        value = metric_improvement(y[idx], candidate[idx], reference[idx], kind)
        if np.isfinite(value):
            values.append(value)
    array = np.asarray(values, float)
    return {
        "observed_improvement": observed,
        "ci_low": float(np.quantile(array, 0.025)) if len(array) else np.nan,
        "ci_high": float(np.quantile(array, 0.975)) if len(array) else np.nan,
        "one_sided_p": float((1 + np.sum(array <= 0)) / (1 + len(array))) if len(array) else np.nan,
        "valid_bootstrap": int(len(array)),
    }


def _adjust_pvalues(values: Iterable[float], method: str) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    result = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return result
    raw = p[valid]
    order = np.argsort(raw)
    ordered = raw[order]
    m = len(ordered)
    if method == "holm":
        adjusted_ordered = np.maximum.accumulate((m - np.arange(m)) * ordered)
    elif method == "bh":
        adjusted_ordered = np.minimum.accumulate((ordered * m / (np.arange(m) + 1))[::-1])[::-1]
    else:
        raise KeyError(method)
    adjusted_ordered = np.clip(adjusted_ordered, 0, 1)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_ordered
    result[valid] = adjusted
    return result


def _prediction_matrix(
    frame: pd.DataFrame,
    *,
    label_column: str = "y_true",
    score_column: str = "y_score",
    label_tolerance: float = 1e-5,
) -> pd.DataFrame:
    """Create paired pipeline columns without using floating labels as keys.

    Frozen predictions and newly fitted controls can serialize the same target
    through float32 and float64.  Pivoting on ``(inchikey_2d, y_true)`` then
    splits genuine pairs because their labels differ by a few ULPs.  Pair only
    on the stable compound identifier and retain a consensus label after
    confirming that any cross-pipeline difference is numerical round-off.
    """
    required = {"inchikey_2d", "pipeline", label_column, score_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Prediction matrix is missing columns: {sorted(missing)}")
    if frame.duplicated(["inchikey_2d", "pipeline"]).any():
        duplicated = frame.loc[
            frame.duplicated(["inchikey_2d", "pipeline"], keep=False),
            ["inchikey_2d", "pipeline"],
        ].head(10)
        raise RuntimeError(f"Duplicate compound-pipeline rows: {duplicated.to_dict('records')}")

    labels = frame.groupby("inchikey_2d", sort=False)[label_column].agg(["min", "max", "median"])
    discrepancy = (labels["max"] - labels["min"]).abs()
    if (discrepancy > label_tolerance).any():
        examples = discrepancy.loc[discrepancy > label_tolerance].head(10).to_dict()
        raise RuntimeError(
            f"Materially inconsistent {label_column} values across pipelines: {examples}"
        )

    scores = frame.pivot(index="inchikey_2d", columns="pipeline", values=score_column)
    wide = labels[["median"]].rename(columns={"median": label_column}).join(scores, how="outer")
    return wide.reset_index()


def paired_information_gain(predictions: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    comparisons = [
        (PIPELINE_FINAL, PRIMARY_MATCHED_ROUTE),
        (PIPELINE_FINAL, PIPELINE_METADATA),
        (PIPELINE_FINAL, "Numeric + adduct/instrument HGB"),
        (PRIMARY_MATCHED_ROUTE, "Numeric nonfragment HGB"),
        (PRIMARY_MATCHED_ROUTE, "Provenance-only HGB"),
    ]
    cohorts = ["development_oof", "selected_novel", "natural_novel", "novel_all", "transfer"]
    rows = []
    for cohort in cohorts:
        for task, spec in TASKS.items():
            base = predictions.loc[
                _cohort_mask(predictions, cohort) & predictions["task"].eq(task),
                ["inchikey_2d", "pipeline", "y_true", "y_score"],
            ].dropna(subset=["y_true", "y_score"])
            wide = _prediction_matrix(base)
            for candidate, reference in comparisons:
                if candidate not in wide or reference not in wide:
                    continue
                paired = wide.dropna(subset=[candidate, reference])
                stats = bootstrap_improvement(
                    paired["y_true"].to_numpy(float), paired[candidate].to_numpy(float),
                    paired[reference].to_numpy(float), spec["kind"],
                    stable_seed(SEED_NAMESPACE, cohort, task, candidate, reference), n_bootstrap,
                )
                rows.append({
                    "cohort": cohort, "task": task, "kind": spec["kind"],
                    "candidate": candidate, "reference": reference,
                    "metric": "pr_auc_gain" if spec["kind"] == "classification" else "mae_reduction",
                    "n": len(paired), **stats,
                })
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["holm_p"] = np.nan
        frame["bh_q"] = np.nan
        for _, idx in frame.groupby(["cohort", "candidate", "reference"], sort=False).groups.items():
            indices = list(idx)
            frame.loc[indices, "holm_p"] = _adjust_pvalues(frame.loc[indices, "one_sided_p"], "holm")
            frame.loc[indices, "bh_q"] = _adjust_pvalues(frame.loc[indices, "one_sided_p"], "bh")
    return frame


def _paired_stratified(
    predictions: pd.DataFrame,
    annotation: pd.DataFrame,
    cohort: str,
    stratum_column: str,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows = []
    subset = predictions.loc[
        _cohort_mask(predictions, cohort)
        & predictions["pipeline"].isin([PIPELINE_FINAL, PRIMARY_MATCHED_ROUTE])
    ].merge(annotation[["inchikey_2d", stratum_column]], on="inchikey_2d", how="inner")
    subset = subset.dropna(subset=["y_true", "y_score", stratum_column]).copy()
    for task, spec in TASKS.items():
        task_frame = subset.loc[subset["task"].eq(task)]
        for stratum, part in task_frame.groupby(stratum_column, observed=True, dropna=False):
            wide = _prediction_matrix(part)
            if PIPELINE_FINAL not in wide or PRIMARY_MATCHED_ROUTE not in wide:
                continue
            paired = wide.dropna(subset=[PIPELINE_FINAL, PRIMARY_MATCHED_ROUTE])
            if len(paired) < 10:
                stats = {"observed_improvement": np.nan, "ci_low": np.nan, "ci_high": np.nan, "one_sided_p": np.nan, "valid_bootstrap": 0}
            else:
                stats = bootstrap_improvement(
                    paired["y_true"].to_numpy(float), paired[PIPELINE_FINAL].to_numpy(float),
                    paired[PRIMARY_MATCHED_ROUTE].to_numpy(float), spec["kind"],
                    stable_seed(SEED_NAMESPACE, cohort, stratum_column, str(stratum), task), n_bootstrap,
                )
            rows.append({
                "cohort": cohort, "stratum": str(stratum), "task": task, "kind": spec["kind"],
                "candidate": PIPELINE_FINAL, "reference": PRIMARY_MATCHED_ROUTE,
                "n": len(paired), **stats,
            })
    return pd.DataFrame(rows)


def source_and_similarity_strata(
    p: dict[str, Path], predictions: pd.DataFrame, n_bootstrap: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev_index = pd.read_csv(p["development_index"], usecols=["inchikey_2d", "source_library"], low_memory=False)
    profiles = (
        dev_index.groupby("inchikey_2d")["source_library"]
        .agg(lambda x: ";".join(sorted(set(map(str, x.dropna())))))
        .rename("source_profile")
        .reset_index()
    )
    profiles["source_stratum"] = profiles["source_profile"].where(
        ~profiles["source_profile"].str.contains(";", regex=False), "multi_library"
    )
    source_frame = _paired_stratified(predictions, profiles, "development_oof", "source_stratum", n_bootstrap)

    dev_app = pd.read_csv(p["development_applicability"], low_memory=False)
    dev_app["similarity_stratum"] = pd.cut(
        dev_app["max_morgan_tanimoto_to_outer_train"], bins=SIMILARITY_BINS,
        labels=SIMILARITY_LABELS, right=False,
    ).astype(str)
    dev_similarity = _paired_stratified(predictions, dev_app, "development_oof", "similarity_stratum", n_bootstrap)

    novel_app = pd.read_csv(p["novel_applicability"], low_memory=False)
    novel_app["similarity_stratum"] = pd.cut(
        novel_app["max_morgan_tanimoto_to_development"], bins=SIMILARITY_BINS,
        labels=SIMILARITY_LABELS, right=False,
    ).astype(str)
    novel_similarity = _paired_stratified(predictions, novel_app, "novel_all", "similarity_stratum", n_bootstrap)
    return source_frame, pd.concat([dev_similarity, novel_similarity], ignore_index=True)


def _hazard_truth_and_direction(task: str, y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if task == "P_screen_not_ready_binary":
        return y.astype(int), score, 0.5
    if task == "B_LOGBCF_REGRESSION":
        threshold = math.log10(2000.0)
        return (y >= threshold).astype(int), score, threshold
    threshold = math.log10(0.01)
    return (y <= threshold).astype(int), -score, -threshold


def screening_utility(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in ["development_oof", "novel_all", "transfer"]:
        for task, spec in TASKS.items():
            base = predictions.loc[
                _cohort_mask(predictions, cohort)
                & predictions["task"].eq(task)
                & predictions["pipeline"].isin([PIPELINE_METADATA, PRIMARY_MATCHED_ROUTE, PIPELINE_FINAL]),
                ["inchikey_2d", "pipeline", "y_true", "y_score"],
            ].dropna()
            if base.empty:
                continue
            wide = _prediction_matrix(base)
            for pipeline in [PIPELINE_METADATA, PRIMARY_MATCHED_ROUTE, PIPELINE_FINAL]:
                if pipeline not in wide:
                    continue
                part = wide.dropna(subset=[pipeline])
                y_hazard, ranking_score, threshold = _hazard_truth_and_direction(
                    task, part["y_true"].to_numpy(float), part[pipeline].to_numpy(float)
                )
                predicted_hazard = ranking_score >= threshold
                n_top = max(1, int(math.ceil(0.20 * len(part))))
                top_index = np.argsort(-ranking_score, kind="stable")[:n_top]
                positives = int(y_hazard.sum())
                captured = int(y_hazard[top_index].sum())
                prevalence = float(y_hazard.mean())
                top_density = float(y_hazard[top_index].mean())
                rows.append({
                    "cohort": cohort, "task": task, "pipeline": pipeline, "n": len(part),
                    "positive_n": positives, "positive_prevalence": prevalence,
                    "balanced_accuracy": float(balanced_accuracy_score(y_hazard, predicted_hazard)) if np.unique(y_hazard).size == 2 else np.nan,
                    "precision": float(precision_score(y_hazard, predicted_hazard, zero_division=0)),
                    "recall": float(recall_score(y_hazard, predicted_hazard, zero_division=0)),
                    "f1": float(f1_score(y_hazard, predicted_hazard, zero_division=0)),
                    "top20_n": n_top, "top20_positive_n": captured,
                    "top20_recall": captured / positives if positives else np.nan,
                    "top20_enrichment": top_density / prevalence if prevalence > 0 else np.nan,
                    "decision_threshold_on_hazard_scale": threshold,
                })

            if PRIMARY_MATCHED_ROUTE in wide and PIPELINE_FINAL in wide:
                part = wide.dropna(subset=[PRIMARY_MATCHED_ROUTE, PIPELINE_FINAL])
                y_hazard, final_rank, final_threshold = _hazard_truth_and_direction(
                    task, part["y_true"].to_numpy(float), part[PIPELINE_FINAL].to_numpy(float)
                )
                _, matched_rank, matched_threshold = _hazard_truth_and_direction(
                    task, part["y_true"].to_numpy(float), part[PRIMARY_MATCHED_ROUTE].to_numpy(float)
                )
                final_decision = final_rank >= final_threshold
                matched_decision = matched_rank >= matched_threshold
                rows.append({
                    "cohort": cohort, "task": task, "pipeline": "Final-versus-matched disagreement",
                    "n": len(part), "positive_n": int(y_hazard.sum()), "positive_prevalence": float(y_hazard.mean()),
                    "balanced_accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan,
                    "top20_n": np.nan, "top20_positive_n": np.nan, "top20_recall": np.nan,
                    "top20_enrichment": np.nan,
                    "decision_threshold_on_hazard_scale": np.nan,
                    "decision_disagreement_fraction": float(np.mean(final_decision != matched_decision)),
                })
    return pd.DataFrame(rows)


def effect_size_context(p: dict[str, Path], metrics: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    reliability = pd.read_csv(p["label_reliability"], low_memory=False).set_index("task")
    rows = []
    for task, spec in TASKS.items():
        median_iqr = (
            float(reliability.at[task, "median_iqr_log_units"])
            if task in reliability.index
            and pd.notna(reliability.at[task, "median_iqr_log_units"])
            else np.nan
        )
        replicated_n = (
            int(reliability.at[task, "n_compound_endpoint_groups_with_replicates"])
            if task in reliability.index
            and pd.notna(reliability.at[task, "n_compound_endpoint_groups_with_replicates"])
            else 0
        )
        gain_row = paired.loc[
            paired["cohort"].eq("development_oof") & paired["task"].eq(task)
            & paired["candidate"].eq(PIPELINE_FINAL) & paired["reference"].eq(PRIMARY_MATCHED_ROUTE)
        ]
        base_row = metrics.loc[
            metrics["cohort"].eq("development_oof") & metrics["task"].eq(task)
            & metrics["pipeline"].eq(PRIMARY_MATCHED_ROUTE)
        ]
        gain = float(gain_row.iloc[0]["observed_improvement"]) if len(gain_row) else np.nan
        base_metric = float(base_row.iloc[0]["primary_value"]) if len(base_row) else np.nan
        rows.append({
            "task": task, "kind": spec["kind"], "matched_baseline_primary_value": base_metric,
            "final_improvement_over_matched": gain,
            "relative_improvement_fraction": gain / base_metric if spec["kind"] == "regression" and base_metric else np.nan,
            "median_cross_record_iqr": median_iqr, "replicated_compound_n": replicated_n,
            "improvement_divided_by_observed_iqr": gain / median_iqr if spec["kind"] == "regression" and median_iqr else np.nan,
            "warning": "Observed IQR is descriptive heterogeneity, not an irreducible-error estimate.",
        })
    return pd.DataFrame(rows)


def _strict_toxicity_records(p: dict[str, Path], development_keys: set[str]) -> pd.DataFrame:
    strict = pd.read_csv(p["strict_records"], low_memory=False)
    strict = strict.loc[strict["target_name"].isin(TOXICITY_TASKS)].copy()
    strict["inchikey_2d"] = strict["global_identity_key"].astype(str).str.replace("^IK2D:", "", regex=True)
    strict = strict.loc[strict["inchikey_2d"].isin(development_keys)].copy()
    strict["rounded_value"] = pd.to_numeric(strict["target_value"], errors="coerce").round(12)

    evidence = pd.read_csv(p["evidence_records"], low_memory=False)
    evidence = evidence.loc[evidence["task_id"].isin(TOXICITY_TASKS)].copy()
    evidence["rounded_value"] = pd.to_numeric(evidence["endpoint_value_log10"], errors="coerce").round(12)
    evidence_keep = evidence[[
        "dataset_id", "source_record_id", "task_id", "rounded_value",
        "organism_group", "duration_hours", "duration_days",
    ]].drop_duplicates()
    merged = strict.merge(
        evidence_keep,
        left_on=["dataset_id", "source_record_id", "target_name", "rounded_value"],
        right_on=["dataset_id", "source_record_id", "task_id", "rounded_value"],
        how="left", validate="one_to_one",
    )
    merged["duration_hours_resolved"] = pd.to_numeric(merged["duration_hours"], errors="coerce")
    missing = merged["duration_hours_resolved"].isna()
    merged.loc[missing, "duration_hours_resolved"] = pd.to_numeric(
        merged.loc[missing, "duration_days"], errors="coerce"
    ) * 24.0
    return merged


def toxicity_species_duration_sensitivity(
    p: dict[str, Path], predictions: pd.DataFrame, n_bootstrap: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    development_keys = set(pd.read_csv(p["development"], usecols=["inchikey_2d"])["inchikey_2d"].astype(str))
    records = _strict_toxicity_records(p, development_keys)
    definitions = [
        ("EC50_crustacean_48h", "T_EC50_MG_L_REGRESSION", "crustacean", 48.0),
        ("EC50_algae_72h", "T_EC50_MG_L_REGRESSION", "algae", 72.0),
        ("EC50_algae_96h", "T_EC50_MG_L_REGRESSION", "algae", 96.0),
        ("LC50_fish_96h", "T_LC50_MG_L_REGRESSION", "fish", 96.0),
        ("NOEC_fish_21d", "T_NOEC_MG_L_REGRESSION", "fish", 504.0),
        ("NOEC_crustacean_21d", "T_NOEC_MG_L_REGRESSION", "crustacean", 504.0),
    ]
    result_rows = []
    count_rows = []
    for stratum, task, organism, hours in definitions:
        selected = records.loc[
            records["target_name"].eq(task)
            & records["organism_group"].astype(str).str.lower().eq(organism)
            & np.isclose(records["duration_hours_resolved"], hours, atol=0.01, equal_nan=False)
        ].copy()
        labels = selected.groupby("inchikey_2d", as_index=False)["target_value"].median().rename(
            columns={"target_value": "stratum_y_true"}
        )
        count_rows.append({
            "stratum": stratum, "task": task, "organism_group": organism,
            "duration_hours": hours, "record_n": len(selected), "compound_n": len(labels),
        })
        task_predictions = predictions.loc[
            predictions["cohort"].eq("development_oof") & predictions["task"].eq(task)
            & predictions["pipeline"].isin([PIPELINE_METADATA, PRIMARY_MATCHED_ROUTE, PIPELINE_FINAL]),
            ["inchikey_2d", "pipeline", "y_score"],
        ].merge(labels, on="inchikey_2d", how="inner")
        wide = task_predictions.pivot(index=["inchikey_2d", "stratum_y_true"], columns="pipeline", values="y_score").reset_index()
        for candidate, reference in [
            (PIPELINE_FINAL, PIPELINE_METADATA),
            (PIPELINE_FINAL, PRIMARY_MATCHED_ROUTE),
        ]:
            if candidate not in wide or reference not in wide:
                continue
            paired = wide.dropna(subset=[candidate, reference])
            stats = bootstrap_improvement(
                paired["stratum_y_true"].to_numpy(float), paired[candidate].to_numpy(float),
                paired[reference].to_numpy(float), "regression",
                stable_seed(SEED_NAMESPACE, stratum, candidate, reference), n_bootstrap,
            ) if len(paired) >= 10 else {
                "observed_improvement": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "one_sided_p": np.nan, "valid_bootstrap": 0,
            }
            result_rows.append({
                "stratum": stratum, "task": task, "candidate": candidate, "reference": reference,
                "n_compounds_paired": len(paired),
                "candidate_mae": float(mean_absolute_error(paired["stratum_y_true"], paired[candidate])) if len(paired) else np.nan,
                "reference_mae": float(mean_absolute_error(paired["stratum_y_true"], paired[reference])) if len(paired) else np.nan,
                **stats,
            })
    return pd.DataFrame(result_rows), pd.DataFrame(count_rows)


def pipeline_selection_stability(p: dict[str, Path], n_bootstrap: int) -> pd.DataFrame:
    common = pd.read_csv(p["direct_common_oof"], low_memory=False)
    registry = pd.read_csv(p["registry"], low_memory=False).set_index("task")["selected_pipeline"].to_dict()
    rows = []
    for task, spec in TASKS.items():
        frame = common.loc[common["task"].eq(task)]
        wide = _prediction_matrix(frame)
        pipelines = [c for c in wide.columns if c not in {"inchikey_2d", "y_true"}]
        # The saved common-subset table can still contain a small number of
        # route-level NaNs.  Selection must be evaluated on the strict
        # intersection of every eligible direct pipeline.
        wide = wide.dropna(subset=pipelines).reset_index(drop=True)
        if not len(wide):
            raise RuntimeError(f"No complete direct-candidate rows for {task}")
        y = wide["y_true"].to_numpy(float)
        scores = {pipeline: wide[pipeline].to_numpy(float) for pipeline in pipelines}
        counts = {pipeline: 0 for pipeline in pipelines}
        valid = 0
        rng = np.random.default_rng(stable_seed(SEED_NAMESPACE, "selection_stability", task))
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(wide), len(wide))
            if spec["kind"] == "classification" and np.unique(y[idx]).size < 2:
                continue
            values = {
                pipeline: _primary_value(y[idx], score[idx], spec["kind"])
                for pipeline, score in scores.items()
            }
            best = sorted(values, key=lambda x: (-values[x], x))[0] if spec["kind"] == "classification" else sorted(values, key=lambda x: (values[x], x))[0]
            counts[best] += 1
            valid += 1
        for pipeline in pipelines:
            rows.append({
                "task": task, "kind": spec["kind"], "pipeline": pipeline,
                "selected_in_original_registry": pipeline == registry.get(task),
                "bootstrap_selection_n": counts[pipeline], "valid_bootstrap": valid,
                "bootstrap_selection_probability": counts[pipeline] / valid if valid else np.nan,
            })
    return pd.DataFrame(rows)


def make_figure(
    paired: pd.DataFrame, similarity: pd.DataFrame, toxicity: pd.DataFrame, p: dict[str, Path]
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), constrained_layout=True)
    colors = {PIPELINE_METADATA: "#d68a34", PRIMARY_MATCHED_ROUTE: "#7b4fa3"}

    ax = axes[0]
    subset = paired.loc[
        paired["cohort"].eq("development_oof") & paired["candidate"].eq(PIPELINE_FINAL)
        & paired["reference"].isin([PIPELINE_METADATA, PRIMARY_MATCHED_ROUTE])
    ]
    for j, reference in enumerate([PIPELINE_METADATA, PRIMARY_MATCHED_ROUTE]):
        part = subset.loc[subset["reference"].eq(reference)].set_index("task").reindex(TASKS)
        y = np.arange(len(TASKS)) + (j - 0.5) * 0.18
        center = part["observed_improvement"].to_numpy(float)
        low = part["ci_low"].to_numpy(float)
        high = part["ci_high"].to_numpy(float)
        ax.errorbar(center, y, xerr=np.vstack([center - low, high - center]), fmt="o", capsize=3,
                    color=colors[reference], label=reference)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(np.arange(len(TASKS)), [TASK_LABELS[t] for t in TASKS])
    ax.invert_yaxis()
    ax.set_xlabel("Paired improvement of frozen final model")
    ax.set_title("A  Matched-control audit")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    sim = similarity.loc[similarity["cohort"].eq("development_oof")]
    for task, color in [("B_LOGBCF_REGRESSION", "#315da8"), ("T_LC50_MG_L_REGRESSION", "#2f7d32")]:
        part = sim.loc[sim["task"].eq(task)].set_index("stratum").reindex(SIMILARITY_LABELS)
        ax.plot(SIMILARITY_LABELS, part["observed_improvement"], "o-", color=color, label=TASK_LABELS[task])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Final minus matched-control improvement")
    ax.set_xlabel("Maximum Morgan similarity to outer training set")
    ax.set_title("B  Structural-similarity sensitivity")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    tox = toxicity.loc[toxicity["reference"].eq(PRIMARY_MATCHED_ROUTE)].copy()
    if len(tox):
        y = np.arange(len(tox))
        center = tox["observed_improvement"].to_numpy(float)
        low = tox["ci_low"].to_numpy(float)
        high = tox["ci_high"].to_numpy(float)
        ax.errorbar(center, y, xerr=np.vstack([center - low, high - center]), fmt="o", capsize=3, color="#9b3f37")
        ax.set_yticks(y, tox["stratum"])
        ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("MAE reduction versus matched control")
    ax.set_title("C  Species-duration sensitivity")
    fig.savefig(p["figure_png"], dpi=300, bbox_inches="tight")
    fig.savefig(p["figure_pdf"], bbox_inches="tight")
    plt.close(fig)


def write_summary(
    paired: pd.DataFrame, effect: pd.DataFrame, toxicity: pd.DataFrame,
    selection: pd.DataFrame, p: dict[str, Path],
) -> None:
    key = paired.loc[
        paired["candidate"].eq(PIPELINE_FINAL)
        & paired["reference"].eq(PRIMARY_MATCHED_ROUTE)
        & paired["cohort"].isin(["development_oof", "novel_all"])
    ]
    lines = [
        "# Notebook 40: presubmission matched-control audit", "",
        "Positive paired values favor the frozen final MS/MS pipeline. The matched control includes all eight Transformer numeric metadata fields plus adduct, instrument category and source library, but no fragment peaks.", "",
        "## Frozen final model versus matched non-fragment HGB", "",
        "| Cohort | Endpoint | n | Improvement | 95% CI | Holm p | BH q |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in key.iterrows():
        lines.append(
            f"| {row['cohort']} | {TASK_LABELS[row['task']]} | {int(row['n'])} | {row['observed_improvement']:.3f} | "
            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] | {row['holm_p']:.4f} | {row['bh_q']:.4f} |"
        )
    lines.extend(["", "## Effect-size context", "", "| Endpoint | Matched baseline | Gain | Relative gain | Replicated-label median IQR |", "|---|---:|---:|---:|---:|"])
    for _, row in effect.iterrows():
        lines.append(
            f"| {TASK_LABELS[row['task']]} | {row['matched_baseline_primary_value']:.3f} | "
            f"{row['final_improvement_over_matched']:.3f} | {row['relative_improvement_fraction']:.3f} | "
            f"{row['median_cross_record_iqr']:.3f} |"
        )
    lines.extend([
        "", "Observed cross-record IQR is descriptive label heterogeneity and must not be described as an irreducible error floor.",
        "", "## Species-duration strata", "",
        "The stratum analysis evaluates the already-frozen predictions against condition-specific compound medians. It is a robustness analysis, not a newly optimized condition-specific model.", "",
    ])
    for _, row in toxicity.loc[toxicity["reference"].eq(PRIMARY_MATCHED_ROUTE)].iterrows():
        lines.append(
            f"- {row['stratum']}: n={int(row['n_compounds_paired'])}, MAE reduction={row['observed_improvement']:.3f} "
            f"(95% CI {row['ci_low']:.3f} to {row['ci_high']:.3f})."
        )
    lines.extend(["", "## Model-selection stability", ""])
    for task in TASKS:
        part = selection.loc[(selection["task"] == task) & selection["selected_in_original_registry"]]
        if len(part):
            row = part.iloc[0]
            lines.append(f"- {TASK_LABELS[task]} selected pipeline bootstrap selection probability: {row['bootstrap_selection_probability']:.3f}.")
    lines.extend([
        "", "## Claim boundary", "",
        "A positive gain over the matched control supports predictive information beyond the complete non-fragment input set used here. It does not by itself establish that every gain is mechanistically caused by molecular fragmentation, and stratum-specific estimates with small n should be described as sensitivity analyses.",
    ])
    p["summary"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_qc(
    p: dict[str, Path], predictions: pd.DataFrame, paired: pd.DataFrame,
    toxicity_counts: pd.DataFrame, input_hashes: dict[str, str],
) -> pd.DataFrame:
    primary_pair_counts = paired.loc[
        paired["candidate"].eq(PIPELINE_FINAL)
        & paired["reference"].eq(PRIMARY_MATCHED_ROUTE),
        ["cohort", "task", "n"],
    ].set_index(["cohort", "task"])["n"]
    pair_count_mismatches = []
    for cohort in ["development_oof", "selected_novel", "natural_novel", "novel_all", "transfer"]:
        for task in TASKS:
            subset = predictions.loc[
                _cohort_mask(predictions, cohort)
                & predictions["task"].eq(task)
                & predictions["pipeline"].isin([PIPELINE_FINAL, PRIMARY_MATCHED_ROUTE]),
                ["inchikey_2d", "pipeline", "y_true", "y_score"],
            ].dropna(subset=["y_true", "y_score"])
            wide = _prediction_matrix(subset)
            expected = (
                len(wide.dropna(subset=[PIPELINE_FINAL, PRIMARY_MATCHED_ROUTE]))
                if {PIPELINE_FINAL, PRIMARY_MATCHED_ROUTE}.issubset(wide.columns)
                else 0
            )
            observed = int(primary_pair_counts.get((cohort, task), -1))
            if observed != expected:
                pair_count_mismatches.append(
                    {"cohort": cohort, "task": task, "observed": observed, "expected": expected}
                )
    checks = [
        ("new_outputs_are_isolated", str(p["out"]).endswith(str(OUTPUT_SUBDIR)), str(p["out"])),
        ("all_new_routes_present", set(ROUTES).issubset(set(predictions["pipeline"])), sorted(predictions["pipeline"].unique())),
        ("all_scores_finite", np.isfinite(predictions["y_score"].to_numpy(float)).all(), len(predictions)),
        ("no_duplicate_prediction_keys", not predictions.duplicated(["cohort", "pipeline", "task", "inchikey_2d"]).any(), "cohort-pipeline-task-compound"),
        ("unlabeled_external_has_no_labels", not any(c in {"y_true"} or c.startswith("label__") for c in pd.read_csv(p["unlabeled_external"], nrows=1).columns), p["unlabeled_external"].name),
        ("prediction_freeze_exists", p["prediction_freeze"].exists(), p["prediction_freeze"].name),
        ("paired_primary_comparison_complete", len(paired.loc[(paired["candidate"] == PIPELINE_FINAL) & (paired["reference"] == PRIMARY_MATCHED_ROUTE)]) == 25, "5 cohorts x 5 endpoints"),
        ("paired_primary_sample_sizes_complete", not pair_count_mismatches, pair_count_mismatches or "all paired n equal complete prediction intersections"),
        ("toxicity_strata_declared", len(toxicity_counts) == 6, toxicity_counts["stratum"].tolist()),
        ("upstream_inputs_unchanged", all(sha256_file(p[name]) == digest for name, digest in input_hashes.items()), len(input_hashes)),
    ]
    frame = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    atomic_csv(frame, p["qc"])
    if not frame["passed"].all():
        raise RuntimeError(f"Notebook 40 QC failed: {frame.loc[~frame['passed'], 'check'].tolist()}")
    return frame


def run_analysis(
    root: str | Path | None = None,
    n_bootstrap: int = 2000,
    selection_bootstrap: int = 2000,
) -> dict[str, Any]:
    root_path = locate_root(root)
    p = paths(root_path)
    p["out"].mkdir(parents=True, exist_ok=True)
    p["figures"].mkdir(parents=True, exist_ok=True)
    validation = validate_inputs(root_path)
    plan = analysis_plan()
    if p["analysis_plan"].exists() and _read_json(p["analysis_plan"]) != plan:
        raise RuntimeError("Existing Notebook 40 analysis plan differs from the fixed plan in code")
    atomic_json(p["analysis_plan"], plan)
    input_names = validation["input"].tolist()
    input_hashes = {name: sha256_file(p[name]) for name in input_names}

    internal, external_unlabeled = fit_new_controls(p)
    predictions = combine_predictions(p, internal, external_unlabeled)
    metrics = calculate_metrics(predictions)
    paired = paired_information_gain(predictions, n_bootstrap)
    source, similarity = source_and_similarity_strata(p, predictions, n_bootstrap)
    utility = screening_utility(predictions)
    effect = effect_size_context(p, metrics, paired)
    toxicity, toxicity_counts = toxicity_species_duration_sensitivity(p, predictions, n_bootstrap)
    selection = pipeline_selection_stability(p, selection_bootstrap)

    atomic_csv(predictions, p["predictions"])
    atomic_csv(metrics, p["metrics"])
    atomic_csv(paired, p["paired"])
    atomic_csv(source, p["source_strata"])
    atomic_csv(similarity, p["similarity_strata"])
    atomic_csv(toxicity, p["toxicity_strata"])
    atomic_csv(toxicity_counts, p["toxicity_counts"])
    atomic_csv(utility, p["utility"])
    atomic_csv(effect, p["effect_context"])
    atomic_csv(selection, p["selection_stability"])
    make_figure(paired, similarity, toxicity, p)
    write_summary(paired, effect, toxicity, selection, p)
    qc = run_qc(p, predictions, paired, toxicity_counts, input_hashes)

    outputs = [
        p["analysis_plan"], p["unlabeled_external"], p["prediction_freeze"], p["predictions"],
        p["metrics"], p["paired"], p["source_strata"], p["similarity_strata"],
        p["toxicity_strata"], p["toxicity_counts"], p["utility"], p["effect_context"],
        p["selection_stability"], p["qc"], p["summary"], p["figure_png"], p["figure_pdf"],
    ]
    manifest = {
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "post_freeze_presubmission_analysis": True,
        "changes_initial_submission_model_registry": False,
        "external_labels_used_for_training_tuning_or_selection": False,
        "n_bootstrap": n_bootstrap,
        "selection_bootstrap": selection_bootstrap,
        "input_hashes": input_hashes,
        "output_hashes": {path.name: sha256_file(path) for path in outputs},
        "software": {"python": sys.version, "python_executable": sys.executable},
    }
    atomic_json(p["manifest"], manifest)
    return {
        "root": root_path, "output_dir": p["out"], "manifest": manifest,
        "metrics": metrics, "paired": paired, "source_strata": source,
        "similarity_strata": similarity, "toxicity_strata": toxicity,
        "utility": utility, "effect_size_context": effect,
        "selection_stability": selection, "qc": qc,
        "summary": p["summary"], "figure": p["figure_png"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--selection-bootstrap", type=int, default=2000)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(validate_inputs(args.root).to_string(index=False))
        return
    result = run_analysis(args.root, args.bootstrap, args.selection_bootstrap)
    print(json.dumps({
        "status": result["manifest"]["status"],
        "output_dir": str(result["output_dir"]),
        "summary": str(result["summary"]),
        "figure": str(result["figure"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
