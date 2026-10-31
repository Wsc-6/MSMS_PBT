from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from .utils import json_default, sha256_file, sha256_lines, write_json


SEED = 42
N_BOOTSTRAP = 1000
N_OUTER_FOLDS = 5

TASKS: dict[str, dict[str, str]] = {
    "P_screen_not_ready_binary": {
        "column": "label__P_screen_not_ready_binary",
        "kind": "classification",
    },
    "B_LOGBCF_REGRESSION": {
        "column": "label__B_LOGBCF_REGRESSION",
        "kind": "regression",
    },
    "T_EC50_MG_L_REGRESSION": {
        "column": "label__T_EC50_MG_L_REGRESSION",
        "kind": "regression",
    },
    "T_LC50_MG_L_REGRESSION": {
        "column": "label__T_LC50_MG_L_REGRESSION",
        "kind": "regression",
    },
    "T_NOEC_MG_L_REGRESSION": {
        "column": "label__T_NOEC_MG_L_REGRESSION",
        "kind": "regression",
    },
}

PRIMARY_FAMILIES = ("linear", "mlp")
SECONDARY_FAMILIES = ("random_forest", "catboost", "svm_rbf")
ALL_FAMILIES = PRIMARY_FAMILIES + SECONDARY_FAMILIES
MAIN_ROUTE = "all_spectra_compound_median"
MAIN_MODEL = "nested_selected_primary"


def stable_seed(*parts: Any) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=json_default)


def _parameter_grid(family: str, kind: str) -> list[dict[str, Any]]:
    if family == "linear":
        name = "C" if kind == "classification" else "alpha"
        values = [0.03, 0.3, 3.0, 30.0] if kind == "classification" else [0.1, 1.0, 10.0, 100.0]
        return [{name: value} for value in values]
    if family == "mlp":
        return [
            {"hidden_layer_sizes": [128, 32], "alpha": 0.001},
            {"hidden_layer_sizes": [128, 32], "alpha": 0.1},
        ]
    if family == "random_forest":
        return [
            {"max_depth": None, "min_samples_leaf": 2},
            {"max_depth": 12, "min_samples_leaf": 3},
        ]
    if family == "catboost":
        return [
            {"depth": 4, "l2_leaf_reg": 3.0},
            {"depth": 6, "l2_leaf_reg": 5.0},
        ]
    if family == "svm_rbf":
        return [
            {"C": 0.3, "gamma": "scale", "epsilon": 0.1},
            {"C": 3.0, "gamma": "scale", "epsilon": 0.1},
            {"C": 30.0, "gamma": "scale", "epsilon": 0.1},
        ]
    raise KeyError(family)


def _make_model(family: str, kind: str, params: dict[str, Any], seed: int):
    if family == "linear":
        estimator = (
            LogisticRegression(
                C=float(params["C"]),
                max_iter=4000,
                solver="liblinear",
                random_state=seed,
            )
            if kind == "classification"
            else Ridge(alpha=float(params["alpha"]))
        )
        return Pipeline([("scale", StandardScaler()), ("model", estimator)])

    if family == "mlp":
        common = dict(
            hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
            alpha=float(params["alpha"]),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            batch_size=64,
            max_iter=350,
            tol=1e-4,
            n_iter_no_change=30,
            early_stopping=False,
            random_state=seed,
        )
        estimator = MLPClassifier(**common) if kind == "classification" else MLPRegressor(**common)
        return Pipeline([("scale", StandardScaler()), ("model", estimator)])

    if family == "random_forest":
        common = dict(
            n_estimators=300,
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
        return RandomForestClassifier(**common) if kind == "classification" else RandomForestRegressor(**common)

    if family == "catboost":
        common = dict(
            iterations=150,
            depth=int(params["depth"]),
            l2_leaf_reg=float(params["l2_leaf_reg"]),
            learning_rate=0.05,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        return (
            CatBoostClassifier(loss_function="Logloss", **common)
            if kind == "classification"
            else CatBoostRegressor(loss_function="RMSE", **common)
        )

    if family == "svm_rbf":
        if kind == "classification":
            estimator = SVC(
                C=float(params["C"]),
                gamma=params["gamma"],
                kernel="rbf",
                probability=False,
                random_state=seed,
            )
        else:
            estimator = SVR(
                C=float(params["C"]),
                gamma=params["gamma"],
                epsilon=float(params["epsilon"]),
                kernel="rbf",
            )
        return Pipeline([("scale", StandardScaler()), ("model", estimator)])
    raise KeyError(family)


def _raw_classification_score(model, family: str, x: np.ndarray) -> np.ndarray:
    if family == "svm_rbf":
        return np.asarray(model.decision_function(x), dtype=float)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def _fit_model(model, x: np.ndarray, y: np.ndarray) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        return model.fit(x, y)


def _inner_search(
    x: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    family: str,
    kind: str,
    task: str,
    outer_fold: int,
    scope: str,
) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    inner_folds = sorted(np.unique(fold).tolist())
    if len(inner_folds) < 2:
        raise ValueError(f"Insufficient inner folds for {scope}/{task}/outer={outer_fold}")

    for parameter_index, params in enumerate(_parameter_grid(family, kind)):
        scores: list[float] = []
        for inner_fold in inner_folds:
            train = fold != inner_fold
            valid = fold == inner_fold
            if kind == "classification" and (np.unique(y[train]).size < 2 or np.unique(y[valid]).size < 2):
                raise ValueError(f"Single-class inner split: {scope}/{task}/outer={outer_fold}/inner={inner_fold}")
            seed = stable_seed(SEED, scope, task, family, outer_fold, inner_fold, parameter_index)
            model = _make_model(family, kind, params, seed)
            _fit_model(model, x[train], y[train])
            if kind == "classification":
                prediction = _raw_classification_score(model, family, x[valid])
                score = float(average_precision_score(y[valid], prediction))
                selection_score = score
            else:
                prediction = np.asarray(model.predict(x[valid]), dtype=float)
                score = float(mean_absolute_error(y[valid], prediction))
                selection_score = -score
            scores.append(score)
            records.append(
                {
                    "analysis_scope": scope,
                    "task": task,
                    "kind": kind,
                    "family": family,
                    "outer_fold": outer_fold,
                    "inner_fold": int(inner_fold),
                    "parameter_index": parameter_index,
                    "parameters": _json_text(params),
                    "selection_metric": "pr_auc" if kind == "classification" else "mae",
                    "inner_score": score,
                }
            )
        mean_score = float(np.mean(scores))
        ranking_score = mean_score if kind == "classification" else -mean_score
        candidates.append((ranking_score, _json_text(params), params))
        for row in records[-len(inner_folds) :]:
            row["mean_inner_score"] = mean_score
    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_params = candidates[0][2]
    best_reported_score = candidates[0][0] if kind == "classification" else -candidates[0][0]
    return best_params, float(best_reported_score), records


def _fit_final_predictor(
    x: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    family: str,
    kind: str,
    params: dict[str, Any],
    task: str,
    outer_fold: int,
    scope: str,
) -> dict[str, Any]:
    seed = stable_seed(SEED, scope, task, family, outer_fold, "final")
    model = _make_model(family, kind, params, seed)
    calibrator = None
    if kind == "classification" and family == "svm_rbf":
        calibration_scores = np.full(len(y), np.nan, dtype=float)
        for inner_fold in sorted(np.unique(fold).tolist()):
            train = fold != inner_fold
            valid = fold == inner_fold
            inner_model = _make_model(
                family,
                kind,
                params,
                stable_seed(seed, "calibration", inner_fold),
            )
            _fit_model(inner_model, x[train], y[train])
            calibration_scores[valid] = inner_model.decision_function(x[valid])
        if not np.isfinite(calibration_scores).all():
            raise ValueError("Incomplete group-aware SVC calibration scores")
        calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, random_state=seed)
        calibrator.fit(calibration_scores[:, None], y)
    _fit_model(model, x, y)
    return {"model": model, "calibrator": calibrator, "family": family, "kind": kind}


def _predict(predictor: dict[str, Any], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = predictor["model"]
    family = predictor["family"]
    kind = predictor["kind"]
    if kind == "classification":
        if family == "svm_rbf":
            raw = np.asarray(model.decision_function(x), dtype=float)
            calibrator = predictor["calibrator"]
            probability = (
                calibrator.predict_proba(raw[:, None])[:, 1]
                if calibrator is not None
                else expit(raw)
            )
        else:
            probability = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
        probability = np.clip(probability, 0.0, 1.0)
        return (probability >= 0.5).astype(float), probability
    prediction = np.asarray(model.predict(x), dtype=float)
    return prediction, np.full(len(prediction), np.nan, dtype=float)


def _classification_metrics(y_true: Iterable[float], probability: Iterable[float]) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    has_both_classes = np.unique(y_true).size == 2
    return {
        "pr_auc": float(average_precision_score(y_true, probability)) if has_both_classes else np.nan,
        "roc_auc": float(roc_auc_score(y_true, probability)) if has_both_classes else np.nan,
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "brier": float(brier_score_loss(y_true, probability)),
    }


def _regression_metrics(y_true: Iterable[float], prediction: Iterable[float]) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    spearman = (
        np.nan
        if np.unique(y_true).size < 2 or np.unique(prediction).size < 2
        else float(spearmanr(y_true, prediction).statistic)
    )
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "r2": float(r2_score(y_true, prediction)),
        "spearman": spearman,
    }


def _metric_table(oof: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["representation_route", "model", "task", "kind"]
    for (route, model, task, kind), frame in oof.groupby(group_columns, sort=True):
        frame = frame.sort_values("inchikey_2d").reset_index(drop=True)
        observed = (
            _classification_metrics(frame["y_true"], frame["y_probability"])
            if kind == "classification"
            else _regression_metrics(frame["y_true"], frame["y_prediction"])
        )
        values = {name: [] for name in observed}
        rng = np.random.default_rng(stable_seed(SEED, route, model, task, "bootstrap"))
        for _ in range(n_bootstrap):
            sample = frame.iloc[rng.integers(0, len(frame), len(frame))]
            if kind == "classification" and sample["y_true"].nunique() < 2:
                continue
            current = (
                _classification_metrics(sample["y_true"], sample["y_probability"])
                if kind == "classification"
                else _regression_metrics(sample["y_true"], sample["y_prediction"])
            )
            for name, value in current.items():
                if np.isfinite(value):
                    values[name].append(float(value))
        row = {
            "representation_route": route,
            "model": model,
            "task": task,
            "kind": kind,
            "n_compounds": len(frame),
            **observed,
        }
        for name, current in values.items():
            row[f"{name}_ci_low"] = float(np.quantile(current, 0.025))
            row[f"{name}_ci_high"] = float(np.quantile(current, 0.975))
        primary = "pr_auc" if kind == "classification" else "mae"
        row.update(
            primary_metric=primary,
            primary_value=observed[primary],
            primary_ci_low=row[f"{primary}_ci_low"],
            primary_ci_high=row[f"{primary}_ci_high"],
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _fold_metric_table(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = ["representation_route", "model", "task", "kind", "outer_fold"]
    for keys, frame in oof.groupby(columns, sort=True):
        route, model, task, kind, outer_fold = keys
        metrics = (
            _classification_metrics(frame["y_true"], frame["y_probability"])
            if kind == "classification"
            else _regression_metrics(frame["y_true"], frame["y_prediction"])
        )
        rows.append(
            {
                "representation_route": route,
                "model": model,
                "task": task,
                "kind": kind,
                "outer_fold": int(outer_fold),
                "n_compounds": len(frame),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _paired_bootstrap(
    main: pd.DataFrame,
    comparator: pd.DataFrame,
    comparator_name: str,
    n_bootstrap: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["inchikey_2d", "task", "kind", "outer_fold", "y_true"]
    for task, spec in TASKS.items():
        left = main[main["task"].eq(task)]
        right = comparator[comparator["task"].eq(task)]
        paired = left.merge(right, on=keys, suffixes=("_dreams", "_comparator"), validate="one_to_one")
        if len(paired) != len(left):
            raise ValueError(f"Comparator {comparator_name} is incomplete for {task}: {len(paired)}/{len(left)}")
        if spec["kind"] == "classification":
            observed = average_precision_score(paired["y_true"], paired["y_probability_dreams"]) - average_precision_score(
                paired["y_true"], paired["y_probability_comparator"]
            )
            definition = "DreaMS_PR_minus_comparator_PR"
        else:
            observed = mean_absolute_error(paired["y_true"], paired["y_prediction_comparator"]) - mean_absolute_error(
                paired["y_true"], paired["y_prediction_dreams"]
            )
            definition = "comparator_MAE_minus_DreaMS_MAE"
        rng = np.random.default_rng(stable_seed(SEED, comparator_name, task, "paired_bootstrap"))
        deltas: list[float] = []
        for _ in range(n_bootstrap):
            sample = paired.iloc[rng.integers(0, len(paired), len(paired))]
            if spec["kind"] == "classification":
                if sample["y_true"].nunique() < 2:
                    continue
                delta = average_precision_score(sample["y_true"], sample["y_probability_dreams"]) - average_precision_score(
                    sample["y_true"], sample["y_probability_comparator"]
                )
            else:
                delta = mean_absolute_error(sample["y_true"], sample["y_prediction_comparator"]) - mean_absolute_error(
                    sample["y_true"], sample["y_prediction_dreams"]
                )
            deltas.append(float(delta))
        rows.append(
            {
                "task": task,
                "kind": spec["kind"],
                "comparator": comparator_name,
                "n_compounds": len(paired),
                "delta_definition": definition,
                "observed_delta_positive_is_better": float(observed),
                "ci_low": float(np.quantile(deltas, 0.025)),
                "ci_high": float(np.quantile(deltas, 0.975)),
                "bootstrap_p_one_sided_dreams_not_better": float(
                    (1 + np.sum(np.asarray(deltas) <= 0)) / (1 + len(deltas))
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_compound_embeddings(
    embedding_path: Path,
    index: pd.DataFrame,
    development: pd.DataFrame,
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.shape != (len(index), 1024):
        raise ValueError(f"Unexpected DreaMS shape {embeddings.shape}")
    expected_rows = np.arange(len(index), dtype=np.int64)
    if not np.array_equal(index["feature_row"].to_numpy(dtype=np.int64), expected_rows):
        raise ValueError("feature_row is not the sequential embedding row key")

    compound_keys = sorted(index["inchikey_2d"].astype(str).unique().tolist())
    development = development[development["inchikey_2d"].isin(compound_keys)].copy()
    development = development.sort_values("inchikey_2d").reset_index(drop=True)
    if development["inchikey_2d"].tolist() != compound_keys:
        raise ValueError("DreaMS compound keys do not exactly match the Development subset")

    n_compounds, dimension = len(compound_keys), embeddings.shape[1]
    arrays = {
        "all_spectra": np.full((n_compounds, dimension), np.nan, dtype=np.float32),
        "format_a_pass": np.full((n_compounds, dimension), np.nan, dtype=np.float32),
        "positive_only": np.full((n_compounds, dimension), np.nan, dtype=np.float32),
        "negative_only": np.full((n_compounds, dimension), np.nan, dtype=np.float32),
        "single_representative": np.full((n_compounds, dimension), np.nan, dtype=np.float32),
    }
    metadata_rows: list[dict[str, Any]] = []
    grouped = {key: frame.sort_values("spectrum_id") for key, frame in index.groupby("inchikey_2d", sort=False)}
    for compound_row, key in enumerate(compound_keys):
        frame = grouped[key]
        rows_all = frame["feature_row"].to_numpy(dtype=np.int64)
        masks = {
            "all_spectra": np.ones(len(frame), dtype=bool),
            "format_a_pass": frame["dreams_format_a_domain_pass"].astype(bool).to_numpy(),
            "positive_only": frame["ion_mode"].eq("Positive").to_numpy(),
            "negative_only": frame["ion_mode"].eq("Negative").to_numpy(),
        }
        for route, mask in masks.items():
            if mask.any():
                arrays[route][compound_row] = np.median(np.asarray(embeddings[rows_all[mask]], dtype=np.float32), axis=0)
        representative = frame.iloc[0]
        arrays["single_representative"][compound_row] = np.asarray(
            embeddings[int(representative["feature_row"])], dtype=np.float32
        )
        metadata_rows.append(
            {
                "compound_row": compound_row,
                "inchikey_2d": key,
                "cv_group": representative["cv_group"],
                "cv_fold": int(representative["cv_fold"]),
                "chemvista_evaluation_role": representative["chemvista_evaluation_role"],
                "n_spectra_all": int(len(frame)),
                "n_spectra_format_a_pass": int(masks["format_a_pass"].sum()),
                "n_spectra_positive": int(masks["positive_only"].sum()),
                "n_spectra_negative": int(masks["negative_only"].sum()),
                "single_spectrum_id": representative["spectrum_id"],
                "single_feature_row": int(representative["feature_row"]),
            }
        )
    matrix_index = pd.DataFrame(metadata_rows).merge(
        development[["inchikey_2d", *[spec["column"] for spec in TASKS.values()]]],
        on="inchikey_2d",
        how="left",
        validate="one_to_one",
    )
    for name, array in arrays.items():
        availability = np.isfinite(array).all(axis=1)
        if name in {"all_spectra", "single_representative"} and not availability.all():
            raise ValueError(f"Missing mandatory compound embeddings for {name}")
        matrix_index[f"has_{name}"] = availability
    np.savez_compressed(output_dir / "dreams_compound_embedding_routes.npz", **arrays)
    matrix_index.to_csv(output_dir / "dreams_compound_embedding_index.csv", index=False)
    return arrays, matrix_index


def _prediction_rows(
    matrix_index: pd.DataFrame,
    selected_rows: np.ndarray,
    task: str,
    kind: str,
    route: str,
    model_name: str,
    prediction: np.ndarray,
    probability: np.ndarray,
    inner_best_score: float,
    best_params: dict[str, Any],
    selected_family: str | None,
    count_column: str,
    analysis_scope: str,
) -> pd.DataFrame:
    base_columns = ["inchikey_2d", "cv_group", "cv_fold", "n_spectra_all"]
    if count_column != "n_spectra_all":
        base_columns.append(count_column)
    frame = matrix_index.iloc[selected_rows][base_columns].copy()
    label_column = TASKS[task]["column"]
    frame["task"] = task
    frame["kind"] = kind
    frame["outer_fold"] = frame["cv_fold"].astype(int)
    frame["representation_route"] = route
    frame["model"] = model_name
    frame["selected_family"] = selected_family
    frame["y_true"] = pd.to_numeric(matrix_index.iloc[selected_rows][label_column], errors="coerce").to_numpy(dtype=float)
    frame["y_prediction"] = prediction
    frame["y_probability"] = probability
    frame["n_spectra_route"] = frame[count_column].astype(int)
    frame["inner_best_score"] = inner_best_score
    frame["best_params"] = _json_text(best_params)
    frame["analysis_scope"] = analysis_scope
    drop_columns = ["cv_fold"]
    if count_column != "n_spectra_all":
        drop_columns.append(count_column)
    return frame.drop(columns=drop_columns)


def _train_scope(
    arrays: dict[str, np.ndarray],
    matrix_index: pd.DataFrame,
    input_route: str,
    output_route: str,
    families: tuple[str, ...],
    analysis_scope: str,
    include_family_outputs: bool,
    count_column: str,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, int, str], dict[str, Any]]]:
    x = arrays[input_route]
    availability = np.isfinite(x).all(axis=1)
    fold = matrix_index["cv_fold"].to_numpy(dtype=int)
    prediction_frames: list[pd.DataFrame] = []
    inner_records: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    fitted: dict[tuple[str, int, str], dict[str, Any]] = {}

    for task, spec in TASKS.items():
        kind = spec["kind"]
        labels = pd.to_numeric(matrix_index[spec["column"]], errors="coerce").to_numpy(dtype=float)
        eligible = availability & np.isfinite(labels)
        for outer_fold in range(N_OUTER_FOLDS):
            print(
                f"[{analysis_scope}] task={task} outer_fold={outer_fold} "
                f"train={int((eligible & (fold != outer_fold)).sum())} "
                f"test={int((eligible & (fold == outer_fold)).sum())}",
                flush=True,
            )
            train_rows = np.flatnonzero(eligible & (fold != outer_fold))
            test_rows = np.flatnonzero(eligible & (fold == outer_fold))
            x_train, y_train = x[train_rows], labels[train_rows]
            inner_fold = fold[train_rows]
            family_results: dict[str, dict[str, Any]] = {}
            for family in families:
                family_started = time.perf_counter()
                best_params, best_score, records = _inner_search(
                    x_train,
                    y_train,
                    inner_fold,
                    family,
                    kind,
                    task,
                    outer_fold,
                    analysis_scope,
                )
                inner_records.extend(records)
                predictor = _fit_final_predictor(
                    x_train,
                    y_train,
                    inner_fold,
                    family,
                    kind,
                    best_params,
                    task,
                    outer_fold,
                    analysis_scope,
                )
                y_prediction, y_probability = _predict(predictor, x[test_rows])
                family_results[family] = {
                    "params": best_params,
                    "score": best_score,
                    "predictor": predictor,
                    "prediction": y_prediction,
                    "probability": y_probability,
                }
                fitted[(task, outer_fold, family)] = predictor
                print(
                    f"  family={family} inner_best={best_score:.6f} "
                    f"elapsed={time.perf_counter() - family_started:.1f}s",
                    flush=True,
                )
                selection_records.append(
                    {
                        "analysis_scope": analysis_scope,
                        "task": task,
                        "kind": kind,
                        "outer_fold": outer_fold,
                        "family": family,
                        "inner_best_score": best_score,
                        "best_params": _json_text(best_params),
                    }
                )
                if include_family_outputs:
                    prediction_frames.append(
                        _prediction_rows(
                            matrix_index,
                            test_rows,
                            task,
                            kind,
                            output_route,
                            family,
                            y_prediction,
                            y_probability,
                            best_score,
                            best_params,
                            None,
                            count_column,
                            analysis_scope,
                        )
                    )
            available_primary = [family for family in PRIMARY_FAMILIES if family in family_results]
            if not available_primary:
                raise ValueError(f"No primary family available for {analysis_scope}/{task}")
            if kind == "classification":
                selected_family = sorted(available_primary, key=lambda name: (-family_results[name]["score"], name))[0]
            else:
                selected_family = sorted(available_primary, key=lambda name: (family_results[name]["score"], name))[0]
            selected = family_results[selected_family]
            prediction_frames.append(
                _prediction_rows(
                    matrix_index,
                    test_rows,
                    task,
                    kind,
                    output_route,
                    MAIN_MODEL,
                    selected["prediction"],
                    selected["probability"],
                    selected["score"],
                    selected["params"],
                    selected_family,
                    count_column,
                    analysis_scope,
                )
            )
    return prediction_frames, inner_records, selection_records, fitted


def _main_route_dummy(matrix_index: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    fold = matrix_index["cv_fold"].to_numpy(dtype=int)
    for task, spec in TASKS.items():
        labels = pd.to_numeric(matrix_index[spec["column"]], errors="coerce").to_numpy(dtype=float)
        eligible = np.isfinite(labels)
        for outer_fold in range(N_OUTER_FOLDS):
            train = eligible & (fold != outer_fold)
            test_rows = np.flatnonzero(eligible & (fold == outer_fold))
            mean = float(np.mean(labels[train]))
            if spec["kind"] == "classification":
                probability = np.full(len(test_rows), mean, dtype=float)
                prediction = (probability >= 0.5).astype(float)
                name = "dummy_prevalence"
            else:
                prediction = np.full(len(test_rows), mean, dtype=float)
                probability = np.full(len(test_rows), np.nan, dtype=float)
                name = "dummy_mean"
            frames.append(
                _prediction_rows(
                    matrix_index,
                    test_rows,
                    task,
                    spec["kind"],
                    MAIN_ROUTE,
                    name,
                    prediction,
                    probability,
                    np.nan,
                    {},
                    None,
                    "n_spectra_all",
                    "main_all_spectra",
                )
            )
    return pd.concat(frames, ignore_index=True)


def _variant_predictions(
    arrays: dict[str, np.ndarray],
    matrix_index: pd.DataFrame,
    main_oof: pd.DataFrame,
    fitted: dict[tuple[str, int, str], dict[str, Any]],
    selections: pd.DataFrame,
) -> list[pd.DataFrame]:
    routes = {
        "single_representative": ("single_representative_inference", "n_spectra_all"),
        "positive_only": ("positive_only_inference", "n_spectra_positive"),
        "negative_only": ("negative_only_inference", "n_spectra_negative"),
        "format_a_pass": ("format_a_pass_inference_all_trained", "n_spectra_format_a_pass"),
    }
    frames: list[pd.DataFrame] = []
    fold = matrix_index["cv_fold"].to_numpy(dtype=int)
    for task, spec in TASKS.items():
        labels = pd.to_numeric(matrix_index[spec["column"]], errors="coerce").to_numpy(dtype=float)
        for outer_fold in range(N_OUTER_FOLDS):
            selected = selections[
                selections["analysis_scope"].eq("main_all_spectra")
                & selections["task"].eq(task)
                & selections["outer_fold"].eq(outer_fold)
                & selections["family"].isin(PRIMARY_FAMILIES)
            ].copy()
            selected = selected.sort_values(
                ["inner_best_score", "family"],
                ascending=[spec["kind"] != "classification", True],
            )
            family = str(selected.iloc[0]["family"])
            predictor = fitted[(task, outer_fold, family)]
            params = json.loads(selected.iloc[0]["best_params"])
            score = float(selected.iloc[0]["inner_best_score"])
            for input_route, (output_route, count_column) in routes.items():
                x = arrays[input_route]
                valid = (
                    (fold == outer_fold)
                    & np.isfinite(labels)
                    & np.isfinite(x).all(axis=1)
                )
                rows = np.flatnonzero(valid)
                prediction, probability = _predict(predictor, x[rows])
                frames.append(
                    _prediction_rows(
                        matrix_index,
                        rows,
                        task,
                        spec["kind"],
                        output_route,
                        MAIN_MODEL,
                        prediction,
                        probability,
                        score,
                        params,
                        family,
                        count_column,
                        "inference_sensitivity_main_model",
                    )
                )
    # Same all-spectra predictions restricted to compounds that have official-domain spectra.
    covered = set(matrix_index.loc[matrix_index["has_format_a_pass"], "inchikey_2d"])
    matched = main_oof[main_oof["inchikey_2d"].isin(covered)].copy()
    matched["representation_route"] = "all_spectra_matched_format_a_coverage"
    matched["analysis_scope"] = "coverage_matched_reference"
    matched["n_spectra_route"] = matched["inchikey_2d"].map(
        matrix_index.set_index("inchikey_2d")["n_spectra_all"]
    ).astype(int)
    frames.append(matched)
    return frames


def _load_comparators(root: Path) -> dict[str, pd.DataFrame]:
    comparators: dict[str, pd.DataFrame] = {}
    simple = pd.read_pickle(
        root / "outputs/pbt_msms_v1/simple_spectrum_baselines_v1/simple_spectrum_compound_level_oof.pkl"
    )
    comparators["metadata_only"] = simple[simple["representation_route"].eq("metadata_only")].copy()
    transformer = pd.read_pickle(
        root / "outputs/pbt_msms_v1/set_transformer_v1/set_transformer_compound_level_oof.pkl"
    )
    comparators["set_transformer_pretrained"] = transformer[
        transformer["representation_route"].eq("set_transformer_pretrained")
    ].copy()
    structure = pd.read_csv(
        root / "outputs/pbt_msms_v1/structure_baselines_v1/structure_baseline_oof_predictions.csv.gz",
        low_memory=False,
    )
    structure = structure[structure["model"].eq("nested_selected")].copy()
    comparators["Morgan_2048_nested_selected"] = structure[structure["representation"].eq("Morgan_2048")].copy()
    comparators["MACCS_167_nested_selected"] = structure[structure["representation"].eq("MACCS_167")].copy()
    return comparators


def _task_counts(matrix_index: pd.DataFrame, spectrum_index: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    spectrum_counts = spectrum_index.groupby("inchikey_2d").size()
    format_counts = spectrum_index[spectrum_index["dreams_format_a_domain_pass"]].groupby("inchikey_2d").size()
    for task, spec in TASKS.items():
        labels = pd.to_numeric(matrix_index[spec["column"]], errors="coerce")
        eligible = labels.notna()
        keys = matrix_index.loc[eligible, "inchikey_2d"]
        row = {
            "task": task,
            "kind": spec["kind"],
            "n_compounds": int(eligible.sum()),
            "n_spectra": int(keys.map(spectrum_counts).sum()),
            "n_format_a_compounds": int((eligible & matrix_index["has_format_a_pass"]).sum()),
            "n_format_a_spectra": int(keys.map(format_counts).fillna(0).sum()),
            "format_a_compound_coverage": float((eligible & matrix_index["has_format_a_pass"]).sum() / eligible.sum()),
        }
        if spec["kind"] == "classification":
            counts = labels[eligible].astype(int).value_counts()
            row.update(n_negative=int(counts.get(0, 0)), n_positive=int(counts.get(1, 0)))
        rows.append(row)
    return pd.DataFrame(rows)


def _outer_training_calibration_scores(
    arrays: dict[str, np.ndarray],
    matrix_index: pd.DataFrame,
    family: str,
    params: dict[str, Any],
    outer_fold: int,
    training_route: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Generate calibration scores without fitting on any outer-test row.

    Hyperparameters have already been selected within this outer training set.
    Each calibration prediction also excludes its own structure-cluster fold.
    Never reuse another outer model's OOF scores: that model saw this outer test.
    """
    task = "P_screen_not_ready_binary"
    fold = matrix_index["cv_fold"].to_numpy(dtype=int)
    y = pd.to_numeric(matrix_index[TASKS[task]["column"]], errors="coerce").to_numpy(float)
    x = arrays[training_route]
    eligible = np.isfinite(y) & np.isfinite(x).all(axis=1)
    outer_train = eligible & (fold != outer_fold)
    outer_test_keys = set(matrix_index.loc[fold == outer_fold, "inchikey_2d"])
    outputs = {route: np.full(len(y), np.nan) for route in arrays}
    audit = []
    for inner_fold in sorted(np.unique(fold[outer_train])):
        train = outer_train & (fold != inner_fold)
        valid = outer_train & (fold == inner_fold)
        if np.unique(y[train]).size != 2 or not valid.any():
            raise ValueError("Invalid inner calibration split")
        train_keys = set(matrix_index.loc[train, "inchikey_2d"])
        valid_keys = set(matrix_index.loc[valid, "inchikey_2d"])
        train_groups = set(matrix_index.loc[train, "cv_group"])
        valid_groups = set(matrix_index.loc[valid, "cv_group"])
        if train_keys & outer_test_keys or valid_keys & outer_test_keys or train_groups & valid_groups:
            raise ValueError("Outer-test identity or held-out structure group entered calibration fitting")
        seed = stable_seed(SEED, "outer_train_inner_oof_platt", training_route, family,
                           _json_text(params), outer_fold, int(inner_fold))
        model = _make_model(family, "classification", params, seed)
        _fit_model(model, x[train], y[train].astype(int))
        for route, features in arrays.items():
            rows = valid & np.isfinite(features).all(axis=1)
            if rows.any():
                outputs[route][rows] = _raw_classification_score(model, family, features[rows])
        audit.append({
            "training_route": training_route, "family": family, "parameters": _json_text(params),
            "outer_fold": int(outer_fold), "inner_fold": int(inner_fold),
            "fit_train_n": int(train.sum()), "calibration_validation_n": int(valid.sum()),
            "fit_train_folds": _json_text(sorted(set(fold[train].tolist()))),
            "fit_train_ik2d_sha256": sha256_lines(train_keys),
            "calibration_validation_ik2d_sha256": sha256_lines(valid_keys),
            "outer_test_ik2d_sha256": sha256_lines(outer_test_keys),
            "outer_test_overlap_n": len((train_keys | valid_keys) & outer_test_keys),
            "inner_group_overlap_n": len(train_groups & valid_groups),
        })
    return outputs, audit


def _nested_probability_calibration(
    oof: pd.DataFrame, arrays: dict[str, np.ndarray], matrix_index: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oof = oof.copy()
    keys = matrix_index["inchikey_2d"].astype(str)
    lookup = pd.Series(np.arange(len(keys)), index=keys)
    folds = matrix_index["cv_fold"].to_numpy(int)
    y = pd.to_numeric(matrix_index[TASKS["P_screen_not_ready_binary"]["column"]], errors="coerce").to_numpy(float)
    route_inputs = {
        MAIN_ROUTE: "all_spectra", "all_spectra_matched_format_a_coverage": "all_spectra",
        "single_representative_inference": "single_representative",
        "positive_only_inference": "positive_only", "negative_only_inference": "negative_only",
        "format_a_pass_inference_all_trained": "format_a_pass",
        "format_a_pass_compound_median_retrained": "format_a_pass",
    }
    score_cache = {}
    calibration_rows, inner_rows, fit_audits = [], [], []
    p = oof[oof["task"].eq("P_screen_not_ready_binary") & oof["kind"].eq("classification")]
    for (route, model_name, outer_fold), group in p.groupby(["representation_route", "model", "outer_fold"], sort=True):
        if model_name == "dummy_prevalence":
            oof.loc[group.index, "calibration_method"] = "outer_training_prevalence"
            continue
        if model_name == "svm_rbf":
            oof.loc[group.index, "calibration_method"] = "inner_group_oof_platt"
            continue
        family = str(group["selected_family"].iloc[0]) if model_name == MAIN_MODEL else model_name
        if group["best_params"].nunique() != 1:
            raise ValueError("Calibration group has inconsistent selected parameters")
        params = json.loads(group["best_params"].iloc[0])
        training_route = "format_a_pass" if route == "format_a_pass_compound_median_retrained" else "all_spectra"
        input_route = route_inputs[route]
        cache_key = (training_route, family, _json_text(params), int(outer_fold))
        if cache_key not in score_cache:
            score_cache[cache_key], audits = _outer_training_calibration_scores(
                arrays, matrix_index, family, params, int(outer_fold), training_route
            )
            fit_audits.extend(audits)
        probability = score_cache[cache_key][input_route]
        train = (folds != outer_fold) & np.isfinite(y) & np.isfinite(probability)
        test_rows = lookup.loc[group["inchikey_2d"]].to_numpy(int)
        if not (folds[test_rows] == outer_fold).all() or np.unique(y[train]).size != 2:
            raise ValueError("Calibration/application identities do not match their frozen folds")
        tr = np.clip(probability[train], 1e-6, 1 - 1e-6)
        te = np.clip(group["y_probability_raw"].to_numpy(float), 1e-6, 1 - 1e-6)
        calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                       random_state=stable_seed(SEED, route, model_name, outer_fold, "nested_platt"))
        calibrator.fit(np.log(tr / (1 - tr))[:, None], y[train].astype(int))
        calibrated = calibrator.predict_proba(np.log(te / (1 - te))[:, None])[:, 1]
        oof.loc[group.index, "y_probability"] = calibrated
        oof.loc[group.index, "y_prediction"] = (calibrated >= 0.5).astype(float)
        oof.loc[group.index, "calibration_method"] = "outer_train_inner_group_oof_platt"
        calibration_rows.append({
            "representation_route": route, "model": model_name, "outer_fold": int(outer_fold),
            "n_calibration_compounds": int(train.sum()), "n_application_compounds": len(group),
            "calibration_train_folds": _json_text(sorted(set(folds[train].tolist()))),
            "current_fold_excluded": True, "base_fit_outer_fold_excluded": True,
            "calibration_method": "outer_train_inner_group_oof_platt",
            "coefficient": float(calibrator.coef_[0, 0]), "intercept": float(calibrator.intercept_[0]),
        })
        for row_index in np.flatnonzero(train):
            inner_rows.append({"representation_route": route, "model": model_name,
                "outer_fold": int(outer_fold), "inner_fold": int(folds[row_index]),
                "inchikey_2d": keys.iloc[row_index], "y_true": float(y[row_index]),
                "y_probability_inner_oof": float(probability[row_index])})
        print(f"Nested calibration {route}/{model_name}/outer={outer_fold}: {int(train.sum())} train, {len(group)} test", flush=True)
    return oof, pd.DataFrame(calibration_rows), pd.DataFrame(inner_rows), pd.DataFrame(fit_audits)


def finalize_existing(root: Path) -> dict[str, Any]:
    """Refit calibration-only inner models, preserving frozen outer raw predictions."""
    started = time.perf_counter()
    root = Path(root).resolve()
    output_dir = root / "outputs/pbt_msms_v1/dreams_heads_v1"
    required = {
        "oof": output_dir / "dreams_head_compound_level_oof.pkl",
        "qc": output_dir / "dreams_head_qc_checks.csv",
        "manifest": output_dir / "dreams_head_run_manifest.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cannot finalize missing DreaMS-head outputs:\n" + "\n".join(missing))

    oof = pd.read_pickle(required["oof"]).copy()
    if "y_probability_raw" not in oof:
        oof["y_probability_raw"] = oof["y_probability"]
    else:
        oof["y_probability"] = oof["y_probability_raw"]
    oof["calibration_method"] = np.where(
        oof["kind"].eq("classification"), "uncalibrated", "not_applicable"
    )

    with np.load(output_dir / "dreams_compound_embedding_routes.npz", allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    matrix_index = pd.read_csv(output_dir / "dreams_compound_embedding_index.csv")
    oof, calibration, calibration_inner, calibration_audit = _nested_probability_calibration(oof, arrays, matrix_index)
    if not calibration["base_fit_outer_fold_excluded"].all() or calibration_audit["outer_test_overlap_n"].ne(0).any():
        raise RuntimeError("Nested calibration fit included outer-test identities")
    calibration_inner.to_csv(output_dir / "dreams_head_calibration_inner_oof.csv.gz", index=False)
    calibration_audit.to_csv(output_dir / "dreams_head_calibration_fit_audit.csv", index=False)

    metrics = _metric_table(oof)
    fold_metrics = _fold_metric_table(oof)
    main_selected = oof[
        oof["representation_route"].eq(MAIN_ROUTE) & oof["model"].eq(MAIN_MODEL)
    ].copy()
    dummy = oof[
        oof["representation_route"].eq(MAIN_ROUTE)
        & oof["model"].isin(["dummy_prevalence", "dummy_mean"])
    ].copy()
    comparisons = [
        _paired_bootstrap(main_selected, comparator, name)
        for name, comparator in _load_comparators(root).items()
    ]
    comparisons.append(_paired_bootstrap(main_selected, dummy, "DreaMS_dummy"))
    paired = pd.concat(comparisons, ignore_index=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        oof.to_pickle(output_dir / "dreams_head_compound_level_oof.pkl")
        oof.to_csv(
            output_dir / "dreams_head_compound_level_oof.csv.gz",
            index=False,
            compression="gzip",
        )
        metrics.to_csv(output_dir / "dreams_head_metrics.csv", index=False)
        fold_metrics.to_csv(output_dir / "dreams_head_fold_metrics.csv", index=False)
        paired.to_csv(output_dir / "dreams_head_paired_bootstrap.csv", index=False)
        calibration.to_csv(output_dir / "dreams_head_probability_calibration.csv", index=False)

    qc = pd.read_csv(required["qc"])
    qc.loc[qc["check_id"].eq("DHEAD015"), "description"] = (
        "classification calibration and its base fits exclude the full outer test fold"
    )
    qc.loc[qc["check_id"].eq("DHEAD015"), "details"] = (
        "SVC: inner-group OOF Platt; other P heads: new inner-group OOF predictions fitted within each outer training set"
    )
    calibration_status = "PASS" if calibration.empty or calibration["current_fold_excluded"].all() else "FAIL"
    qc = pd.concat(
        [
            qc,
            pd.DataFrame(
                [
                    {
                        "check_id": "DHEAD017",
                        "description": "P calibration scores come from outer-training-only inner OOF fits; raw outer predictions preserved",
                        "status": calibration_status,
                        "details": f"calibrators={len(calibration)}, raw_probability_column=y_probability_raw",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    qc = qc.drop_duplicates("check_id", keep="last").sort_values("check_id")
    qc.to_csv(required["qc"], index=False)
    if qc["status"].ne("PASS").any():
        raise RuntimeError("Finalized DreaMS-head QC contains FAIL")

    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    artifact_names = [
        "dreams_compound_embedding_routes.npz",
        "dreams_compound_embedding_index.csv",
        "dreams_head_task_counts.csv",
        "dreams_head_compound_level_oof.pkl",
        "dreams_head_compound_level_oof.csv.gz",
        "dreams_head_nested_cv_scores.csv",
        "dreams_head_selected_hyperparameters.csv",
        "dreams_head_metrics.csv",
        "dreams_head_fold_metrics.csv",
        "dreams_head_paired_bootstrap.csv",
        "dreams_head_probability_calibration.csv",
        "dreams_head_calibration_inner_oof.csv.gz",
        "dreams_head_calibration_fit_audit.csv",
        "dreams_head_fold_integrity.csv",
        "dreams_head_qc_checks.csv",
    ]
    manifest["generated_at_utc"] = utc_now()
    manifest["classification_calibration"] = {
        "primary_and_non_svm_heads": "Platt fitted to new inner-group OOF predictions generated exclusively within each outer training set",
        "svm_rbf": "Platt calibration fit on group-aware inner OOF decision scores",
        "raw_probability_retained_as": "y_probability_raw",
    }
    manifest["postprocessing"] = {
        "status": "PASS",
        "elapsed_seconds": time.perf_counter() - started,
        "constant_prediction_spearman": "reported as NaN without warning",
    }
    headline = metrics[
        metrics["representation_route"].eq(MAIN_ROUTE) & metrics["model"].eq(MAIN_MODEL)
    ].sort_values("task")
    manifest["headline_internal_metrics"] = headline.to_dict("records")
    manifest["pipeline_code"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__).resolve()),
    }
    manifest["artifacts"] = {
        name: {
            "path": str(output_dir / name),
            "size_bytes": (output_dir / name).stat().st_size,
            "sha256": sha256_file(output_dir / name),
        }
        for name in artifact_names
    }
    manifest["qc_status"] = "PASS"
    manifest["status"] = "PASS"
    manifest["external_test_unlocked"] = False
    write_json(required["manifest"], manifest)
    return manifest


def run_pipeline(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(root).resolve()
    output_dir = root / "outputs/pbt_msms_v1/dreams_heads_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = root / "outputs/pbt_msms_v1/dreams_v1/batch_embeddings_v1"
    paths = {
        "development": root / "outputs/pbt_msms_v1/split_v1/development_compounds.csv",
        "split_freeze": root / "outputs/pbt_msms_v1/split_v1/split_freeze_manifest.json",
        "split_qc": root / "outputs/pbt_msms_v1/split_v1/leakage_checks.csv",
        "embedding": batch_dir / "development_dreams_embeddings.npy",
        "embedding_index": batch_dir / "development_dreams_input_index.csv.gz",
        "batch_manifest": batch_dir / "dreams_batch_run_manifest.json",
        "batch_qc": batch_dir / "dreams_batch_qc_checks.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing DreaMS-head inputs:\n" + "\n".join(missing))

    batch_manifest = json.loads(paths["batch_manifest"].read_text(encoding="utf-8"))
    batch_qc = pd.read_csv(paths["batch_qc"])
    split_qc = pd.read_csv(paths["split_qc"])
    if batch_manifest.get("status") != "PASS" or not bool(batch_manifest.get("qc_all_pass")):
        raise ValueError("Upstream DreaMS batch is not frozen PASS")
    if bool(batch_manifest.get("external_test_unlocked")):
        raise ValueError("ChemVista external test must remain locked")
    if batch_qc["status"].ne("PASS").any() or split_qc["status"].ne("PASS").any():
        raise ValueError("Upstream QC contains FAIL")

    development = pd.read_csv(paths["development"], low_memory=False)
    spectrum_index = pd.read_csv(paths["embedding_index"], low_memory=False)
    arrays, matrix_index = _build_compound_embeddings(
        paths["embedding"], spectrum_index, development, output_dir
    )
    task_counts = _task_counts(matrix_index, spectrum_index)
    task_counts.to_csv(output_dir / "dreams_head_task_counts.csv", index=False)

    main_frames, inner_records, selection_records, fitted = _train_scope(
        arrays,
        matrix_index,
        input_route="all_spectra",
        output_route=MAIN_ROUTE,
        families=ALL_FAMILIES,
        analysis_scope="main_all_spectra",
        include_family_outputs=True,
        count_column="n_spectra_all",
    )
    main_family_oof = pd.concat(main_frames, ignore_index=True)
    dummy_oof = _main_route_dummy(matrix_index)
    main_selected = main_family_oof[
        main_family_oof["representation_route"].eq(MAIN_ROUTE)
        & main_family_oof["model"].eq(MAIN_MODEL)
    ].copy()
    selections = pd.DataFrame(selection_records)

    variant_frames = _variant_predictions(arrays, matrix_index, main_selected, fitted, selections)

    format_frames, format_inner, format_selections, _ = _train_scope(
        arrays,
        matrix_index,
        input_route="format_a_pass",
        output_route="format_a_pass_compound_median_retrained",
        families=PRIMARY_FAMILIES,
        analysis_scope="format_a_pass_retrained",
        include_family_outputs=False,
        count_column="n_spectra_format_a_pass",
    )
    inner_records.extend(format_inner)
    selection_records.extend(format_selections)

    oof = pd.concat(
        [main_family_oof, dummy_oof, *variant_frames, *format_frames],
        ignore_index=True,
    )
    oof = oof.sort_values(
        ["representation_route", "model", "task", "outer_fold", "inchikey_2d"]
    ).reset_index(drop=True)
    inner_df = pd.DataFrame(inner_records).sort_values(
        ["analysis_scope", "task", "outer_fold", "family", "parameter_index", "inner_fold"]
    )
    selection_df = pd.DataFrame(selection_records).sort_values(
        ["analysis_scope", "task", "outer_fold", "family"]
    )

    metrics = _metric_table(oof)
    fold_metrics = _fold_metric_table(oof)
    comparisons: list[pd.DataFrame] = []
    for name, comparator in _load_comparators(root).items():
        comparisons.append(_paired_bootstrap(main_selected, comparator, name))
    internal_dummy = dummy_oof.copy()
    comparisons.append(_paired_bootstrap(main_selected, internal_dummy, "DreaMS_dummy"))
    paired = pd.concat(comparisons, ignore_index=True)

    oof.to_pickle(output_dir / "dreams_head_compound_level_oof.pkl")
    oof.to_csv(output_dir / "dreams_head_compound_level_oof.csv.gz", index=False, compression="gzip")
    inner_df.to_csv(output_dir / "dreams_head_nested_cv_scores.csv", index=False)
    selection_df.to_csv(output_dir / "dreams_head_selected_hyperparameters.csv", index=False)
    metrics.to_csv(output_dir / "dreams_head_metrics.csv", index=False)
    fold_metrics.to_csv(output_dir / "dreams_head_fold_metrics.csv", index=False)
    paired.to_csv(output_dir / "dreams_head_paired_bootstrap.csv", index=False)

    # Fold-level integrity is recorded independently of model metrics.
    fold_integrity_rows = []
    for outer_fold in range(N_OUTER_FOLDS):
        train_groups = set(matrix_index.loc[matrix_index["cv_fold"].ne(outer_fold), "cv_group"])
        test_groups = set(matrix_index.loc[matrix_index["cv_fold"].eq(outer_fold), "cv_group"])
        fold_integrity_rows.append(
            {
                "outer_fold": outer_fold,
                "n_train_compounds": int(matrix_index["cv_fold"].ne(outer_fold).sum()),
                "n_test_compounds": int(matrix_index["cv_fold"].eq(outer_fold).sum()),
                "n_train_groups": len(train_groups),
                "n_test_groups": len(test_groups),
                "group_overlap": len(train_groups & test_groups),
                "status": "PASS" if not (train_groups & test_groups) else "FAIL",
            }
        )
    fold_integrity = pd.DataFrame(fold_integrity_rows)
    fold_integrity.to_csv(output_dir / "dreams_head_fold_integrity.csv", index=False)

    expected_task_total = sum(task_counts["n_compounds"])
    main_key_counts = main_selected.groupby(["task", "inchikey_2d"]).size()
    qc_rows = [
        ("DHEAD001", "upstream split and DreaMS batch QC contain no FAIL", "PASS" if batch_qc["status"].eq("PASS").all() and split_qc["status"].eq("PASS").all() else "FAIL", ""),
        ("DHEAD002", "embedding and spectrum index row counts match", "PASS" if len(spectrum_index) == batch_manifest["n_spectra"] else "FAIL", f"rows={len(spectrum_index)}"),
        ("DHEAD003", "feature_row is unique and sequential", "PASS" if spectrum_index["feature_row"].is_unique and np.array_equal(spectrum_index["feature_row"], np.arange(len(spectrum_index))) else "FAIL", ""),
        ("DHEAD004", "all mandatory compound embeddings are finite", "PASS" if np.isfinite(arrays["all_spectra"]).all() and np.isfinite(arrays["single_representative"]).all() else "FAIL", f"shape={arrays['all_spectra'].shape}"),
        ("DHEAD005", "DreaMS compounds exactly match the frozen 20 ppm Development subset", "PASS" if len(matrix_index) == batch_manifest["n_compounds"] == 1230 else "FAIL", f"compounds={len(matrix_index)}"),
        ("DHEAD006", "frozen cv_group is confined to one cv_fold", "PASS" if matrix_index.groupby("cv_group")["cv_fold"].nunique().max() == 1 else "FAIL", ""),
        ("DHEAD007", "only public spectra are used and ChemVista spectra are not read", "PASS" if set(spectrum_index["source_library"]) <= {"GNPS", "MassBank", "MoNA"} else "FAIL", _json_text(spectrum_index["source_library"].value_counts().to_dict())),
        ("DHEAD008", "all five frozen outer folds are represented", "PASS" if set(matrix_index["cv_fold"].astype(int)) == set(range(5)) else "FAIL", _json_text(matrix_index["cv_fold"].value_counts().sort_index().to_dict())),
        ("DHEAD009", "P classification has both classes in every fold", "PASS" if all(matrix_index.loc[matrix_index["cv_fold"].eq(fold), TASKS["P_screen_not_ready_binary"]["column"]].dropna().nunique() == 2 for fold in range(5)) else "FAIL", ""),
        ("DHEAD010", "main nested-selected OOF contains every labeled compound exactly once", "PASS" if len(main_selected) == expected_task_total and main_key_counts.eq(1).all() else "FAIL", f"rows={len(main_selected)}, expected={expected_task_total}"),
        ("DHEAD011", "outer train/test structure groups never overlap", "PASS" if fold_integrity["status"].eq("PASS").all() else "FAIL", _json_text(fold_integrity["group_overlap"].tolist())),
        ("DHEAD012", "required primary and secondary model families were evaluated", "PASS" if set(ALL_FAMILIES) <= set(main_family_oof["model"]) else "FAIL", _json_text(sorted(set(main_family_oof["model"])))),
        ("DHEAD013", "official DreaMS DataFormatA coverage is retained and explicitly reported", "PASS" if int(matrix_index["has_format_a_pass"].sum()) == int(batch_manifest["n_format_a_domain_compounds"]) else "FAIL", f"covered_compounds={int(matrix_index['has_format_a_pass'].sum())}"),
        ("DHEAD014", "compound-level bootstrap confidence intervals were produced", "PASS" if metrics["primary_ci_low"].notna().all() and metrics["primary_ci_high"].notna().all() else "FAIL", f"resamples={N_BOOTSTRAP}"),
        ("DHEAD015", "SVC classification uses group-aware out-of-fold Platt calibration", "PASS", "calibration folds are the remaining frozen structure folds"),
        ("DHEAD016", "ChemVista external test remains locked", "PASS" if not bool(batch_manifest["external_test_unlocked"]) else "FAIL", "external_test_unlocked=false"),
    ]
    qc = pd.DataFrame(qc_rows, columns=["check_id", "description", "status", "details"])
    qc.to_csv(output_dir / "dreams_head_qc_checks.csv", index=False)
    if qc["status"].ne("PASS").any():
        raise RuntimeError("DreaMS-head QC failed:\n" + qc[qc["status"].ne("PASS")].to_string(index=False))

    artifact_names = [
        "dreams_compound_embedding_routes.npz",
        "dreams_compound_embedding_index.csv",
        "dreams_head_task_counts.csv",
        "dreams_head_compound_level_oof.pkl",
        "dreams_head_compound_level_oof.csv.gz",
        "dreams_head_nested_cv_scores.csv",
        "dreams_head_selected_hyperparameters.csv",
        "dreams_head_metrics.csv",
        "dreams_head_fold_metrics.csv",
        "dreams_head_paired_bootstrap.csv",
        "dreams_head_fold_integrity.csv",
        "dreams_head_qc_checks.csv",
    ]
    artifacts = {
        name: {
            "path": str(output_dir / name),
            "size_bytes": (output_dir / name).stat().st_size,
            "sha256": sha256_file(output_dir / name),
        }
        for name in artifact_names
    }
    import catboost
    import sklearn

    headline = metrics[
        metrics["representation_route"].eq(MAIN_ROUTE) & metrics["model"].eq(MAIN_MODEL)
    ].sort_values("task")
    manifest = {
        "run_version": "PBT_MSMS_DREAMS_HEADS_V1",
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "external_test_unlocked": False,
        "development_definition": "frozen 20 ppm Development compounds with public spectra; ChemVista spectra not read",
        "representation": {
            "source": "frozen DreaMS embeddings",
            "dimension": 1024,
            "main_compound_aggregation": "coordinate-wise median across all eligible public spectra",
            "equal_compound_weighting": True,
            "inference_sensitivities": [
                "single representative spectrum",
                "positive-mode spectra",
                "negative-mode spectra",
                "official DataFormatA-pass spectra",
            ],
        },
        "outer_cv": {"method": "frozen Morgan-Butina structure clusters", "n_folds": 5},
        "inner_cv": {"method": "remaining frozen structure-cluster folds", "n_folds": 4},
        "primary_head_families": list(PRIMARY_FAMILIES),
        "secondary_head_families": list(SECONDARY_FAMILIES),
        "primary_oof_rule": "per outer fold, select linear or two-layer MLP by inner PR-AUC/MAE only",
        "classification_calibration": "SVC uses Platt logistic calibration fit on group-aware inner OOF decision scores",
        "tasks": TASKS,
        "task_counts": task_counts.to_dict("records"),
        "headline_internal_metrics": headline.to_dict("records"),
        "bootstrap_resamples": N_BOOTSTRAP,
        "bootstrap_scope": "compound-level conditional OOF bootstrap; no model retraining in bootstrap",
        "input_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "development_ik2d_sha256": sha256_lines(matrix_index["inchikey_2d"]),
        "pipeline_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "software": {
            "python": sys.version,
            "executable": sys.executable,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "catboost": catboost.__version__,
            "platform": platform.platform(),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": artifacts,
        "qc_status": "PASS",
    }
    write_json(output_dir / "dreams_head_run_manifest.json", manifest)
    # A fresh training run must use the same isolated calibration as a refresh.
    return finalize_existing(root)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Frozen DreaMS embedding P/B/T heads")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Reuse outer predictions and embeddings; fit inner calibration heads and recompute metrics/QC",
    )
    args = parser.parse_args(argv)
    manifest = finalize_existing(args.root) if args.finalize_existing else run_pipeline(args.root)
    print(json.dumps(manifest["headline_internal_metrics"], ensure_ascii=False, indent=2, default=json_default))
    print(f"STATUS={manifest['status']}")
    print(f"ELAPSED_SECONDS={manifest['elapsed_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
