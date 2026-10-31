"""Reproducible MS/MS fragment-to-structure-to-property interpretation pipeline.

The module reuses the frozen development folds, feature matrices, token caches,
checkpoints, and OOF predictions produced by notebooks 05, 08, and 18.  It does
not read ChemVista labels and it never fills scientific outputs with simulated
values.

The public entry point is :func:`run_pipeline`.  A small command-line interface
is included so that the DreaMS embedding step can be executed in ``env_dreams``
while the rest of the analysis stays in ``env_ms``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import sys
import tempfile
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from pbt_msms.deployment import deployment_dreams_calibration_path


TASKS: dict[str, dict[str, str]] = {
    "P_screen_not_ready_binary": {
        "short": "P",
        "column": "label__P_screen_not_ready_binary",
        "kind": "classification",
        "model_family": "dreams",
    },
    "B_LOGBCF_REGRESSION": {
        "short": "BCF",
        "column": "label__B_LOGBCF_REGRESSION",
        "kind": "regression",
        "model_family": "transformer",
    },
    "T_EC50_MG_L_REGRESSION": {
        "short": "EC50",
        "column": "label__T_EC50_MG_L_REGRESSION",
        "kind": "regression",
        "model_family": "binned",
    },
    "T_LC50_MG_L_REGRESSION": {
        "short": "LC50",
        "column": "label__T_LC50_MG_L_REGRESSION",
        "kind": "regression",
        "model_family": "binned",
    },
    "T_NOEC_MG_L_REGRESSION": {
        "short": "NOEC",
        "column": "label__T_NOEC_MG_L_REGRESSION",
        "kind": "regression",
        "model_family": "transformer",
    },
}
TASK_NAMES = list(TASKS)


@dataclass(frozen=True)
class RunConfig:
    mode: str = "pilot"
    max_compounds_per_endpoint: int | None = 50
    top_features_per_spectrum: int = 30
    integrated_gradient_steps: int = 16
    permutation_controls: int = 1
    dreams_chunk_max_spectra: int = 8_000
    dreams_batch_size: int = 16
    random_seed: int = 42
    transformer_batch_size: int = 128
    run_dreams_peak_occlusion: bool = True

    @classmethod
    def for_mode(cls, mode: str, *, run_dreams_peak_occlusion: bool = True) -> "RunConfig":
        mode = str(mode).lower()
        if mode == "smoke":
            return cls(
                mode=mode,
                max_compounds_per_endpoint=1,
                top_features_per_spectrum=2,
                integrated_gradient_steps=4,
                permutation_controls=0,
                dreams_chunk_max_spectra=2_000,
                transformer_batch_size=16,
                run_dreams_peak_occlusion=run_dreams_peak_occlusion,
            )
        if mode == "pilot":
            return cls(mode=mode, run_dreams_peak_occlusion=run_dreams_peak_occlusion)
        if mode == "representative":
            return cls(
                mode=mode,
                max_compounds_per_endpoint=200,
                top_features_per_spectrum=40,
                integrated_gradient_steps=32,
                permutation_controls=3,
                dreams_chunk_max_spectra=8_000,
                run_dreams_peak_occlusion=run_dreams_peak_occlusion,
            )
        if mode == "full":
            return cls(
                mode=mode,
                max_compounds_per_endpoint=None,
                top_features_per_spectrum=40,
                integrated_gradient_steps=32,
                permutation_controls=5,
                dreams_chunk_max_spectra=8_000,
                run_dreams_peak_occlusion=run_dreams_peak_occlusion,
            )
        raise ValueError(f"Unknown run mode: {mode}")


ATTRIBUTION_COLUMNS = [
    "inchikey_2d",
    "endpoint",
    "endpoint_short",
    "outer_fold",
    "seed",
    "model",
    "sample_id",
    "method",
    "feature_type",
    "mz_or_bin",
    "intensity",
    "signed_attribution",
    "absolute_rank",
    "prediction_before",
    "prediction_after",
    "label_permuted",
    "notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_seed(*parts: Any) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def locate_root(root: str | Path | None = None) -> Path:
    if root is not None:
        result = Path(root).resolve()
        if not (result / "outputs" / "pbt_msms_v1").is_dir():
            raise FileNotFoundError(f"Not a PBT-MS/MS project root: {result}")
        return result
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents, Path(r"E:\谱图预测PBT_终版")]:
        if (candidate / "outputs" / "pbt_msms_v1").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the project root")


def output_directory(root: Path) -> Path:
    path = root / "论文绘图框架_最终版" / "figures" / "SI" / "FigS04" / "executed_outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def select_compounds(
    frame: pd.DataFrame,
    endpoint: str,
    maximum: int | None,
    seed: int,
) -> pd.DataFrame:
    """Select a deterministic fold- and class-aware attribution subset."""
    columns = ["inchikey_2d", "outer_fold", "y_true"]
    candidates = frame[columns].drop_duplicates().copy()
    candidates["selection_stratum"] = candidates["outer_fold"].astype(str)
    if TASKS[endpoint]["kind"] == "classification":
        candidates["selection_stratum"] += "|" + candidates["y_true"].astype(int).astype(str)
    candidates["selection_hash"] = candidates["inchikey_2d"].map(
        lambda key: stable_seed(seed, endpoint, key)
    )
    candidates = candidates.sort_values(["selection_stratum", "selection_hash", "inchikey_2d"])
    if maximum is None or len(candidates) <= maximum:
        selected = candidates
    else:
        strata = list(candidates.groupby("selection_stratum", sort=True))
        selected_indices: list[int] = []
        cursor = 0
        while len(selected_indices) < maximum:
            made_progress = False
            for _, part in strata:
                if cursor < len(part) and len(selected_indices) < maximum:
                    selected_indices.append(int(part.index[cursor]))
                    made_progress = True
            if not made_progress:
                break
            cursor += 1
        selected = candidates.loc[selected_indices]
    return selected.sort_values(["outer_fold", "inchikey_2d"]).reset_index(drop=True)


def rank_attributions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    frame = pd.DataFrame(rows)
    group = ["inchikey_2d", "endpoint", "outer_fold", "seed", "method", "sample_id"]
    frame["_absolute_attribution"] = frame["signed_attribution"].abs()
    frame["absolute_rank"] = frame.groupby(group, dropna=False)["_absolute_attribution"].rank(
        method="first", ascending=False
    )
    frame = frame.drop(columns="_absolute_attribution")
    return frame[ATTRIBUTION_COLUMNS].to_dict("records")


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def _load_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    return torch, nn, functional


class _TransformerDefinitions:
    """Factory that delays torch imports until Transformer analysis is requested."""

    @staticmethod
    def build(config: dict[str, Any], vocab_sizes: dict[str, int], n_tasks: int):
        torch, nn, functional = _load_torch()

        class SpectrumSetTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                d_model = config["d_model"]
                self.peak_projection = nn.Sequential(
                    nn.Linear(config["peak_input_dim"] + 1, d_model),
                    nn.GELU(),
                    nn.LayerNorm(d_model),
                )
                self.metadata_projection = nn.Sequential(
                    nn.Linear(config["metadata_numeric_dim"], d_model),
                    nn.GELU(),
                    nn.LayerNorm(d_model),
                )
                self.adduct_embedding = nn.Embedding(vocab_sizes["adduct"], d_model)
                self.instrument_embedding = nn.Embedding(vocab_sizes["instrument_category"], d_model)
                self.source_embedding = nn.Embedding(vocab_sizes["source_library"], d_model)
                self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=config["n_heads"],
                    dim_feedforward=config["dim_feedforward"],
                    dropout=config["dropout"],
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=config["n_layers"])
                self.final_norm = nn.LayerNorm(d_model)
                self.reconstruction_head = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, config["peak_input_dim"]),
                )
                self.projection_head = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, config["projection_dim"]),
                )
                nn.init.normal_(self.cls_token, std=0.02)

            def forward(self, peaks, padding, mask_flags, metadata, adduct, instrument, source):
                peak_input = torch.cat([peaks, mask_flags.unsqueeze(-1)], dim=-1)
                peak_tokens = self.peak_projection(peak_input)
                condition = (
                    self.metadata_projection(metadata)
                    + self.adduct_embedding(adduct)
                    + self.instrument_embedding(instrument)
                    + self.source_embedding(source)
                )
                cls = self.cls_token.expand(len(peaks), -1, -1) + condition.unsqueeze(1)
                tokens = torch.cat([cls, peak_tokens], dim=1)
                cls_padding = torch.zeros((len(peaks), 1), dtype=torch.bool, device=padding.device)
                full_padding = torch.cat([cls_padding, padding], dim=1)
                encoded = self.final_norm(self.encoder(tokens, src_key_padding_mask=full_padding))
                embedding = functional.normalize(self.projection_head(encoded[:, 0]), dim=-1)
                reconstructed_peaks = self.reconstruction_head(encoded[:, 1:])
                return embedding, reconstructed_peaks

        class MultiTaskSpectrumModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = SpectrumSetTransformer()
                for parameter in self.encoder.reconstruction_head.parameters():
                    parameter.requires_grad = False
                self.head = nn.Sequential(
                    nn.LayerNorm(config["projection_dim"]),
                    nn.Linear(config["projection_dim"], config["projection_dim"]),
                    nn.GELU(),
                    nn.Dropout(0.20),
                    nn.Linear(config["projection_dim"], n_tasks),
                )

            def forward(self, peaks, padding, metadata, adduct, instrument, source):
                mask_flags = torch.zeros(peaks.shape[:2], dtype=peaks.dtype, device=peaks.device)
                embedding, _ = self.encoder(
                    peaks, padding, mask_flags, metadata, adduct, instrument, source
                )
                return self.head(embedding)

        return MultiTaskSpectrumModel()


def _simple_assets(root: Path) -> dict[str, Any]:
    base = root / "outputs" / "pbt_msms_v1"
    directory = base / "simple_spectrum_baselines_v1"
    spectra = pd.read_csv(directory / "spectrum_feature_row_index.csv")
    spectra["inchikey_2d"] = spectra["inchikey_2d"].astype(str)
    labels = pd.read_csv(base / "split_v1" / "development_compounds.csv", low_memory=False)
    labels["inchikey_2d"] = labels["inchikey_2d"].astype(str)
    labels = labels.set_index("inchikey_2d", drop=False)
    assets = {
        "directory": directory,
        "spectra": spectra,
        "labels": labels,
        "fragment": sparse.load_npz(directory / "fragment_intensity_1Da.npz").tocsr(),
        "neutral": sparse.load_npz(directory / "neutral_loss_intensity_1Da.npz").tocsr(),
        "metadata": np.load(directory / "spectrum_metadata.npy", allow_pickle=False),
        "oof": pd.read_pickle(directory / "simple_spectrum_compound_level_oof.pkl"),
    }
    return assets


def _sample_training_rows(
    spectra: pd.DataFrame,
    compound_keys: Iterable[str],
    endpoint: str,
    outer_fold: int,
    repeats: int = 5,
) -> np.ndarray:
    grouped = {
        key: part["feature_row"].to_numpy(dtype=int)
        for key, part in spectra.groupby("inchikey_2d", sort=True)
    }
    sampled: list[int] = []
    for repeat in range(repeats):
        rng = np.random.default_rng(stable_seed(42, endpoint, outer_fold, repeat))
        for key in sorted(map(str, compound_keys)):
            sampled.append(int(rng.choice(grouped[key])))
    return np.asarray(sampled, dtype=int)


def _make_binned_design(
    assets: dict[str, Any], train_rows: np.ndarray, test_rows: np.ndarray
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, StandardScaler]:
    metadata = assets["metadata"]
    scaler = StandardScaler().fit(metadata[train_rows])
    train_meta = sparse.csr_matrix(scaler.transform(metadata[train_rows]).astype(np.float32))
    test_meta = sparse.csr_matrix(scaler.transform(metadata[test_rows]).astype(np.float32))
    train = sparse.hstack(
        [assets["fragment"][train_rows], assets["neutral"][train_rows], train_meta],
        format="csr",
    )
    test = sparse.hstack(
        [assets["fragment"][test_rows], assets["neutral"][test_rows], test_meta],
        format="csr",
    )
    return train, test, scaler


def _aggregate_regression_predictions(
    spectra: pd.DataFrame, rows: np.ndarray, predictions: np.ndarray
) -> pd.DataFrame:
    frame = spectra.iloc[rows][["spectrum_id", "inchikey_2d"]].copy()
    frame["prediction"] = np.asarray(predictions, dtype=float)
    return (
        frame.sort_values(["inchikey_2d", "spectrum_id"])
        .groupby("inchikey_2d", as_index=False)
        .agg(prediction=("prediction", "median"), n_spectra=("spectrum_id", "size"))
    )


def run_binned_analysis(root: Path, config: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refit frozen binned models, reproduce OOF predictions, and explain EC50/LC50."""
    assets = _simple_assets(root)
    spectra = assets["spectra"]
    labels = assets["labels"]
    oof = assets["oof"]
    rows_by_key = {
        key: part["feature_row"].to_numpy(dtype=int)
        for key, part in spectra.groupby("inchikey_2d", sort=True)
    }
    id_to_row = spectra.set_index("spectrum_id")["feature_row"].to_dict()
    attribution_rows: list[dict[str, Any]] = []
    reproduction_rows: list[dict[str, Any]] = []
    selection_rows: list[pd.DataFrame] = []

    for endpoint in ["T_EC50_MG_L_REGRESSION", "T_LC50_MG_L_REGRESSION"]:
        frozen = oof[
            oof["task"].eq(endpoint)
            & oof["representation_route"].eq("fragment_neutral_loss_metadata")
        ].copy()
        selected = select_compounds(
            frozen, endpoint, config.max_compounds_per_endpoint, config.random_seed
        )
        selected["endpoint"] = endpoint
        selected["model_family"] = "binned"
        selection_rows.append(selected)
        selected_keys = set(selected["inchikey_2d"])
        label_column = TASKS[endpoint]["column"]

        spectrum_keys = set(spectra["inchikey_2d"])
        valid = labels[
            pd.to_numeric(labels[label_column], errors="coerce").notna()
            & labels["inchikey_2d"].isin(spectrum_keys)
        ].copy()
        label_map = pd.to_numeric(valid[label_column], errors="coerce").to_dict()
        for outer_fold in range(5):
            train_keys = valid.loc[valid["cv_fold"].astype(int).ne(outer_fold), "inchikey_2d"]
            test_keys = valid.loc[valid["cv_fold"].astype(int).eq(outer_fold), "inchikey_2d"]
            train_rows = _sample_training_rows(spectra, train_keys, endpoint, outer_fold)
            test_rows = np.concatenate([rows_by_key[key] for key in sorted(test_keys)]).astype(int)
            x_train, x_test, _ = _make_binned_design(assets, train_rows, test_rows)
            train_row_keys = spectra.iloc[train_rows]["inchikey_2d"].to_numpy()
            y_train = np.asarray([label_map[key] for key in train_row_keys], dtype=float)
            estimator = Ridge(alpha=10.0, solver="lsqr", tol=1e-4, max_iter=3000)
            estimator.fit(x_train, y_train)
            predictions = np.asarray(estimator.predict(x_test), dtype=float)
            aggregate = _aggregate_regression_predictions(spectra, test_rows, predictions)
            frozen_fold = frozen[frozen["outer_fold"].eq(outer_fold)][
                ["inchikey_2d", "y_prediction"]
            ]
            comparison = aggregate.merge(frozen_fold, on="inchikey_2d", validate="one_to_one")
            difference = comparison["prediction"] - comparison["y_prediction"]
            reproduction_rows.append(
                {
                    "model": "Binned MS/MS",
                    "endpoint": endpoint,
                    "outer_fold": outer_fold,
                    "seed": 42,
                    "n_compounds": len(comparison),
                    "mae_against_frozen_oof": float(np.mean(np.abs(difference))),
                    "max_abs_against_frozen_oof": float(np.max(np.abs(difference))),
                    "pearson_against_frozen_oof": float(
                        np.corrcoef(comparison["prediction"], comparison["y_prediction"])[0, 1]
                    ),
                    "passed": bool(np.max(np.abs(difference)) <= 1e-5),
                }
            )

            fold_selected = selected[
                selected["outer_fold"].eq(outer_fold)
                & selected["inchikey_2d"].isin(selected_keys)
            ]
            if fold_selected.empty:
                continue
            train_mean = np.asarray(x_train.mean(axis=0)).ravel()
            coefficient = np.asarray(estimator.coef_, dtype=float).ravel()
            for _, selection in fold_selected.iterrows():
                key = str(selection["inchikey_2d"])
                frozen_row = frozen[frozen["inchikey_2d"].eq(key)].iloc[0]
                sample_id = str(frozen_row["single_spectrum_id"])
                feature_row = int(id_to_row[sample_id])
                _, x_one, _ = _make_binned_design(
                    assets, train_rows, np.asarray([feature_row], dtype=int)
                )
                x_dense = np.asarray(x_one.toarray(), dtype=float).ravel()
                before = float(estimator.predict(x_one)[0])
                contribution = coefficient * x_dense
                linear_shap = coefficient * (x_dense - train_mean)
                spectral_indices = np.arange(4000)
                active = spectral_indices[
                    (np.abs(contribution[:4000]) > 0) | (np.abs(linear_shap[:4000]) > 0)
                ]
                order = active[
                    np.argsort(
                        np.maximum(
                            np.abs(contribution[active]), np.abs(linear_shap[active])
                        )
                    )[::-1]
                ][: config.top_features_per_spectrum]
                local_rows: list[dict[str, Any]] = []
                for feature_index in order:
                    feature_type = "fragment_mz_bin" if feature_index < 2000 else "neutral_loss_bin"
                    mz_bin = float(feature_index if feature_index < 2000 else feature_index - 2000)
                    common = {
                        "inchikey_2d": key,
                        "endpoint": endpoint,
                        "endpoint_short": TASKS[endpoint]["short"],
                        "outer_fold": outer_fold,
                        "seed": 42,
                        "model": "Binned MS/MS ridge",
                        "sample_id": sample_id,
                        "feature_type": feature_type,
                        "mz_or_bin": mz_bin,
                        "intensity": float(x_dense[feature_index]),
                        "prediction_before": before,
                        "label_permuted": False,
                        "notes": "1 Da bin; representative spectrum; frozen fold refit",
                    }
                    effect = float(contribution[feature_index])
                    local_rows.append(
                        {
                            **common,
                            "method": "mass_bin_occlusion",
                            "signed_attribution": effect,
                            "prediction_after": before - effect,
                        }
                    )
                    local_rows.append(
                        {
                            **common,
                            "method": "coefficient_times_input",
                            "signed_attribution": effect,
                            "prediction_after": before - effect,
                        }
                    )
                    shap_effect = float(linear_shap[feature_index])
                    local_rows.append(
                        {
                            **common,
                            "method": "linear_shap",
                            "signed_attribution": shap_effect,
                            "prediction_after": before - shap_effect,
                        }
                    )
                attribution_rows.extend(rank_attributions(local_rows))

    return (
        pd.DataFrame(attribution_rows, columns=ATTRIBUTION_COLUMNS),
        pd.DataFrame(reproduction_rows),
        pd.concat(selection_rows, ignore_index=True),
    )


class _TokenStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.caches: dict[str, Any] = {}

    def load(self, name: str):
        if name not in self.caches:
            self.caches[name] = np.load(self.directory / name, mmap_mode="r", allow_pickle=False)
        return self.caches[name]

    def numpy_sample(self, name: str, row: int) -> dict[str, Any]:
        cache = self.load(name)
        return {
            "peaks": np.asarray(cache["peak_tokens"][row], dtype=np.float32),
            "padding": np.asarray(cache["padding_mask"][row], dtype=bool),
            "metadata": np.asarray(cache["metadata_numeric"][row], dtype=np.float32),
            "adduct": int(cache["adduct_code"][row]),
            "instrument": int(cache["instrument_code"][row]),
            "source": int(cache["source_code"][row]),
        }


def _transformer_batch(samples: list[dict[str, Any]], device: Any) -> dict[str, Any]:
    torch, _, _ = _load_torch()
    return {
        "peaks": torch.from_numpy(np.stack([x["peaks"] for x in samples])).to(device),
        "padding": torch.from_numpy(np.stack([x["padding"] for x in samples])).to(device),
        "metadata": torch.from_numpy(np.stack([x["metadata"] for x in samples])).to(device),
        "adduct": torch.tensor([x["adduct"] for x in samples], dtype=torch.long, device=device),
        "instrument": torch.tensor(
            [x["instrument"] for x in samples], dtype=torch.long, device=device
        ),
        "source": torch.tensor([x["source"] for x in samples], dtype=torch.long, device=device),
    }


def _transformer_original_output(raw: Any, endpoint: str, metadata: dict[str, Any]) -> Any:
    task_index = TASK_NAMES.index(endpoint)
    values = raw[..., task_index]
    if TASKS[endpoint]["kind"] == "classification":
        return values.sigmoid()
    stats = metadata["regression_stats"][endpoint]
    return values * float(stats["std"]) + float(stats["mean"])


def _transformer_predict_compounds(
    model: Any,
    index: pd.DataFrame,
    store: _TokenStore,
    endpoint: str,
    checkpoint_metadata: dict[str, Any],
    device: Any,
    batch_size: int,
    use_mixed_precision: bool = False,
) -> pd.DataFrame:
    torch, _, _ = _load_torch()
    accumulator: dict[str, list[float]] = {}
    model.eval()
    for start in range(0, len(index), batch_size):
        part = index.iloc[start : start + batch_size]
        samples = [
            store.numpy_sample(str(row.cache_name), int(row.cache_row))
            for row in part.itertuples(index=False)
        ]
        batch = _transformer_batch(samples, device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(use_mixed_precision and device.type == "cuda"),
        ):
            # Notebook 18 generated the frozen OOF table under CUDA float16
            # autocast, then converted raw outputs to float32 before applying
            # endpoint-specific inverse transforms.
            raw = model(**batch)
        values = raw[..., TASK_NAMES.index(endpoint)].float().detach().cpu().numpy()
        if TASKS[endpoint]["kind"] == "classification":
            prediction = expit(np.clip(values, -40.0, 40.0))
        else:
            stats = checkpoint_metadata["regression_stats"][endpoint]
            prediction = values * float(stats["std"]) + float(stats["mean"])
        for key, value in zip(part["inchikey_2d"].astype(str), prediction):
            accumulator.setdefault(key, []).append(float(value))
    return pd.DataFrame(
        {
            "inchikey_2d": list(accumulator),
            "prediction": [float(np.mean(values)) for values in accumulator.values()],
            "n_spectra": [len(values) for values in accumulator.values()],
        }
    )


def _transformer_attention_rollout(model: Any, batch: dict[str, Any]) -> np.ndarray:
    """Compute CLS-to-peak rollout from the frozen encoder self-attention weights."""
    torch, _, _ = _load_torch()
    captured: list[Any] = []
    hooks = []
    for layer in model.encoder.encoder.layers:
        hooks.append(
            layer.register_forward_pre_hook(
                lambda module, inputs, captured=captured: captured.append(inputs[0].detach())
            )
        )
    with torch.inference_mode():
        model(**batch)
    for hook in hooks:
        hook.remove()
    padding = batch["padding"]
    cls_padding = torch.zeros((len(padding), 1), dtype=torch.bool, device=padding.device)
    full_padding = torch.cat([cls_padding, padding], dim=1)
    token_n = full_padding.shape[1]
    rollout = torch.eye(token_n, device=padding.device).unsqueeze(0).repeat(len(padding), 1, 1)
    with torch.inference_mode():
        for layer, layer_input in zip(model.encoder.encoder.layers, captured):
            query = layer.norm1(layer_input) if layer.norm_first else layer_input
            _, weight = layer.self_attn(
                query,
                query,
                query,
                key_padding_mask=full_padding,
                need_weights=True,
                average_attn_weights=False,
            )
            attention = weight.mean(dim=1)
            attention = attention + torch.eye(token_n, device=attention.device).unsqueeze(0)
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            rollout = attention @ rollout
    return rollout[:, 0, 1:].detach().cpu().numpy()


def _transformer_explain_sample(
    model: Any,
    sample: dict[str, Any],
    endpoint: str,
    checkpoint_metadata: dict[str, Any],
    device: Any,
    steps: int,
    top_k: int,
) -> list[dict[str, float]]:
    torch, _, _ = _load_torch()
    batch = _transformer_batch([sample], device)
    model.eval()
    with torch.inference_mode():
        baseline_raw = model(**batch)
        before = float(
            _transformer_original_output(baseline_raw, endpoint, checkpoint_metadata).item()
        )
    valid = np.flatnonzero(~sample["padding"])
    if not len(valid):
        return []
    intensity = sample["peaks"][valid, 1]
    chosen = valid[np.argsort(intensity)[::-1][:top_k]]

    # Exact peak removal: mark one peak as padding while keeping every other input fixed.
    occluded_predictions: dict[int, float] = {}
    for peak_index in chosen:
        changed = {key: value.clone() for key, value in batch.items()}
        changed["padding"][0, int(peak_index)] = True
        with torch.inference_mode():
            raw = model(**changed)
            after = float(_transformer_original_output(raw, endpoint, checkpoint_metadata).item())
        occluded_predictions[int(peak_index)] = after

    # Intensity-only Integrated Gradients preserves m/z and neutral-loss coordinates.
    observed = batch["peaks"].detach()
    baseline = observed.clone()
    baseline[..., 1] = 0.0
    accumulated = torch.zeros_like(observed[..., 1])
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=device)
    for alpha in alphas:
        interpolated = (baseline + alpha * (observed - baseline)).detach().requires_grad_(True)
        raw = model(
            interpolated,
            batch["padding"],
            batch["metadata"],
            batch["adduct"],
            batch["instrument"],
            batch["source"],
        )
        target = _transformer_original_output(raw, endpoint, checkpoint_metadata).sum()
        gradient = torch.autograd.grad(target, interpolated, retain_graph=False)[0][..., 1]
        accumulated += gradient.detach()
    integrated = ((observed[..., 1] - baseline[..., 1]) * accumulated / steps)[0]
    rollout = _transformer_attention_rollout(model, batch)[0]

    results: list[dict[str, float]] = []
    for peak_index in chosen:
        index = int(peak_index)
        results.append(
            {
                "peak_index": index,
                "mz": float(sample["peaks"][index, 0] * 2000.0),
                "intensity": float(sample["peaks"][index, 1]),
                "prediction_before": before,
                "prediction_after": occluded_predictions[index],
                "occlusion": before - occluded_predictions[index],
                "integrated_gradients": float(integrated[index].detach().cpu()),
                "attention_rollout": float(rollout[index]),
            }
        )
    return results


def run_transformer_analysis(
    root: Path, config: RunConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    torch, _, _ = _load_torch()
    base = root / "outputs" / "pbt_msms_v1"
    freeze_dir = base / "transformer_data_v1"
    model_dir = base / "transformer_finetuning_v1"
    labels = pd.read_csv(freeze_dir / "transformer_compound_labels.csv.gz")
    labels["inchikey_2d"] = labels["inchikey_2d"].astype(str)
    vocab = json.loads((freeze_dir / "transformer_categorical_vocab.json").read_text(encoding="utf-8"))
    store = _TokenStore(freeze_dir / "token_caches")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attribution_rows: list[dict[str, Any]] = []
    reproduction_rows: list[dict[str, Any]] = []
    selection_rows: list[pd.DataFrame] = []

    for endpoint in ["B_LOGBCF_REGRESSION", "T_NOEC_MG_L_REGRESSION"]:
        prediction_frames = []
        for path in sorted((model_dir / "outer_test_predictions").rglob("B_real_pretrain__seed_*.csv.gz")):
            part = pd.read_csv(path)
            prediction_frames.append(part[part["task"].eq(endpoint)])
        frozen_all = pd.concat(prediction_frames, ignore_index=True)
        frozen_for_selection = (
            frozen_all.groupby(["inchikey_2d", "outer_fold", "y_true"], as_index=False)
            .agg(y_prediction=("y_prediction", "mean"))
        )
        selected = select_compounds(
            frozen_for_selection, endpoint, config.max_compounds_per_endpoint, config.random_seed
        )
        selected["endpoint"] = endpoint
        selected["model_family"] = "transformer"
        selection_rows.append(selected)

        for outer_fold in range(5):
            evaluation = pd.read_csv(
                freeze_dir / "fold_indices" / f"outer_{outer_fold}" / "real_evaluation_index.csv.gz"
            )
            evaluation["inchikey_2d"] = evaluation["inchikey_2d"].astype(str)
            evaluation = evaluation[evaluation["split_role"].eq("outer_test")].copy()
            available = set(
                labels.loc[pd.to_numeric(labels[TASKS[endpoint]["column"]], errors="coerce").notna(), "inchikey_2d"]
            )
            evaluation = evaluation[evaluation["inchikey_2d"].isin(available)].reset_index(drop=True)
            fold_selected = selected[selected["outer_fold"].eq(outer_fold)]

            for seed_index in range(3):
                checkpoint_path = (
                    model_dir
                    / "checkpoints"
                    / f"outer_{outer_fold}"
                    / f"B_real_pretrain__seed_{seed_index}__best.pt"
                )
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                metadata = checkpoint["metadata"]
                model = _TransformerDefinitions.build(
                    metadata["model_config"], metadata["vocab_sizes"], len(TASKS)
                )
                model.load_state_dict(checkpoint["model_state"], strict=True)
                model.to(device).eval()

                frozen = frozen_all[
                    frozen_all["outer_fold"].eq(outer_fold)
                    & frozen_all["seed_index"].eq(seed_index)
                ][["inchikey_2d", "y_prediction", "n_spectra"]]
                if config.mode == "full":
                    audit_keys = set(frozen["inchikey_2d"])
                else:
                    # Reproduce a deterministic low/medium-spectrum audit subset in smoke,
                    # pilot, and representative modes. Full mode verifies every compound.
                    ordered_audit = frozen.sort_values(["n_spectra", "inchikey_2d"])
                    audit_n = 3 if config.mode == "smoke" else 10
                    audit_keys = set(ordered_audit.head(audit_n)["inchikey_2d"])
                evaluation_audit = evaluation[evaluation["inchikey_2d"].isin(audit_keys)]
                reproduced = _transformer_predict_compounds(
                    model,
                    evaluation_audit,
                    store,
                    endpoint,
                    metadata,
                    device,
                    config.transformer_batch_size,
                    use_mixed_precision=True,
                )
                comparison = reproduced.merge(
                    frozen[["inchikey_2d", "y_prediction"]],
                    on="inchikey_2d",
                    validate="one_to_one",
                )
                difference = comparison["prediction"] - comparison["y_prediction"]
                reproduction_rows.append(
                    {
                        "model": "Transformer B real pretrain",
                        "endpoint": endpoint,
                        "outer_fold": outer_fold,
                        "seed": seed_index,
                        "n_compounds": len(comparison),
                        "mae_against_frozen_oof": float(np.mean(np.abs(difference))),
                        "max_abs_against_frozen_oof": float(np.max(np.abs(difference))),
                        "pearson_against_frozen_oof": float(
                            np.corrcoef(comparison["prediction"], comparison["y_prediction"])[0, 1]
                        ),
                        # Reproduction uses Notebook 18's CUDA float16 autocast;
                        # the separate attribution path below remains float32.
                        "passed": bool(
                            np.max(np.abs(difference)) <= 3e-3
                            and np.corrcoef(
                                comparison["prediction"], comparison["y_prediction"]
                            )[0, 1]
                            >= 0.999
                        ),
                    }
                )

                for _, selection in fold_selected.iterrows():
                    key = str(selection["inchikey_2d"])
                    compound_rows = evaluation[evaluation["inchikey_2d"].eq(key)].sort_values("sample_id")
                    if compound_rows.empty:
                        continue
                    representative = compound_rows.iloc[0]
                    sample = store.numpy_sample(
                        str(representative["cache_name"]), int(representative["cache_row"])
                    )
                    explanations = _transformer_explain_sample(
                        model,
                        sample,
                        endpoint,
                        metadata,
                        device,
                        config.integrated_gradient_steps,
                        config.top_features_per_spectrum,
                    )
                    local_rows: list[dict[str, Any]] = []
                    for explanation in explanations:
                        common = {
                            "inchikey_2d": key,
                            "endpoint": endpoint,
                            "endpoint_short": TASKS[endpoint]["short"],
                            "outer_fold": outer_fold,
                            "seed": seed_index,
                            "model": "Transformer B real pretrain",
                            "sample_id": str(representative["sample_id"]),
                            "feature_type": "fragment_peak",
                            "mz_or_bin": explanation["mz"],
                            "intensity": explanation["intensity"],
                            "prediction_before": explanation["prediction_before"],
                            "label_permuted": False,
                            "notes": "Representative real outer-test spectrum; no external labels",
                        }
                        local_rows.extend(
                            [
                                {
                                    **common,
                                    "method": "peak_occlusion",
                                    "signed_attribution": explanation["occlusion"],
                                    "prediction_after": explanation["prediction_after"],
                                },
                                {
                                    **common,
                                    "method": "integrated_gradients_intensity",
                                    "signed_attribution": explanation["integrated_gradients"],
                                    "prediction_after": np.nan,
                                },
                                {
                                    **common,
                                    "method": "attention_rollout",
                                    "signed_attribution": explanation["attention_rollout"],
                                    "prediction_after": np.nan,
                                },
                            ]
                        )
                    attribution_rows.extend(rank_attributions(local_rows))
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    return (
        pd.DataFrame(attribution_rows, columns=ATTRIBUTION_COLUMNS),
        pd.DataFrame(reproduction_rows),
        pd.concat(selection_rows, ignore_index=True),
    )


def _deployment_platt(parameters: pd.DataFrame, outer_fold: int) -> LogisticRegression:
    """Load the recorded deployment map without refitting it from OOF labels."""
    rows = parameters.loc[
        parameters["representation_route"].eq("all_spectra_compound_median")
        & parameters["model"].eq("nested_selected_primary")
        & parameters["outer_fold"].eq(outer_fold)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one deployed P calibration map for fold {outer_fold}")
    row = rows.iloc[0]
    calibrator = LogisticRegression()
    calibrator.coef_ = np.array([[float(row["coefficient"])]])
    calibrator.intercept_ = np.array([float(row["intercept"])])
    calibrator.classes_ = np.array([0, 1])
    calibrator.n_features_in_ = 1
    return calibrator


def _calibrated_dreams_probability(predictor: dict[str, Any], calibrator: Any, x: np.ndarray) -> np.ndarray:
    from pbt_msms.dreams_heads import _predict

    _, raw_probability = _predict(predictor, np.asarray(x, dtype=np.float32))
    clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped))[:, None]
    return calibrator.predict_proba(logit)[:, 1]


def _dreams_embedding_integrated_gradients(
    predictor: dict[str, Any],
    calibrator: LogisticRegression,
    x: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, float]:
    """Exact-weight IG through the refitted sklearn MLP and frozen Platt map."""
    torch, _, functional = _load_torch()
    pipeline = predictor["model"]
    scaler = pipeline.named_steps["scale"]
    mlp = pipeline.named_steps["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observed = torch.tensor(np.asarray(x, dtype=np.float32), device=device)
    baseline_np = np.asarray(scaler.mean_, dtype=np.float32)
    baseline = torch.tensor(baseline_np, device=device)
    scale = torch.tensor(np.asarray(scaler.scale_, dtype=np.float32), device=device)
    weights = [torch.tensor(w, dtype=torch.float32, device=device) for w in mlp.coefs_]
    biases = [torch.tensor(b, dtype=torch.float32, device=device) for b in mlp.intercepts_]
    platt_weight = torch.tensor(float(calibrator.coef_[0, 0]), device=device)
    platt_bias = torch.tensor(float(calibrator.intercept_[0]), device=device)

    def forward(value):
        hidden = (value - baseline) / scale
        for weight, bias in zip(weights[:-1], biases[:-1]):
            hidden = functional.relu(hidden @ weight + bias)
        raw_logit = (hidden @ weights[-1] + biases[-1]).squeeze(-1)
        return torch.sigmoid(platt_weight * raw_logit + platt_bias)

    accumulated = torch.zeros_like(observed)
    for alpha in torch.linspace(1.0 / steps, 1.0, steps, device=device):
        point = (baseline + alpha * (observed - baseline)).detach().requires_grad_(True)
        value = forward(point)
        accumulated += torch.autograd.grad(value, point)[0].detach()
    attribution = (observed - baseline) * accumulated / steps
    prediction = float(forward(observed).detach().cpu())
    return attribution.detach().cpu().numpy(), prediction


def _dreams_embed_only(mgf_path: Path, output_path: Path, batch_size: int) -> None:
    root = locate_root()
    from pbt_msms.dreams_mvp import configure_writable_runtime_cache

    configure_writable_runtime_cache()
    dreams_source = root / "third_party" / "DreaMS"
    if str(dreams_source) not in sys.path:
        sys.path.insert(0, str(dreams_source))
    from dreams.api import dreams_embeddings

    embeddings = dreams_embeddings(
        mgf_path,
        batch_size=int(batch_size),
        progress_bar=True,
        store_embs=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != 1024 or not np.isfinite(embeddings).all():
        raise RuntimeError(f"Invalid DreaMS embeddings: shape={embeddings.shape}")
    np.save(output_path, embeddings, allow_pickle=False)


def _call_dreams_environment(root: Path, mgf_path: Path, output_path: Path, batch_size: int) -> None:
    python = Path(r"D:\Appinstallation\Anaconda\envs\env_dreams\python.exe")
    if not python.is_file():
        raise FileNotFoundError(f"DreaMS environment is missing: {python}")
    environment = os.environ.copy()
    source_path = str(root / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PBT_MSMS_ROOT"] = str(root)
    command = [
        str(python),
        "-m",
        "pbt_msms.interpretability_chain",
        "--embed-mgf",
        str(mgf_path),
        "--embedding-output",
        str(output_path),
        "--dreams-batch-size",
        str(batch_size),
    ]
    digest = hashlib.sha256(mgf_path.read_bytes()).hexdigest()
    digest_path = output_path.with_suffix(output_path.suffix + ".source_sha256.txt")
    if output_path.is_file() and digest_path.is_file():
        if digest_path.read_text(encoding="utf-8").strip() == digest:
            cached = np.load(output_path, mmap_mode="r", allow_pickle=False)
            if cached.ndim == 2 and cached.shape[1] == 1024 and np.isfinite(cached).all():
                return

    log_path = output_path.with_suffix(output_path.suffix + ".log")
    with log_path.open("wb") as log_handle:
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        tail = log_path.read_bytes()[-12_000:].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"DreaMS embedding failed for {mgf_path.name} with exit code "
            f"{completed.returncode}. Full log: {log_path}\n--- log tail ---\n{tail}"
        )
    digest_path.write_text(digest + "\n", encoding="utf-8")


def _prepare_dreams_occlusion_jobs(
    root: Path,
    selected: pd.DataFrame,
    config: RunConfig,
    work_dir: Path,
) -> list[tuple[Path, Path, Path]]:
    """Write chunked MGF files with one compound-wide 1 Da peak-bin deletion per job."""
    from pbt_msms.dreams_batch import mgf_block
    from pbt_msms.dreams_mvp import extract_peaks

    base = root / "outputs" / "pbt_msms_v1"
    dreams_dir = base / "dreams_v1" / "batch_embeddings_v1"
    index = pd.read_csv(dreams_dir / "development_dreams_input_index.csv.gz")
    index["inchikey_2d"] = index["inchikey_2d"].astype(str)
    fragment = sparse.load_npz(base / "simple_spectrum_baselines_v1" / "fragment_intensity_1Da.npz").tocsr()
    stream_path = root / "质谱数据" / "EnvMolBench_Cleaned.pkl"
    chunk_paths: list[tuple[Path, Path, Path]] = []
    chunk_index = 0
    current_blocks = 0
    mgf_handle = None
    mapping_rows: list[dict[str, Any]] = []
    mgf_path: Path | None = None

    def open_chunk():
        nonlocal chunk_index, mgf_handle, mgf_path, mapping_rows, current_blocks
        mgf_path = work_dir / f"dreams_occlusion_chunk_{chunk_index:03d}.mgf"
        mgf_handle = mgf_path.open("w", encoding="utf-8", newline="\n")
        mapping_rows = []
        current_blocks = 0

    def close_chunk():
        nonlocal chunk_index, mgf_handle, mgf_path, mapping_rows
        if mgf_handle is None or mgf_path is None:
            return
        mgf_handle.close()
        map_path = work_dir / f"dreams_occlusion_chunk_{chunk_index:03d}_map.csv"
        embedding_path = work_dir / f"dreams_occlusion_chunk_{chunk_index:03d}_embeddings.npy"
        write_csv(map_path, pd.DataFrame(mapping_rows))
        chunk_paths.append((mgf_path, map_path, embedding_path))
        chunk_index += 1
        mgf_handle = None

    open_chunk()
    with stream_path.open("rb") as stream:
        for selection in selected.itertuples(index=False):
            key = str(selection.inchikey_2d)
            compound = index[index["inchikey_2d"].eq(key)].sort_values("feature_row")
            if compound.empty:
                continue
            rows = compound["feature_row"].to_numpy(dtype=int)
            mean_spectrum = np.asarray(fragment[rows].mean(axis=0)).ravel()
            active = np.flatnonzero(mean_spectrum > 0)
            chosen_bins = active[np.argsort(mean_spectrum[active])[::-1]][
                : config.top_features_per_spectrum
            ]
            loaded: list[tuple[pd.Series, np.ndarray]] = []
            for _, row in compound.iterrows():
                stream.seek(int(row["stream_offset"]))
                actual_key, record = pickle.load(stream)
                if str(actual_key) != key:
                    raise RuntimeError(f"DreaMS stream identity mismatch for {row['spectrum_id']}")
                loaded.append((row, extract_peaks(record)))

            for mz_bin in chosen_bins:
                required = len(loaded)
                if current_blocks and current_blocks + required > config.dreams_chunk_max_spectra:
                    close_chunk()
                    open_chunk()
                perturbation_id = f"{key}|{int(selection.outer_fold)}|{int(mz_bin)}"
                for row, peaks in loaded:
                    keep = np.floor(peaks[:, 0]).astype(int) != int(mz_bin)
                    modified = peaks[keep]
                    min_peak_guard_applied = len(modified) < 3
                    if min_peak_guard_applied:
                        # DreaMS Format A requires at least three valid peaks.  Reuse the original
                        # spectrum when a complete 1 Da deletion would make the perturbation invalid;
                        # the mapping records this conservative zero-effect guard for audit.
                        modified = peaks
                    feature_id = len(mapping_rows)
                    mgf_handle.write(
                        mgf_block(
                            feature_id,
                            row,
                            modified,
                            int(row["charge"]),
                            str(row["source_library"]),
                        )
                    )
                    mapping_rows.append(
                        {
                            "embedding_row": feature_id,
                            "perturbation_id": perturbation_id,
                            "inchikey_2d": key,
                            "outer_fold": int(selection.outer_fold),
                            "mz_bin": int(mz_bin),
                            "mean_bin_intensity": float(mean_spectrum[mz_bin]),
                            "spectrum_id": str(row["spectrum_id"]),
                            "min_peak_guard_applied": bool(min_peak_guard_applied),
                        }
                    )
                    current_blocks += 1
    close_chunk()
    return chunk_paths


def _run_dreams_peak_occlusion(
    root: Path,
    selected: pd.DataFrame,
    config: RunConfig,
    predictors: dict[int, dict[str, Any]],
    calibrators: dict[int, LogisticRegression],
    original_embeddings: np.ndarray,
    matrix_index: pd.DataFrame,
    work_dir: Path,
) -> pd.DataFrame:
    chunks = _prepare_dreams_occlusion_jobs(root, selected, config, work_dir)
    original_by_key = {
        str(row.inchikey_2d): np.asarray(original_embeddings[int(row.compound_row)], dtype=np.float32)
        for row in matrix_index.itertuples(index=False)
    }
    result_rows: list[dict[str, Any]] = []
    for mgf_path, map_path, embedding_path in chunks:
        _call_dreams_environment(root, mgf_path, embedding_path, config.dreams_batch_size)
        mapping = pd.read_csv(map_path)
        embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
        if len(mapping) != len(embeddings):
            raise RuntimeError(f"DreaMS occlusion row mismatch for {mgf_path.name}")
        for perturbation_id, group in mapping.groupby("perturbation_id", sort=False):
            first = group.iloc[0]
            key = str(first["inchikey_2d"])
            fold = int(first["outer_fold"])
            changed = np.median(np.asarray(embeddings[group.index], dtype=np.float32), axis=0)
            before = float(
                _calibrated_dreams_probability(
                    predictors[fold], calibrators[fold], original_by_key[key][None, :]
                )[0]
            )
            after = float(
                _calibrated_dreams_probability(
                    predictors[fold], calibrators[fold], changed[None, :]
                )[0]
            )
            result_rows.append(
                {
                    "inchikey_2d": key,
                    "endpoint": "P_screen_not_ready_binary",
                    "endpoint_short": "P",
                    "outer_fold": fold,
                    "seed": 42,
                    "model": "DreaMS + selected MLP head",
                    "sample_id": f"all_spectra(n={len(group)})",
                    "method": "compound_peak_bin_occlusion_dreams",
                    "feature_type": "fragment_mz_bin",
                    "mz_or_bin": float(first["mz_bin"]),
                    "intensity": float(first["mean_bin_intensity"]),
                    "signed_attribution": before - after,
                    "absolute_rank": np.nan,
                    "prediction_before": before,
                    "prediction_after": after,
                    "label_permuted": False,
                    "notes": (
                        "1 Da bin removed before compound-median DreaMS aggregation; "
                        f"minimum-3-peak guard reused {int(group['min_peak_guard_applied'].sum())} "
                        f"of {len(group)} spectra"
                    ),
                }
            )
        del embeddings
    return pd.DataFrame(rank_attributions(result_rows), columns=ATTRIBUTION_COLUMNS)


def run_dreams_analysis(
    root: Path, config: RunConfig, work_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from pbt_msms.dreams_heads import _fit_model, _make_model, _predict, stable_seed as dreams_seed

    base = root / "outputs" / "pbt_msms_v1"
    directory = base / "dreams_heads_v1"
    arrays = np.load(directory / "dreams_compound_embedding_routes.npz", allow_pickle=False)
    x = np.asarray(arrays["all_spectra"], dtype=np.float32)
    matrix_index = pd.read_csv(directory / "dreams_compound_embedding_index.csv")
    matrix_index["inchikey_2d"] = matrix_index["inchikey_2d"].astype(str)
    frozen_path = base / "frozen_deployment_v1/dreams_head_compound_level_oof.pkl"
    if not frozen_path.exists():
        frozen_path = directory / "dreams_head_compound_level_oof.pkl"
    frozen_all = pd.read_pickle(frozen_path)
    deployment_parameters = pd.read_csv(deployment_dreams_calibration_path(directory))
    frozen = frozen_all[
        frozen_all["task"].eq("P_screen_not_ready_binary")
        & frozen_all["representation_route"].eq("all_spectra_compound_median")
        & frozen_all["model"].eq("nested_selected_primary")
    ].copy()
    selected = select_compounds(
        frozen, "P_screen_not_ready_binary", config.max_compounds_per_endpoint, config.random_seed
    )
    selected["endpoint"] = "P_screen_not_ready_binary"
    selected["model_family"] = "dreams"
    labels = pd.to_numeric(
        matrix_index[TASKS["P_screen_not_ready_binary"]["column"]], errors="coerce"
    ).to_numpy(float)
    folds = matrix_index["cv_fold"].to_numpy(int)
    available = np.isfinite(x).all(axis=1) & np.isfinite(labels)
    attribution_rows: list[dict[str, Any]] = []
    reproduction_rows: list[dict[str, Any]] = []
    predictors: dict[int, dict[str, Any]] = {}
    calibrators: dict[int, LogisticRegression] = {}

    for outer_fold in range(5):
        frozen_fold = frozen[frozen["outer_fold"].eq(outer_fold)]
        if frozen_fold.empty:
            continue
        family = str(frozen_fold["selected_family"].iloc[0])
        params = json.loads(str(frozen_fold["best_params"].iloc[0]))
        train_rows = np.flatnonzero(available & (folds != outer_fold))
        test_rows = np.flatnonzero(available & (folds == outer_fold))
        estimator = _make_model(
            family,
            "classification",
            params,
            dreams_seed(42, "main_all_spectra", "P_screen_not_ready_binary", family, outer_fold, "final"),
        )
        _fit_model(estimator, x[train_rows], labels[train_rows].astype(int))
        predictor = {
            "model": estimator,
            "calibrator": None,
            "family": family,
            "kind": "classification",
        }
        calibrator = _deployment_platt(deployment_parameters, outer_fold)
        predictors[outer_fold] = predictor
        calibrators[outer_fold] = calibrator
        _, raw_probability = _predict(predictor, x[test_rows])
        calibrated = _calibrated_dreams_probability(predictor, calibrator, x[test_rows])
        reproduced = pd.DataFrame(
            {
                "inchikey_2d": matrix_index.iloc[test_rows]["inchikey_2d"].to_numpy(),
                "raw_reproduced": raw_probability,
                "calibrated_reproduced": calibrated,
            }
        )
        comparison = reproduced.merge(
            frozen_fold[["inchikey_2d", "y_probability_raw", "y_probability"]],
            on="inchikey_2d",
            validate="one_to_one",
        )
        raw_difference = comparison["raw_reproduced"] - comparison["y_probability_raw"]
        calibrated_difference = comparison["calibrated_reproduced"] - comparison["y_probability"]
        reproduction_rows.append(
            {
                "model": "DreaMS + selected MLP head",
                "endpoint": "P_screen_not_ready_binary",
                "outer_fold": outer_fold,
                "seed": 42,
                "n_compounds": len(comparison),
                "mae_against_frozen_oof": float(np.mean(np.abs(calibrated_difference))),
                "max_abs_against_frozen_oof": float(np.max(np.abs(calibrated_difference))),
                "raw_max_abs_against_frozen_oof": float(np.max(np.abs(raw_difference))),
                "pearson_against_frozen_oof": float(
                    np.corrcoef(comparison["calibrated_reproduced"], comparison["y_probability"])[0, 1]
                ),
                "passed": bool(
                    np.max(np.abs(raw_difference)) <= 2e-5
                    and np.max(np.abs(calibrated_difference)) <= 2e-5
                ),
            }
        )

        fold_selected = selected[selected["outer_fold"].eq(outer_fold)]
        key_to_row = matrix_index.set_index("inchikey_2d")["compound_row"].to_dict()
        for selection in fold_selected.itertuples(index=False):
            key = str(selection.inchikey_2d)
            compound_row = int(key_to_row[key])
            attribution, prediction = _dreams_embedding_integrated_gradients(
                predictor, calibrator, x[compound_row], config.integrated_gradient_steps
            )
            order = np.argsort(np.abs(attribution))[::-1][: config.top_features_per_spectrum]
            local_rows: list[dict[str, Any]] = []
            for dimension in order:
                local_rows.append(
                    {
                        "inchikey_2d": key,
                        "endpoint": "P_screen_not_ready_binary",
                        "endpoint_short": "P",
                        "outer_fold": outer_fold,
                        "seed": 42,
                        "model": "DreaMS + selected MLP head",
                        "sample_id": "all_spectra_compound_median",
                        "method": "embedding_integrated_gradients",
                        "feature_type": "dreams_embedding_dimension",
                        "mz_or_bin": float(dimension),
                        "intensity": float(x[compound_row, dimension]),
                        "signed_attribution": float(attribution[dimension]),
                        "absolute_rank": np.nan,
                        "prediction_before": prediction,
                        "prediction_after": np.nan,
                        "label_permuted": False,
                        "notes": "Baseline is the outer-training embedding mean; includes frozen cross-fitted Platt map",
                    }
                )
            attribution_rows.extend(rank_attributions(local_rows))

    embedding_attribution = pd.DataFrame(attribution_rows, columns=ATTRIBUTION_COLUMNS)
    if config.run_dreams_peak_occlusion:
        peak_attribution = _run_dreams_peak_occlusion(
            root,
            selected,
            config,
            predictors,
            calibrators,
            x,
            matrix_index,
            work_dir,
        )
        embedding_attribution = pd.concat(
            [embedding_attribution, peak_attribution], ignore_index=True
        )
    return embedding_attribution, pd.DataFrame(reproduction_rows), selected


def summarize_stability(attributions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if attributions.empty:
        return pd.DataFrame()
    frame = attributions.copy()
    frame["feature_key"] = (
        frame["feature_type"].astype(str)
        + "|"
        + frame["mz_or_bin"].round(0).astype("Int64").astype(str)
    )
    frame["replicate"] = frame["outer_fold"].astype(str) + "|" + frame["seed"].astype(str)
    aggregated = (
        frame.groupby(["endpoint", "method", "replicate", "feature_key"], as_index=False)
        .agg(mean_abs_attribution=("signed_attribution", lambda x: float(np.mean(np.abs(x)))))
    )
    for (endpoint, method), part in aggregated.groupby(["endpoint", "method"], sort=True):
        vectors = {
            replicate: group.set_index("feature_key")["mean_abs_attribution"]
            for replicate, group in part.groupby("replicate")
        }
        correlations: list[float] = []
        names = sorted(vectors)
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                union = vectors[left_name].index.union(vectors[right_name].index)
                left = vectors[left_name].reindex(union, fill_value=0.0)
                right = vectors[right_name].reindex(union, fill_value=0.0)
                if left.nunique() > 1 and right.nunique() > 1:
                    correlations.append(float(spearmanr(left, right).statistic))
        rows.append(
            {
                "endpoint": endpoint,
                "method": method,
                "n_replicates": len(vectors),
                "n_pairwise_comparisons": len(correlations),
                "median_pairwise_spearman": float(np.median(correlations)) if correlations else np.nan,
                "q25_pairwise_spearman": float(np.quantile(correlations, 0.25)) if correlations else np.nan,
                "q75_pairwise_spearman": float(np.quantile(correlations, 0.75)) if correlations else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _functional_group_matrix(structures: pd.DataFrame) -> pd.DataFrame:
    from rdkit import Chem

    patterns = {
        "aromatic_ring": "a1aaaaa1",
        "heteroaromatic_atom": "[n,o,s]",
        "halogenated": "[F,Cl,Br,I]",
        "carboxylic_acid": "C(=O)[O;H1,-1]",
        "ester": "C(=O)O[#6]",
        "amide": "C(=O)N",
        "amine": "[N;!$(N-C=O);!$(N-S=O)]",
        "sulfur_oxidized": "S(=O)",
        "phosphate": "P(=O)(O)O",
        "nitrile": "C#N",
        "phenol": "c[O;H1]",
    }
    queries = {name: Chem.MolFromSmarts(smarts) for name, smarts in patterns.items()}
    rows: list[dict[str, Any]] = []
    for record in structures[["inchikey_2d", "canonical_smiles"]].drop_duplicates().itertuples(index=False):
        molecule = Chem.MolFromSmiles(str(record.canonical_smiles)) if pd.notna(record.canonical_smiles) else None
        row: dict[str, Any] = {"inchikey_2d": str(record.inchikey_2d)}
        for name, query in queries.items():
            row[name] = bool(molecule is not None and query is not None and molecule.HasSubstructMatch(query))
        rows.append(row)
    return pd.DataFrame(rows)


def build_fragment_structure_links(
    root: Path, attributions: pd.DataFrame, selections: pd.DataFrame
) -> pd.DataFrame:
    relevant = attributions[
        attributions["feature_type"].isin(["fragment_peak", "fragment_mz_bin"])
        & ~attributions["method"].eq("attention_rollout")
    ].copy()
    if relevant.empty:
        return pd.DataFrame()
    relevant["mz_bin"] = np.floor(relevant["mz_or_bin"].astype(float)).astype(int)
    structures = pd.read_csv(
        root / "outputs" / "pbt_msms_v1" / "split_v1" / "development_compounds.csv",
        low_memory=False,
        usecols=["inchikey_2d", "canonical_smiles"],
    )
    structures["inchikey_2d"] = structures["inchikey_2d"].astype(str)
    group_matrix = _functional_group_matrix(structures)
    selected_keys = selections[["endpoint", "inchikey_2d"]].drop_duplicates()
    selected_keys = selected_keys.merge(group_matrix, on="inchikey_2d", how="left")
    structure_columns = [column for column in group_matrix if column != "inchikey_2d"]
    rows: list[dict[str, Any]] = []

    for (endpoint, method), part in relevant.groupby(["endpoint", "method"], sort=True):
        importance = (
            part.groupby("mz_bin")["signed_attribution"]
            .apply(lambda values: float(np.mean(np.abs(values))))
            .sort_values(ascending=False)
        )
        top_bins = importance.head(20).index
        endpoint_keys = selected_keys[selected_keys["endpoint"].eq(endpoint)]
        for mz_bin in top_bins:
            bin_values = (
                part[part["mz_bin"].eq(mz_bin)]
                .groupby("inchikey_2d", as_index=False)
                .agg(
                    signed_attribution=("signed_attribution", "mean"),
                    absolute_attribution=("signed_attribution", lambda x: float(np.mean(np.abs(x)))),
                )
            )
            complete = endpoint_keys.merge(bin_values, on="inchikey_2d", how="left")
            complete[["signed_attribution", "absolute_attribution"]] = complete[
                ["signed_attribution", "absolute_attribution"]
            ].fillna(0.0)
            for group_name in structure_columns:
                present = complete[group_name].fillna(False).astype(bool)
                if present.sum() < 5 or (~present).sum() < 5:
                    continue
                yes = complete.loc[present, "absolute_attribution"].to_numpy(float)
                no = complete.loc[~present, "absolute_attribution"].to_numpy(float)
                test = mannwhitneyu(yes, no, alternative="two-sided")
                rows.append(
                    {
                        "endpoint": endpoint,
                        "endpoint_short": TASKS[endpoint]["short"],
                        "method": method,
                        "mz_bin": int(mz_bin),
                        "structure_group": group_name,
                        "n_group_present": int(present.sum()),
                        "n_group_absent": int((~present).sum()),
                        "median_abs_attribution_present": float(np.median(yes)),
                        "median_abs_attribution_absent": float(np.median(no)),
                        "delta_median_abs_attribution": float(np.median(yes) - np.median(no)),
                        "median_signed_attribution_present": float(
                            np.median(complete.loc[present, "signed_attribution"])
                        ),
                        "mannwhitney_p": float(test.pvalue),
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr_q"] = benjamini_hochberg(result["mannwhitney_p"])
        result = result.sort_values(
            ["endpoint", "method", "fdr_q", "delta_median_abs_attribution"],
            ascending=[True, True, True, False],
        )
    return result


def collect_negative_controls(root: Path) -> pd.DataFrame:
    path = (
        root
        / "outputs"
        / "pbt_msms_v1"
        / "simple_spectrum_baselines_v1"
        / "simple_spectrum_label_permutation_summary.csv"
    )
    frame = pd.read_csv(path)
    frame = frame[frame["task"].isin(["T_EC50_MG_L_REGRESSION", "T_LC50_MG_L_REGRESSION"])].copy()
    frame["control_scope"] = "frozen binned MS/MS label permutation"
    return frame


def _preflight(root: Path, out: Path) -> pd.DataFrame:
    base = root / "outputs" / "pbt_msms_v1"
    required = {
        "Transformer B checkpoints": list(
            (base / "transformer_finetuning_v1" / "checkpoints").rglob(
                "B_real_pretrain__seed_*__best.pt"
            )
        ),
        "Transformer tokens": [
            base / "transformer_data_v1" / "token_caches" / "real_development_tokens.npz"
        ],
        "Binned feature matrices": [
            base / "simple_spectrum_baselines_v1" / "fragment_intensity_1Da.npz",
            base / "simple_spectrum_baselines_v1" / "neutral_loss_intensity_1Da.npz",
            base / "simple_spectrum_baselines_v1" / "spectrum_metadata.npy",
        ],
        "DreaMS compound embeddings": [
            base / "dreams_heads_v1" / "dreams_compound_embedding_routes.npz"
        ],
        "DreaMS spectrum embeddings": [
            base / "dreams_v1" / "batch_embeddings_v1" / "development_dreams_embeddings.npy"
        ],
        "DreaMS frozen checkpoint": [
            root / "third_party" / "DreaMS" / "dreams" / "models" / "pretrained" / "embedding_model.ckpt"
        ],
        "env_dreams": [Path(r"D:\Appinstallation\Anaconda\envs\env_dreams\python.exe")],
    }
    rows = []
    for component, paths in required.items():
        rows.append(
            {
                "component": component,
                "available": bool(paths) and all(Path(path).is_file() for path in paths),
                "count": len(paths),
                "paths": " | ".join(str(path) for path in paths),
            }
        )
    frame = pd.DataFrame(rows)
    write_csv(out / "preflight_status.csv", frame)
    if not frame["available"].all():
        missing = frame.loc[~frame["available"], "component"].tolist()
        raise FileNotFoundError(f"Interpretability preflight failed: {missing}")
    return frame


def _mode_gate(out: Path, config: RunConfig) -> None:
    if config.mode != "full":
        return
    pilot_manifest = out / "run_manifest_pilot.json"
    if not pilot_manifest.is_file():
        raise RuntimeError("Full mode is locked until a successful pilot run exists")
    pilot = json.loads(pilot_manifest.read_text(encoding="utf-8"))
    if pilot.get("status") != "PASS":
        raise RuntimeError("Full mode is locked because the pilot manifest is not PASS")


def run_pipeline(
    root: str | Path | None = None,
    *,
    mode: str = "pilot",
    run_dreams_peak_occlusion: bool = True,
    stages: Sequence[str] = ("binned", "transformer", "dreams", "summarize"),
) -> dict[str, Any]:
    root = locate_root(root)
    out = output_directory(root)
    config = RunConfig.for_mode(
        mode, run_dreams_peak_occlusion=run_dreams_peak_occlusion
    )
    _mode_gate(out, config)
    preflight = _preflight(root, out)
    started = utc_now()
    manifest_path = out / f"run_manifest_{config.mode}.json"
    work_dir = out / f"work_{config.mode}"
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "run_version": "PBT_MSMS_INTERPRETABILITY_V1",
        "started_utc": started,
        "status": "RUNNING",
        "config": asdict(config),
        "stages": list(stages),
        "external_test_labels_used": False,
        "root": str(root),
        "output_directory": str(out),
    }
    write_json(manifest_path, manifest)
    attribution_frames: list[pd.DataFrame] = []
    reproduction_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    try:
        if "binned" in stages:
            attribution, reproduction, selection = run_binned_analysis(root, config)
            attribution_frames.append(attribution)
            reproduction_frames.append(reproduction)
            selection_frames.append(selection)
        if "transformer" in stages:
            attribution, reproduction, selection = run_transformer_analysis(root, config)
            attribution_frames.append(attribution)
            reproduction_frames.append(reproduction)
            selection_frames.append(selection)
        if "dreams" in stages:
            attribution, reproduction, selection = run_dreams_analysis(root, config, work_dir)
            attribution_frames.append(attribution)
            reproduction_frames.append(reproduction)
            selection_frames.append(selection)

        attributions = (
            pd.concat(attribution_frames, ignore_index=True)
            if attribution_frames
            else pd.DataFrame(columns=ATTRIBUTION_COLUMNS)
        )
        reproduction = (
            pd.concat(reproduction_frames, ignore_index=True)
            if reproduction_frames
            else pd.DataFrame()
        )
        selections = (
            pd.concat(selection_frames, ignore_index=True)
            if selection_frames
            else pd.DataFrame()
        )
        write_csv(out / f"attribution_records_{config.mode}.csv", attributions)
        write_csv(out / f"model_reproduction_qc_{config.mode}.csv", reproduction)
        write_csv(out / f"selected_compounds_{config.mode}.csv", selections)

        if "summarize" in stages:
            stability = summarize_stability(attributions)
            links = build_fragment_structure_links(root, attributions, selections)
            controls = collect_negative_controls(root)
            write_csv(out / f"attribution_stability_{config.mode}.csv", stability)
            write_csv(out / f"fragment_structure_property_links_{config.mode}.csv", links)
            write_csv(out / f"negative_controls_{config.mode}.csv", controls)
        else:
            stability = links = controls = pd.DataFrame()

        all_reproduced = bool(
            not reproduction.empty and reproduction["passed"].fillna(False).astype(bool).all()
        )
        manifest.update(
            {
                "completed_utc": utc_now(),
                "status": "PASS" if all_reproduced else "FAIL_REPRODUCTION_QC",
                "preflight_all_available": bool(preflight["available"].all()),
                "model_reproduction_all_passed": all_reproduced,
                "n_attribution_records": int(len(attributions)),
                "n_selected_endpoint_compounds": int(len(selections)),
                "n_stability_rows": int(len(stability)),
                "n_fragment_structure_links": int(len(links)),
                "outputs": {
                    "attributions": str(out / f"attribution_records_{config.mode}.csv"),
                    "reproduction_qc": str(out / f"model_reproduction_qc_{config.mode}.csv"),
                    "selected_compounds": str(out / f"selected_compounds_{config.mode}.csv"),
                    "stability": str(out / f"attribution_stability_{config.mode}.csv"),
                    "fragment_structure_links": str(
                        out / f"fragment_structure_property_links_{config.mode}.csv"
                    ),
                    "negative_controls": str(out / f"negative_controls_{config.mode}.csv"),
                },
            }
        )
        write_json(manifest_path, manifest)
        write_json(out / "run_manifest.json", manifest)
        if not all_reproduced:
            failed = reproduction.loc[~reproduction["passed"].fillna(False).astype(bool)]
            raise RuntimeError(
                "Frozen OOF reproduction failed; attribution outputs are retained for audit but full mode remains locked. "
                + failed[["model", "endpoint", "outer_fold", "seed"]].to_json(orient="records")
            )
        return manifest
    except Exception as error:
        manifest.update(
            {
                "completed_utc": utc_now(),
                "status": "ERROR",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        write_json(manifest_path, manifest)
        write_json(out / "run_manifest.json", manifest)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--mode", choices=["smoke", "pilot", "representative", "full"], default="pilot")
    parser.add_argument("--skip-dreams-peak-occlusion", action="store_true")
    parser.add_argument("--stages", nargs="+", default=["binned", "transformer", "dreams", "summarize"])
    parser.add_argument("--embed-mgf", type=Path)
    parser.add_argument("--embedding-output", type=Path)
    parser.add_argument("--dreams-batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    if args.embed_mgf is not None:
        if args.embedding_output is None:
            parser.error("--embedding-output is required with --embed-mgf")
        _dreams_embed_only(args.embed_mgf, args.embedding_output, args.dreams_batch_size)
        return 0
    manifest = run_pipeline(
        args.root,
        mode=args.mode,
        run_dreams_peak_occlusion=not args.skip_dreams_peak_occlusion,
        stages=args.stages,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




