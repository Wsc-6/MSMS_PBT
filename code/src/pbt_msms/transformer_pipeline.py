from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
import platform
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Required by CUDA/cuBLAS when deterministic algorithms are requested.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
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
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


SEED = 42
MAX_MZ = 2000.0
MAX_PEAKS = 120
N_OUTER_FOLDS = 5
N_PRETRAIN_PAIRS = 16_000
PRETRAIN_EPOCHS = 3
MAX_FINETUNE_EPOCHS = 18
EARLY_STOPPING_PATIENCE = 4
N_BOOTSTRAP = 1000

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
TASK_NAMES = list(TASKS)

ROUTES: dict[str, dict[str, Any]] = {
    "set_transformer_pretrained": {"pretrained": True, "variant": "full"},
    "set_transformer_scratch": {"pretrained": False, "variant": "full"},
    "set_transformer_no_neutral_loss": {
        "pretrained": True,
        "variant": "no_neutral_loss",
    },
    "set_transformer_no_collision_energy": {
        "pretrained": True,
        "variant": "no_collision_energy",
    },
}
MAIN_ROUTE = "set_transformer_pretrained"


def stable_seed(*parts: Any) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(map(str, values)):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def extract_record(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one trusted cleaned record to deployable peak and metadata features."""
    tensor = np.asarray(record["tensor"], dtype=np.float32)
    valid = ~np.asarray(record["padding_mask"], dtype=bool)
    valid[0] = False  # row zero is the precursor token, not a fragment peak
    mz = tensor[valid, 0].astype(np.float32)
    intensity = np.clip(tensor[valid, 3].astype(np.float32), 0.0, None)
    if intensity.size and float(intensity.max()) > 0:
        intensity = np.sqrt(intensity / float(intensity.max()))
    precursor = float(record.get("precursor_mz", np.nan))
    precursor_norm = precursor / MAX_MZ if np.isfinite(precursor) else 0.0
    neutral_loss = np.clip(precursor_norm - mz, 0.0, 1.0)
    keep = (
        np.isfinite(mz)
        & np.isfinite(intensity)
        & (mz >= 15.0 / MAX_MZ)
        & (mz < 1.0)
        & (intensity > 0)
    )
    peaks = np.stack([mz[keep], intensity[keep], neutral_loss[keep]], axis=1)
    if len(peaks) > MAX_PEAKS:
        chosen = np.argsort(peaks[:, 1])[-MAX_PEAKS:]
        peaks = peaks[chosen]
    if len(peaks):
        peaks = peaks[np.argsort(peaks[:, 0], kind="stable")]

    features = np.zeros((MAX_PEAKS, 3), dtype=np.float32)
    padding = np.ones(MAX_PEAKS, dtype=bool)
    features[: len(peaks)] = peaks
    padding[: len(peaks)] = False

    ce = record.get("ce_real", np.nan)
    try:
        ce = float(ce)
    except (TypeError, ValueError):
        ce = np.nan
    ce_missing = not np.isfinite(ce) or ce < 0
    ion_text = str(record.get("ionmode", "")).lower()
    polarity = 1.0 if "positive" in ion_text else (-1.0 if "negative" in ion_text else 0.0)
    metadata = np.asarray(
        [
            np.clip(precursor_norm, 0.0, 1.5),
            polarity,
            0.0 if ce_missing else np.clip(ce / 100.0, 0.0, 2.0),
            float(ce_missing),
            math.log1p(len(peaks)) / math.log1p(MAX_PEAKS),
            float(not np.isfinite(precursor)),
        ],
        dtype=np.float32,
    )
    return features, padding, metadata


def build_labeled_cache(
    spectra: pd.DataFrame,
    stream_path: Path,
    cache_path: Path,
) -> dict[str, np.ndarray]:
    features = np.zeros((len(spectra), MAX_PEAKS, 3), dtype=np.float16)
    padding = np.ones((len(spectra), MAX_PEAKS), dtype=bool)
    metadata = np.zeros((len(spectra), 6), dtype=np.float32)
    with stream_path.open("rb") as handle:
        for row_index, row in tqdm(
            spectra.iterrows(),
            total=len(spectra),
            desc="缓存Development峰token",
            mininterval=2.0,
        ):
            handle.seek(int(row["stream_offset"]))
            actual_key, record = pickle.load(handle)
            if actual_key != row["inchikey_2d"]:
                raise ValueError(f"Spectrum index mismatch at {row['spectrum_id']}")
            x, pad, meta = extract_record(record)
            features[row_index] = x.astype(np.float16)
            padding[row_index] = pad
            metadata[row_index] = meta
    np.savez_compressed(cache_path, features=features, padding=padding, metadata=metadata)
    return {"features": features, "padding": padding, "metadata": metadata}


def load_labeled_cache(cache_path: Path) -> dict[str, np.ndarray]:
    cached = np.load(cache_path)
    return {name: cached[name] for name in ["features", "padding", "metadata"]}


def build_pretraining_cache(
    stream_path: Path,
    index_path: Path,
    excluded_keys: set[str],
    cache_path: Path,
    key_path: Path,
    n_pairs: int,
    seed: int,
) -> dict[str, np.ndarray]:
    with index_path.open("rb") as handle:
        index: dict[str, list[int]] = pickle.load(handle)
    candidates = sorted(
        key for key, offsets in index.items() if key not in excluded_keys and len(offsets) >= 2
    )
    if len(candidates) < n_pairs:
        raise ValueError(f"Only {len(candidates)} background compounds have >=2 spectra")
    rng = np.random.default_rng(seed)
    selected = sorted(rng.choice(candidates, size=n_pairs, replace=False).tolist())
    requests: list[tuple[int, int, int, str]] = []
    for pair_index, key in enumerate(selected):
        offsets = np.asarray(index[key], dtype=np.int64)
        pair_offsets = rng.choice(offsets, size=2, replace=False)
        for side, offset in enumerate(pair_offsets):
            requests.append((int(offset), pair_index, side, key))
    requests.sort(key=lambda item: item[0])

    features = np.zeros((n_pairs, 2, MAX_PEAKS, 3), dtype=np.float16)
    padding = np.ones((n_pairs, 2, MAX_PEAKS), dtype=bool)
    metadata = np.zeros((n_pairs, 2, 6), dtype=np.float32)
    with stream_path.open("rb") as handle:
        for offset, pair_index, side, expected_key in tqdm(
            requests, desc="缓存无标签同结构谱图对", mininterval=2.0
        ):
            handle.seek(offset)
            actual_key, record = pickle.load(handle)
            if actual_key != expected_key:
                raise ValueError(f"Pretraining index mismatch at offset {offset}")
            x, pad, meta = extract_record(record)
            features[pair_index, side] = x.astype(np.float16)
            padding[pair_index, side] = pad
            metadata[pair_index, side] = meta
    np.savez_compressed(cache_path, features=features, padding=padding, metadata=metadata)
    pd.DataFrame({"pair_index": np.arange(n_pairs), "inchikey_2d": selected}).to_csv(
        key_path, index=False
    )
    return {"features": features, "padding": padding, "metadata": metadata}


def load_pretraining_cache(cache_path: Path) -> dict[str, np.ndarray]:
    cached = np.load(cache_path)
    return {name: cached[name] for name in ["features", "padding", "metadata"]}


def apply_variant(
    features: np.ndarray, metadata: np.ndarray, variant: str
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32).copy()
    metadata = np.asarray(metadata, dtype=np.float32).copy()
    if variant == "no_neutral_loss":
        features[..., 2] = 0.0
    elif variant == "no_collision_energy":
        metadata[..., 2:4] = 0.0
    elif variant != "full":
        raise KeyError(variant)
    return features, metadata


class PairPretrainingDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], seed: int = SEED):
        self.arrays = arrays
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.arrays["features"])

    def __getitem__(self, index: int):
        x = self.arrays["features"][index].astype(np.float32)
        pad = self.arrays["padding"][index].copy()
        meta = self.arrays["metadata"][index].astype(np.float32)
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index * 37)
        for side in range(2):
            valid = np.where(~pad[side])[0]
            if len(valid) > 4:
                drop = valid[rng.random(len(valid)) < 0.08]
                if len(valid) - len(drop) >= 3:
                    pad[side, drop] = True
            remaining = np.where(~pad[side])[0]
            if len(remaining):
                x[side, remaining, 1] = np.clip(
                    x[side, remaining, 1]
                    * np.exp(rng.normal(0.0, 0.03, len(remaining))),
                    0.0,
                    1.0,
                )
                permuted = rng.permutation(remaining)
                x[side, remaining] = x[side, permuted]
        return torch.from_numpy(x), torch.from_numpy(pad), torch.from_numpy(meta)


class CompoundSpectrumDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        spectra: pd.DataFrame,
        compounds: pd.DataFrame,
        keys: list[str],
        variant: str,
        augment: bool,
        seed: int,
    ):
        self.arrays = arrays
        self.variant = variant
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.keys = sorted(keys)
        self.rows_by_key = {
            key: group["feature_row"].to_numpy(dtype=int)
            for key, group in spectra[spectra["inchikey_2d"].isin(self.keys)].groupby(
                "inchikey_2d", sort=True
            )
        }
        label_frame = compounds.set_index("inchikey_2d")
        self.labels = np.asarray(
            [
                [pd.to_numeric(label_frame.at[key, TASKS[task]["column"]], errors="coerce") for task in TASK_NAMES]
                for key in self.keys
            ],
            dtype=np.float32,
        )
        missing = [key for key in self.keys if key not in self.rows_by_key]
        if missing:
            raise ValueError(f"Compounds without spectra: {missing[:5]}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index * 9176)
        row = int(rng.choice(self.rows_by_key[self.keys[index]]))
        x = self.arrays["features"][row].astype(np.float32)
        pad = self.arrays["padding"][row].copy()
        meta = self.arrays["metadata"][row].astype(np.float32)
        x, meta = apply_variant(x, meta, self.variant)
        if self.augment:
            valid = np.where(~pad)[0]
            if len(valid) > 4:
                drop = valid[rng.random(len(valid)) < 0.08]
                if len(valid) - len(drop) >= 3:
                    pad[drop] = True
            remaining = np.where(~pad)[0]
            if len(remaining):
                x[remaining, 1] = np.clip(
                    x[remaining, 1] * np.exp(rng.normal(0.0, 0.03, len(remaining))),
                    0.0,
                    1.0,
                )
                x[remaining] = x[rng.permutation(remaining)]
        return (
            torch.from_numpy(x),
            torch.from_numpy(pad),
            torch.from_numpy(meta),
            torch.from_numpy(self.labels[index]),
        )


class SpectrumRowDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        row_indices: np.ndarray,
        variant: str,
    ):
        self.arrays = arrays
        self.row_indices = np.asarray(row_indices, dtype=int)
        self.variant = variant

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, index: int):
        row = int(self.row_indices[index])
        x, meta = apply_variant(
            self.arrays["features"][row], self.arrays["metadata"][row], self.variant
        )
        return (
            torch.from_numpy(x),
            torch.from_numpy(self.arrays["padding"][row].copy()),
            torch.from_numpy(meta),
            row,
        )


class FourierPeakEmbedding(nn.Module):
    def __init__(self, d_model: int, n_frequencies: int = 8):
        super().__init__()
        frequencies = math.pi * (2.0 ** torch.arange(n_frequencies, dtype=torch.float32))
        self.register_buffer("frequencies", frequencies, persistent=True)
        input_dim = 4 * n_frequencies + 2
        self.network = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, features: torch.Tensor, mask_flag: torch.Tensor | None = None):
        mz = features[..., 0:1]
        intensity = features[..., 1:2]
        neutral_loss = features[..., 2:3]
        if mask_flag is None:
            mask_flag = torch.zeros_like(intensity)
        else:
            mask_flag = mask_flag.unsqueeze(-1).to(features.dtype)
        mz_phase = mz * self.frequencies
        neutral_phase = neutral_loss * self.frequencies
        encoded = torch.cat(
            [
                torch.sin(mz_phase),
                torch.cos(mz_phase),
                torch.sin(neutral_phase),
                torch.cos(neutral_phase),
                intensity,
                mask_flag,
            ],
            dim=-1,
        )
        return self.network(encoded)


class PeakSetEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.peak_embedding = FourierPeakEmbedding(d_model)
        self.metadata_embedding = nn.Sequential(
            nn.Linear(6, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3), nn.Sigmoid()
        )

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
        metadata: torch.Tensor,
        mask_flag: torch.Tensor | None = None,
        return_tokens: bool = False,
    ):
        peak_tokens = self.peak_embedding(features, mask_flag)
        cls = self.cls_token.expand(len(features), -1, -1) + self.metadata_embedding(metadata).unsqueeze(1)
        tokens = torch.cat([cls, peak_tokens], dim=1)
        cls_padding = torch.zeros((len(features), 1), dtype=torch.bool, device=padding_mask.device)
        full_padding = torch.cat([cls_padding, padding_mask], dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=full_padding)
        embedding = encoded[:, 0]
        if return_tokens:
            return embedding, encoded[:, 1:]
        return embedding


class MultiTaskSpectrumModel(nn.Module):
    def __init__(self, encoder: PeakSetEncoder):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(encoder.d_model, encoder.d_model // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(encoder.d_model // 2, 1),
                )
                for task in TASK_NAMES
            }
        )

    def forward(self, features, padding_mask, metadata):
        embedding = self.encoder(features, padding_mask, metadata)
        return torch.cat([self.heads[task](embedding) for task in TASK_NAMES], dim=1)


def make_encoder(model_config: dict[str, Any]) -> PeakSetEncoder:
    return PeakSetEncoder(
        d_model=int(model_config["d_model"]),
        nhead=int(model_config["nhead"]),
        num_layers=int(model_config["num_layers"]),
        dim_feedforward=int(model_config["dim_feedforward"]),
        dropout=float(model_config["dropout"]),
    )


def _autocast(device: torch.device):
    return torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda")


def pretrain_encoder(
    encoder: PeakSetEncoder,
    dataset: PairPretrainingDataset,
    device: torch.device,
    epochs: int = PRETRAIN_EPOCHS,
    batch_size: int = 32,
    seed: int = SEED,
) -> pd.DataFrame:
    encoder.to(device)
    projection = nn.Sequential(
        nn.Linear(encoder.d_model, encoder.d_model),
        nn.GELU(),
        nn.Linear(encoder.d_model, 64),
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projection.parameters()), lr=2e-4, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history = []
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
            generator=generator,
        )
        encoder.train()
        projection.train()
        total_loss = total_reconstruction = total_contrastive = 0.0
        n_batches = 0
        progress = tqdm(loader, desc=f"自监督预训练 epoch {epoch + 1}/{epochs}", mininterval=2.0)
        for features, padding, metadata in progress:
            batch_size_actual = len(features)
            features = features.reshape(-1, MAX_PEAKS, 3).to(device, non_blocking=True)
            padding = padding.reshape(-1, MAX_PEAKS).to(device, non_blocking=True)
            metadata = metadata.reshape(-1, 6).to(device, non_blocking=True)
            mask_flag = (torch.rand(padding.shape, device=device) < 0.15) & ~padding
            masked_features = features.clone()
            masked_features[mask_flag] = 0.0
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                embedding, peak_tokens = encoder(
                    masked_features,
                    padding,
                    metadata,
                    mask_flag=mask_flag,
                    return_tokens=True,
                )
                reconstructed = encoder.reconstruction_head(peak_tokens)
                reconstruction_loss = nn.functional.smooth_l1_loss(
                    reconstructed[mask_flag], features[mask_flag]
                )
                projected = nn.functional.normalize(projection(embedding), dim=1)
                first = projected[0::2]
                second = projected[1::2]
                logits = first @ second.T / 0.1
                labels = torch.arange(batch_size_actual, device=device)
                contrastive_loss = 0.5 * (
                    nn.functional.cross_entropy(logits, labels)
                    + nn.functional.cross_entropy(logits.T, labels)
                )
                loss = reconstruction_loss + 0.2 * contrastive_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(projection.parameters()), max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            total_reconstruction += float(reconstruction_loss.detach())
            total_contrastive += float(contrastive_loss.detach())
            n_batches += 1
            progress.set_postfix(loss=f"{total_loss / n_batches:.4f}")
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / n_batches,
                "reconstruction_loss": total_reconstruction / n_batches,
                "contrastive_loss": total_contrastive / n_batches,
            }
        )
    return pd.DataFrame(history)


@dataclass
class LabelStatistics:
    mean: np.ndarray
    std: np.ndarray
    positive_weight: float


def compute_label_statistics(compounds: pd.DataFrame, keys: list[str]) -> LabelStatistics:
    frame = compounds[compounds["inchikey_2d"].isin(keys)]
    means = np.zeros(len(TASK_NAMES), dtype=np.float32)
    stds = np.ones(len(TASK_NAMES), dtype=np.float32)
    for index, task in enumerate(TASK_NAMES):
        if TASKS[task]["kind"] == "regression":
            values = pd.to_numeric(frame[TASKS[task]["column"]], errors="coerce").dropna()
            means[index] = float(values.mean())
            stds[index] = max(float(values.std(ddof=0)), 1e-6)
    p_values = pd.to_numeric(
        frame[TASKS["P_screen_not_ready_binary"]["column"]], errors="coerce"
    ).dropna()
    positives = int(p_values.eq(1).sum())
    negatives = int(p_values.eq(0).sum())
    positive_weight = negatives / positives if positives else 1.0
    return LabelStatistics(means, stds, float(positive_weight))


def supervised_loss(
    outputs: torch.Tensor, labels: torch.Tensor, stats: LabelStatistics
) -> tuple[torch.Tensor, int]:
    losses = []
    observed_tasks = 0
    for index, task in enumerate(TASK_NAMES):
        observed = torch.isfinite(labels[:, index])
        if not observed.any():
            continue
        observed_tasks += 1
        if TASKS[task]["kind"] == "classification":
            pos_weight = torch.tensor(stats.positive_weight, device=outputs.device)
            task_loss = nn.functional.binary_cross_entropy_with_logits(
                outputs[observed, index], labels[observed, index], pos_weight=pos_weight
            )
        else:
            target = (labels[observed, index] - stats.mean[index]) / stats.std[index]
            task_loss = nn.functional.smooth_l1_loss(outputs[observed, index], target)
        losses.append(task_loss)
    if not losses:
        raise ValueError("A supervised batch contains no observed targets")
    return torch.stack(losses).mean(), observed_tasks


def evaluate_supervised_loss(
    model: MultiTaskSpectrumModel,
    dataset: CompoundSpectrumDataset,
    stats: LabelStatistics,
    device: torch.device,
    batch_size: int,
) -> float:
    dataset.set_epoch(0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    losses = []
    with torch.no_grad():
        for features, padding, metadata, labels in loader:
            features = features.to(device)
            padding = padding.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)
            with _autocast(device):
                outputs = model(features, padding, metadata)
                loss, _ = supervised_loss(outputs, labels, stats)
            losses.append(float(loss))
    return float(np.mean(losses))


def train_supervised(
    model: MultiTaskSpectrumModel,
    train_dataset: CompoundSpectrumDataset,
    stats: LabelStatistics,
    device: torch.device,
    max_epochs: int,
    batch_size: int,
    seed: int,
    validation_dataset: CompoundSpectrumDataset | None = None,
    patience: int | None = None,
) -> tuple[pd.DataFrame, int]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history = []
    best_epoch = max_epochs
    best_value = float("inf")
    stale = 0
    for epoch in range(max_epochs):
        train_dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
            generator=generator,
        )
        model.train()
        running = []
        for features, padding, metadata, labels in loader:
            features = features.to(device, non_blocking=True)
            padding = padding.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                outputs = model(features, padding, metadata)
                loss, _ = supervised_loss(outputs, labels, stats)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running.append(float(loss.detach()))
        validation_loss = (
            evaluate_supervised_loss(model, validation_dataset, stats, device, batch_size)
            if validation_dataset is not None
            else np.nan
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(running)),
                "validation_loss": validation_loss,
            }
        )
        if validation_dataset is not None:
            if validation_loss < best_value - 1e-5:
                best_value = validation_loss
                best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1
                if patience is not None and stale >= patience:
                    break
    return pd.DataFrame(history), int(best_epoch)


def predict_rows(
    model: MultiTaskSpectrumModel,
    arrays: dict[str, np.ndarray],
    row_indices: np.ndarray,
    variant: str,
    stats: LabelStatistics,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = SpectrumRowDataset(arrays, row_indices, variant)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    raw_outputs, observed_rows = [], []
    model.eval()
    with torch.no_grad():
        for features, padding, metadata, rows in loader:
            with _autocast(device):
                output = model(
                    features.to(device, non_blocking=True),
                    padding.to(device, non_blocking=True),
                    metadata.to(device, non_blocking=True),
                )
            raw_outputs.append(output.float().cpu().numpy())
            observed_rows.append(rows.numpy())
    raw = np.concatenate(raw_outputs)
    rows = np.concatenate(observed_rows).astype(int)
    converted = raw.copy()
    for index, task in enumerate(TASK_NAMES):
        if TASKS[task]["kind"] == "classification":
            converted[:, index] = 1.0 / (1.0 + np.exp(-raw[:, index]))
        else:
            converted[:, index] = raw[:, index] * stats.std[index] + stats.mean[index]
    return rows, converted


def classification_metrics(y_true, probability) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "brier": float(brier_score_loss(y_true, probability)),
    }


def regression_metrics(y_true, prediction) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "r2": float(r2_score(y_true, prediction)),
        "spearman": float(spearmanr(y_true, prediction).statistic),
    }


def metric_table(compound_oof: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows = []
    for (route, task, kind), frame in compound_oof.groupby(
        ["representation_route", "task", "kind"], sort=True
    ):
        observed = (
            classification_metrics(frame["y_true"], frame["y_probability"])
            if kind == "classification"
            else regression_metrics(frame["y_true"], frame["y_prediction"])
        )
        metric_names = list(observed)
        bootstrap_values = {name: [] for name in metric_names}
        rng = np.random.default_rng(stable_seed(seed, route, task, "bootstrap"))
        for _ in range(n_bootstrap):
            sample = frame.iloc[rng.integers(0, len(frame), len(frame))]
            if kind == "classification" and sample["y_true"].nunique() < 2:
                continue
            result = (
                classification_metrics(sample["y_true"], sample["y_probability"])
                if kind == "classification"
                else regression_metrics(sample["y_true"], sample["y_prediction"])
            )
            for name, value in result.items():
                if np.isfinite(value):
                    bootstrap_values[name].append(value)
        row = {
            "representation_route": route,
            "task": task,
            "kind": kind,
            "n_compounds": len(frame),
            **observed,
        }
        for name, values in bootstrap_values.items():
            row[f"{name}_ci_low"] = float(np.quantile(values, 0.025))
            row[f"{name}_ci_high"] = float(np.quantile(values, 0.975))
        primary = "pr_auc" if kind == "classification" else "mae"
        row.update(
            {
                "primary_metric": primary,
                "primary_value": observed[primary],
                "primary_ci_low": row[f"{primary}_ci_low"],
                "primary_ci_high": row[f"{primary}_ci_high"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(
    main: pd.DataFrame,
    comparator: pd.DataFrame,
    comparator_name: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for task, spec in TASKS.items():
        left = main[main["task"].eq(task)]
        right = comparator[comparator["task"].eq(task)]
        paired = left.merge(
            right,
            on=["inchikey_2d", "task", "kind", "outer_fold", "y_true"],
            suffixes=("_transformer", "_comparator"),
            validate="one_to_one",
        )
        if spec["kind"] == "classification":
            observed = average_precision_score(
                paired["y_true"], paired["y_probability_transformer"]
            ) - average_precision_score(paired["y_true"], paired["y_probability_comparator"])
            definition = "transformer_PR_minus_comparator_PR"
        else:
            observed = mean_absolute_error(
                paired["y_true"], paired["y_prediction_comparator"]
            ) - mean_absolute_error(paired["y_true"], paired["y_prediction_transformer"])
            definition = "comparator_MAE_minus_transformer_MAE"
        rng = np.random.default_rng(stable_seed(seed, comparator_name, task))
        deltas = []
        for _ in range(n_bootstrap):
            sample = paired.iloc[rng.integers(0, len(paired), len(paired))]
            if spec["kind"] == "classification":
                if sample["y_true"].nunique() < 2:
                    continue
                delta = average_precision_score(
                    sample["y_true"], sample["y_probability_transformer"]
                ) - average_precision_score(
                    sample["y_true"], sample["y_probability_comparator"]
                )
            else:
                delta = mean_absolute_error(
                    sample["y_true"], sample["y_prediction_comparator"]
                ) - mean_absolute_error(sample["y_true"], sample["y_prediction_transformer"])
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
                "bootstrap_p_one_sided_transformer_not_better": float(
                    (1 + np.sum(np.asarray(deltas) <= 0)) / (1 + len(deltas))
                ),
            }
        )
    return pd.DataFrame(rows)


def _model_from_initialization(
    model_config: dict[str, Any],
    pretrained_state: dict[str, torch.Tensor] | None,
    seed: int,
) -> MultiTaskSpectrumModel:
    set_global_seed(seed)
    encoder = make_encoder(model_config)
    if pretrained_state is not None:
        encoder.load_state_dict(pretrained_state, strict=True)
    return MultiTaskSpectrumModel(encoder)


def _aggregate_predictions(
    spectra: pd.DataFrame,
    compounds: pd.DataFrame,
    row_indices: np.ndarray,
    predictions: np.ndarray,
    route: str,
    outer_fold: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    prediction_frame = spectra.iloc[row_indices][
        ["feature_row", "spectrum_id", "inchikey_2d"]
    ].copy()
    prediction_frame = prediction_frame.reset_index(drop=True)
    compound_lookup = compounds.set_index("inchikey_2d")
    compound_frames, spectrum_frames = [], []
    for task_index, task in enumerate(TASK_NAMES):
        label_column = TASKS[task]["column"]
        eligible_keys = set(
            compounds.loc[
                compounds["cv_fold"].eq(outer_fold)
                & pd.to_numeric(compounds[label_column], errors="coerce").notna(),
                "inchikey_2d",
            ]
        )
        current = prediction_frame[prediction_frame["inchikey_2d"].isin(eligible_keys)].copy()
        current["y_true"] = current["inchikey_2d"].map(compound_lookup[label_column]).astype(float)
        current["outer_fold"] = outer_fold
        current["task"] = task
        current["kind"] = TASKS[task]["kind"]
        if TASKS[task]["kind"] == "classification":
            current["y_probability"] = predictions[current.index.to_numpy(), task_index]
            current["y_prediction"] = (current["y_probability"] >= 0.5).astype(float)
        else:
            current["y_prediction"] = predictions[current.index.to_numpy(), task_index]
            current["y_probability"] = np.nan
        if route == MAIN_ROUTE:
            spectrum_frames.append(current.copy())
        aggregate_rows = []
        for key, group in current.sort_values(["inchikey_2d", "spectrum_id"]).groupby(
            "inchikey_2d", sort=True
        ):
            first = group.iloc[0]
            if TASKS[task]["kind"] == "classification":
                probability = float(group["y_probability"].median())
                prediction = float(probability >= 0.5)
                dispersion = float(group["y_probability"].std(ddof=0))
                single_prediction = float(first["y_probability"] >= 0.5)
                single_probability = float(first["y_probability"])
            else:
                prediction = float(group["y_prediction"].median())
                probability = np.nan
                dispersion = float(group["y_prediction"].std(ddof=0))
                single_prediction = float(first["y_prediction"])
                single_probability = np.nan
            aggregate_rows.append(
                {
                    "inchikey_2d": key,
                    "outer_fold": outer_fold,
                    "y_true": float(first["y_true"]),
                    "n_spectra": len(group),
                    "y_prediction": prediction,
                    "y_probability": probability,
                    "spectrum_prediction_sd": dispersion,
                    "single_y_prediction": single_prediction,
                    "single_y_probability": single_probability,
                    "single_spectrum_id": first["spectrum_id"],
                    "representation_route": route,
                    "task": task,
                    "kind": TASKS[task]["kind"],
                }
            )
        compound_frames.append(pd.DataFrame(aggregate_rows))
    return compound_frames, spectrum_frames


def _permutation_invariance_difference(
    model: MultiTaskSpectrumModel,
    arrays: dict[str, np.ndarray],
    rows: np.ndarray,
    variant: str,
    device: torch.device,
) -> float:
    rows = np.asarray(rows[: min(32, len(rows))], dtype=int)
    features = arrays["features"][rows].astype(np.float32)
    padding = arrays["padding"][rows].copy()
    metadata = arrays["metadata"][rows].astype(np.float32)
    features, metadata = apply_variant(features, metadata, variant)
    permuted = features.copy()
    rng = np.random.default_rng(SEED + 88_000)
    for index in range(len(rows)):
        valid = np.where(~padding[index])[0]
        permuted[index, valid] = features[index, rng.permutation(valid)]
    model.eval()
    # The architecture is permutation invariant.  Verify this property in FP32;
    # FP16 attention can differ at about 1e-3 solely from reduction-order rounding.
    with torch.no_grad():
        first = model(
            torch.from_numpy(features).to(device),
            torch.from_numpy(padding).to(device),
            torch.from_numpy(metadata).to(device),
        )
        second = model(
            torch.from_numpy(permuted).to(device),
            torch.from_numpy(padding).to(device),
            torch.from_numpy(metadata).to(device),
        )
    return float(torch.max(torch.abs(first.float() - second.float())).cpu())


def run_pipeline(root: Path, model_config: dict[str, Any]) -> dict[str, Any]:
    root = Path(root).resolve()
    output_dir = root / "outputs" / "pbt_msms_v1" / "set_transformer_v1"
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "development": root / "outputs" / "pbt_msms_v1" / "split_v1" / "development_compounds.csv",
        "split_manifest": root / "outputs" / "pbt_msms_v1" / "split_v1" / "compound_split_manifest.csv",
        "split_freeze": root / "outputs" / "pbt_msms_v1" / "split_v1" / "split_freeze_manifest.json",
        "split_qc": root / "outputs" / "pbt_msms_v1" / "split_v1" / "leakage_checks.csv",
        "public_manifest": root / "outputs" / "pbt_msms_v1" / "spectrum_audit_v1" / "public_relevant_spectrum_manifest.pkl",
        "spectrum_audit": root / "outputs" / "pbt_msms_v1" / "spectrum_audit_v1" / "spectrum_audit_manifest.json",
        "spectrum_qc": root / "outputs" / "pbt_msms_v1" / "spectrum_audit_v1" / "spectrum_audit_qc_checks.csv",
        "public_stream": root / "质谱数据" / "EnvMolBench_Cleaned.pkl",
        "public_index": root / "质谱数据" / "EnvMolBench_Cleaned_Index.pkl",
        "simple_oof": root / "outputs" / "pbt_msms_v1" / "simple_spectrum_baselines_v1" / "simple_spectrum_compound_level_oof.pkl",
        "simple_manifest": root / "outputs" / "pbt_msms_v1" / "simple_spectrum_baselines_v1" / "simple_spectrum_run_manifest.json",
        "structure_best": root / "outputs" / "pbt_msms_v1" / "structure_baselines_v1" / "structure_baseline_best_models.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing transformer inputs:\n" + "\n".join(missing))

    set_global_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    development = pd.read_csv(paths["development"], low_memory=False)
    development["cv_fold"] = development["cv_fold"].astype(int)
    split_manifest = pd.read_csv(paths["split_manifest"], low_memory=False)
    split_qc = pd.read_csv(paths["split_qc"])
    spectrum_qc = pd.read_csv(paths["spectrum_qc"])
    public_manifest = pd.read_pickle(paths["public_manifest"])
    with paths["spectrum_audit"].open("r", encoding="utf-8") as handle:
        spectrum_audit = json.load(handle)
    with paths["simple_manifest"].open("r", encoding="utf-8") as handle:
        simple_manifest = json.load(handle)

    spectra = public_manifest.loc[public_manifest["primary_train_eligible_20ppm"]].copy()
    spectra = spectra.sort_values(["stream_offset", "spectrum_id"]).reset_index(drop=True)
    spectra["feature_row"] = np.arange(len(spectra), dtype=int)
    eligible_keys = set(spectra["inchikey_2d"])
    modeling = development[development["inchikey_2d"].isin(eligible_keys)].copy()
    modeling = modeling.sort_values("inchikey_2d").reset_index(drop=True)

    labeled_cache_path = output_dir / "development_peak_token_cache.npz"
    arrays = build_labeled_cache(spectra, paths["public_stream"], labeled_cache_path)
    spectra[
        [
            "feature_row",
            "spectrum_id",
            "inchikey_2d",
            "stream_offset",
            "cv_fold",
            "precursor_mz",
            "ion_mode",
            "collision_energy_value",
            "cleaned_peak_n",
        ]
    ].to_csv(output_dir / "development_peak_token_index.csv", index=False)

    external_keys = set(
        split_manifest.loc[
            split_manifest["chemvista_evaluation_role"].astype(str).ne("none"), "inchikey_2d"
        ].dropna()
    )
    excluded_pretraining_keys = set(development["inchikey_2d"]) | external_keys
    pretrain_cache_path = output_dir / "background_pretraining_pairs.npz"
    pretrain_key_path = output_dir / "background_pretraining_pair_keys.csv"
    pretrain_arrays = build_pretraining_cache(
        paths["public_stream"],
        paths["public_index"],
        excluded_pretraining_keys,
        pretrain_cache_path,
        pretrain_key_path,
        N_PRETRAIN_PAIRS,
        SEED,
    )
    pretrain_keys = set(pd.read_csv(pretrain_key_path)["inchikey_2d"])

    encoder = make_encoder(model_config)
    pretrain_history = pretrain_encoder(
        encoder,
        PairPretrainingDataset(pretrain_arrays),
        device,
        epochs=PRETRAIN_EPOCHS,
        batch_size=32,
        seed=SEED,
    )
    pretrained_state = {key: value.detach().cpu() for key, value in encoder.state_dict().items()}
    pretrained_checkpoint = checkpoint_dir / "background_pretrained_encoder.pt"
    torch.save(pretrained_state, pretrained_checkpoint)
    pretrain_history.to_csv(output_dir / "pretraining_history.csv", index=False)
    del encoder, pretrain_arrays
    if device.type == "cuda":
        torch.cuda.empty_cache()

    all_compound_frames: list[pd.DataFrame] = []
    main_spectrum_frames: list[pd.DataFrame] = []
    training_history_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    fold_integrity_rows: list[dict[str, Any]] = []
    permutation_differences: list[float] = []

    all_keys = modeling["inchikey_2d"].tolist()
    for route, route_config in ROUTES.items():
        variant = route_config["variant"]
        for outer_fold in range(N_OUTER_FOLDS):
            test_keys = modeling.loc[modeling["cv_fold"].eq(outer_fold), "inchikey_2d"].tolist()
            outer_train_keys = modeling.loc[modeling["cv_fold"].ne(outer_fold), "inchikey_2d"].tolist()
            inner_validation_fold = (outer_fold + 1) % N_OUTER_FOLDS
            inner_validation_keys = modeling.loc[
                modeling["cv_fold"].eq(inner_validation_fold), "inchikey_2d"
            ].tolist()
            inner_train_keys = modeling.loc[
                ~modeling["cv_fold"].isin([outer_fold, inner_validation_fold]), "inchikey_2d"
            ].tolist()
            fold_integrity_rows.append(
                {
                    "representation_route": route,
                    "outer_fold": outer_fold,
                    "inner_validation_fold": inner_validation_fold,
                    "n_inner_train": len(inner_train_keys),
                    "n_inner_validation": len(inner_validation_keys),
                    "n_outer_train": len(outer_train_keys),
                    "n_outer_test": len(test_keys),
                    "train_test_overlap": len(set(outer_train_keys) & set(test_keys)),
                    "inner_train_validation_overlap": len(
                        set(inner_train_keys) & set(inner_validation_keys)
                    ),
                }
            )

            initialization = pretrained_state if route_config["pretrained"] else None
            selection_seed = stable_seed(SEED, route, outer_fold, "selection")
            selection_model = _model_from_initialization(
                model_config, initialization, selection_seed
            )
            inner_stats = compute_label_statistics(modeling, inner_train_keys)
            inner_train_dataset = CompoundSpectrumDataset(
                arrays,
                spectra,
                modeling,
                inner_train_keys,
                variant,
                augment=True,
                seed=selection_seed,
            )
            inner_validation_dataset = CompoundSpectrumDataset(
                arrays,
                spectra,
                modeling,
                inner_validation_keys,
                variant,
                augment=False,
                seed=selection_seed + 1,
            )
            selection_history, best_epoch = train_supervised(
                selection_model,
                inner_train_dataset,
                inner_stats,
                device,
                max_epochs=MAX_FINETUNE_EPOCHS,
                batch_size=64,
                seed=selection_seed,
                validation_dataset=inner_validation_dataset,
                patience=EARLY_STOPPING_PATIENCE,
            )
            selection_history["representation_route"] = route
            selection_history["outer_fold"] = outer_fold
            selection_history["phase"] = "inner_epoch_selection"
            training_history_frames.append(selection_history)
            selection_rows.append(
                {
                    "representation_route": route,
                    "outer_fold": outer_fold,
                    "inner_validation_fold": inner_validation_fold,
                    "best_epoch": best_epoch,
                    "epochs_evaluated": len(selection_history),
                    "best_validation_loss": float(
                        selection_history.loc[
                            selection_history["epoch"].eq(best_epoch), "validation_loss"
                        ].iloc[0]
                    ),
                }
            )
            del selection_model, inner_train_dataset, inner_validation_dataset
            if device.type == "cuda":
                torch.cuda.empty_cache()

            final_seed = stable_seed(SEED, route, outer_fold, "final")
            final_model = _model_from_initialization(model_config, initialization, final_seed)
            outer_stats = compute_label_statistics(modeling, outer_train_keys)
            outer_train_dataset = CompoundSpectrumDataset(
                arrays,
                spectra,
                modeling,
                outer_train_keys,
                variant,
                augment=True,
                seed=final_seed,
            )
            final_history, _ = train_supervised(
                final_model,
                outer_train_dataset,
                outer_stats,
                device,
                max_epochs=best_epoch,
                batch_size=64,
                seed=final_seed,
                validation_dataset=None,
                patience=None,
            )
            final_history["representation_route"] = route
            final_history["outer_fold"] = outer_fold
            final_history["phase"] = "outer_train_refit"
            training_history_frames.append(final_history)
            checkpoint = checkpoint_dir / f"{route}_fold{outer_fold}.pt"
            torch.save(
                {
                    "model_state": {k: v.detach().cpu() for k, v in final_model.state_dict().items()},
                    "label_mean": outer_stats.mean,
                    "label_std": outer_stats.std,
                    "positive_weight": outer_stats.positive_weight,
                    "best_epoch": best_epoch,
                    "variant": variant,
                    "outer_fold": outer_fold,
                },
                checkpoint,
            )

            test_rows = spectra.loc[
                spectra["inchikey_2d"].isin(test_keys), "feature_row"
            ].to_numpy(dtype=int)
            predicted_rows, predictions = predict_rows(
                final_model,
                arrays,
                test_rows,
                variant,
                outer_stats,
                device,
                batch_size=256,
            )
            compound_frames, spectrum_frames = _aggregate_predictions(
                spectra,
                modeling,
                predicted_rows,
                predictions,
                route,
                outer_fold,
            )
            all_compound_frames.extend(compound_frames)
            main_spectrum_frames.extend(spectrum_frames)
            if route == MAIN_ROUTE:
                permutation_differences.append(
                    _permutation_invariance_difference(
                        final_model, arrays, test_rows, variant, device
                    )
                )
            del final_model, outer_train_dataset
            if device.type == "cuda":
                torch.cuda.empty_cache()

    compound_oof = pd.concat(all_compound_frames, ignore_index=True)
    spectrum_oof = pd.concat(main_spectrum_frames, ignore_index=True)
    training_history = pd.concat(training_history_frames, ignore_index=True)
    selection = pd.DataFrame(selection_rows)
    fold_integrity = pd.DataFrame(fold_integrity_rows)
    metrics = metric_table(compound_oof, N_BOOTSTRAP, SEED)

    simple_oof = pd.read_pickle(paths["simple_oof"])
    simple_main = simple_oof[simple_oof["representation_route"].eq("fragment_neutral_loss_metadata")]
    metadata_only = simple_oof[simple_oof["representation_route"].eq("metadata_only")]
    transformer_main = compound_oof[compound_oof["representation_route"].eq(MAIN_ROUTE)]
    transformer_scratch = compound_oof[
        compound_oof["representation_route"].eq("set_transformer_scratch")
    ]
    comparisons = pd.concat(
        [
            paired_bootstrap(
                transformer_main,
                simple_main,
                "simple_fragment_neutral_loss_metadata",
                N_BOOTSTRAP,
                SEED,
            ),
            paired_bootstrap(
                transformer_main, metadata_only, "metadata_only", N_BOOTSTRAP, SEED
            ),
            paired_bootstrap(
                transformer_main,
                transformer_scratch,
                "set_transformer_scratch",
                N_BOOTSTRAP,
                SEED,
            ),
        ],
        ignore_index=True,
    )

    structure_best = pd.read_csv(paths["structure_best"])
    transformer_comparison = metrics.copy()
    transformer_comparison["comparison_family"] = "set_transformer_internal_oof"
    transformer_comparison["model_or_route"] = transformer_comparison["representation_route"]
    structure_comparison = structure_best.copy()
    structure_comparison["comparison_family"] = "structure_upper_bound_nested_selected"
    structure_comparison["model_or_route"] = structure_comparison["representation"]
    comparison_columns = [
        "comparison_family",
        "model_or_route",
        "task",
        "kind",
        "n_compounds",
        "primary_metric",
        "primary_value",
        "primary_ci_low",
        "primary_ci_high",
    ]
    all_baselines = pd.concat(
        [transformer_comparison[comparison_columns], structure_comparison[comparison_columns]],
        ignore_index=True,
    )

    checks = []

    def add_check(check_id: str, description: str, passed: bool, details: str = ""):
        checks.append(
            {
                "check_id": check_id,
                "description": description,
                "status": "PASS" if passed else "FAIL",
                "details": details,
            }
        )

    add_check("TRANS001", "上游拆分与谱图审计无FAIL", not split_qc.status.eq("FAIL").any() and not spectrum_qc.status.eq("FAIL").any())
    add_check("TRANS002", "仅使用20ppm冻结Development谱图", len(spectra) == spectrum_audit["counts"]["primary_train_20ppm_spectra"], f"{len(spectra)}/{spectrum_audit['counts']['primary_train_20ppm_spectra']}")
    add_check("TRANS003", "峰token缓存与谱图清单行数一致", arrays["features"].shape == (len(spectra), MAX_PEAKS, 3))
    add_check("TRANS004", "无标签预训练结构与Development及全部冻结外测结构不重叠", not bool(pretrain_keys & excluded_pretraining_keys), f"overlap={len(pretrain_keys & excluded_pretraining_keys)}")
    add_check("TRANS005", "无标签预训练使用同结构双谱图而不读取PBT标签", len(pretrain_keys) == N_PRETRAIN_PAIRS and pretrain_keys.isdisjoint(set(modeling.inchikey_2d)))
    expected_coverage = True
    for route in ROUTES:
        for task, spec in TASKS.items():
            expected = int(pd.to_numeric(modeling[spec["column"]], errors="coerce").notna().sum())
            observed = compound_oof[
                compound_oof.representation_route.eq(route) & compound_oof.task.eq(task)
            ]
            expected_coverage &= len(observed) == observed.inchikey_2d.nunique() == expected
    add_check("TRANS006", "每条路线每个任务均覆盖全部有效化合物一次", expected_coverage)
    add_check("TRANS007", "化合物OOF键唯一", not compound_oof.duplicated(["representation_route", "task", "inchikey_2d"]).any())
    add_check("TRANS008", "所有路线任务均覆盖5个冻结结构簇fold", compound_oof.groupby(["representation_route", "task"]).outer_fold.nunique().eq(N_OUTER_FOLDS).all())
    add_check("TRANS009", "训练/验证/测试化合物严格隔离", fold_integrity.train_test_overlap.eq(0).all() and fold_integrity.inner_train_validation_overlap.eq(0).all())
    add_check("TRANS010", "峰顺序置换预测不变", max(permutation_differences) < 1e-4, f"max_abs_diff={max(permutation_differences):.8g}")
    add_check("TRANS011", "连续端点缺失标签通过mask保留为多任务训练", np.isnan(CompoundSpectrumDataset(arrays, spectra, modeling, all_keys, "full", False, SEED).labels).any())
    add_check("TRANS012", "每个训练epoch按化合物等权抽取一张谱图", len(CompoundSpectrumDataset(arrays, spectra, modeling, all_keys, "full", False, SEED)) == len(modeling))
    add_check("TRANS013", "全部预测为有限数", np.isfinite(compound_oof.y_prediction).all() and np.isfinite(compound_oof.loc[compound_oof.kind.eq("classification"), "y_probability"]).all())
    add_check("TRANS014", "预训练和20个fold模型检查点均存在", pretrained_checkpoint.exists() and len(list(checkpoint_dir.glob("*_fold*.pt"))) == len(ROUTES) * N_OUTER_FOLDS)
    add_check("TRANS015", "ChemVista外测保持锁定", simple_manifest["external_test_unlocked"] is False)
    add_check("TRANS016", "模型输入仅含峰、母离子与采集元数据，不含结构指纹", arrays["features"].shape[-1] == 3 and arrays["metadata"].shape[-1] == 6)
    qc = pd.DataFrame(checks)
    if not qc.status.eq("PASS").all():
        raise AssertionError(qc[qc.status.eq("FAIL")].to_dict("records"))

    compound_oof.to_pickle(output_dir / "set_transformer_compound_level_oof.pkl")
    compound_oof.to_csv(
        output_dir / "set_transformer_compound_level_oof.csv.gz",
        index=False,
        compression="gzip",
    )
    spectrum_oof.to_csv(
        output_dir / "set_transformer_main_spectrum_level_oof.csv.gz",
        index=False,
        compression="gzip",
    )
    metrics.to_csv(output_dir / "set_transformer_metrics.csv", index=False)
    comparisons.to_csv(output_dir / "set_transformer_paired_bootstrap.csv", index=False)
    all_baselines.to_csv(output_dir / "set_transformer_vs_structure_upper_bound.csv", index=False)
    training_history.to_csv(output_dir / "set_transformer_training_history.csv", index=False)
    selection.to_csv(output_dir / "set_transformer_epoch_selection.csv", index=False)
    fold_integrity.to_csv(output_dir / "set_transformer_fold_integrity.csv", index=False)
    qc.to_csv(output_dir / "set_transformer_qc_checks.csv", index=False)

    main_metrics = metrics[metrics.representation_route.eq(MAIN_ROUTE)]
    run_manifest = {
        "run_version": "PBT_MSMS_SET_TRANSFORMER_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": SEED,
        "external_test_unlocked": False,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_development_spectra": len(spectra),
        "n_development_compounds": len(modeling),
        "pretraining": {
            "n_same_structure_pairs": N_PRETRAIN_PAIRS,
            "epochs": PRETRAIN_EPOCHS,
            "objectives": ["masked_peak_reconstruction", "same_structure_contrastive"],
            "excluded_structure_count": len(excluded_pretraining_keys),
            "development_overlap": len(pretrain_keys & set(development.inchikey_2d)),
            "frozen_external_overlap": len(pretrain_keys & external_keys),
        },
        "architecture": {
            **model_config,
            "max_peaks": MAX_PEAKS,
            "peak_inputs": ["fragment_mz", "sqrt_relative_intensity", "neutral_loss"],
            "metadata_inputs": ["precursor_mz", "polarity", "collision_energy", "collision_energy_missing", "peak_count", "precursor_missing"],
            "positional_encoding": False,
            "parameter_count": int(sum(parameter.numel() for parameter in _model_from_initialization(model_config, None, SEED).parameters())),
        },
        "supervised_training": {
            "routes": ROUTES,
            "outer_cv": "frozen Morgan-Butina structure cluster 5-fold",
            "epoch_selection": "one held-out frozen structure-cluster fold inside each outer training set",
            "max_epochs": MAX_FINETUNE_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "compound_equal_sampling": "one spectrum per compound per epoch",
            "multitask_missing_label_mask": True,
        },
        "evaluation": {
            "compound_aggregation": "median across all eligible spectra",
            "bootstrap_resamples": N_BOOTSTRAP,
            "regression_metrics": ["MAE", "RMSE", "R2", "Spearman"],
        },
        "main_internal_metrics": main_metrics.to_dict("records"),
        "permutation_invariance_max_abs_difference": max(permutation_differences),
        "input_hashes": {
            name: sha256_file(path)
            for name, path in paths.items()
            if name not in {"public_stream"}
        },
        "public_stream_sha256_from_passed_upstream_audit": spectrum_audit["public_stream"]["sha256"],
        "software": {
            "python": sys.version,
            "executable": sys.executable,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "qc_status": "PASS",
    }
    write_json(output_dir / "set_transformer_run_manifest.json", run_manifest)
    return {
        "output_dir": output_dir,
        "metrics": metrics,
        "comparisons": comparisons,
        "qc": qc,
        "selection": selection,
        "pretrain_history": pretrain_history,
        "manifest": run_manifest,
    }
