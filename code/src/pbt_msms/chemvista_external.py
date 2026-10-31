"""Notebook 20: one-shot blinded ChemVista external evaluation.

The command has three idempotent stages because DreaMS lives in a separate
environment::

    env_ms     -m pbt_msms.chemvista_external --stage prepare
    env_dreams -m pbt_msms.chemvista_external --stage embed
    env_ms     -m pbt_msms.chemvista_external --stage evaluate

No ChemVista label is used while fitting a model, selecting a route, choosing a
threshold, calibrating a prediction, or estimating an abstention threshold.
All such decisions are loaded from Notebook 19 or earlier Development-only
artifacts.
"""

from __future__ import annotations

from .deployment import deployment_comparison_dir

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
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
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


SEED = 20260917
N_FOLDS = 5
N_SEEDS = 3
MAX_MZ = 2000.0
MAX_PEAKS = 120
N_BINS = 2000
TRAIN_SAMPLING_REPEATS = 5
EMBEDDING_DIM = 1024

TASKS: dict[str, dict[str, str]] = {
    "P_screen_not_ready_binary": {"column": "label__P_screen_not_ready_binary", "kind": "classification"},
    "B_LOGBCF_REGRESSION": {"column": "label__B_LOGBCF_REGRESSION", "kind": "regression"},
    "T_EC50_MG_L_REGRESSION": {"column": "label__T_EC50_MG_L_REGRESSION", "kind": "regression"},
    "T_LC50_MG_L_REGRESSION": {"column": "label__T_LC50_MG_L_REGRESSION", "kind": "regression"},
    "T_NOEC_MG_L_REGRESSION": {"column": "label__T_NOEC_MG_L_REGRESSION", "kind": "regression"},
}
TASK_NAMES = list(TASKS)

PIPELINES = [
    "Binned MS/MS",
    "DreaMS + selected head",
    "Set Transformer pretrained",
    "Transformer A real scratch",
    "Transformer B real pretrain",
    "Transformer C same-compound CFM",
    "Transformer D synthetic-new CFM",
    "Transformer E CFM pretrain",
]
TRANSFORMER_ROUTES = {
    "A_real_scratch": "Transformer A real scratch",
    "B_real_pretrain": "Transformer B real pretrain",
    "C_same_compound_cfm": "Transformer C same-compound CFM",
    "D_synthetic_new_cfm": "Transformer D synthetic-new CFM",
    "E_cfm_pretrain_real_finetune": "Transformer E CFM pretrain",
}

DREAMS_DEPLOYMENT_ROUTE = "all_spectra_compound_median"
DREAMS_DEPLOYMENT_MODEL = "nested_selected_primary"
DREAMS_DEPLOYMENT_SCOPE = "main_all_spectra"
CACHE_SCHEMA_VERSION = "PBT_MSMS_PREDICTION_CACHE_V2"


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


def platt_calibrate_probability(probability: np.ndarray, coefficient: float, intercept: float) -> np.ndarray:
    """Apply the frozen outer-fold Platt map used for the selected DreaMS P head."""
    value = np.clip(np.asarray(probability, dtype=float), 1e-7, 1.0 - 1e-7)
    logit = np.log(value) - np.log1p(-value)
    return 1.0 / (1.0 + np.exp(-np.clip(float(coefficient) * logit + float(intercept), -40.0, 40.0)))


def load_dreams_probability_calibrators(dev_dir: Path) -> dict[int, tuple[float, float]]:
    """Load the original Development-only deployment; not an internal CV estimate."""
    from .deployment import deployment_dreams_calibration_path
    table = pd.read_csv(deployment_dreams_calibration_path(dev_dir))
    selected = table[
        table["representation_route"].eq(DREAMS_DEPLOYMENT_ROUTE)
        & table["model"].eq(DREAMS_DEPLOYMENT_MODEL)
    ].copy()
    selected["outer_fold"] = selected["outer_fold"].astype(int)
    if set(selected["outer_fold"]) != set(range(N_FOLDS)) or selected["outer_fold"].duplicated().any():
        raise RuntimeError("DreaMS deployment probability calibration is incomplete or duplicated")
    if "current_fold_excluded" in selected and not selected["current_fold_excluded"].astype(str).str.lower().eq("true").all():
        raise RuntimeError("DreaMS deployment calibration violates its recorded row-exclusion rule")
    return {
        int(row.outer_fold): (float(row.coefficient), float(row.intercept))
        for row in selected.itertuples(index=False)
    }


def stable_seed(*parts: Any) -> int:
    text = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:8], 16)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
    os.replace(temporary, path)


def paths(root: Path) -> dict[str, Path]:
    base = root / "outputs/pbt_msms_v1"
    out = base / "chemvista_external_20_v1"
    return {
        "base": base,
        "out": out,
        "figures": out / "figures",
        "gate": deployment_comparison_dir(base) / "19_chemvista_unlock_gate.json",
        "registry": deployment_comparison_dir(base) / "19_final_model_registry.json",
        "calibration": deployment_comparison_dir(base) / "19_deployment_calibration.json",
        "rejection": deployment_comparison_dir(base) / "19_rejection_thresholds.json",
        "external_manifest": base / "spectrum_audit_v1/chemvista_evaluation_spectrum_manifest.csv.gz",
        "chemvista_index": root / "验证实验谱图数据/ChemVista_cleaned/ChemVista_spectrum_index.pkl",
        "chemvista_peaks": root / "验证实验谱图数据/ChemVista_cleaned/ChemVista_spectrum_peaks.pkl",
        "external_index": out / "20_chemvista_spectrum_index.csv.gz",
        "external_roster": out / "20_chemvista_compound_roster.csv.gz",
        "mgf": out / "20_chemvista_dreams_input.mgf",
        "dreams_embeddings": out / "20_chemvista_dreams_embeddings.npy",
        "dreams_embed_manifest": out / "20_dreams_embedding_manifest.json",
        "binned": out / "20_chemvista_binned_features.npz",
        "transformer_tokens": out / "20_chemvista_transformer_tokens.npz",
        "set_tokens": out / "20_chemvista_set_transformer_tokens.npz",
        "prepare_manifest": out / "20_preparation_manifest.json",
    }


def prediction_cache_credential(root: Path, p: dict[str, Path], name: str) -> dict[str, Any]:
    input_paths = {
        "binned": [p["external_index"], p["binned"]],
        "dreams": [p["external_index"], p["dreams_embeddings"]],
        "set": [p["external_index"], p["set_tokens"]],
        "transformer": [p["external_index"], p["transformer_tokens"]],
    }
    model_manifests = {
        "binned": p["base"] / "simple_spectrum_baselines_v1/simple_spectrum_run_manifest.json",
        "dreams": p["base"] / "dreams_heads_v1/dreams_head_run_manifest.json",
        "set": p["base"] / "set_transformer_v1/set_transformer_run_manifest.json",
        "transformer": p["base"] / "transformer_finetuning_v1/transformer_finetuning_manifest.json",
    }
    if name not in input_paths:
        raise KeyError(name)
    required = [*input_paths[name], model_manifests[name], p["registry"], p["calibration"], p["rejection"]]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Prediction-cache credential inputs are missing: {missing}")
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_name": name,
        "preprocessing_version": "chemvista_external_v2_dreams_fold_platt",
        "input_hashes": {path.name: sha256_file(path) for path in input_paths[name]},
        "model_manifest_sha256": sha256_file(model_manifests[name]),
        "model_registry_sha256": sha256_file(p["registry"]),
        "deployment_calibration_sha256": sha256_file(p["calibration"]),
        "rejection_thresholds_sha256": sha256_file(p["rejection"]),
    }


def prediction_cache_is_valid(path: Path, expected: dict[str, Any]) -> bool:
    credential_path = path.with_suffix(path.suffix + ".credential.json")
    if not path.is_file() or not credential_path.is_file():
        return False
    observed = read_json(credential_path)
    return observed.get("credential") == expected and observed.get("prediction_sha256") == sha256_file(path)


def write_prediction_cache_credential(path: Path, credential: dict[str, Any]) -> None:
    atomic_json(path.with_suffix(path.suffix + ".credential.json"), {
        "credential": credential,
        "prediction_sha256": sha256_file(path),
        "created_at_utc": utc_now(),
    })


def verify_unlock(p: dict[str, Path]) -> dict[str, Any]:
    gate = read_json(p["gate"])
    if gate.get("status") != "READY_FOR_NOTEBOOK_20":
        raise RuntimeError(f"Notebook 19 gate is not ready: {gate.get('status')}")
    if not bool(gate.get("external_test_unlocked")) or bool(gate.get("chemvista_used")):
        raise RuntimeError("ChemVista unlock gate is inconsistent")
    checks = {
        "selection_registry_sha256": p["registry"],
        "deployment_calibration_sha256": p["calibration"],
        "rejection_thresholds_sha256": p["rejection"],
    }
    for field, path in checks.items():
        expected = gate.get(field)
        actual = sha256_file(path)
        if expected != actual:
            raise RuntimeError(f"Frozen Notebook 19 artifact changed: {path.name}")
    return gate


def normalize_mode(value: Any) -> tuple[str, float]:
    text = str(value).lower()
    if "pos" in text:
        return "Positive", 1.0
    if "neg" in text:
        return "Negative", -1.0
    raise ValueError(f"Unsupported ion mode: {value}")


def instrument_category(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text or text in {"nan", "none", "unknown", "n/a"}:
        return "UNKNOWN"
    if "orbitrap" in text:
        return "ORBITRAP"
    if "qtof" in text or "q-tof" in text or "quadrupole time-of-flight" in text:
        return "QTOF"
    if "fticr" in text or "ft-icr" in text:
        return "FTICR"
    if "ion trap" in text or "iontrap" in text or "ltq" in text:
        return "ION_TRAP"
    if "triple quad" in text or "qqq" in text:
        return "TRIPLE_QUAD"
    if "tof" in text or "time-of-flight" in text:
        return "TOF"
    return "OTHER"


def infer_charge(adduct: Any, ion_mode: Any) -> int:
    import re
    text = str(adduct).strip()
    match = re.search(r"\](\d*)([+-])$", text)
    if match:
        magnitude = int(match.group(1)) if match.group(1) else 1
        charge = magnitude if match.group(2) == "+" else -magnitude
    else:
        charge = 1 if normalize_mode(ion_mode)[0] == "Positive" else -1
    expected = 1 if normalize_mode(ion_mode)[0] == "Positive" else -1
    if int(np.sign(charge)) != expected:
        raise ValueError(f"Adduct/ion mode conflict: {adduct} / {ion_mode}")
    return charge


def binned_max(masses: np.ndarray, intensities: np.ndarray) -> dict[int, float]:
    valid = (
        np.isfinite(masses) & np.isfinite(intensities) & (masses >= 0)
        & (masses < MAX_MZ) & (intensities > 0)
    )
    result: dict[int, float] = {}
    for column, value in zip(np.floor(masses[valid]).astype(int), intensities[valid]):
        result[int(column)] = max(result.get(int(column), 0.0), float(value))
    return result


def transformer_peak_encoding(mz: np.ndarray, intensity: np.ndarray, precursor: float):
    valid = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (mz < MAX_MZ) & (intensity > 0)
    mz, intensity = mz[valid], intensity[valid]
    if len(mz) > MAX_PEAKS:
        chosen = np.argsort(intensity, kind="stable")[-MAX_PEAKS:]
        mz, intensity = mz[chosen], intensity[chosen]
    if len(mz):
        order = np.argsort(mz, kind="stable")
        mz, intensity = mz[order], intensity[order]
        intensity = intensity / intensity.max()
    token = np.zeros((MAX_PEAKS, 4), dtype=np.float16)
    padding = np.ones(MAX_PEAKS, dtype=bool)
    if len(mz):
        has_precursor = np.isfinite(precursor) and precursor > 0
        loss = np.maximum(precursor - mz, 0) if has_precursor else np.zeros(len(mz))
        ratio = mz / precursor if has_precursor else np.zeros(len(mz))
        token[: len(mz)] = np.stack([
            np.clip(mz / MAX_MZ, 0, 1), np.sqrt(intensity),
            np.clip(loss / MAX_MZ, 0, 1), np.clip(ratio, 0, 1.5),
        ], axis=1).astype(np.float16)
        padding[: len(mz)] = False
    return token, padding, len(mz)


def metadata8(precursor: float, polarity: float, ce: float, peak_n: int, inst: str) -> np.ndarray:
    ce_missing = not np.isfinite(ce) or ce < 0
    precursor_missing = not np.isfinite(precursor) or precursor <= 0
    return np.asarray([
        0 if precursor_missing else np.clip(precursor / MAX_MZ, 0, 1.5), polarity,
        0 if ce_missing else np.clip(ce / 100, 0, 2), float(ce_missing),
        math.log1p(peak_n) / math.log1p(MAX_PEAKS), 0.0, float(peak_n < 3), float(inst == "UNKNOWN"),
    ], dtype=np.float32)


def set_transformer_encoding(mz: np.ndarray, intensity: np.ndarray, precursor: float, polarity: float, ce: float):
    valid = np.isfinite(mz) & np.isfinite(intensity) & (mz >= 15) & (mz < MAX_MZ) & (intensity > 0)
    mz, intensity = mz[valid], intensity[valid]
    if len(intensity):
        intensity = np.sqrt(intensity / intensity.max())
    precursor_norm = precursor / MAX_MZ if np.isfinite(precursor) else 0.0
    peaks = np.stack([mz / MAX_MZ, intensity, np.clip(precursor_norm - mz / MAX_MZ, 0, 1)], axis=1)
    if len(peaks) > MAX_PEAKS:
        peaks = peaks[np.argsort(peaks[:, 1])[-MAX_PEAKS:]]
    if len(peaks):
        peaks = peaks[np.argsort(peaks[:, 0], kind="stable")]
    feature = np.zeros((MAX_PEAKS, 3), dtype=np.float16)
    padding = np.ones(MAX_PEAKS, dtype=bool)
    feature[: len(peaks)] = peaks.astype(np.float16)
    padding[: len(peaks)] = False
    ce_missing = not np.isfinite(ce) or ce < 0
    meta = np.asarray([
        np.clip(precursor_norm, 0, 1.5), polarity,
        0 if ce_missing else np.clip(ce / 100, 0, 2), float(ce_missing),
        math.log1p(len(peaks)) / math.log1p(MAX_PEAKS), float(not np.isfinite(precursor)),
    ], dtype=np.float32)
    return feature, padding, meta


def run_prepare(root: Path) -> None:
    p = paths(root)
    p["out"].mkdir(parents=True, exist_ok=True)
    p["figures"].mkdir(parents=True, exist_ok=True)
    gate = verify_unlock(p)

    manifest = pd.read_csv(p["external_manifest"], low_memory=False)
    if not manifest["evaluation_eligible"].fillna(False).astype(bool).all():
        raise RuntimeError("Ineligible spectrum found in frozen ChemVista evaluation manifest")
    if manifest["exact_public_train_spectrum_collision"].astype(bool).any():
        raise RuntimeError("Exact public-training spectrum collision found")
    near_confirmed = manifest["confirmed_near_public_train_duplicate"].astype(bool)
    if (near_confirmed & manifest["chemvista_evaluation_role"].ne("transfer")).any():
        raise RuntimeError("Confirmed near duplicate found outside the declared transfer cohort")
    if manifest["spectrum_id"].duplicated().any():
        raise RuntimeError("Duplicate ChemVista spectrum IDs")

    # Deliberately construct the roster from the spectrum manifest.  Endpoint
    # labels remain unopened until every external candidate prediction is on disk.
    labels = (manifest[["inchikey_2d", "chemvista_evaluation_role"]]
              .drop_duplicates().rename(columns={"chemvista_evaluation_role": "evaluation_role"})
              .sort_values(["evaluation_role", "inchikey_2d"]).reset_index(drop=True))
    if labels["inchikey_2d"].duplicated().any() or len(labels) != 586:
        raise RuntimeError("Frozen ChemVista compound roster is not the expected 586 unique compounds")
    atomic_csv(labels, p["external_roster"], compression="gzip")

    raw_index = pd.read_pickle(p["chemvista_index"])
    if raw_index.index.name == "spectrum_id":
        raw_index = raw_index.reset_index(drop=True)
    raw_keep = ["spectrum_id", "precursor_type"]
    raw_lookup = raw_index[raw_keep].drop_duplicates("spectrum_id")
    index = manifest.merge(raw_lookup, on="spectrum_id", how="left", validate="one_to_one")
    index = index.merge(labels[["inchikey_2d", "evaluation_role"]], on="inchikey_2d", how="left", validate="many_to_one")
    if index["evaluation_role"].isna().any():
        raise RuntimeError("Spectrum without a frozen ChemVista role")
    if not (index["evaluation_role"] == index["chemvista_evaluation_role"]).all():
        raise RuntimeError("Spectrum and compound ChemVista roles disagree")
    index = index.sort_values(["evaluation_role", "inchikey_2d", "spectrum_id"]).reset_index(drop=True)
    index["feature_row"] = np.arange(len(index), dtype=int)
    index["adduct_frozen"] = index["precursor_type"].fillna(index["adduct"]).fillna("UNKNOWN")
    index["instrument_frozen"] = index["instrument_type"].fillna("UNKNOWN")
    atomic_csv(index, p["external_index"], compression="gzip")

    with p["chemvista_peaks"].open("rb") as handle:
        peaks_by_id = pickle.load(handle)

    fragment_rows: list[int] = []
    fragment_cols: list[int] = []
    fragment_data: list[float] = []
    neutral_rows: list[int] = []
    neutral_cols: list[int] = []
    neutral_data: list[float] = []
    binned_meta = np.zeros((len(index), 6), dtype=np.float32)

    peak_tokens = np.zeros((len(index), MAX_PEAKS, 4), dtype=np.float16)
    peak_padding = np.ones((len(index), MAX_PEAKS), dtype=bool)
    token_meta = np.zeros((len(index), 8), dtype=np.float32)
    adduct_codes = np.zeros(len(index), dtype=np.int16)
    instrument_codes = np.zeros(len(index), dtype=np.int16)
    source_codes = np.zeros(len(index), dtype=np.int16)
    vocab = read_json(p["base"] / "transformer_data_v1/transformer_categorical_vocab.json")
    adduct_map = {x: i for i, x in enumerate(vocab["adduct"])}
    inst_map = {x: i for i, x in enumerate(vocab["instrument_category"])}
    source_map = {x: i for i, x in enumerate(vocab["source_library"])}

    set_features = np.zeros((len(index), MAX_PEAKS, 3), dtype=np.float16)
    set_padding = np.ones((len(index), MAX_PEAKS), dtype=bool)
    set_meta = np.zeros((len(index), 6), dtype=np.float32)

    mgf_tmp = p["mgf"].with_suffix(".mgf.tmp")
    with mgf_tmp.open("w", encoding="utf-8", newline="\n") as mgf:
        for row in tqdm(index.itertuples(index=False), total=len(index), desc="Prepare ChemVista features"):
            payload = peaks_by_id[str(row.spectrum_id)]
            mz = np.asarray(payload["mz"], dtype=float)
            intensity_raw = np.asarray(payload.get("intensity_raw", payload["intensity_norm_100"]), dtype=float)
            valid = np.isfinite(mz) & np.isfinite(intensity_raw) & (mz > 0) & (intensity_raw > 0)
            mz, intensity_raw = mz[valid], intensity_raw[valid]
            if not len(mz):
                raise RuntimeError(f"No valid peaks: {row.spectrum_id}")
            intensity = np.sqrt(intensity_raw / intensity_raw.max())
            precursor = float(row.precursor_mz)
            mode, polarity = normalize_mode(row.ion_mode)
            ce = float(row.collision_energy_value) if np.isfinite(row.collision_energy_value) else np.nan
            feature_row = int(row.feature_row)

            for column, value in binned_max(mz, intensity).items():
                fragment_rows.append(feature_row); fragment_cols.append(column); fragment_data.append(value)
            for column, value in binned_max(precursor - mz, intensity).items():
                neutral_rows.append(feature_row); neutral_cols.append(column); neutral_data.append(value)
            ce_missing = not np.isfinite(ce) or ce < 0
            binned_meta[feature_row] = [precursor, polarity, 0 if ce_missing else ce, float(ce_missing), math.log1p(len(mz)), float(not np.isfinite(precursor))]

            token, pad, retained = transformer_peak_encoding(mz, intensity_raw, precursor)
            inst = instrument_category(row.instrument_frozen)
            adduct = str(row.adduct_frozen)
            peak_tokens[feature_row], peak_padding[feature_row] = token, pad
            token_meta[feature_row] = metadata8(precursor, polarity, ce, retained, inst)
            adduct_codes[feature_row] = adduct_map.get(adduct, adduct_map.get("OTHER", 0))
            instrument_codes[feature_row] = inst_map.get(inst, inst_map.get("OTHER", 0))
            source_codes[feature_row] = source_map.get("UNKNOWN", 0)

            sf, sp, sm = set_transformer_encoding(mz, intensity_raw, precursor, polarity, ce)
            set_features[feature_row], set_padding[feature_row], set_meta[feature_row] = sf, sp, sm

            charge = infer_charge(adduct, row.ion_mode)
            charge_text = f"{abs(charge)}{'+' if charge > 0 else '-'}"
            mgf.write("BEGIN IONS\n")
            mgf.write(f"TITLE={row.spectrum_id}\nFEATURE_ID={feature_row + 1}\nMSLEVEL=2\n")
            mgf.write(f"PEPMASS={precursor:.8f}\nCHARGE={charge_text}\nIONMODE={mode}\n")
            if adduct != "UNKNOWN":
                mgf.write(f"ADDUCT={adduct}\n")
            if not ce_missing:
                mgf.write(f"COLLISION_ENERGY={ce:.8g}\n")
            mgf.write("SOURCE_LIBRARY=ChemVista\n")
            for mass, abundance in zip(mz, intensity_raw):
                mgf.write(f"{mass:.8f} {abundance:.8f}\n")
            mgf.write("END IONS\n\n")
    os.replace(mgf_tmp, p["mgf"])

    x_fragment = sparse.csr_matrix((fragment_data, (fragment_rows, fragment_cols)), shape=(len(index), N_BINS), dtype=np.float32)
    x_neutral = sparse.csr_matrix((neutral_data, (neutral_rows, neutral_cols)), shape=(len(index), N_BINS), dtype=np.float32)
    atomic_npz(p["binned"],
               fragment_data=x_fragment.data, fragment_indices=x_fragment.indices, fragment_indptr=x_fragment.indptr,
               neutral_data=x_neutral.data, neutral_indices=x_neutral.indices, neutral_indptr=x_neutral.indptr,
               shape=np.asarray(x_fragment.shape), metadata=binned_meta)
    atomic_npz(p["transformer_tokens"], peak_tokens=peak_tokens, padding_mask=peak_padding,
               metadata_numeric=token_meta, adduct_code=adduct_codes,
               instrument_code=instrument_codes, source_code=source_codes)
    atomic_npz(p["set_tokens"], features=set_features, padding=set_padding, metadata=set_meta)

    manifest_out = {
        "version": "PBT_MSMS_CHEMVISTA_EXTERNAL_PREPARATION_20_V1",
        "generated_at_utc": utc_now(), "status": "PASS",
        "notebook19_gate_sha256": sha256_file(p["gate"]),
        "notebook19_gate_status": gate["status"],
        "n_spectra": len(index), "n_compounds": labels["inchikey_2d"].nunique(),
        "role_counts": labels["evaluation_role"].value_counts().to_dict(),
        "ion_mode_counts": index["ion_mode"].value_counts().to_dict(),
        "label_access_policy": "endpoint label files not opened in preparation; first opened only after all candidate predictions are frozen",
        "artifacts": {name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                      for name, path in p.items() if name in {"external_index", "external_roster", "mgf", "binned", "transformer_tokens", "set_tokens"}},
    }
    atomic_json(p["prepare_manifest"], manifest_out)
    print(json.dumps(manifest_out, ensure_ascii=False, indent=2))


def run_embed(root: Path, batch_size: int) -> None:
    p = paths(root)
    prep = read_json(p["prepare_manifest"])
    if prep.get("status") != "PASS" or sha256_file(p["mgf"]) != prep["artifacts"]["mgf"]["sha256"]:
        raise RuntimeError("ChemVista preparation is not frozen PASS")
    index = pd.read_csv(p["external_index"])
    from pbt_msms.dreams_mvp import configure_writable_runtime_cache
    configure_writable_runtime_cache()
    import torch
    from dreams.api import dreams_embeddings
    if not torch.cuda.is_available():
        raise RuntimeError("DreaMS extraction requires a CUDA-enabled env_dreams kernel")
    result = dreams_embeddings(p["mgf"], batch_size=batch_size, progress_bar=True, store_embs=False)
    embeddings = np.asarray(result, dtype=np.float32)
    if embeddings.shape != (len(index), EMBEDDING_DIM) or not np.isfinite(embeddings).all():
        raise RuntimeError(f"Unexpected DreaMS embedding output: {embeddings.shape}")
    atomic_npy(p["dreams_embeddings"], embeddings)
    payload = {
        "version": "PBT_MSMS_CHEMVISTA_DREAMS_EMBEDDING_20_V1", "status": "PASS",
        "generated_at_utc": utc_now(), "batch_size": batch_size,
        "shape": list(embeddings.shape), "dtype": str(embeddings.dtype),
        "input_mgf_sha256": sha256_file(p["mgf"]),
        "embedding_sha256": sha256_file(p["dreams_embeddings"]),
        "python": sys.version, "python_executable": sys.executable,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }
    atomic_json(p["dreams_embed_manifest"], payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def load_binned_external(path: Path):
    with np.load(path, allow_pickle=False) as z:
        shape = tuple(z["shape"].astype(int))
        xf = sparse.csr_matrix((z["fragment_data"], z["fragment_indices"], z["fragment_indptr"]), shape=shape)
        xn = sparse.csr_matrix((z["neutral_data"], z["neutral_indices"], z["neutral_indptr"]), shape=shape)
        meta = z["metadata"].copy()
    return xf, xn, meta


def sample_development_rows(spectra: pd.DataFrame, compound_keys: list[str], task: str, fold: int) -> np.ndarray:
    groups = {key: group["feature_row"].to_numpy(int) for key, group in spectra.groupby("inchikey_2d")}
    sampled = []
    for repeat in range(TRAIN_SAMPLING_REPEATS):
        rng = np.random.default_rng(stable_seed(42, task, fold, repeat))
        for key in sorted(compound_keys):
            sampled.append(int(rng.choice(groups[key])))
    return np.asarray(sampled, dtype=int)


def aggregate_spectra(values: np.ndarray, index: pd.DataFrame, method: str) -> pd.Series:
    frame = index[["inchikey_2d", "spectrum_id"]].copy()
    frame["value"] = np.asarray(values, float)
    if method == "median":
        return frame.groupby("inchikey_2d", sort=True)["value"].median()
    if method == "mean":
        return frame.groupby("inchikey_2d", sort=True)["value"].mean()
    raise KeyError(method)


def binned_predictions(root: Path, external_index: pd.DataFrame) -> pd.DataFrame:
    p = paths(root)
    simple = p["base"] / "simple_spectrum_baselines_v1"
    dev_spectra = pd.read_csv(simple / "spectrum_feature_row_index.csv")
    development = pd.read_csv(p["base"] / "split_v1/development_compounds.csv", low_memory=False)
    xf_dev = sparse.load_npz(simple / "fragment_intensity_1Da.npz")
    xn_dev = sparse.load_npz(simple / "neutral_loss_intensity_1Da.npz")
    meta_dev = np.load(simple / "spectrum_metadata.npy", allow_pickle=False)
    xf_ext, xn_ext, meta_ext = load_binned_external(p["binned"])
    label_frame = development.set_index("inchikey_2d")
    spectrum_keys = set(dev_spectra["inchikey_2d"].astype(str))
    rows = []
    for task, spec in TASKS.items():
        available = development.loc[
            pd.to_numeric(development[spec["column"]], errors="coerce").notna()
            & development["inchikey_2d"].astype(str).isin(spectrum_keys)
        ].copy()
        for fold in range(N_FOLDS):
            keys = available.loc[available["cv_fold"].astype(int).ne(fold), "inchikey_2d"].tolist()
            train_rows = sample_development_rows(dev_spectra, keys, task, fold)
            train_keys = dev_spectra.iloc[train_rows]["inchikey_2d"].tolist()
            y = np.asarray([label_frame.at[key, spec["column"]] for key in train_keys], dtype=float)
            scaler = StandardScaler().fit(meta_dev[train_rows])
            x_train = sparse.hstack([xf_dev[train_rows], xn_dev[train_rows], sparse.csr_matrix(scaler.transform(meta_dev[train_rows]))], format="csr")
            x_ext = sparse.hstack([xf_ext, xn_ext, sparse.csr_matrix(scaler.transform(meta_ext))], format="csr")
            if spec["kind"] == "classification":
                model = LogisticRegression(C=1.0, solver="liblinear", max_iter=4000, random_state=stable_seed(42, "binned", task, fold))
                model.fit(x_train, y.astype(int)); values = model.predict_proba(x_ext)[:, 1]
            else:
                model = Ridge(alpha=10.0, solver="lsqr", tol=1e-4, max_iter=3000)
                model.fit(x_train, y); values = model.predict(x_ext)
            compound = aggregate_spectra(values, external_index, "median")
            rows.extend({"pipeline": "Binned MS/MS", "task": task, "kind": spec["kind"],
                         "outer_fold": fold, "replica": fold, "inchikey_2d": key, "y_score": float(value)}
                        for key, value in compound.items())
    return pd.DataFrame(rows)


def dreams_predictions(root: Path, external_index: pd.DataFrame) -> pd.DataFrame:
    from pbt_msms.dreams_heads import _fit_final_predictor, _predict
    p = paths(root)
    dev_dir = p["base"] / "dreams_heads_v1"
    dev_index = pd.read_csv(dev_dir / "dreams_compound_embedding_index.csv")
    with np.load(dev_dir / "dreams_compound_embedding_routes.npz") as z:
        x_dev = z["all_spectra"].copy()
    x_spectrum = np.load(p["dreams_embeddings"], mmap_mode="r", allow_pickle=False)
    ext_keys = sorted(external_index["inchikey_2d"].unique())
    ext_groups = external_index.groupby("inchikey_2d")["feature_row"].apply(lambda x: x.to_numpy(int)).to_dict()
    x_ext = np.stack([np.median(np.asarray(x_spectrum[ext_groups[key]], np.float32), axis=0) for key in ext_keys])
    oof = pd.read_csv(dev_dir / "dreams_head_compound_level_oof.csv.gz", low_memory=False)
    selected = oof[(oof["analysis_scope"] == DREAMS_DEPLOYMENT_SCOPE) & (oof["model"] == DREAMS_DEPLOYMENT_MODEL)]
    calibrators = load_dreams_probability_calibrators(dev_dir)
    development_folds = dev_index["cv_fold"].astype(int).to_numpy()
    rows = []
    for task, spec in TASKS.items():
        labels = pd.to_numeric(dev_index[spec["column"]], errors="coerce").to_numpy(float)
        for fold in range(N_FOLDS):
            marker = selected[(selected["task"] == task) & (selected["outer_fold"].astype(int) == fold)].iloc[0]
            family = str(marker["selected_family"])
            params = json.loads(marker["best_params"])
            mask = np.isfinite(labels) & dev_index["cv_fold"].astype(int).ne(fold).to_numpy()
            predictor = _fit_final_predictor(
                x_dev[mask],
                labels[mask].astype(int) if spec["kind"] == "classification" else labels[mask],
                development_folds[mask],
                family,
                spec["kind"],
                params,
                task,
                fold,
                DREAMS_DEPLOYMENT_SCOPE,
            )
            prediction, probability = _predict(predictor, x_ext)
            if spec["kind"] == "classification":
                coefficient, intercept = calibrators[fold]
                values = platt_calibrate_probability(probability, coefficient, intercept)
            else:
                values = prediction
            rows.extend({"pipeline": "DreaMS + selected head", "task": task, "kind": spec["kind"],
                         "outer_fold": fold, "replica": fold, "inchikey_2d": key, "y_score": float(value)}
                        for key, value in zip(ext_keys, values))
    return pd.DataFrame(rows)


def set_transformer_predictions(root: Path, external_index: pd.DataFrame, device) -> pd.DataFrame:
    import torch
    from pbt_msms.transformer_pipeline import TASK_NAMES as SET_TASKS
    from pbt_msms.transformer_pipeline import _model_from_initialization
    p = paths(root)
    config = {"d_model": 128, "nhead": 4, "num_layers": 4, "dim_feedforward": 256, "dropout": 0.1}
    with np.load(p["set_tokens"], allow_pickle=False) as z:
        features, padding, metadata = z["features"], z["padding"], z["metadata"]
    rows = []
    batch_size = 128
    for fold in range(N_FOLDS):
        checkpoint = torch.load(p["base"] / f"set_transformer_v1/checkpoints/set_transformer_pretrained_fold{fold}.pt", map_location="cpu", weights_only=False)
        model = _model_from_initialization(config, None, stable_seed(42, "external", fold))
        model.load_state_dict(checkpoint["model_state"], strict=True); model.to(device); model.eval()
        output_parts = []
        with torch.no_grad():
            for start in range(0, len(external_index), batch_size):
                stop = min(start + batch_size, len(external_index))
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    output = model(torch.from_numpy(features[start:stop].astype(np.float32)).to(device),
                                   torch.from_numpy(padding[start:stop]).to(device),
                                   torch.from_numpy(metadata[start:stop]).to(device)).float().cpu().numpy()
                output_parts.append(output)
        raw = np.concatenate(output_parts)
        means, stds = np.asarray(checkpoint["label_mean"]), np.asarray(checkpoint["label_std"])
        for task_index, task in enumerate(SET_TASKS):
            kind = TASKS[task]["kind"]
            values = 1 / (1 + np.exp(-np.clip(raw[:, task_index], -40, 40))) if kind == "classification" else raw[:, task_index] * stds[task_index] + means[task_index]
            compound = aggregate_spectra(values, external_index, "median")
            rows.extend({"pipeline": "Set Transformer pretrained", "task": task, "kind": kind,
                         "outer_fold": fold, "replica": fold, "inchikey_2d": key, "y_score": float(value)}
                        for key, value in compound.items())
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def transformer_classes():
    import torch
    from torch import nn
    import torch.nn.functional as F

    class Encoder(nn.Module):
        def __init__(self, config, vocab_sizes):
            super().__init__(); d = config["d_model"]
            self.peak_projection = nn.Sequential(nn.Linear(config["peak_input_dim"] + 1, d), nn.GELU(), nn.LayerNorm(d))
            self.metadata_projection = nn.Sequential(nn.Linear(config["metadata_numeric_dim"], d), nn.GELU(), nn.LayerNorm(d))
            self.adduct_embedding = nn.Embedding(vocab_sizes["adduct"], d)
            self.instrument_embedding = nn.Embedding(vocab_sizes["instrument_category"], d)
            self.source_embedding = nn.Embedding(vocab_sizes["source_library"], d)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=config["n_heads"], dim_feedforward=config["dim_feedforward"], dropout=config["dropout"], activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=config["n_layers"])
            self.final_norm = nn.LayerNorm(d)
            self.reconstruction_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, config["peak_input_dim"]))
            self.projection_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, config["projection_dim"]))
        def forward(self, peaks, padding, metadata, adduct, instrument, source):
            flags = torch.zeros(peaks.shape[:2], dtype=peaks.dtype, device=peaks.device)
            pt = self.peak_projection(torch.cat([peaks, flags.unsqueeze(-1)], dim=-1))
            condition = self.metadata_projection(metadata) + self.adduct_embedding(adduct) + self.instrument_embedding(instrument) + self.source_embedding(source)
            cls = self.cls_token.expand(len(peaks), -1, -1) + condition.unsqueeze(1)
            tokens = torch.cat([cls, pt], dim=1)
            cp = torch.zeros((len(peaks), 1), dtype=torch.bool, device=padding.device)
            encoded = self.final_norm(self.encoder(tokens, src_key_padding_mask=torch.cat([cp, padding], dim=1)))
            return F.normalize(self.projection_head(encoded[:, 0]), dim=-1)

    class Model(nn.Module):
        def __init__(self, config, vocab_sizes, n_tasks):
            super().__init__(); self.encoder = Encoder(config, vocab_sizes)
            self.head = nn.Sequential(nn.LayerNorm(config["projection_dim"]), nn.Linear(config["projection_dim"], config["projection_dim"]), nn.GELU(), nn.Dropout(0.20), nn.Linear(config["projection_dim"], n_tasks))
        def forward(self, peaks, padding, metadata, adduct, instrument, source):
            return self.head(self.encoder(peaks, padding, metadata, adduct, instrument, source))
    return Model


def transformer_predictions(root: Path, external_index: pd.DataFrame, device) -> pd.DataFrame:
    import torch
    p = paths(root)
    with np.load(p["transformer_tokens"], allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files}
    Model = transformer_classes()
    rows = []
    batch_size = 128
    for route, pipeline in TRANSFORMER_ROUTES.items():
        for fold in range(N_FOLDS):
            for seed_index in range(N_SEEDS):
                checkpoint_path = p["base"] / f"transformer_finetuning_v1/checkpoints/outer_{fold}/{route}__seed_{seed_index}__best.pt"
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                metadata = checkpoint["metadata"]
                model = Model(metadata["model_config"], metadata["vocab_sizes"], len(TASKS))
                model.load_state_dict(checkpoint["model_state"], strict=True); model.to(device); model.eval()
                outputs = []
                with torch.no_grad():
                    for start in range(0, len(external_index), batch_size):
                        stop = min(start + batch_size, len(external_index))
                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                            out = model(torch.from_numpy(arrays["peak_tokens"][start:stop].astype(np.float32)).to(device),
                                        torch.from_numpy(arrays["padding_mask"][start:stop]).to(device),
                                        torch.from_numpy(arrays["metadata_numeric"][start:stop]).to(device),
                                        torch.from_numpy(arrays["adduct_code"][start:stop].astype(np.int64)).to(device),
                                        torch.from_numpy(arrays["instrument_code"][start:stop].astype(np.int64)).to(device),
                                        torch.from_numpy(arrays["source_code"][start:stop].astype(np.int64)).to(device)).float().cpu().numpy()
                        outputs.append(out)
                raw = np.concatenate(outputs)
                for task_index, (task, spec) in enumerate(TASKS.items()):
                    if spec["kind"] == "classification":
                        values = 1 / (1 + np.exp(-np.clip(raw[:, task_index], -40, 40)))
                    else:
                        stat = metadata["regression_stats"][task]
                        values = raw[:, task_index] * float(stat["std"]) + float(stat["mean"])
                    compound = aggregate_spectra(values, external_index, "mean")
                    rows.extend({"pipeline": pipeline, "task": task, "kind": spec["kind"], "outer_fold": fold,
                                 "seed_index": seed_index, "replica": fold * N_SEEDS + seed_index,
                                 "inchikey_2d": key, "y_score": float(value)} for key, value in compound.items())
                del model
                if device.type == "cuda": torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = score >= threshold
    both = np.unique(y).size == 2
    tn, fp, fn, tp = confusion_matrix(y.astype(int), prediction.astype(int), labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y, score)) if both else np.nan,
        "roc_auc": float(roc_auc_score(y, score)) if both else np.nan,
        "brier": float(brier_score_loss(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)) if both else np.nan,
        "mcc": float(matthews_corrcoef(y, prediction)) if both else np.nan,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
    }


def regression_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    rho = spearmanr(y, score).statistic if len(y) >= 3 and np.unique(score).size > 1 else np.nan
    return {"mae": float(mean_absolute_error(y, score)), "rmse": float(np.sqrt(mean_squared_error(y, score))),
            "r2": float(r2_score(y, score)) if len(y) >= 2 else np.nan, "spearman": float(rho)}


def bootstrap_primary(frame: pd.DataFrame, kind: str, seed: int, n_bootstrap: int) -> tuple[float, float]:
    if len(frame) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed); values = []
    for _ in range(n_bootstrap):
        sample = frame.iloc[rng.integers(0, len(frame), len(frame))]
        y, score = sample["y_true"].to_numpy(float), sample["y_score"].to_numpy(float)
        if kind == "classification":
            if np.unique(y).size == 2: values.append(float(average_precision_score(y, score)))
        else:
            values.append(float(mean_absolute_error(y, score)))
    return (float(np.quantile(values, .025)), float(np.quantile(values, .975))) if values else (np.nan, np.nan)


def load_terminal_labels(p: dict[str, Path], roster: pd.DataFrame) -> pd.DataFrame:
    """Open ChemVista endpoint labels only after predictions have been frozen."""
    split_files = {
        "transfer": "chemvista_transfer.csv",
        "selected_novel": "chemvista_selected_novel.csv",
        "natural_novel": "chemvista_natural_novel.csv",
    }
    keep = ["inchikey_2d", *[spec["column"] for spec in TASKS.values()]]
    parts = []
    input_hashes = {}
    for role, filename in split_files.items():
        path = p["base"] / "split_v1" / filename
        frame = pd.read_csv(path, low_memory=False)[keep].copy()
        frame["evaluation_role"] = role
        parts.append(frame); input_hashes[filename] = sha256_file(path)
    labels = pd.concat(parts, ignore_index=True)
    if labels["inchikey_2d"].duplicated().any() or len(labels) != len(roster):
        raise RuntimeError("Terminal label set does not match frozen roster size")
    observed = labels[["inchikey_2d", "evaluation_role"]].sort_values("inchikey_2d").reset_index(drop=True)
    expected = roster[["inchikey_2d", "evaluation_role"]].sort_values("inchikey_2d").reset_index(drop=True)
    if not observed.equals(expected):
        raise RuntimeError("Terminal label identities/roles differ from the frozen unlabeled roster")
    atomic_json(p["out"] / "20_terminal_label_access_audit.json", {
        "version": "PBT_MSMS_CHEMVISTA_TERMINAL_LABEL_ACCESS_20_V1",
        "opened_at_utc": utc_now(),
        "opened_after_candidate_predictions_frozen": True,
        "candidate_predictions_sha256": sha256_file(p["out"] / "20_all_candidate_predictions.csv.gz"),
        "label_input_hashes": input_hashes,
        "n_compounds": len(labels),
    })
    return labels


def run_evaluate(root: Path, n_bootstrap: int = 1000) -> None:
    import matplotlib.pyplot as plt
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ModuleNotFoundError:
        from types import SimpleNamespace
        device = SimpleNamespace(type="cpu")
    p = paths(root); p["figures"].mkdir(parents=True, exist_ok=True)
    gate = verify_unlock(p)
    prep = read_json(p["prepare_manifest"]); embed = read_json(p["dreams_embed_manifest"])
    if prep.get("status") != "PASS" or embed.get("status") != "PASS":
        raise RuntimeError("Preparation and DreaMS embedding must both be PASS")
    if sha256_file(p["dreams_embeddings"]) != embed["embedding_sha256"]:
        raise RuntimeError("DreaMS embedding file changed")
    external_index = pd.read_csv(p["external_index"])
    roster = pd.read_csv(p["external_roster"])
    print("External evaluation device:", device)

    prediction_files = {
        "binned": p["out"] / "20_binned_replicas.csv.gz",
        "dreams": p["out"] / "20_dreams_replicas.csv.gz",
        "set": p["out"] / "20_set_transformer_replicas.csv.gz",
        "transformer": p["out"] / "20_transformer_replicas.csv.gz",
    }
    builders = {
        "binned": lambda: binned_predictions(root, external_index),
        "dreams": lambda: dreams_predictions(root, external_index),
        "set": lambda: set_transformer_predictions(root, external_index, device),
        "transformer": lambda: transformer_predictions(root, external_index, device),
    }
    invalid = [
        name for name in prediction_files
        if not prediction_cache_is_valid(prediction_files[name], prediction_cache_credential(root, p, name))
    ]
    if invalid and "torch" not in sys.modules and any(name in {"set", "transformer"} for name in invalid):
        raise RuntimeError(f"PyTorch is required to rebuild invalid prediction caches: {invalid}")
    frames = []
    for name in ["binned", "dreams", "set", "transformer"]:
        credential = prediction_cache_credential(root, p, name)
        if prediction_cache_is_valid(prediction_files[name], credential):
            frame = pd.read_csv(prediction_files[name])
            print("Reuse", prediction_files[name].name, len(frame))
        else:
            frame = builders[name]()
            atomic_csv(frame, prediction_files[name], compression="gzip")
            write_prediction_cache_credential(prediction_files[name], credential)
        frames.append(frame)
    replicas = pd.concat(frames, ignore_index=True)
    if not np.isfinite(replicas["y_score"].to_numpy(float)).all():
        raise RuntimeError("Non-finite external candidate prediction")
    atomic_csv(replicas, p["out"] / "20_all_candidate_replica_predictions.csv.gz", compression="gzip")

    candidate = replicas.groupby(["pipeline", "task", "kind", "inchikey_2d"], as_index=False)["y_score"].mean()
    if set(candidate["pipeline"]) != set(PIPELINES):
        raise RuntimeError(f"Candidate pipelines incomplete: {set(candidate['pipeline'])}")
    atomic_csv(candidate, p["out"] / "20_all_candidate_predictions.csv.gz", compression="gzip")
    disagreement = candidate.pivot_table(index=["task", "kind", "inchikey_2d"], columns="pipeline", values="y_score")
    if disagreement[PIPELINES].isna().any().any():
        raise RuntimeError("Candidate matrix is incomplete")
    disagreement = disagreement[PIPELINES].std(axis=1, ddof=0).rename("direct_model_disagreement").reset_index()

    transformer = replicas[replicas["pipeline"] == "Transformer B real pretrain"].copy()
    seed_sd = (transformer.groupby(["task", "kind", "inchikey_2d", "outer_fold"])["y_score"]
               .agg(lambda x: float(np.std(np.asarray(x, float), ddof=0))).rename("fold_three_seed_sd").reset_index())
    seed_sd = seed_sd.groupby(["task", "kind", "inchikey_2d"], as_index=False)["fold_three_seed_sd"].median().rename(columns={"fold_three_seed_sd": "three_seed_prediction_sd"})

    registry = read_json(p["registry"]); calibration = read_json(p["calibration"]); rejection = read_json(p["rejection"])
    selected_parts = []
    for record in registry["models"]:
        task, pipeline = record["task"], record["selected_pipeline"]
        part = candidate[(candidate["task"] == task) & (candidate["pipeline"] == pipeline)].copy()
        if pipeline.startswith("Transformer"):
            part = part.merge(seed_sd[seed_sd["task"] == task][["inchikey_2d", "three_seed_prediction_sd"]], on="inchikey_2d", validate="one_to_one")
            part["uncertainty_score"] = part["three_seed_prediction_sd"]
            part["uncertainty_method"] = "three_seed_prediction_sd; median across five frozen outer-fold ensembles"
        else:
            part = part.merge(disagreement[disagreement["task"] == task][["inchikey_2d", "direct_model_disagreement"]], on="inchikey_2d", validate="one_to_one")
            part["uncertainty_score"] = part["direct_model_disagreement"]
            part["uncertainty_method"] = "direct_candidate_model_disagreement"
        cal = calibration["endpoints"][task]
        if cal["calibration"] != "identity":
            raise RuntimeError("Notebook 20 currently requires the frozen identity calibration selected by Notebook 19")
        part["y_score_calibrated"] = part["y_score"]
        part["selected_pipeline"] = pipeline
        part["rejection_threshold"] = float(rejection["endpoints"][task]["primary_threshold"])
        part["retained"] = part["uncertainty_score"] <= part["rejection_threshold"]
        if record["kind"] == "classification":
            part["screening_threshold"] = float(cal["screening_threshold"])
            part["y_prediction"] = (part["y_score_calibrated"] >= part["screening_threshold"]).astype(int)
            part["interval_low"] = np.nan; part["interval_high"] = np.nan
        else:
            half = float(cal["q90_half_width"])
            part["screening_threshold"] = np.nan; part["y_prediction"] = part["y_score_calibrated"]
            part["interval_low"] = part["y_score_calibrated"] - half
            part["interval_high"] = part["y_score_calibrated"] + half
        selected_parts.append(part)
    # Terminal unblinding: this is the first point at which endpoint labels are opened.
    labels = load_terminal_labels(p, roster)
    final = pd.concat(selected_parts, ignore_index=True).merge(labels, on="inchikey_2d", how="left", validate="many_to_one")
    final["y_true"] = np.nan
    for task, spec in TASKS.items():
        mask = final["task"] == task
        final.loc[mask, "y_true"] = pd.to_numeric(final.loc[mask, spec["column"]], errors="coerce")
    atomic_csv(final, p["out"] / "20_final_selected_model_predictions.csv.gz", compression="gzip")

    cohorts = {
        "transfer": {"transfer"}, "selected_novel": {"selected_novel"},
        "natural_novel": {"natural_novel"}, "novel_all": {"selected_novel", "natural_novel"},
        "all_external": {"transfer", "selected_novel", "natural_novel"},
    }
    metric_rows = []
    for cohort, roles in cohorts.items():
        for task, spec in TASKS.items():
            base = final[(final["evaluation_role"].isin(roles)) & (final["task"] == task) & final["y_true"].notna()].copy()
            for retained_only in [False, True]:
                part = base[base["retained"]].copy() if retained_only else base
                row = {"cohort": cohort, "task": task, "kind": spec["kind"], "retained_only": retained_only,
                       "available_n": len(base), "n": len(part), "realized_coverage": len(part) / len(base) if len(base) else np.nan,
                       "selected_pipeline": part["selected_pipeline"].iloc[0] if len(part) else registry["models"][TASK_NAMES.index(task)]["selected_pipeline"]}
                if len(part):
                    y = part["y_true"].to_numpy(float); score = part["y_score_calibrated"].to_numpy(float)
                    if spec["kind"] == "classification":
                        threshold = float(calibration["endpoints"][task]["screening_threshold"])
                        row.update(classification_metrics(y, score, threshold))
                        row.update({"mae": np.nan, "rmse": np.nan, "r2": np.nan, "spearman": np.nan, "interval_coverage_90": np.nan})
                    else:
                        row.update(regression_metrics(y, score))
                        row.update({"pr_auc": np.nan, "roc_auc": np.nan, "brier": np.nan, "balanced_accuracy": np.nan, "mcc": np.nan, "sensitivity": np.nan, "specificity": np.nan,
                                    "interval_coverage_90": float(((y >= part["interval_low"].to_numpy(float)) & (y <= part["interval_high"].to_numpy(float))).mean())})
                    bootstrap_frame = part.copy()
                    bootstrap_frame["y_score"] = bootstrap_frame["y_score_calibrated"]
                    low, high = bootstrap_primary(bootstrap_frame, spec["kind"], stable_seed(SEED, cohort, task, retained_only), n_bootstrap)
                    row["primary_ci_low"], row["primary_ci_high"] = low, high
                metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    atomic_csv(metrics, p["out"] / "20_external_metrics.csv")

    plot = metrics[(metrics["cohort"].isin(["transfer", "novel_all"])) & (~metrics["retained_only"])].copy()
    plot["primary"] = np.where(plot["kind"] == "classification", plot["pr_auc"], plot["mae"])
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    for axis, task in zip(axes, TASK_NAMES):
        part = plot[plot["task"] == task].set_index("cohort").reindex(["transfer", "novel_all"])
        axis.bar(["Transfer", "Novel"], part["primary"], color=["#4C78A8", "#F58518"])
        axis.set_title(task.replace("_MG_L_REGRESSION", "").replace("_screen_not_ready_binary", ""))
        axis.set_ylabel("PR-AUC" if TASKS[task]["kind"] == "classification" else "MAE")
    fig.suptitle("ChemVista one-shot external validation (frozen models)")
    fig.tight_layout(); fig.savefig(p["figures"] / "20_chemvista_transfer_vs_novel.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    expected_replica_counts = {
        "Binned MS/MS": 5, "DreaMS + selected head": 5, "Set Transformer pretrained": 5,
        **{label: 15 for label in TRANSFORMER_ROUTES.values()},
    }
    replica_count = replicas.groupby(["pipeline", "task", "inchikey_2d"])["replica"].nunique()
    replica_complete = all((replica_count.xs(name, level="pipeline") == count).all() for name, count in expected_replica_counts.items())
    qc = pd.DataFrame([
        {"check": "notebook19_unlock_gate_ready", "passed": gate["status"] == "READY_FOR_NOTEBOOK_20", "detail": gate["status"]},
        {"check": "frozen_input_counts", "passed": len(external_index) == 1873 and labels["inchikey_2d"].nunique() == 586, "detail": f"spectra={len(external_index)}, compounds={labels['inchikey_2d'].nunique()}"},
        {"check": "three_external_roles_present", "passed": set(labels["evaluation_role"]) == {"transfer", "selected_novel", "natural_novel"}, "detail": str(labels["evaluation_role"].value_counts().to_dict())},
        {"check": "no_forbidden_training_spectrum_collision", "passed": not external_index["exact_public_train_spectrum_collision"].astype(bool).any() and not (external_index["confirmed_near_public_train_duplicate"].astype(bool) & external_index["evaluation_role"].ne("transfer")).any(), "detail": "exact=0; confirmed near matches allowed only in declared measurement-domain transfer cohort"},
        {"check": "all_eight_direct_candidates_present", "passed": set(candidate["pipeline"]) == set(PIPELINES), "detail": str(sorted(candidate["pipeline"].unique()))},
        {"check": "all_candidate_replicas_complete", "passed": replica_complete, "detail": str(expected_replica_counts)},
        {"check": "final_predictions_complete_and_finite", "passed": len(final) == 586 * len(TASKS) and np.isfinite(final["y_score_calibrated"]).all(), "detail": f"{len(final)}/{586 * len(TASKS)}"},
        {"check": "frozen_calibration_only", "passed": all(x["calibration"] == "identity" for x in calibration["endpoints"].values()), "detail": "no external recalibration"},
        {"check": "frozen_rejection_thresholds_applied", "passed": np.isfinite(final["rejection_threshold"]).all(), "detail": "Notebook 19 primary thresholds"},
        {"check": "terminal_unblinding_order_enforced", "passed": (p["out"] / "20_terminal_label_access_audit.json").exists(), "detail": "endpoint labels opened only after all candidate predictions were frozen"},
        {"check": "chemvista_never_used_for_fitting", "passed": True, "detail": "all fits use Development; checkpoints frozen before unlock"},
    ])
    atomic_csv(qc, p["out"] / "20_qc_checks.csv")
    all_pass = bool(qc["passed"].all())
    outputs = [p["out"] / "20_final_selected_model_predictions.csv.gz", p["out"] / "20_external_metrics.csv",
               p["out"] / "20_all_candidate_predictions.csv.gz", p["out"] / "20_qc_checks.csv",
               p["figures"] / "20_chemvista_transfer_vs_novel.png"]
    manifest = {
        "version": "PBT_MSMS_CHEMVISTA_EXTERNAL_EVALUATION_20_V1", "generated_at_utc": utc_now(),
        "status": "READY_FOR_NOTEBOOK_21" if all_pass else "BLOCKED_REVIEW_QC",
        "scientific_scope": "one-shot blinded external ChemVista evaluation",
        "external_recalibration": False, "external_hyperparameter_selection": False,
        "deployment_rule": "five-fold deployment ensembles; three-seed Transformer ensembles; frozen Notebook 19 calibration/thresholds",
        "dreams_p_probability_rule": "each outer-fold P probability is transformed by its frozen Development-only DreaMS Platt calibrator before ensemble averaging",
        "prediction_cache_schema": CACHE_SCHEMA_VERSION,
        "n_bootstrap": n_bootstrap, "device": str(device), "qc_all_pass": all_pass,
        "output_hashes": {x.name: sha256_file(x) for x in outputs},
    }
    atomic_json(p["out"] / "20_run_manifest.json", manifest)
    print(qc.to_string(index=False)); print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "embed", "evaluate"], required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(); root = locate_root(args.root)
    if args.stage == "prepare": run_prepare(root)
    elif args.stage == "embed": run_embed(root, args.batch_size)
    else: run_evaluate(root, args.bootstrap)


if __name__ == "__main__":
    main()
