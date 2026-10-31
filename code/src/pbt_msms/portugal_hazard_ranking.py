"""Notebook 24: identity-blind frozen-model inference for Portugal wastewater.

Stages
------
prepare
    Freeze the 79,867 representative DDA spectra from Notebook 23, build the
    binned and Transformer inputs, assign deterministic cross-sample feature
    groups, and write a DreaMS MGF file.  Chemical identity fields are neither
    read nor written.
embed
    Extract DreaMS embeddings in the dedicated ``env_dreams`` environment.
predict
    Recreate the frozen Development deployment ensembles used by Notebook 20,
    apply Notebook 19 calibration/rejection rules without recalibration, and
    freeze feature- and occurrence-level hazard-priority rankings before any
    identity matching.

The eight candidate pipelines are required for the exact Notebook 19
``direct_candidate_model_disagreement`` uncertainty score.  Only the single
preselected pipeline for each endpoint supplies the reported endpoint value.
"""

from __future__ import annotations

from .deployment import deployment_comparison_dir

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from pbt_msms.chemvista_external import (
    EMBEDDING_DIM,
    MAX_PEAKS,
    N_BINS,
    N_FOLDS,
    N_SEEDS,
    PIPELINES,
    TASKS,
    TASK_NAMES,
    TRANSFORMER_ROUTES,
    binned_max,
    load_binned_external,
    metadata8,
    sample_development_rows,
    stable_seed,
    transformer_classes,
    transformer_peak_encoding,
    load_dreams_probability_calibrators,
    platt_calibrate_probability,
    DREAMS_DEPLOYMENT_MODEL,
    DREAMS_DEPLOYMENT_SCOPE,
)
from pbt_msms.hazard_scoring import (
    CANONICAL_HAZARD_FORMULA,
    TASK_ORDER as HAZARD_TASK_ORDER,
    harmonized_full_hazard,
    load_development_references,
)


VERSION = "PBT_MSMS_PORTUGAL_HAZARD_RANKING_24_V2"
CACHE_SCHEMA_VERSION = "PBT_MSMS_PREDICTION_CACHE_V2"
EXPECTED_FEATURES = 79_867
GROUP_PPM = 10.0
GROUP_RT_SECONDS = 15.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        compression=compression,
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
    os.replace(temporary, path)


def locate_root(root: str | Path | None = None) -> Path:
    if root is not None:
        result = Path(root).resolve()
    else:
        cwd = Path.cwd().resolve()
        result = next((p for p in [cwd, *cwd.parents] if (p / "outputs/pbt_msms_v1").exists()), cwd)
    if not (result / "outputs/pbt_msms_v1").exists():
        raise FileNotFoundError(f"Cannot locate project root: {result}")
    return result


def paths(root: Path, tag: str = "") -> dict[str, Path]:
    if tag and not tag.startswith("_"):
        raise ValueError("A non-empty tag must start with '_' (for example _dryrun)")
    base = root / "outputs/pbt_msms_v1"
    case = root / "external_data/portugal_wastewater_5830725"
    manifests = case / "manifests"
    out = case / f"04_hazard_ranking_24{tag}"
    prefix = f"24{tag}"
    return {
        "base": base,
        "case": case,
        "manifests": manifests,
        "out": out,
        "figures": out / "figures",
        "candidate": out / "candidate_pipeline_means",
        "nb23": manifests / "23_feature_extraction_run_manifest.json",
        "nb23_index": manifests / "23_sample_ms2_feature_index.csv.gz",
        "nb23_summary": manifests / "23_sample_consensus_summary.csv",
        "registry": deployment_comparison_dir(base) / "19_final_model_registry.json",
        "calibration": deployment_comparison_dir(base) / "19_deployment_calibration.json",
        "rejection": deployment_comparison_dir(base) / "19_rejection_thresholds.json",
        "development_oof": deployment_comparison_dir(base) / "19_selected_model_oof_calibrated.csv.gz",
        "index": out / f"{prefix}_portugal_feature_index.csv.gz",
        "binned": out / f"{prefix}_portugal_binned_features.npz",
        "tokens": out / f"{prefix}_portugal_transformer_tokens.npz",
        "mgf": out / f"{prefix}_portugal_dreams_input.mgf",
        "embeddings": out / f"{prefix}_portugal_dreams_embeddings.npy",
        "prepare_manifest": manifests / f"{prefix}_preparation_manifest.json",
        "embed_manifest": manifests / f"{prefix}_dreams_embedding_manifest.json",
        "feature_predictions": out / f"{prefix}_feature_predictions_identity_blind.csv.gz",
        "group_predictions": out / f"{prefix}_case_feature_groups_identity_blind.csv.gz",
        "sample_summary": out / f"{prefix}_sample_priority_summary.csv",
        "site_summary": out / f"{prefix}_site_campaign_priority_summary.csv",
        "qc": manifests / f"{prefix}_hazard_ranking_qc_checks.csv",
        "freeze": manifests / f"{prefix}_identity_blind_freeze.json",
        "manifest": manifests / f"{prefix}_hazard_ranking_run_manifest.json",
        "figure": out / "figures" / f"{prefix}_identity_blind_overview.png",
    }


def verify_frozen_inputs(p: dict[str, Path], official: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    nb23 = read_json(p["nb23"])
    if nb23.get("status") != "READY_FOR_NOTEBOOK_24" or not nb23.get("qc_all_required_pass"):
        raise RuntimeError(f"Notebook 23 is not READY_FOR_NOTEBOOK_24: {nb23.get('status')}")
    registry = read_json(p["registry"])
    expected = {
        "P_screen_not_ready_binary": "DreaMS + selected head",
        "B_LOGBCF_REGRESSION": "Transformer B real pretrain",
        "T_EC50_MG_L_REGRESSION": "Binned MS/MS",
        "T_LC50_MG_L_REGRESSION": "Binned MS/MS",
        "T_NOEC_MG_L_REGRESSION": "Transformer B real pretrain",
    }
    observed = {row["task"]: row["selected_pipeline"] for row in registry["models"]}
    if observed != expected:
        raise RuntimeError(f"Notebook 19 model registry changed: {observed}")
    if official and nb23["counts"]["sample_consensus_model_eligible_ms2_features"] != EXPECTED_FEATURES:
        raise RuntimeError("Notebook 23 eligible-feature count changed")
    return nb23, registry


def assign_case_groups(index: pd.DataFrame) -> pd.Series:
    """Greedy centroid grouping within mode, 10 ppm and 15 s.

    A group can contain at most one feature from a sample.  The procedure is
    deterministic and uses no chemical identity or endpoint prediction.
    """
    assignments: dict[int, str] = {}
    for mode, part in index.groupby("mode", sort=True):
        ordered = part.sort_values(["precursor_mz", "rt_seconds", "sample_id", "sample_feature_id"])
        active: list[dict[str, Any]] = []
        serial = 0
        for row in tqdm(ordered.itertuples(), total=len(ordered), desc=f"Group {mode} features"):
            mz = float(row.precursor_mz)
            rt = float(row.rt_seconds)
            sample = str(row.sample_id)
            kept = []
            for cluster in active:
                ppm_forward = (mz - cluster["mz"]) / max(mz, 1e-12) * 1e6
                if ppm_forward <= GROUP_PPM:
                    kept.append(cluster)
            active = kept
            candidates: list[tuple[float, int]] = []
            for position, cluster in enumerate(active):
                if sample in cluster["samples"]:
                    continue
                ppm = abs(mz - cluster["mz"]) / mz * 1e6
                rt_error = abs(rt - cluster["rt"])
                if ppm <= GROUP_PPM and rt_error <= GROUP_RT_SECONDS:
                    score = (ppm / GROUP_PPM) ** 2 + (rt_error / GROUP_RT_SECONDS) ** 2
                    candidates.append((score, position))
            if candidates:
                _, position = min(candidates)
                cluster = active[position]
                count = cluster["n"] + 1
                cluster["mz"] = (cluster["mz"] * cluster["n"] + mz) / count
                cluster["rt"] = (cluster["rt"] * cluster["n"] + rt) / count
                cluster["n"] = count
                cluster["samples"].add(sample)
            else:
                serial += 1
                cluster = {
                    "id": f"PTWW_{mode}_G{serial:06d}",
                    "mz": mz,
                    "rt": rt,
                    "n": 1,
                    "samples": {sample},
                }
                active.append(cluster)
            assignments[int(row.Index)] = cluster["id"]
    return pd.Series(assignments).reindex(index.index)


def load_peak_slice(row: Any, cache: dict[str, dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    path = str(row.peak_file)
    if path not in cache:
        # A/B files from one sample are interleaved by precursor m/z.  Keeping
        # four files avoids repeated reads while preventing all 96 variable-
        # length peak arrays from accumulating in RAM.
        while len(cache) >= 4:
            cache.pop(next(iter(cache)))
        with np.load(path, allow_pickle=False) as payload:
            cache[path] = {name: payload[name] for name in payload.files}
    payload = cache[path]
    start = int(row.peak_offset)
    stop = start + int(row.peak_count)
    mz = np.asarray(payload["mz"][start:stop], dtype=np.float32)
    intensity = np.asarray(payload["intensity"][start:stop], dtype=np.float32)
    if len(mz) != int(row.clean_peak_n) or len(mz) < 3:
        raise RuntimeError(f"Peak slice mismatch: {row.sample_feature_id}")
    if not np.isfinite(mz).all() or not np.isfinite(intensity).all() or np.any(intensity <= 0):
        raise RuntimeError(f"Invalid peak slice: {row.sample_feature_id}")
    return mz, intensity


def run_prepare(root: Path, tag: str = "", limit: int | None = None) -> None:
    p = paths(root, tag)
    official = not tag
    nb23, _ = verify_frozen_inputs(p, official=official)
    p["out"].mkdir(parents=True, exist_ok=True)
    p["figures"].mkdir(parents=True, exist_ok=True)
    p["candidate"].mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(p["nb23_index"], low_memory=False)
    if not index["model_eligible"].astype(bool).all() or index["sample_feature_id"].duplicated().any():
        raise RuntimeError("Notebook 23 primary feature index is invalid")
    forbidden = {"name", "smiles", "inchikey", "inchikey_2d", "formula", "identity", "annotation"}
    if forbidden.intersection({column.lower() for column in index.columns}):
        raise RuntimeError("Chemical-identity field found in the Notebook 23 feature index")
    index = index.sort_values(["mode", "sample_id", "precursor_mz", "rt_seconds", "sample_feature_id"]).reset_index(drop=True)
    if limit is not None:
        if official:
            raise ValueError("--limit is permitted only with a non-empty --tag")
        index = index.head(int(limit)).copy()
    index["feature_row"] = np.arange(len(index), dtype=np.int64)

    sample_summary = pd.read_csv(p["nb23_summary"])
    median_depth = float(sample_summary["model_eligible_ms2_features"].median())
    low_depth_ids = set(
        sample_summary.loc[
            sample_summary["model_eligible_ms2_features"] < max(100.0, 0.25 * median_depth),
            "sample_id",
        ].astype(str)
    )
    index["low_ms2_depth_sample"] = index["sample_id"].astype(str).isin(low_depth_ids)
    index["case_feature_id"] = assign_case_groups(index).astype(str)

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
    adduct_map = {value: i for i, value in enumerate(vocab["adduct"])}
    instrument_map = {value: i for i, value in enumerate(vocab["instrument_category"])}
    source_map = {value: i for i, value in enumerate(vocab["source_library"])}
    peak_cache: dict[str, dict[str, np.ndarray]] = {}
    mgf_temporary = p["mgf"].with_suffix(".mgf.tmp")

    with mgf_temporary.open("w", encoding="utf-8", newline="\n") as mgf:
        for row in tqdm(index.itertuples(index=False), total=len(index), desc="Prepare Portugal model inputs"):
            mz, intensity_raw = load_peak_slice(row, peak_cache)
            intensity_sqrt = np.sqrt(intensity_raw / float(intensity_raw.max()))
            precursor = float(row.precursor_mz)
            polarity = 1.0 if str(row.mode) == "Pos" else -1.0
            mode_text = "Positive" if polarity > 0 else "Negative"
            ce = float(row.collision_energy_ev) if np.isfinite(row.collision_energy_ev) else np.nan
            feature_row = int(row.feature_row)

            for column, value in binned_max(mz, intensity_sqrt).items():
                fragment_rows.append(feature_row)
                fragment_cols.append(column)
                fragment_data.append(value)
            for column, value in binned_max(precursor - mz, intensity_sqrt).items():
                neutral_rows.append(feature_row)
                neutral_cols.append(column)
                neutral_data.append(value)
            ce_missing = not np.isfinite(ce) or ce < 0
            binned_meta[feature_row] = [
                precursor,
                polarity,
                0.0 if ce_missing else ce,
                float(ce_missing),
                math.log1p(len(mz)),
                float(not np.isfinite(precursor)),
            ]

            token, padding, retained_n = transformer_peak_encoding(mz, intensity_raw, precursor)
            peak_tokens[feature_row] = token
            peak_padding[feature_row] = padding
            token_meta[feature_row] = metadata8(precursor, polarity, ce, retained_n, "QTOF")
            adduct_codes[feature_row] = adduct_map.get("UNKNOWN", 0)
            instrument_codes[feature_row] = instrument_map.get("QTOF", instrument_map.get("UNKNOWN", 0))
            source_codes[feature_row] = source_map.get("UNKNOWN", 0)

            charge_text = "1+" if polarity > 0 else "1-"
            mgf.write("BEGIN IONS\n")
            mgf.write(f"TITLE={row.sample_feature_id}\nFEATURE_ID={feature_row + 1}\nMSLEVEL=2\n")
            mgf.write(f"PEPMASS={precursor:.8f}\nCHARGE={charge_text}\nIONMODE={mode_text}\n")
            if not ce_missing:
                mgf.write(f"COLLISION_ENERGY={ce:.8g}\n")
            mgf.write("SOURCE_LIBRARY=PortugalWastewater\n")
            for mass, abundance in zip(mz, intensity_raw):
                mgf.write(f"{float(mass):.8f} {float(abundance):.8f}\n")
            mgf.write("END IONS\n\n")
    os.replace(mgf_temporary, p["mgf"])

    shape = (len(index), N_BINS)
    x_fragment = sparse.csr_matrix((fragment_data, (fragment_rows, fragment_cols)), shape=shape, dtype=np.float32)
    x_neutral = sparse.csr_matrix((neutral_data, (neutral_rows, neutral_cols)), shape=shape, dtype=np.float32)
    atomic_npz(
        p["binned"],
        fragment_data=x_fragment.data,
        fragment_indices=x_fragment.indices,
        fragment_indptr=x_fragment.indptr,
        neutral_data=x_neutral.data,
        neutral_indices=x_neutral.indices,
        neutral_indptr=x_neutral.indptr,
        shape=np.asarray(shape),
        metadata=binned_meta,
    )
    atomic_npz(
        p["tokens"],
        peak_tokens=peak_tokens,
        padding_mask=peak_padding,
        metadata_numeric=token_meta,
        adduct_code=adduct_codes,
        instrument_code=instrument_codes,
        source_code=source_codes,
    )
    atomic_csv(index, p["index"], compression="gzip")

    group_sizes = index.groupby("case_feature_id")["sample_id"].nunique()
    payload = {
        "version": VERSION,
        "stage": "prepare",
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "official_run": official,
        "tag": tag,
        "n_features": len(index),
        "n_samples": int(index["sample_id"].nunique()),
        "n_case_feature_groups": int(index["case_feature_id"].nunique()),
        "mode_counts": index["mode"].value_counts().to_dict(),
        "low_ms2_depth_samples": sorted(low_depth_ids),
        "identity_blind": True,
        "identity_fields_read": [],
        "grouping": {
            "ppm": GROUP_PPM,
            "rt_seconds": GROUP_RT_SECONDS,
            "maximum_one_feature_per_sample_per_group": True,
            "max_observed_sample_count_per_group": int(group_sizes.max()),
        },
        "notebook23_manifest_sha256": sha256_file(p["nb23"]),
        "frozen_model_artifact_hashes": {
            path.name: sha256_file(path)
            for path in [p["registry"], p["calibration"], p["rejection"]]
        },
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in [p["index"], p["binned"], p["tokens"], p["mgf"]]
        },
    }
    atomic_json(p["prepare_manifest"], payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_embed(root: Path, tag: str = "", batch_size: int = 16) -> None:
    p = paths(root, tag)
    prep = read_json(p["prepare_manifest"])
    if prep.get("status") != "PASS" or sha256_file(p["mgf"]) != prep["artifacts"][p["mgf"].name]["sha256"]:
        raise RuntimeError("Notebook 24 preparation is not frozen PASS")
    from pbt_msms.dreams_mvp import configure_writable_runtime_cache

    configure_writable_runtime_cache()
    import torch
    from dreams.api import dreams_embeddings

    if not torch.cuda.is_available():
        raise RuntimeError("DreaMS extraction requires a CUDA-enabled env_dreams kernel")
    result = dreams_embeddings(p["mgf"], batch_size=batch_size, progress_bar=True, store_embs=False)
    embeddings = np.asarray(result, dtype=np.float32)
    expected = (int(prep["n_features"]), EMBEDDING_DIM)
    if embeddings.shape != expected or not np.isfinite(embeddings).all():
        raise RuntimeError(f"Unexpected DreaMS embeddings: {embeddings.shape}, expected {expected}")
    atomic_npy(p["embeddings"], embeddings)
    payload = {
        "version": VERSION,
        "stage": "embed",
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "batch_size": batch_size,
        "input_mgf_sha256": sha256_file(p["mgf"]),
        "embedding_sha256": sha256_file(p["embeddings"]),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    atomic_json(p["embed_manifest"], payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def pipeline_path(p: dict[str, Path], pipeline: str) -> Path:
    slug = (
        pipeline.lower()
        .replace(" + ", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )
    return p["candidate"] / f"{slug}.npz"


def candidate_cache_credential(p: dict[str, Path], pipeline: str) -> dict[str, Any]:
    model_manifests = {
        "Binned MS/MS": p["base"] / "simple_spectrum_baselines_v1/simple_spectrum_run_manifest.json",
        "DreaMS + selected head": p["base"] / "dreams_heads_v1/dreams_head_run_manifest.json",
        "Set Transformer pretrained": p["base"] / "set_transformer_v1/set_transformer_run_manifest.json",
    }
    default_manifest = p["base"] / "transformer_finetuning_v1/transformer_finetuning_manifest.json"
    input_path = p["binned"] if pipeline == "Binned MS/MS" else p["embeddings"] if pipeline == "DreaMS + selected head" else p["tokens"]
    manifest = model_manifests.get(pipeline, default_manifest)
    required = [p["index"], input_path, manifest, p["registry"], p["calibration"], p["rejection"]]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Candidate-cache credential inputs are missing: {missing}")
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "pipeline": pipeline,
        "preprocessing_version": "portugal_hazard_ranking_v2_dreams_fold_platt",
        "feature_index_sha256": sha256_file(p["index"]),
        "inference_input_sha256": sha256_file(input_path),
        "model_manifest_sha256": sha256_file(manifest),
        "model_registry_sha256": sha256_file(p["registry"]),
        "deployment_calibration_sha256": sha256_file(p["calibration"]),
        "rejection_thresholds_sha256": sha256_file(p["rejection"]),
    }


def candidate_cache_is_valid(path: Path, expected: dict[str, Any]) -> bool:
    credential_path = path.with_suffix(path.suffix + ".credential.json")
    if not path.is_file() or not credential_path.is_file():
        return False
    observed = read_json(credential_path)
    return observed.get("credential") == expected and observed.get("prediction_sha256") == sha256_file(path)


def save_pipeline(
    path: Path,
    predictions: np.ndarray,
    seed_sd: np.ndarray | None = None,
    credential: dict[str, Any] | None = None,
) -> None:
    arrays = {
        "predictions": np.asarray(predictions, dtype=np.float32),
        "task_names": np.asarray(TASK_NAMES, dtype="U64"),
    }
    if seed_sd is not None:
        arrays["three_seed_prediction_sd"] = np.asarray(seed_sd, dtype=np.float32)
    atomic_npz(path, **arrays)
    if credential is not None:
        atomic_json(path.with_suffix(path.suffix + ".credential.json"), {
            "credential": credential,
            "prediction_sha256": sha256_file(path),
            "created_at_utc": utc_now(),
        })


def load_pipeline(path: Path, n_features: int) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as payload:
        predictions = np.asarray(payload["predictions"], dtype=np.float32)
        seed_sd = np.asarray(payload["three_seed_prediction_sd"], dtype=np.float32) if "three_seed_prediction_sd" in payload.files else None
        names = payload["task_names"].astype(str).tolist()
    if predictions.shape != (n_features, len(TASK_NAMES)) or names != TASK_NAMES:
        raise RuntimeError(f"Invalid candidate prediction artifact: {path}")
    if not np.isfinite(predictions).all() or (seed_sd is not None and not np.isfinite(seed_sd).all()):
        raise RuntimeError(f"Non-finite candidate prediction artifact: {path}")
    return predictions, seed_sd


def build_binned(root: Path, p: dict[str, Path], n_features: int) -> np.ndarray:
    simple = p["base"] / "simple_spectrum_baselines_v1"
    dev_spectra = pd.read_csv(simple / "spectrum_feature_row_index.csv")
    development = pd.read_csv(p["base"] / "split_v1/development_compounds.csv", low_memory=False)
    xf_dev = sparse.load_npz(simple / "fragment_intensity_1Da.npz")
    xn_dev = sparse.load_npz(simple / "neutral_loss_intensity_1Da.npz")
    meta_dev = np.load(simple / "spectrum_metadata.npy", allow_pickle=False)
    xf_ext, xn_ext, meta_ext = load_binned_external(p["binned"])
    label_frame = development.set_index("inchikey_2d")
    spectrum_keys = set(dev_spectra["inchikey_2d"].astype(str))
    result = np.zeros((n_features, len(TASK_NAMES)), dtype=np.float64)
    for task_index, (task, spec) in enumerate(TASKS.items()):
        available = development.loc[
            pd.to_numeric(development[spec["column"]], errors="coerce").notna()
            & development["inchikey_2d"].astype(str).isin(spectrum_keys)
        ].copy()
        fold_predictions = []
        for fold in range(N_FOLDS):
            print(f"Binned {task} fold {fold + 1}/{N_FOLDS}", flush=True)
            keys = available.loc[available["cv_fold"].astype(int).ne(fold), "inchikey_2d"].tolist()
            train_rows = sample_development_rows(dev_spectra, keys, task, fold)
            train_keys = dev_spectra.iloc[train_rows]["inchikey_2d"].tolist()
            y = np.asarray([label_frame.at[key, spec["column"]] for key in train_keys], dtype=float)
            scaler = StandardScaler().fit(meta_dev[train_rows])
            x_train = sparse.hstack(
                [xf_dev[train_rows], xn_dev[train_rows], sparse.csr_matrix(scaler.transform(meta_dev[train_rows]))],
                format="csr",
            )
            x_ext = sparse.hstack(
                [xf_ext, xn_ext, sparse.csr_matrix(scaler.transform(meta_ext))],
                format="csr",
            )
            if spec["kind"] == "classification":
                model = LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    max_iter=4000,
                    random_state=stable_seed(42, "binned", task, fold),
                )
                model.fit(x_train, y.astype(int))
                values = model.predict_proba(x_ext)[:, 1]
            else:
                model = Ridge(alpha=10.0, solver="lsqr", tol=1e-4, max_iter=3000)
                model.fit(x_train, y)
                values = model.predict(x_ext)
            fold_predictions.append(np.asarray(values, dtype=np.float64))
        result[:, task_index] = np.mean(fold_predictions, axis=0)
    return result


def build_dreams(root: Path, p: dict[str, Path], n_features: int) -> np.ndarray:
    from pbt_msms.dreams_heads import _fit_final_predictor, _predict

    dev_dir = p["base"] / "dreams_heads_v1"
    dev_index = pd.read_csv(dev_dir / "dreams_compound_embedding_index.csv")
    with np.load(dev_dir / "dreams_compound_embedding_routes.npz", allow_pickle=False) as payload:
        x_dev = payload["all_spectra"].copy()
    x_ext = np.load(p["embeddings"], mmap_mode="r", allow_pickle=False)
    oof = pd.read_csv(dev_dir / "dreams_head_compound_level_oof.csv.gz", low_memory=False)
    selected = oof[(oof["analysis_scope"] == DREAMS_DEPLOYMENT_SCOPE) & (oof["model"] == DREAMS_DEPLOYMENT_MODEL)]
    calibrators = load_dreams_probability_calibrators(dev_dir)
    development_folds = dev_index["cv_fold"].astype(int).to_numpy()
    result = np.zeros((n_features, len(TASK_NAMES)), dtype=np.float64)
    for task_index, (task, spec) in enumerate(TASKS.items()):
        labels = pd.to_numeric(dev_index[spec["column"]], errors="coerce").to_numpy(float)
        fold_predictions = []
        for fold in range(N_FOLDS):
            print(f"DreaMS head {task} fold {fold + 1}/{N_FOLDS}", flush=True)
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
                probability = platt_calibrate_probability(probability, coefficient, intercept)
                fold_predictions.append(probability)
            else:
                fold_predictions.append(prediction)
        result[:, task_index] = np.mean(fold_predictions, axis=0)
    return result


def build_set_transformer(p: dict[str, Path], n_features: int, device: Any, batch_size: int) -> np.ndarray:
    import torch
    from pbt_msms.transformer_pipeline import TASK_NAMES as SET_TASKS
    from pbt_msms.transformer_pipeline import _model_from_initialization

    if list(SET_TASKS) != TASK_NAMES:
        raise RuntimeError("Set Transformer task order changed")
    config = {"d_model": 128, "nhead": 4, "num_layers": 4, "dim_feedforward": 256, "dropout": 0.1}
    # Reconstruct the three-input Set Transformer representation directly from
    # the four-channel frozen peak tokens: m/z, intensity and neutral loss.
    with np.load(p["tokens"], allow_pickle=False) as payload:
        peak = payload["peak_tokens"].astype(np.float32)
        padding = payload["padding_mask"]
        meta8 = payload["metadata_numeric"].astype(np.float32)
    features = peak[:, :, [0, 1, 2]].astype(np.float32)
    metadata = meta8[:, :6].astype(np.float32)
    result_folds = []
    for fold in range(N_FOLDS):
        print(f"Set Transformer fold {fold + 1}/{N_FOLDS}", flush=True)
        checkpoint = torch.load(
            p["base"] / f"set_transformer_v1/checkpoints/set_transformer_pretrained_fold{fold}.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = _model_from_initialization(config, None, stable_seed(42, "external", fold))
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device).eval()
        parts = []
        with torch.no_grad():
            for start in tqdm(range(0, n_features, batch_size), desc=f"Set fold {fold}"):
                stop = min(start + batch_size, n_features)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    output = model(
                        torch.from_numpy(features[start:stop]).to(device),
                        torch.from_numpy(padding[start:stop]).to(device),
                        torch.from_numpy(metadata[start:stop]).to(device),
                    ).float().cpu().numpy()
                parts.append(output)
        raw = np.concatenate(parts)
        means = np.asarray(checkpoint["label_mean"])
        stds = np.asarray(checkpoint["label_std"])
        converted = raw.copy()
        for task_index, task in enumerate(TASK_NAMES):
            if TASKS[task]["kind"] == "classification":
                converted[:, task_index] = 1.0 / (1.0 + np.exp(-np.clip(raw[:, task_index], -40, 40)))
            else:
                converted[:, task_index] = raw[:, task_index] * stds[task_index] + means[task_index]
        result_folds.append(converted)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.mean(result_folds, axis=0)


def build_transformer_route(
    p: dict[str, Path], route: str, n_features: int, device: Any, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    with np.load(p["tokens"], allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    Model = transformer_classes()
    fold_means = []
    fold_seed_sd = []
    for fold in range(N_FOLDS):
        seed_predictions = []
        for seed_index in range(N_SEEDS):
            print(f"Transformer {route} fold {fold + 1}/{N_FOLDS} seed {seed_index + 1}/{N_SEEDS}", flush=True)
            checkpoint_path = p["base"] / f"transformer_finetuning_v1/checkpoints/outer_{fold}/{route}__seed_{seed_index}__best.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model_metadata = checkpoint["metadata"]
            model = Model(model_metadata["model_config"], model_metadata["vocab_sizes"], len(TASK_NAMES))
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.to(device).eval()
            parts = []
            with torch.no_grad():
                for start in tqdm(range(0, n_features, batch_size), desc=f"{route} f{fold} s{seed_index}"):
                    stop = min(start + batch_size, n_features)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=device.type == "cuda",
                    ):
                        raw = model(
                            torch.from_numpy(arrays["peak_tokens"][start:stop].astype(np.float32)).to(device),
                            torch.from_numpy(arrays["padding_mask"][start:stop]).to(device),
                            torch.from_numpy(arrays["metadata_numeric"][start:stop]).to(device),
                            torch.from_numpy(arrays["adduct_code"][start:stop].astype(np.int64)).to(device),
                            torch.from_numpy(arrays["instrument_code"][start:stop].astype(np.int64)).to(device),
                            torch.from_numpy(arrays["source_code"][start:stop].astype(np.int64)).to(device),
                        ).float().cpu().numpy()
                    parts.append(raw)
            raw = np.concatenate(parts)
            converted = raw.copy()
            for task_index, (task, spec) in enumerate(TASKS.items()):
                if spec["kind"] == "classification":
                    converted[:, task_index] = 1.0 / (1.0 + np.exp(-np.clip(raw[:, task_index], -40, 40)))
                else:
                    stat = model_metadata["regression_stats"][task]
                    converted[:, task_index] = raw[:, task_index] * float(stat["std"]) + float(stat["mean"])
            seed_predictions.append(converted)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        seed_stack = np.stack(seed_predictions, axis=0)
        fold_means.append(seed_stack.mean(axis=0))
        fold_seed_sd.append(seed_stack.std(axis=0, ddof=0))
    return np.mean(fold_means, axis=0), np.median(fold_seed_sd, axis=0)


def add_within_group_rank(frame: pd.DataFrame, score: str, group: str, output: str) -> None:
    frame[output] = frame.groupby(group, sort=False)[score].rank(method="min", ascending=False, na_option="bottom")


def freeze_rankings(
    p: dict[str, Path], index: pd.DataFrame, candidates: dict[str, np.ndarray], seed_sd: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry = read_json(p["registry"])
    calibration = read_json(p["calibration"])
    rejection = read_json(p["rejection"])
    selected = {row["task"]: row["selected_pipeline"] for row in registry["models"]}
    task_index = {task: i for i, task in enumerate(TASK_NAMES)}
    predictions = index.copy()

    direct_stack = np.stack([candidates[pipeline] for pipeline in PIPELINES], axis=0)
    direct_disagreement = direct_stack.std(axis=0, ddof=0)
    endpoint_columns = {
        "P_screen_not_ready_binary": "p_probability",
        "B_LOGBCF_REGRESSION": "b_logbcf",
        "T_EC50_MG_L_REGRESSION": "t_ec50_log10_mg_l",
        "T_LC50_MG_L_REGRESSION": "t_lc50_log10_mg_l",
        "T_NOEC_MG_L_REGRESSION": "t_noec_log10_mg_l",
    }
    prefixes = {
        "P_screen_not_ready_binary": "p",
        "B_LOGBCF_REGRESSION": "b",
        "T_EC50_MG_L_REGRESSION": "t_ec50",
        "T_LC50_MG_L_REGRESSION": "t_lc50",
        "T_NOEC_MG_L_REGRESSION": "t_noec",
    }
    for task in TASK_NAMES:
        column = endpoint_columns[task]
        prefix = prefixes[task]
        i = task_index[task]
        pipeline = selected[task]
        predictions[column] = candidates[pipeline][:, i]
        if pipeline.startswith("Transformer"):
            uncertainty = seed_sd[:, i]
            method = "three_seed_prediction_sd; median across five frozen outer-fold ensembles"
        else:
            uncertainty = direct_disagreement[:, i]
            method = "direct_candidate_model_disagreement"
        threshold = float(rejection["endpoints"][task]["primary_threshold"])
        predictions[f"{prefix}_uncertainty"] = uncertainty
        predictions[f"{prefix}_rejection_threshold"] = threshold
        predictions[f"{prefix}_retained"] = uncertainty <= threshold
        predictions[f"{prefix}_uncertainty_method"] = method
        predictions[f"{prefix}_selected_pipeline"] = pipeline
        cal = calibration["endpoints"][task]
        if cal["calibration"] != "identity":
            raise RuntimeError("Notebook 24 requires Notebook 19 identity calibration")
        if TASKS[task]["kind"] == "classification":
            predictions["p_screen_positive"] = predictions[column] >= float(cal["screening_threshold"])
            predictions["p_screening_threshold"] = float(cal["screening_threshold"])
        else:
            half = float(cal["q90_half_width"])
            predictions[f"{prefix}_interval_low"] = predictions[column] - half
            predictions[f"{prefix}_interval_high"] = predictions[column] + half

    if tuple(TASK_NAMES) != HAZARD_TASK_ORDER:
        raise RuntimeError("Endpoint order differs from the shared canonical hazard definition")
    references = load_development_references(p["development_oof"])
    raw_predictions = {
        task: predictions[endpoint_columns[task]].to_numpy(float) for task in TASK_NAMES
    }
    endpoint_hazard, toxicity, joint_hazard = harmonized_full_hazard(raw_predictions, references)
    hazard_columns = {}
    for task in TASK_NAMES:
        prefix = prefixes[task]
        hazard_column = f"{prefix}_development_anchored_hazard_percentile"
        predictions[hazard_column] = endpoint_hazard[task]
        hazard_columns[task] = hazard_column

    predictions["toxicity_hazard_percentile"] = toxicity
    predictions["joint_pbt_hazard_score"] = joint_hazard
    retained_columns = [f"{prefixes[task]}_retained" for task in TASK_NAMES]
    predictions["retained_endpoint_n"] = predictions[retained_columns].sum(axis=1).astype(int)
    predictions["all_endpoints_retained"] = predictions["retained_endpoint_n"].eq(len(TASK_NAMES))
    predictions["primary_hazard_score"] = predictions["joint_pbt_hazard_score"].where(
        predictions["all_endpoints_retained"]
    )
    predictions["overall_hazard_rank"] = predictions["primary_hazard_score"].rank(
        method="min", ascending=False, na_option="bottom"
    )
    add_within_group_rank(predictions, "primary_hazard_score", "mode", "mode_hazard_rank")
    add_within_group_rank(predictions, "primary_hazard_score", "sample_id", "sample_hazard_rank")
    predictions["sample_eligible_n"] = predictions.groupby("sample_id")["all_endpoints_retained"].transform("sum")
    predictions["sample_hazard_rank_fraction"] = predictions["sample_hazard_rank"] / predictions["sample_eligible_n"].replace(0, np.nan)
    for cutoff in (5, 10, 20):
        predictions[f"sample_top_{cutoff}pct_hazard"] = (
            predictions["all_endpoints_retained"]
            & (predictions["sample_hazard_rank_fraction"] <= cutoff / 100.0)
        )

    log_area = np.log10(np.maximum(predictions["intensity_area_mean_detected"].to_numpy(float), 0.0) + 1.0)
    predictions["log10_intensity_area"] = log_area
    predictions["sample_abundance_percentile"] = predictions.groupby("sample_id")["log10_intensity_area"].rank(
        pct=True, method="average"
    )

    aggregate_columns = [
        "precursor_mz",
        "rt_seconds",
        "primary_hazard_score",
        "joint_pbt_hazard_score",
        "p_probability",
        "b_logbcf",
        "t_ec50_log10_mg_l",
        "t_lc50_log10_mg_l",
        "t_noec_log10_mg_l",
        hazard_columns["P_screen_not_ready_binary"],
        hazard_columns["B_LOGBCF_REGRESSION"],
        "toxicity_hazard_percentile",
        "sample_abundance_percentile",
    ]
    groups = predictions.groupby(["case_feature_id", "mode"], as_index=False).agg(
        **{f"median_{column}": (column, "median") for column in aggregate_columns},
        detected_sample_n=("sample_id", "nunique"),
        detected_site_n=("site_code", "nunique"),
        detected_campaign_n=("campaign", "nunique"),
        retained_feature_n=("all_endpoints_retained", "sum"),
        member_feature_n=("sample_feature_id", "size"),
    )
    sample_denominator = predictions.groupby("mode")["sample_id"].nunique().to_dict()
    groups["detection_frequency_fraction"] = [
        row.detected_sample_n / sample_denominator[row.mode] for row in groups.itertuples(index=False)
    ]
    groups["retained_feature_fraction"] = groups["retained_feature_n"] / groups["member_feature_n"]
    groups["occurrence_assisted_score"] = np.sqrt(
        groups["median_joint_pbt_hazard_score"] * groups["detection_frequency_fraction"]
    )
    groups["hazard_rank"] = groups["median_primary_hazard_score"].rank(
        method="min", ascending=False, na_option="bottom"
    )
    groups["occurrence_assisted_rank"] = groups["occurrence_assisted_score"].rank(
        method="min", ascending=False, na_option="bottom"
    )
    groups["abundance_baseline_rank"] = groups["median_sample_abundance_percentile"].rank(
        method="min", ascending=False, na_option="bottom"
    )
    groups["frequency_baseline_rank"] = groups["detection_frequency_fraction"].rank(
        method="min", ascending=False, na_option="bottom"
    )

    sample_summary = predictions.groupby(
        ["sample_id", "mode", "site_code", "campaign", "low_ms2_depth_sample"], as_index=False
    ).agg(
        feature_n=("sample_feature_id", "size"),
        retained_all_endpoint_n=("all_endpoints_retained", "sum"),
        median_joint_hazard_score=("joint_pbt_hazard_score", "median"),
        median_primary_hazard_score=("primary_hazard_score", "median"),
        p90_primary_hazard_score=("primary_hazard_score", lambda x: x.quantile(0.90)),
        top_5pct_n=("sample_top_5pct_hazard", "sum"),
        top_10pct_n=("sample_top_10pct_hazard", "sum"),
        top_20pct_n=("sample_top_20pct_hazard", "sum"),
    )
    sample_summary["retained_all_endpoint_fraction"] = (
        sample_summary["retained_all_endpoint_n"] / sample_summary["feature_n"]
    )
    site_summary = predictions.groupby(["mode", "site_code", "campaign"], as_index=False).agg(
        feature_n=("sample_feature_id", "size"),
        retained_all_endpoint_n=("all_endpoints_retained", "sum"),
        median_joint_hazard_score=("joint_pbt_hazard_score", "median"),
        p90_primary_hazard_score=("primary_hazard_score", lambda x: x.quantile(0.90)),
        top_10pct_n=("sample_top_10pct_hazard", "sum"),
    )
    site_summary["retained_all_endpoint_fraction"] = (
        site_summary["retained_all_endpoint_n"] / site_summary["feature_n"]
    )
    return predictions, groups, sample_summary, site_summary


def make_figure(p: dict[str, Path], predictions: pd.DataFrame, sample_summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for mode, color in [("Pos", "#4C78A8"), ("Neg", "#F58518")]:
        values = predictions.loc[
            (predictions["mode"] == mode) & predictions["primary_hazard_score"].notna(),
            "primary_hazard_score",
        ]
        axes[0].hist(values, bins=40, alpha=0.55, label=mode, color=color)
    axes[0].set_xlabel("Development-anchored joint P/B/T hazard score")
    axes[0].set_ylabel("Feature count")
    axes[0].legend(frameon=False)

    retention_columns = ["p_retained", "b_retained", "t_ec50_retained", "t_lc50_retained", "t_noec_retained"]
    retention = [float(predictions[column].mean()) for column in retention_columns]
    axes[1].bar(["P", "B", "EC50", "LC50", "NOEC"], retention, color="#59A14F")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Frozen-threshold retained fraction")

    ordered = sample_summary.sort_values(["mode", "site_code", "campaign"])
    labels = [f"{r.mode}:{r.site_code}:C{r.campaign}" for r in ordered.itertuples(index=False)]
    colors = ["#4C78A8" if value == "Pos" else "#F58518" for value in ordered["mode"]]
    axes[2].bar(np.arange(len(ordered)), ordered["p90_primary_hazard_score"], color=colors)
    axes[2].set_xticks(np.arange(len(ordered)))
    axes[2].set_xticklabels(labels, rotation=90, fontsize=6)
    axes[2].set_ylabel("Within-sample 90th percentile hazard score")
    figure.suptitle("Portugal wastewater identity-blind frozen-model prioritization")
    figure.tight_layout()
    p["figure"].parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(p["figure"], dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_predict(root: Path, tag: str = "", batch_size: int = 256) -> None:
    p = paths(root, tag)
    official = not tag
    nb23, _ = verify_frozen_inputs(p, official=official)
    prep = read_json(p["prepare_manifest"])
    embed = read_json(p["embed_manifest"])
    if prep.get("status") != "PASS" or embed.get("status") != "PASS":
        raise RuntimeError("Preparation and embedding stages must both be PASS")
    if sha256_file(p["index"]) != prep["artifacts"][p["index"].name]["sha256"]:
        raise RuntimeError("Frozen Notebook 24 feature index changed")
    if sha256_file(p["embeddings"]) != embed["embedding_sha256"]:
        raise RuntimeError("Frozen Notebook 24 DreaMS embeddings changed")

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    invalid_caches = [
        pipeline for pipeline in PIPELINES
        if not candidate_cache_is_valid(pipeline_path(p, pipeline), candidate_cache_credential(p, pipeline))
    ]
    gpu_rebuilds = [pipeline for pipeline in invalid_caches if pipeline == "Set Transformer pretrained" or pipeline.startswith("Transformer")]
    if official and device.type != "cuda" and gpu_rebuilds:
        raise RuntimeError(f"Official Notebook 24 GPU candidate caches require rebuilding: {gpu_rebuilds}")
    print("Notebook 24 inference device:", device, flush=True)
    index = pd.read_csv(p["index"], low_memory=False)
    n_features = len(index)

    builders = {
        "Binned MS/MS": lambda: (build_binned(root, p, n_features), None),
        "DreaMS + selected head": lambda: (build_dreams(root, p, n_features), None),
        "Set Transformer pretrained": lambda: (build_set_transformer(p, n_features, device, batch_size), None),
    }
    for route, pipeline in TRANSFORMER_ROUTES.items():
        builders[pipeline] = lambda route=route: build_transformer_route(p, route, n_features, device, batch_size)

    candidates: dict[str, np.ndarray] = {}
    transformer_b_seed_sd: np.ndarray | None = None
    for pipeline in PIPELINES:
        output = pipeline_path(p, pipeline)
        credential = candidate_cache_credential(p, pipeline)
        if candidate_cache_is_valid(output, credential):
            prediction, seed_sd = load_pipeline(output, n_features)
            print("Reuse", output.name, flush=True)
        else:
            prediction, seed_sd = builders[pipeline]()
            save_pipeline(output, prediction, seed_sd, credential)
        candidates[pipeline] = prediction
        if pipeline == "Transformer B real pretrain":
            transformer_b_seed_sd = seed_sd
    if transformer_b_seed_sd is None:
        raise RuntimeError("Transformer B three-seed uncertainty is missing")

    predictions, groups, sample_summary, site_summary = freeze_rankings(
        p, index, candidates, transformer_b_seed_sd
    )
    atomic_csv(predictions, p["feature_predictions"], compression="gzip")
    atomic_csv(groups, p["group_predictions"], compression="gzip")
    atomic_csv(sample_summary, p["sample_summary"])
    atomic_csv(site_summary, p["site_summary"])
    make_figure(p, predictions, sample_summary)

    duplicate_sample_group = predictions.duplicated(["case_feature_id", "sample_id"]).any()
    identity_like = [
        column for column in predictions.columns
        if any(token in column.lower() for token in ["smiles", "inchikey", "formula", "compound_name", "annotation"])
    ]
    qc = pd.DataFrame([
        {"check": "notebook23_ready", "passed": nb23["status"] == "READY_FOR_NOTEBOOK_24", "detail": nb23["status"]},
        {"check": "official_feature_count", "passed": (n_features == EXPECTED_FEATURES) if official else n_features > 0, "detail": n_features},
        {"check": "unique_sample_feature_ids", "passed": not predictions["sample_feature_id"].duplicated().any(), "detail": predictions["sample_feature_id"].nunique()},
        {"check": "identity_blind_columns", "passed": not identity_like, "detail": str(identity_like)},
        {"check": "all_eight_candidate_pipelines", "passed": set(candidates) == set(PIPELINES), "detail": str(sorted(candidates))},
        {"check": "all_predictions_finite", "passed": all(np.isfinite(array).all() for array in candidates.values()), "detail": f"{len(candidates)} pipeline means"},
        {"check": "frozen_thresholds_only", "passed": True, "detail": "Notebook 19 calibration and rejection JSON; no Portugal recalibration"},
        {"check": "group_has_at_most_one_feature_per_sample", "passed": not duplicate_sample_group, "detail": "10 ppm / 15 s deterministic grouping"},
        {"check": "low_depth_sample_flagged", "passed": bool(predictions["low_ms2_depth_sample"].any()) if official else True, "detail": str(sorted(predictions.loc[predictions["low_ms2_depth_sample"], "sample_id"].unique()))},
        {"check": "blank_limitation_carried_forward", "passed": "No explicit blank/QC" in nb23["blank_limitation"], "detail": nb23["blank_limitation"]},
        {"check": "primary_ranking_nonempty", "passed": predictions["primary_hazard_score"].notna().any(), "detail": int(predictions["primary_hazard_score"].notna().sum())},
    ])
    atomic_csv(qc, p["qc"])
    all_pass = bool(qc["passed"].all())
    outputs = [
        p["feature_predictions"],
        p["group_predictions"],
        p["sample_summary"],
        p["site_summary"],
        p["figure"],
        p["qc"],
    ]
    freeze = {
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "status": "IDENTITY_BLIND_RANKING_FROZEN" if all_pass else "BLOCKED_REVIEW_QC",
        "identity_blind": True,
        "identity_fields_read": [],
        "ranking_formula": {
            "formula_id": CANONICAL_HAZARD_FORMULA,
            "endpoint_components": "smoothed empirical hazard percentiles anchored to Development OOF selected-model predictions",
            "toxicity_component": "maximum of inverse EC50, LC50 and NOEC Development-anchored percentiles",
            "joint_pbt_hazard_score": "geometric mean of P, B and toxicity components",
            "primary_hazard_score": "joint score only when all five frozen endpoint rejection rules retain the feature",
            "occurrence_assisted_score": "square root of group median joint hazard score times within-mode sample detection frequency",
            "identity_or_label_based_tuning": False,
        },
        "n_features": n_features,
        "n_primary_ranked": int(predictions["primary_hazard_score"].notna().sum()),
        "n_case_feature_groups": len(groups),
        "low_ms2_depth_samples": sorted(predictions.loc[predictions["low_ms2_depth_sample"], "sample_id"].unique()),
        "blank_limitation": nb23["blank_limitation"],
        "output_hashes": {path.name: sha256_file(path) for path in outputs},
    }
    atomic_json(p["freeze"], freeze)
    manifest = {
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "status": "READY_FOR_NOTEBOOK_25_IDENTITY_REVEAL" if all_pass else "BLOCKED_REVIEW_QC",
        "scientific_scope": "identity-blind frozen-model hazard prioritization of Portugal wastewater DDA-MS/MS features",
        "device": str(device),
        "candidate_pipelines": PIPELINES,
        "selected_models": {row["task"]: row["selected_pipeline"] for row in read_json(p["registry"])["models"]},
        "external_recalibration": False,
        "dreams_p_probability_rule": "each outer-fold P probability is transformed by its frozen Development-only DreaMS Platt calibrator before ensemble averaging",
        "prediction_cache_schema": CACHE_SCHEMA_VERSION,
        "identity_access": False,
        "qc_all_required_pass": all_pass,
        "preparation_manifest_sha256": sha256_file(p["prepare_manifest"]),
        "embedding_manifest_sha256": sha256_file(p["embed_manifest"]),
        "identity_blind_freeze_sha256": sha256_file(p["freeze"]),
        "candidate_artifact_hashes": {
            pipeline_path(p, pipeline).name: sha256_file(pipeline_path(p, pipeline))
            for pipeline in PIPELINES
        },
        "output_hashes": {path.name: sha256_file(path) for path in outputs},
        "next_step": "Notebook 25: post-freeze identity matching, known-pollutant recovery, and baseline enrichment evaluation",
    }
    atomic_json(p["manifest"], manifest)
    print(qc.to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "embed", "predict"], required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    root = locate_root(args.root)
    if args.stage == "prepare":
        run_prepare(root, tag=args.tag, limit=args.limit)
    elif args.stage == "embed":
        run_embed(root, tag=args.tag, batch_size=args.batch_size)
    else:
        run_predict(root, tag=args.tag, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
