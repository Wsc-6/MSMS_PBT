"""Prepare, embed, and validate frozen Development spectra with DreaMS.

Run the three stages with separate interpreters::

    env_ms      -m pbt_msms.dreams_batch --stage prepare
    env_dreams  -m pbt_msms.dreams_batch --stage embed
    env_ms      -m pbt_msms.dreams_batch --stage validate

ChemVista spectrum files are deliberately not referenced by this module.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import locale
import os
import pickle
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pbt_msms.dreams_mvp import (
    DREAMS_ARCHIVE,
    DREAMS_COMMIT,
    DREAMS_SOURCE,
    PROJECT_ROOT,
    PUBLIC_MANIFEST,
    PUBLIC_STREAM,
    atomic_json,
    checkpoint_hashes,
    configure_writable_runtime_cache,
    extract_peaks,
    package_versions,
    sha256_array,
    sha256_file,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pbt_msms_v1" / "dreams_v1" / "batch_embeddings_v1"
UPSTREAM_AUDIT = (
    PROJECT_ROOT / "outputs" / "pbt_msms_v1" / "spectrum_audit_v1" / "spectrum_audit_manifest.json"
)
UPSTREAM_QC = (
    PROJECT_ROOT / "outputs" / "pbt_msms_v1" / "spectrum_audit_v1" / "spectrum_audit_qc_checks.csv"
)
MVP_DIR = PROJECT_ROOT / "outputs" / "pbt_msms_v1" / "dreams_v1" / "mvp_5_spectra"
INPUT_MGF = OUTPUT_DIR / "development_dreams_input.mgf"
INPUT_HDF5 = OUTPUT_DIR / "development_dreams_input.hdf5"
INPUT_INDEX = OUTPUT_DIR / "development_dreams_input_index.csv.gz"
PREP_FAILURES = OUTPUT_DIR / "development_dreams_preparation_failures.csv"
PREP_MANIFEST = OUTPUT_DIR / "dreams_batch_preparation_manifest.json"
EMBEDDINGS = OUTPUT_DIR / "development_dreams_embeddings.npy"
EMBED_MANIFEST = OUTPUT_DIR / "dreams_batch_embedding_stage_manifest.json"
PIP_FREEZE = OUTPUT_DIR / "env_dreams_pip_freeze.txt"
QC_PATH = OUTPUT_DIR / "dreams_batch_qc_checks.csv"
RUN_MANIFEST = OUTPUT_DIR / "dreams_batch_run_manifest.json"
EXPECTED_SPECTRA = 58_687
EXPECTED_COMPOUNDS = 1_230
EMBEDDING_DIM = 1_024
PUBLIC_LIBRARIES = {"GNPS", "MassBank", "MoNA"}
ALLOWED_EVALUATION_ROLES = {"none", "transfer"}
SEED = 42


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.GzipFile(temporary, mode="wb", compresslevel=6, mtime=0) as compressed:
        frame.to_csv(compressed, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, path)


def atomic_npy(array: np.ndarray, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
    os.replace(temporary, path)


def normalize_ion_mode(value: Any) -> str:
    text = str(value).strip().lower()
    if text.startswith("pos") or text == "+":
        return "Positive"
    if text.startswith("neg") or text == "-":
        return "Negative"
    raise ValueError(f"Unsupported ion mode: {value}")


def infer_charge(adduct: Any, ion_mode: Any) -> tuple[int, str]:
    adduct_text = str(adduct).strip()
    match = re.search(r"\](\d*)([+-])$", adduct_text)
    if match:
        magnitude = int(match.group(1)) if match.group(1) else 1
        sign = 1 if match.group(2) == "+" else -1
        charge = sign * magnitude
        method = "parsed_from_adduct"
    else:
        charge = 1 if normalize_ion_mode(ion_mode) == "Positive" else -1
        method = "fallback_from_ion_mode"
    expected_sign = 1 if normalize_ion_mode(ion_mode) == "Positive" else -1
    if int(np.sign(charge)) != expected_sign:
        raise ValueError(f"Adduct/ion-mode charge conflict: adduct={adduct_text}, ion_mode={ion_mode}")
    return charge, method


def dreams_format_a_reason(peaks: np.ndarray, precursor_mz: float, charge: int) -> str:
    """Mirror the first-failure semantics of DreaMS DataFormatA.val_spec."""
    if precursor_mz > 1000.0:
        return "precursor_mz_gt_1000"
    if charge < 1 or charge > 1:
        return "charge_not_plus_one"
    if len(peaks) < 3 or len(peaks) > 128:
        return "peak_count_outside_3_128"
    if float(np.max(peaks[:, 0])) > 1000.0:
        return "fragment_mz_gt_1000"
    relative = peaks[:, 1] / float(np.max(peaks[:, 1]))
    if float(np.max(relative) / np.min(relative)) < 20.0:
        return "intensity_amplitude_lt_20"
    if int(np.sum(relative > 0.1)) < 3:
        return "fewer_than_3_peaks_above_10pct"
    return "PASS"


def eligible_manifest() -> tuple[pd.DataFrame, dict[str, Any]]:
    upstream = read_json(UPSTREAM_AUDIT)
    upstream_qc = pd.read_csv(UPSTREAM_QC)
    if upstream_qc["status"].astype(str).eq("FAIL").any():
        raise AssertionError("Upstream spectrum audit contains FAIL")
    manifest = pd.read_csv(PUBLIC_MANIFEST)
    eligible = manifest.loc[manifest["primary_train_eligible_20ppm"].fillna(False).astype(bool)].copy()
    eligible = eligible.sort_values(["stream_offset", "spectrum_id"]).reset_index(drop=True)
    if len(eligible) != int(upstream["counts"]["primary_train_20ppm_spectra"]):
        raise AssertionError("Eligible spectrum count differs from frozen upstream audit")
    if eligible["inchikey_2d"].nunique() != int(upstream["counts"]["primary_train_20ppm_compounds"]):
        raise AssertionError("Eligible compound count differs from frozen upstream audit")
    if not eligible["supervised_compound_role"].astype(str).eq("development").all():
        raise AssertionError("Non-Development compound entered DreaMS preparation")
    roles = set(eligible["chemvista_evaluation_role"].astype(str).unique())
    if not roles.issubset(ALLOWED_EVALUATION_ROLES):
        raise AssertionError(f"Forbidden ChemVista evaluation role in Development: {roles}")
    return eligible, upstream


def mgf_block(
    feature_row: int,
    row: pd.Series,
    peaks: np.ndarray,
    charge: int,
    source_library: str,
) -> str:
    charge_text = f"{abs(charge)}{'+' if charge > 0 else '-'}"
    lines = [
        "BEGIN IONS",
        f"TITLE={row['spectrum_id']}",
        f"FEATURE_ID={feature_row + 1}",
        "MSLEVEL=2",
        f"PEPMASS={float(row['precursor_mz']):.8f}",
        f"CHARGE={charge_text}",
        f"IONMODE={normalize_ion_mode(row['ion_mode'])}",
    ]
    if pd.notna(row["adduct"]):
        lines.append(f"ADDUCT={row['adduct']}")
    collision_energy = pd.to_numeric(row["collision_energy_value"], errors="coerce")
    if np.isfinite(collision_energy) and float(collision_energy) >= 0:
        lines.append(f"COLLISION_ENERGY={float(collision_energy):.8g}")
    lines.append(f"SOURCE_LIBRARY={source_library}")
    lines.extend(f"{mz:.8f} {intensity:.8f}" for mz, intensity in peaks)
    lines.extend(["END IONS", ""])
    return "\n".join(lines) + "\n"


def run_prepare() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    eligible, upstream = eligible_manifest()
    temporary_mgf = INPUT_MGF.with_suffix(INPUT_MGF.suffix + ".tmp")
    index_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()

    with PUBLIC_STREAM.open("rb") as stream, temporary_mgf.open(
        "w", encoding="utf-8", newline="\n"
    ) as mgf:
        for _, row in tqdm(eligible.iterrows(), total=len(eligible), desc="导出DreaMS Development MGF"):
            spectrum_id = str(row["spectrum_id"])
            try:
                stream.seek(int(row["stream_offset"]))
                actual_key, record = pickle.load(stream)
                if str(actual_key) != str(row["inchikey_2d"]):
                    raise ValueError("stream IK2D mismatch")
                if str(record.get("inchikey_2d")) != str(row["inchikey_2d"]):
                    raise ValueError("record IK2D mismatch")
                source_library = str(record.get("source_library"))
                if source_library not in PUBLIC_LIBRARIES:
                    raise ValueError(f"non-public source library: {source_library}")
                record_precursor = float(record.get("precursor_mz", np.nan))
                manifest_precursor = float(row["precursor_mz"])
                if not np.isfinite(record_precursor) or abs(record_precursor - manifest_precursor) > 1e-6:
                    raise ValueError(
                        f"precursor mismatch: manifest={manifest_precursor}, record={record_precursor}"
                    )
                record_ion_mode = normalize_ion_mode(record.get("ionmode"))
                manifest_ion_mode = normalize_ion_mode(row["ion_mode"])
                if record_ion_mode != manifest_ion_mode:
                    raise ValueError("ion-mode mismatch")

                peaks = extract_peaks(record)
                record_peak_n = int(record.get("cleaned_peak_n", -1))
                if len(peaks) != record_peak_n:
                    raise ValueError(
                        f"record/exported peak mismatch: record={record_peak_n}, exported={len(peaks)}"
                    )
                charge, charge_method = infer_charge(row["adduct"], row["ion_mode"])
                format_reason = dreams_format_a_reason(peaks, manifest_precursor, charge)
                feature_row = len(index_rows)
                mgf.write(mgf_block(feature_row, row, peaks, charge, source_library))

                index_rows.append(
                    {
                        "feature_row": feature_row,
                        "spectrum_id": spectrum_id,
                        "inchikey_2d": str(row["inchikey_2d"]),
                        "stream_offset": int(row["stream_offset"]),
                        "cv_group": str(row["cv_group"]),
                        "cv_fold": int(row["cv_fold"]),
                        "source_library": source_library,
                        "ion_mode": manifest_ion_mode,
                        "adduct": str(row["adduct"]),
                        "charge": int(charge),
                        "charge_method": charge_method,
                        "precursor_mz": manifest_precursor,
                        "collision_energy_value": float(row["collision_energy_value"]),
                        "manifest_cleaned_peak_n": int(row["cleaned_peak_n"]),
                        "record_cleaned_peak_n": record_peak_n,
                        "exported_peak_n": int(len(peaks)),
                        "dreams_retained_peak_n": int(min(len(peaks), 100)),
                        "peak_count_manifest_record_discrepancy": bool(
                            int(row["cleaned_peak_n"]) != record_peak_n
                        ),
                        "max_fragment_mz": float(np.max(peaks[:, 0])),
                        "dreams_format_a_domain_pass": format_reason == "PASS",
                        "dreams_format_a_first_failure": format_reason,
                        "identity_ppm_error": float(row["identity_ppm_error"]),
                        "spectrum_hash_common": str(row["spectrum_hash_common"]),
                        "peak_hash_common": str(row["peak_hash_common"]),
                        "chemvista_evaluation_role": str(row["chemvista_evaluation_role"]),
                        "supervised_compound_role": str(row["supervised_compound_role"]),
                        "primary_train_eligible_20ppm": bool(row["primary_train_eligible_20ppm"]),
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "spectrum_id": spectrum_id,
                        "inchikey_2d": str(row.get("inchikey_2d", "")),
                        "stream_offset": row.get("stream_offset", np.nan),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )

    os.replace(temporary_mgf, INPUT_MGF)
    index = pd.DataFrame(index_rows)
    failures_frame = pd.DataFrame(
        failures, columns=["spectrum_id", "inchikey_2d", "stream_offset", "reason"]
    )
    atomic_csv_gzip(index, INPUT_INDEX)
    failures_frame.to_csv(PREP_FAILURES, index=False, encoding="utf-8", lineterminator="\n")
    status = "PASS" if len(index) == len(eligible) and failures_frame.empty else "FAIL"
    preparation_manifest = {
        "run_version": "PBT_MSMS_DREAMS_BATCH_PREPARATION_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "external_test_unlocked": False,
        "source_spectrum_files": [str(PUBLIC_STREAM)],
        "chemvista_spectrum_files_read": [],
        "n_upstream_eligible_spectra": int(len(eligible)),
        "n_exported_spectra": int(len(index)),
        "n_exported_compounds": int(index["inchikey_2d"].nunique()) if len(index) else 0,
        "n_failures": int(len(failures_frame)),
        "n_manifest_record_peak_count_discrepancies": int(
            index["peak_count_manifest_record_discrepancy"].sum()
        ) if len(index) else 0,
        "n_dreams_format_a_domain_pass": int(index["dreams_format_a_domain_pass"].sum()) if len(index) else 0,
        "format_a_failure_counts": (
            index["dreams_format_a_first_failure"].value_counts(dropna=False).to_dict()
            if len(index)
            else {}
        ),
        "source_library_counts": index["source_library"].value_counts().to_dict() if len(index) else {},
        "ion_mode_counts": index["ion_mode"].value_counts().to_dict() if len(index) else {},
        "evaluation_role_counts": (
            index["chemvista_evaluation_role"].value_counts().to_dict() if len(index) else {}
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "input_mgf": {
            "path": str(INPUT_MGF),
            "size_bytes": INPUT_MGF.stat().st_size,
            "sha256": sha256_file(INPUT_MGF),
        },
        "input_index": {
            "path": str(INPUT_INDEX),
            "size_bytes": INPUT_INDEX.stat().st_size,
            "sha256": sha256_file(INPUT_INDEX),
        },
        "upstream_audit_sha256": sha256_file(UPSTREAM_AUDIT),
        "upstream_public_stream_sha256": upstream["public_stream"]["sha256"],
    }
    atomic_json(PREP_MANIFEST, preparation_manifest)
    print(json.dumps(preparation_manifest, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise AssertionError("DreaMS batch preparation did not cover every frozen Development spectrum")


def environment_freeze() -> str:
    freeze_bytes = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True
    ).stdout
    return freeze_bytes.decode(locale.getpreferredencoding(False), errors="replace")


def run_embed(batch_size: int) -> None:
    configure_writable_runtime_cache()
    preparation = read_json(PREP_MANIFEST)
    if preparation.get("status") != "PASS":
        raise AssertionError("DreaMS preparation manifest is not PASS")
    if sha256_file(INPUT_MGF) != preparation["input_mgf"]["sha256"]:
        raise AssertionError("Prepared MGF hash changed before DreaMS extraction")
    if sha256_file(INPUT_INDEX) != preparation["input_index"]["sha256"]:
        raise AssertionError("Prepared input index hash changed before DreaMS extraction")
    index = pd.read_csv(INPUT_INDEX)
    if len(index) != EXPECTED_SPECTRA:
        raise AssertionError("Prepared DreaMS index has unexpected row count")

    import torch
    import h5py
    from dreams.api import dreams_embeddings

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen DreaMS batch extraction")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    embeddings_raw = dreams_embeddings(
        INPUT_MGF,
        batch_size=batch_size,
        progress_bar=True,
        store_embs=False,
    )
    elapsed = time.perf_counter() - started
    embeddings = np.asarray(embeddings_raw, dtype=np.float32)
    if embeddings.shape != (len(index), EMBEDDING_DIM):
        raise AssertionError(f"Unexpected DreaMS batch embedding shape: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise AssertionError("DreaMS batch embeddings contain non-finite values")
    atomic_npy(embeddings, EMBEDDINGS)
    PIP_FREEZE.write_text(environment_freeze(), encoding="utf-8", newline="\n")
    with h5py.File(INPUT_HDF5, "r") as handle:
        hdf5_rows = int(len(handle["spectrum"]))

    stage_manifest = {
        "run_version": "PBT_MSMS_DREAMS_BATCH_EMBEDDING_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "external_test_unlocked": False,
        "batch_size": int(batch_size),
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "embedding_array_sha256": sha256_array(embeddings),
        "embedding_file": {
            "path": str(EMBEDDINGS),
            "size_bytes": EMBEDDINGS.stat().st_size,
            "sha256": sha256_file(EMBEDDINGS),
        },
        "generated_hdf5": {
            "path": str(INPUT_HDF5),
            "exists": INPUT_HDF5.exists(),
            "size_bytes": INPUT_HDF5.stat().st_size if INPUT_HDF5.exists() else None,
            "sha256": sha256_file(INPUT_HDF5) if INPUT_HDF5.exists() else None,
            "row_count": hdf5_rows,
        },
        "elapsed_seconds": elapsed,
        "spectra_per_second": len(index) / elapsed,
        "python": sys.version,
        "python_executable": sys.executable,
        "package_versions": package_versions(),
        "torch_cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "dreams_source_commit": DREAMS_COMMIT,
        "checkpoints": checkpoint_hashes(),
        "prepared_mgf_sha256": preparation["input_mgf"]["sha256"],
        "prepared_index_sha256": preparation["input_index"]["sha256"],
        "pip_freeze_sha256": sha256_file(PIP_FREEZE),
    }
    atomic_json(EMBED_MANIFEST, stage_manifest)
    print(json.dumps(stage_manifest, ensure_ascii=False, indent=2))


def count_mgf_blocks_and_structure_fields(path: Path) -> tuple[int, list[str]]:
    blocks = 0
    forbidden: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            upper = line.upper()
            if upper.rstrip() == "BEGIN IONS":
                blocks += 1
            if upper.startswith(("SMILES=", "INCHI=", "INCHIKEY=", "INCHIAUX=")):
                forbidden.append(f"line {line_number}: {line.strip()[:80]}")
    return blocks, forbidden


def finite_embedding_stats(path: Path, chunk_size: int = 4096) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    finite_count = 0
    total_count = 0
    norms: list[np.ndarray] = []
    for start in range(0, len(array), chunk_size):
        chunk = np.asarray(array[start : start + chunk_size], dtype=np.float32)
        finite_count += int(np.isfinite(chunk).sum())
        total_count += int(chunk.size)
        norms.append(np.linalg.norm(chunk, axis=1))
    all_norms = np.concatenate(norms)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite_count": finite_count,
        "total_count": total_count,
        "finite_fraction": finite_count / total_count,
        "l2_norm_min": float(np.min(all_norms)),
        "l2_norm_median": float(np.median(all_norms)),
        "l2_norm_max": float(np.max(all_norms)),
        "l2_norm_zero_count": int(np.sum(all_norms == 0)),
    }


def mvp_reproducibility(index: pd.DataFrame) -> dict[str, Any]:
    mvp_index_path = MVP_DIR / "development_5_spectra_selection.csv"
    mvp_embeddings_path = MVP_DIR / "development_5_spectra_dreams_embeddings.npy"
    if not mvp_index_path.exists() or not mvp_embeddings_path.exists():
        return {"available": False, "status": "NOT_RUN"}
    mvp_index = pd.read_csv(mvp_index_path)
    mvp_embeddings = np.load(mvp_embeddings_path, allow_pickle=False)
    batch_embeddings = np.load(EMBEDDINGS, mmap_mode="r", allow_pickle=False)
    row_map = index.set_index("spectrum_id")["feature_row"].to_dict()
    missing = [spectrum_id for spectrum_id in mvp_index["spectrum_id"] if spectrum_id not in row_map]
    if missing:
        return {"available": True, "status": "FAIL", "missing_spectrum_ids": missing}
    batch_rows = np.asarray(
        [batch_embeddings[int(row_map[spectrum_id])] for spectrum_id in mvp_index["spectrum_id"]],
        dtype=np.float32,
    )
    difference = np.abs(batch_rows - mvp_embeddings)
    return {
        "available": True,
        "status": "PASS" if float(np.max(difference)) <= 1e-5 else "FAIL",
        "n_spectra": int(len(mvp_index)),
        "max_abs_difference": float(np.max(difference)),
        "mean_abs_difference": float(np.mean(difference)),
    }


def run_validate() -> None:
    preparation = read_json(PREP_MANIFEST)
    embedding_stage = read_json(EMBED_MANIFEST)
    upstream_qc = pd.read_csv(UPSTREAM_QC)
    index = pd.read_csv(INPUT_INDEX)
    failures = pd.read_csv(PREP_FAILURES)
    upstream_manifest = pd.read_csv(PUBLIC_MANIFEST)
    upstream_eligible = upstream_manifest.loc[
        upstream_manifest["primary_train_eligible_20ppm"].fillna(False).astype(bool)
    ]
    stats = finite_embedding_stats(EMBEDDINGS)
    mgf_blocks, forbidden_structure_fields = count_mgf_blocks_and_structure_fields(INPUT_MGF)
    hdf5_rows = int(embedding_stage["generated_hdf5"]["row_count"])
    hdf5_hash_current = sha256_file(INPUT_HDF5)

    exact_spectrum_set = set(index["spectrum_id"]) == set(upstream_eligible["spectrum_id"])
    feature_rows_sequential = np.array_equal(
        pd.to_numeric(index["feature_row"], errors="raise").to_numpy(dtype=int),
        np.arange(len(index), dtype=int),
    )
    mvp_check = mvp_reproducibility(index)
    checkpoints = embedding_stage["checkpoints"]
    checkpoint_complete = all(
        item.get("exists") and item.get("sha256") for item in checkpoints.values()
    )

    checks: list[dict[str, Any]] = []

    def add(check_id: str, description: str, passed: bool, details: str = "") -> None:
        checks.append(
            {
                "check_id": check_id,
                "description": description,
                "status": "PASS" if passed else "FAIL",
                "details": details,
            }
        )

    add("DREAMS001", "upstream spectrum audit contains no FAIL", not upstream_qc["status"].eq("FAIL").any())
    add(
        "DREAMS002",
        "batch input covers all frozen Development spectra and compounds",
        len(index) == EXPECTED_SPECTRA and index["inchikey_2d"].nunique() == EXPECTED_COMPOUNDS,
        f"spectra={len(index)}, compounds={index['inchikey_2d'].nunique()}",
    )
    add("DREAMS003", "batch input spectrum IDs exactly match upstream eligibility", exact_spectrum_set)
    add(
        "DREAMS004",
        "only public libraries are read and no ChemVista spectrum is used",
        set(index["source_library"]).issubset(PUBLIC_LIBRARIES)
        and preparation.get("chemvista_spectrum_files_read") == [],
        str(index["source_library"].value_counts().to_dict()),
    )
    add(
        "DREAMS005",
        "only Development compounds with none/transfer evaluation roles are included",
        index["supervised_compound_role"].eq("development").all()
        and set(index["chemvista_evaluation_role"]).issubset(ALLOWED_EVALUATION_ROLES),
        str(index["chemvista_evaluation_role"].value_counts().to_dict()),
    )
    add(
        "DREAMS006",
        "MGF contains no SMILES InChI or InChIKey fields",
        not forbidden_structure_fields,
        str(forbidden_structure_fields[:3]),
    )
    add("DREAMS007", "preparation has zero failed spectra", failures.empty, f"n_failures={len(failures)}")
    add("DREAMS008", "MGF block count equals input index", mgf_blocks == len(index), f"blocks={mgf_blocks}")
    add(
        "DREAMS009",
        "DreaMS HDF5 row count and hash match the embedding-stage record",
        hdf5_rows == len(index)
        and hdf5_hash_current == embedding_stage["generated_hdf5"]["sha256"],
        f"rows={hdf5_rows}",
    )
    add(
        "DREAMS010",
        "embedding shape equals 58687 x 1024",
        stats["shape"] == [EXPECTED_SPECTRA, EMBEDDING_DIM],
        str(stats["shape"]),
    )
    add(
        "DREAMS011",
        "all embedding values are finite and all row norms are nonzero",
        stats["finite_fraction"] == 1.0 and stats["l2_norm_zero_count"] == 0,
        f"finite_fraction={stats['finite_fraction']}, zero_norm={stats['l2_norm_zero_count']}",
    )
    add(
        "DREAMS012",
        "feature_row is a unique sequential embedding-row key",
        feature_rows_sequential and index["spectrum_id"].is_unique,
    )
    add(
        "DREAMS013",
        "all five frozen structure-cluster folds are represented",
        set(pd.to_numeric(index["cv_fold"], errors="raise").astype(int)) == set(range(5)),
        str(index["cv_fold"].value_counts().sort_index().to_dict()),
    )
    add("DREAMS014", "both official DreaMS checkpoints are present and hashed", checkpoint_complete)
    add(
        "DREAMS015",
        "batch embeddings reproduce the five-spectrum MVP",
        mvp_check.get("status") == "PASS",
        json.dumps(mvp_check, ensure_ascii=False),
    )
    add(
        "DREAMS016",
        "ChemVista external test remains locked",
        preparation.get("external_test_unlocked") is False
        and embedding_stage.get("external_test_unlocked") is False,
    )

    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(QC_PATH, index=False, encoding="utf-8", lineterminator="\n")

    coverage = (
        index.groupby(
            ["source_library", "ion_mode", "chemvista_evaluation_role", "dreams_format_a_domain_pass"],
            dropna=False,
        )
        .agg(n_spectra=("spectrum_id", "size"), n_compounds=("inchikey_2d", "nunique"))
        .reset_index()
    )
    coverage.to_csv(
        OUTPUT_DIR / "dreams_batch_coverage_by_source_mode.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    compound_coverage = (
        index.groupby("inchikey_2d", as_index=False)
        .agg(
            n_spectra=("spectrum_id", "size"),
            cv_fold=("cv_fold", "first"),
            n_sources=("source_library", "nunique"),
            n_ion_modes=("ion_mode", "nunique"),
            n_format_a_domain_spectra=("dreams_format_a_domain_pass", "sum"),
            chemvista_evaluation_role=("chemvista_evaluation_role", "first"),
        )
    )
    compound_coverage.to_csv(
        OUTPUT_DIR / "dreams_batch_compound_coverage.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    discrepancy = index.loc[index["peak_count_manifest_record_discrepancy"]].copy()
    discrepancy.to_csv(
        OUTPUT_DIR / "dreams_batch_peak_count_discrepancies.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    pd.DataFrame([stats]).to_csv(
        OUTPUT_DIR / "dreams_batch_embedding_summary.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    artifact_paths = [
        INPUT_MGF,
        INPUT_HDF5,
        INPUT_INDEX,
        PREP_FAILURES,
        PREP_MANIFEST,
        EMBEDDINGS,
        EMBED_MANIFEST,
        PIP_FREEZE,
        QC_PATH,
        OUTPUT_DIR / "dreams_batch_coverage_by_source_mode.csv",
        OUTPUT_DIR / "dreams_batch_compound_coverage.csv",
        OUTPUT_DIR / "dreams_batch_peak_count_discrepancies.csv",
        OUTPUT_DIR / "dreams_batch_embedding_summary.csv",
    ]
    artifacts = {
        path.name: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in artifact_paths
    }
    run_manifest = {
        "run_version": "PBT_MSMS_DREAMS_BATCH_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if checks_frame["status"].eq("PASS").all() else "FAIL",
        "external_test_unlocked": False,
        "n_spectra": int(len(index)),
        "n_compounds": int(index["inchikey_2d"].nunique()),
        "embedding_shape": stats["shape"],
        "embedding_stats": stats,
        "n_format_a_domain_spectra": int(index["dreams_format_a_domain_pass"].sum()),
        "n_format_a_domain_compounds": int(
            index.loc[index["dreams_format_a_domain_pass"], "inchikey_2d"].nunique()
        ),
        "format_a_failure_counts": index["dreams_format_a_first_failure"].value_counts().to_dict(),
        "n_manifest_record_peak_count_discrepancies": int(
            index["peak_count_manifest_record_discrepancy"].sum()
        ),
        "mvp_reproducibility": mvp_check,
        "dreams_source": {
            "repository": "https://github.com/pluskal-lab/DreaMS",
            "commit": DREAMS_COMMIT,
            "archive_path": str(DREAMS_ARCHIVE),
            "archive_sha256": sha256_file(DREAMS_ARCHIVE),
        },
        "pipeline_code": {
            Path(__file__).name: {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "dreams_mvp.py": {
                "path": str(Path(__file__).resolve().with_name("dreams_mvp.py")),
                "sha256": sha256_file(Path(__file__).resolve().with_name("dreams_mvp.py")),
            },
        },
        "checkpoints": checkpoints,
        "embedding_environment": embedding_stage,
        "artifacts": artifacts,
        "qc_all_pass": bool(checks_frame["status"].eq("PASS").all()),
    }
    atomic_json(RUN_MANIFEST, run_manifest)
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2))
    if not run_manifest["qc_all_pass"]:
        failed = checks_frame.loc[checks_frame["status"].eq("FAIL")]
        raise AssertionError(f"DreaMS batch validation failed:\n{failed.to_string(index=False)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["prepare", "embed", "validate"])
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        run_prepare()
    elif args.stage == "embed":
        run_embed(args.batch_size)
    else:
        run_validate()


if __name__ == "__main__":
    main()
