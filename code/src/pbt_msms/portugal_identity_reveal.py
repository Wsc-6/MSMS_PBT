"""Post-freeze identity reveal for the Portugal wastewater case study.

Notebook 24 freezes identity-blind feature and group rankings.  This module
opens identities only after verifying those frozen artifacts.  It links the
published CEC list to public measured reference spectra and performs a strict
precursor-plus-MS/MS library search.  Unmatched features are treated as
unlabelled, never as true negatives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import hypergeom
from tqdm.auto import tqdm


VERSION = "PBT_MSMS_PORTUGAL_IDENTITY_REVEAL_25_V1"
MAX_MZ = 2000.0
PRECURSOR_PPM = 20.0
FRAGMENT_TOLERANCE_DA = 0.02
MAIN_COSINE = 0.70
MAIN_MATCHED_PEAKS = 6
HIGH_COSINE = 0.85
HIGH_MATCHED_PEAKS = 8
AMBIGUITY_DELTA = 0.05
REFERENCE_MAX_PER_STRATUM = 2
RANDOM_SEED = 42


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(frame: pd.DataFrame, path: Path, *, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    compression: Any = {"method": "gzip", "mtime": 0} if compressed else None
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def paths(root: Path, output_dir: Path | None = None) -> dict[str, Path]:
    case = root / "external_data" / "portugal_wastewater_5830725"
    inputs = case / "06_revealed_evaluation" / "inputs"
    out = output_dir or case / "06_revealed_evaluation" / "25_identity_reveal"
    manifests = case / "manifests"
    spectrum_audit = root / "outputs" / "pbt_msms_v1" / "spectrum_audit_v1"
    return {
        "root": root,
        "case": case,
        "inputs": inputs,
        "out": out,
        "figures": out / "figures",
        "manifests": manifests,
        "manifest_out": out / "manifests" if output_dir else manifests,
        "nb24": case / "04_hazard_ranking_24",
        "freeze24": manifests / "24_identity_blind_freeze.json",
        "run24": manifests / "24_hazard_ranking_run_manifest.json",
        "supplement_manifest": inputs / "25_supplement_extraction_manifest.json",
        "pubchem_manifest": inputs / "25_pubchem_identity_manifest.json",
        "published_rows": inputs / "25_published_cec_rows_table_sm10.csv",
        "reference_map": inputs / "25_published_cec_reference_map.csv",
        "pubchem": inputs / "25_pubchem_identity_resolution.csv",
        "watchlist": inputs / "25_published_watchlist_cecs.csv",
        "public_stream": root / "质谱数据" / "EnvMolBench_Cleaned.pkl",
        "public_index": root / "质谱数据" / "EnvMolBench_Cleaned_Index.pkl",
        "spectrum_audit_manifest": spectrum_audit / "spectrum_audit_manifest.json",
        "feature_predictions": case / "04_hazard_ranking_24" / "24_feature_predictions_identity_blind.csv.gz",
        "case_groups": case / "04_hazard_ranking_24" / "24_case_feature_groups_identity_blind.csv.gz",
        "reference_index": out / "25_public_reference_spectrum_index.csv.gz",
        "reference_mz": out / "25_public_reference_mz.npy",
        "reference_intensity": out / "25_public_reference_intensity.npy",
        "reference_offsets": out / "25_public_reference_offsets.npy",
        "case_mz": out / "25_case_feature_mz.npy",
        "case_intensity": out / "25_case_feature_intensity.npy",
        "case_offsets": out / "25_case_feature_offsets.npy",
        "target_coverage": out / "25_study_target_reference_coverage.csv",
        "prepare_manifest": (out / "manifests" if output_dir else manifests) / "25_identity_reference_preparation_manifest.json",
        "feature_matches": out / "25_feature_library_matches.csv.gz",
        "group_matches": out / "25_case_group_library_matches.csv.gz",
        "match_manifest": (out / "manifests" if output_dir else manifests) / "25_library_match_manifest.json",
        "revealed_priority": out / "25_revealed_priority_table.csv.gz",
        "target_recovery": out / "25_known_target_recovery.csv",
        "enrichment": out / "25_top_fraction_enrichment.csv",
        "site_summary": out / "25_site_known_pollutant_summary.csv",
        "qc": (out / "manifests" if output_dir else manifests) / "25_identity_reveal_qc_checks.csv",
        "run_manifest": (out / "manifests" if output_dir else manifests) / "25_identity_reveal_run_manifest.json",
    }


def software_versions() -> dict[str, str]:
    import matplotlib
    import rdkit
    import scipy

    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "rdkit": rdkit.__version__,
    }


def verify_frozen_inputs(p: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    run24 = json.loads(p["run24"].read_text(encoding="utf-8"))
    freeze24 = json.loads(p["freeze24"].read_text(encoding="utf-8"))
    if run24.get("status") != "READY_FOR_NOTEBOOK_25_IDENTITY_REVEAL":
        raise AssertionError(run24.get("status"))
    if freeze24.get("status") != "IDENTITY_BLIND_RANKING_FROZEN":
        raise AssertionError(freeze24.get("status"))
    if run24.get("identity_access") is not False or freeze24.get("identity_blind") is not True:
        raise AssertionError("Notebook 24 was not identity blind")
    for name, expected in run24["output_hashes"].items():
        candidates = [p["nb24"] / name, p["nb24"] / "figures" / name, p["manifests"] / name]
        candidate = next((value for value in candidates if value.is_file()), None)
        if candidate is None or sha256_file(candidate) != expected:
            raise AssertionError(f"Frozen Notebook 24 artifact changed: {name}")
    return run24, freeze24


def standardized_connectivity_key(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return ""
    fragments = Chem.GetMolFrags(molecule, asMols=True)
    if not fragments:
        return ""
    molecule = max(fragments, key=lambda item: item.GetNumHeavyAtoms())
    Chem.RemoveStereochemistry(molecule)
    key = Chem.MolToInchiKey(molecule)
    return key.split("-")[0] if key else ""


def build_target_table(p: dict[str, Path], max_targets: int | None = None) -> pd.DataFrame:
    reference = pd.read_csv(p["reference_map"], keep_default_na=False)
    pubchem = pd.read_csv(p["pubchem"], keep_default_na=False)
    unique_reference = (
        reference.sort_values(["target_id", "compound_name_library"])
        .groupby("target_id", as_index=False)
        .agg(
            compound_name_published=("compound_name_published", "first"),
            normalized_name=("normalized_name", "first"),
            published_row_n=("published_row_n", "first"),
            published_detection_percent_max=("published_detection_percent_max", "first"),
            main_use=("main_use", "first"),
            compound_class=("compound_class", "first"),
            formula_table_sm3=("formula", "first"),
            cas_table_sm3=("cas", "first"),
            library_resolution_method=("library_resolution_method", "first"),
            library_resolution_n=("library_resolution_n", "max"),
        )
    )
    targets = unique_reference.merge(pubchem, on=[
        "target_id", "compound_name_published", "formula_table_sm3", "cas_table_sm3"
    ], how="left", validate="one_to_one")
    watchlist = pd.read_csv(p["watchlist"], keep_default_na=False)
    if len(watchlist) != 8 or watchlist["target_id"].nunique() != 8:
        raise AssertionError("The article-defined Watch-List CEC set must contain exactly eight targets")
    targets["published_watchlist_cec"] = targets["target_id"].isin(set(watchlist["target_id"]))
    targets["reference_inchikey_2d"] = targets["pubchem_smiles"].map(standardized_connectivity_key)
    targets["pubchem_salt_or_fragment_normalized"] = targets["reference_inchikey_2d"].ne(
        targets["pubchem_inchikey_2d"].astype(str)
    )
    if max_targets is not None:
        targets = targets.head(int(max_targets)).copy()
    return targets


def record_peaks(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    tensor = np.asarray(record["tensor"], dtype=np.float32)
    mask = ~np.asarray(record["padding_mask"], dtype=bool)
    valid = tensor[mask]
    if len(valid) <= 1:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    mz = valid[1:, 0] * MAX_MZ
    intensity = valid[1:, 3]
    order = np.argsort(mz, kind="stable")
    return mz[order].astype(np.float32), intensity[order].astype(np.float32)


def save_ragged(mz_arrays: list[np.ndarray], intensity_arrays: list[np.ndarray], mz_path: Path,
                intensity_path: Path, offsets_path: Path) -> None:
    counts = np.asarray([len(value) for value in mz_arrays], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    mz = np.concatenate(mz_arrays).astype(np.float32) if mz_arrays else np.empty(0, dtype=np.float32)
    intensity = (
        np.concatenate(intensity_arrays).astype(np.float32)
        if intensity_arrays else np.empty(0, dtype=np.float32)
    )
    np.save(mz_path, mz, allow_pickle=False)
    np.save(intensity_path, intensity, allow_pickle=False)
    np.save(offsets_path, offsets, allow_pickle=False)


def prepare_case_peak_store(features: pd.DataFrame, p: dict[str, Path]) -> None:
    if not features["feature_row"].astype(int).is_monotonic_increasing:
        features = features.sort_values("feature_row")
    expected_rows = np.arange(len(features), dtype=int)
    if not np.array_equal(features["feature_row"].to_numpy(dtype=int), expected_rows):
        raise AssertionError("Feature rows are not zero-based and consecutive")
    counts = features["peak_count"].to_numpy(dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    all_mz = np.empty(int(offsets[-1]), dtype=np.float32)
    all_intensity = np.empty(int(offsets[-1]), dtype=np.float32)
    for peak_file, group in tqdm(features.groupby("peak_file", sort=True), desc="Compact case peak store"):
        with np.load(str(peak_file), allow_pickle=False) as archive:
            source_mz = archive["mz"]
            source_intensity = archive["intensity"]
            for row in group.itertuples(index=False):
                target_start, target_end = offsets[int(row.feature_row) : int(row.feature_row) + 2]
                source_start = int(row.peak_offset)
                source_end = source_start + int(row.peak_count)
                all_mz[target_start:target_end] = source_mz[source_start:source_end]
                all_intensity[target_start:target_end] = source_intensity[source_start:source_end]
    np.save(p["case_mz"], all_mz, allow_pickle=False)
    np.save(p["case_intensity"], all_intensity, allow_pickle=False)
    np.save(p["case_offsets"], offsets, allow_pickle=False)


def prepare(root: Path, output_dir: Path | None = None, max_targets: int | None = None) -> dict[str, Any]:
    p = paths(root, output_dir)
    p["out"].mkdir(parents=True, exist_ok=True)
    p["figures"].mkdir(parents=True, exist_ok=True)
    p["manifest_out"].mkdir(parents=True, exist_ok=True)
    run24, freeze24 = verify_frozen_inputs(p)
    supplement_manifest = json.loads(p["supplement_manifest"].read_text(encoding="utf-8"))
    pubchem_manifest = json.loads(p["pubchem_manifest"].read_text(encoding="utf-8"))
    if supplement_manifest["counts"]["published_rows"] != 108:
        raise AssertionError("Published CEC table extraction is incomplete")
    if supplement_manifest["counts"]["published_unique_normalized_names"] != 107:
        raise AssertionError("Unexpected unique published CEC count")
    if pubchem_manifest["counts"]["resolved"] != 107:
        raise AssertionError("PubChem identity bridge is incomplete")
    for name, details in supplement_manifest["outputs"].items():
        candidate = p["inputs"] / name
        if not candidate.is_file() or sha256_file(candidate) != details["sha256"]:
            raise AssertionError(f"Supplement extraction artifact changed: {name}")
    for name, details in pubchem_manifest["outputs"].items():
        candidate = p["inputs"] / name
        if not candidate.is_file() or sha256_file(candidate) != details["sha256"]:
            raise AssertionError(f"PubChem identity artifact changed: {name}")

    targets = build_target_table(p, max_targets=max_targets)
    with p["public_index"].open("rb") as handle:
        public_index: dict[str, list[int]] = pickle.load(handle)
    targets["public_reference_available"] = targets["reference_inchikey_2d"].isin(public_index)
    targets["public_reference_raw_spectrum_n"] = targets["reference_inchikey_2d"].map(
        lambda key: len(public_index.get(key, []))
    )

    candidate_metadata: list[dict[str, Any]] = []
    candidate_mz: list[np.ndarray] = []
    candidate_intensity: list[np.ndarray] = []
    with p["public_stream"].open("rb") as stream:
        for target in tqdm(targets.itertuples(index=False), total=len(targets), desc="Load target spectra"):
            key = str(target.reference_inchikey_2d)
            for offset in public_index.get(key, []):
                stream.seek(int(offset))
                actual_key, record = pickle.load(stream)
                if str(actual_key) != key or str(record.get("inchikey_2d", "")) != key:
                    raise AssertionError(f"Public spectrum index mismatch at {offset}")
                ppm_error = float(record.get("identity_ppm_error", np.nan))
                ion_mode = str(record.get("ionmode", ""))
                cleaned_peak_n = int(record.get("cleaned_peak_n", 0))
                if not np.isfinite(ppm_error) or ppm_error > PRECURSOR_PPM:
                    continue
                if ion_mode not in {"Positive", "Negative"} or cleaned_peak_n < MAIN_MATCHED_PEAKS:
                    continue
                mz, intensity = record_peaks(record)
                if len(mz) < MAIN_MATCHED_PEAKS:
                    continue
                raw_index = len(candidate_mz)
                candidate_mz.append(mz)
                candidate_intensity.append(intensity)
                candidate_metadata.append(
                    {
                        "raw_reference_index": raw_index,
                        "target_id": target.target_id,
                        "compound_name_published": target.compound_name_published,
                        "reference_inchikey_2d": key,
                        "source_library": str(record.get("source_library", "")),
                        "source_record_ordinal": int(record.get("source_record_ordinal", -1)),
                        "original_spectrum_id": str(record.get("original_spectrum_id", "")),
                        "instrument": str(record.get("instrument", "")),
                        "ion_mode": "Pos" if ion_mode == "Positive" else "Neg",
                        "adduct": str(record.get("adduct", "")),
                        "collision_energy": record.get("ce_real", np.nan),
                        "collision_energy_bucket": record.get("ce_bucket", np.nan),
                        "reference_precursor_mz": float(record["precursor_mz"]),
                        "reference_identity_ppm_error": ppm_error,
                        "reference_peak_n": int(len(mz)),
                        "spectrum_hash": str(record.get("spectrum_hash", "")),
                        "stream_offset": int(offset),
                    }
                )
    raw = pd.DataFrame(candidate_metadata)
    if raw.empty:
        raise AssertionError("No public reference spectra survived the frozen quality rules")
    raw["ce_stratum"] = raw["collision_energy_bucket"].fillna(-1).astype(str)
    raw = raw.sort_values(
        ["target_id", "ion_mode", "adduct", "ce_stratum", "source_library",
         "reference_identity_ppm_error", "reference_peak_n", "stream_offset"],
        ascending=[True, True, True, True, True, True, False, True],
    )
    raw = raw.drop_duplicates(["target_id", "ion_mode", "spectrum_hash"], keep="first")
    selected = (
        raw.groupby(["target_id", "ion_mode", "adduct", "ce_stratum", "source_library"],
                    sort=False, dropna=False)
        .head(REFERENCE_MAX_PER_STRATUM)
        .copy()
        .reset_index(drop=True)
    )
    selected["reference_row"] = np.arange(len(selected), dtype=int)
    selected_mz = [candidate_mz[int(index)] for index in selected["raw_reference_index"]]
    selected_intensity = [candidate_intensity[int(index)] for index in selected["raw_reference_index"]]
    save_ragged(selected_mz, selected_intensity, p["reference_mz"], p["reference_intensity"], p["reference_offsets"])
    selected = selected.drop(columns=["raw_reference_index", "ce_stratum"])
    write_csv(selected, p["reference_index"], compressed=True)

    target_reference_counts = selected.groupby("target_id").size()
    targets["public_reference_selected_spectrum_n"] = targets["target_id"].map(target_reference_counts).fillna(0).astype(int)
    targets["spectral_matching_eligible"] = targets["public_reference_selected_spectrum_n"].gt(0)
    write_csv(targets, p["target_coverage"])

    features = pd.read_csv(p["feature_predictions"])
    if len(features) != int(freeze24["n_features"]):
        raise AssertionError("Frozen feature count changed")
    prepare_case_peak_store(features, p)

    artifacts = [
        p["reference_index"], p["reference_mz"], p["reference_intensity"], p["reference_offsets"],
        p["case_mz"], p["case_intensity"], p["case_offsets"], p["target_coverage"],
    ]
    manifest = {
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "stage": "prepare",
        "status": "PASS",
        "identity_opened_after_frozen_ranking": True,
        "notebook24_freeze_sha256": sha256_file(p["freeze24"]),
        "notebook24_feature_predictions_sha256": run24["output_hashes"][p["feature_predictions"].name],
        "supplement_extraction_manifest_sha256": sha256_file(p["supplement_manifest"]),
        "pubchem_manifest_sha256": sha256_file(p["pubchem_manifest"]),
        "published_watchlist_cecs_sha256": sha256_file(p["watchlist"]),
        "counts": {
            "published_rows": int(pd.read_csv(p["published_rows"]).shape[0]),
            "published_unique_targets": int(targets["target_id"].nunique()),
            "published_watchlist_targets": int(targets["published_watchlist_cec"].sum()),
            "public_reference_covered_targets": int(targets["public_reference_available"].sum()),
            "spectral_matching_eligible_targets": int(targets["spectral_matching_eligible"].sum()),
            "raw_reference_spectra": int(len(raw)),
            "selected_reference_spectra": int(len(selected)),
            "case_features": int(len(features)),
            "case_peaks": int(np.load(p["case_offsets"], mmap_mode="r")[-1]),
        },
        "matching_rules": {
            "reference_identity_ppm_max": PRECURSOR_PPM,
            "minimum_reference_peaks": MAIN_MATCHED_PEAKS,
            "max_reference_spectra_per_target_mode_adduct_ce_source_stratum": REFERENCE_MAX_PER_STRATUM,
        },
        "artifacts": {path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in artifacts},
        "software": software_versions(),
    }
    write_json(p["prepare_manifest"], manifest)
    return manifest


def modified_cosine_score(
    mz_a: np.ndarray,
    intensity_a: np.ndarray,
    precursor_a: float,
    mz_b: np.ndarray,
    intensity_b: np.ndarray,
    precursor_b: float,
    tolerance: float = FRAGMENT_TOLERANCE_DA,
) -> tuple[float, int]:
    mz_a = np.asarray(mz_a, dtype=float)
    mz_b = np.asarray(mz_b, dtype=float)
    intensity_a = np.sqrt(np.clip(np.asarray(intensity_a, dtype=float), 0, None))
    intensity_b = np.sqrt(np.clip(np.asarray(intensity_b, dtype=float), 0, None))
    norm_a, norm_b = np.linalg.norm(intensity_a), np.linalg.norm(intensity_b)
    if not len(mz_a) or not len(mz_b) or norm_a == 0 or norm_b == 0:
        return 0.0, 0
    intensity_a, intensity_b = intensity_a / norm_a, intensity_b / norm_b
    precursor_shift = float(precursor_b) - float(precursor_a)
    candidates: list[tuple[float, int, int]] = []
    for first, mass_a in enumerate(mz_a):
        for shift in (0.0, precursor_shift):
            target = mass_a + shift
            left = int(np.searchsorted(mz_b, target - tolerance, side="left"))
            right = int(np.searchsorted(mz_b, target + tolerance, side="right"))
            for second in range(left, right):
                candidates.append((float(intensity_a[first] * intensity_b[second]), first, second))
    used_a: set[int] = set()
    used_b: set[int] = set()
    score = 0.0
    matches = 0
    for product, first, second in sorted(candidates, reverse=True):
        if first in used_a or second in used_b:
            continue
        used_a.add(first)
        used_b.add(second)
        score += product
        matches += 1
    return float(min(score, 1.0)), int(matches)


def match(root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    p = paths(root, output_dir)
    verify_frozen_inputs(p)
    preparation = json.loads(p["prepare_manifest"].read_text(encoding="utf-8"))
    if preparation.get("status") != "PASS":
        raise AssertionError("Reference preparation is not complete")
    for name, details in preparation["artifacts"].items():
        path = p["out"] / name
        if not path.is_file() or sha256_file(path) != details["sha256"]:
            raise AssertionError(f"Prepared identity artifact changed: {name}")

    features = pd.read_csv(p["feature_predictions"])
    reference = pd.read_csv(p["reference_index"])
    ref_mz = np.load(p["reference_mz"], mmap_mode="r")
    ref_intensity = np.load(p["reference_intensity"], mmap_mode="r")
    ref_offsets = np.load(p["reference_offsets"], mmap_mode="r")
    case_mz = np.load(p["case_mz"], mmap_mode="r")
    case_intensity = np.load(p["case_intensity"], mmap_mode="r")
    case_offsets = np.load(p["case_offsets"], mmap_mode="r")

    mode_lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for mode in ("Pos", "Neg"):
        eligible = features.loc[
            features["mode"].eq(mode) & features["clean_peak_n"].ge(MAIN_MATCHED_PEAKS),
            ["feature_row", "precursor_mz"],
        ].sort_values(["precursor_mz", "feature_row"])
        mode_lookup[mode] = (
            eligible["precursor_mz"].to_numpy(dtype=float),
            eligible["feature_row"].to_numpy(dtype=int),
        )

    best_pairs: dict[tuple[int, str], dict[str, Any]] = {}
    comparison_n = 0
    for ref in tqdm(reference.itertuples(index=False), total=len(reference), desc="Public library matching"):
        sorted_mass, sorted_rows = mode_lookup[str(ref.ion_mode)]
        tolerance = float(ref.reference_precursor_mz) * PRECURSOR_PPM * 1e-6
        left = int(np.searchsorted(sorted_mass, float(ref.reference_precursor_mz) - tolerance, side="left"))
        right = int(np.searchsorted(sorted_mass, float(ref.reference_precursor_mz) + tolerance, side="right"))
        rstart, rend = ref_offsets[int(ref.reference_row) : int(ref.reference_row) + 2]
        rmz = ref_mz[rstart:rend]
        rint = ref_intensity[rstart:rend]
        for feature_row in sorted_rows[left:right]:
            feature_row = int(feature_row)
            cstart, cend = case_offsets[feature_row : feature_row + 2]
            cmz = case_mz[cstart:cend]
            cint = case_intensity[cstart:cend]
            score, matched_peaks = modified_cosine_score(
                rmz, rint, float(ref.reference_precursor_mz), cmz, cint,
                float(features.iloc[feature_row]["precursor_mz"]),
            )
            comparison_n += 1
            if score < 0.50 or matched_peaks < 4:
                continue
            case_precursor = float(features.iloc[feature_row]["precursor_mz"])
            ppm_error = abs(case_precursor - float(ref.reference_precursor_mz)) / float(ref.reference_precursor_mz) * 1e6
            candidate = {
                "feature_row": feature_row,
                "target_id": str(ref.target_id),
                "compound_name_published": str(ref.compound_name_published),
                "reference_row": int(ref.reference_row),
                "modified_cosine": score,
                "matched_peaks": matched_peaks,
                "precursor_ppm_error": ppm_error,
                "reference_precursor_mz": float(ref.reference_precursor_mz),
                "reference_ion_mode": str(ref.ion_mode),
                "reference_adduct": str(ref.adduct),
                "reference_collision_energy": ref.collision_energy,
                "reference_source_library": str(ref.source_library),
                "reference_original_spectrum_id": str(ref.original_spectrum_id),
                "reference_instrument": str(ref.instrument),
            }
            key = (feature_row, str(ref.target_id))
            incumbent = best_pairs.get(key)
            candidate_order = (score, matched_peaks, -ppm_error)
            incumbent_order = (
                float(incumbent["modified_cosine"]), int(incumbent["matched_peaks"]),
                -float(incumbent["precursor_ppm_error"]),
            ) if incumbent else None
            if incumbent is None or candidate_order > incumbent_order:
                best_pairs[key] = candidate

    pairs = pd.DataFrame(best_pairs.values())
    if pairs.empty:
        pairs = pd.DataFrame(columns=[
            "feature_row", "target_id", "compound_name_published", "reference_row",
            "modified_cosine", "matched_peaks", "precursor_ppm_error",
        ])
    identity_columns = [
        "feature_row", "sample_id", "sample_feature_id", "mode", "site_code", "campaign",
        "case_feature_id", "precursor_mz", "rt_seconds", "low_ms2_depth_sample",
        "primary_hazard_score", "overall_hazard_rank", "sample_abundance_percentile",
    ]
    pairs = pairs.merge(features[identity_columns], on="feature_row", how="left", validate="many_to_one")
    pairs["passes_main_library_rule"] = (
        pairs["modified_cosine"].ge(MAIN_COSINE)
        & pairs["matched_peaks"].ge(MAIN_MATCHED_PEAKS)
        & pairs["precursor_ppm_error"].le(PRECURSOR_PPM)
    )
    pairs["passes_high_library_rule"] = (
        pairs["modified_cosine"].ge(HIGH_COSINE)
        & pairs["matched_peaks"].ge(HIGH_MATCHED_PEAKS)
        & pairs["precursor_ppm_error"].le(PRECURSOR_PPM)
    )
    write_csv(pairs.sort_values(["feature_row", "modified_cosine"], ascending=[True, False]),
              p["feature_matches"], compressed=True)

    passing = pairs.loc[pairs["passes_main_library_rule"]].copy()
    group_target = (
        passing.sort_values(
            ["case_feature_id", "target_id", "modified_cosine", "matched_peaks", "precursor_ppm_error"],
            ascending=[True, True, False, False, True],
        )
        .drop_duplicates(["case_feature_id", "target_id"], keep="first")
    )
    group_rows: list[dict[str, Any]] = []
    for case_feature_id, group in group_target.groupby("case_feature_id", sort=True):
        ordered = group.sort_values(
            ["modified_cosine", "matched_peaks", "precursor_ppm_error", "target_id"],
            ascending=[False, False, True, True],
        )
        best = ordered.iloc[0]
        runner = ordered.iloc[1] if len(ordered) > 1 else None
        runner_score = float(runner["modified_cosine"]) if runner is not None else np.nan
        ambiguous = bool(
            runner is not None
            and runner_score >= MAIN_COSINE
            and float(best["modified_cosine"]) - runner_score <= AMBIGUITY_DELTA
        )
        if ambiguous:
            status = "ambiguous_library_match"
        elif bool(best["passes_high_library_rule"]):
            status = "level_2a_high_confidence_library_match"
        else:
            status = "level_2a_probable_library_match"
        row = best.to_dict()
        row.update(
            {
                "case_feature_id": case_feature_id,
                "library_match_status": status,
                "runner_up_target_id": str(runner["target_id"]) if runner is not None else "",
                "runner_up_compound_name": str(runner["compound_name_published"]) if runner is not None else "",
                "runner_up_modified_cosine": runner_score,
                "best_runner_up_score_delta": (
                    float(best["modified_cosine"]) - runner_score if runner is not None else np.nan
                ),
                "candidate_identity_n": int(len(ordered)),
                "is_nonambiguous_published_cec_match": not ambiguous,
            }
        )
        group_rows.append(row)
    group_matches = pd.DataFrame(group_rows)
    write_csv(group_matches, p["group_matches"], compressed=True)

    manifest = {
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "stage": "match",
        "status": "PASS",
        "identity_claim": "Schymanski level 2a public-library match; no retention-time standard, so never Level 1",
        "rules": {
            "precursor_ppm_max": PRECURSOR_PPM,
            "fragment_tolerance_da": FRAGMENT_TOLERANCE_DA,
            "main_modified_cosine_min": MAIN_COSINE,
            "main_matched_peaks_min": MAIN_MATCHED_PEAKS,
            "high_modified_cosine_min": HIGH_COSINE,
            "high_matched_peaks_min": HIGH_MATCHED_PEAKS,
            "ambiguity_best_runner_up_delta_max": AMBIGUITY_DELTA,
        },
        "counts": {
            "reference_spectra": int(len(reference)),
            "precursor_window_comparisons": int(comparison_n),
            "feature_target_pairs_saved": int(len(pairs)),
            "feature_target_pairs_passing_main": int(pairs["passes_main_library_rule"].sum()),
            "case_groups_with_any_main_match": int(group_matches["case_feature_id"].nunique()) if len(group_matches) else 0,
            "case_groups_nonambiguous": int(group_matches["is_nonambiguous_published_cec_match"].sum()) if len(group_matches) else 0,
            "unique_published_targets_recovered": int(
                group_matches.loc[group_matches["is_nonambiguous_published_cec_match"], "target_id"].nunique()
            ) if len(group_matches) else 0,
        },
        "artifacts": {
            p["feature_matches"].name: {"sha256": sha256_file(p["feature_matches"]), "rows": int(len(pairs))},
            p["group_matches"].name: {"sha256": sha256_file(p["group_matches"]), "rows": int(len(group_matches))},
        },
        "software": software_versions(),
    }
    write_json(p["match_manifest"], manifest)
    return manifest


def add_qc(checks: list[dict[str, Any]], check: str, passed: bool, detail: Any) -> None:
    checks.append({"check": check, "passed": bool(passed), "detail": str(detail)})


def top_enrichment_table(universe: pd.DataFrame, winners: pd.DataFrame) -> pd.DataFrame:
    methods = {
        "Frozen PBT hazard": "median_primary_hazard_score",
        "Occurrence-assisted hazard": "occurrence_assisted_score",
        "Abundance baseline": "median_sample_abundance_percentile",
        "Detection-frequency baseline": "detection_frequency_fraction",
    }
    winner_target = winners.set_index("case_feature_id")["target_id"].to_dict()
    output: list[dict[str, Any]] = []
    scopes = {
        "All published CEC library matches": "known_hit",
        "Published Watch-List CEC matches": "watchlist_hit",
    }
    total_groups = int(len(universe))
    for label_scope, label_column in scopes.items():
        total_hits = int(universe[label_column].sum())
        background = total_hits / total_groups if total_groups else np.nan
        eligible_target_ids = set(
            winners.loc[
                winners["case_feature_id"].isin(universe["case_feature_id"])
                & (winners["published_watchlist_cec"].astype(bool) if label_column == "watchlist_hit" else True),
                "target_id",
            ]
        )
        for method, score_column in methods.items():
            ordered = universe.sort_values([score_column, "case_feature_id"], ascending=[False, True]).reset_index(drop=True)
            for fraction in (0.05, 0.10, 0.20):
                top_n = max(1, int(math.ceil(total_groups * fraction)))
                selected = ordered.head(top_n)
                hits = int(selected[label_column].sum())
                density = hits / top_n
                enrichment = density / background if background and np.isfinite(background) else np.nan
                p_value = float(hypergeom.sf(hits - 1, total_groups, total_hits, top_n)) if total_hits else np.nan
                captured_targets = {
                    winner_target[feature_id]
                    for feature_id in selected.loc[selected[label_column], "case_feature_id"]
                    if feature_id in winner_target and winner_target[feature_id] in eligible_target_ids
                }

                cutoff_score = float(selected.iloc[-1][score_column])
                above = ordered[score_column].gt(cutoff_score)
                tied = ordered[score_column].eq(cutoff_score)
                required_from_tie = top_n - int(above.sum())
                tie_total = int(tied.sum())
                tie_hits = int(ordered.loc[tied, label_column].sum())
                rng = np.random.default_rng(
                    RANDOM_SEED + int(fraction * 1000) + list(methods).index(method)
                    + (10000 if label_column == "watchlist_hit" else 0)
                )
                if tie_total and required_from_tie > 0:
                    draws = rng.hypergeometric(tie_hits, tie_total - tie_hits, required_from_tie, size=20000)
                    fixed_hits = int(ordered.loc[above, label_column].sum())
                    tie_mean = float(fixed_hits + draws.mean())
                    tie_low, tie_high = np.quantile(fixed_hits + draws, [0.025, 0.975])
                else:
                    tie_mean = float(hits)
                    tie_low = tie_high = float(hits)
                output.append(
                    {
                        "label_scope": label_scope,
                        "method": method,
                        "score_column": score_column,
                        "top_fraction": fraction,
                        "universe_group_n": total_groups,
                        "top_group_n": top_n,
                        "known_hit_group_n": hits,
                        "known_hit_density": density,
                        "background_known_hit_prevalence": background,
                        "enrichment_fold": enrichment,
                        "hypergeometric_p_value": p_value,
                        "captured_unique_target_n": len(captured_targets),
                        "recoverable_unique_target_n_in_universe": len(eligible_target_ids),
                        "unique_target_capture_fraction": len(captured_targets) / len(eligible_target_ids) if eligible_target_ids else np.nan,
                        "cutoff_score": cutoff_score,
                        "cutoff_tie_group_n": tie_total,
                        "cutoff_tie_selected_n": required_from_tie,
                        "tie_randomized_expected_hit_n": tie_mean,
                        "tie_randomized_hit_n_ci_low": float(tie_low),
                        "tie_randomized_hit_n_ci_high": float(tie_high),
                    }
                )
    return pd.DataFrame(output)


def plot_results(enrichment: pd.DataFrame, feature_counts: dict[str, int], p: dict[str, Path]) -> list[Path]:
    p["figures"].mkdir(parents=True, exist_ok=True)
    method_colors = {
        "Frozen PBT hazard": "#1f77b4",
        "Occurrence-assisted hazard": "#2ca02c",
        "Abundance baseline": "#ff7f0e",
        "Detection-frequency baseline": "#9467bd",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    capture = enrichment.loc[enrichment["label_scope"].eq("Published Watch-List CEC matches")]
    all_cec = enrichment.loc[enrichment["label_scope"].eq("All published CEC library matches")]
    for method, group in capture.groupby("method", sort=False):
        ordered = group.sort_values("top_fraction")
        axes[0].plot(ordered["top_fraction"] * 100, ordered["unique_target_capture_fraction"],
                     marker="o", label=method, color=method_colors[method])
    for method, group in all_cec.groupby("method", sort=False):
        ordered = group.sort_values("top_fraction")
        axes[1].plot(ordered["top_fraction"] * 100, ordered["enrichment_fold"],
                     marker="o", label=method, color=method_colors[method])
    axes[0].set(xlabel="Top-ranked case groups (%)", ylabel="Recovered published targets",
                title="Published Watch-List target capture", ylim=(0, 1.03))
    axes[1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[1].set(xlabel="Top-ranked case groups (%)", ylabel="Known-hit enrichment (fold)",
                title="All-published-CEC enrichment")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    enrichment_path = p["figures"] / "25_known_pollutant_enrichment.png"
    fig.savefig(enrichment_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    labels = ["Model-scored\nfeatures", "All-endpoint\nretained", "Library-matched\nfeatures"]
    values = [feature_counts["model_scored"], feature_counts["all_endpoint_retained"], feature_counts["library_matched"]]
    fig, axis = plt.subplots(figsize=(7.5, 4.6))
    bars = axis.bar(labels, values, color=["#4c78a8", "#59a14f", "#e15759"])
    axis.set_ylabel("Sample-level MS/MS features")
    axis.set_title("Direct-model deployment coverage versus post-freeze library identification")
    axis.bar_label(bars, fmt="%d", padding=3)
    fig.tight_layout()
    coverage_path = p["figures"] / "25_model_vs_library_coverage.png"
    fig.savefig(coverage_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [enrichment_path, coverage_path]


def evaluate(root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    p = paths(root, output_dir)
    run24, freeze24 = verify_frozen_inputs(p)
    preparation = json.loads(p["prepare_manifest"].read_text(encoding="utf-8"))
    matching = json.loads(p["match_manifest"].read_text(encoding="utf-8"))
    if preparation.get("status") != "PASS" or matching.get("status") != "PASS":
        raise AssertionError("Notebook 25 upstream stage is incomplete")

    targets = pd.read_csv(p["target_coverage"], keep_default_na=False)
    features = pd.read_csv(p["feature_predictions"])
    groups = pd.read_csv(p["case_groups"])
    feature_matches = pd.read_csv(p["feature_matches"])
    group_matches = pd.read_csv(p["group_matches"]) if p["group_matches"].stat().st_size else pd.DataFrame()
    if group_matches.empty:
        winners = pd.DataFrame(columns=[
            "case_feature_id", "target_id", "compound_name_published", "published_watchlist_cec"
        ])
    else:
        winners = group_matches.loc[group_matches["is_nonambiguous_published_cec_match"].astype(bool)].copy()
        winners = winners.merge(
            targets[["target_id", "published_watchlist_cec"]],
            on="target_id", how="left", validate="many_to_one",
        )
        winners["published_watchlist_cec"] = winners["published_watchlist_cec"].fillna(False).astype(bool)

    winner_columns = [
        "case_feature_id", "target_id", "compound_name_published", "library_match_status",
        "modified_cosine", "matched_peaks", "precursor_ppm_error", "reference_source_library",
        "reference_original_spectrum_id", "candidate_identity_n", "runner_up_target_id",
        "runner_up_modified_cosine",
    ]
    revealed = groups.merge(winners[winner_columns] if len(winners) else winners,
                            on="case_feature_id", how="left", validate="one_to_one")
    revealed["known_published_cec_hit"] = revealed["target_id"].notna()
    write_csv(revealed, p["revealed_priority"], compressed=True)

    recovered = (
        winners.groupby(["target_id", "compound_name_published"], as_index=False)
        .agg(
            recovered_case_group_n=("case_feature_id", "nunique"),
            best_modified_cosine=("modified_cosine", "max"),
            best_matched_peaks=("matched_peaks", "max"),
        ) if len(winners) else pd.DataFrame(columns=[
            "target_id", "compound_name_published", "recovered_case_group_n",
            "best_modified_cosine", "best_matched_peaks",
        ])
    )
    target_recovery = targets.merge(recovered, on=["target_id", "compound_name_published"], how="left")
    target_recovery["recovered_case_group_n"] = target_recovery["recovered_case_group_n"].fillna(0).astype(int)
    target_recovery["recovered"] = target_recovery["recovered_case_group_n"].gt(0)
    if len(winners):
        best_group = winners.sort_values(["target_id", "modified_cosine"], ascending=[True, False]).drop_duplicates("target_id")
        best_group = best_group[["target_id", "case_feature_id"]].merge(
            groups[["case_feature_id", "hazard_rank", "occurrence_assisted_rank", "abundance_baseline_rank", "frequency_baseline_rank"]],
            on="case_feature_id", how="left",
        )
        target_recovery = target_recovery.merge(best_group, on="target_id", how="left")
    write_csv(target_recovery, p["target_recovery"])

    common = revealed.loc[
        revealed[["median_primary_hazard_score", "occurrence_assisted_score",
                  "median_sample_abundance_percentile", "detection_frequency_fraction"]]
        .notna().all(axis=1)
    ].copy()
    low_only = (
        features.groupby("case_feature_id")["low_ms2_depth_sample"].all().to_dict()
    )
    common["low_depth_only_group"] = common["case_feature_id"].map(low_only).fillna(False)
    common = common.loc[~common["low_depth_only_group"]].copy()
    common["known_hit"] = common["known_published_cec_hit"].astype(bool)
    watchlist_ids = set(targets.loc[targets["published_watchlist_cec"].astype(bool), "target_id"])
    common["watchlist_hit"] = common["target_id"].isin(watchlist_ids)
    enrichment = top_enrichment_table(common, winners)
    write_csv(enrichment, p["enrichment"])

    if len(winners):
        propagated = features.merge(
            winners[["case_feature_id", "target_id", "compound_name_published"]],
            on="case_feature_id", how="inner", validate="many_to_one",
        )
        site_summary = (
            propagated.groupby(["target_id", "compound_name_published", "mode", "site_code", "campaign"], as_index=False)
            .agg(
                sample_n=("sample_id", "nunique"),
                feature_n=("sample_feature_id", "nunique"),
                median_primary_hazard_score=("primary_hazard_score", "median"),
                median_intensity_area=("intensity_area_mean_detected", "median"),
            )
        )
    else:
        site_summary = pd.DataFrame(columns=[
            "target_id", "compound_name_published", "mode", "site_code", "campaign",
            "sample_n", "feature_n", "median_primary_hazard_score", "median_intensity_area",
        ])
    write_csv(site_summary, p["site_summary"])

    feature_counts = {
        "model_scored": int(len(features)),
        "all_endpoint_retained": int(features["all_endpoints_retained"].astype(bool).sum()),
        "library_matched": int(
            feature_matches.loc[
                feature_matches["passes_main_library_rule"].astype(bool), "feature_row"
            ].nunique()
        ),
    }
    figure_paths = plot_results(enrichment, feature_counts, p)

    checks: list[dict[str, Any]] = []
    add_qc(checks, "notebook24_frozen_and_unchanged", True, run24["status"])
    add_qc(checks, "identity_reveal_after_freeze", True, "Identity bridge and library search occur only in Notebook 25")
    add_qc(checks, "published_table_rows", len(pd.read_csv(p["published_rows"])) == 108, 108)
    add_qc(checks, "published_unique_targets", targets["target_id"].nunique() == 107 or len(targets) < 107, targets["target_id"].nunique())
    add_qc(checks, "published_watchlist_targets", targets["published_watchlist_cec"].astype(bool).sum() == 8, targets["published_watchlist_cec"].astype(bool).sum())
    add_qc(checks, "pubchem_identity_bridge_complete", targets["reference_inchikey_2d"].ne("").all(), targets["reference_inchikey_2d"].ne("").sum())
    add_qc(checks, "public_reference_coverage_nonzero", targets["spectral_matching_eligible"].astype(bool).any(), targets["spectral_matching_eligible"].astype(bool).sum())
    add_qc(checks, "precursor_rule_respected", bool(feature_matches.empty or feature_matches.loc[feature_matches["passes_main_library_rule"].astype(bool), "precursor_ppm_error"].le(PRECURSOR_PPM + 1e-9).all()), PRECURSOR_PPM)
    add_qc(checks, "fragment_rule_respected", bool(feature_matches.empty or feature_matches.loc[feature_matches["passes_main_library_rule"].astype(bool), "matched_peaks"].ge(MAIN_MATCHED_PEAKS).all()), MAIN_MATCHED_PEAKS)
    add_qc(checks, "no_level1_claim", True, "No retention-time standard; all nonambiguous calls remain Level 2a")
    add_qc(checks, "unlabelled_not_negative", True, "Precision/AUROC are not computed against incomplete annotations")
    add_qc(checks, "common_universe_nonempty", len(common) > 0, len(common))
    add_qc(checks, "known_hit_enrichment_computable", len(winners) > 0, len(winners))
    add_qc(checks, "low_depth_sample_carried_forward", "PTWW_Neg_PS-F_C3" in freeze24["low_ms2_depth_samples"], freeze24["low_ms2_depth_samples"])
    qc = pd.DataFrame(checks)
    write_csv(qc, p["qc"])
    all_pass = bool(qc["passed"].all())

    output_artifacts = [
        p["revealed_priority"], p["target_recovery"], p["enrichment"], p["site_summary"],
        p["feature_matches"], p["group_matches"], p["qc"], *figure_paths,
    ]
    run_manifest = {
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "status": "READY_FOR_NOTEBOOK_26_PORTUGAL_CASE_INTERPRETATION" if all_pass else "FAILED_QC",
        "scientific_scope": "post-freeze identity matching and known-pollutant enrichment in Portugal wastewater DDA-MS/MS",
        "identity_opened_only_after_notebook24_freeze": True,
        "qc_all_required_pass": all_pass,
        "counts": {
            "published_cec_rows": int(pd.read_csv(p["published_rows"]).shape[0]),
            "published_unique_targets": int(targets["target_id"].nunique()),
            "published_watchlist_targets": int(targets["published_watchlist_cec"].astype(bool).sum()),
            "public_spectral_library_covered_targets": int(targets["spectral_matching_eligible"].astype(bool).sum()),
            "recovered_unique_targets": int(target_recovery["recovered"].sum()),
            "recovered_watchlist_targets": int(
                target_recovery.loc[target_recovery["published_watchlist_cec"].astype(bool), "recovered"].sum()
            ),
            "nonambiguous_matched_case_groups": int(len(winners)),
            "model_scored_features": feature_counts["model_scored"],
            "all_endpoint_retained_features": feature_counts["all_endpoint_retained"],
            "library_matched_sample_features": feature_counts["library_matched"],
            "common_enrichment_universe_groups": int(len(common)),
        },
        "limitations": [
            "No explicit blank/QC raw files are identifiable in the Zenodo archive; no blank-subtraction claim is made.",
            "No retention-time standards are available; library matches are Schymanski Level 2a, not Level 1.",
            "Published and public-library annotations are incomplete; unmatched features remain unlabelled and are not negatives.",
            "The eight article-defined Watch-List CECs are a regulatory relevance benchmark, not binary PBT ground truth.",
            "The source Supplementary Table SM-10 prints Losartan twice (100% and 96%); 108 rows correspond to 107 unique printed names.",
            "Detection frequency and peak area are not concentrations; results are hazard prioritization, not quantitative risk.",
        ],
        "output_hashes": {path.name: sha256_file(path) for path in output_artifacts},
        "next_step": "Notebook 26: interpret site/campaign patterns, perform sensitivity analyses, and assemble manuscript-ready Portugal case figures",
        "software": software_versions(),
    }
    write_json(p["run_manifest"], run_manifest)
    return run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=["prepare", "match", "evaluate", "all"], default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-targets", type=int)
    args = parser.parse_args()
    if args.stage in {"prepare", "all"}:
        result = prepare(args.root, args.output_dir, max_targets=args.max_targets)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.stage in {"match", "all"}:
        result = match(args.root, args.output_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.stage in {"evaluate", "all"}:
        result = evaluate(args.root, args.output_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
