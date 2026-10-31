"""Post-freeze public measured-library matching for Bangladesh DIA cohorts.

Notebook 33 consumes only the identity-blind cohorts frozen by Notebook 32.
It searches the locally cleaned GNPS, MassBank, and MoNA collection and never
changes the frozen ranking or cohort membership.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # Progress bars are optional for archived/minimal runtimes.
    def tqdm(iterable, *args, **kwargs):
        return iterable

VERSION = "PBT_MSMS_BANGLADESH_LIBRARY_33_V2"
MAX_MZ = 2000.0
PRECURSOR_PPM = 20.0
FRAGMENT_TOLERANCE_DA = 0.02
MAIN_COSINE = 0.70
MAIN_MATCHED_PEAKS = 6
HIGH_COSINE = 0.85
HIGH_MATCHED_PEAKS = 8
AMBIGUITY_DELTA = 0.05
REFERENCE_MAX_PER_STRATUM = 2


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(frame: pd.DataFrame, path: Path, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    compression: Any = {"method": "gzip", "mtime": 0} if compressed else None
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def paths(root: Path) -> dict[str, Path]:
    case = root / "external_data" / "bangladesh_surface_water"
    cohort = case / "06_revealed_evaluation" / "32_identity_reveal_cohort_v2"
    out = case / "06_revealed_evaluation" / "33_public_library_matching_v1"
    public = root / "质谱数据"
    return {
        "root": root,
        "out": out,
        "cohort": cohort,
        "cohort_manifest": cohort / "32_run_manifest.json",
        "cohort_clusters": cohort / "32_frozen_cluster_cohort.csv",
        "cohort_members": cohort / "32_frozen_feature_members.csv.gz",
        "freeze": case / "04_blinded_inputs" / "28_full_cleaned_v2",
        "public_stream": public / "EnvMolBench_Cleaned.pkl",
        "public_metadata": public / "EnvMolBench_Cleaned_Metadata.csv.gz",
        "public_summary": public / "EnvMolBench_Cleaned_Summary.json",
        "case_index": out / "33_case_spectrum_index.csv.gz",
        "case_mz": out / "33_case_mz.npy",
        "case_intensity": out / "33_case_intensity.npy",
        "case_offsets": out / "33_case_offsets.npy",
        "reference_index": out / "33_candidate_reference_index.csv.gz",
        "reference_mz": out / "33_reference_mz.npy",
        "reference_intensity": out / "33_reference_intensity.npy",
        "reference_offsets": out / "33_reference_offsets.npy",
        "prepare_manifest": out / "33_preparation_manifest.json",
        "candidate_pairs": out / "33_feature_identity_candidate_pairs.csv.gz",
        "feature_summary": out / "33_feature_best_library_match.csv.gz",
        "cluster_summary": out / "33_cluster_library_match_summary.csv",
        "without_nonambiguous_match": out / "33_clusters_without_nonambiguous_library_match.csv",
        "legacy_sirius_followup": out / "33_clusters_for_sirius_followup.csv",
        "cohort_audit": out / "33_cohort_match_audit.csv",
        "run_manifest": out / "33_run_manifest.json",
    }


def _software() -> dict[str, str]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def _verify_stage32(p: dict[str, Path]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    manifest = json.loads(p["cohort_manifest"].read_text(encoding="utf-8"))
    if manifest.get("status") != "IDENTITY_REVEAL_COHORT_FROZEN":
        raise RuntimeError("Notebook 32 identity-reveal cohort is not frozen")
    for name, expected in manifest["output_hashes"].items():
        candidate = p["cohort"] / name
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise RuntimeError(f"Frozen Notebook 32 artifact changed: {name}")
    clusters = pd.read_csv(p["cohort_clusters"])
    members = pd.read_csv(p["cohort_members"])
    if len(clusters) != 400 or not clusters["cross_polarity_cluster"].is_unique:
        raise RuntimeError("Notebook 32 must contain 400 unique clusters")
    if clusters.groupby("cohort").size().ne(100).any():
        raise RuntimeError("Notebook 32 cohorts are not four groups of 100")
    if members["spectrum_id"].duplicated().any() or len(members) != manifest["n_feature_members_selected"]:
        raise RuntimeError("Notebook 32 feature-member table is inconsistent")
    return manifest, clusters, members


def _save_ragged(mz_arrays: list[np.ndarray], intensity_arrays: list[np.ndarray], mz_path: Path,
                 intensity_path: Path, offsets_path: Path) -> None:
    counts = np.asarray([len(value) for value in mz_arrays], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    mz = np.concatenate(mz_arrays).astype(np.float32) if mz_arrays else np.empty(0, dtype=np.float32)
    intensity = np.concatenate(intensity_arrays).astype(np.float32) if intensity_arrays else np.empty(0, dtype=np.float32)
    np.save(mz_path, mz, allow_pickle=False)
    np.save(intensity_path, intensity, allow_pickle=False)
    np.save(offsets_path, offsets, allow_pickle=False)


def _record_peaks(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    tensor = np.asarray(record["tensor"], dtype=np.float32)
    mask = ~np.asarray(record["padding_mask"], dtype=bool)
    valid = tensor[mask]
    if len(valid) <= 1:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    mz = valid[1:, 0] * MAX_MZ
    intensity = valid[1:, 3]
    order = np.argsort(mz, kind="stable")
    return mz[order].astype(np.float32), intensity[order].astype(np.float32)


def _within_any_reference_window(reference_mz: np.ndarray, query_mz: np.ndarray) -> np.ndarray:
    """Return whether each reference precursor lies within 20 ppm of any query."""
    if len(reference_mz) == 0 or len(query_mz) == 0:
        return np.zeros(len(reference_mz), dtype=bool)
    query_mz = np.sort(np.asarray(query_mz, dtype=float))
    reference_mz = np.asarray(reference_mz, dtype=float)
    position = np.searchsorted(query_mz, reference_mz, side="left")
    right = np.clip(position, 0, len(query_mz) - 1)
    left = np.clip(position - 1, 0, len(query_mz) - 1)
    difference = np.minimum(np.abs(reference_mz - query_mz[left]), np.abs(reference_mz - query_mz[right]))
    return difference <= reference_mz * PRECURSOR_PPM * 1e-6


def _prepare_case_spectra(p: dict[str, Path], members: pd.DataFrame) -> pd.DataFrame:
    wanted = set(members["spectrum_id"].astype(str))
    records: dict[str, dict[str, Any]] = {}
    for mode in ("Pos", "Neg"):
        source = p["freeze"] / mode / "28_cleaned_spectra.jsonl"
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    spectrum_id = str(record["spectrum_id"])
                    if spectrum_id in wanted:
                        records[spectrum_id] = record
    missing = sorted(wanted.difference(records))
    if missing:
        raise RuntimeError(f"Missing frozen spectra from Notebook 28: {missing[:3]}")
    index = members.copy().sort_values(["cohort", "selection_order", "mode", "spectrum_id"]).reset_index(drop=True)
    index.insert(0, "case_row", np.arange(len(index), dtype=int))
    mz_arrays, intensity_arrays, peak_counts, spectrum_hashes = [], [], [], []
    for spectrum_id in index["spectrum_id"].astype(str):
        record = records[spectrum_id]
        mz = np.asarray(record["mz"], dtype=np.float32)
        intensity = np.asarray(record["intensity"], dtype=np.float32)
        order = np.argsort(mz, kind="stable")
        mz, intensity = mz[order], intensity[order]
        mz_arrays.append(mz); intensity_arrays.append(intensity)
        peak_counts.append(len(mz)); spectrum_hashes.append(str(record["spectrum_hash"]))
    index["case_peak_n"] = peak_counts
    index["case_spectrum_hash"] = spectrum_hashes
    _save_ragged(mz_arrays, intensity_arrays, p["case_mz"], p["case_intensity"], p["case_offsets"])
    write_csv(index, p["case_index"], compressed=True)
    return index


def prepare(root: Path, chunksize: int = 200_000) -> dict[str, Any]:
    """Freeze case peak arrays and precursor-window candidate reference spectra."""
    p = paths(root)
    p["out"].mkdir(parents=True, exist_ok=True)
    stage32, clusters, members = _verify_stage32(p)
    summary = json.loads(p["public_summary"].read_text(encoding="utf-8"))
    if summary.get("status") != "PASS" or summary.get("schema_version") != "ENVMOLBENCH_CLEANED_V2":
        raise RuntimeError("Local public spectral library has not passed its cleaning audit")
    for key, path in (("stream", p["public_stream"]), ("metadata", p["public_metadata"])):
        if not path.is_file() or path.stat().st_size != int(summary["outputs"][key]["size_bytes"]):
            raise RuntimeError(f"Public-library {key} file is absent or has changed size")
    if sha256_file(p["public_metadata"]) != summary["outputs"]["metadata"]["sha256"]:
        raise RuntimeError("Public-library metadata SHA-256 changed")

    case = _prepare_case_spectra(p, members)
    query_groups = {
        (mode, adduct): group["precursor_mz"].to_numpy(dtype=float)
        for (mode, adduct), group in case.groupby(["mode", "adduct"], sort=True)
    }
    columns = [
        "stream_offset", "source_library", "source_record_ordinal", "original_spectrum_id",
        "instrument", "inchikey_full", "inchikey_2d", "canonical_smiles", "precursor_mz",
        "ionmode", "adduct", "identity_ppm_error", "ce_real", "ce_bucket",
        "cleaned_peak_n", "spectrum_hash",
    ]
    selected_chunks: list[pd.DataFrame] = []
    scanned = 0
    reader = pd.read_csv(p["public_metadata"], usecols=columns, chunksize=chunksize, low_memory=False)
    for chunk in tqdm(reader, total=int(np.ceil(summary["export"]["record_count"] / chunksize)), desc="Scan public-library metadata"):
        scanned += len(chunk)
        chunk = chunk[
            pd.to_numeric(chunk["identity_ppm_error"], errors="coerce").le(PRECURSOR_PPM)
            & pd.to_numeric(chunk["cleaned_peak_n"], errors="coerce").ge(MAIN_MATCHED_PEAKS)
            & chunk["ionmode"].isin(["Positive", "Negative"])
        ].copy()
        if chunk.empty:
            continue
        chunk["mode"] = chunk["ionmode"].map({"Positive": "Pos", "Negative": "Neg"})
        keep = np.zeros(len(chunk), dtype=bool)
        for (mode, adduct), masses in query_groups.items():
            positions = np.flatnonzero(chunk["mode"].eq(mode).to_numpy() & chunk["adduct"].eq(adduct).to_numpy())
            if len(positions):
                keep[positions] = _within_any_reference_window(
                    chunk.iloc[positions]["precursor_mz"].to_numpy(dtype=float), masses
                )
        if keep.any():
            selected_chunks.append(chunk.iloc[np.flatnonzero(keep)].copy())
    if scanned != int(summary["export"]["record_count"]):
        raise RuntimeError(f"Public metadata scan incomplete: {scanned}")
    raw = pd.concat(selected_chunks, ignore_index=True) if selected_chunks else pd.DataFrame(columns=columns + ["mode"])
    if raw.empty:
        raise RuntimeError("No public reference spectra fall inside the frozen precursor/adduct windows")
    raw_n = len(raw)
    collision_hashes = raw.groupby("spectrum_hash")["inchikey_2d"].nunique()
    collision_hashes = set(collision_hashes[collision_hashes.gt(1)].index)
    raw = raw[~raw["spectrum_hash"].isin(collision_hashes)].copy()
    raw = raw.sort_values(
        ["inchikey_2d", "mode", "adduct", "source_library", "identity_ppm_error", "cleaned_peak_n", "stream_offset"],
        ascending=[True, True, True, True, True, False, True],
    ).drop_duplicates(["inchikey_2d", "mode", "spectrum_hash"], keep="first")
    raw["ce_stratum"] = pd.to_numeric(raw["ce_bucket"], errors="coerce").fillna(-1).astype(str)
    reference = (
        raw.groupby(["inchikey_2d", "mode", "adduct", "ce_stratum", "source_library"], sort=False, dropna=False)
        .head(REFERENCE_MAX_PER_STRATUM)
        .copy()
        .reset_index(drop=True)
    )
    reference = reference.sort_values(
        ["precursor_mz", "mode", "adduct", "inchikey_2d", "source_library", "stream_offset"]
    ).reset_index(drop=True)
    reference.insert(0, "reference_row", np.arange(len(reference), dtype=int))
    mz_arrays, intensity_arrays = [], []
    with p["public_stream"].open("rb") as stream:
        for row in tqdm(reference.itertuples(index=False), total=len(reference), desc="Load candidate reference spectra"):
            stream.seek(int(row.stream_offset))
            key, record = pickle.load(stream)
            if str(key) != str(row.inchikey_2d) or str(record.get("spectrum_hash")) != str(row.spectrum_hash):
                raise RuntimeError(f"Public stream/index mismatch at offset {row.stream_offset}")
            mz, intensity = _record_peaks(record)
            if len(mz) < MAIN_MATCHED_PEAKS:
                raise RuntimeError("Prepared reference contains fewer peaks than the frozen minimum")
            mz_arrays.append(mz); intensity_arrays.append(intensity)
    _save_ragged(mz_arrays, intensity_arrays, p["reference_mz"], p["reference_intensity"], p["reference_offsets"])
    reference = reference.drop(columns=["ce_stratum"])
    write_csv(reference, p["reference_index"], compressed=True)

    artifacts = [p["case_index"], p["case_mz"], p["case_intensity"], p["case_offsets"],
                 p["reference_index"], p["reference_mz"], p["reference_intensity"], p["reference_offsets"]]
    manifest = {
        "version": VERSION,
        "stage": "prepare",
        "status": "LIBRARY_CANDIDATES_PREPARED",
        "generated_at_utc": utc_now(),
        "identity_reveal_after_freeze": True,
        "stage32_manifest_sha256": sha256_file(p["cohort_manifest"]),
        "stage32_output_hashes": stage32["output_hashes"],
        "public_library": {
            "schema": summary["schema_version"],
            "libraries": summary["export"]["library_counts"],
            "total_records_scanned": scanned,
            "metadata_sha256": summary["outputs"]["metadata"]["sha256"],
            "stream_sha256_from_cleaning_manifest": summary["outputs"]["stream"]["sha256"],
        },
        "rules": {
            "ion_mode": "exact",
            "adduct": "exact normalized adduct",
            "precursor_ppm_max": PRECURSOR_PPM,
            "reference_identity_ppm_max": PRECURSOR_PPM,
            "minimum_reference_peaks": MAIN_MATCHED_PEAKS,
            "cross_identity_exact_spectrum_hashes_excluded": len(collision_hashes),
            "max_references_per_identity_mode_adduct_ce_source": REFERENCE_MAX_PER_STRATUM,
        },
        "counts": {
            "frozen_clusters": len(clusters),
            "case_spectra": len(case),
            "raw_precursor_window_reference_spectra": raw_n,
            "selected_reference_spectra": len(reference),
            "candidate_connectivity_keys": int(reference["inchikey_2d"].nunique()),
        },
        "artifacts": {path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in artifacts},
        "software": _software(),
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
    used_a: set[int] = set(); used_b: set[int] = set()
    score = 0.0; matches = 0
    for product, first, second in sorted(candidates, reverse=True):
        if first in used_a or second in used_b:
            continue
        used_a.add(first); used_b.add(second)
        score += product; matches += 1
    return float(min(score, 1.0)), int(matches)


def _empty_pairs() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "case_row", "spectrum_id", "cross_polarity_cluster", "cohort", "mode", "adduct",
        "case_precursor_mz", "inchikey_2d", "inchikey_full", "canonical_smiles", "reference_row",
        "modified_cosine", "matched_peaks", "precursor_ppm_error", "reference_precursor_mz",
        "reference_source_library", "reference_original_spectrum_id", "reference_instrument",
        "reference_collision_energy", "passes_main_library_rule", "passes_high_library_rule",
    ])


def _no_match_payload() -> dict[str, Any]:
    """Fixed empty annotation schema so a scientifically valid zero-hit run is stable."""
    return {
        "inchikey_2d": "",
        "inchikey_full": "",
        "canonical_smiles": "",
        "reference_row": np.nan,
        "modified_cosine": np.nan,
        "matched_peaks": np.nan,
        "precursor_ppm_error": np.nan,
        "reference_precursor_mz": np.nan,
        "reference_source_library": "",
        "reference_original_spectrum_id": "",
        "reference_instrument": "",
        "reference_collision_energy": np.nan,
        "passes_main_library_rule": False,
        "passes_high_library_rule": False,
        "runner_up_inchikey_2d": "",
        "runner_up_modified_cosine": np.nan,
        "best_runner_up_score_delta": np.nan,
    }


def _feature_summary(case: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    passing = pairs[pairs["passes_main_library_rule"]].copy() if len(pairs) else pairs.copy()
    for row in case.itertuples(index=False):
        group = passing[passing["case_row"].eq(int(row.case_row))]
        if group.empty:
            result = _no_match_payload()
            result.update({"case_row": int(row.case_row), "library_match_status": "no_main_library_match",
                           "has_any_main_match": False, "is_nonambiguous_library_match": False,
                           "candidate_identity_n": 0})
            rows.append(result)
            continue
        ordered = group.sort_values(
            ["modified_cosine", "matched_peaks", "precursor_ppm_error", "inchikey_2d"],
            ascending=[False, False, True, True],
        ).drop_duplicates("inchikey_2d")
        best = ordered.iloc[0]; runner = ordered.iloc[1] if len(ordered) > 1 else None
        runner_score = float(runner["modified_cosine"]) if runner is not None else np.nan
        ambiguous = bool(runner is not None and float(best["modified_cosine"]) - runner_score <= AMBIGUITY_DELTA)
        status = "ambiguous_library_match" if ambiguous else (
            "high_confidence_library_candidate" if bool(best["passes_high_library_rule"])
            else "probable_library_candidate"
        )
        result = best.to_dict()
        result.update({
            "case_row": int(row.case_row), "library_match_status": status, "has_any_main_match": True,
            "is_nonambiguous_library_match": not ambiguous, "candidate_identity_n": len(ordered),
            "runner_up_inchikey_2d": str(runner["inchikey_2d"]) if runner is not None else "",
            "runner_up_modified_cosine": runner_score,
            "best_runner_up_score_delta": float(best["modified_cosine"]) - runner_score if runner is not None else np.nan,
        })
        rows.append(result)
    result = case.merge(pd.DataFrame(rows), on="case_row", how="left", validate="one_to_one", suffixes=("", "__match"))
    return result


def _cluster_summary(clusters: pd.DataFrame, case: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    passing = pairs[pairs["passes_main_library_rule"]].copy() if len(pairs) else pairs.copy()
    rows: list[dict[str, Any]] = []
    for cluster in clusters["cross_polarity_cluster"].astype(str):
        group = passing[passing["cross_polarity_cluster"].eq(cluster)]
        if group.empty:
            result = _no_match_payload()
            result.update({"cross_polarity_cluster": cluster, "library_match_status": "no_main_library_match",
                           "has_any_main_match": False, "is_nonambiguous_library_match": False,
                           "candidate_identity_n": 0, "supporting_spectrum_n": 0, "supporting_mode_n": 0,
                           "supporting_modes": ""})
            rows.append(result)
            continue
        identity = (
            group.sort_values(["inchikey_2d", "modified_cosine", "matched_peaks", "precursor_ppm_error"],
                              ascending=[True, False, False, True])
            .groupby("inchikey_2d", as_index=False).first()
        )
        support = group.groupby("inchikey_2d").agg(
            supporting_spectrum_n=("spectrum_id", "nunique"),
            supporting_mode_n=("mode", "nunique"),
            supporting_modes=("mode", lambda x: ";".join(sorted(set(x)))),
        ).reset_index()
        identity = identity.merge(support, on="inchikey_2d", validate="one_to_one")
        ordered = identity.sort_values(
            ["modified_cosine", "supporting_mode_n", "supporting_spectrum_n", "matched_peaks", "precursor_ppm_error", "inchikey_2d"],
            ascending=[False, False, False, False, True, True],
        )
        best = ordered.iloc[0]; runner = ordered.iloc[1] if len(ordered) > 1 else None
        runner_score = float(runner["modified_cosine"]) if runner is not None else np.nan
        ambiguous = bool(runner is not None and float(best["modified_cosine"]) - runner_score <= AMBIGUITY_DELTA)
        status = "ambiguous_library_match" if ambiguous else (
            "high_confidence_library_candidate" if bool(best["passes_high_library_rule"])
            else "probable_library_candidate"
        )
        result = best.to_dict()
        result.update({
            "cross_polarity_cluster": cluster, "library_match_status": status, "has_any_main_match": True,
            "is_nonambiguous_library_match": not ambiguous, "candidate_identity_n": len(ordered),
            "runner_up_inchikey_2d": str(runner["inchikey_2d"]) if runner is not None else "",
            "runner_up_modified_cosine": runner_score,
            "best_runner_up_score_delta": float(best["modified_cosine"]) - runner_score if runner is not None else np.nan,
        })
        rows.append(result)
    match_frame = pd.DataFrame(rows)
    drop_overlap = [column for column in match_frame.columns if column in clusters.columns and column != "cross_polarity_cluster"]
    match_frame = match_frame.drop(columns=drop_overlap)
    return clusters.merge(match_frame, on="cross_polarity_cluster", how="left", validate="one_to_one")


def match(root: Path) -> dict[str, Any]:
    """Run strict precursor/adduct/fragment matching and aggregate to frozen clusters."""
    p = paths(root)
    stage32, clusters, members = _verify_stage32(p)
    preparation = json.loads(p["prepare_manifest"].read_text(encoding="utf-8"))
    if preparation.get("status") != "LIBRARY_CANDIDATES_PREPARED":
        raise RuntimeError("Notebook 33 preparation has not completed")
    for name, details in preparation["artifacts"].items():
        candidate = p["out"] / name
        if not candidate.is_file() or sha256_file(candidate) != details["sha256"]:
            raise RuntimeError(f"Prepared Notebook 33 artifact changed: {name}")
    case = pd.read_csv(p["case_index"])
    reference = pd.read_csv(p["reference_index"])
    case_mz = np.load(p["case_mz"], mmap_mode="r"); case_intensity = np.load(p["case_intensity"], mmap_mode="r")
    case_offsets = np.load(p["case_offsets"], mmap_mode="r")
    ref_mz = np.load(p["reference_mz"], mmap_mode="r"); ref_intensity = np.load(p["reference_intensity"], mmap_mode="r")
    ref_offsets = np.load(p["reference_offsets"], mmap_mode="r")

    query_lookup: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for key, group in case.groupby(["mode", "adduct"], sort=True):
        group = group.sort_values(["precursor_mz", "case_row"])
        query_lookup[key] = (group["precursor_mz"].to_numpy(float), group["case_row"].to_numpy(int))
    best_pairs: dict[tuple[int, str], dict[str, Any]] = {}
    comparison_n = 0
    for ref in tqdm(reference.itertuples(index=False), total=len(reference), desc="Modified-cosine library matching"):
        masses, rows = query_lookup.get((str(ref.mode), str(ref.adduct)), (np.empty(0), np.empty(0, dtype=int)))
        tolerance = float(ref.precursor_mz) * PRECURSOR_PPM * 1e-6
        left = int(np.searchsorted(masses, float(ref.precursor_mz) - tolerance, side="left"))
        right = int(np.searchsorted(masses, float(ref.precursor_mz) + tolerance, side="right"))
        rstart, rend = ref_offsets[int(ref.reference_row):int(ref.reference_row) + 2]
        rmz, rint = ref_mz[rstart:rend], ref_intensity[rstart:rend]
        for case_row in rows[left:right]:
            case_row = int(case_row)
            cstart, cend = case_offsets[case_row:case_row + 2]
            cmz, cint = case_mz[cstart:cend], case_intensity[cstart:cend]
            case_precursor = float(case.iloc[case_row]["precursor_mz"])
            score, matched_peaks = modified_cosine_score(
                rmz, rint, float(ref.precursor_mz), cmz, cint, case_precursor
            )
            comparison_n += 1
            if score < 0.50 or matched_peaks < 4:
                continue
            ppm_error = abs(case_precursor - float(ref.precursor_mz)) / float(ref.precursor_mz) * 1e6
            case_row_data = case.iloc[case_row]
            candidate = {
                "case_row": case_row,
                "spectrum_id": str(case_row_data["spectrum_id"]),
                "cross_polarity_cluster": str(case_row_data["cross_polarity_cluster"]),
                "cohort": str(case_row_data["cohort"]),
                "mode": str(case_row_data["mode"]),
                "adduct": str(case_row_data["adduct"]),
                "case_precursor_mz": case_precursor,
                "inchikey_2d": str(ref.inchikey_2d),
                "inchikey_full": str(ref.inchikey_full),
                "canonical_smiles": str(ref.canonical_smiles),
                "reference_row": int(ref.reference_row),
                "modified_cosine": score,
                "matched_peaks": matched_peaks,
                "precursor_ppm_error": ppm_error,
                "reference_precursor_mz": float(ref.precursor_mz),
                "reference_source_library": str(ref.source_library),
                "reference_original_spectrum_id": str(ref.original_spectrum_id),
                "reference_instrument": str(ref.instrument),
                "reference_collision_energy": ref.ce_real,
            }
            key = (case_row, str(ref.inchikey_2d))
            incumbent = best_pairs.get(key)
            order = (score, matched_peaks, -ppm_error)
            old_order = (float(incumbent["modified_cosine"]), int(incumbent["matched_peaks"]),
                         -float(incumbent["precursor_ppm_error"])) if incumbent else None
            if incumbent is None or order > old_order:
                best_pairs[key] = candidate
    pairs = pd.DataFrame(best_pairs.values()) if best_pairs else _empty_pairs()
    pairs["passes_main_library_rule"] = (
        pd.to_numeric(pairs["modified_cosine"], errors="coerce").ge(MAIN_COSINE)
        & pd.to_numeric(pairs["matched_peaks"], errors="coerce").ge(MAIN_MATCHED_PEAKS)
        & pd.to_numeric(pairs["precursor_ppm_error"], errors="coerce").le(PRECURSOR_PPM)
    )
    pairs["passes_high_library_rule"] = (
        pd.to_numeric(pairs["modified_cosine"], errors="coerce").ge(HIGH_COSINE)
        & pd.to_numeric(pairs["matched_peaks"], errors="coerce").ge(HIGH_MATCHED_PEAKS)
        & pd.to_numeric(pairs["precursor_ppm_error"], errors="coerce").le(PRECURSOR_PPM)
    )
    pairs = pairs.sort_values(["case_row", "modified_cosine", "matched_peaks"], ascending=[True, False, False])
    write_csv(pairs, p["candidate_pairs"], compressed=True)
    feature = _feature_summary(case, pairs)
    cluster = _cluster_summary(clusters, case, pairs)
    write_csv(feature, p["feature_summary"], compressed=True)
    write_csv(cluster, p["cluster_summary"])
    followup = cluster[
        ~cluster["is_nonambiguous_library_match"].fillna(False).astype(bool)
    ].copy()
    followup["library_resolution_reason"] = np.where(
        followup["library_match_status"].eq("ambiguous_library_match"),
        "ambiguous measured-library candidates", "no measured-library candidate meeting the main rule"
    )
    write_csv(followup, p["without_nonambiguous_match"])
    if p["legacy_sirius_followup"].is_file():
        p["legacy_sirius_followup"].unlink()
    audit = cluster.groupby("cohort", sort=False).agg(
        n_clusters=("cross_polarity_cluster", "size"),
        any_main_match=("has_any_main_match", "sum"),
        nonambiguous_library_match=("is_nonambiguous_library_match", "sum"),
        high_confidence=("library_match_status", lambda x: int(x.eq("high_confidence_library_candidate").sum())),
        probable=("library_match_status", lambda x: int(x.eq("probable_library_candidate").sum())),
        ambiguous=("library_match_status", lambda x: int(x.eq("ambiguous_library_match").sum())),
        no_main_match=("library_match_status", lambda x: int(x.eq("no_main_library_match").sum())),
        without_nonambiguous_library_match=("is_nonambiguous_library_match", lambda x: int((~x.fillna(False).astype(bool)).sum())),
    ).reset_index()
    write_csv(audit, p["cohort_audit"])

    artifacts = [p["candidate_pairs"], p["feature_summary"], p["cluster_summary"], p["without_nonambiguous_match"], p["cohort_audit"]]
    manifest = {
        "version": VERSION,
        "stage": 33,
        "status": "PUBLIC_LIBRARY_MATCHING_COMPLETE",
        "generated_at_utc": utc_now(),
        "identity_claim": "Measured-library connectivity candidate after frozen ranking; DIA pseudo-MS/MS and no authentic-standard RT, therefore never Level 1",
        "unmatched_interpretation": "Unmatched means unlabelled, never a confirmed negative",
        "rules": {
            "precursor_ppm_max": PRECURSOR_PPM,
            "ion_mode": "exact",
            "adduct": "exact normalized adduct",
            "fragment_tolerance_da": FRAGMENT_TOLERANCE_DA,
            "main_modified_cosine_min": MAIN_COSINE,
            "main_matched_peaks_min": MAIN_MATCHED_PEAKS,
            "high_modified_cosine_min": HIGH_COSINE,
            "high_matched_peaks_min": HIGH_MATCHED_PEAKS,
            "ambiguity_delta_max": AMBIGUITY_DELTA,
        },
        "counts": {
            "frozen_clusters": len(cluster),
            "case_spectra": len(case),
            "candidate_reference_spectra": len(reference),
            "precursor_window_comparisons": comparison_n,
            "saved_feature_identity_pairs": len(pairs),
            "pairs_passing_main": int(pairs["passes_main_library_rule"].sum()),
            "clusters_with_any_main_match": int(cluster["has_any_main_match"].sum()),
            "clusters_with_nonambiguous_match": int(cluster["is_nonambiguous_library_match"].sum()),
            "clusters_without_nonambiguous_library_match": len(followup),
        },
        "input_hashes": {
            p["cohort_manifest"].name: sha256_file(p["cohort_manifest"]),
            p["prepare_manifest"].name: sha256_file(p["prepare_manifest"]),
        },
        "artifacts": {path.name: {"sha256": sha256_file(path), "rows": int(pd.read_csv(path).shape[0])} for path in artifacts},
        "software": _software(),
    }
    write_json(p["run_manifest"], manifest)
    return manifest
