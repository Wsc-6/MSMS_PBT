"""MS1 feature finding and DDA-MS2 association for the Portugal case study.

Notebook 23 executes this module with the ``env_dreams`` interpreter because
that environment already contains pyOpenMS.  The workflow is incremental at
the mzML-run level and never modifies the frozen raw or mzML files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VERSION = "PBT_MSMS_PORTUGAL_FEATURE_EXTRACTION_23_V1"
EXPECTED_RUNS = 96
EXPECTED_SAMPLES = 48
MASS_ERROR_PPM = 10.0
NOISE_THRESHOLD_INT = 200.0
CHROM_PEAK_SNR = 3.0
MIN_TRACE_LENGTH_SECONDS = 5.0
MIN_FWHM_SECONDS = 2.0
MAX_FWHM_SECONDS = 60.0
MS2_ASSOC_PPM = 20.0
MS2_ASSOC_MIN_RT_SECONDS = 15.0
AB_ALIGN_PPM = 10.0
AB_ALIGN_RT_SECONDS = 15.0
MIN_FRAGMENT_MZ = 15.0
MAX_FRAGMENT_MZ = 2000.0
MIN_RELATIVE_INTENSITY = 0.001
MODEL_MIN_PEAKS = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n", compression=compression)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
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


def paths(root: Path) -> dict[str, Path]:
    case = root / "external_data/portugal_wastewater_5830725"
    out = case / "03_feature_extraction_23"
    manifests = case / "manifests"
    return {
        "case": case,
        "nb22": manifests / "22_msconvert_run_manifest.json",
        "mzml_status": manifests / "22_msconvert_file_status.csv",
        "dataset_manifest": manifests / "21_portugal_dda_dataset_manifest.csv",
        "out": out,
        "features": out / "per_run_features",
        "ms2": out / "per_run_ms2",
        "peaks": out / "per_run_peaks",
        "receipts": out / "receipts",
        "sample_features": out / "sample_consensus_features",
        "run_status": manifests / "23_feature_extraction_run_status.csv",
        "sample_summary": manifests / "23_sample_consensus_summary.csv",
        "all_ms2_index": manifests / "23_all_ms2_event_index.csv.gz",
        "sample_ms2_index": manifests / "23_sample_ms2_feature_index.csv.gz",
        "qc": manifests / "23_feature_extraction_qc_checks.csv",
        "manifest": manifests / "23_feature_extraction_run_manifest.json",
    }


def ensure_ascii_alias(root: Path) -> Path:
    """Return an ASCII-only junction to the project for pyOpenMS on Windows.

    The native OpenMS Windows reader can fail on Unicode paths.  A directory
    junction points to the same files and does not copy or modify them.
    """
    if all(ord(character) < 128 for character in str(root)):
        return root
    alias = Path(tempfile.gettempdir()) / "pbt_msms_project_ascii"
    if alias.exists():
        try:
            if alias.resolve() == root.resolve():
                return alias
        except OSError:
            pass
        raise RuntimeError(f"ASCII alias exists but points elsewhere: {alias}")
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(alias), str(root)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0 or not alias.exists():
        raise RuntimeError(f"Cannot create pyOpenMS ASCII junction {alias}: {result.stdout} {result.stderr}")
    return alias


def parameter_payload() -> dict[str, Any]:
    return {
        "mass_trace": {
            "mass_error_ppm": MASS_ERROR_PPM,
            "noise_threshold_int": NOISE_THRESHOLD_INT,
            "chrom_peak_snr": CHROM_PEAK_SNR,
            "min_trace_length": MIN_TRACE_LENGTH_SECONDS,
            "quant_method": "area",
        },
        "elution_peak": {
            "chrom_peak_snr": CHROM_PEAK_SNR,
            "width_filtering": "fixed",
            "min_fwhm": MIN_FWHM_SECONDS,
            "max_fwhm": MAX_FWHM_SECONDS,
        },
        "feature_finding_metabo": {
            "charge_lower_bound": 1,
            "charge_upper_bound": 2,
            "remove_single_traces": "false",
            "isotope_filtering_model": "metabolites (5% RMS)",
        },
        "ms2_association": {"ppm": MS2_ASSOC_PPM, "minimum_rt_seconds": MS2_ASSOC_MIN_RT_SECONDS},
        "fragment_cleaning": {
            "minimum_mz": MIN_FRAGMENT_MZ,
            "maximum_mz": MAX_FRAGMENT_MZ,
            "minimum_relative_intensity": MIN_RELATIVE_INTENSITY,
            "minimum_model_peaks": MODEL_MIN_PEAKS,
        },
        "iterative_injection_alignment": {"ppm": AB_ALIGN_PPM, "rt_seconds": AB_ALIGN_RT_SECONDS},
    }


def load_inputs(root: Path) -> tuple[dict[str, Path], dict[str, Any], pd.DataFrame]:
    p = paths(root)
    nb22 = json.loads(p["nb22"].read_text(encoding="utf-8"))
    if nb22.get("status") != "READY_FOR_NOTEBOOK_23" or not nb22.get("qc_all_pass"):
        raise RuntimeError(f"Notebook 22 is not READY_FOR_NOTEBOOK_23: {nb22.get('status')}")
    mzml = pd.read_csv(p["mzml_status"])
    datasets = pd.read_csv(p["dataset_manifest"])
    if len(mzml) != EXPECTED_RUNS or not mzml["qc_pass"].astype(bool).all():
        raise RuntimeError("Notebook 22 file-level QC is incomplete")
    plan = datasets.merge(
        mzml[["dataset_id", "output_mzml", "mzml_sha256", "spectra_n", "ms1_n", "ms2_n"]],
        on="dataset_id", how="inner", validate="one_to_one",
    )
    if len(plan) != EXPECTED_RUNS or plan["dataset_id"].nunique() != EXPECTED_RUNS:
        raise RuntimeError(f"Expected {EXPECTED_RUNS} unique mzML runs, found {len(plan)}")
    missing = [value for value in plan["output_mzml"] if not Path(value).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing mzML files: {missing[:3]}")
    return p, nb22, plan.sort_values(["mode", "site_code", "campaign", "injection"]).reset_index(drop=True)


def clean_fragments(mz: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    keep = (
        np.isfinite(mz) & np.isfinite(intensity) & (intensity > 0)
        & (mz >= MIN_FRAGMENT_MZ) & (mz <= MAX_FRAGMENT_MZ)
    )
    mz, intensity = mz[keep], intensity[keep]
    if not len(intensity) or float(intensity.max()) <= 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    intensity = intensity / float(intensity.max())
    keep = intensity >= MIN_RELATIVE_INTENSITY
    mz, intensity = mz[keep], intensity[keep]
    order = np.argsort(mz, kind="stable")
    mz, intensity = mz[order], intensity[order]

    # Match the isotope-duplicate suppression used for the public training spectra.
    keep = np.ones(len(mz), dtype=bool)
    for first in range(len(mz)):
        if not keep[first]:
            continue
        for second in range(first + 1, len(mz)):
            difference = mz[second] - mz[first]
            if difference > 1.01:
                break
            if 1.00085 < difference < 1.00585 and intensity[second] < intensity[first] * 0.8:
                keep[second] = False
    return mz[keep].astype(np.float32), intensity[keep].astype(np.float32)


def associate_feature(
    precursor_mz: float,
    rt: float,
    feature_mz: np.ndarray,
    feature_rt: np.ndarray,
    feature_width: np.ndarray,
    mz_order: np.ndarray,
) -> tuple[int | None, float, float]:
    sorted_mz = feature_mz[mz_order]
    delta = precursor_mz * MS2_ASSOC_PPM * 1e-6
    left = int(np.searchsorted(sorted_mz, precursor_mz - delta, side="left"))
    right = int(np.searchsorted(sorted_mz, precursor_mz + delta, side="right"))
    best: tuple[float, int, float, float] | None = None
    for sorted_index in range(left, right):
        index = int(mz_order[sorted_index])
        ppm = abs(feature_mz[index] - precursor_mz) / precursor_mz * 1e6
        rt_error = abs(feature_rt[index] - rt)
        rt_tolerance = max(MS2_ASSOC_MIN_RT_SECONDS, 1.5 * max(feature_width[index], 0.0))
        if rt_error > rt_tolerance:
            continue
        score = ppm / MS2_ASSOC_PPM + rt_error / rt_tolerance
        candidate = (score, index, ppm, rt_error)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None, math.nan, math.nan
    return best[1], best[2], best[3]


def extract_one_run(row: Any, p: dict[str, Path], params_hash: str) -> dict[str, Any]:
    import pyopenms as oms

    dataset_id = str(row.dataset_id)
    feature_path = p["features"] / f"{dataset_id}.csv.gz"
    ms2_path = p["ms2"] / f"{dataset_id}.csv.gz"
    peak_path = p["peaks"] / f"{dataset_id}.npz"
    receipt_path = p["receipts"] / f"{dataset_id}.json"
    expected = {
        "version": VERSION,
        "dataset_id": dataset_id,
        "mzml_sha256": str(row.mzml_sha256),
        "parameters_sha256": params_hash,
    }
    if receipt_path.exists() and feature_path.exists() and ms2_path.exists() and peak_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if all(receipt.get(key) == value for key, value in expected.items()):
            return {**receipt, "action": "skipped_valid_existing"}

    source_mzml = Path(row.output_mzml)
    pyopenms_root = p.get("pyopenms_root", p["case"].parents[1])
    project_root = p["case"].parents[1]
    pyopenms_mzml = pyopenms_root / source_mzml.relative_to(project_root)
    experiment = oms.MSExperiment()
    oms.MzMLFile().load(str(pyopenms_mzml), experiment)
    all_spectra = experiment.getSpectra()
    ms1_experiment = oms.MSExperiment()
    ms1_experiment.setSpectra([spectrum for spectrum in all_spectra if spectrum.getMSLevel() == 1])
    ms1_experiment.sortSpectra(True)

    detector = oms.MassTraceDetection()
    detector_params = detector.getDefaults()
    detector_params.setValue("mass_error_ppm", MASS_ERROR_PPM)
    detector_params.setValue("noise_threshold_int", NOISE_THRESHOLD_INT)
    detector_params.setValue("chrom_peak_snr", CHROM_PEAK_SNR)
    detector_params.setValue("min_trace_length", MIN_TRACE_LENGTH_SECONDS)
    detector_params.setValue("quant_method", "area")
    detector.setParameters(detector_params)
    mass_traces: list[Any] = []
    detector.run(ms1_experiment, mass_traces, 0)

    peak_detector = oms.ElutionPeakDetection()
    peak_params = peak_detector.getDefaults()
    peak_params.setValue("chrom_peak_snr", CHROM_PEAK_SNR)
    peak_params.setValue("width_filtering", "fixed")
    peak_params.setValue("min_fwhm", MIN_FWHM_SECONDS)
    peak_params.setValue("max_fwhm", MAX_FWHM_SECONDS)
    peak_detector.setParameters(peak_params)
    split_traces: list[Any] = []
    peak_detector.detectPeaks(mass_traces, split_traces)

    feature_finder = oms.FeatureFindingMetabo()
    finder_params = feature_finder.getDefaults()
    finder_params.setValue("charge_lower_bound", 1)
    finder_params.setValue("charge_upper_bound", 2)
    finder_params.setValue("remove_single_traces", "false")
    finder_params.setValue("report_convex_hulls", "false")
    finder_params.setValue("report_chromatograms", "false")
    feature_finder.setParameters(finder_params)
    feature_map = oms.FeatureMap()
    chromatograms: list[Any] = []
    feature_finder.run(split_traces, feature_map, chromatograms)

    feature_rows = []
    for index, feature in enumerate(feature_map):
        feature_rows.append({
            "dataset_id": dataset_id,
            "run_feature_id": f"{dataset_id}_F{index + 1:06d}",
            "mode": row.mode,
            "site_code": row.site_code,
            "campaign": int(row.campaign),
            "injection": row.injection,
            "mz": float(feature.getMZ()),
            "rt_seconds": float(feature.getRT()),
            "intensity_area": float(feature.getIntensity()),
            "charge": int(feature.getCharge()),
            "width_seconds": float(feature.getWidth()),
            "quality": float(feature.getOverallQuality()),
        })
    features = pd.DataFrame(feature_rows)
    if features.empty:
        raise RuntimeError(f"No MS1 features found for {dataset_id}")
    feature_mz = features["mz"].to_numpy(dtype=float)
    feature_rt = features["rt_seconds"].to_numpy(dtype=float)
    feature_width = features["width_seconds"].to_numpy(dtype=float)
    mz_order = np.argsort(feature_mz, kind="stable")

    event_rows: list[dict[str, Any]] = []
    mz_values: list[np.ndarray] = []
    intensity_values: list[np.ndarray] = []
    offsets = [0]
    ms2_ordinal = 0
    for spectrum in all_spectra:
        if spectrum.getMSLevel() != 2:
            continue
        ms2_ordinal += 1
        precursors = spectrum.getPrecursors()
        if not precursors:
            continue
        precursor = precursors[0]
        precursor_mz = float(precursor.getMZ())
        precursor_intensity = float(precursor.getIntensity())
        precursor_charge = int(precursor.getCharge())
        collision_energy = math.nan
        if precursor.metaValueExists("collision energy"):
            try:
                collision_energy = float(precursor.getMetaValue("collision energy"))
            except (TypeError, ValueError):
                collision_energy = math.nan
        rt = float(spectrum.getRT())
        raw_mz, raw_intensity = spectrum.get_peaks()
        clean_mz, clean_intensity = clean_fragments(raw_mz, raw_intensity)
        associated, ppm_error, rt_error = associate_feature(
            precursor_mz, rt, feature_mz, feature_rt, feature_width, mz_order
        )
        event_id = f"{dataset_id}_MS2_{ms2_ordinal:06d}"
        offset = offsets[-1]
        mz_values.append(clean_mz)
        intensity_values.append(clean_intensity)
        offsets.append(offset + len(clean_mz))
        event_rows.append({
            "dataset_id": dataset_id,
            "ms2_event_id": event_id,
            "mode": row.mode,
            "site_code": row.site_code,
            "campaign": int(row.campaign),
            "injection": row.injection,
            "native_id": spectrum.getNativeID(),
            "rt_seconds": rt,
            "precursor_mz": precursor_mz,
            "precursor_intensity": precursor_intensity,
            "precursor_charge": precursor_charge,
            "collision_energy_ev": collision_energy,
            "raw_peak_n": int(len(raw_mz)),
            "clean_peak_n": int(len(clean_mz)),
            "model_eligible": bool(len(clean_mz) >= MODEL_MIN_PEAKS),
            "fragment_tic_raw": float(np.asarray(raw_intensity, dtype=float).sum()),
            "associated_feature_id": "" if associated is None else features.iloc[associated]["run_feature_id"],
            "association_ppm_error": ppm_error,
            "association_rt_error_seconds": rt_error,
            "peak_file": str(peak_path),
            "peak_offset": offset,
            "peak_count": int(len(clean_mz)),
        })
    events = pd.DataFrame(event_rows)
    if len(events) != int(row.ms2_n):
        raise RuntimeError(f"MS2 count mismatch for {dataset_id}: {len(events)} != {int(row.ms2_n)}")

    features["ms2_event_n"] = 0
    features["model_eligible_ms2_n"] = 0
    features["best_ms2_event_id"] = ""
    if not events.empty:
        linked = events.loc[events["associated_feature_id"].astype(str).ne("")].copy()
        if not linked.empty:
            counts = linked.groupby("associated_feature_id").size()
            eligible_counts = linked.loc[linked["model_eligible"]].groupby("associated_feature_id").size()
            features["ms2_event_n"] = features["run_feature_id"].map(counts).fillna(0).astype(int)
            features["model_eligible_ms2_n"] = features["run_feature_id"].map(eligible_counts).fillna(0).astype(int)
            ranked = linked.sort_values(
                ["associated_feature_id", "model_eligible", "clean_peak_n", "fragment_tic_raw", "precursor_intensity"],
                ascending=[True, False, False, False, False], kind="stable",
            ).drop_duplicates("associated_feature_id")
            best_map = ranked.set_index("associated_feature_id")["ms2_event_id"]
            features["best_ms2_event_id"] = features["run_feature_id"].map(best_map).fillna("")

    concatenated_mz = np.concatenate(mz_values).astype(np.float32) if mz_values else np.asarray([], dtype=np.float32)
    concatenated_intensity = np.concatenate(intensity_values).astype(np.float32) if intensity_values else np.asarray([], dtype=np.float32)
    atomic_npz(
        peak_path,
        offsets=np.asarray(offsets, dtype=np.int64),
        mz=concatenated_mz,
        intensity=concatenated_intensity,
    )
    atomic_csv(features, feature_path, compression="gzip")
    atomic_csv(events, ms2_path, compression="gzip")

    linked_n = int(events["associated_feature_id"].astype(str).ne("").sum())
    eligible_n = int(events["model_eligible"].sum())
    receipt = {
        **expected,
        "generated_at_utc": utc_now(),
        "action": "processed",
        "feature_file": str(feature_path),
        "ms2_file": str(ms2_path),
        "peak_file": str(peak_path),
        "feature_file_sha256": sha256_file(feature_path),
        "ms2_file_sha256": sha256_file(ms2_path),
        "peak_file_sha256": sha256_file(peak_path),
        "ms1_spectra_n": int(row.ms1_n),
        "ms2_spectra_n": int(row.ms2_n),
        "mass_traces_n": len(mass_traces),
        "split_mass_traces_n": len(split_traces),
        "features_n": len(features),
        "linked_ms2_n": linked_n,
        "linked_ms2_fraction": linked_n / len(events) if len(events) else 0.0,
        "model_eligible_ms2_n": eligible_n,
        "model_eligible_ms2_fraction": eligible_n / len(events) if len(events) else 0.0,
        "features_with_model_eligible_ms2_n": int(features["model_eligible_ms2_n"].gt(0).sum()),
    }
    atomic_json(receipt_path, receipt)
    return receipt


def match_ab_features(a: pd.DataFrame, b: pd.DataFrame) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    a_mz = a["mz"].to_numpy(dtype=float)
    a_rt = a["rt_seconds"].to_numpy(dtype=float)
    order = np.argsort(a_mz, kind="stable")
    sorted_mz = a_mz[order]
    used_a: set[int] = set()
    pairs: list[tuple[int, int]] = []
    b_order = b["intensity_area"].sort_values(ascending=False, kind="stable").index.to_list()
    for b_index in b_order:
        mz = float(b.at[b_index, "mz"])
        rt = float(b.at[b_index, "rt_seconds"])
        delta = mz * AB_ALIGN_PPM * 1e-6
        left = int(np.searchsorted(sorted_mz, mz - delta, side="left"))
        right = int(np.searchsorted(sorted_mz, mz + delta, side="right"))
        best: tuple[float, int] | None = None
        for sorted_index in range(left, right):
            a_index = int(order[sorted_index])
            if a_index in used_a:
                continue
            ppm = abs(a_mz[a_index] - mz) / mz * 1e6
            rt_error = abs(a_rt[a_index] - rt)
            if rt_error > AB_ALIGN_RT_SECONDS:
                continue
            candidate = (ppm / AB_ALIGN_PPM + rt_error / AB_ALIGN_RT_SECONDS, a_index)
            if best is None or candidate < best:
                best = candidate
        if best is not None:
            used_a.add(best[1])
            pairs.append((best[1], int(b_index)))
    paired_b = {b_index for _, b_index in pairs}
    return pairs, sorted(set(a.index) - used_a), sorted(set(b.index) - paired_b)


def build_sample_consensus(plan: pd.DataFrame, p: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    master_ms2_rows: list[pd.DataFrame] = []
    grouping = ["mode", "site_code", "campaign"]
    for keys, group in plan.groupby(grouping, sort=True):
        mode, site, campaign = keys
        if set(group["injection"]) != {"A", "B"} or len(group) != 2:
            raise RuntimeError(f"Expected A/B pair for {keys}")
        rows = {item.injection: item for item in group.itertuples(index=False)}
        feature_frames = {
            injection: pd.read_csv(p["features"] / f"{rows[injection].dataset_id}.csv.gz")
            for injection in ["A", "B"]
        }
        event_frames = {
            injection: pd.read_csv(p["ms2"] / f"{rows[injection].dataset_id}.csv.gz", low_memory=False)
            for injection in ["A", "B"]
        }
        for injection in ["A", "B"]:
            feature_frames[injection] = feature_frames[injection].reset_index(drop=True)
            event_frames[injection]["injection"] = injection
        pairs, only_a, only_b = match_ab_features(feature_frames["A"], feature_frames["B"])
        groups: list[list[tuple[str, int]]] = (
            [[("A", a_index), ("B", b_index)] for a_index, b_index in pairs]
            + [[("A", index)] for index in only_a]
            + [[("B", index)] for index in only_b]
        )
        event_lookup = pd.concat(event_frames.values(), ignore_index=True).set_index("ms2_event_id", drop=False)
        sample_id = f"PTWW_{mode}_{str(site).replace(' ', '-')}_C{int(campaign)}"
        consensus_rows = []
        for ordinal, members in enumerate(groups, start=1):
            member_rows = [feature_frames[injection].iloc[index] for injection, index in members]
            intensities = np.asarray([max(float(item["intensity_area"]), 0.0) for item in member_rows])
            weights = intensities if intensities.sum() > 0 else np.ones(len(member_rows))
            mz = float(np.average([float(item["mz"]) for item in member_rows], weights=weights))
            rt = float(np.average([float(item["rt_seconds"]) for item in member_rows], weights=weights))
            candidate_events = []
            for item in member_rows:
                event_id = str(item.get("best_ms2_event_id", ""))
                if event_id and event_id != "nan" and event_id in event_lookup.index:
                    candidate_events.append(event_lookup.loc[event_id])
            representative = None
            if candidate_events:
                representative = sorted(
                    candidate_events,
                    key=lambda item: (
                        bool(item["model_eligible"]), int(item["clean_peak_n"]),
                        float(item["fragment_tic_raw"]), float(item["precursor_intensity"]),
                    ), reverse=True,
                )[0]
            consensus_rows.append({
                "sample_id": sample_id,
                "sample_feature_id": f"{sample_id}_F{ordinal:06d}",
                "mode": mode,
                "site_code": site,
                "campaign": int(campaign),
                "mz": mz,
                "rt_seconds": rt,
                "intensity_area_mean_detected": float(intensities.mean()),
                "intensity_area_max": float(intensities.max()),
                "detected_in_A": any(injection == "A" for injection, _ in members),
                "detected_in_B": any(injection == "B" for injection, _ in members),
                "detected_injection_n": len(members),
                "run_feature_ids": "|".join(str(item["run_feature_id"]) for item in member_rows),
                "representative_ms2_event_id": "" if representative is None else representative["ms2_event_id"],
                "representative_dataset_id": "" if representative is None else representative["dataset_id"],
                "representative_injection": "" if representative is None else representative["injection"],
                "precursor_mz": math.nan if representative is None else float(representative["precursor_mz"]),
                "precursor_intensity": math.nan if representative is None else float(representative["precursor_intensity"]),
                "collision_energy_ev": math.nan if representative is None else float(representative["collision_energy_ev"]),
                "clean_peak_n": 0 if representative is None else int(representative["clean_peak_n"]),
                "model_eligible": False if representative is None else bool(representative["model_eligible"]),
                "peak_file": "" if representative is None else representative["peak_file"],
                "peak_offset": -1 if representative is None else int(representative["peak_offset"]),
                "peak_count": 0 if representative is None else int(representative["peak_count"]),
            })
        consensus = pd.DataFrame(consensus_rows).sort_values(["mz", "rt_seconds"]).reset_index(drop=True)
        output = p["sample_features"] / f"{sample_id}.csv.gz"
        atomic_csv(consensus, output, compression="gzip")
        eligible = consensus.loc[consensus["model_eligible"].astype(bool)].copy()
        master_ms2_rows.append(eligible)
        summary_rows.append({
            "sample_id": sample_id,
            "mode": mode,
            "site_code": site,
            "campaign": int(campaign),
            "features_total": len(consensus),
            "features_detected_both_injections": int(consensus["detected_injection_n"].eq(2).sum()),
            "features_with_ms2": int(consensus["representative_ms2_event_id"].astype(str).ne("").sum()),
            "model_eligible_ms2_features": len(eligible),
            "output_file": str(output),
            "output_sha256": sha256_file(output),
        })
    summary = pd.DataFrame(summary_rows)
    master = pd.concat(master_ms2_rows, ignore_index=True) if master_ms2_rows else pd.DataFrame()
    atomic_csv(summary, p["sample_summary"])
    atomic_csv(master, p["sample_ms2_index"], compression="gzip")
    return summary, master


def run(root: Path) -> dict[str, Any]:
    import pyopenms as oms

    p, nb22, plan = load_inputs(root)
    p["pyopenms_root"] = ensure_ascii_alias(root)
    for key in ["features", "ms2", "peaks", "receipts", "sample_features"]:
        p[key].mkdir(parents=True, exist_ok=True)
    params = parameter_payload()
    params_hash = stable_hash(params)
    records: list[dict[str, Any]] = []
    for number, row in enumerate(plan.itertuples(index=False), start=1):
        print(f"[{number:02d}/{len(plan)}] FEATURE {row.dataset_id}", flush=True)
        try:
            receipt = extract_one_run(row, p, params_hash)
            records.append({"dataset_id": row.dataset_id, "mode": row.mode, "site_code": row.site_code,
                            "campaign": int(row.campaign), "injection": row.injection, "status": "complete", **receipt})
        except Exception as exc:
            records.append({"dataset_id": row.dataset_id, "mode": row.mode, "site_code": row.site_code,
                            "campaign": int(row.campaign), "injection": row.injection, "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}"})
            atomic_csv(pd.DataFrame(records), p["run_status"])
            raise
        atomic_csv(pd.DataFrame(records), p["run_status"])

    status = pd.DataFrame(records)
    all_ms2 = pd.concat(
        [pd.read_csv(p["ms2"] / f"{dataset_id}.csv.gz", low_memory=False) for dataset_id in plan["dataset_id"]],
        ignore_index=True,
    )
    atomic_csv(all_ms2, p["all_ms2_index"], compression="gzip")
    summary, sample_ms2 = build_sample_consensus(plan, p)
    total_ms2_expected = int(plan["ms2_n"].sum())
    total_ms2 = int(status["ms2_spectra_n"].sum())
    linked_fraction = float(status["linked_ms2_n"].sum() / total_ms2) if total_ms2 else 0.0
    eligible_fraction = float(status["model_eligible_ms2_n"].sum() / total_ms2) if total_ms2 else 0.0
    qc_rows = [
        {"check": "notebook22_ready", "passed": nb22.get("status") == "READY_FOR_NOTEBOOK_23", "detail": nb22.get("status")},
        {"check": "all_96_runs_complete", "passed": len(status) == EXPECTED_RUNS and status["status"].eq("complete").all(), "detail": f"{len(status)}/{EXPECTED_RUNS}"},
        {"check": "ms2_count_matches_notebook22", "passed": total_ms2 == total_ms2_expected, "detail": f"{total_ms2}/{total_ms2_expected}"},
        {"check": "all_ms2_event_index_complete", "passed": len(all_ms2) == total_ms2_expected, "detail": f"{len(all_ms2)}/{total_ms2_expected}"},
        {"check": "every_run_has_ms1_features", "passed": status["features_n"].gt(0).all(), "detail": int(status["features_n"].min())},
        {"check": "every_run_has_model_eligible_ms2", "passed": status["model_eligible_ms2_n"].gt(0).all(), "detail": int(status["model_eligible_ms2_n"].min())},
        {"check": "dda_ms2_link_rate_at_least_50pct", "passed": linked_fraction >= 0.50, "detail": linked_fraction},
        {"check": "sample_units_48", "passed": len(summary) == EXPECTED_SAMPLES, "detail": len(summary)},
        {"check": "every_sample_has_model_eligible_ms2", "passed": summary["model_eligible_ms2_features"].gt(0).all(), "detail": int(summary["model_eligible_ms2_features"].min())},
        {"check": "primary_sample_ms2_index_nonempty", "passed": len(sample_ms2) > 0, "detail": len(sample_ms2)},
        {"check": "explicit_blank_subtraction", "passed": False, "required": False,
         "detail": "Public archive contains no filename-identifiable blank/QC runs; no blank subtraction claimed"},
    ]
    qc = pd.DataFrame(qc_rows)
    atomic_csv(qc, p["qc"])
    required = qc.loc[qc["required"].isna() | qc["required"].eq(True)]
    all_pass = bool(required["passed"].astype(bool).all())
    manifest = {
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "status": "READY_FOR_NOTEBOOK_24" if all_pass else "BLOCKED_FEATURE_QC",
        "scientific_scope": "OpenMS MS1 feature finding, DDA-MS2 association, and A/B iterative-injection consolidation",
        "notebook22_manifest_sha256": sha256_file(p["nb22"]),
        "mzml_status_sha256": sha256_file(p["mzml_status"]),
        "software": {"pyopenms": oms.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "pyopenms_ascii_alias": str(p["pyopenms_root"]),
        "parameters": params,
        "parameters_sha256": params_hash,
        "counts": {
            "runs": len(status), "sample_units": len(summary),
            "ms1_features": int(status["features_n"].sum()),
            "ms2_events": total_ms2,
            "linked_ms2_events": int(status["linked_ms2_n"].sum()),
            "linked_ms2_fraction": linked_fraction,
            "model_eligible_ms2_events": int(status["model_eligible_ms2_n"].sum()),
            "model_eligible_ms2_fraction": eligible_fraction,
            "sample_consensus_features": int(summary["features_total"].sum()),
            "sample_consensus_model_eligible_ms2_features": len(sample_ms2),
        },
        "blank_limitation": "No explicit blank/QC raw files are identifiable in the public archive; this workflow does not claim blank subtraction.",
        "interpretation_limit": "Detection frequency derived later from these files is MS1-feature occurrence; DDA MS2 availability remains acquisition-dependent.",
        "qc_all_required_pass": all_pass,
        "artifacts": {
            p["run_status"].name: {"sha256": sha256_file(p["run_status"]), "size_bytes": p["run_status"].stat().st_size},
            p["sample_summary"].name: {"sha256": sha256_file(p["sample_summary"]), "size_bytes": p["sample_summary"].stat().st_size},
            p["all_ms2_index"].name: {"sha256": sha256_file(p["all_ms2_index"]), "size_bytes": p["all_ms2_index"].stat().st_size},
            p["sample_ms2_index"].name: {"sha256": sha256_file(p["sample_ms2_index"]), "size_bytes": p["sample_ms2_index"].stat().st_size},
            p["qc"].name: {"sha256": sha256_file(p["qc"]), "size_bytes": p["qc"].stat().st_size},
        },
        "next_step": "Notebook 24: frozen-model inference on representative real-water MS/MS and hazard-priority ranking",
    }
    atomic_json(p["manifest"], manifest)
    print(qc.to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not all_pass:
        raise RuntimeError("Notebook 23 required QC failed")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    run(locate_root(args.root))


if __name__ == "__main__":
    main()
