"""Reproducible Bangladesh surface-water DIA cleaning and frozen-model inference.

All public entry points are designed to be called from the Bangladesh Notebooks 28 onward.
Raw MS-DIAL exports are read-only; all derived artifacts use versioned folders.
"""
from __future__ import annotations

from .deployment import deployment_comparison_dir

import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from pbt_msms import chemvista_external as ce
from pbt_msms.hazard_scoring import (
    B_TASK,
    CANONICAL_HAZARD_FORMULA,
    EC50_TASK,
    LC50_TASK,
    NOEC_TASK,
    P_TASK,
    TASK_ORDER as HAZARD_TASK_ORDER,
    harmonized_full_hazard,
    load_development_references,
)

VERSION = "PBT_MSMS_BANGLADESH_DIA_V3"
EXPECTED_STRICT = {"Pos": 10651, "Neg": 7670}
ADDUCTS = {
    "Pos": {"[M+H]+", "[M+NH4]+", "[M+Na]+", "[M+K]+", "[M+2H]2+"},
    "Neg": {"[M-H]-", "[M+Cl]-", "[M-2H]2-"},
}
SUSPICIOUS = {"Pos": "[M+2H]+", "Neg": "[M-2H]-"}
MAX_MZ, MAX_PEAKS, MIN_RELATIVE = 2000.0, 120, 0.001


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def paths(root: Path) -> dict[str, Path]:
    case = root / "external_data" / "bangladesh_surface_water"
    prepared = case / "04_blinded_inputs" / "29_pbt_model_inputs_v2"
    return {
        "case": case,
        "source": case / "03_msdial_exports" / "28_full",
        "clean": case / "04_dia_cleaned" / "28_full_v2",
        "freeze": case / "04_blinded_inputs" / "28_full_cleaned_v2",
        "prepared": prepared,
        "prediction": case / "05_frozen_predictions" / "30_model_inference_v2",
        "base": root / "outputs" / "pbt_msms_v1",
        "development_oof": deployment_comparison_dir(root / "outputs/pbt_msms_v1") / "19_selected_model_oof_calibrated.csv.gz",
        "index": prepared / "29_bangladesh_spectrum_index.csv.gz",
        "mgf": prepared / "29_bangladesh_dreams_input.mgf",
        "binned": prepared / "29_bangladesh_binned_features.npz",
        "transformer_tokens": prepared / "29_bangladesh_transformer_tokens.npz",
        "set_tokens": prepared / "29_bangladesh_set_transformer_tokens.npz",
        "dreams_embeddings": prepared / "29_bangladesh_dreams_embeddings.npy",
        "prepare_manifest": prepared / "29_preparation_manifest.json",
        "embed_manifest": prepared / "29_dreams_embedding_manifest.json",
    }


def _pick(folder: Path, token: str, suffix: str | None = None) -> Path:
    found = [p for p in folder.iterdir() if token.lower() in p.name.lower() and (suffix is None or p.suffix.lower() == suffix)]
    if len(found) != 1:
        raise RuntimeError(f"Expected one {token} file in {folder}, found {len(found)}")
    return found[0]


def _number(value: Any) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _read_height(path: Path) -> tuple[list[list[str]], list[str], Iterable[list[str]], Any]:
    handle = path.open("r", encoding="utf-8-sig", errors="replace", newline="")
    reader = csv.reader(handle, delimiter="\t")
    # MS-DIAL alignment matrices contain four pre-header rows:
    # Class, File type, Injection order, and Batch ID.
    metadata = [next(reader), next(reader), next(reader), next(reader)]
    header = next(reader)
    if not header or header[0] != "Alignment ID":
        handle.close()
        raise RuntimeError(f"Unexpected MS-DIAL header in {path}")
    return metadata, header, reader, handle


def _iter_msp(path: Path):
    block: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if not line.strip():
                if block:
                    yield block
                    block = []
            else:
                block.append(line)
    if block:
        yield block


def _msp_record(block: list[str]) -> dict[str, Any]:
    fields: dict[str, str] = {}
    peaks: list[tuple[float, float]] = []
    in_peaks = False
    for line in block:
        if in_peaks:
            parts = re.split(r"[\t ]+", line.strip())
            if len(parts) >= 2:
                try:
                    peaks.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip().upper()] = value.strip()
            if key.strip().upper() == "NUM PEAKS":
                in_peaks = True
    name = fields.get("NAME", "")
    match = re.search(r"(?:^|\|)ID=(\d+)(?:\||$)", name)
    if not match:
        comment = fields.get("COMMENT", "")
        match = re.search(r"(?:PEAKID|ID)=(\d+)", comment)
    return {
        "alignment_id": int(match.group(1)) if match else None,
        "name": name,
        "precursor_mz": _number(fields.get("PRECURSORMZ")),
        "adduct": fields.get("PRECURSORTYPE", ""),
        "ion_mode": fields.get("IONMODE", ""),
        "rt": _number(fields.get("RETENTIONTIME")),
        "peaks": peaks,
    }


def _mgf_ids(path: Path) -> tuple[list[int], int]:
    ids, blocks = [], 0
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if text.upper() == "BEGIN IONS":
                blocks += 1
            elif text.upper().startswith("TITLE="):
                match = re.search(r"(?:^|\|)ID=(\d+)(?:\||$)", text.split("=", 1)[1])
                if match:
                    ids.append(int(match.group(1)))
    return ids, blocks


def _clean_peaks(peaks: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw_n = len(peaks)
    if not peaks:
        return np.array([]), np.array([]), {"raw_peak_n": 0, "nonpositive_removed": 0, "relative_removed": 0, "isotope_removed": 0}
    array = np.asarray(peaks, dtype=float)
    valid = np.isfinite(array).all(axis=1) & (array[:, 0] >= 15.0) & (array[:, 0] <= MAX_MZ) & (array[:, 1] > 0)
    nonpositive_removed = int(raw_n - valid.sum())
    mz, intensity = array[valid, 0], array[valid, 1]
    if not len(intensity):
        return mz, intensity, {"raw_peak_n": raw_n, "nonpositive_removed": nonpositive_removed, "relative_removed": 0, "isotope_removed": 0}
    intensity = intensity / intensity.max()
    keep = intensity >= MIN_RELATIVE
    relative_removed = int(len(mz) - keep.sum())
    mz, intensity = mz[keep], intensity[keep]
    order = np.argsort(mz, kind="stable")
    mz, intensity = mz[order], intensity[order]
    isotope_keep = np.ones(len(mz), dtype=bool)
    for first in range(len(mz)):
        if not isotope_keep[first]:
            continue
        for second in range(first + 1, len(mz)):
            difference = mz[second] - mz[first]
            if difference > 1.01:
                break
            if 1.00085 < difference < 1.00585 and intensity[second] < intensity[first] * 0.8:
                isotope_keep[second] = False
    isotope_removed = int((~isotope_keep).sum())
    return mz[isotope_keep], intensity[isotope_keep], {
        "raw_peak_n": raw_n,
        "nonpositive_removed": nonpositive_removed,
        "relative_removed": relative_removed,
        "isotope_removed": isotope_removed,
    }


def _peak_hash(precursor: float, mode: str, mz: np.ndarray, intensity: np.ndarray) -> str:
    payload = [f"{precursor:.5f}", mode.lower()]
    payload.extend(f"{m:.5f}:{i:.6f}" for m, i in zip(mz, intensity))
    return hashlib.sha256("|".join(payload).encode()).hexdigest()


def clean_mode(root: Path, mode: str) -> dict[str, Any]:
    p = paths(root)
    source = p["source"] / mode
    out = p["clean"] / mode
    freeze = p["freeze"] / mode
    out.mkdir(parents=True, exist_ok=True)
    freeze.mkdir(parents=True, exist_ok=True)
    height = _pick(source, "Height", ".txt")
    source_msp = _pick(source, "Msp_", ".msp")
    source_mgf = _pick(source, "Mgf_", ".mgf")

    metadata, header, rows, handle = _read_height(height)
    classes, file_types = metadata[0], metadata[1]
    sample_cols = [i for i, value in enumerate(file_types) if value == "Sample"]
    blank_cols = [i for i, value in enumerate(file_types) if value == "Blank"]
    qc_cols = [i for i, value in enumerate(file_types) if value == "QC"]
    if (len(sample_cols), len(blank_cols), len(qc_cols)) != (24, 3, 5):
        handle.close()
        raise RuntimeError(f"Unexpected injection classes: {len(sample_cols)}/{len(blank_cols)}/{len(qc_cols)}")
    groups: dict[str, list[int]] = defaultdict(list)
    for column in sample_cols:
        groups[classes[column]].append(column)

    feature_rows: list[dict[str, Any]] = []
    for row in rows:
        if not row or not row[0].isdigit():
            continue
        aid = int(row[0])
        samples = np.asarray([_number(row[i]) for i in sample_cols], float)
        blanks = np.asarray([_number(row[i]) for i in blank_cols], float)
        qcs = np.asarray([_number(row[i]) for i in qc_cols], float)
        sample_max = float(samples.max(initial=0))
        blank_mean = float(blanks.mean())
        sn = _number(row[31])
        qc_detected = int((qcs > 0).sum())
        duplicate_pair = any(len(cols) >= 2 and all(_number(row[i]) > 0 for i in cols) for cols in groups.values())
        checks = {
            "blank_ratio_pass": bool(sample_max >= 5.0 * blank_mean),
            "sn_pass": bool(sn >= 10.0),
            "qc_pass": bool(qc_detected >= 3),
            "duplicate_pair_pass": bool(duplicate_pair),
        }
        strict = all(checks.values())
        adduct = row[4].strip()
        reasons = [name.replace("_pass", "") for name, passed in checks.items() if not passed]
        if strict and adduct not in ADDUCTS[mode]:
            reasons.append("adduct_excluded")
        feature_rows.append({
            "alignment_id": aid, "average_rt_min": _number(row[1]), "average_mz": _number(row[2]),
            "metabolite_name": row[3], "adduct": adduct, "msms_assigned": row[7],
            "sample_max_height": sample_max, "blank_mean_height": blank_mean,
            "sample_blank_ratio": sample_max / blank_mean if blank_mean > 0 else np.inf,
            "sn_average": sn, "qc_detected": qc_detected, "duplicate_pair_pass": duplicate_pair,
            **checks, "strict_quality_pass": strict,
            "adduct_pass": adduct in ADDUCTS[mode], "suspicious_adduct": adduct == SUSPICIOUS[mode],
            "pre_spectrum_pass": strict and adduct in ADDUCTS[mode], "feature_filter_reason": ";".join(reasons),
        })
    handle.close()
    features = pd.DataFrame(feature_rows).sort_values("alignment_id").reset_index(drop=True)
    strict_n = int(features["strict_quality_pass"].sum())
    if strict_n != EXPECTED_STRICT[mode]:
        raise RuntimeError(f"Strict filter count drift for {mode}: {strict_n} != {EXPECTED_STRICT[mode]}")

    mgf_ids, mgf_blocks = _mgf_ids(source_mgf)
    records: dict[int, dict[str, Any]] = {}
    msp_ids: list[int] = []
    for block in _iter_msp(source_msp):
        record = _msp_record(block)
        if record["alignment_id"] is None:
            raise RuntimeError("MSP record without Alignment ID")
        aid = int(record["alignment_id"])
        if aid in records:
            raise RuntimeError(f"Duplicate MSP Alignment ID: {aid}")
        records[aid] = record
        msp_ids.append(aid)
    feature_ids = set(features["alignment_id"].astype(int))
    if set(msp_ids) != feature_ids or set(mgf_ids) != feature_ids or mgf_blocks != len(feature_ids):
        raise RuntimeError("MSP/MGF/feature Alignment ID sets are not identical")

    candidates = features[features["pre_spectrum_pass"]].copy().set_index("alignment_id")
    audit_rows, cleaned = [], {}
    for aid, row in candidates.iterrows():
        rec = records[int(aid)]
        mz, intensity, stats = _clean_peaks(rec["peaks"])
        precursor = float(rec["precursor_mz"] or row["average_mz"])
        total = float(intensity.sum()) if len(intensity) else 0.0
        precursor_fraction = float(intensity[np.abs(mz - precursor) <= 1.5].sum() / total) if total > 0 else 0.0
        above_fraction = float(intensity[mz > precursor + 2.0].sum() / total) if total > 0 else 0.0
        low_peaks = len(mz) < 3
        severe_precursor = precursor_fraction >= 0.95
        severe_above = above_fraction >= 0.80
        anomaly = low_peaks or severe_precursor or severe_above or precursor <= 0
        digest = _peak_hash(precursor, mode, mz, intensity) if len(mz) else ""
        audit_rows.append({
            "alignment_id": int(aid), "precursor_mz": precursor, "rt_min": float(rec["rt"] or row["average_rt_min"]),
            "adduct": str(row["adduct"]), **stats, "cleaned_peak_n": int(len(mz)),
            "precursor_window_intensity_fraction": precursor_fraction,
            "above_precursor_intensity_fraction": above_fraction,
            "low_peak_anomaly": low_peaks, "precursor_pollution_anomaly": severe_precursor,
            "above_precursor_anomaly": severe_above, "spectrum_anomaly": anomaly,
            "spectrum_hash": digest,
        })
        cleaned[int(aid)] = {**rec, "precursor_mz": precursor, "mz": mz, "intensity": intensity, "spectrum_hash": digest}
    audit = pd.DataFrame(audit_rows).sort_values("alignment_id").reset_index(drop=True)
    duplicate_of: dict[int, int] = {}
    for _, group in audit[~audit["spectrum_anomaly"]].groupby("spectrum_hash", sort=True):
        if len(group) <= 1:
            continue
        ranked = group.merge(features[["alignment_id", "sn_average", "sample_max_height"]], on="alignment_id")
        ranked = ranked.sort_values(["sn_average", "sample_max_height", "alignment_id"], ascending=[False, False, True])
        representative = int(ranked.iloc[0]["alignment_id"])
        for aid in ranked.iloc[1:]["alignment_id"].astype(int):
            duplicate_of[aid] = representative
    audit["exact_duplicate"] = audit["alignment_id"].isin(duplicate_of)
    audit["duplicate_of_alignment_id"] = audit["alignment_id"].map(duplicate_of)
    audit["spectrum_pass"] = ~audit["spectrum_anomaly"] & ~audit["exact_duplicate"]

    final_ids = set(audit.loc[audit["spectrum_pass"], "alignment_id"].astype(int))
    features = features.merge(audit[["alignment_id", "spectrum_anomaly", "exact_duplicate", "duplicate_of_alignment_id", "spectrum_pass"]], on="alignment_id", how="left")
    features[["spectrum_anomaly", "exact_duplicate", "spectrum_pass"]] = features[["spectrum_anomaly", "exact_duplicate", "spectrum_pass"]].fillna(False)
    features["final_pass"] = features["pre_spectrum_pass"] & features["spectrum_pass"]
    features["final_filter_reason"] = features["feature_filter_reason"]
    features.loc[features["pre_spectrum_pass"] & features["spectrum_anomaly"], "final_filter_reason"] = "spectrum_anomaly"
    features.loc[features["pre_spectrum_pass"] & features["exact_duplicate"], "final_filter_reason"] = "exact_duplicate"

    table_path = out / "28_feature_screening.tsv"
    audit_path = out / "28_spectrum_audit.tsv"
    features.to_csv(table_path, sep="\t", index=False)
    audit.to_csv(audit_path, sep="\t", index=False)
    jsonl_path = out / "28_cleaned_spectra.jsonl"
    msp_path = out / "28_standardized.msp"
    mgf_path = out / "28_standardized.mgf"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as fj, msp_path.open("w", encoding="utf-8", newline="\n") as fs, mgf_path.open("w", encoding="utf-8", newline="\n") as fg:
        for aid in sorted(final_ids):
            rec = cleaned[aid]
            spectrum_id = f"Bangladesh_{mode}_Alignment_{aid}"
            payload = {"spectrum_id": spectrum_id, "alignment_id": aid, "mode": mode,
                       "precursor_mz": rec["precursor_mz"], "rt_min": rec["rt"], "adduct": rec["adduct"],
                       "mz": rec["mz"].tolist(), "intensity": rec["intensity"].tolist(), "spectrum_hash": rec["spectrum_hash"]}
            fj.write(json.dumps(payload, ensure_ascii=False) + "\n")
            fs.write(f"NAME: {spectrum_id}\nPRECURSORMZ: {rec['precursor_mz']:.8f}\nPRECURSORTYPE: {rec['adduct']}\nIONMODE: {'Positive' if mode == 'Pos' else 'Negative'}\nRETENTIONTIME: {rec['rt']:.8f}\nCOMMENT: AlignmentID={aid};Source=MS-DIAL_deconvoluted_DIA\nNum Peaks: {len(rec['mz'])}\n")
            fg.write(f"BEGIN IONS\nTITLE={spectrum_id}\nALIGNMENT_ID={aid}\nPEPMASS={rec['precursor_mz']:.8f}\nCHARGE={'2+' if '2+' in rec['adduct'] else ('2-' if '2-' in rec['adduct'] else ('1+' if mode == 'Pos' else '1-'))}\nIONMODE={'Positive' if mode == 'Pos' else 'Negative'}\nADDUCT={rec['adduct']}\nSOURCE_LIBRARY=Bangladesh_MSDIAL_DIA\n")
            for mass, abundance in zip(rec["mz"], rec["intensity"]):
                fs.write(f"{mass:.8f}\t{abundance:.8f}\n")
                fg.write(f"{mass:.8f} {abundance:.8f}\n")
            fs.write("\n")
            fg.write("END IONS\n\n")

    manifest = {
        "version": VERSION, "stage": 28, "mode": mode, "status": "FROZEN_FOR_PBT_INPUT_PREPARATION",
        "generated_at_utc": utc_now(),
        "parameters": {"blank_fold": 5.0, "sn_min": 10.0, "qc_min_of_5": 3, "sample_duplicate_pair_required": True,
                       "adduct_whitelist": sorted(ADDUCTS[mode]), "min_mz": 15.0, "max_mz": MAX_MZ,
                       "min_relative_intensity": MIN_RELATIVE, "min_cleaned_peaks": 3,
                       "precursor_pollution_flag_fraction": 0.95, "above_precursor_flag_fraction": 0.80,
                       "deduplication": "exact standardized spectrum hash; retain highest S/N then intensity"},
        "counts": {"all_features": len(features), "strict_quality_pass": strict_n,
                   "adduct_pass_after_strict": int(features["pre_spectrum_pass"].sum()),
                   "spectrum_anomalies": int(audit["spectrum_anomaly"].sum()),
                   "exact_duplicates": int(audit["exact_duplicate"].sum()), "final_spectra": len(final_ids)},
        "id_integrity": {"feature_ids": len(feature_ids), "msp_ids": len(set(msp_ids)), "mgf_ids": len(set(mgf_ids)), "all_equal": True},
        "source_files": {x.name: {"bytes": x.stat().st_size, "sha256": sha256_file(x)} for x in (height, source_msp, source_mgf)},
        "artifacts": {x.name: {"bytes": x.stat().st_size, "sha256": sha256_file(x)} for x in (table_path, audit_path, jsonl_path, msp_path, mgf_path)},
    }
    write_json(out / "28_run_manifest.json", manifest)
    for artifact in (table_path, audit_path, jsonl_path, msp_path, mgf_path, out / "28_run_manifest.json"):
        target = freeze / artifact.name
        target.write_bytes(artifact.read_bytes())
    write_json(freeze / "28_freeze_status.json", {"status": "FROZEN_FOR_PBT_INPUT_PREPARATION", "mode": mode, "final_spectra": len(final_ids), "source_manifest_sha256": sha256_file(out / "28_run_manifest.json")})
    return manifest


def run_cleaning(root: Path) -> dict[str, Any]:
    manifests = {mode: clean_mode(root, mode) for mode in ("Pos", "Neg")}
    p = paths(root)
    summary = {"version": VERSION, "stage": 28, "status": "FROZEN_FOR_PBT_INPUT_PREPARATION", "generated_at_utc": utc_now(),
               "modes": {mode: data["counts"] for mode, data in manifests.items()}}
    write_json(p["freeze"] / "28_freeze_manifest.json", summary)
    return summary


def _load_cleaned(p: dict[str, Path]) -> list[dict[str, Any]]:
    records = []
    for mode in ("Pos", "Neg"):
        status = json.loads((p["freeze"] / mode / "28_freeze_status.json").read_text(encoding="utf-8"))
        if status.get("status") != "FROZEN_FOR_PBT_INPUT_PREPARATION":
            raise RuntimeError(f"Notebook 28 freeze is not ready for {mode}")
        with (p["freeze"] / mode / "28_cleaned_spectra.jsonl").open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def run_prepare(root: Path) -> dict[str, Any]:
    p = paths(root)
    p["prepared"].mkdir(parents=True, exist_ok=True)
    records = _load_cleaned(p)
    index = pd.DataFrame([{k: rec[k] for k in ("spectrum_id", "alignment_id", "mode", "precursor_mz", "rt_min", "adduct", "spectrum_hash")} for rec in records])
    if index["spectrum_id"].duplicated().any() or index["spectrum_hash"].duplicated().any():
        raise RuntimeError("Notebook 28 final spectra are not unique")
    index["feature_row"] = np.arange(len(index), dtype=int)
    index["inchikey_2d"] = index["spectrum_id"]  # one unknown feature is one deployment unit
    index["ion_mode"] = np.where(index["mode"].eq("Pos"), "Positive", "Negative")
    index.to_csv(p["index"], index=False, compression="gzip")

    n = len(index)
    fr, fc, fd, nr, nc, nd = [], [], [], [], [], []
    bmeta = np.zeros((n, 6), np.float32)
    peak_tokens = np.zeros((n, MAX_PEAKS, 4), np.float16)
    peak_padding = np.ones((n, MAX_PEAKS), bool)
    token_meta = np.zeros((n, 8), np.float32)
    adduct_codes = np.zeros(n, np.int16); instrument_codes = np.zeros(n, np.int16); source_codes = np.zeros(n, np.int16)
    set_features = np.zeros((n, MAX_PEAKS, 3), np.float16); set_padding = np.ones((n, MAX_PEAKS), bool); set_meta = np.zeros((n, 6), np.float32)
    vocab = json.loads((p["base"] / "transformer_data_v1" / "transformer_categorical_vocab.json").read_text(encoding="utf-8"))
    maps = {key: {value: i for i, value in enumerate(vocab[key])} for key in ("adduct", "instrument_category", "source_library")}
    with p["mgf"].open("w", encoding="utf-8", newline="\n") as mgf:
        for row, rec in zip(index.itertuples(index=False), records):
            mz = np.asarray(rec["mz"], float); intensity = np.asarray(rec["intensity"], float)
            precursor = float(rec["precursor_mz"]); polarity = 1.0 if rec["mode"] == "Pos" else -1.0; ce_value = np.nan
            root_intensity = np.sqrt(intensity / intensity.max())
            for column, value in ce.binned_max(mz, root_intensity).items(): fr.append(row.feature_row); fc.append(column); fd.append(value)
            for column, value in ce.binned_max(precursor - mz, root_intensity).items(): nr.append(row.feature_row); nc.append(column); nd.append(value)
            bmeta[row.feature_row] = [precursor, polarity, 0, 1, math.log1p(len(mz)), 0]
            token, pad, retained = ce.transformer_peak_encoding(mz, intensity, precursor)
            peak_tokens[row.feature_row], peak_padding[row.feature_row] = token, pad
            token_meta[row.feature_row] = ce.metadata8(precursor, polarity, ce_value, retained, "UNKNOWN")
            adduct_codes[row.feature_row] = maps["adduct"].get(rec["adduct"], maps["adduct"].get("OTHER", 0))
            instrument_codes[row.feature_row] = maps["instrument_category"].get("UNKNOWN", maps["instrument_category"].get("OTHER", 0))
            source_codes[row.feature_row] = maps["source_library"].get("UNKNOWN", 0)
            sf, sp, sm = ce.set_transformer_encoding(mz, intensity, precursor, polarity, ce_value)
            set_features[row.feature_row], set_padding[row.feature_row], set_meta[row.feature_row] = sf, sp, sm
            charge = "2+" if "2+" in rec["adduct"] else ("2-" if "2-" in rec["adduct"] else ("1+" if rec["mode"] == "Pos" else "1-"))
            mgf.write(f"BEGIN IONS\nTITLE={rec['spectrum_id']}\nFEATURE_ID={row.feature_row + 1}\nMSLEVEL=2\nPEPMASS={precursor:.8f}\nCHARGE={charge}\nIONMODE={row.ion_mode}\nADDUCT={rec['adduct']}\nSOURCE_LIBRARY=Bangladesh_MSDIAL_DIA\n")
            for mass, abundance in zip(mz, intensity): mgf.write(f"{mass:.8f} {abundance:.8f}\n")
            mgf.write("END IONS\n\n")
    xf = sparse.csr_matrix((fd, (fr, fc)), shape=(n, ce.N_BINS), dtype=np.float32)
    xn = sparse.csr_matrix((nd, (nr, nc)), shape=(n, ce.N_BINS), dtype=np.float32)
    np.savez_compressed(p["binned"], fragment_data=xf.data, fragment_indices=xf.indices, fragment_indptr=xf.indptr,
                        neutral_data=xn.data, neutral_indices=xn.indices, neutral_indptr=xn.indptr,
                        shape=np.asarray(xf.shape), metadata=bmeta)
    np.savez_compressed(p["transformer_tokens"], peak_tokens=peak_tokens, padding_mask=peak_padding, metadata_numeric=token_meta,
                        adduct_code=adduct_codes, instrument_code=instrument_codes, source_code=source_codes)
    np.savez_compressed(p["set_tokens"], features=set_features, padding=set_padding, metadata=set_meta)
    manifest = {"version": VERSION, "stage": 29, "status": "READY_FOR_DREAMS_EMBEDDING", "generated_at_utc": utc_now(),
                "n_spectra": n, "mode_counts": index["mode"].value_counts().to_dict(),
                "contracts": {"binned": "Notebook20 exact feature builder", "transformer": "Notebook18 exact 4-channel token schema",
                              "set_transformer": "Notebook06 exact 3-channel token schema", "dreams": "MGF for DreaMS embedding"},
                "artifacts": {key: {"sha256": sha256_file(p[key]), "bytes": p[key].stat().st_size} for key in ("index", "mgf", "binned", "transformer_tokens", "set_tokens")}}
    write_json(p["prepare_manifest"], manifest)
    return manifest


def run_dreams_embedding(root: Path, batch_size: int = 32) -> dict[str, Any]:
    p = paths(root)
    prep = json.loads(p["prepare_manifest"].read_text(encoding="utf-8"))
    if prep.get("status") != "READY_FOR_DREAMS_EMBEDDING" or sha256_file(p["mgf"]) != prep["artifacts"]["mgf"]["sha256"]:
        raise RuntimeError("Notebook 29 preparation is not intact")
    from pbt_msms.dreams_mvp import configure_writable_runtime_cache
    configure_writable_runtime_cache()
    import torch
    from dreams.api import dreams_embeddings
    if not torch.cuda.is_available():
        raise RuntimeError("DreaMS embedding requires the env_dreams kernel with CUDA")
    embeddings = np.asarray(dreams_embeddings(p["mgf"], batch_size=batch_size, progress_bar=True, store_embs=False), np.float32)
    if embeddings.shape != (prep["n_spectra"], ce.EMBEDDING_DIM) or not np.isfinite(embeddings).all():
        raise RuntimeError(f"Unexpected DreaMS embeddings: {embeddings.shape}")
    np.save(p["dreams_embeddings"], embeddings, allow_pickle=False)
    manifest = {"version": VERSION, "stage": 29, "status": "READY_FOR_FROZEN_MODEL_INFERENCE", "generated_at_utc": utc_now(),
                "shape": list(embeddings.shape), "batch_size": batch_size, "input_mgf_sha256": sha256_file(p["mgf"]),
                "embedding_sha256": sha256_file(p["dreams_embeddings"]), "gpu": torch.cuda.get_device_name(0)}
    write_json(p["embed_manifest"], manifest)
    return manifest


def _prediction_path_map(root: Path) -> dict[str, Path]:
    p = paths(root)
    return {"base": p["base"], "binned": p["binned"], "transformer_tokens": p["transformer_tokens"],
            "set_tokens": p["set_tokens"], "dreams_embeddings": p["dreams_embeddings"]}


def run_prediction(root: Path) -> dict[str, Any]:
    import torch
    p = paths(root); p["prediction"].mkdir(parents=True, exist_ok=True)
    prep = json.loads(p["prepare_manifest"].read_text(encoding="utf-8")); embed = json.loads(p["embed_manifest"].read_text(encoding="utf-8"))
    if prep.get("status") != "READY_FOR_DREAMS_EMBEDDING" or embed.get("status") != "READY_FOR_FROZEN_MODEL_INFERENCE":
        raise RuntimeError("Notebook 29 is not ready")
    if sha256_file(p["dreams_embeddings"]) != embed["embedding_sha256"]:
        raise RuntimeError("DreaMS embeddings changed after freezing")
    index = pd.read_csv(p["index"])
    registry = json.loads((deployment_comparison_dir(p["base"]) / "19_final_model_registry.json").read_text(encoding="utf-8"))
    calibration = json.loads((deployment_comparison_dir(p["base"]) / "19_deployment_calibration.json").read_text(encoding="utf-8"))
    rejection = json.loads((deployment_comparison_dir(p["base"]) / "19_rejection_thresholds.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_paths = ce.paths
    ce.paths = lambda ignored: _prediction_path_map(root)
    try:
        frames = [ce.binned_predictions(root, index), ce.dreams_predictions(root, index),
                  ce.set_transformer_predictions(root, index, device), ce.transformer_predictions(root, index, device)]
    finally:
        ce.paths = original_paths
    replicas = pd.concat(frames, ignore_index=True)
    if not np.isfinite(replicas["y_score"].to_numpy(float)).all():
        raise RuntimeError("Non-finite model prediction")
    replicas = replicas.rename(columns={"inchikey_2d": "spectrum_id"})
    replicas.to_csv(p["prediction"] / "30_all_candidate_replica_predictions.csv.gz", index=False, compression="gzip")
    candidate = replicas.groupby(["pipeline", "task", "kind", "spectrum_id"], as_index=False)["y_score"].mean()
    expected_pipelines = set(ce.PIPELINES)
    if set(candidate["pipeline"]) != expected_pipelines:
        raise RuntimeError(f"Candidate pipelines incomplete: {sorted(set(candidate['pipeline']))}")
    candidate.to_csv(p["prediction"] / "30_all_candidate_predictions.csv.gz", index=False, compression="gzip")
    disagreement = candidate.pivot_table(index=["task", "kind", "spectrum_id"], columns="pipeline", values="y_score")
    if disagreement[list(ce.PIPELINES)].isna().any().any():
        raise RuntimeError("Candidate prediction matrix incomplete")
    disagreement = disagreement[list(ce.PIPELINES)].std(axis=1, ddof=0).rename("direct_model_disagreement").reset_index()
    transformer = replicas[replicas["pipeline"] == "Transformer B real pretrain"]
    seed_sd = transformer.groupby(["task", "kind", "spectrum_id", "outer_fold"])["y_score"].agg(lambda x: float(np.std(np.asarray(x), ddof=0))).rename("fold_three_seed_sd").reset_index()
    seed_sd = seed_sd.groupby(["task", "kind", "spectrum_id"], as_index=False)["fold_three_seed_sd"].median().rename(columns={"fold_three_seed_sd": "three_seed_prediction_sd"})
    selected = []
    for record in registry["models"]:
        task, pipeline = record["task"], record["selected_pipeline"]
        part = candidate[(candidate["task"] == task) & (candidate["pipeline"] == pipeline)].copy()
        if pipeline.startswith("Transformer"):
            part = part.merge(seed_sd[seed_sd["task"] == task][["spectrum_id", "three_seed_prediction_sd"]], on="spectrum_id", validate="one_to_one")
            part["uncertainty_score"] = part["three_seed_prediction_sd"]; part["uncertainty_method"] = "three_seed_prediction_sd"
        else:
            part = part.merge(disagreement[disagreement["task"] == task][["spectrum_id", "direct_model_disagreement"]], on="spectrum_id", validate="one_to_one")
            part["uncertainty_score"] = part["direct_model_disagreement"]; part["uncertainty_method"] = "eight_direct_candidate_model_disagreement"
        cal = calibration["endpoints"][task]
        if cal["calibration"] != "identity":
            raise RuntimeError("Frozen calibration is not identity; implementation update required")
        part["selected_pipeline"] = pipeline; part["y_score_calibrated"] = part["y_score"]
        part["rejection_threshold"] = float(rejection["endpoints"][task]["primary_threshold"])
        part["retained"] = part["uncertainty_score"] <= part["rejection_threshold"]
        if record["kind"] == "classification":
            part["screening_threshold"] = float(cal["screening_threshold"])
            part["y_prediction"] = (part["y_score_calibrated"] >= part["screening_threshold"]).astype(int)
            part["interval_low"] = np.nan; part["interval_high"] = np.nan
        else:
            half = float(cal["q90_half_width"]); part["screening_threshold"] = np.nan
            part["y_prediction"] = part["y_score_calibrated"]; part["interval_low"] = part["y_score_calibrated"] - half; part["interval_high"] = part["y_score_calibrated"] + half
        selected.append(part)
    final = pd.concat(selected, ignore_index=True).merge(index[["spectrum_id", "alignment_id", "mode", "precursor_mz", "rt_min", "adduct"]], on="spectrum_id", validate="many_to_one")
    final.to_csv(p["prediction"] / "30_final_selected_model_predictions.csv.gz", index=False, compression="gzip")
    for mode in ("Pos", "Neg"):
        final[final["mode"] == mode].to_csv(p["prediction"] / f"30_Bangladesh_DIA_{mode}_PBT_predictions.csv.gz", index=False, compression="gzip")
    manifest = {"version": VERSION, "stage": 30, "status": "PREDICTIONS_FROZEN", "generated_at_utc": utc_now(),
                "device": str(device), "n_spectra": len(index), "n_prediction_rows": len(final),
                "method_correction": "DreaMS P fold probabilities use the same Development-only Platt maps as internal evaluation",
                "dreams_p_calibration_scope": "fold-specific Platt calibration fitted on Development OOF predictions only",
                "retained_by_task": final.groupby("task")["retained"].agg(["sum", "count"]).to_dict(orient="index"),
                "selected_pipelines": {x["task"]: x["selected_pipeline"] for x in registry["models"]}}
    write_json(p["prediction"] / "30_run_manifest.json", manifest)
    return manifest


NEUTRAL_MASS_RULES = {
    "[M+H]+": lambda mz: mz - 1.007276466621,
    "[M+NH4]+": lambda mz: mz - 18.033823,
    "[M+Na]+": lambda mz: mz - 22.989218,
    "[M+K]+": lambda mz: mz - 38.963158,
    "[M+2H]2+": lambda mz: mz * 2.0 - 2.0 * 1.007276466621,
    "[M-H]-": lambda mz: mz + 1.007276466621,
    "[M+Cl]-": lambda mz: mz - 34.969401261,
    "[M-2H]2-": lambda mz: mz * 2.0 + 2.0 * 1.007276466621,
}


def _abundance_summary(root: Path, mode: str, wanted: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = paths(root)
    height = _pick(p["source"] / mode, "Height", ".txt")
    metadata, header, rows, handle = _read_height(height)
    classes, file_types = metadata[0], metadata[1]
    sample_cols = [i for i, value in enumerate(file_types) if value == "Sample"]
    blank_cols = [i for i, value in enumerate(file_types) if value == "Blank"]
    qc_cols = [i for i, value in enumerate(file_types) if value == "QC"]
    groups: dict[str, list[int]] = defaultdict(list)
    for column in sample_cols:
        groups[classes[column]].append(column)
    summaries, sample_rows = [], []
    for row in rows:
        if not row or not row[0].isdigit() or int(row[0]) not in wanted:
            continue
        aid = int(row[0])
        sample = np.asarray([_number(row[i]) for i in sample_cols], float)
        blank = np.asarray([_number(row[i]) for i in blank_cols], float)
        qc = np.asarray([_number(row[i]) for i in qc_cols], float)
        blank_mean = float(blank.mean())
        corrected = np.clip(sample - blank_mean, 0, None)
        qc_mean = float(qc.mean())
        group_both = sum(len(cols) >= 2 and all(_number(row[i]) > 0 for i in cols) for cols in groups.values())
        summaries.append({
            "mode": mode, "alignment_id": aid, "sample_detected_n": int((sample > 0).sum()),
            "sample_detection_fraction": float((sample > 0).mean()), "sample_group_n": len(groups),
            "sample_groups_both_detected_n": int(group_both), "sample_group_both_fraction": float(group_both / len(groups)),
            "sample_max_height": float(sample.max()), "sample_median_height": float(np.median(sample)),
            "blank_mean_height": blank_mean, "blank_corrected_max_height": float(corrected.max()),
            "blank_corrected_median_height": float(np.median(corrected)), "qc_mean_height": qc_mean,
            "qc_cv": float(qc.std(ddof=1) / qc_mean) if qc_mean > 0 and len(qc) > 1 else np.nan,
        })
        for column, value, corrected_value in zip(sample_cols, sample, corrected):
            sample_rows.append({"mode": mode, "alignment_id": aid, "sample_name": header[column],
                                "sample_group": classes[column], "height": float(value),
                                "blank_corrected_height": float(corrected_value)})
    handle.close()
    result = pd.DataFrame(summaries)
    if set(result["alignment_id"].astype(int)) != wanted:
        raise RuntimeError(f"Abundance linkage incomplete for {mode}")
    return result, pd.DataFrame(sample_rows)


def _percentile(series: pd.Series, ascending: bool = True) -> pd.Series:
    return series.rank(method="average", pct=True, ascending=ascending).astype(float)


def _neutral_clusters(frame: pd.DataFrame, ppm: float = 10.0, rt_tolerance: float = 0.15) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result["neutral_mass"] = [NEUTRAL_MASS_RULES[a](float(m)) for a, m in zip(result["adduct"], result["precursor_mz"])]
    result["cross_polarity_cluster"] = ""
    pos = result[result["mode"] == "Pos"].sort_values("neutral_mass")
    neg = result[result["mode"] == "Neg"].sort_values("neutral_mass")
    neg_indices = neg.index.to_numpy(); neg_mass = neg["neutral_mass"].to_numpy(float)
    parent = list(range(len(result)))
    rt_minimum = result["rt_min"].astype(float).to_numpy(copy=True)
    rt_maximum = rt_minimum.copy()
    mass_minimum = result["neutral_mass"].astype(float).to_numpy(copy=True)
    mass_maximum = mass_minimum.copy()
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        combined_rt_min = min(rt_minimum[ra], rt_minimum[rb])
        combined_rt_max = max(rt_maximum[ra], rt_maximum[rb])
        combined_mass_min = min(mass_minimum[ra], mass_minimum[rb])
        combined_mass_max = max(mass_maximum[ra], mass_maximum[rb])
        reference_mass = max((combined_mass_min + combined_mass_max) / 2.0, 1e-12)
        if combined_rt_max - combined_rt_min > rt_tolerance + 1e-12:
            return
        if (combined_mass_max - combined_mass_min) / reference_mass * 1e6 > ppm + 1e-12:
            return
        if ra > rb:
            ra, rb = rb, ra
        parent[rb] = ra
        rt_minimum[ra], rt_maximum[ra] = combined_rt_min, combined_rt_max
        mass_minimum[ra], mass_maximum[ra] = combined_mass_min, combined_mass_max
    edges = []
    for pidx, prow in pos.iterrows():
        mass = float(prow.neutral_mass); tolerance = mass * ppm * 1e-6
        left = np.searchsorted(neg_mass, mass - tolerance, side="left")
        right = np.searchsorted(neg_mass, mass + tolerance, side="right")
        for nidx in neg_indices[left:right]:
            if abs(float(result.at[nidx, "rt_min"]) - float(prow.rt_min)) <= rt_tolerance:
                mass_error = abs(float(result.at[nidx, "neutral_mass"]) - mass) / max(mass, 1e-12) * 1e6
                rt_error = abs(float(result.at[nidx, "rt_min"]) - float(prow.rt_min))
                edges.append(((mass_error / ppm) ** 2 + (rt_error / rt_tolerance) ** 2, int(pidx), int(nidx)))
    for _, pidx, nidx in sorted(edges):
        union(pidx, nidx)
    roots = [find(i) for i in range(len(result))]
    root_order = {root: n + 1 for n, root in enumerate(sorted(set(roots)))}
    result["cross_polarity_cluster"] = [f"BD_DIA_{root_order[root]:06d}" for root in roots]
    result["cross_polarity_member_n"] = result.groupby("cross_polarity_cluster")["spectrum_id"].transform("size")
    result["cross_polarity_supported"] = result.groupby("cross_polarity_cluster")["mode"].transform("nunique").eq(2)
    result["cross_polarity_rt_span_min"] = result.groupby("cross_polarity_cluster")["rt_min"].transform(lambda x: float(x.max() - x.min()))
    result["cross_polarity_neutral_mass_span_ppm"] = result.groupby("cross_polarity_cluster")["neutral_mass"].transform(
        lambda x: float((x.max() - x.min()) / max((x.max() + x.min()) / 2.0, 1e-12) * 1e6)
    )
    if (result["cross_polarity_rt_span_min"] > rt_tolerance + 1e-12).any():
        raise RuntimeError("Cross-polarity cluster exceeds the full-cluster RT span limit")
    if (result["cross_polarity_neutral_mass_span_ppm"] > ppm + 1e-12).any():
        raise RuntimeError("Cross-polarity cluster exceeds the full-cluster neutral-mass span limit")
    return result


def run_ranking(root: Path) -> dict[str, Any]:
    p = paths(root)
    out = p["case"] / "06_revealed_evaluation" / "31_feature_prioritization_v2"
    out.mkdir(parents=True, exist_ok=True)
    prediction_manifest = json.loads((p["prediction"] / "30_run_manifest.json").read_text(encoding="utf-8"))
    if prediction_manifest.get("status") != "PREDICTIONS_FROZEN":
        raise RuntimeError("Notebook 30 predictions are not frozen")
    pred = pd.read_csv(p["prediction"] / "30_final_selected_model_predictions.csv.gz")
    if pred.duplicated(["spectrum_id", "task"]).any() or pred.groupby("spectrum_id")["task"].nunique().ne(5).any():
        raise RuntimeError("Prediction table is not one row per spectrum and task")
    meta = pred[["spectrum_id", "alignment_id", "mode", "precursor_mz", "rt_min", "adduct"]].drop_duplicates()
    score = pred.pivot(index="spectrum_id", columns="task", values="y_score_calibrated")
    retained = pred.pivot(index="spectrum_id", columns="task", values="retained").add_prefix("retained__")
    uncertainty = pred.pivot(index="spectrum_id", columns="task", values="uncertainty_score").add_prefix("uncertainty__")
    wide = meta.merge(score.reset_index(), on="spectrum_id", validate="one_to_one")
    wide = wide.merge(retained.reset_index(), on="spectrum_id", validate="one_to_one").merge(uncertainty.reset_index(), on="spectrum_id", validate="one_to_one")
    abundance_parts, sample_parts = [], []
    for mode in ("Pos", "Neg"):
        wanted = set(wide.loc[wide["mode"] == mode, "alignment_id"].astype(int))
        abundance, sample = _abundance_summary(root, mode, wanted)
        abundance_parts.append(abundance); sample_parts.append(sample)
    abundance = pd.concat(abundance_parts, ignore_index=True)
    sample_long = pd.concat(sample_parts, ignore_index=True)
    wide = wide.merge(abundance, on=["mode", "alignment_id"], validate="one_to_one")

    task_names = list(ce.TASKS)
    if tuple(task_names) != HAZARD_TASK_ORDER:
        raise RuntimeError("Endpoint order differs from the shared canonical hazard definition")
    wide["retained_endpoint_n"] = wide[[f"retained__{task}" for task in task_names]].sum(axis=1).astype(int)
    wide["all_five_endpoints_retained"] = wide["retained_endpoint_n"].eq(5)
    references = load_development_references(p["development_oof"])
    endpoint_hazard, toxicity, hazard = harmonized_full_hazard(
        {task: wide[task].to_numpy(float) for task in ce.TASKS}, references
    )
    component_columns = {
        P_TASK: "p_development_anchored_hazard_percentile",
        B_TASK: "b_development_anchored_hazard_percentile",
        EC50_TASK: "t_ec50_development_anchored_hazard_percentile",
        LC50_TASK: "t_lc50_development_anchored_hazard_percentile",
        NOEC_TASK: "t_noec_development_anchored_hazard_percentile",
    }
    for task, column in component_columns.items():
        wide[column] = endpoint_hazard[task]
    wide["persistence_component"] = endpoint_hazard[P_TASK]
    wide["bioaccumulation_component"] = endpoint_hazard[B_TASK]
    wide["toxicity_component"] = toxicity
    wide["hazard_component"] = hazard
    wide["occurrence_component"] = wide[["sample_detection_fraction", "sample_group_both_fraction"]].mean(axis=1)
    wide["abundance_component"] = _percentile(np.log1p(wide["blank_corrected_max_height"]), ascending=True)
    wide["exposure_component"] = wide[["occurrence_component", "abundance_component"]].mean(axis=1)
    wide["confidence_component"] = wide["retained_endpoint_n"] / 5.0
    wide["priority_score_primary"] = (0.70 * wide["hazard_component"] + 0.30 * wide["exposure_component"]) * wide["confidence_component"]
    wide["priority_score_hazard_heavy"] = (0.85 * wide["hazard_component"] + 0.15 * wide["exposure_component"]) * wide["confidence_component"]
    wide["priority_score_exposure_heavy"] = (0.55 * wide["hazard_component"] + 0.45 * wide["exposure_component"]) * wide["confidence_component"]
    wide = _neutral_clusters(wide)
    wide["feature_rank"] = wide["priority_score_primary"].rank(method="first", ascending=False).astype(int)
    wide["cluster_rank_within"] = wide.groupby("cross_polarity_cluster")["priority_score_primary"].rank(method="first", ascending=False).astype(int)
    cluster = wide.sort_values(["priority_score_primary", "spectrum_id"], ascending=[False, True]).groupby("cross_polarity_cluster", as_index=False).first()
    members = wide.groupby("cross_polarity_cluster").agg(cluster_member_n=("spectrum_id", "size"), modes=("mode", lambda x: ";".join(sorted(set(x)))), member_spectrum_ids=("spectrum_id", lambda x: ";".join(sorted(x)))).reset_index()
    cluster = cluster.drop(columns=["cross_polarity_member_n", "cross_polarity_supported"], errors="ignore").merge(members, on="cross_polarity_cluster", validate="one_to_one")
    cluster["cluster_rank"] = cluster["priority_score_primary"].rank(method="first", ascending=False).astype(int)
    cluster = cluster.sort_values("cluster_rank")
    wide = wide.sort_values("feature_rank")
    wide.to_csv(out / "31_feature_priority_ranking.csv.gz", index=False, compression="gzip")
    cluster.to_csv(out / "31_cross_polarity_cluster_ranking.csv.gz", index=False, compression="gzip")
    sample_long.to_csv(out / "31_selected_feature_sample_abundance_long.csv.gz", index=False, compression="gzip")
    wide.head(100).to_csv(out / "31_top100_features.csv", index=False)
    cluster.head(100).to_csv(out / "31_top100_clusters.csv", index=False)
    sensitivity = pd.DataFrame({
        "comparison": ["hazard_heavy", "exposure_heavy"],
        "spearman_vs_primary": [wide["priority_score_primary"].corr(wide["priority_score_hazard_heavy"], method="spearman"),
                                wide["priority_score_primary"].corr(wide["priority_score_exposure_heavy"], method="spearman")],
        "top100_overlap": [len(set(wide.nsmallest(100, "feature_rank")["spectrum_id"]) & set(wide.nlargest(100, "priority_score_hazard_heavy")["spectrum_id"])),
                           len(set(wide.nsmallest(100, "feature_rank")["spectrum_id"]) & set(wide.nlargest(100, "priority_score_exposure_heavy")["spectrum_id"]))],
    })
    sensitivity.to_csv(out / "31_ranking_sensitivity.csv", index=False)
    manifest = {"version": VERSION, "stage": 31, "status": "FEATURE_PRIORITIZATION_FROZEN", "generated_at_utc": utc_now(),
                "scientific_scope": "exploratory unknown-feature prioritization; not chemical identification or regulatory PBT classification",
                "n_features": len(wide), "n_cross_polarity_clusters": len(cluster),
                "cross_polarity_supported_clusters": int(cluster["modes"].str.contains("Pos;Neg|Neg;Pos", regex=True).sum()),
                "all_five_endpoints_retained": int(wide["all_five_endpoints_retained"].sum()),
                "hazard_formula_id": CANONICAL_HAZARD_FORMULA,
                "hazard_formula": "geometric mean of Development-OOF-referenced P, B and max inverse-toxicity percentiles",
                "priority_formula": "(0.70*harmonized geometric P/B/T hazard+0.30*mean(occurrence, abundance percentile))*retained_endpoint_fraction",
                "development_oof_sha256": sha256_file(p["development_oof"]),
                "cross_polarity_rule": "adduct-derived neutral mass full-cluster span <=10 ppm and RT full-cluster span <=0.15 min; Pos-Neg links only",
                "sensitivity": sensitivity.to_dict(orient="records")}
    write_json(out / "31_run_manifest.json", manifest)
    return manifest


def _stable_hash_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Version-independent deterministic pseudo-random sample by cluster ID."""
    if len(frame) < n:
        raise RuntimeError(f"Sampling pool contains {len(frame)} clusters, fewer than requested {n}")
    sampled = frame.copy()
    sampled["_sample_key"] = sampled["cross_polarity_cluster"].map(
        lambda value: hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()
    )
    return sampled.sort_values(["_sample_key", "cross_polarity_cluster"]).head(n).drop(columns="_sample_key")


def run_identity_reveal_cohort(
    root: Path,
    n_per_cohort: int = 100,
    seed: int = 20260920,
    middle_bounds: tuple[float, float] = (0.45, 0.55),
    low_bounds: tuple[float, float] = (0.90, 1.00),
    high_risk_quantile: float = 0.80,
    high_uncertainty_quantile: float = 0.80,
) -> dict[str, Any]:
    """Freeze identity-blind cluster cohorts before any library or structure search.

    Selection is performed at the cross-polarity-cluster level to avoid counting
    multiple ion/adduct representations of the same candidate as independent items.
    """
    p = paths(root)
    rank_dir = p["case"] / "06_revealed_evaluation" / "31_feature_prioritization_v2"
    out = p["case"] / "06_revealed_evaluation" / "32_identity_reveal_cohort_v2"
    out.mkdir(parents=True, exist_ok=True)
    rank_manifest_path = rank_dir / "31_run_manifest.json"
    feature_path = rank_dir / "31_feature_priority_ranking.csv.gz"
    cluster_path = rank_dir / "31_cross_polarity_cluster_ranking.csv.gz"
    rank_manifest = json.loads(rank_manifest_path.read_text(encoding="utf-8"))
    if rank_manifest.get("status") != "FEATURE_PRIORITIZATION_FROZEN":
        raise RuntimeError("Notebook 31 feature prioritization is not frozen")
    if not (0 <= middle_bounds[0] < middle_bounds[1] <= 1):
        raise ValueError("middle_bounds must satisfy 0 <= lower < upper <= 1")
    if not (0 <= low_bounds[0] < low_bounds[1] <= 1):
        raise ValueError("low_bounds must satisfy 0 <= lower < upper <= 1")

    features = pd.read_csv(feature_path)
    clusters = pd.read_csv(cluster_path).sort_values("cluster_rank").reset_index(drop=True)
    if clusters["cross_polarity_cluster"].duplicated().any():
        raise RuntimeError("Notebook 31 cluster identifiers are not unique")
    if set(features["cross_polarity_cluster"]) != set(clusters["cross_polarity_cluster"]):
        raise RuntimeError("Notebook 31 feature and cluster tables contain different cluster IDs")
    if not clusters["cluster_rank"].tolist() == list(range(1, len(clusters) + 1)):
        raise RuntimeError("Notebook 31 cluster rank is not a complete 1..N sequence")

    uncertainty_columns = sorted(column for column in features if column.startswith("uncertainty__"))
    if len(uncertainty_columns) != 5:
        raise RuntimeError(f"Expected five uncertainty columns, found {uncertainty_columns}")
    feature_uncertainty = features[uncertainty_columns].mean(axis=1)
    uncertainty = (
        features.assign(_feature_uncertainty_mean=feature_uncertainty)
        .groupby("cross_polarity_cluster", as_index=False)
        .agg(
            cluster_uncertainty_mean=("_feature_uncertainty_mean", "mean"),
            cluster_uncertainty_max=("_feature_uncertainty_mean", "max"),
        )
    )
    clusters = clusters.merge(uncertainty, on="cross_polarity_cluster", validate="one_to_one")
    clusters["rank_fraction"] = clusters["cluster_rank"] / len(clusters)
    hazard_threshold = float(clusters["hazard_component"].quantile(high_risk_quantile))
    uncertainty_threshold = float(clusters["cluster_uncertainty_mean"].quantile(high_uncertainty_quantile))

    chosen: list[pd.DataFrame] = []
    used: set[str] = set()

    def register(frame: pd.DataFrame, cohort: str, reason: str) -> None:
        nonlocal used
        frame = frame.copy()
        overlap = used.intersection(frame["cross_polarity_cluster"])
        if overlap:
            raise RuntimeError(f"Cohort overlap detected for {cohort}: {sorted(overlap)[:3]}")
        frame["cohort"] = cohort
        frame["selection_order"] = np.arange(1, len(frame) + 1)
        frame["selection_reason"] = reason
        chosen.append(frame)
        used.update(frame["cross_polarity_cluster"])

    top = clusters.nsmallest(n_per_cohort, "cluster_rank")
    register(top, "TOP_PRIORITY", f"cluster_rank 1..{n_per_cohort}")

    high_uncertainty_pool = clusters[
        (~clusters["cross_polarity_cluster"].isin(used))
        & (clusters["hazard_component"] >= hazard_threshold)
        & (clusters["cluster_uncertainty_mean"] >= uncertainty_threshold)
    ].copy()
    high_uncertainty_pool["_tie_key"] = high_uncertainty_pool["cross_polarity_cluster"].map(
        lambda value: hashlib.sha256(f"{seed}|uncertainty|{value}".encode("utf-8")).hexdigest()
    )
    high_uncertainty = high_uncertainty_pool.sort_values(
        ["cluster_uncertainty_mean", "hazard_component", "_tie_key"],
        ascending=[False, False, True],
    ).head(n_per_cohort).drop(columns="_tie_key")
    if len(high_uncertainty) != n_per_cohort:
        raise RuntimeError("Insufficient non-overlapping high-risk/high-uncertainty clusters")
    register(
        high_uncertainty,
        "HIGH_RISK_HIGH_UNCERTAINTY",
        f"hazard >= q{high_risk_quantile:.2f} and mean uncertainty >= q{high_uncertainty_quantile:.2f}; then uncertainty descending",
    )

    mid_pool = clusters[
        (~clusters["cross_polarity_cluster"].isin(used))
        & clusters["rank_fraction"].between(middle_bounds[0], middle_bounds[1], inclusive="both")
    ]
    middle = _stable_hash_sample(mid_pool, n_per_cohort, seed + 1).sort_values("cluster_rank")
    register(middle, "MID_PRIORITY", f"stable seeded sample from rank fraction {middle_bounds[0]:.2f}..{middle_bounds[1]:.2f}")

    low_pool = clusters[
        (~clusters["cross_polarity_cluster"].isin(used))
        & clusters["rank_fraction"].between(low_bounds[0], low_bounds[1], inclusive="both")
    ]
    low = _stable_hash_sample(low_pool, n_per_cohort, seed + 2).sort_values("cluster_rank")
    register(low, "LOW_PRIORITY", f"stable seeded sample from rank fraction {low_bounds[0]:.2f}..{low_bounds[1]:.2f}")

    cohort_order = ["TOP_PRIORITY", "MID_PRIORITY", "LOW_PRIORITY", "HIGH_RISK_HIGH_UNCERTAINTY"]
    selected = pd.concat(chosen, ignore_index=True)
    selected["cohort"] = pd.Categorical(selected["cohort"], cohort_order, ordered=True)
    selected = selected.sort_values(["cohort", "selection_order"]).reset_index(drop=True)
    selected["cohort"] = selected["cohort"].astype("string")
    if len(selected) != n_per_cohort * len(cohort_order):
        raise RuntimeError("Unexpected number of selected clusters")
    if selected["cross_polarity_cluster"].duplicated().any():
        raise RuntimeError("Frozen cohorts are not mutually exclusive")

    member_map = selected[["cross_polarity_cluster", "cohort", "selection_order", "selection_reason", "cluster_rank"]]
    members = features.merge(
        member_map,
        on="cross_polarity_cluster",
        how="inner",
        validate="many_to_one",
    )
    members["cluster_rank"] = members["cluster_rank"].astype(int)
    members = members.sort_values(["cohort", "selection_order", "mode", "spectrum_id"]).reset_index(drop=True)
    observed_member_counts = members.groupby("cross_polarity_cluster").size()
    expected_member_counts = selected.set_index("cross_polarity_cluster")["cluster_member_n"].astype(int)
    observed_aligned = observed_member_counts.reindex(expected_member_counts.index).to_numpy(dtype=int)
    if not np.array_equal(observed_aligned, expected_member_counts.to_numpy(dtype=int)):
        raise RuntimeError("Selected cluster member counts do not match Notebook 31")

    cluster_output = out / "32_frozen_cluster_cohort.csv"
    member_output = out / "32_frozen_feature_members.csv.gz"
    audit_output = out / "32_cohort_audit.csv"
    selected.to_csv(cluster_output, index=False)
    members.to_csv(member_output, index=False, compression={"method": "gzip", "mtime": 0})
    audit = (
        selected.groupby("cohort", sort=False, observed=True)
        .agg(
            n_clusters=("cross_polarity_cluster", "size"),
            min_cluster_rank=("cluster_rank", "min"),
            median_cluster_rank=("cluster_rank", "median"),
            max_cluster_rank=("cluster_rank", "max"),
            median_priority_score=("priority_score_primary", "median"),
            median_hazard_component=("hazard_component", "median"),
            median_uncertainty=("cluster_uncertainty_mean", "median"),
            pos_only=("modes", lambda x: int((x == "Pos").sum())),
            neg_only=("modes", lambda x: int((x == "Neg").sum())),
            cross_polarity=("modes", lambda x: int(x.str.contains("Pos").mul(x.str.contains("Neg")).sum())),
        )
        .reset_index()
    )
    audit.to_csv(audit_output, index=False)
    output_hashes = {
        cluster_output.name: sha256_file(cluster_output),
        member_output.name: sha256_file(member_output),
        audit_output.name: sha256_file(audit_output),
    }
    manifest = {
        "version": VERSION,
        "stage": 32,
        "status": "IDENTITY_REVEAL_COHORT_FROZEN",
        "generated_at_utc": utc_now(),
        "identity_blinding": "No compound name, formula, structure, library hit, or original-paper identity was read during selection",
        "sampling_unit": "cross-polarity cluster",
        "n_clusters_total": len(clusters),
        "n_clusters_selected": len(selected),
        "n_feature_members_selected": len(members),
        "n_per_cohort": n_per_cohort,
        "cohort_order": cohort_order,
        "seed": seed,
        "selection_parameters": {
            "top_rank_max": n_per_cohort,
            "middle_rank_fraction": list(middle_bounds),
            "low_rank_fraction": list(low_bounds),
            "high_risk_quantile": high_risk_quantile,
            "high_uncertainty_quantile": high_uncertainty_quantile,
            "hazard_threshold": hazard_threshold,
            "uncertainty_threshold": uncertainty_threshold,
        },
        "input_hashes": {
            rank_manifest_path.name: sha256_file(rank_manifest_path),
            feature_path.name: sha256_file(feature_path),
            cluster_path.name: sha256_file(cluster_path),
        },
        "output_hashes": output_hashes,
        "next_stage": "Public measured-library matching followed by a uniform SIRIUS pass over all 400 frozen clusters",
    }
    write_json(out / "32_run_manifest.json", manifest)
    return manifest


def run_sirius_export(root: Path, top_n_clusters: int = 100) -> dict[str, Any]:
    p = paths(root)
    rank_dir = p["case"] / "06_revealed_evaluation" / "31_feature_prioritization_v2"
    out = p["case"] / "06_revealed_evaluation" / "32_sirius_annotation_v2"
    pos_dir, neg_dir = out / "positive_ms", out / "negative_ms"
    pos_dir.mkdir(parents=True, exist_ok=True); neg_dir.mkdir(parents=True, exist_ok=True)
    for folder in (pos_dir, neg_dir):
        for stale in folder.glob("*.ms"):
            stale.unlink()
    ranking = pd.read_csv(rank_dir / "31_feature_priority_ranking.csv.gz")
    clusters = pd.read_csv(rank_dir / "31_cross_polarity_cluster_ranking.csv.gz").nsmallest(top_n_clusters, "cluster_rank")
    selected = ranking[ranking["cross_polarity_cluster"].isin(set(clusters["cross_polarity_cluster"]))].copy()
    records: dict[str, dict[str, Any]] = {}
    for mode in ("Pos", "Neg"):
        with (p["freeze"] / mode / "28_cleaned_spectra.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rec = json.loads(line); records[rec["spectrum_id"]] = rec
    manifest_rows = []
    for row in selected.sort_values(["cross_polarity_cluster", "mode", "spectrum_id"]).itertuples(index=False):
        rec = records[str(row.spectrum_id)]
        target = pos_dir if row.mode == "Pos" else neg_dir
        ms_path = target / f"{row.spectrum_id}.ms"
        lines = [f">compound {row.spectrum_id}", f">feature_id {row.spectrum_id}",
                 f">parentmass {float(row.precursor_mz):.8f}", f">ionization {row.adduct}",
                 f">rt {float(row.rt_min) * 60.0:.4f}s", "", ">ms2"]
        lines.extend(f"{float(m):.8f} {float(i):.8f}" for m, i in zip(rec["mz"], rec["intensity"]))
        ms_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        manifest_rows.append({"spectrum_id": row.spectrum_id, "alignment_id": int(row.alignment_id), "mode": row.mode,
                              "cross_polarity_cluster": row.cross_polarity_cluster, "cluster_rank": int(row.cluster_rank_within) if hasattr(row, "cluster_rank_within") else np.nan,
                              "precursor_mz": float(row.precursor_mz), "neutral_mass": float(row.neutral_mass), "rt_min": float(row.rt_min),
                              "adduct": row.adduct, "peak_n": len(rec["mz"]), "ms_file": str(ms_path.relative_to(root)),
                              "ms_sha256": sha256_file(ms_path)})
    export = pd.DataFrame(manifest_rows)
    cluster_rank_lookup = clusters.set_index("cross_polarity_cluster")["cluster_rank"]
    export["cluster_rank"] = export["cross_polarity_cluster"].map(cluster_rank_lookup).astype(int)
    export = export.sort_values(["cluster_rank", "mode", "spectrum_id"]).reset_index(drop=True)
    export.to_csv(out / "32_sirius_input_manifest.csv", index=False)
    readme = """# SIRIUS Top-100 cluster annotation inputs\n\n1. Start SIRIUS 6 and sign in.\n2. Create/open separate positive and negative projects.\n3. Import every `.ms` file from `positive_ms` into the positive project and every file from `negative_ms` into the negative project.\n4. Run Molecular Formula Identification, CSI:FingerID structure database search, and spectral library search with one frozen setting for every feature.\n5. Keep both projects open, then continue the REST-extraction cells in Notebook 32.\n\nDo not supply a known formula or structure. Do not tune parameters based on PBT rank.\n"""
    (out / "SIRIUS_运行说明.md").write_text(readme, encoding="utf-8")
    manifest = {"version": VERSION, "stage": 32, "status": "READY_FOR_SIRIUS_GUI_COMPUTATION", "generated_at_utc": utc_now(),
                "top_n_clusters": top_n_clusters, "n_clusters": int(export["cross_polarity_cluster"].nunique()),
                "n_sirius_features": len(export), "mode_counts": export["mode"].value_counts().to_dict(),
                "identity_blinding": "no formula, SMILES, InChI, or compound name supplied",
                "input_manifest_sha256": sha256_file(out / "32_sirius_input_manifest.csv")}
    write_json(out / "32_export_manifest.json", manifest)
    return manifest


def _sirius_api_request(base_url: str, method: str, path: str, params: dict[str, Any] | None = None, timeout: int = 180):
    import urllib.parse, urllib.request
    query = urllib.parse.urlencode(params or {}, doseq=True)
    url = base_url + path + (("?" + query) if query else "")
    data = b"" if method.upper() in {"POST", "PUT"} else None
    request = urllib.request.Request(url, data=data, method=method.upper(), headers={"Accept": "application/json, text/plain"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8-sig")
    return json.loads(raw) if raw.strip() else None


def run_sirius_result_import(root: Path, top_structure_n: int = 10) -> dict[str, Any]:
    import urllib.parse
    p = paths(root)
    out = p["case"] / "06_revealed_evaluation" / "32_sirius_annotation_v2"
    export_manifest = json.loads((out / "32_export_manifest.json").read_text(encoding="utf-8"))
    inputs = pd.read_csv(out / "32_sirius_input_manifest.csv")
    port_file = Path.home() / ".sirius" / "sirius-6.port"
    if not port_file.is_file():
        raise RuntimeError("SIRIUS GUI/REST 未运行。请启动 SIRIUS 6、打开已计算的正负项目，再重新运行本单元格。")
    base_url = f"http://127.0.0.1:{int(port_file.read_text(encoding='utf-8').strip())}"
    info = _sirius_api_request(base_url, "GET", "/api/info", {"includeConsumables": "false", "includeVersions": "true"})
    projects = _sirius_api_request(base_url, "GET", "/api/projects")
    expected_by_mode = {mode: set(inputs.loc[inputs["mode"] == mode, "spectrum_id"]) for mode in ("Pos", "Neg")}
    matched: dict[str, tuple[str, pd.DataFrame]] = {}
    for project in projects:
        pid = str(project["projectId"]); key = urllib.parse.quote(pid, safe="")
        features = pd.DataFrame(_sirius_api_request(base_url, "GET", f"/api/projects/{key}/aligned-features", {"includeMergedMs2": "false"}))
        if features.empty:
            continue
        features["spectrum_id"] = features["externalFeatureId"].fillna(features["name"])
        observed = set(features["spectrum_id"].astype(str))
        for mode, expected in expected_by_mode.items():
            if expected and expected.issubset(observed):
                if mode in matched:
                    raise RuntimeError(f"Multiple SIRIUS projects match {mode}")
                matched[mode] = (pid, features[features["spectrum_id"].isin(expected)].copy())
    if set(matched) != {"Pos", "Neg"}:
        raise RuntimeError(f"未同时找到精确匹配的正负项目。已匹配：{sorted(matched)}")
    formula_rows, structure_rows, library_rows = [], [], []
    for mode, (pid, features) in matched.items():
        pkey = urllib.parse.quote(pid, safe="")
        for feature in features.itertuples(index=False):
            fid = str(feature.alignedFeatureId); fkey = urllib.parse.quote(fid, safe="")
            base = f"/api/projects/{pkey}/aligned-features/{fkey}"
            try: formulas = _sirius_api_request(base_url, "GET", base + "/formulas") or []
            except Exception: formulas = []
            formulas = sorted(formulas, key=lambda x: (x.get("rank", 10**9), -float(x.get("siriusScore") or -1e30)))
            rank1 = formulas[0] if formulas else {}
            formula_rows.append({"mode": mode, "spectrum_id": feature.spectrum_id, "aligned_feature_id": fid,
                                 "formula_status": "success" if rank1 else "no_formula_candidates",
                                 "molecular_formula": rank1.get("molecularFormula"), "formula_adduct": rank1.get("adduct"),
                                 "formula_rank": rank1.get("rank"), "sirius_score": rank1.get("siriusScore"),
                                 "sirius_score_normalized": rank1.get("siriusScoreNormalized"), "formula_id": rank1.get("formulaId")})
            try: structures = _sirius_api_request(base_url, "GET", base + "/db-structures") or []
            except Exception: structures = []
            structures = sorted(structures, key=lambda x: -float(x.get("csiScore") or -1e30))[:top_structure_n]
            for rank, candidate in enumerate(structures, 1):
                structure_rows.append({"mode": mode, "spectrum_id": feature.spectrum_id, "structure_rank": rank,
                                       "structure_name": candidate.get("structureName"), "inchi_key": candidate.get("inchiKey"),
                                       "smiles": candidate.get("smiles"), "candidate_formula": candidate.get("molecularFormula"),
                                       "candidate_adduct": candidate.get("adduct"), "csi_score": candidate.get("csiScore"),
                                       "tanimoto_similarity": candidate.get("tanimotoSimilarity"), "xlogp": candidate.get("xlogP")})
            try: matches = _sirius_api_request(base_url, "GET", base + "/spectral-library-matches") or []
            except Exception: matches = []
            for rank, match in enumerate(matches[:top_structure_n], 1):
                library_rows.append({"mode": mode, "spectrum_id": feature.spectrum_id, "library_rank": rank,
                                     "library_name": match.get("referenceSpectrumName") or match.get("structureName") or match.get("name"),
                                     "library_similarity": match.get("similarity") or match.get("cosine"),
                                     "library_shared_peaks": match.get("sharedPeaks"), "library_id": match.get("referenceSpectrumId")})
    formulas = inputs.merge(pd.DataFrame(formula_rows), on=["mode", "spectrum_id"], how="left", validate="one_to_one")
    structures = pd.DataFrame(structure_rows)
    libraries = pd.DataFrame(library_rows)
    formulas.to_csv(out / "32_formula_annotations.csv", index=False)
    structures.to_csv(out / "32_structure_candidates_top10.csv.gz", index=False, compression="gzip")
    libraries.to_csv(out / "32_spectral_library_matches_top10.csv.gz", index=False, compression="gzip")
    top_structure = structures[structures["structure_rank"] == 1].drop(columns="structure_rank") if len(structures) else pd.DataFrame(columns=["mode", "spectrum_id"])
    top_library = libraries[libraries["library_rank"] == 1].drop(columns="library_rank") if len(libraries) else pd.DataFrame(columns=["mode", "spectrum_id"])
    annotation = formulas.merge(top_structure, on=["mode", "spectrum_id"], how="left", validate="one_to_one").merge(top_library, on=["mode", "spectrum_id"], how="left", validate="one_to_one")
    similarity = pd.to_numeric(annotation.get("library_similarity"), errors="coerce")
    annotation["annotation_level"] = np.select(
        [similarity.ge(0.8), annotation.get("inchi_key", pd.Series(index=annotation.index, dtype=object)).notna(), annotation["molecular_formula"].notna()],
        ["2_probable_library_match", "3_tentative_structure", "4_formula_only"], default="5_exact_mass_only")
    annotation.to_csv(out / "32_feature_annotation_summary.csv", index=False)
    cluster_summary = annotation.sort_values(["cluster_rank", "annotation_level", "spectrum_id"]).groupby("cross_polarity_cluster", as_index=False).first()
    cluster_summary.to_csv(out / "32_cluster_annotation_summary.csv", index=False)
    manifest = {"version": VERSION, "stage": 32, "status": "SIRIUS_ANNOTATIONS_IMPORTED", "generated_at_utc": utc_now(),
                "sirius_version": info.get("siriusVersion"), "n_features": len(annotation),
                "formula_success": int(annotation["molecular_formula"].notna().sum()),
                "structure_success": int(annotation.get("inchi_key", pd.Series(dtype=object)).notna().sum()),
                "library_level2": int(annotation["annotation_level"].eq("2_probable_library_match").sum()),
                "annotation_level_counts": annotation["annotation_level"].value_counts().to_dict(),
                "important_scope": "No level-1 confirmation without authentic standard RT and MS/MS match"}
    write_json(out / "32_run_manifest.json", manifest)
    return manifest
