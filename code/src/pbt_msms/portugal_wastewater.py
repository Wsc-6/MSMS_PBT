"""Audit and safely extract the Portugal wastewater DDA case-study archive.

Notebook 21 deliberately stops before mzML conversion and feature extraction.
It freezes the raw-data inventory and extracts only AutoMSMS DDA data; the DIA
subtree remains in the source archive for an optional later domain study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
from tqdm.auto import tqdm


EXPECTED_ARCHIVE_SHA256 = "2b4b642ef24bc31069719b51d67f7e28cf5805ac3cf666f2bb028d445b20859c"
EXPECTED_DDA_DATASETS = 96
EXPECTED_DDA_BY_MODE = {"Pos": 48, "Neg": 48}
EXPECTED_SITES = {"CP-VC", "PS-F", "PS-T", "PS-VC", "PVZ-1", "PVZ-2", "WWTP Eff", "WWTP In"}
EXPECTED_CAMPAIGNS = {1, 2, 3}
EXPECTED_INJECTIONS = {"A", "B"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
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


def project_paths(root: Path) -> dict[str, Path]:
    case = root / "external_data/portugal_wastewater_5830725"
    return {
        "case": case,
        "archive": case / "00_archive/Portugal_wastewater_LCHRMS_dataset.zip",
        "archive_hash_manifest": case / "manifests/portugal_archive_sha256.csv",
        "raw": case / "01_raw_agilent_d",
        "manifests": case / "manifests",
        "nb20": root / "outputs/pbt_msms_v1/chemvista_external_20_v1/20_run_manifest.json",
        "entry_manifest": case / "manifests/21_portugal_dda_archive_entries.csv",
        "dataset_manifest": case / "manifests/21_portugal_dda_dataset_manifest.csv",
        "qc": case / "manifests/21_portugal_dda_qc_checks.csv",
        "run_manifest": case / "manifests/21_portugal_dda_extraction_manifest.json",
        "conversion_plan": case / "manifests/21_portugal_msconvert_plan.csv",
    }


def safe_relative_target(filename: str) -> tuple[str, Path] | None:
    parts = PurePosixPath(filename).parts
    if len(parts) < 4:
        return None
    if parts[0] != "Portugal_wastewater_LCHRMS_dataset" or parts[1] != "AutoMSMS_DDA":
        return None
    mode = parts[2]
    if mode not in {"Pos", "Neg"}:
        return None
    relative = Path(*parts[2:])
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe archive entry: {filename}")
    return mode, relative


def parse_dataset_name(name: str) -> tuple[str, int, str]:
    normalized = re.sub(r"\s+", " ", name.strip())
    match = re.fullmatch(r"(.+?)\s+([123])_([AB])\.d", normalized, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unrecognized DDA dataset name: {name}")
    return match.group(1), int(match.group(2)), match.group(3).upper()


def inventory_archive(p: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    archive = p["archive"]
    if not archive.is_file():
        raise FileNotFoundError(archive)
    actual_hash = sha256_file(archive)
    if actual_hash != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(f"Portugal archive SHA-256 mismatch: {actual_hash}")
    if p["archive_hash_manifest"].exists():
        frozen_hashes = pd.read_csv(p["archive_hash_manifest"])
        values = set(frozen_hashes["Hash"].astype(str).str.lower())
        if actual_hash not in values:
            raise RuntimeError("Archive hash disagrees with download manifest")

    entry_rows: list[dict[str, Any]] = []
    unsafe = []
    total_uncompressed = 0
    dia_uncompressed = 0
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            total_uncompressed += int(info.file_size)
            if "/All_Ions_DIA/" in info.filename:
                dia_uncompressed += int(info.file_size)
            selected = safe_relative_target(info.filename)
            if selected is None or info.is_dir():
                continue
            mode, relative = selected
            target = (p["raw"] / relative).resolve()
            raw_resolved = p["raw"].resolve()
            try:
                target.relative_to(raw_resolved)
            except ValueError:
                unsafe.append(info.filename)
                continue
            parts = PurePosixPath(info.filename).parts
            dataset_name = parts[3]
            site, campaign, injection = parse_dataset_name(dataset_name)
            entry_rows.append({
                "archive_path": info.filename,
                "target_relative_path": relative.as_posix(),
                "mode": mode, "dataset_name_raw": dataset_name,
                "site_code": site, "campaign": campaign, "injection": injection,
                "file_size": int(info.file_size), "compressed_size": int(info.compress_size),
                "crc32": f"{int(info.CRC):08x}",
            })
    if unsafe:
        raise RuntimeError(f"Unsafe ZIP entries found: {unsafe[:3]}")
    entries = pd.DataFrame(entry_rows).sort_values(["mode", "site_code", "campaign", "injection", "archive_path"]).reset_index(drop=True)
    if entries.empty:
        raise RuntimeError("No AutoMSMS DDA files found in archive")

    dataset_rows = []
    for keys, group in entries.groupby(["mode", "dataset_name_raw", "site_code", "campaign", "injection"], sort=True):
        mode, dataset_name, site, campaign, injection = keys
        dataset_rows.append({
            "dataset_id": f"PTWW_{mode}_{site.replace(' ', '-')}_C{campaign}_{injection}",
            "mode": mode, "site_code": site, "campaign": int(campaign), "injection": injection,
            "iterative_injection_role": "first" if injection == "A" else "second",
            "dataset_name_raw": dataset_name,
            "raw_d_relative_path": f"{mode}/{dataset_name}",
            "entry_count": len(group), "uncompressed_bytes": int(group["file_size"].sum()),
            "has_mspeak_bin": bool(group["archive_path"].str.endswith("/AcqData/MSPeak.bin").any()),
            "has_msscan_bin": bool(group["archive_path"].str.endswith("/AcqData/MSScan.bin").any()),
            "sample_class": "environmental_sample",
            "blank_or_qc": False,
        })
    datasets = pd.DataFrame(dataset_rows).sort_values(["mode", "site_code", "campaign", "injection"]).reset_index(drop=True)
    metadata = {
        "archive_sha256": actual_hash,
        "archive_size_bytes": archive.stat().st_size,
        "zip_total_uncompressed_bytes": total_uncompressed,
        "dda_uncompressed_bytes": int(entries["file_size"].sum()),
        "dia_uncompressed_bytes": dia_uncompressed,
        "unsafe_entry_count": len(unsafe),
    }
    return entries, datasets, metadata


def conversion_plan(datasets: pd.DataFrame, p: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for row in datasets.itertuples(index=False):
        output_subdir = f"{row.mode}"
        output_name = f"{row.dataset_id}.mzML"
        rows.append({
            "dataset_id": row.dataset_id,
            "mode": row.mode,
            "site_code": row.site_code,
            "campaign": row.campaign,
            "injection": row.injection,
            "input_d": str((p["raw"] / row.raw_d_relative_path).resolve()),
            "output_directory": str((p["case"] / "02_mzml" / output_subdir).resolve()),
            "output_filename": output_name,
            "planned_msconvert_filters": "peakPicking true 1-; zeroSamples removeExtra; titleMaker <RunId>.<ScanNumber>",
            "centroiding_note": "vendor peakPicking must run before other filters",
        })
    return pd.DataFrame(rows)


def extract_entries(entries: pd.DataFrame, p: dict[str, Path], reextract: bool) -> dict[str, int]:
    p["raw"].mkdir(parents=True, exist_ok=True)
    missing_bytes = 0
    for row in entries.itertuples(index=False):
        target = p["raw"] / Path(row.target_relative_path)
        if reextract or not target.is_file() or target.stat().st_size != int(row.file_size):
            missing_bytes += int(row.file_size)
    free_bytes = shutil.disk_usage(p["raw"]).free
    safety_margin = 5 * 1024**3
    if free_bytes < missing_bytes + safety_margin:
        raise RuntimeError(
            f"Insufficient free space: need missing {missing_bytes / 1024**3:.2f} GB plus 5 GB margin; "
            f"available {free_bytes / 1024**3:.2f} GB"
        )

    extracted = skipped = 0
    with zipfile.ZipFile(p["archive"]) as handle:
        for row in tqdm(entries.itertuples(index=False), total=len(entries), desc="Extract Portugal AutoMSMS DDA"):
            target = p["raw"] / Path(row.target_relative_path)
            if not reextract and target.is_file() and target.stat().st_size == int(row.file_size):
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            if temporary.exists():
                temporary.unlink()
            with handle.open(row.archive_path) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            if temporary.stat().st_size != int(row.file_size):
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"Extracted size mismatch: {row.archive_path}")
            os.replace(temporary, target)
            extracted += 1
    return {"extracted_files": extracted, "skipped_existing_files": skipped, "free_bytes_before": free_bytes}


def validate_extraction(entries: pd.DataFrame, datasets: pd.DataFrame, p: dict[str, Path]) -> tuple[bool, list[str]]:
    errors = []
    for row in entries.itertuples(index=False):
        target = p["raw"] / Path(row.target_relative_path)
        if not target.is_file():
            errors.append(f"missing:{row.target_relative_path}")
        elif target.stat().st_size != int(row.file_size):
            errors.append(f"size:{row.target_relative_path}")
    for row in datasets.itertuples(index=False):
        directory = p["raw"] / row.raw_d_relative_path
        if not directory.is_dir():
            errors.append(f"dataset_dir:{row.raw_d_relative_path}")
    return not errors, errors


def run(root: Path, stage: str, reextract: bool = False) -> dict[str, Any]:
    p = project_paths(root)
    p["manifests"].mkdir(parents=True, exist_ok=True)
    nb20 = json.loads(p["nb20"].read_text(encoding="utf-8"))
    if nb20.get("status") != "READY_FOR_NOTEBOOK_21" or not nb20.get("qc_all_pass"):
        raise RuntimeError("Notebook 20 is not READY_FOR_NOTEBOOK_21")

    entries, datasets, archive_meta = inventory_archive(p)
    atomic_csv(entries, p["entry_manifest"])
    atomic_csv(datasets, p["dataset_manifest"])
    plan = conversion_plan(datasets, p)
    atomic_csv(plan, p["conversion_plan"])

    extraction = {"extracted_files": 0, "skipped_existing_files": 0}
    if stage == "extract":
        extraction = extract_entries(entries, p, reextract)
    extraction_ok, errors = validate_extraction(entries, datasets, p)

    counts_by_mode = datasets["mode"].value_counts().to_dict()
    combinations = datasets.groupby(["mode", "site_code", "campaign", "injection"]).size()
    expected_combinations = 2 * len(EXPECTED_SITES) * len(EXPECTED_CAMPAIGNS) * len(EXPECTED_INJECTIONS)
    qc_rows = [
        {"check": "notebook20_ready", "passed": nb20.get("status") == "READY_FOR_NOTEBOOK_21", "detail": nb20.get("status")},
        {"check": "archive_sha256_matches", "passed": archive_meta["archive_sha256"] == EXPECTED_ARCHIVE_SHA256, "detail": archive_meta["archive_sha256"]},
        {"check": "zip_paths_safe", "passed": archive_meta["unsafe_entry_count"] == 0, "detail": archive_meta["unsafe_entry_count"]},
        {"check": "dda_dataset_count_96", "passed": len(datasets) == EXPECTED_DDA_DATASETS, "detail": len(datasets)},
        {"check": "dda_mode_balance", "passed": counts_by_mode == EXPECTED_DDA_BY_MODE, "detail": str(counts_by_mode)},
        {"check": "site_codes_complete", "passed": set(datasets["site_code"]) == EXPECTED_SITES, "detail": str(sorted(datasets["site_code"].unique()))},
        {"check": "campaigns_complete", "passed": set(datasets["campaign"].astype(int)) == EXPECTED_CAMPAIGNS, "detail": str(sorted(datasets["campaign"].unique()))},
        {"check": "iterative_injections_complete", "passed": set(datasets["injection"]) == EXPECTED_INJECTIONS and len(combinations) == expected_combinations and combinations.eq(1).all(), "detail": f"{len(combinations)}/{expected_combinations}"},
        {"check": "vendor_binary_components_present", "passed": datasets[["has_mspeak_bin", "has_msscan_bin"]].all().all(), "detail": "MSPeak.bin and MSScan.bin in every .d"},
        {"check": "explicit_blanks_present", "passed": False, "required": False, "detail": "No filename-designated blank/QC runs in public archive; record as case-study limitation"},
        {"check": "dda_extraction_complete", "passed": extraction_ok if stage == "extract" else False, "required": stage == "extract", "detail": "PASS" if extraction_ok else f"pending/errors={len(errors)}"},
    ]
    qc = pd.DataFrame(qc_rows)
    atomic_csv(qc, p["qc"])
    # Missing values mean "required"; only an explicit False marks an informational check.
    required_mask = qc["required"].isna() | qc["required"].eq(True)
    required = qc[required_mask]
    all_required_pass = bool(required["passed"].astype(bool).all())
    status = (
        "READY_FOR_NOTEBOOK_22" if stage == "extract" and all_required_pass
        else "READY_FOR_DDA_EXTRACTION" if stage == "audit" and all_required_pass
        else "BLOCKED_REVIEW_QC"
    )
    manifest = {
        "version": "PBT_MSMS_PORTUGAL_DDA_EXTRACTION_21_V1",
        "generated_at_utc": utc_now(), "stage": stage, "status": status,
        "scientific_scope": "Portugal municipal wastewater AutoMSMS DDA raw-data freeze",
        "notebook20_manifest_sha256": sha256_file(p["nb20"]),
        "archive": archive_meta,
        "selection": "AutoMSMS_DDA only; All_Ions_DIA retained in archive and not extracted",
        "counts": {
            "dda_datasets": len(datasets), "dda_archive_entries": len(entries),
            "mode": counts_by_mode, "sites": len(set(datasets["site_code"])),
            "campaigns": len(set(datasets["campaign"])), "iterative_injections": sorted(set(datasets["injection"])),
            "explicit_blank_or_qc_datasets": int(datasets["blank_or_qc"].sum()),
        },
        "extraction": extraction,
        "blank_limitation": "No explicit blank/QC raw files are identifiable from archive filenames; do not claim blank subtraction unless independent metadata establishes blanks.",
        "next_step": "Notebook 22: ProteoWizard vendor centroiding and mzML conversion with frozen conversion plan",
        "qc_all_required_pass": all_required_pass,
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in [p["entry_manifest"], p["dataset_manifest"], p["conversion_plan"], p["qc"]]
        },
    }
    atomic_json(p["run_manifest"], manifest)
    print(qc.to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if status == "BLOCKED_REVIEW_QC":
        raise RuntimeError("Notebook 21 QC failed")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["audit", "extract"], required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--reextract", action="store_true")
    args = parser.parse_args()
    run(locate_root(args.root), args.stage, args.reextract)


if __name__ == "__main__":
    main()
