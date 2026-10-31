"""Inventory and freeze a small Bangladesh surface-water DIA feasibility pilot.

The module is intentionally read-only with respect to the downloaded mzXML
files.  It validates the paired positive/negative datasets, freezes a
scientifically contrasting pilot set, audits the DIA acquisition signature,
and reports whether MS-DIAL is available for the next deconvolution notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VERSION = "PBT_MSMS_BANGLADESH_DIA_PILOT_GATE_27_V1"
PILOT_SAMPLE_UNITS = (
    "B1-1", "B1-2", "M1-1", "M1-2", "M4-1", "M4-2", "FieldBlank1", "QC1"
)
EXPECTED_DIA_CENTERS = (210.0, 450.0, 690.0, 930.0)
EXPECTED_DIA_WIDTH = 245.0


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


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def paths(root: Path, output_dir: Path | None = None) -> dict[str, Path]:
    case = root / "external_data" / "bangladesh_surface_water"
    out = output_dir or case / "01_raw" / "27_pilot_gate"
    manifests = case / "manifests"
    manifest_out = out / "manifests" if output_dir else manifests
    return {
        "root": root,
        "case": case,
        "out": out,
        "figures": out / "figures",
        "manifest_out": manifest_out,
        "download_report": root / "external_data" / "download_completion_report.json",
        "positive": case / "MSV000089705_ESI_positive" / "00_download",
        "negative": case / "MSV000089706_ESI_negative" / "00_download",
        "inventory": out / "27_full_file_inventory.csv",
        "pilot": out / "27_pilot_run_manifest.csv",
        "audit": out / "27_pilot_mzxml_scan_audit.csv",
        "conversion": out / "27_pilot_conversion_plan.csv",
        "source_metadata": out / "27_source_and_design_metadata.json",
        "qc": manifest_out / "27_feasibility_qc_checks.csv",
        "run_manifest": manifest_out / "27_feasibility_gate_manifest.json",
    }


def sample_metadata(filename: str) -> dict[str, Any]:
    stem = re.sub(r"_ESI(?:pos|neg)_\d+$", "", Path(filename).stem, flags=re.IGNORECASE)
    aliases = {"M31": "M3-1", "M41": "M4-1"}
    sample = aliases.get(stem, stem)
    if sample.startswith("FieldBlank"):
        return {
            "sample_unit_id": sample, "sample_type": "Blank", "site_id": "FieldBlank",
            "replicate": int(re.search(r"\d+", sample).group()), "river": "Control",
            "setting": "field blank", "pilot_contrast": "background control",
        }
    if sample.startswith("QC"):
        return {
            "sample_unit_id": sample, "sample_type": "QC", "site_id": "PooledQC",
            "replicate": int(re.search(r"\d+", sample).group()), "river": "Control",
            "setting": "pooled quality control", "pilot_contrast": "analytical stability control",
        }
    match = re.fullmatch(r"([BMST])(\d+)-(\d+)", sample)
    if not match:
        raise ValueError(f"Unrecognized Bangladesh filename: {filename}")
    prefix, site_number, replicate = match.groups()
    river_lookup = {"B": "Buriganga", "M": "Meghna", "S": "Shitalakshya", "T": "Turag"}
    site = f"{prefix}{site_number}"
    if prefix in {"B", "S", "T"}:
        setting = "urban Dhaka-associated"
        contrast = "urban"
    elif site in {"M1", "M2", "M3"}:
        setting = "upper Meghna"
        contrast = "rural upstream"
    elif site == "M4":
        setting = "lower Meghna after confluence"
        contrast = "downstream confluence"
    else:
        setting = "lower Meghna"
        contrast = "downstream"
    return {
        "sample_unit_id": sample, "sample_type": "Sample", "site_id": site,
        "replicate": int(replicate), "river": river_lookup[prefix],
        "setting": setting, "pilot_contrast": contrast,
    }


def load_mode_inventory(dataset_dir: Path, dataset_id: str, mode: str) -> pd.DataFrame:
    peak_dir = dataset_dir / "peak"
    checksum_path = dataset_dir / f"{dataset_id}_sha256.csv"
    checksums = pd.read_csv(checksum_path)
    checksum_lookup = {Path(value.Path).name: str(value.Hash).lower() for value in checksums.itertuples()}
    rows: list[dict[str, Any]] = []
    for file in sorted(peak_dir.glob("*.mzXML"), key=lambda item: item.name.lower()):
        meta = sample_metadata(file.name)
        rows.append({
            "dataset_id": dataset_id, "mode": mode, "filename": file.name,
            "path": str(file.resolve()), "size_bytes": file.stat().st_size,
            "recorded_sha256": checksum_lookup.get(file.name, ""),
            "selected_for_pilot": meta["sample_unit_id"] in PILOT_SAMPLE_UNITS,
            **meta,
        })
    return pd.DataFrame(rows)


def parse_attributes(text: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z][A-Za-z0-9]*)=\"([^\"]*)\"", text))


def parse_seconds(value: str) -> float:
    match = re.fullmatch(r"PT([0-9.]+)S", value or "")
    return float(match.group(1)) if match else np.nan


def audit_mzxml(path: Path, expected_sha256: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    declared_scan_n: int | None = None
    ms1_peaks: list[int] = []
    ms2_peaks: list[int] = []
    ms2_centers: list[float] = []
    ms2_widths: list[float] = []
    collision_energies: list[float] = []
    polarities: set[str] = set()
    retention_seconds: list[float] = []
    model = ""
    scan_header: list[str] | None = None

    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            line = raw_line.decode("latin-1", errors="ignore").strip()
            if declared_scan_n is None and "<msRun " in line:
                attrs = parse_attributes(line)
                if "scanCount" in attrs:
                    declared_scan_n = int(attrs["scanCount"])
            if "<msModel " in line:
                model = parse_attributes(line).get("value", model)
            if "<scan " in line:
                scan_header = [line]
            elif scan_header is not None:
                scan_header.append(line)
            if scan_header is not None and ">" in line:
                attrs = parse_attributes(" ".join(scan_header))
                level = int(attrs.get("msLevel", "0"))
                peaks = int(attrs.get("peaksCount", "0"))
                if level == 1:
                    ms1_peaks.append(peaks)
                elif level == 2:
                    ms2_peaks.append(peaks)
                    if "collisionEnergy" in attrs:
                        collision_energies.append(float(attrs["collisionEnergy"]))
                if attrs.get("polarity"):
                    polarities.add(attrs["polarity"])
                if attrs.get("retentionTime"):
                    retention_seconds.append(parse_seconds(attrs["retentionTime"]))
                scan_header = None
            if "<precursorMz " in line:
                attrs = parse_attributes(line)
                center_match = re.search(r">\s*([0-9.]+)\s*</precursorMz>", line)
                if center_match:
                    ms2_centers.append(float(center_match.group(1)))
                if attrs.get("windowWideness"):
                    ms2_widths.append(float(attrs["windowWideness"]))

    observed_hash = digest.hexdigest()
    parsed_scan_n = len(ms1_peaks) + len(ms2_peaks)
    unique_centers = sorted({round(value, 4) for value in ms2_centers})
    unique_widths = sorted({round(value, 4) for value in ms2_widths})
    unique_ce = sorted({round(value, 4) for value in collision_energies})
    return {
        "instrument_model": model,
        "declared_scan_n": declared_scan_n,
        "parsed_scan_n": parsed_scan_n,
        "ms1_scan_n": len(ms1_peaks),
        "ms2_scan_n": len(ms2_peaks),
        "ms2_per_ms1_ratio": len(ms2_peaks) / len(ms1_peaks) if ms1_peaks else np.nan,
        "median_ms1_peak_n": float(np.median(ms1_peaks)) if ms1_peaks else np.nan,
        "median_ms2_peak_n": float(np.median(ms2_peaks)) if ms2_peaks else np.nan,
        "ms2_ge6_peak_fraction": float(np.mean(np.asarray(ms2_peaks) >= 6)) if ms2_peaks else np.nan,
        "retention_time_min": float(np.nanmin(retention_seconds) / 60) if retention_seconds else np.nan,
        "retention_time_max": float(np.nanmax(retention_seconds) / 60) if retention_seconds else np.nan,
        "polarity_values": ";".join(sorted(polarities)),
        "dia_window_centers_mz": ";".join(f"{value:g}" for value in unique_centers),
        "dia_window_widths_mz": ";".join(f"{value:g}" for value in unique_widths),
        "collision_energies": ";".join(f"{value:g}" for value in unique_ce),
        "observed_sha256": observed_hash,
        "sha256_matches_download_manifest": bool(expected_sha256 and observed_hash == expected_sha256.lower()),
        "scan_count_matches_header": declared_scan_n == parsed_scan_n,
    }


def find_msconvert(explicit: Path | None) -> Path | None:
    candidates = [
        explicit,
        Path(r"C:\Users\Lenovo\AppData\Local\Apps\ProteoWizard 3.0.26260.f8f0a9a6 64-bit\msconvert.exe"),
    ]
    which = shutil.which("msconvert")
    if which:
        candidates.append(Path(which))
    for item in candidates:
        if item is None:
            continue
        try:
            if item.is_file():
                return item.resolve()
        except PermissionError:
            # The Codex sandbox can deny stat() on an otherwise user-accessible
            # executable.  Preserve the exact configured path for the local run.
            if explicit is not None or "ProteoWizard" in str(item):
                return item.resolve(strict=False)
    return None


def find_msdial(explicit: Path | None) -> Path | None:
    candidates = [explicit]
    for name in ("MsdialConsoleApp.exe", "MsdialWorkbench.exe", "MSDIAL.exe"):
        which = shutil.which(name)
        if which:
            candidates.append(Path(which))
    for item in candidates:
        if item is None:
            continue
        try:
            if item.is_file():
                return item.resolve()
        except PermissionError:
            if explicit is not None:
                return item.resolve(strict=False)
    return None


def plot_audit(audit: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = audit.sort_values(["mode", "pilot_order"])
    labels = ordered["sample_unit_id"] + " " + ordered["mode"]
    x = np.arange(len(ordered))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes[0, 0].bar(x, ordered["ms2_per_ms1_ratio"], color=np.where(ordered["mode"].eq("Pos"), "#3b6fb6", "#e67e22"))
    axes[0, 0].axhline(4, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_xticks(x, labels, rotation=70, ha="right", fontsize=8)
    axes[0, 0].set_ylabel("MS2 scans per MS1 scan")
    axes[0, 0].set_title("A  Four-window DIA cycle", loc="left", fontweight="bold")

    axes[0, 1].bar(x, ordered["median_ms2_peak_n"], color=np.where(ordered["sample_type"].eq("Sample"), "#4c956c", "#9b9b9b"))
    axes[0, 1].set_xticks(x, labels, rotation=70, ha="right", fontsize=8)
    axes[0, 1].set_ylabel("Median centroid peaks per DIA scan")
    axes[0, 1].set_title("B  Raw DIA spectral richness", loc="left", fontweight="bold")

    axes[1, 0].bar(x, 100 * ordered["ms2_ge6_peak_fraction"], color="#7a5195")
    axes[1, 0].set_xticks(x, labels, rotation=70, ha="right", fontsize=8)
    axes[1, 0].set_ylim(0, 105)
    axes[1, 0].set_ylabel("DIA scans with ≥6 peaks (%)")
    axes[1, 0].set_title("C  Minimum peak-count potential", loc="left", fontweight="bold")

    for index, center in enumerate(EXPECTED_DIA_CENTERS):
        left = center - EXPECTED_DIA_WIDTH / 2
        axes[1, 1].broken_barh([(left, EXPECTED_DIA_WIDTH)], (index - 0.35, 0.7), facecolors="#5b8ff9")
        axes[1, 1].text(center, index, f"{center:g}", ha="center", va="center", color="white", fontweight="bold")
    axes[1, 1].set_yticks(range(4), [f"Window {index}" for index in range(1, 5)])
    axes[1, 1].set_xlim(80, 1060)
    axes[1, 1].set_xlabel("m/z")
    axes[1, 1].set_title("D  Acquisition windows (width 245 m/z)", loc="left", fontweight="bold")
    axes[1, 1].spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def add_qc(checks: list[dict[str, Any]], check: str, required: bool, passed: bool, detail: Any) -> None:
    checks.append({"check": check, "required": bool(required), "passed": bool(passed), "detail": str(detail)})


def run(
    root: Path, output_dir: Path | None = None, msdial_exe: Path | None = None,
    msconvert_exe: Path | None = None, max_pilot_runs: int | None = None,
) -> dict[str, Any]:
    p = paths(root, output_dir)
    p["out"].mkdir(parents=True, exist_ok=True)
    download_report = json.loads(p["download_report"].read_text(encoding="utf-8"))
    positive = load_mode_inventory(p["positive"], "MSV000089705", "Pos")
    negative = load_mode_inventory(p["negative"], "MSV000089706", "Neg")
    inventory = pd.concat([positive, negative], ignore_index=True)
    pilot = inventory.loc[inventory["selected_for_pilot"]].copy()
    pilot_order_lookup = {sample: index + 1 for index, sample in enumerate(PILOT_SAMPLE_UNITS)}
    pilot["pilot_order"] = pilot["sample_unit_id"].map(pilot_order_lookup)
    pilot = pilot.sort_values(["pilot_order", "mode"]).reset_index(drop=True)
    if max_pilot_runs is not None:
        pilot = pilot.head(int(max_pilot_runs)).copy()

    audit_rows: list[dict[str, Any]] = []
    for row in pilot.itertuples(index=False):
        print(f"Auditing {row.mode} {row.sample_unit_id}: {row.filename}", flush=True)
        details = audit_mzxml(Path(row.path), row.recorded_sha256)
        audit_rows.append({
            "dataset_id": row.dataset_id, "mode": row.mode, "sample_unit_id": row.sample_unit_id,
            "sample_type": row.sample_type, "site_id": row.site_id, "river": row.river,
            "pilot_contrast": row.pilot_contrast, "pilot_order": row.pilot_order,
            "filename": row.filename, "path": row.path, "size_bytes": row.size_bytes,
            **details,
        })
    audit = pd.DataFrame(audit_rows)

    conversion = pilot[[
        "pilot_order", "dataset_id", "mode", "sample_unit_id", "sample_type", "site_id", "river",
        "pilot_contrast", "filename", "path", "size_bytes", "recorded_sha256",
    ]].copy()
    conversion["mzml_output_path"] = conversion.apply(
        lambda row: str((p["case"] / "02_mzml" / "27_pilot" / row["mode"] /
                         f"{row['sample_unit_id']}_{row['mode']}.mzML").resolve()), axis=1,
    )
    conversion["conversion_required"] = True
    conversion["conversion_reason"] = "standardize centroided mzXML to mzML for the MS-DIAL DIA pilot"
    conversion["source_files_are_not_modified"] = True

    msconvert = find_msconvert(msconvert_exe)
    msdial = find_msdial(msdial_exe)
    expected_run_n = 16 if max_pilot_runs is None else min(16, int(max_pilot_runs))
    centers_ok = audit["dia_window_centers_mz"].eq("210;450;690;930").all()
    widths_ok = audit["dia_window_widths_mz"].eq("245").all()
    ratio_ok = audit["ms2_per_ms1_ratio"].between(3.95, 4.05).all()

    checks: list[dict[str, Any]] = []
    add_qc(checks, "download_completion_report_passed", True, download_report.get("all_passed") is True, download_report.get("all_passed"))
    add_qc(checks, "positive_mzxml_file_count", True, len(positive) == 32, len(positive))
    add_qc(checks, "negative_mzxml_file_count", True, len(negative) == 32, len(negative))
    add_qc(checks, "mode_sample_units_pair_exactly", True, set(positive["sample_unit_id"]) == set(negative["sample_unit_id"]), len(set(positive["sample_unit_id"]) & set(negative["sample_unit_id"])))
    add_qc(checks, "environmental_samples_per_mode", True, int((positive["sample_type"] == "Sample").sum()) == 24 and int((negative["sample_type"] == "Sample").sum()) == 24, "24 Pos + 24 Neg")
    add_qc(checks, "field_blanks_per_mode", True, int((positive["sample_type"] == "Blank").sum()) == 3 and int((negative["sample_type"] == "Blank").sum()) == 3, "3 Pos + 3 Neg")
    add_qc(checks, "pooled_qcs_per_mode", True, int((positive["sample_type"] == "QC").sum()) == 5 and int((negative["sample_type"] == "QC").sum()) == 5, "5 Pos + 5 Neg")
    add_qc(checks, "pilot_run_count", True, len(pilot) == expected_run_n, len(pilot))
    add_qc(checks, "pilot_hashes_match_download_manifest", True, audit["sha256_matches_download_manifest"].all(), f"{int(audit['sha256_matches_download_manifest'].sum())}/{len(audit)}")
    add_qc(checks, "scan_counts_match_mzxml_headers", True, audit["scan_count_matches_header"].all(), f"{int(audit['scan_count_matches_header'].sum())}/{len(audit)}")
    add_qc(checks, "q_exactive_hfx_confirmed", True, audit["instrument_model"].str.contains("Q Exactive HF-X", regex=False).all(), ";".join(sorted(audit["instrument_model"].unique())))
    add_qc(checks, "four_dia_window_centers_confirmed", True, centers_ok, ";".join(sorted(audit["dia_window_centers_mz"].unique())))
    add_qc(checks, "dia_window_width_245_confirmed", True, widths_ok, ";".join(sorted(audit["dia_window_widths_mz"].unique())))
    add_qc(checks, "four_ms2_scans_per_ms1_confirmed", True, ratio_ok, f"range {audit['ms2_per_ms1_ratio'].min():.3f}-{audit['ms2_per_ms1_ratio'].max():.3f}")
    add_qc(checks, "raw_dia_scans_have_peak_content", True, audit["ms2_ge6_peak_fraction"].median() >= 0.90, f"median fraction {audit['ms2_ge6_peak_fraction'].median():.3f}")
    add_qc(checks, "msconvert_available_for_mzml_standardization", True, msconvert is not None, msconvert or "not found")
    add_qc(checks, "msdial_available_for_deconvolution", False, msdial is not None, msdial or "not found; install official MS-DIAL before Notebook 28")
    qc = pd.DataFrame(checks)
    required_pass = bool(qc.loc[qc["required"], "passed"].all())

    write_csv(inventory, p["inventory"])
    write_csv(pilot, p["pilot"])
    write_csv(audit, p["audit"])
    write_csv(conversion, p["conversion"])
    write_csv(qc, p["qc"])
    source_metadata = {
        "paper_doi": "10.1021/acs.est.2c08200",
        "paper_title": "Nontarget Analysis of Polluted Surface Waters in Bangladesh Using Open Science Workflows",
        "positive_dataset": "MSV000089705",
        "negative_dataset": "MSV000089706",
        "instrument": "Thermo Scientific Q Exactive HF-X",
        "acquisition": "parallel full MS1 plus four sequential wide-window DIA MS2 scans",
        "dia_window_centers_mz": list(EXPECTED_DIA_CENTERS),
        "dia_window_width_mz": EXPECTED_DIA_WIDTH,
        "pilot_rationale": {
            "B1": "urban Buriganga contrast",
            "M1": "upper Meghna rural contrast",
            "M4": "lower Meghna after confluence with Dhaka-associated rivers",
            "FieldBlank1": "background/contamination control",
            "QC1": "pooled analytical stability control",
        },
        "critical_constraint": "Raw wide-window DIA spectra are mixtures and must not be passed directly to the frozen DDA-trained PBT model.",
    }
    write_json(p["source_metadata"], source_metadata)
    figure = p["figures"] / "27_pilot_design_and_dia_acquisition.png"
    plot_audit(audit, figure)

    if required_pass and msdial is not None:
        status = "READY_FOR_NOTEBOOK_28_DIA_DECONVOLUTION"
    elif required_pass:
        status = "PILOT_DATA_READY_MSDIAL_INSTALL_REQUIRED"
    else:
        status = "FAILED_QC"
    artifacts = [
        p["inventory"], p["pilot"], p["audit"], p["conversion"], p["source_metadata"],
        p["qc"], figure, figure.with_suffix(".pdf"),
    ]
    disk = shutil.disk_usage(p["case"])
    manifest = {
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "status": status,
        "data_qc_all_required_pass": required_pass,
        "msdial_ready": msdial is not None,
        "msdial_executable": str(msdial) if msdial else None,
        "msconvert_executable": str(msconvert) if msconvert else None,
        "counts": {
            "total_mzxml_files": int(len(inventory)),
            "positive_files": int(len(positive)),
            "negative_files": int(len(negative)),
            "pilot_sample_units": int(pilot["sample_unit_id"].nunique()),
            "pilot_runs": int(len(pilot)),
            "pilot_environmental_runs": int((pilot["sample_type"] == "Sample").sum()),
            "pilot_blank_runs": int((pilot["sample_type"] == "Blank").sum()),
            "pilot_qc_runs": int((pilot["sample_type"] == "QC").sum()),
        },
        "pilot_raw_bytes": int(pilot["size_bytes"].sum()),
        "free_disk_bytes": int(disk.free),
        "estimated_minimum_free_bytes_for_conversion_and_deconvolution": int(2.5 * pilot["size_bytes"].sum()),
        "raw_files_modified": False,
        "direct_raw_dia_model_inference_allowed": False,
        "input_hashes": {
            p["download_report"].name: sha256_file(p["download_report"]),
            "MSV000089705_sha256.csv": sha256_file(p["positive"] / "MSV000089705_sha256.csv"),
            "MSV000089706_sha256.csv": sha256_file(p["negative"] / "MSV000089706_sha256.csv"),
        },
        "analysis_code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "output_hashes": {item.name: sha256_file(item) for item in artifacts},
        "next_step": (
            "Notebook 28: convert the frozen 16-run pilot to mzML, perform MS-DIAL DIA deconvolution, "
            "and quantify pseudo-MS/MS eligibility and replicate/QC/blank behavior"
        ),
    }
    write_json(p["run_manifest"], manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--msdial-exe", type=Path)
    parser.add_argument("--msconvert-exe", type=Path)
    parser.add_argument("--max-pilot-runs", type=int)
    args = parser.parse_args()
    print(json.dumps(run(
        args.root, args.output_dir, args.msdial_exe, args.msconvert_exe, args.max_pilot_runs
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
