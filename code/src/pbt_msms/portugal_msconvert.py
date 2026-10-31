"""Convert the frozen Portugal wastewater Agilent DDA datasets to mzML.

This module is the executable backend for Notebook 22.  Conversion is
incremental, one dataset at a time, and every produced mzML is parsed to EOF
and checked for MS1, MS2, centroiding, precursor metadata, and polarity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_DATASETS = 96
FILTERS = [
    # Explicit modern spelling of the vendor centroiding requested by the
    # Notebook 21 plan (whose legacy spelling was "peakPicking true 1-").
    "peakPicking vendor msLevel=1-",
    "zeroSamples removeExtra",
    "titleMaker <RunId>.<ScanNumber>",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    manifests = case / "manifests"
    return {
        "case": case,
        "nb21": manifests / "21_portugal_dda_extraction_manifest.json",
        "plan": manifests / "21_portugal_msconvert_plan.csv",
        "raw": case / "01_raw_agilent_d",
        "mzml": case / "02_mzml",
        "logs": case / "logs/msconvert_22",
        "status": manifests / "22_msconvert_file_status.csv",
        "qc": manifests / "22_msconvert_qc_checks.csv",
        "manifest": manifests / "22_msconvert_run_manifest.json",
    }


def locate_msconvert(explicit: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("MSCONVERT_EXE"):
        candidates.append(Path(os.environ["MSCONVERT_EXE"]))
    found = shutil.which("msconvert") or shutil.which("msconvert.exe")
    if found:
        candidates.append(Path(found))

    local = Path(os.environ.get("LOCALAPPDATA", "C:/Users/Default/AppData/Local"))
    glob_patterns = [
        Path("C:/Program Files/ProteoWizard"),
        Path("C:/Program Files (x86)/ProteoWizard"),
        local / "Apps",
        Path("D:/Appinstallation"),
    ]
    for base in glob_patterns:
        if not base.exists():
            continue
        candidates.extend(base.glob("msconvert.exe"))
        candidates.extend(base.glob("*/msconvert.exe"))
        candidates.extend(base.glob("ProteoWizard*/msconvert.exe"))
        candidates.extend(base.glob("ProteoWizard*/**/msconvert.exe"))

    valid = {candidate.resolve() for candidate in candidates if candidate.is_file()}
    if not valid:
        return None
    return max(valid, key=lambda candidate: candidate.stat().st_mtime)


def msconvert_version(executable: Path) -> str:
    for argument in ("--version", "--help"):
        result = subprocess.run(
            [str(executable), argument], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=60,
        )
        lines = [line.strip() for line in (result.stdout + "\n" + result.stderr).splitlines() if line.strip()]
        if result.returncode == 0 and lines and not any("unrecognised option" in line.lower() for line in lines):
            return lines[0][:500]
    match = re.search(r"ProteoWizard\s+(.+?)\s+64-bit", executable.parent.name, flags=re.IGNORECASE)
    if match:
        return f"ProteoWizard {match.group(1)} 64-bit (from installation directory)"
    return "version unavailable"


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def validate_mzml(path: Path, expected_mode: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "xml_parse_success": False,
        "spectra_n": 0,
        "ms1_n": 0,
        "ms2_n": 0,
        "positive_spectra_n": 0,
        "negative_spectra_n": 0,
        "centroid_spectra_n": 0,
        "profile_spectra_n": 0,
        "ms2_with_precursor_n": 0,
        "ms2_precursor_fraction": 0.0,
        "qc_pass": False,
        "qc_detail": "not parsed",
    }
    if not path.is_file() or path.stat().st_size == 0:
        result["qc_detail"] = "missing or empty mzML"
        return result

    try:
        for _event, element in ET.iterparse(path, events=("end",)):
            if local_name(element.tag) != "spectrum":
                continue
            result["spectra_n"] += 1
            cv = [child for child in element.iter() if local_name(child.tag) == "cvParam"]
            accessions = {child.attrib.get("accession", "") for child in cv}
            ms_level = None
            for child in cv:
                if child.attrib.get("accession") == "MS:1000511":
                    try:
                        ms_level = int(float(child.attrib.get("value", "nan")))
                    except ValueError:
                        ms_level = None
                    break
            if ms_level == 1:
                result["ms1_n"] += 1
            elif ms_level == 2:
                result["ms2_n"] += 1
                if "MS:1000744" in accessions or "MS:1000827" in accessions:
                    result["ms2_with_precursor_n"] += 1
            if "MS:1000130" in accessions:
                result["positive_spectra_n"] += 1
            if "MS:1000129" in accessions:
                result["negative_spectra_n"] += 1
            if "MS:1000127" in accessions:
                result["centroid_spectra_n"] += 1
            if "MS:1000128" in accessions:
                result["profile_spectra_n"] += 1
            element.clear()
        result["xml_parse_success"] = True
    except Exception as exc:  # Keep the exact parse failure in the audit table.
        result["qc_detail"] = f"{type(exc).__name__}: {exc}"[:1000]
        return result

    if result["ms2_n"]:
        result["ms2_precursor_fraction"] = result["ms2_with_precursor_n"] / result["ms2_n"]
    expected_polarity = result["positive_spectra_n"] if expected_mode == "Pos" else result["negative_spectra_n"]
    unexpected_polarity = result["negative_spectra_n"] if expected_mode == "Pos" else result["positive_spectra_n"]
    checks = {
        "spectra_present": result["spectra_n"] > 0,
        "ms1_present": result["ms1_n"] > 0,
        "ms2_present": result["ms2_n"] > 0,
        "expected_polarity_present": expected_polarity > 0,
        "unexpected_polarity_absent": unexpected_polarity == 0,
        "all_spectra_centroided": result["centroid_spectra_n"] == result["spectra_n"],
        "no_profile_spectra": result["profile_spectra_n"] == 0,
        "ms2_precursor_metadata_95pct": result["ms2_precursor_fraction"] >= 0.95,
    }
    result["qc_pass"] = all(checks.values())
    result["qc_detail"] = "; ".join(f"{key}={value}" for key, value in checks.items())
    return result


def load_and_validate_inputs(root: Path) -> tuple[dict[str, Path], dict[str, Any], pd.DataFrame]:
    paths = project_paths(root)
    nb21 = json.loads(paths["nb21"].read_text(encoding="utf-8"))
    if nb21.get("status") != "READY_FOR_NOTEBOOK_22" or not nb21.get("qc_all_required_pass"):
        raise RuntimeError(f"Notebook 21 is not READY_FOR_NOTEBOOK_22: {nb21.get('status')}")
    plan = pd.read_csv(paths["plan"])
    if len(plan) != EXPECTED_DATASETS or plan["dataset_id"].nunique() != EXPECTED_DATASETS:
        raise RuntimeError(f"Expected {EXPECTED_DATASETS} unique conversion rows, found {len(plan)}")
    if set(plan["mode"]) != {"Pos", "Neg"} or plan["mode"].value_counts().to_dict() != {"Neg": 48, "Pos": 48}:
        raise RuntimeError("Frozen conversion plan is not balanced at 48 Pos / 48 Neg")
    missing = [value for value in plan["input_d"] if not Path(value).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing Agilent .d inputs: {missing[:3]}")
    return paths, nb21, plan.sort_values(["mode", "dataset_id"]).reset_index(drop=True)


def run(root: Path, executable: Path, overwrite: bool = False) -> dict[str, Any]:
    paths, nb21, plan = load_and_validate_inputs(root)
    paths["mzml"].mkdir(parents=True, exist_ok=True)
    paths["logs"].mkdir(parents=True, exist_ok=True)
    version = msconvert_version(executable)
    free_before = shutil.disk_usage(paths["mzml"]).free
    if free_before < 15 * 1024**3:
        raise RuntimeError(f"At least 15 GiB free space is required; found {free_before / 1024**3:.2f} GiB")

    records: list[dict[str, Any]] = []
    stopped_after_failure = False
    for number, row in enumerate(plan.itertuples(index=False), start=1):
        output_directory = Path(row.output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        output = output_directory / row.output_filename
        log_path = paths["logs"] / f"{row.dataset_id}.log"
        existing_qc = validate_mzml(output, row.mode) if output.exists() else {"qc_pass": False}
        action = "converted"
        return_code = 0
        command: list[str] = []
        message = ""

        if output.exists() and existing_qc.get("qc_pass") and not overwrite:
            action = "skipped_valid_existing"
            qc = existing_qc
            print(f"[{number:02d}/{len(plan)}] SKIP {row.dataset_id} (valid existing mzML)", flush=True)
        else:
            temporary_name = f"{row.dataset_id}.partial.mzML"
            temporary = output_directory / temporary_name
            if temporary.exists():
                temporary.unlink()
            command = [
                str(executable), str(Path(row.input_d)), "--mzML", "--zlib", "--64",
                "--filter", FILTERS[0], "--filter", FILTERS[1], "--filter", FILTERS[2],
                "--outdir", str(output_directory), "--outfile", temporary_name,
            ]
            print(f"[{number:02d}/{len(plan)}] CONVERT {row.dataset_id}", flush=True)
            completed = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            return_code = completed.returncode
            log_text = (
                f"command: {subprocess.list2cmdline(command)}\n"
                f"return_code: {return_code}\n\nSTDOUT\n{completed.stdout}\n\nSTDERR\n{completed.stderr}\n"
            )
            log_path.write_text(log_text, encoding="utf-8")
            if not temporary.exists():
                appended = output_directory / f"{temporary_name}.mzML"
                if appended.exists():
                    temporary = appended
            qc = validate_mzml(temporary, row.mode)
            if return_code == 0 and qc.get("qc_pass"):
                os.replace(temporary, output)
                qc = validate_mzml(output, row.mode)
            else:
                action = "failed"
                message = (completed.stderr or completed.stdout)[-2000:]
                stopped_after_failure = True

        record = {
            "dataset_id": row.dataset_id,
            "mode": row.mode,
            "site_code": row.site_code,
            "campaign": int(row.campaign),
            "injection": row.injection,
            "input_d": row.input_d,
            "output_mzml": str(output),
            "action": action,
            "return_code": return_code,
            "command": subprocess.list2cmdline(command) if command else "",
            "log_path": str(log_path) if log_path.exists() else "",
            "failure_tail": message,
            **qc,
        }
        if output.is_file() and qc.get("qc_pass"):
            record["mzml_size_bytes"] = output.stat().st_size
            record["mzml_sha256"] = sha256_file(output)
        else:
            record["mzml_size_bytes"] = 0
            record["mzml_sha256"] = ""
        records.append(record)
        atomic_csv(pd.DataFrame(records), paths["status"])
        if stopped_after_failure:
            print(f"Stopped after first failure. Inspect: {log_path}", flush=True)
            break

    status_frame = pd.DataFrame(records)
    success_n = int(status_frame.get("qc_pass", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    complete = len(records) == EXPECTED_DATASETS and success_n == EXPECTED_DATASETS
    mode_success = (
        status_frame.loc[status_frame.get("qc_pass", False).fillna(False).astype(bool), "mode"].value_counts().to_dict()
        if not status_frame.empty else {}
    )
    qc_rows = [
        {"check": "notebook21_ready", "passed": nb21.get("status") == "READY_FOR_NOTEBOOK_22", "detail": nb21.get("status")},
        {"check": "msconvert_executable_present", "passed": executable.is_file(), "detail": str(executable)},
        {"check": "conversion_plan_has_96_unique_datasets", "passed": len(plan) == EXPECTED_DATASETS and plan["dataset_id"].nunique() == EXPECTED_DATASETS, "detail": len(plan)},
        {"check": "all_96_mzml_pass_file_qc", "passed": complete, "detail": f"{success_n}/{EXPECTED_DATASETS}"},
        {"check": "mode_balance_48_48", "passed": mode_success == {"Neg": 48, "Pos": 48}, "detail": str(mode_success)},
        {"check": "no_profile_spectra", "passed": complete and status_frame["profile_spectra_n"].sum() == 0, "detail": int(status_frame["profile_spectra_n"].sum()) if not status_frame.empty else 0},
        {"check": "all_spectra_centroided", "passed": complete and status_frame["centroid_spectra_n"].sum() == status_frame["spectra_n"].sum(), "detail": "per-file check"},
        {"check": "every_file_contains_ms1_and_ms2", "passed": complete and status_frame[["ms1_n", "ms2_n"]].gt(0).all().all(), "detail": "per-file check"},
        {"check": "every_file_has_95pct_ms2_precursors", "passed": complete and status_frame["ms2_precursor_fraction"].ge(0.95).all(), "detail": "per-file check"},
    ]
    qc_frame = pd.DataFrame(qc_rows)
    atomic_csv(qc_frame, paths["qc"])
    final_status = "READY_FOR_NOTEBOOK_23" if bool(qc_frame["passed"].all()) else "BLOCKED_MS_CONVERSION"
    manifest = {
        "version": "PBT_MSMS_PORTUGAL_MSCONVERT_22_V1",
        "generated_at_utc": utc_now(),
        "status": final_status,
        "scientific_scope": "Vendor-centroided mzML conversion of 96 frozen Portugal wastewater AutoMSMS DDA datasets",
        "notebook21_manifest_sha256": sha256_file(paths["nb21"]),
        "conversion_plan_sha256": sha256_file(paths["plan"]),
        "msconvert": {"executable": str(executable), "version": version, "filters_in_order": FILTERS},
        "encoding": {"format": "mzML", "binary_precision": "64-bit", "compression": "zlib"},
        "counts": {"planned": EXPECTED_DATASETS, "processed_this_run": len(records), "passed": success_n, "passed_by_mode": mode_success},
        "disk_free_bytes_before": free_before,
        "stopped_after_failure": stopped_after_failure,
        "qc_all_pass": bool(qc_frame["passed"].all()),
        "artifacts": {
            paths["status"].name: {"sha256": sha256_file(paths["status"]), "size_bytes": paths["status"].stat().st_size},
            paths["qc"].name: {"sha256": sha256_file(paths["qc"]), "size_bytes": paths["qc"].stat().st_size},
        },
        "next_step": "Notebook 23: mzML feature extraction, DDA MS/MS association, and sample-level QC",
    }
    atomic_json(paths["manifest"], manifest)
    print(qc_frame.to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if final_status != "READY_FOR_NOTEBOOK_23":
        raise RuntimeError(f"Notebook 22 did not pass: {success_n}/{EXPECTED_DATASETS} mzML files valid")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--msconvert", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--locate-only", action="store_true")
    args = parser.parse_args()
    executable = locate_msconvert(args.msconvert)
    if executable is None:
        raise FileNotFoundError(
            "msconvert.exe was not found. Install 64-bit ProteoWizard with vendor readers, "
            "then restart Jupyter; or pass --msconvert with its full path."
        )
    print(f"msconvert: {executable}")
    print(f"version: {msconvert_version(executable)}")
    if not args.locate_only:
        run(locate_root(args.root), executable, args.overwrite)


if __name__ == "__main__":
    main()
