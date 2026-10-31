"""Export the frozen 20 ppm Development spectra as leakage-safe SIRIUS inputs.

The exported ``.ms`` files contain only information available from an unknown
LC-HRMS/MS feature: precursor m/z, ion/adduct annotation, collision energy (if
available), and fragment peaks.  True structures and molecular formulae are
kept out of the SIRIUS inputs.  ChemVista spectra are never opened here.

Run in ``env_ms`` from the project root::

    python -m pbt_msms.sirius_export
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pbt_msms.dreams_mvp import PROJECT_ROOT, PUBLIC_STREAM, extract_peaks, sha256_file


EXPORT_VERSION = "SIRIUS_DEVELOPMENT_INPUT_V1"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pbt_msms_v1" / "sirius_v1" / "development_inputs_v1"
POSITIVE_DIR = OUTPUT_DIR / "positive_ms"
NEGATIVE_DIR = OUTPUT_DIR / "negative_ms"
INPUT_INDEX = (
    PROJECT_ROOT
    / "outputs"
    / "pbt_msms_v1"
    / "dreams_v1"
    / "batch_embeddings_v1"
    / "development_dreams_input_index.csv.gz"
)
DEVELOPMENT_COMPOUNDS = (
    PROJECT_ROOT / "outputs" / "pbt_msms_v1" / "split_v1" / "development_compounds.csv"
)
UPSTREAM_AUDIT = (
    PROJECT_ROOT
    / "outputs"
    / "pbt_msms_v1"
    / "spectrum_audit_v1"
    / "spectrum_audit_manifest.json"
)
UPSTREAM_QC = (
    PROJECT_ROOT
    / "outputs"
    / "pbt_msms_v1"
    / "spectrum_audit_v1"
    / "spectrum_audit_qc_checks.csv"
)

FEATURE_MANIFEST = OUTPUT_DIR / "development_sirius_feature_manifest.csv.gz"
SELECTED_MANIFEST = OUTPUT_DIR / "development_sirius_selected_spectra.csv.gz"
ADDUCT_AUDIT = OUTPUT_DIR / "development_sirius_adduct_audit.csv"
QC_PATH = OUTPUT_DIR / "development_sirius_input_qc_checks.csv"
FAILURES_PATH = OUTPUT_DIR / "development_sirius_export_failures.csv"
RUN_MANIFEST = OUTPUT_DIR / "development_sirius_input_manifest.json"
README_PATH = OUTPUT_DIR / "SIRIUS_运行说明.md"
POSITIVE_LIST = OUTPUT_DIR / "positive_input_file_list.txt"
NEGATIVE_LIST = OUTPUT_DIR / "negative_input_file_list.txt"

EXPECTED_SPECTRA = 58_687
EXPECTED_COMPOUNDS = 1_230
EXPECTED_POSITIVE_COMPOUNDS = 1_080
EXPECTED_NEGATIVE_COMPOUNDS = 548
EXPECTED_BOTH_MODE_COMPOUNDS = 398
PUBLIC_LIBRARIES = {"GNPS", "MassBank", "MoNA"}
ALLOWED_EVALUATION_ROLES = {"none", "transfer"}
CLUSTER_PPM = 20.0
MAX_SPECTRA_PER_FEATURE = 3
KNOWN_CE_MINIMUM = 0.0

# SIRIUS v6 nomenclature uses elemental formulae rather than informal
# abbreviations.  The remaining observed adducts already follow that syntax.
ADDUCT_NORMALIZATION = {
    "[M]+.": "[M]+",
    "[M]-.": "[M]-",
    "[M+FA]-": "[M+CHO2]-",
    "[M+Ac]-": "[M+C2H3O2]-",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, path)


def atomic_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.GzipFile(temporary, mode="wb", compresslevel=6, mtime=0) as compressed:
        frame.to_csv(compressed, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(temporary, path)


def hash_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def normalize_ion_mode(value: Any) -> str:
    text = str(value).strip().lower()
    if text.startswith("pos") or text == "+":
        return "Positive"
    if text.startswith("neg") or text == "-":
        return "Negative"
    raise ValueError(f"Unsupported ion mode: {value}")


def normalize_adduct(value: Any, ion_mode: str) -> tuple[str, str]:
    raw = re.sub(r"\s+", "", str(value).strip())
    if not raw or raw.lower() in {"nan", "none", "unknown"}:
        normalized = "[M+?]+" if ion_mode == "Positive" else "[M+?]-"
        return normalized, "fallback_unknown_by_mode"
    normalized = ADDUCT_NORMALIZATION.get(raw, raw)
    method = "explicit_mapping" if normalized != raw else "unchanged"
    if not re.fullmatch(r"\[M[^\]]*\][+-]", normalized):
        raise ValueError(f"Adduct does not follow singly charged SIRIUS syntax: {raw}")
    expected_sign = "+" if ion_mode == "Positive" else "-"
    if not normalized.endswith(expected_sign):
        raise ValueError(f"Adduct/ion-mode conflict: {raw}, {ion_mode}")
    return normalized, method


def ppm_error(values: np.ndarray, reference: float) -> np.ndarray:
    return np.abs(values - reference) / reference * 1e6


def precursor_clusters(group: pd.DataFrame) -> pd.Series:
    """Deterministically split an adduct group into <=20 ppm precursor clusters."""
    ordered = group.sort_values(["precursor_mz", "spectrum_id"]).copy()
    clusters: list[list[int]] = []
    for row_index, precursor in zip(ordered.index, ordered["precursor_mz"].astype(float)):
        if not clusters:
            clusters.append([int(row_index)])
            continue
        trial = clusters[-1] + [int(row_index)]
        trial_values = group.loc[trial, "precursor_mz"].astype(float).to_numpy()
        median = float(np.median(trial_values))
        if float(np.max(ppm_error(trial_values, median))) <= CLUSTER_PPM + 1e-9:
            clusters[-1].append(int(row_index))
        else:
            clusters.append([int(row_index)])
    assignments: dict[int, int] = {}
    for cluster_number, indices in enumerate(clusters, start=1):
        assignments.update({idx: cluster_number for idx in indices})
    return pd.Series(assignments, dtype="int64")


def _quality_sorted(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["_abs_identity_ppm"] = pd.to_numeric(
        ranked["identity_ppm_error"], errors="coerce"
    ).abs().fillna(np.inf)
    ranked["_peak_n"] = pd.to_numeric(
        ranked["exported_peak_n"], errors="coerce"
    ).fillna(-1)
    ranked = ranked.sort_values(
        ["_peak_n", "_abs_identity_ppm", "source_library", "spectrum_id"],
        ascending=[False, True, True, True],
    )
    return ranked.drop(columns=["_abs_identity_ppm", "_peak_n"])


def select_representative_spectra(group: pd.DataFrame) -> pd.DataFrame:
    """Choose at most low/mid/high CE spectra, with deterministic quality fills."""
    candidates = _quality_sorted(group)
    candidates = candidates.drop_duplicates("spectrum_hash_common", keep="first").copy()
    candidates["_ce"] = pd.to_numeric(candidates["collision_energy_value"], errors="coerce")
    known = candidates.loc[np.isfinite(candidates["_ce"]) & (candidates["_ce"] >= KNOWN_CE_MINIMUM)]

    selected_indices: list[int] = []
    selected_roles: dict[int, str] = {}
    if not known.empty:
        ce_values = known["_ce"].astype(float).to_numpy()
        targets = [
            (float(np.min(ce_values)), "ce_low"),
            (float(np.median(ce_values)), "ce_mid"),
            (float(np.max(ce_values)), "ce_high"),
        ]
        for target, role in targets:
            available = known.loc[~known.index.isin(selected_indices)].copy()
            if available.empty:
                break
            available["_distance"] = (available["_ce"] - target).abs()
            available = available.sort_values(
                ["_distance", "exported_peak_n", "identity_ppm_error", "spectrum_id"],
                ascending=[True, False, True, True],
            )
            chosen = int(available.index[0])
            selected_indices.append(chosen)
            selected_roles[chosen] = role

    # Prefer acquisition-source diversity when CE is unknown or non-diverse.
    for _, source_group in candidates.loc[~candidates.index.isin(selected_indices)].groupby(
        "source_library", sort=True
    ):
        if len(selected_indices) >= MAX_SPECTRA_PER_FEATURE:
            break
        chosen = int(_quality_sorted(source_group).index[0])
        selected_indices.append(chosen)
        selected_roles[chosen] = "unknown_ce_or_source_diversity"

    for idx in candidates.index:
        if len(selected_indices) >= MAX_SPECTRA_PER_FEATURE:
            break
        idx = int(idx)
        if idx not in selected_indices:
            selected_indices.append(idx)
            selected_roles[idx] = "quality_fill"

    selected = group.loc[selected_indices].copy()
    selected["selection_role"] = [selected_roles[int(idx)] for idx in selected.index]
    role_order = {
        "ce_low": 0,
        "ce_mid": 1,
        "ce_high": 2,
        "unknown_ce_or_source_diversity": 3,
        "quality_fill": 4,
    }
    selected["_role_order"] = selected["selection_role"].map(role_order)
    selected = selected.sort_values(["_role_order", "spectrum_id"]).drop(columns="_role_order")
    selected["spectrum_position_in_feature"] = np.arange(1, len(selected) + 1)
    return selected


def load_and_validate_inputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    for path in [INPUT_INDEX, DEVELOPMENT_COMPOUNDS, UPSTREAM_AUDIT, UPSTREAM_QC, PUBLIC_STREAM]:
        if not path.exists():
            raise FileNotFoundError(path)
    upstream_qc = pd.read_csv(UPSTREAM_QC)
    if upstream_qc["status"].astype(str).eq("FAIL").any():
        raise AssertionError("Upstream spectrum audit contains FAIL")
    upstream = read_json(UPSTREAM_AUDIT)
    index = pd.read_csv(INPUT_INDEX)
    development = pd.read_csv(DEVELOPMENT_COMPOUNDS, usecols=["inchikey_2d"])

    if len(index) != EXPECTED_SPECTRA or index["inchikey_2d"].nunique() != EXPECTED_COMPOUNDS:
        raise AssertionError("Frozen DreaMS index no longer has 58,687 spectra / 1,230 compounds")
    if len(index) != int(upstream["counts"]["primary_train_20ppm_spectra"]):
        raise AssertionError("DreaMS index differs from upstream 20 ppm spectrum count")
    if index["inchikey_2d"].nunique() != int(
        upstream["counts"]["primary_train_20ppm_compounds"]
    ):
        raise AssertionError("DreaMS index differs from upstream 20 ppm compound count")
    if not set(index["inchikey_2d"]).issubset(set(development["inchikey_2d"])):
        raise AssertionError("A spectrum identity is absent from the frozen Development split")
    if not index["supervised_compound_role"].astype(str).eq("development").all():
        raise AssertionError("Non-Development identity entered SIRIUS export")
    roles = set(index["chemvista_evaluation_role"].astype(str).unique())
    if not roles.issubset(ALLOWED_EVALUATION_ROLES):
        raise AssertionError(f"ChemVista novel identity entered SIRIUS export: {roles}")
    if not index["primary_train_eligible_20ppm"].fillna(False).astype(bool).all():
        raise AssertionError("Non-primary (50/100 ppm) spectrum entered SIRIUS export")
    if not set(index["source_library"].astype(str)).issubset(PUBLIC_LIBRARIES):
        raise AssertionError("Non-public spectrum source entered SIRIUS export")
    if index["spectrum_id"].duplicated().any():
        raise AssertionError("Duplicate spectrum_id in frozen input")

    index["ion_mode"] = index["ion_mode"].map(normalize_ion_mode)
    normalized = [normalize_adduct(a, m) for a, m in zip(index["adduct"], index["ion_mode"])]
    index["sirius_adduct"] = [item[0] for item in normalized]
    index["adduct_normalization_method"] = [item[1] for item in normalized]
    index["precursor_mz"] = pd.to_numeric(index["precursor_mz"], errors="raise")
    if (~np.isfinite(index["precursor_mz"]) | (index["precursor_mz"] <= 0)).any():
        raise AssertionError("Invalid precursor m/z")
    if not index["charge"].astype(int).abs().eq(1).all():
        raise AssertionError("SIRIUS v6 input contains a non-singly charged spectrum")
    return index, upstream


def build_feature_tables(index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = index.copy()
    index["precursor_cluster"] = -1
    group_columns = ["inchikey_2d", "ion_mode", "sirius_adduct"]
    for _, group in index.groupby(group_columns, sort=True, dropna=False):
        assignments = precursor_clusters(group)
        index.loc[assignments.index, "precursor_cluster"] = assignments.astype(int)
    if (index["precursor_cluster"] < 1).any():
        raise AssertionError("Precursor clustering left unassigned rows")

    feature_groups = [*group_columns, "precursor_cluster"]
    feature_rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    mode_counters = {"Positive": 0, "Negative": 0}
    grouped = index.groupby(feature_groups, sort=True, dropna=False)
    for key, group in grouped:
        inchikey_2d, ion_mode, sirius_adduct, precursor_cluster = key
        mode_counters[ion_mode] += 1
        prefix = "POS" if ion_mode == "Positive" else "NEG"
        feature_id = f"DEV_{prefix}_{mode_counters[ion_mode]:06d}"
        selected = select_representative_spectra(group)
        selected["sirius_feature_id"] = feature_id
        selected_frames.append(selected)

        precursor_values = group["precursor_mz"].astype(float).to_numpy()
        parentmass = float(np.median(precursor_values))
        selected_ce = pd.to_numeric(selected["collision_energy_value"], errors="coerce")
        feature_rows.append(
            {
                "sirius_feature_id": feature_id,
                "inchikey_2d": inchikey_2d,
                "ion_mode": ion_mode,
                "sirius_adduct": sirius_adduct,
                "precursor_cluster": int(precursor_cluster),
                "parentmass_median": parentmass,
                "precursor_min": float(np.min(precursor_values)),
                "precursor_max": float(np.max(precursor_values)),
                "max_parentmass_deviation_ppm": float(np.max(ppm_error(precursor_values, parentmass))),
                "available_spectrum_n": int(len(group)),
                "unique_exact_spectrum_n": int(group["spectrum_hash_common"].nunique()),
                "selected_spectrum_n": int(len(selected)),
                "selected_known_ce_n": int((np.isfinite(selected_ce) & (selected_ce >= 0)).sum()),
                "cv_group": str(group["cv_group"].iloc[0]),
                "cv_fold": int(group["cv_fold"].iloc[0]),
                "chemvista_evaluation_role": str(group["chemvista_evaluation_role"].iloc[0]),
                "supervised_compound_role": str(group["supervised_compound_role"].iloc[0]),
            }
        )

    features = pd.DataFrame(feature_rows)
    selected = pd.concat(selected_frames, ignore_index=False).copy()
    selected = selected.merge(
        features[["sirius_feature_id", "parentmass_median"]], on="sirius_feature_id", how="left"
    )
    selected = selected.sort_values(
        ["ion_mode", "sirius_feature_id", "spectrum_position_in_feature"]
    ).reset_index(drop=True)
    selected["sirius_input_row"] = np.arange(len(selected), dtype=int)
    return features, selected


def clear_stale_ms_files(directory: Path) -> None:
    resolved_output = OUTPUT_DIR.resolve()
    resolved_directory = directory.resolve()
    if resolved_output not in resolved_directory.parents:
        raise AssertionError(f"Refusing to clean outside fixed SIRIUS output: {resolved_directory}")
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.ms"):
        path.unlink()


def format_ms_file(feature: pd.Series, spectra: list[tuple[pd.Series, np.ndarray]]) -> str:
    lines = [
        f">compound {feature['sirius_feature_id']}",
        f">feature_id {feature['sirius_feature_id']}",
        f">parentmass {float(feature['parentmass_median']):.8f}",
        f">ionization {feature['sirius_adduct']}",
        "",
    ]
    for row, peaks in spectra:
        ce = pd.to_numeric(row["collision_energy_value"], errors="coerce")
        if np.isfinite(ce) and float(ce) >= KNOWN_CE_MINIMUM:
            lines.append(f">collision {float(ce):.8g}")
        else:
            lines.append(">ms2")
        lines.extend(f"{mz:.8f} {intensity:.8f}" for mz, intensity in peaks)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_ms_files(
    features: pd.DataFrame, selected: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    clear_stale_ms_files(POSITIVE_DIR)
    clear_stale_ms_files(NEGATIVE_DIR)
    selected = selected.copy()
    selected["record_peak_n"] = -1
    selected["exported_peak_n_sirius"] = -1
    selected["ms_file_sha256"] = ""
    selected["ms_file_relative_path"] = ""
    selected["record_source_library"] = ""
    failures: list[dict[str, Any]] = []

    feature_lookup = features.set_index("sirius_feature_id")
    grouped_selected = selected.groupby("sirius_feature_id", sort=True)
    with PUBLIC_STREAM.open("rb") as stream:
        for feature_id, group in tqdm(
            grouped_selected, total=selected["sirius_feature_id"].nunique(), desc="导出SIRIUS .ms"
        ):
            feature = feature_lookup.loc[feature_id].copy()
            # The identifier became the index in ``feature_lookup``; restore it
            # explicitly for the anonymous metadata written into the .ms file.
            feature["sirius_feature_id"] = feature_id
            spectra: list[tuple[pd.Series, np.ndarray]] = []
            try:
                for row_index, row in group.sort_values("spectrum_position_in_feature").iterrows():
                    stream.seek(int(row["stream_offset"]))
                    actual_key, record = pickle.load(stream)
                    if str(actual_key) != str(row["inchikey_2d"]):
                        raise ValueError(f"stream IK2D mismatch for {row['spectrum_id']}")
                    if str(record.get("inchikey_2d")) != str(row["inchikey_2d"]):
                        raise ValueError(f"record IK2D mismatch for {row['spectrum_id']}")
                    source_library = str(record.get("source_library"))
                    if source_library not in PUBLIC_LIBRARIES:
                        raise ValueError(f"non-public source for {row['spectrum_id']}: {source_library}")
                    if source_library != str(row["source_library"]):
                        raise ValueError(f"source-library mismatch for {row['spectrum_id']}")
                    peaks = extract_peaks(record)
                    record_peak_n = int(record.get("cleaned_peak_n", -1))
                    if record_peak_n != int(row["exported_peak_n"]) or len(peaks) != record_peak_n:
                        raise ValueError(f"peak-count mismatch for {row['spectrum_id']}")
                    if not np.isfinite(peaks).all() or (peaks <= 0).any():
                        raise ValueError(f"invalid peak values for {row['spectrum_id']}")
                    spectra.append((row, peaks))
                    selected.loc[row_index, "record_peak_n"] = record_peak_n
                    selected.loc[row_index, "exported_peak_n_sirius"] = len(peaks)
                    selected.loc[row_index, "record_source_library"] = source_library

                directory = POSITIVE_DIR if feature["ion_mode"] == "Positive" else NEGATIVE_DIR
                output_path = directory / f"{feature_id}.ms"
                atomic_text(output_path, format_ms_file(feature, spectra))
                relative_path = str(output_path.relative_to(PROJECT_ROOT))
                file_hash = sha256_file(output_path)
                selected.loc[group.index, "ms_file_relative_path"] = relative_path
                selected.loc[group.index, "ms_file_sha256"] = file_hash
            except Exception as exc:
                failures.append(
                    {
                        "sirius_feature_id": feature_id,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
    return selected, failures


def validate_ms_file(path: Path, expected_feature_id: str, expected_spectra: int) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    required = {
        f">compound {expected_feature_id}",
        f">feature_id {expected_feature_id}",
    }
    if not required.issubset(set(lines)):
        errors.append("missing_or_wrong_feature_metadata")
    if sum(line.startswith(">parentmass ") for line in lines) != 1:
        errors.append("parentmass_count_not_one")
    if sum(line.startswith(">ionization ") for line in lines) != 1:
        errors.append("ionization_count_not_one")
    n_ms2 = sum(line == ">ms2" or line.startswith(">collision ") for line in lines)
    if n_ms2 != int(expected_spectra):
        errors.append("ms2_block_count_mismatch")
    forbidden = [">formula", "SMILES", "InChI", "inchikey"]
    if any(token.lower() in text.lower() for token in forbidden):
        errors.append("forbidden_structure_or_formula_metadata")
    allowed_meta = (">compound ", ">feature_id ", ">parentmass ", ">ionization ", ">collision ")
    for line in lines:
        if line.startswith(">") and line != ">ms2" and not line.startswith(allowed_meta):
            errors.append("unknown_ms_metadata")
            break
    return errors


def make_qc(
    index: pd.DataFrame,
    features: pd.DataFrame,
    selected: pd.DataFrame,
    failures: list[dict[str, Any]],
) -> pd.DataFrame:
    file_errors: list[tuple[str, str]] = []
    for feature in features.itertuples(index=False):
        directory = POSITIVE_DIR if feature.ion_mode == "Positive" else NEGATIVE_DIR
        path = directory / f"{feature.sirius_feature_id}.ms"
        if not path.exists():
            file_errors.append((feature.sirius_feature_id, "missing_file"))
            continue
        for error in validate_ms_file(path, feature.sirius_feature_id, feature.selected_spectrum_n):
            file_errors.append((feature.sirius_feature_id, error))

    compound_modes = features.groupby("inchikey_2d")["ion_mode"].agg(set)
    n_both = int(compound_modes.map(lambda modes: modes == {"Positive", "Negative"}).sum())
    n_pos_compounds = int(features.loc[features["ion_mode"] == "Positive", "inchikey_2d"].nunique())
    n_neg_compounds = int(features.loc[features["ion_mode"] == "Negative", "inchikey_2d"].nunique())
    n_pos_files = len(list(POSITIVE_DIR.glob("*.ms")))
    n_neg_files = len(list(NEGATIVE_DIR.glob("*.ms")))

    checks: list[dict[str, Any]] = []

    def add(check: str, passed: bool, observed: Any, expected: Any) -> None:
        checks.append(
            {
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "observed": observed,
                "expected": expected,
            }
        )

    add("frozen_spectrum_count", len(index) == EXPECTED_SPECTRA, len(index), EXPECTED_SPECTRA)
    add(
        "frozen_compound_count",
        index["inchikey_2d"].nunique() == EXPECTED_COMPOUNDS,
        index["inchikey_2d"].nunique(),
        EXPECTED_COMPOUNDS,
    )
    add(
        "all_20ppm_primary",
        index["primary_train_eligible_20ppm"].astype(bool).all(),
        int(index["primary_train_eligible_20ppm"].astype(bool).sum()),
        len(index),
    )
    add(
        "public_spectrum_sources_only",
        set(index["source_library"].astype(str)).issubset(PUBLIC_LIBRARIES),
        "|".join(sorted(set(index["source_library"].astype(str)))),
        "GNPS|MassBank|MoNA",
    )
    add(
        "no_chemvista_novel_identity",
        set(index["chemvista_evaluation_role"].astype(str)).issubset(ALLOWED_EVALUATION_ROLES),
        "|".join(sorted(set(index["chemvista_evaluation_role"].astype(str)))),
        "none|transfer",
    )
    add("positive_compound_coverage", n_pos_compounds == EXPECTED_POSITIVE_COMPOUNDS, n_pos_compounds, EXPECTED_POSITIVE_COMPOUNDS)
    add("negative_compound_coverage", n_neg_compounds == EXPECTED_NEGATIVE_COMPOUNDS, n_neg_compounds, EXPECTED_NEGATIVE_COMPOUNDS)
    add("both_mode_compound_coverage", n_both == EXPECTED_BOTH_MODE_COMPOUNDS, n_both, EXPECTED_BOTH_MODE_COMPOUNDS)
    add("all_compounds_exported", features["inchikey_2d"].nunique() == EXPECTED_COMPOUNDS, features["inchikey_2d"].nunique(), EXPECTED_COMPOUNDS)
    add("unique_feature_ids", not features["sirius_feature_id"].duplicated().any(), features["sirius_feature_id"].nunique(), len(features))
    add("positive_file_count", n_pos_files == int((features["ion_mode"] == "Positive").sum()), n_pos_files, int((features["ion_mode"] == "Positive").sum()))
    add("negative_file_count", n_neg_files == int((features["ion_mode"] == "Negative").sum()), n_neg_files, int((features["ion_mode"] == "Negative").sum()))
    add("one_to_three_spectra_per_feature", features["selected_spectrum_n"].between(1, MAX_SPECTRA_PER_FEATURE).all(), f"min={features['selected_spectrum_n'].min()},max={features['selected_spectrum_n'].max()}", "1..3")
    add("precursor_cluster_within_20ppm", (features["max_parentmass_deviation_ppm"] <= CLUSTER_PPM + 1e-9).all(), float(features["max_parentmass_deviation_ppm"].max()), f"<={CLUSTER_PPM}")
    add("selected_peak_counts_match", selected["record_peak_n"].eq(selected["exported_peak_n_sirius"]).all() and selected["record_peak_n"].gt(0).all(), int((selected["record_peak_n"] == selected["exported_peak_n_sirius"]).sum()), len(selected))
    add("selected_sources_match", selected["source_library"].astype(str).eq(selected["record_source_library"].astype(str)).all(), int(selected["source_library"].astype(str).eq(selected["record_source_library"].astype(str)).sum()), len(selected))
    add("no_export_failures", len(failures) == 0, len(failures), 0)
    add("all_ms_files_parse_contract", len(file_errors) == 0, len(file_errors), 0)
    add("no_structure_or_formula_in_ms", not any(error == "forbidden_structure_or_formula_metadata" for _, error in file_errors), sum(error == "forbidden_structure_or_formula_metadata" for _, error in file_errors), 0)
    return pd.DataFrame(checks)


def directory_hash(directory: Path) -> str:
    entries = [f"{path.name}|{sha256_file(path)}" for path in sorted(directory.glob("*.ms"))]
    return hash_lines(entries)


def write_supporting_files(
    index: pd.DataFrame,
    features: pd.DataFrame,
    selected: pd.DataFrame,
    qc: pd.DataFrame,
    upstream: dict[str, Any],
) -> dict[str, Any]:
    feature_export = features.copy()
    feature_export["ms_file_relative_path"] = feature_export.apply(
        lambda row: str(
            (
                (POSITIVE_DIR if row["ion_mode"] == "Positive" else NEGATIVE_DIR)
                / f"{row['sirius_feature_id']}.ms"
            ).relative_to(PROJECT_ROOT)
        ),
        axis=1,
    )
    feature_export["ms_file_sha256"] = feature_export["ms_file_relative_path"].map(
        lambda path: sha256_file(PROJECT_ROOT / path)
    )
    atomic_csv_gzip(feature_export, FEATURE_MANIFEST)

    selected_columns = [
        "sirius_input_row",
        "sirius_feature_id",
        "spectrum_position_in_feature",
        "selection_role",
        "spectrum_id",
        "inchikey_2d",
        "stream_offset",
        "cv_group",
        "cv_fold",
        "source_library",
        "record_source_library",
        "ion_mode",
        "adduct",
        "sirius_adduct",
        "adduct_normalization_method",
        "charge",
        "precursor_mz",
        "parentmass_median",
        "precursor_cluster",
        "collision_energy_value",
        "record_peak_n",
        "exported_peak_n_sirius",
        "identity_ppm_error",
        "spectrum_hash_common",
        "peak_hash_common",
        "chemvista_evaluation_role",
        "supervised_compound_role",
        "primary_train_eligible_20ppm",
        "ms_file_relative_path",
        "ms_file_sha256",
    ]
    atomic_csv_gzip(selected[selected_columns], SELECTED_MANIFEST)

    adduct_audit = (
        index.groupby(
            ["ion_mode", "adduct", "sirius_adduct", "adduct_normalization_method"],
            dropna=False,
        )
        .agg(spectrum_n=("spectrum_id", "size"), compound_n=("inchikey_2d", "nunique"))
        .reset_index()
        .sort_values(["ion_mode", "spectrum_n"], ascending=[True, False])
    )
    atomic_csv(adduct_audit, ADDUCT_AUDIT)
    atomic_csv(qc, QC_PATH)

    pos_paths = feature_export.loc[feature_export["ion_mode"] == "Positive", "ms_file_relative_path"]
    neg_paths = feature_export.loc[feature_export["ion_mode"] == "Negative", "ms_file_relative_path"]
    atomic_text(POSITIVE_LIST, "\n".join(pos_paths.astype(str)) + "\n")
    atomic_text(NEGATIVE_LIST, "\n".join(neg_paths.astype(str)) + "\n")

    readme = f"""# SIRIUS Development 输入（{EXPORT_VERSION}）

本目录只包含冻结的公共谱图 Development 数据：{EXPECTED_COMPOUNDS:,} 个化合物，ChemVista 谱图未被读取或导出。

## 推荐手动运行方式

1. 在 SIRIUS 6 GUI 中新建项目，先导入 `positive_ms` 整个目录；负模式在另一个项目中导入 `negative_ms`。
2. 导入后先核对 feature 数、母离子、离子模式和加合物。不要补入真实分子式；主分析必须让 SIRIUS 从谱图推断分子式。
3. 选择 Molecular Formula Identification 和 CSI:FingerID Fingerprint Prediction；不需要运行 structure database search 才能完成本课题的概率指纹路线。
4. 主分析采用 20 ppm 上限，并在 SIRIUS 中保存 preset；点击 **Show Command** 和 **Show JSON**，把完整参数、SIRIUS 精确版本、指纹维数保存到结果目录。
5. 正、负模式分别导出项目和结果。不得根据 P/B/T 标签或 ChemVista 表现调整 SIRIUS 参数。

## 输入语义

- 一个 `.ms` 文件是一个 SIRIUS feature：同一 IK2D、离子模式、规范化加合物及 20 ppm 母离子质量簇。
- 每个 feature 最多包含 3 张代表谱图：优先低/中/高碰撞能；碰撞能为 `-1` 时按未知处理并使用 `>ms2`。
- `.ms` 文件使用匿名 `DEV_POS/NEG_*` ID，不含 IK2D、SMILES 或正确分子式；真实映射只在清单中保存。
- `transfer` 表示相同化合物可存在于 Development 公共谱图中，但 ChemVista 实验谱图仍被锁定；`novel` 身份没有进入本输入。

## 关键文件

- `development_sirius_feature_manifest.csv.gz`：feature 到化合物、fold、输入文件的映射。
- `development_sirius_selected_spectra.csv.gz`：每个输入 feature 实际使用的谱图。
- `development_sirius_adduct_audit.csv`：加合物标准化审计。
- `development_sirius_input_qc_checks.csv`：导出前质量控制。
- `development_sirius_input_manifest.json`：哈希、版本、数量和方法冻结记录。

只有在 GUI/CLI 成功导入并且 feature 数与清单一致后，才进入“批量计算概率指纹”步骤。
"""
    atomic_text(README_PATH, readme)

    manifest = {
        "export_version": EXPORT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": "20 ppm public-spectrum Development only; no ChemVista spectra",
        "input_hashes": {
            "development_dreams_input_index": sha256_file(INPUT_INDEX),
            "development_compounds": sha256_file(DEVELOPMENT_COMPOUNDS),
            "upstream_spectrum_audit": sha256_file(UPSTREAM_AUDIT),
            "public_stream": upstream["public_stream"]["sha256"],
        },
        "method": {
            "feature_group": "IK2D + ion mode + normalized adduct + precursor cluster",
            "precursor_cluster_max_deviation_ppm": CLUSTER_PPM,
            "max_spectra_per_feature": MAX_SPECTRA_PER_FEATURE,
            "representative_selection": "known CE low/median/high, then source-diverse quality fill",
            "collision_energy_minus_one_semantics": "missing; exported as >ms2",
            "parentmass": "median of all available spectra in feature cluster",
            "formula_in_input": False,
            "structure_in_input": False,
            "identifier_in_ms": "anonymous DEV_POS/NEG sequence",
        },
        "counts": {
            "frozen_available_spectra": int(len(index)),
            "frozen_development_compounds": int(index["inchikey_2d"].nunique()),
            "selected_spectra": int(len(selected)),
            "sirius_features": int(len(features)),
            "positive_features": int((features["ion_mode"] == "Positive").sum()),
            "negative_features": int((features["ion_mode"] == "Negative").sum()),
            "positive_compounds": int(features.loc[features["ion_mode"] == "Positive", "inchikey_2d"].nunique()),
            "negative_compounds": int(features.loc[features["ion_mode"] == "Negative", "inchikey_2d"].nunique()),
            "both_mode_compounds": int(features.groupby("inchikey_2d")["ion_mode"].agg(set).map(lambda modes: modes == {"Positive", "Negative"}).sum()),
            "features_with_known_collision_energy": int((features["selected_known_ce_n"] > 0).sum()),
        },
        "output_hashes": {
            "feature_manifest": sha256_file(FEATURE_MANIFEST),
            "selected_spectrum_manifest": sha256_file(SELECTED_MANIFEST),
            "adduct_audit": sha256_file(ADDUCT_AUDIT),
            "qc_checks": sha256_file(QC_PATH),
            "positive_ms_directory": directory_hash(POSITIVE_DIR),
            "negative_ms_directory": directory_hash(NEGATIVE_DIR),
        },
        "sirius_execution": {
            "status": "not_run_by_exporter",
            "required_major_version": 6,
            "main_formula_mode": "spectrum-inferred; do not supply correct formula",
            "primary_mass_deviation_ppm": 20,
            "required_downstream_tool": "CSI:FingerID fingerprints",
            "must_record_after_run": [
                "exact SIRIUS version",
                "GUI Show Command",
                "GUI Show JSON",
                "element settings",
                "fingerprint dimension",
                "imported/success/failed feature counts",
            ],
        },
        "software": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "qc_status": "PASS" if not qc["status"].eq("FAIL").any() else "FAIL",
    }
    atomic_json(RUN_MANIFEST, manifest)
    return manifest


def run_export() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index, upstream = load_and_validate_inputs()
    features, selected = build_feature_tables(index)
    selected, failures = export_ms_files(features, selected)
    atomic_csv(
        pd.DataFrame(failures, columns=["sirius_feature_id", "reason"]),
        FAILURES_PATH,
    )
    if failures:
        sample = pd.DataFrame(failures).head(10)
        raise RuntimeError(
            f"{len(failures)} SIRIUS features failed during export; "
            f"see {FAILURES_PATH}. First failures:\n{sample.to_string(index=False)}"
        )
    qc = make_qc(index, features, selected, failures)
    manifest = write_supporting_files(index, features, selected, qc, upstream)
    if qc["status"].eq("FAIL").any():
        failed = qc.loc[qc["status"] == "FAIL", ["check", "observed", "expected"]]
        raise AssertionError(f"SIRIUS input QC failed:\n{failed.to_string(index=False)}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    manifest = run_export()
    counts = manifest["counts"]
    print(
        "SIRIUS Development input export PASS: "
        f"{counts['frozen_development_compounds']} compounds, "
        f"{counts['sirius_features']} features, "
        f"{counts['selected_spectra']} selected spectra."
    )
    print(f"Positive input: {POSITIVE_DIR}")
    print(f"Negative input: {NEGATIVE_DIR}")
    print(f"QC: {QC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
