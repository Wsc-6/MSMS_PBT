from __future__ import annotations

import atexit
import csv
import gzip
import hashlib
import json
import logging
import math
import os
import pickle
import platform
import random
import re
import shutil
import sqlite3
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Descriptors
from tqdm.auto import tqdm


SCHEMA_VERSION = "ENVMOLBENCH_CLEANED_V2"
RANDOM_SEED = 42
MAX_PEAKS = 120
INPUT_DIM = 7
MAX_MZ = 2000.0
MIN_MZ = 15.0
MIN_RELATIVE_INTENSITY = 0.001
PRIMARY_PPM = 20.0
SENSITIVITY_PPM = (50.0, 100.0)
MASTER_ACCEPTANCE_PPM = 100.0
COMMIT_INTERVAL = 10_000
MIN_FREE_DISK_GB = 30
EXPECTED_RAW_RECORD_COUNTS = {
    "GNPS": 2_091_371,
    "MassBank": 134_440,
    "MoNA": 153_632,
}

ADDUCT_DICT_POS = {
    "[M+H]+": (1.007276, 1),
    "[M+Na]+": (22.989769, 1),
    "[M+NH4]+": (18.033823, 1),
    "[M+K]+": (38.963158, 1),
    "[M-H2O+H]+": (-17.003284, 1),
    "[M]+.": (-0.0005485, 1),
}
ADDUCT_DICT_NEG = {
    "[M-H]-": (-1.007276, 1),
    "[M+Cl]-": (34.968853, 1),
    "[M+FA]-": (44.99765, 1),
    "[M+Ac]-": (59.01330, 1),
    "[M-H2O-H]-": (-19.01784, 1),
    "[M]-.": (0.0005485, 1),
}

METADATA_COLUMNS = [
    "accepted_rank_global",
    "accepted_rank_in_library",
    "stream_offset",
    "source_library",
    "source_file",
    "source_record_ordinal",
    "original_spectrum_id",
    "original_spectrum_id_status",
    "instrument",
    "instrument_status",
    "inchikey_full",
    "inchikey_2d",
    "canonical_smiles",
    "precursor_mz",
    "ionmode",
    "raw_adduct",
    "adduct",
    "adduct_selection_method",
    "theoretical_precursor_mz",
    "identity_ppm_error",
    "best_physical_ppm_error",
    "ppm_tier",
    "ce_raw",
    "ce_real",
    "ce_bucket",
    "raw_peak_n",
    "cleaned_peak_n",
    "raw_peak_hash",
    "cleaned_peak_hash",
    "spectrum_hash",
    "schema_version",
]


class PeakList:
    def __init__(self, mz: list[float], intensities: list[float]):
        self.mz = np.asarray(mz, dtype=np.float64)
        self.intensities = np.asarray(intensities, dtype=np.float64)


class RawSpectrum:
    """Minimal spectrum object used by the streaming MGF/MSP readers."""

    def __init__(self, metadata: dict[str, str], mz: list[float], intensities: list[float]):
        self.metadata = metadata
        self.peaks = PeakList(mz, intensities)

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(_normalize_metadata_key(key), default)


def _normalize_metadata_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def _append_metadata(metadata: dict[str, str], key: str, value: str) -> None:
    normalized = _normalize_metadata_key(key)
    if not normalized:
        return
    value = str(value).strip()
    if normalized in metadata and metadata[normalized] != value:
        metadata[normalized] = metadata[normalized] + " | " + value
    else:
        metadata[normalized] = value


def _parse_peak_line(line: str) -> tuple[float, float] | None:
    match = re.match(
        r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        r"[\s,;]+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        line,
    )
    if not match:
        return None
    try:
        return float(match.group(1)), float(match.group(2))
    except ValueError:
        return None


def load_mgf_stream(path: Path) -> Iterable[RawSpectrum]:
    metadata: dict[str, str] | None = None
    mz: list[float] = []
    intensity: list[float] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            upper = stripped.upper()
            if upper == "BEGIN IONS":
                metadata, mz, intensity = {}, [], []
                continue
            if upper == "END IONS":
                if metadata is not None:
                    yield RawSpectrum(metadata, mz, intensity)
                metadata = None
                continue
            if metadata is None or not stripped:
                continue
            if "=" in stripped and _parse_peak_line(stripped) is None:
                key, value = stripped.split("=", 1)
                _append_metadata(metadata, key, value)
            else:
                peak = _parse_peak_line(stripped)
                if peak is not None:
                    mz.append(peak[0])
                    intensity.append(peak[1])


def load_msp_stream(path: Path) -> Iterable[RawSpectrum]:
    metadata: dict[str, str] = {}
    mz: list[float] = []
    intensity: list[float] = []

    def emit() -> RawSpectrum | None:
        if not metadata and not mz:
            return None
        return RawSpectrum(dict(metadata), list(mz), list(intensity))

    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                record = emit()
                if record is not None:
                    yield record
                metadata, mz, intensity = {}, [], []
                continue
            peak = _parse_peak_line(stripped)
            if peak is not None:
                mz.append(peak[0])
                intensity.append(peak[1])
            elif ":" in stripped:
                key, value = stripped.split(":", 1)
                _append_metadata(metadata, key, value)
        record = emit()
        if record is not None:
            yield record

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(map(str, values)):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _json_write(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def resolve_project_root(start: Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        spectrum_dir = candidate / "质谱数据"
        if (spectrum_dir / "GNPS").is_dir() and (candidate / "configs").is_dir():
            return candidate
    if current.name == "质谱数据" and (current / "GNPS").is_dir():
        return current.parent
    raise FileNotFoundError("无法从当前工作目录定位项目根目录和质谱数据目录。")


def default_raw_files(project_root: Path) -> dict[str, Path]:
    base = Path(project_root) / "质谱数据"
    return {
        "GNPS": base / "GNPS" / "ALL_GNPS.mgf",
        "MassBank": base / "MassBank Europe" / "MassBank_NISTformat.msp",
        "MoNA": base / "MoNA" / "MoNA-export-LC-MS-MS_Spectra.msp",
    }


def _metadata_value(spectrum: Any, names: Iterable[str]) -> tuple[str, str]:
    for name in names:
        try:
            value = spectrum.get(name, None)
        except Exception:
            value = None
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null", "n/a", "unknown"}:
            return text, name
    return "", ""


def _extract_precursor_mz(spectrum: Any) -> float | None:
    value, _ = _metadata_value(
        spectrum,
        ["precursor_mz", "precursormz", "pepmass", "parent_mass", "parentmass"],
    )
    if not value:
        return None
    match = re.search(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _extract_smiles(spectrum: Any) -> str:
    direct, _ = _metadata_value(
        spectrum,
        ["smiles", "canonical_smiles", "canonicalsmiles", "isomeric_smiles"],
    )
    if direct:
        return direct.split(" | ", 1)[0].strip()
    comments, _ = _metadata_value(spectrum, ["comments", "comment"])
    if not comments:
        return ""
    quoted = re.findall(r'"([^"=]*smiles)\s*=\s*([^"]+)"', comments, flags=re.I)
    for label, value in quoted:
        if label.strip().lower() == "smiles":
            return value.strip()
    if quoted:
        return quoted[0][1].strip()
    plain = re.search(r"(?:^|\s)SMILES\s*=\s*([^\s\"]+)", comments, flags=re.I)
    return plain.group(1).strip() if plain else ""


def _peak_hash(mz: np.ndarray, intensity: np.ndarray) -> str:
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    valid = np.isfinite(mz) & np.isfinite(intensity)
    mz, intensity = mz[valid], intensity[valid]
    if len(mz):
        order = np.argsort(mz, kind="stable")
        mz, intensity = mz[order], intensity[order]
        maximum = float(np.max(np.clip(intensity, 0.0, None)))
        if maximum > 0:
            intensity = intensity / maximum
    packed = np.column_stack([np.round(mz, 5), np.round(intensity, 6)]).astype("<f8")
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _spectrum_hash(
    precursor_mz: float,
    ionmode: str,
    mz: np.ndarray,
    intensity: np.ndarray,
) -> str:
    peak_digest = _peak_hash(mz, intensity)
    payload = f"{precursor_mz:.5f}|{ionmode}|{peak_digest}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def extract_standard_ce(spectrum: Any) -> dict[str, Any]:
    ce_text, _ = _metadata_value(
        spectrum,
        ["collision_energy", "collisionenergy", "collision energy", "ce", "nce"],
    )
    lowered = ce_text.lower()
    if not lowered:
        return {"ce_raw": "", "ce_real": -1.0, "ce_bucket": -1.0}
    numbers = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", lowered)]
    if not numbers:
        return {"ce_raw": ce_text, "ce_real": -1.0, "ce_bucket": -1.0}
    ramped = len(numbers) >= 2 and (
        any(marker in lowered for marker in ["ramp", "step", "range", " to ", ",", "/"])
        or bool(re.search(r"\d\s*[-–—]\s*\d", lowered))
    )
    value = float(np.mean(numbers)) if ramped else float(numbers[0])
    if ramped:
        bucket = -2.0
    elif 10 <= value <= 25:
        bucket = 20.0
    elif 25 < value <= 45:
        bucket = 40.0
    elif 45 < value <= 65:
        bucket = 60.0
    else:
        bucket = -1.0
    return {"ce_raw": ce_text, "ce_real": value, "ce_bucket": bucket}


def _text_adduct_hint(raw_adduct: str, ionmode: str) -> str | None:
    text = raw_adduct.replace(" ", "").upper()
    if ionmode == "Positive":
        if "M+H" in text and "H2O" not in text:
            return "[M+H]+"
        if "M+NA" in text:
            return "[M+Na]+"
        if "M+NH4" in text:
            return "[M+NH4]+"
        if "M+K" in text:
            return "[M+K]+"
        if "M-H2O" in text or "M+H-H2O" in text:
            return "[M-H2O+H]+"
        if text in {"M+.", "[M]+."}:
            return "[M]+."
    else:
        if "M-H" in text and "H2O" not in text:
            return "[M-H]-"
        if "M+CL" in text:
            return "[M+Cl]-"
        if "M+FA" in text or "HCOO" in text:
            return "[M+FA]-"
        if "M+AC" in text or "CH3COO" in text:
            return "[M+Ac]-"
        if "M-H2O" in text:
            return "[M-H2O-H]-"
        if text in {"M-.", "[M]-."}:
            return "[M]-."
    return None


def _ppm_tier(ppm_error: float) -> str:
    if ppm_error <= PRIMARY_PPM:
        return "20ppm"
    if ppm_error <= SENSITIVITY_PPM[0]:
        return "50ppm"
    if ppm_error <= SENSITIVITY_PPM[1]:
        return "100ppm"
    return "rejected"


def preprocess_to_tensor(
    raw_spectrum: Any,
    *,
    source_library: str,
    source_file: Path,
    source_record_ordinal: int,
    error_logger: logging.Logger,
    max_peaks: int = MAX_PEAKS,
) -> tuple[str, dict[str, Any] | None]:
    if raw_spectrum is None:
        return "INVALID_METADATA", None
    original_id, original_id_field = _metadata_value(
        raw_spectrum,
        [
            "spectrum_id",
            "spectrumid",
            "accession",
            "db",
            "scan_number",
            "scans",
            "feature_id",
            "title",
            "name",
        ],
    )
    if original_id:
        original_id_status = f"source_field:{original_id_field}"
    else:
        original_id = f"{source_library}:{source_record_ordinal:09d}"
        original_id_status = "generated_from_source_ordinal"
    instrument, instrument_field = _metadata_value(
        raw_spectrum,
        [
            "instrument",
            "instrument_type",
            "instrumenttype",
            "instrument_name",
            "ms_instrument",
            "source_instrument",
        ],
    )
    instrument_status = f"source_field:{instrument_field}" if instrument else "unavailable"
    raw_adduct, _ = _metadata_value(raw_spectrum, ["adduct", "precursor_type", "ion_type"])
    try:
        raw_mz = np.asarray(raw_spectrum.peaks.mz, dtype=np.float64)
        raw_intensity = np.asarray(raw_spectrum.peaks.intensities, dtype=np.float64)
    except Exception:
        raw_mz = np.asarray([], dtype=np.float64)
        raw_intensity = np.asarray([], dtype=np.float64)
    raw_peak_n = int(len(raw_mz))
    raw_peak_hash = _peak_hash(raw_mz, raw_intensity)

    try:
        spectrum = raw_spectrum

        ion_text, _ = _metadata_value(
            spectrum,
            ["ionmode", "ion_mode", "polarity", "ionization_mode", "ionizationmode"],
        )
        lowered_ion = ion_text.lower().strip()
        if lowered_ion in {"p", "+", "positive", "pos"} or any(
            value in lowered_ion for value in ["positive", "pos", "+"]
        ):
            ionmode = "Positive"
        elif lowered_ion in {"n", "-", "negative", "neg"} or any(
            value in lowered_ion for value in ["negative", "neg", "-"]
        ):
            ionmode = "Negative"
        else:
            return "INVALID_METADATA", None

        precursor_mz = _extract_precursor_mz(spectrum)
        if precursor_mz is None:
            return "INVALID_METADATA", None
        if not np.isfinite(precursor_mz) or precursor_mz <= 0 or precursor_mz > MAX_MZ:
            return "INVALID_METADATA", None

        raw_smiles = _extract_smiles(spectrum)
        mol = Chem.MolFromSmiles(raw_smiles) if raw_smiles else None
        if mol is None:
            return "INVALID_SMILES", None
        fragments = Chem.GetMolFrags(mol, asMols=True)
        if not fragments:
            return "INVALID_SMILES", None
        mol = max(fragments, key=lambda item: item.GetNumHeavyAtoms())
        Chem.RemoveStereochemistry(mol)
        # Do not enumerate all tautomers here: pathological structures can cause
        # combinatorial runtimes. Standard InChI normalization supplies the
        # structure identity, while tautomers retain the same exact formula/mass.
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        inchikey_full = Chem.MolToInchiKey(mol)
        if not inchikey_full:
            return "INVALID_SMILES", None
        inchikey_2d = inchikey_full.split("-")[0]

        exact_mass = float(Descriptors.ExactMolWt(mol))
        adduct_dictionary = ADDUCT_DICT_POS if ionmode == "Positive" else ADDUCT_DICT_NEG
        candidates: dict[str, dict[str, float]] = {}
        for adduct_name, (mass_delta, charge) in adduct_dictionary.items():
            theoretical_mz = (exact_mass + mass_delta) / charge
            ppm_error = abs(precursor_mz - theoretical_mz) / theoretical_mz * 1e6
            candidates[adduct_name] = {
                "theoretical_mz": theoretical_mz,
                "ppm_error": ppm_error,
            }
        best_adduct = min(candidates, key=lambda key: candidates[key]["ppm_error"])
        best_error = float(candidates[best_adduct]["ppm_error"])
        hint = _text_adduct_hint(raw_adduct, ionmode)
        if hint in candidates and candidates[hint]["ppm_error"] <= PRIMARY_PPM:
            clean_adduct = hint
            selection_method = "source_hint_within_20ppm"
        elif best_error <= SENSITIVITY_PPM[0]:
            clean_adduct = best_adduct
            selection_method = "best_physical_match_within_50ppm"
        elif best_error <= MASTER_ACCEPTANCE_PPM:
            clean_adduct = best_adduct
            selection_method = "best_physical_match_within_100ppm"
        else:
            return "PPM_FAIL", None
        chosen = candidates[clean_adduct]
        identity_ppm_error = float(chosen["ppm_error"])

        mz_array = np.asarray(spectrum.peaks.mz, dtype=np.float64)
        intensity_array = np.asarray(spectrum.peaks.intensities, dtype=np.float64)
        finite = (
            np.isfinite(mz_array)
            & np.isfinite(intensity_array)
            & (intensity_array > 0)
            & (mz_array >= MIN_MZ)
            & (mz_array <= MAX_MZ)
        )
        mz_array, intensity_array = mz_array[finite], intensity_array[finite]
        if not len(intensity_array) or float(intensity_array.max()) <= 0:
            return "LOW_PEAKS", None
        intensity_array = intensity_array / float(intensity_array.max())
        abundance_keep = intensity_array >= MIN_RELATIVE_INTENSITY
        mz_array, intensity_array = mz_array[abundance_keep], intensity_array[abundance_keep]
        mz_order = np.argsort(mz_array, kind="stable")
        mz_array, intensity_array = mz_array[mz_order], intensity_array[mz_order]
        keep = np.ones(len(mz_array), dtype=bool)
        for first in range(len(mz_array)):
            if not keep[first]:
                continue
            for second in range(first + 1, len(mz_array)):
                difference = mz_array[second] - mz_array[first]
                if difference > 1.01:
                    break
                if 1.00085 < difference < 1.00585 and intensity_array[second] < intensity_array[first] * 0.8:
                    keep[second] = False
        filtered_mz = mz_array[keep]
        filtered_intensity = intensity_array[keep]
        if len(filtered_mz) < 3:
            return "LOW_PEAKS", None

        abundance_order = np.argsort(filtered_intensity, kind="stable")[::-1]
        selected = abundance_order[: max_peaks - 1]
        selected_mz = filtered_mz[selected]
        selected_intensity = filtered_intensity[selected]
        selected_ranks = np.arange(1, len(selected) + 1)
        mass_order = np.argsort(selected_mz, kind="stable")
        mz_sorted = selected_mz[mass_order]
        intensity_sorted = selected_intensity[mass_order]
        ranks_sorted = selected_ranks[mass_order]
        n_fragments = len(mz_sorted)

        tensor = np.zeros((max_peaks, INPUT_DIM), dtype=np.float32)
        padding_mask = np.ones(max_peaks, dtype=bool)
        tensor[0] = [precursor_mz / MAX_MZ, 1.0, 0.0, 1.0, 1.0, np.log1p(1.0), 0.0]
        padding_mask[0] = False
        tensor[1 : n_fragments + 1, 0] = mz_sorted / MAX_MZ
        tensor[1 : n_fragments + 1, 1] = mz_sorted / precursor_mz
        tensor[1 : n_fragments + 1, 2] = (precursor_mz - mz_sorted) / MAX_MZ
        tensor[1 : n_fragments + 1, 3] = intensity_sorted
        tensor[1 : n_fragments + 1, 4] = np.sqrt(intensity_sorted)
        tensor[1 : n_fragments + 1, 5] = np.log1p(intensity_sorted)
        tensor[1 : n_fragments + 1, 6] = ranks_sorted / n_fragments
        padding_mask[1 : n_fragments + 1] = False

        ce = extract_standard_ce(spectrum)
        cleaned_peak_hash = _peak_hash(mz_sorted, intensity_sorted)
        spectrum_hash = _spectrum_hash(precursor_mz, ionmode, mz_sorted, intensity_sorted)
        return "SUCCESS", {
            "schema_version": SCHEMA_VERSION,
            "source_library": source_library,
            "source_file": str(source_file.name),
            "source_record_ordinal": int(source_record_ordinal),
            "original_spectrum_id": original_id,
            "original_spectrum_id_status": original_id_status,
            "instrument": instrument,
            "instrument_status": instrument_status,
            "inchikey_full": inchikey_full,
            "inchikey_2d": inchikey_2d,
            "smiles": canonical_smiles,
            "precursor_mz": precursor_mz,
            "tensor": tensor,
            "padding_mask": padding_mask,
            "ionmode": ionmode,
            "raw_adduct": raw_adduct,
            "adduct": clean_adduct,
            "adduct_selection_method": selection_method,
            "theoretical_precursor_mz": float(chosen["theoretical_mz"]),
            "identity_ppm_error": identity_ppm_error,
            "best_physical_ppm_error": best_error,
            "ppm_tier": _ppm_tier(identity_ppm_error),
            **ce,
            "raw_peak_n": raw_peak_n,
            "cleaned_peak_n": n_fragments,
            "raw_peak_hash": raw_peak_hash,
            "cleaned_peak_hash": cleaned_peak_hash,
            "spectrum_hash": spectrum_hash,
        }
    except Exception:
        error_logger.error(
            "[source=%s ordinal=%s id=%s]\n%s",
            source_library,
            source_record_ordinal,
            original_id,
            traceback.format_exc(),
        )
        return "ERROR", None


class SqliteCacheEngine:
    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
        self.connection = sqlite3.connect(self.database_path)
        self.cursor = self.connection.cursor()
        self.cursor.execute("PRAGMA journal_mode=WAL")
        self.cursor.execute("PRAGMA synchronous=NORMAL")
        self.cursor.execute(
            """
            CREATE TABLE ms_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inchikey_2d TEXT NOT NULL,
                source_library TEXT NOT NULL,
                tensor_dict BLOB NOT NULL
            )
            """
        )
        self.cursor.execute("CREATE INDEX idx_key ON ms_data(inchikey_2d)")
        self.connection.commit()
        self.uncommitted = 0
        self.closed = False
        atexit.register(self.close)

    def add(self, record: dict[str, Any]) -> None:
        self.cursor.execute(
            "INSERT INTO ms_data (inchikey_2d, source_library, tensor_dict) VALUES (?, ?, ?)",
            (
                record["inchikey_2d"],
                record["source_library"],
                pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL),
            ),
        )
        self.uncommitted += 1
        if self.uncommitted >= COMMIT_INTERVAL:
            self.connection.commit()
            self.uncommitted = 0

    def unique_compound_count(self) -> int:
        self.connection.commit()
        return int(
            self.connection.execute("SELECT COUNT(DISTINCT inchikey_2d) FROM ms_data").fetchone()[0]
        )

    def export(
        self,
        stream_path: Path,
        index_path: Path,
        metadata_path: Path,
        header: dict[str, Any],
    ) -> dict[str, Any]:
        self.connection.commit()
        index: dict[str, list[int]] = defaultdict(list)
        library_rank: Counter[str] = Counter()
        record_count = 0
        with stream_path.open("wb") as stream, gzip.open(
            metadata_path, "wt", encoding="utf-8", newline=""
        ) as metadata_handle:
            pickle.dump(("CONFIG", header), stream, protocol=pickle.HIGHEST_PROTOCOL)
            writer = csv.DictWriter(metadata_handle, fieldnames=METADATA_COLUMNS)
            writer.writeheader()
            cursor = self.connection.execute(
                "SELECT inchikey_2d, source_library, tensor_dict FROM ms_data ORDER BY id ASC"
            )
            for inchikey_2d, source_library, blob in tqdm(
                cursor, desc="流式导出V2与可追溯元数据", mininterval=2.0
            ):
                record = pickle.loads(blob)
                offset = int(stream.tell())
                index[inchikey_2d].append(offset)
                pickle.dump((inchikey_2d, record), stream, protocol=pickle.HIGHEST_PROTOCOL)
                record_count += 1
                library_rank[source_library] += 1
                metadata_row = {
                    "accepted_rank_global": record_count,
                    "accepted_rank_in_library": library_rank[source_library],
                    "stream_offset": offset,
                    "source_library": record["source_library"],
                    "source_file": record["source_file"],
                    "source_record_ordinal": record["source_record_ordinal"],
                    "original_spectrum_id": record["original_spectrum_id"],
                    "original_spectrum_id_status": record["original_spectrum_id_status"],
                    "instrument": record["instrument"],
                    "instrument_status": record["instrument_status"],
                    "inchikey_full": record["inchikey_full"],
                    "inchikey_2d": record["inchikey_2d"],
                    "canonical_smiles": record["smiles"],
                    "precursor_mz": record["precursor_mz"],
                    "ionmode": record["ionmode"],
                    "raw_adduct": record["raw_adduct"],
                    "adduct": record["adduct"],
                    "adduct_selection_method": record["adduct_selection_method"],
                    "theoretical_precursor_mz": record["theoretical_precursor_mz"],
                    "identity_ppm_error": record["identity_ppm_error"],
                    "best_physical_ppm_error": record["best_physical_ppm_error"],
                    "ppm_tier": record["ppm_tier"],
                    "ce_raw": record["ce_raw"],
                    "ce_real": record["ce_real"],
                    "ce_bucket": record["ce_bucket"],
                    "raw_peak_n": record["raw_peak_n"],
                    "cleaned_peak_n": record["cleaned_peak_n"],
                    "raw_peak_hash": record["raw_peak_hash"],
                    "cleaned_peak_hash": record["cleaned_peak_hash"],
                    "spectrum_hash": record["spectrum_hash"],
                    "schema_version": record["schema_version"],
                }
                writer.writerow(metadata_row)
        with index_path.open("wb") as handle:
            pickle.dump(dict(index), handle, protocol=pickle.HIGHEST_PROTOCOL)
        return {
            "record_count": record_count,
            "unique_compound_count": len(index),
            "library_counts": dict(library_rank),
        }

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.connection.commit()
            self.connection.close()
        except sqlite3.ProgrammingError:
            pass
        self.closed = True


def _configure_logging(log_path: Path) -> logging.Logger:
    RDLogger.DisableLog("rdApp.*")
    rdBase.DisableLog("rdApp.error")
    rdBase.DisableLog("rdApp.warning")
    rdBase.DisableLog("rdApp.info")
    logger = logging.getLogger(f"EnvMolBenchCleaningV2:{log_path}")
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _validate_build(
    stream_path: Path,
    index_path: Path,
    metadata_path: Path,
    expected_records: int,
    seed: int,
) -> dict[str, Any]:
    with index_path.open("rb") as handle:
        index = pickle.load(handle)
    if not isinstance(index, dict):
        raise TypeError("V2索引不是字典。")
    indexed_records = sum(len(offsets) for offsets in index.values())
    if indexed_records != expected_records:
        raise ValueError(f"索引记录数不守恒：{indexed_records}/{expected_records}")
    with stream_path.open("rb") as handle:
        marker, header = pickle.load(handle)
        if marker != "CONFIG" or header.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("V2主流文件头不正确。")
        all_pairs = [(key, offset) for key, offsets in index.items() for offset in offsets]
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(all_pairs), size=min(1000, len(all_pairs)), replace=False)
        for position in chosen:
            expected_key, offset = all_pairs[int(position)]
            handle.seek(int(offset))
            actual_key, record = pickle.load(handle)
            if actual_key != expected_key or record.get("inchikey_2d") != expected_key:
                raise ValueError(f"随机访问错位：{expected_key}@{offset}")
            if record.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("记录schema_version不正确。")
            tensor = np.asarray(record.get("tensor"))
            mask = np.asarray(record.get("padding_mask"))
            if tensor.shape != (MAX_PEAKS, INPUT_DIM) or tensor.dtype != np.float32:
                raise ValueError("记录tensor格式不正确。")
            if mask.shape != (MAX_PEAKS,) or mask.dtype != np.bool_:
                raise ValueError("记录padding_mask格式不正确。")
            if not np.isfinite(tensor).all():
                raise ValueError("记录tensor包含非有限值。")
            if float(record.get("identity_ppm_error", np.inf)) > MASTER_ACCEPTANCE_PPM:
                raise ValueError("记录超过master acceptance ppm。")

    metadata_count = 0
    metadata_library_counts: Counter[str] = Counter()
    required_columns = set(METADATA_COLUMNS)
    with gzip.open(metadata_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = sorted(required_columns - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(f"V2追溯元数据缺列：{missing_columns}")
        for row in reader:
            metadata_count += 1
            metadata_library_counts[row["source_library"]] += 1
            if row["schema_version"] != SCHEMA_VERSION:
                raise ValueError("V2追溯元数据schema_version不正确。")
            if not row["original_spectrum_id"].strip():
                raise ValueError("V2追溯元数据存在空original_spectrum_id。")
            ppm_error = float(row["identity_ppm_error"])
            if not np.isfinite(ppm_error) or ppm_error > MASTER_ACCEPTANCE_PPM:
                raise ValueError("V2追溯元数据存在非法identity ppm。")
    if metadata_count != expected_records:
        raise ValueError(f"V2追溯元数据记录数不守恒：{metadata_count}/{expected_records}")
    return {
        "indexed_record_count": indexed_records,
        "indexed_compound_count": len(index),
        "random_access_records_checked": min(1000, len(all_pairs)),
        "metadata_record_count": metadata_count,
        "metadata_library_counts": dict(metadata_library_counts),
        "status": "PASS",
    }


def _atomic_promote(artifacts: list[tuple[Path, Path]]) -> None:
    backups: dict[Path, Path] = {}
    promoted: list[Path] = []
    try:
        for _, final in artifacts:
            backup = final.with_name(final.name + ".previous")
            if backup.exists():
                if final.exists():
                    backup.unlink()
                else:
                    os.replace(backup, final)
            if final.exists():
                os.replace(final, backup)
                backups[final] = backup
        for temporary, final in artifacts:
            os.replace(temporary, final)
            promoted.append(final)
    except Exception:
        for final in promoted:
            if final.exists():
                final.unlink()
        for final, backup in backups.items():
            if backup.exists():
                os.replace(backup, final)
        raise
    # Promotion has fully succeeded. Backup cleanup is best-effort and must never
    # trigger rollback after a new validated artifact is already in place.
    for backup in backups.values():
        if backup.exists():
            try:
                backup.unlink()
            except OSError:
                pass


def run_public_spectrum_cleaning(
    project_root: Path | None = None,
    *,
    overwrite: bool = False,
    output_prefix: str = "EnvMolBench_Cleaned",
    max_records_per_library: int | None = None,
) -> dict[str, Any]:
    """Build the public spectrum stream; existing outputs survive until validation passes."""
    root = resolve_project_root(project_root)
    base = root / "质谱数据"
    raw_files = default_raw_files(root)
    missing = [str(path) for path in raw_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少原始谱库：\n" + "\n".join(missing))

    final_paths = {
        "stream": base / f"{output_prefix}.pkl",
        "index": base / f"{output_prefix}_Index.pkl",
        "metadata": base / f"{output_prefix}_Metadata.csv.gz",
        "summary": base / f"{output_prefix}_Summary.json",
        "error_log": base / f"{output_prefix}_Errors.log",
    }
    existing = [path for path in final_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "输出已经存在；只有明确传入overwrite=True才允许在完整验证后替换。\n"
            + "\n".join(map(str, existing))
        )
    disk_free = shutil.disk_usage(base).free
    if disk_free < MIN_FREE_DISK_GB * 1024**3:
        raise OSError(
            f"目标磁盘可用空间不足{MIN_FREE_DISK_GB} GB：{disk_free / 1024**3:.2f} GB"
        )

    temporary_paths = {
        name: path.with_name(path.name + ".building") for name, path in final_paths.items()
    }
    database_path = base / f"{output_prefix}_Temp.sqlite"
    for path in [*temporary_paths.values(), database_path, database_path.with_name(database_path.name + "-wal"), database_path.with_name(database_path.name + "-shm")]:
        if path.exists():
            path.unlink()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    error_logger = _configure_logging(temporary_paths["error_log"])
    stats: Counter[str] = Counter()
    stats_by_library: dict[str, Counter[str]] = defaultdict(Counter)
    cache = SqliteCacheEngine(database_path)
    started = utc_now()
    try:
        for library, source_file in raw_files.items():
            loader = load_mgf_stream if source_file.suffix.lower() == ".mgf" else load_msp_stream
            progress = tqdm(
                loader(source_file),
                desc=f"清洗{library}",
                mininterval=2.0,
            )
            accepted_in_library = 0
            for ordinal, spectrum in enumerate(progress, start=1):
                if max_records_per_library is not None and ordinal > max_records_per_library:
                    break
                stats["TOTAL_RAW"] += 1
                stats_by_library[library]["TOTAL_RAW"] += 1
                status, record = preprocess_to_tensor(
                    spectrum,
                    source_library=library,
                    source_file=source_file,
                    source_record_ordinal=ordinal,
                    error_logger=error_logger,
                )
                stats[status] += 1
                stats_by_library[library][status] += 1
                if status == "SUCCESS" and record is not None:
                    cache.add(record)
                    accepted_in_library += 1
                if ordinal % 1000 == 0:
                    progress.set_postfix(accepted=accepted_in_library, refresh=False)

            progress.set_postfix(accepted=accepted_in_library, refresh=True)
            if max_records_per_library is None:
                parsed_count = int(stats_by_library[library]["TOTAL_RAW"])
                expected_raw_count = EXPECTED_RAW_RECORD_COUNTS[library]
                if parsed_count != expected_raw_count:
                    raise ValueError(
                        f"{library}原始记录数不符合冻结审计值：{parsed_count}/{expected_raw_count}"
                    )

        unique_compounds = cache.unique_compound_count()
        header = {
            "project": "EnvMolBench",
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": utc_now(),
            "max_peaks": MAX_PEAKS,
            "input_dim": INPUT_DIM,
            "max_mz": MAX_MZ,
            "min_mz": MIN_MZ,
            "min_relative_intensity": MIN_RELATIVE_INTENSITY,
            "neutral_loss_included": True,
            "master_acceptance_ppm": MASTER_ACCEPTANCE_PPM,
            "primary_analysis_ppm": PRIMARY_PPM,
            "sensitivity_analysis_ppm": list(SENSITIVITY_PPM),
            "identity_ppm_stored_per_record": True,
            "source_metadata_stored_per_record": True,
            "random_seed": RANDOM_SEED,
            "source_library_order": list(raw_files),
            "expected_raw_record_counts": EXPECTED_RAW_RECORD_COUNTS,
            "tautomer_handling": "standard InChI normalization; no exhaustive tautomer enumeration",
        }
        export = cache.export(
            temporary_paths["stream"],
            temporary_paths["index"],
            temporary_paths["metadata"],
            header,
        )
        if export["record_count"] != int(stats["SUCCESS"]):
            raise ValueError(
                f"SQLite导出记录不守恒：{export['record_count']}/{stats['SUCCESS']}"
            )
        validation = _validate_build(
            temporary_paths["stream"],
            temporary_paths["index"],
            temporary_paths["metadata"],
            int(stats["SUCCESS"]),
            RANDOM_SEED,
        )
        if validation["metadata_library_counts"] != export["library_counts"]:
            raise ValueError(
                "V2追溯元数据分库记录数与SQLite导出不一致："
                f"{validation['metadata_library_counts']} / {export['library_counts']}"
            )
        for handler in error_logger.handlers:
            handler.flush()
        output_hashes = {
            name: {
                "sha256": sha256_file(temporary_paths[name]),
                "size_bytes": temporary_paths[name].stat().st_size,
            }
            for name in ["stream", "index", "metadata", "error_log"]
        }
        summary = {
            "schema_version": SCHEMA_VERSION,
            "started_at_utc": started,
            "completed_at_utc": utc_now(),
            "project_root": str(root),
            "output_prefix": output_prefix,
            "overwrite_requested": overwrite,
            "max_records_per_library": max_records_per_library,
            "raw_inputs": {
                library: {
                    "path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for library, path in raw_files.items()
            },
            "configuration": header,
            "counts": dict(stats),
            "counts_by_library": {
                library: dict(counts) for library, counts in stats_by_library.items()
            },
            "unique_compound_count": unique_compounds,
            "export": export,
            "validation": validation,
            "outputs": output_hashes,
            "software": {
                "python": sys.version,
                "executable": sys.executable,
                "numpy": np.__version__,
                "rdkit": rdBase.rdkitVersion,
                "spectrum_parser": "built-in streaming MGF/MSP V1",
                "platform": platform.platform(),
            },
            "status": "PASS",
        }
        _json_write(temporary_paths["summary"], summary)
        _close_logger(error_logger)
        artifacts = [
            (temporary_paths[name], final_paths[name])
            for name in ["stream", "index", "metadata", "summary", "error_log"]
        ]
        _atomic_promote(artifacts)
        return summary
    finally:
        cache.close()
        _close_logger(error_logger)
        for suffix in ["", "-wal", "-shm"]:
            path = Path(str(database_path) + suffix)
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
