"""Run a five-spectrum DreaMS embedding smoke test without touching ChemVista.

This script is intentionally standalone so it can be executed with the isolated
``env_dreams`` interpreter while the main PBT pipeline remains in ``env_ms``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import locale
import os
import pickle
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pbt_msms_v1" / "dreams_v1" / "mvp_5_spectra"
RUNTIME_CACHE = OUTPUT_DIR.parent / "runtime_cache"
PUBLIC_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "pbt_msms_v1"
    / "spectrum_audit_v1"
    / "public_relevant_spectrum_manifest.csv.gz"
)
PUBLIC_STREAM = PROJECT_ROOT / "质谱数据" / "EnvMolBench_Cleaned.pkl"
DREAMS_SOURCE = PROJECT_ROOT / "third_party" / "DreaMS"
DREAMS_ARCHIVE = PROJECT_ROOT / "third_party" / "DreaMS-main.zip"
DREAMS_COMMIT = "dbec3a0b514a99e5056cfccde4559fda8cfe8129"
MAX_MZ = 2000.0
N_SPECTRA = 5
SEED = 42


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def configure_writable_runtime_cache() -> None:
    cache_paths = {
        "NUMBA_CACHE_DIR": RUNTIME_CACHE / "numba",
        "MPLCONFIGDIR": RUNTIME_CACHE / "matplotlib",
        "HF_HOME": RUNTIME_CACHE / "huggingface",
    }
    for variable, path in cache_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["WANDB_MODE"] = "disabled"


def select_five_development_spectra() -> pd.DataFrame:
    manifest = pd.read_csv(PUBLIC_MANIFEST)
    required = {
        "spectrum_id",
        "inchikey_2d",
        "stream_offset",
        "precursor_mz",
        "ion_mode",
        "adduct",
        "collision_energy_value",
        "cleaned_peak_n",
        "identity_ppm_error",
        "cv_fold",
        "primary_train_eligible_20ppm",
        "supervised_compound_role",
        "chemvista_evaluation_role",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Missing spectrum manifest columns: {missing}")

    eligible = manifest.loc[
        manifest["primary_train_eligible_20ppm"].fillna(False).astype(bool)
        & manifest["supervised_compound_role"].astype(str).eq("development")
        & manifest["chemvista_evaluation_role"].astype(str).eq("none")
        & pd.to_numeric(manifest["cleaned_peak_n"], errors="coerce").ge(10)
        & pd.to_numeric(manifest["precursor_mz"], errors="coerce").notna()
    ].copy()
    eligible["cv_fold"] = pd.to_numeric(eligible["cv_fold"], errors="raise").astype(int)

    # One compound from every frozen outer fold and alternating ion polarity.
    desired_polarity = {0: "Positive", 1: "Negative", 2: "Positive", 3: "Negative", 4: "Positive"}
    selected_rows: list[pd.Series] = []
    used_keys: set[str] = set()
    with PUBLIC_STREAM.open("rb") as stream:
        for fold in range(N_SPECTRA):
            candidates = eligible.loc[
                eligible["cv_fold"].eq(fold)
                & eligible["ion_mode"].astype(str).eq(desired_polarity[fold])
                & ~eligible["inchikey_2d"].astype(str).isin(used_keys)
            ].copy()
            candidates = candidates.sort_values(
                ["cleaned_peak_n", "identity_ppm_error", "spectrum_id"],
                ascending=[False, True, True],
            ).drop_duplicates("inchikey_2d")
            selected_row: pd.Series | None = None
            for _, candidate in candidates.iterrows():
                stream.seek(int(candidate["stream_offset"]))
                actual_key, record = pickle.load(stream)
                record_peak_n = int(record.get("cleaned_peak_n", -1))
                if (
                    str(actual_key) == str(candidate["inchikey_2d"])
                    and str(record.get("source_library")) != "ChemVista"
                    and record_peak_n == int(candidate["cleaned_peak_n"])
                ):
                    selected_row = candidate
                    break
            if selected_row is None:
                raise ValueError(f"No internally consistent DreaMS MVP spectrum for frozen fold {fold}")
            selected_rows.append(selected_row)
            used_keys.add(str(selected_row["inchikey_2d"]))

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected.insert(0, "mvp_row", np.arange(N_SPECTRA, dtype=int))
    if selected["inchikey_2d"].nunique() != N_SPECTRA:
        raise AssertionError("MVP spectra are not five distinct compounds")
    if set(selected["cv_fold"]) != set(range(N_SPECTRA)):
        raise AssertionError("MVP spectra do not cover all five frozen folds")
    if not selected["chemvista_evaluation_role"].astype(str).eq("none").all():
        raise AssertionError("ChemVista evaluation identity entered the MVP input")
    return selected


def charge_text(ion_mode: str) -> str:
    if str(ion_mode).lower().startswith("pos"):
        return "1+"
    if str(ion_mode).lower().startswith("neg"):
        return "1-"
    raise ValueError(f"Unsupported ion mode for MVP: {ion_mode}")


def extract_peaks(record: dict[str, Any]) -> np.ndarray:
    tensor = np.asarray(record["tensor"], dtype=np.float64)
    padding = np.asarray(record["padding_mask"], dtype=bool)
    if tensor.ndim != 2 or tensor.shape[1] < 4 or padding.shape != (len(tensor),):
        raise ValueError("Unexpected trusted public spectrum tensor schema")
    valid = ~padding
    valid[0] = False  # Row zero is the precursor token, not a fragment.
    mz = tensor[valid, 0] * MAX_MZ
    intensity = tensor[valid, 3] * 100.0
    keep = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    peaks = np.column_stack([mz[keep], intensity[keep]])
    if len(peaks) < 3:
        raise ValueError("DreaMS MVP spectrum has fewer than three valid fragment peaks")
    return peaks


def write_mgf(selected: pd.DataFrame, mgf_path: Path) -> pd.DataFrame:
    audit_rows: list[dict[str, Any]] = []
    temporary = mgf_path.with_suffix(mgf_path.suffix + ".tmp")
    with PUBLIC_STREAM.open("rb") as stream, temporary.open("w", encoding="utf-8", newline="\n") as out:
        for row in selected.itertuples(index=False):
            stream.seek(int(row.stream_offset))
            actual_key, record = pickle.load(stream)
            if str(actual_key) != str(row.inchikey_2d):
                raise ValueError(f"Stream identity mismatch for {row.spectrum_id}")
            if str(record.get("inchikey_2d")) != str(row.inchikey_2d):
                raise ValueError(f"Record identity mismatch for {row.spectrum_id}")
            if str(record.get("source_library")) == "ChemVista":
                raise AssertionError("ChemVista spectrum entered the MVP input")
            peaks = extract_peaks(record)
            record_peak_n = int(record.get("cleaned_peak_n", -1))
            if record_peak_n != int(row.cleaned_peak_n) or len(peaks) != record_peak_n:
                raise AssertionError(
                    f"Peak-count mismatch for {row.spectrum_id}: "
                    f"manifest={row.cleaned_peak_n}, record={record_peak_n}, exported={len(peaks)}"
                )

            out.write("BEGIN IONS\n")
            out.write(f"TITLE={row.spectrum_id}\n")
            out.write(f"FEATURE_ID={int(row.mvp_row) + 1}\n")
            out.write("MSLEVEL=2\n")
            out.write(f"PEPMASS={float(row.precursor_mz):.8f}\n")
            out.write(f"CHARGE={charge_text(row.ion_mode)}\n")
            out.write(f"IONMODE={row.ion_mode}\n")
            if pd.notna(row.adduct):
                out.write(f"ADDUCT={row.adduct}\n")
            if np.isfinite(float(row.collision_energy_value)) and float(row.collision_energy_value) >= 0:
                out.write(f"COLLISION_ENERGY={float(row.collision_energy_value):.8g}\n")
            out.write(f"SOURCE_LIBRARY={record.get('source_library', 'unknown')}\n")
            for mz, intensity in peaks:
                out.write(f"{mz:.8f} {intensity:.8f}\n")
            out.write("END IONS\n\n")

            audit_rows.append(
                {
                    "mvp_row": int(row.mvp_row),
                    "spectrum_id": str(row.spectrum_id),
                    "inchikey_2d": str(row.inchikey_2d),
                    "cv_fold": int(row.cv_fold),
                    "ion_mode": str(row.ion_mode),
                    "adduct": str(row.adduct),
                    "precursor_mz": float(row.precursor_mz),
                    "collision_energy_value": float(row.collision_energy_value),
                    "manifest_cleaned_peak_n": int(row.cleaned_peak_n),
                    "exported_peak_n": int(len(peaks)),
                    "source_library": str(record.get("source_library", "unknown")),
                    "chemvista_evaluation_role": str(row.chemvista_evaluation_role),
                    "primary_train_eligible_20ppm": bool(row.primary_train_eligible_20ppm),
                }
            )
    os.replace(temporary, mgf_path)
    return pd.DataFrame(audit_rows)


def package_versions() -> dict[str, str]:
    packages = [
        "dreams",
        "torch",
        "numpy",
        "pandas",
        "pyopenms",
        "pytorch-lightning",
        "torchmetrics",
        "rdkit",
        "matchms",
        "numba",
        "umap-learn",
        "msml",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return versions


def checkpoint_hashes() -> dict[str, dict[str, Any]]:
    checkpoint_dir = DREAMS_SOURCE / "dreams" / "models" / "pretrained"
    hashes: dict[str, dict[str, Any]] = {}
    for name in ["embedding_model.ckpt", "ssl_model.ckpt"]:
        path = checkpoint_dir / name
        hashes[name] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() else None,
        }
    return hashes


def prepare_input() -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = select_five_development_spectra()
    mgf_path = OUTPUT_DIR / "development_5_spectra.mgf"
    selection_audit = write_mgf(selected, mgf_path)
    selection_path = OUTPUT_DIR / "development_5_spectra_selection.csv"
    selection_audit.to_csv(selection_path, index=False)
    print(json.dumps({
        "status": "PREPARED",
        "mgf_path": str(mgf_path),
        "selection_path": str(selection_path),
        "n_spectra": int(len(selection_audit)),
        "n_compounds": int(selection_audit["inchikey_2d"].nunique()),
        "external_test_unlocked": False,
    }, ensure_ascii=False, indent=2))
    return mgf_path, selection_path


def run_embedding() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_writable_runtime_cache()
    mgf_path = OUTPUT_DIR / "development_5_spectra.mgf"
    selection_path = OUTPUT_DIR / "development_5_spectra_selection.csv"
    if not mgf_path.exists() or not selection_path.exists():
        raise FileNotFoundError(
            "Prepared MGF input is missing. Run --stage prepare with env_ms first."
        )
    selection_audit = pd.read_csv(selection_path)
    if len(selection_audit) != N_SPECTRA:
        raise AssertionError(f"Prepared selection has {len(selection_audit)} rows")
    if selection_audit["inchikey_2d"].nunique() != N_SPECTRA:
        raise AssertionError("Prepared selection does not contain five distinct IK2D structures")
    if set(pd.to_numeric(selection_audit["cv_fold"], errors="raise").astype(int)) != set(range(N_SPECTRA)):
        raise AssertionError("Prepared selection does not cover all five frozen folds")
    if not selection_audit["chemvista_evaluation_role"].astype(str).eq("none").all():
        raise AssertionError("Prepared selection contains a ChemVista evaluation identity")

    # Import after configuring the writable Numba cache.
    import torch
    from dreams.api import dreams_embeddings
    from dreams.utils.io import read_mgf

    parsed = read_mgf(mgf_path)
    if len(parsed) != N_SPECTRA:
        raise AssertionError(f"DreaMS parsed {len(parsed)} spectra, expected {N_SPECTRA}")
    embeddings_raw = dreams_embeddings(
        mgf_path,
        batch_size=2,
        progress_bar=True,
        store_embs=False,
    )
    if isinstance(embeddings_raw, torch.Tensor):
        embeddings = embeddings_raw.detach().cpu().numpy()
    else:
        embeddings = np.asarray(embeddings_raw)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape != (N_SPECTRA, 1024):
        raise AssertionError(f"Unexpected DreaMS embedding shape: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise AssertionError("DreaMS embeddings contain non-finite values")

    embeddings_path = OUTPUT_DIR / "development_5_spectra_dreams_embeddings.npy"
    np.save(embeddings_path, embeddings, allow_pickle=False)
    embedding_summary = selection_audit[["mvp_row", "spectrum_id", "inchikey_2d", "cv_fold"]].copy()
    embedding_summary["embedding_dim"] = embeddings.shape[1]
    embedding_summary["finite_fraction"] = np.isfinite(embeddings).mean(axis=1)
    embedding_summary["l2_norm"] = np.linalg.norm(embeddings, axis=1)
    embedding_summary["mean"] = embeddings.mean(axis=1)
    embedding_summary["std"] = embeddings.std(axis=1)
    embedding_summary["embedding_sha256"] = [sha256_array(row) for row in embeddings]
    summary_path = OUTPUT_DIR / "development_5_spectra_embedding_summary.csv"
    embedding_summary.to_csv(summary_path, index=False)

    pip_freeze_path = OUTPUT_DIR / "env_dreams_pip_freeze.txt"
    pip_freeze_bytes = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
    ).stdout
    pip_freeze = pip_freeze_bytes.decode(locale.getpreferredencoding(False), errors="replace")
    pip_freeze_path.write_text(pip_freeze, encoding="utf-8", newline="\n")

    checks = [
        {"check_id": "DREAMS_MVP001", "description": "exactly five spectra", "status": "PASS"},
        {"check_id": "DREAMS_MVP002", "description": "five distinct IK2D structures", "status": "PASS"},
        {"check_id": "DREAMS_MVP003", "description": "one spectrum per frozen outer fold", "status": "PASS"},
        {"check_id": "DREAMS_MVP004", "description": "no ChemVista evaluation identity or spectrum", "status": "PASS"},
        {"check_id": "DREAMS_MVP005", "description": "all spectra are primary 20 ppm Development", "status": "PASS"},
        {"check_id": "DREAMS_MVP006", "description": "DreaMS parsed all five MGF spectra", "status": "PASS"},
        {"check_id": "DREAMS_MVP007", "description": "embedding shape is 5 x 1024", "status": "PASS"},
        {"check_id": "DREAMS_MVP008", "description": "all embedding values are finite", "status": "PASS"},
        {"check_id": "DREAMS_MVP009", "description": "CUDA inference environment available", "status": "PASS" if torch.cuda.is_available() else "FAIL"},
    ]
    checks_path = OUTPUT_DIR / "dreams_mvp_qc_checks.csv"
    pd.DataFrame(checks).to_csv(checks_path, index=False)
    if any(check["status"] != "PASS" for check in checks):
        raise AssertionError("One or more DreaMS MVP QC checks failed")

    artifacts = {
        path.name: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in [mgf_path, selection_path, embeddings_path, summary_path, pip_freeze_path, checks_path]
    }
    run_manifest = {
        "run_version": "PBT_MSMS_DREAMS_MVP_5_SPECTRA_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": SEED,
        "external_test_unlocked": False,
        "selection_rule": "one primary 20 ppm Development spectrum from each frozen outer fold; alternating polarity; ChemVista role none",
        "n_spectra": N_SPECTRA,
        "n_compounds": int(selection_audit["inchikey_2d"].nunique()),
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "embedding_array_sha256": sha256_array(embeddings),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dreams_source": {
            "repository": "https://github.com/pluskal-lab/DreaMS",
            "commit": DREAMS_COMMIT,
            "archive_path": str(DREAMS_ARCHIVE),
            "archive_sha256": sha256_file(DREAMS_ARCHIVE),
        },
        "checkpoints": checkpoint_hashes(),
        "inputs": {
            "public_manifest": str(PUBLIC_MANIFEST),
            "public_manifest_sha256": sha256_file(PUBLIC_MANIFEST),
            "public_stream": str(PUBLIC_STREAM),
            "public_stream_sha256": sha256_file(PUBLIC_STREAM),
        },
        "artifacts": artifacts,
        "qc_all_pass": True,
    }
    atomic_json(OUTPUT_DIR / "dreams_mvp_run_manifest.json", run_manifest)
    print(json.dumps({
        "status": "PASS",
        "output_dir": str(OUTPUT_DIR),
        "embedding_shape": list(embeddings.shape),
        "gpu": run_manifest["gpu"],
        "external_test_unlocked": False,
    }, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["prepare", "embed", "all"],
        default="all",
        help="Use env_ms for prepare and env_dreams for embed. 'all' is only for compatible NumPy pickle versions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"prepare", "all"}:
        prepare_input()
    if args.stage in {"embed", "all"}:
        run_embedding()


if __name__ == "__main__":
    main()
