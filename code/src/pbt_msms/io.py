from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


class PublicSpectrumStore:
    """Random-access reader for the existing streamed EnvMolBench pickle.

    The stream begins with ("CONFIG", mapping), followed by pickled
    (inchikey_2d, record) tuples. Offsets are provided by the companion index.
    Only trusted project-generated pickle files should be opened.
    """

    def __init__(self, stream_path: Path, index_path: Path):
        self.stream_path = Path(stream_path)
        self.index_path = Path(index_path)
        with self.index_path.open("rb") as handle:
            index = pickle.load(handle)
        if not isinstance(index, dict):
            raise TypeError("Public spectrum index must be a dictionary")
        self.index: dict[str, list[int]] = index

    def keys(self) -> set[str]:
        return set(self.index)

    def spectrum_count(self, inchikey_2d: str) -> int:
        return len(self.index.get(inchikey_2d, []))

    def read_at(self, offset: int) -> tuple[str, dict[str, Any]]:
        with self.stream_path.open("rb") as handle:
            handle.seek(int(offset))
            key, record = pickle.load(handle)
        return key, record

    def iter_compound(self, inchikey_2d: str) -> Iterator[dict[str, Any]]:
        with self.stream_path.open("rb") as handle:
            for offset in self.index.get(inchikey_2d, []):
                handle.seek(int(offset))
                key, record = pickle.load(handle)
                if key != inchikey_2d:
                    raise ValueError(
                        f"Index corruption at {offset}: expected {inchikey_2d}, got {key}"
                    )
                yield record

    def header(self) -> dict[str, Any]:
        with self.stream_path.open("rb") as handle:
            marker, payload = pickle.load(handle)
        if marker != "CONFIG":
            raise ValueError("Unexpected public spectrum stream header")
        return payload


class ChemVistaStore:
    """Reader for the trusted ChemVista index/peak dictionaries."""

    def __init__(self, index_path: Path, peaks_path: Path):
        self.index = pd.read_pickle(index_path).copy()
        self.peaks: dict[str, dict[str, np.ndarray]] = pd.read_pickle(peaks_path)
        if "spectrum_id" not in self.index.columns:
            self.index["spectrum_id"] = self.index.index.astype(str)
        self.index["spectrum_id"] = self.index["spectrum_id"].astype(str)

    def compound_keys(self) -> set[str]:
        return set(self.index["inchikey_2d"].dropna().astype(str))

    def spectrum_ids(self, inchikey_2d: str) -> list[str]:
        frame = self.index[self.index["inchikey_2d"].astype(str).eq(inchikey_2d)]
        return frame["spectrum_id"].astype(str).tolist()

    def get(self, spectrum_id: str) -> tuple[pd.Series, dict[str, np.ndarray]]:
        rows = self.index[self.index["spectrum_id"].eq(str(spectrum_id))]
        if len(rows) != 1:
            raise KeyError(f"Expected one ChemVista spectrum {spectrum_id}, found {len(rows)}")
        return rows.iloc[0], self.peaks[str(spectrum_id)]

