from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

from pbt_msms.cli import NOTEBOOKS, _find_notebook
from pbt_msms.chemvista_external import platt_calibrate_probability
from pbt_msms.bangladesh_dia_external import _neutral_clusters
from pbt_msms.hazard_scoring import (
    B_TASK,
    EC50_TASK,
    LC50_TASK,
    NOEC_TASK,
    P_TASK,
    harmonized_full_hazard,
)
from pbt_msms.config import load_config
from pbt_msms.io import ChemVistaStore, PublicSpectrumStore
from pbt_msms.public_spectrum_cleaning import (
    MASTER_ACCEPTANCE_PPM,
    PRIMARY_PPM,
    SCHEMA_VERSION,
    SENSITIVITY_PPM,
    _atomic_promote,
    default_raw_files,
    load_mgf_stream,
    load_msp_stream,
)


ROOT = Path(__file__).resolve().parents[1]


class PipelineContractTests(unittest.TestCase):
    def test_environmental_cases_share_harmonized_hazard_formula(self):
        references = {
            P_TASK: np.asarray([0.1, 0.5, 0.9]),
            B_TASK: np.asarray([1.0, 2.0, 3.0]),
            EC50_TASK: np.asarray([-2.0, -1.0, 0.0]),
            LC50_TASK: np.asarray([-1.5, -0.5, 0.5]),
            NOEC_TASK: np.asarray([-3.0, -2.0, -1.0]),
        }
        predictions = {
            P_TASK: np.asarray([0.5, 0.9]),
            B_TASK: np.asarray([2.0, 3.0]),
            EC50_TASK: np.asarray([-1.0, -2.0]),
            LC50_TASK: np.asarray([-0.5, -1.5]),
            NOEC_TASK: np.asarray([-2.0, -3.0]),
        }
        endpoint, toxicity, hazard = harmonized_full_hazard(predictions, references)
        expected_toxicity = np.max(
            np.column_stack([endpoint[EC50_TASK], endpoint[LC50_TASK], endpoint[NOEC_TASK]]),
            axis=1,
        )
        np.testing.assert_allclose(toxicity, expected_toxicity)
        np.testing.assert_allclose(
            hazard,
            np.cbrt(endpoint[P_TASK] * endpoint[B_TASK] * expected_toxicity),
        )
        self.assertTrue(((hazard > 0) & (hazard < 1)).all())

    def test_dreams_outer_platt_probability_map(self):
        raw = np.asarray([0.1, 0.5, 0.9])
        calibrated = platt_calibrate_probability(raw, 0.176332098019564, 0.16387604632860078)
        expected = 1.0 / (1.0 + np.exp(-(0.176332098019564 * np.log(raw / (1 - raw)) + 0.16387604632860078)))
        np.testing.assert_allclose(calibrated, expected)
        self.assertGreater(float(np.mean(np.abs(calibrated - raw))), 0.1)

    def test_cross_polarity_clustering_limits_full_cluster_span(self):
        frame = pd.DataFrame({
            "spectrum_id": ["p1", "n1", "p2"],
            "mode": ["Pos", "Neg", "Pos"],
            "adduct": ["[M+H]+", "[M-H]-", "[M+H]+"],
            "precursor_mz": [101.007276466621, 98.992723533379, 101.007276466621],
            "rt_min": [1.00, 1.14, 1.28],
        })
        clustered = _neutral_clusters(frame, ppm=10.0, rt_tolerance=0.15)
        self.assertEqual(clustered["cross_polarity_cluster"].nunique(), 2)
        self.assertLessEqual(clustered["cross_polarity_rt_span_min"].max(), 0.15 + 1e-12)

    def test_configuration_matches_frozen_analysis(self):
        config = load_config(ROOT / "configs" / "default.yaml")
        self.assertEqual(config.data["split"]["inner_folds"], 4)
        self.assertEqual(config.data["split"]["grouping_method"], "morgan_butina")
        self.assertEqual(config.data["evaluation"]["bootstrap_iterations"], 1000)
        self.assertFalse(config.data["evaluation"]["unlock_external"])

    def test_notebook_chain_is_unique(self):
        for prefix in NOTEBOOKS.values():
            self.assertTrue(_find_notebook(ROOT, prefix).is_file())

    def test_public_cleaning_contract_and_notebook_syntax(self):
        self.assertEqual(SCHEMA_VERSION, "ENVMOLBENCH_CLEANED_V2")
        self.assertEqual(PRIMARY_PPM, 20.0)
        self.assertEqual(SENSITIVITY_PPM, (50.0, 100.0))
        self.assertEqual(MASTER_ACCEPTANCE_PPM, 100.0)

        notebook_path = ROOT / "质谱数据" / "质谱数据清洗.ipynb"
        if not notebook_path.is_file():
            self.skipTest("原始公共谱库清洗 Notebook 不在当前归档资料包中")
        payload = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload["cells"]
            if cell.get("cell_type") == "code"
        )
        compile(source, str(notebook_path), "exec")
        self.assertIn("run_public_spectrum_cleaning", source)
        self.assertIn("overwrite=OVERWRITE_EXISTING", source)
        self.assertNotIn(r"E:\质谱数据集", source)

    def test_raw_public_parsers_can_read_first_real_record(self):
        raw_files = default_raw_files(ROOT)
        if not all(path.is_file() for path in raw_files.values()):
            self.skipTest("原始公共谱库不在当前工作副本中")
        first_gnps = next(iter(load_mgf_stream(raw_files["GNPS"])))
        first_massbank = next(iter(load_msp_stream(raw_files["MassBank"])))
        first_mona = next(iter(load_msp_stream(raw_files["MoNA"])))
        for spectrum in [first_gnps, first_massbank, first_mona]:
            self.assertGreaterEqual(len(spectrum.peaks.mz), 1)
            self.assertEqual(len(spectrum.peaks.mz), len(spectrum.peaks.intensities))

    def test_atomic_public_output_promotion_and_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            final_a, final_b = directory / "a.pkl", directory / "b.pkl"
            temp_a, temp_b = directory / "a.building", directory / "b.building"
            final_a.write_text("old-a", encoding="utf-8")
            final_b.write_text("old-b", encoding="utf-8")
            temp_a.write_text("new-a", encoding="utf-8")
            temp_b.write_text("new-b", encoding="utf-8")
            _atomic_promote([(temp_a, final_a), (temp_b, final_b)])
            self.assertEqual(final_a.read_text(encoding="utf-8"), "new-a")
            self.assertEqual(final_b.read_text(encoding="utf-8"), "new-b")

            rollback_a = directory / "rollback-a.building"
            missing_b = directory / "missing-b.building"
            rollback_a.write_text("bad-new-a", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                _atomic_promote([(rollback_a, final_a), (missing_b, final_b)])
            self.assertEqual(final_a.read_text(encoding="utf-8"), "new-a")
            self.assertEqual(final_b.read_text(encoding="utf-8"), "new-b")

    def test_frozen_manifests_are_locked_and_pass(self):
        split = ROOT / "outputs" / "pbt_msms_v1" / "split_v1" / "split_freeze_manifest.json"
        baseline = ROOT / "outputs" / "pbt_msms_v1" / "structure_baselines_v1" / "structure_baseline_run_manifest.json"
        transformer = ROOT / "outputs" / "pbt_msms_v1" / "set_transformer_v1" / "set_transformer_run_manifest.json"
        if split.exists():
            self.assertIn("novel", split.read_text(encoding="utf-8").lower())
        if baseline.exists():
            payload = json.loads(baseline.read_text(encoding="utf-8"))
            self.assertFalse(payload["external_test_unlocked"])
            self.assertEqual(payload["qc_status"], "PASS")
        if transformer.exists():
            payload = json.loads(transformer.read_text(encoding="utf-8"))
            self.assertFalse(payload["external_test_unlocked"])
            self.assertEqual(payload["qc_status"], "PASS")
            self.assertEqual(payload["pretraining"]["development_overlap"], 0)
            self.assertEqual(payload["pretraining"]["frozen_external_overlap"], 0)

    def test_chemvista_auxiliary_pickles_are_real_pickles(self):
        directory = ROOT / "验证实验谱图数据" / "ChemVista_cleaned"
        for name in [
            "ChemVista_removed_exact_duplicates.pkl",
            "ChemVista_cross_identity_spectrum_collisions.pkl",
            "ChemVista_failed_files.pkl",
        ]:
            path = directory / name
            if path.exists():
                self.assertIsInstance(pd.read_pickle(path), pd.DataFrame)

    def test_dreams_head_outputs_are_locked_grouped_and_calibrated(self):
        output = ROOT / "outputs" / "pbt_msms_v1" / "dreams_heads_v1"
        manifest_path = output / "dreams_head_run_manifest.json"
        if not manifest_path.exists():
            self.skipTest("DreaMS head pipeline has not been run")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "PASS")
        self.assertEqual(manifest["qc_status"], "PASS")
        self.assertFalse(manifest["external_test_unlocked"])
        self.assertEqual(manifest["representation"]["dimension"], 1024)
        self.assertEqual(manifest["outer_cv"]["n_folds"], 5)
        self.assertEqual(manifest["inner_cv"]["n_folds"], 4)

        qc = pd.read_csv(output / "dreams_head_qc_checks.csv")
        self.assertTrue(qc["status"].eq("PASS").all())
        self.assertIn("DHEAD017", set(qc["check_id"]))
        fold_integrity = pd.read_csv(output / "dreams_head_fold_integrity.csv")
        self.assertTrue(fold_integrity["group_overlap"].eq(0).all())

        oof = pd.read_pickle(output / "dreams_head_compound_level_oof.pkl")
        self.assertIn("y_probability_raw", oof.columns)
        self.assertIn("calibration_method", oof.columns)
        main = oof[
            oof["representation_route"].eq("all_spectra_compound_median")
            & oof["model"].eq("nested_selected_primary")
        ]
        self.assertEqual(main.groupby(["task", "inchikey_2d"]).size().max(), 1)
        self.assertEqual(main["task"].nunique(), 5)

        # Publication bundles and correction backups contain archival copies.
        notebooks = list((ROOT / "建模分析").glob("08_*.ipynb"))
        self.assertEqual(len(notebooks), 1)
        payload = json.loads(notebooks[0].read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload["cells"]
            if cell.get("cell_type") == "code"
        )
        compile(source, str(notebooks[0]), "exec")


if __name__ == "__main__":
    unittest.main()
