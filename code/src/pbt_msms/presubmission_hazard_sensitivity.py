"""Presubmission harmonized hazard-score and field-ranking sensitivity.

The analysis reuses frozen endpoint predictions.  It converts every endpoint
to the same Development-OOF-referenced hazard percentile and applies identical
hazard formulas in the Portugal and Bangladesh cases.  It is a ranking
sensitivity analysis, not validation that unresolved features are hazardous.
"""

from __future__ import annotations

from .deployment import deployment_comparison_dir

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr

from pbt_msms.chemvista_external import TASKS, locate_root


VERSION = "PBT_MSMS_PRESUBMISSION_HAZARD_SENSITIVITY_41_V1_3"
OUTPUT_SUBDIR = Path("outputs/pbt_msms_v1/presubmission_robustness_v1/hazard_score_sensitivity_41")
PRIMARY_VARIANT = "harmonized_full_max_geometric"
PORTUGAL_AGGREGATION_COMPARATOR = "full_median_geometric"
BANGLADESH_AGGREGATION_COMPARATOR = "full_max_arithmetic"
TASK_ORDER = list(TASKS)
P_TASK = "P_screen_not_ready_binary"
B_TASK = "B_LOGBCF_REGRESSION"
EC50_TASK = "T_EC50_MG_L_REGRESSION"
LC50_TASK = "T_LC50_MG_L_REGRESSION"
NOEC_TASK = "T_NOEC_MG_L_REGRESSION"
TOX_TASKS = [EC50_TASK, LC50_TASK, NOEC_TASK]

VARIANTS = {
    "harmonized_full_max_geometric": {
        "required": TASK_ORDER,
        "toxicity": "max",
        "aggregation": "geometric_p_b_t",
        "interpretation": "Proposed harmonized full P/B/T score; any high toxicity endpoint can define T.",
    },
    "full_median_geometric": {
        "required": TASK_ORDER,
        "toxicity": "median",
        "aggregation": "geometric_p_b_t",
        "interpretation": "Toxicity-operator sensitivity on the common Development-referenced percentile scale.",
    },
    "full_max_arithmetic": {
        "required": TASK_ORDER,
        "toxicity": "max",
        "aggregation": "arithmetic_p_b_t",
        "interpretation": "Aggregation-operator sensitivity on the common percentile scale.",
    },
    "internally_supported_p_b_lc50": {
        "required": [P_TASK, B_TASK, LC50_TASK],
        "toxicity": "lc50_only",
        "aggregation": "geometric_p_b_t",
        "interpretation": "Sensitivity restricted to endpoints supported against the original six-field metadata baseline in internal evaluation.",
    },
    "novel_supported_b_lc50": {
        "required": [B_TASK, LC50_TASK],
        "toxicity": "lc50_only",
        "aggregation": "geometric_b_t",
        "interpretation": "Sensitivity restricted to endpoints supported against the original six-field metadata baseline in the held-out Novel cohort; not a full PBT score.",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    compression = "gzip" if path.name.endswith(".gz") else None
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def paths(root: Path) -> dict[str, Path]:
    out = root / OUTPUT_SUBDIR
    portugal = root / "external_data/portugal_wastewater_5830725"
    bangladesh = root / "external_data/bangladesh_surface_water"
    return {
        "root": root,
        "out": out,
        "figures": out / "figures",
        "development_oof": deployment_comparison_dir(root / "outputs/pbt_msms_v1") / "19_selected_model_oof_calibrated.csv.gz",
        "portugal_features": portugal / "04_hazard_ranking_24/24_feature_predictions_identity_blind.csv.gz",
        "portugal_groups": portugal / "04_hazard_ranking_24/24_case_feature_groups_identity_blind.csv.gz",
        "portugal_recovery": portugal / "06_revealed_evaluation/25_identity_reveal/25_known_target_recovery.csv",
        "portugal_group_matches": portugal / "06_revealed_evaluation/25_identity_reveal/25_case_group_library_matches.csv.gz",
        "bangladesh_clusters": bangladesh / "06_revealed_evaluation/31_feature_prioritization_v2/31_cross_polarity_cluster_ranking.csv.gz",
        "analysis_plan": out / "41_analysis_plan.json",
        "portugal_rankings": out / "41_portugal_harmonized_rankings.csv.gz",
        "bangladesh_rankings": out / "41_bangladesh_harmonized_rankings.csv.gz",
        "rank_comparison": out / "41_rank_stability_summary.csv",
        "target_recovery": out / "41_portugal_known_target_recovery_sensitivity.csv",
        "orthogonality": out / "41_hazard_occurrence_orthogonality.csv",
        "migration": out / "41_top_candidate_migration.csv.gz",
        "formula_table": out / "41_hazard_formula_definitions.csv",
        "qc": out / "41_qc_checks.csv",
        "summary": out / "41_manuscript_result_summary.md",
        "manifest": out / "41_run_manifest.json",
        "figure_png": out / "figures/41_hazard_score_sensitivity.png",
        "figure_pdf": out / "figures/41_hazard_score_sensitivity.pdf",
    }


def analysis_plan() -> dict[str, Any]:
    return {
        "version": VERSION,
        "status": "FIXED_BEFORE_EXECUTION",
        "revision_note": (
            "V1.3 treats the harmonized full max/geometric score as the canonical "
            "formula in both field cases and retains alternative endpoint and "
            "aggregation choices only as sensitivity analyses."
        ),
        "purpose": "Presubmission post-freeze hazard-score and ranking sensitivity analysis.",
        "common_scale": "Development selected-model OOF empirical hazard percentiles",
        "direction": {"P": "higher is hazard", "B": "higher is hazard", "toxicity": "lower effect concentration is hazard"},
        "primary_variant": PRIMARY_VARIANT,
        "evaluation_universes": {
            "Portugal": "Canonical 16,627-group common ranking universe.",
            "Bangladesh": (
                "All 16,768 frozen candidate clusters; endpoint rejection is represented by "
                "the frozen confidence component rather than exclusion from the ranking universe."
            ),
        },
        "variants": VARIANTS,
        "case_specific_operational_layers": {
            "Portugal": "occurrence-assisted score = sqrt(hazard * detection frequency)",
            "Bangladesh": "priority = (0.70*hazard + 0.30*exposure) * confidence",
        },
        "known_target_boundary": (
            "Portugal known-CEC recovery measures continuity with previously monitored contaminants; "
            "it is not a hazard-classification truth set."
        ),
        "unknown_boundary": (
            "Neither field case validates that unresolved high-ranking features are hazardous."
        ),
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs(root: str | Path | None = None) -> pd.DataFrame:
    root_path = locate_root(root)
    p = paths(root_path)
    required = [
        "development_oof", "portugal_features", "portugal_groups",
        "portugal_recovery", "portugal_group_matches", "bangladesh_clusters",
    ]
    rows = [{"input": name, "exists": p[name].exists(), "path": str(p[name])} for name in required]
    frame = pd.DataFrame(rows)
    if not frame["exists"].all():
        raise FileNotFoundError("Missing upstream artifacts:\n" + "\n".join(frame.loc[~frame["exists"], "path"]))
    pt = pd.read_csv(p["portugal_features"], nrows=5, low_memory=False)
    bd = pd.read_csv(p["bangladesh_clusters"], nrows=5, low_memory=False)
    required_pt = {
        "case_feature_id", "p_probability", "b_logbcf", "t_ec50_log10_mg_l",
        "t_lc50_log10_mg_l", "t_noec_log10_mg_l", "joint_pbt_hazard_score",
    }
    required_bd = {
        "cross_polarity_cluster", *TASK_ORDER, "hazard_component",
        "exposure_component", "confidence_component", "priority_score_primary",
    }
    if not required_pt.issubset(pt.columns):
        raise RuntimeError(f"Portugal schema mismatch: {required_pt - set(pt.columns)}")
    if not required_bd.issubset(bd.columns):
        raise RuntimeError(f"Bangladesh schema mismatch: {required_bd - set(bd.columns)}")
    return frame


def empirical_percentile(values: np.ndarray, reference: np.ndarray, high_is_hazard: bool) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    values = np.asarray(values, dtype=float)
    if not len(reference) or not np.isfinite(reference).all():
        raise ValueError("Development reference must be non-empty and finite")
    if high_is_hazard:
        ranks = np.searchsorted(reference, values, side="right")
    else:
        ranks = len(reference) - np.searchsorted(reference, values, side="left")
    return (ranks + 0.5) / (len(reference) + 1.0)


def development_references(path: Path) -> dict[str, np.ndarray]:
    oof = pd.read_csv(path, low_memory=False)
    references = {}
    for task in TASK_ORDER:
        values = pd.to_numeric(
            oof.loc[oof["task"].eq(task), "y_score_calibrated"], errors="coerce"
        ).dropna().to_numpy(float)
        if not len(values):
            raise RuntimeError(f"No Development OOF reference values for {task}")
        references[task] = values
    return references


def _add_hazard_percentiles(
    frame: pd.DataFrame,
    raw_columns: dict[str, str],
    references: dict[str, np.ndarray],
) -> pd.DataFrame:
    result = frame.copy()
    for task in TASK_ORDER:
        values = pd.to_numeric(result[raw_columns[task]], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite frozen prediction values for {task}")
        result[f"h__{task}"] = empirical_percentile(
            values, references[task], high_is_hazard=task not in TOX_TASKS
        )
    return result


def _retained_mask(frame: pd.DataFrame, required: list[str], retained_columns: dict[str, str]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for task in required:
        column = retained_columns[task]
        if column not in frame:
            raise KeyError(f"Missing retained column: {column}")
        mask &= frame[column].fillna(False).astype(bool)
    return mask


def _variant_score(frame: pd.DataFrame, specification: dict[str, Any]) -> tuple[pd.Series, pd.Series, pd.Series]:
    p = frame[f"h__{P_TASK}"]
    b = frame[f"h__{B_TASK}"]
    tox = frame[[f"h__{task}" for task in TOX_TASKS]]
    if specification["toxicity"] == "max":
        t = tox.max(axis=1)
    elif specification["toxicity"] == "median":
        t = tox.median(axis=1)
    elif specification["toxicity"] == "lc50_only":
        t = frame[f"h__{LC50_TASK}"]
    else:
        raise KeyError(specification["toxicity"])
    if specification["aggregation"] == "geometric_p_b_t":
        score = np.cbrt(p * b * t)
    elif specification["aggregation"] == "arithmetic_p_b_t":
        score = (p + b + t) / 3.0
    elif specification["aggregation"] == "geometric_b_t":
        score = np.sqrt(b * t)
    else:
        raise KeyError(specification["aggregation"])
    return pd.Series(score, index=frame.index), p, t


def prepare_portugal(
    p: dict[str, Path], references: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, float]:
    features = pd.read_csv(p["portugal_features"], low_memory=False)
    groups = pd.read_csv(p["portugal_groups"], low_memory=False)
    low_depth_only = (
        features.assign(
            _low=features["low_ms2_depth_sample"].fillna(False).astype(bool)
        )
        .groupby("case_feature_id", sort=False)["_low"]
        .all()
    )
    frozen_common_ids = set(
        groups.loc[
            groups["median_primary_hazard_score"].notna()
            & ~groups["case_feature_id"].map(low_depth_only).fillna(False),
            "case_feature_id",
        ]
    )
    raw = {
        P_TASK: "p_probability", B_TASK: "b_logbcf", EC50_TASK: "t_ec50_log10_mg_l",
        LC50_TASK: "t_lc50_log10_mg_l", NOEC_TASK: "t_noec_log10_mg_l",
    }
    retained = {
        P_TASK: "p_retained", B_TASK: "b_retained", EC50_TASK: "t_ec50_retained",
        LC50_TASK: "t_lc50_retained", NOEC_TASK: "t_noec_retained",
    }
    features = _add_hazard_percentiles(features, raw, references)
    # Reuse the canonical Notebook 24 percentiles for Portugal. Recomputing from
    # rounded CSV predictions can move a few exact-tie P values by one empirical
    # rank (1 / 646), even though all other endpoint percentiles reproduce to
    # floating-point precision.  The frozen columns are the authoritative
    # Development-anchored values and make canonical-formula reproduction exact.
    frozen_percentiles = {
        P_TASK: "p_development_anchored_hazard_percentile",
        B_TASK: "b_development_anchored_hazard_percentile",
        EC50_TASK: "t_ec50_development_anchored_hazard_percentile",
        LC50_TASK: "t_lc50_development_anchored_hazard_percentile",
        NOEC_TASK: "t_noec_development_anchored_hazard_percentile",
    }
    for task, column in frozen_percentiles.items():
        features[f"h__{task}"] = pd.to_numeric(features[column], errors="raise")
    reproduction_score, _, _ = _variant_score(features, VARIANTS[PRIMARY_VARIANT])
    reproduction_error = float(np.nanmax(np.abs(
        reproduction_score.to_numpy(float) - features["joint_pbt_hazard_score"].to_numpy(float)
    )))

    common_group_columns = [
        "case_feature_id", "mode", "median_precursor_mz", "median_rt_seconds",
        "detection_frequency_fraction", "median_sample_abundance_percentile",
        "frequency_baseline_rank", "abundance_baseline_rank",
    ]
    group_metadata = groups[common_group_columns].drop_duplicates("case_feature_id")
    group_metadata["frozen_common_universe"] = group_metadata["case_feature_id"].isin(frozen_common_ids)
    rows = []
    for variant, specification in VARIANTS.items():
        score, _, _ = _variant_score(features, specification)
        eligible = _retained_mask(features, specification["required"], retained)
        working = features[["case_feature_id"]].copy()
        working["hazard_score"] = score.where(eligible)
        working["eligible"] = eligible
        aggregated = working.groupby("case_feature_id", as_index=False).agg(
            hazard_score=("hazard_score", "median"),
            eligible_feature_n=("eligible", "sum"),
            member_feature_n=("eligible", "size"),
        ).merge(group_metadata, on="case_feature_id", how="left", validate="one_to_one")
        aggregated = aggregated.loc[aggregated["frozen_common_universe"]].copy()
        aggregated["case"] = "Portugal"
        aggregated["variant"] = variant
        aggregated["eligible"] = aggregated["hazard_score"].notna()
        aggregated["hazard_rank"] = aggregated["hazard_score"].rank(method="min", ascending=False, na_option="bottom")
        aggregated["operational_score"] = np.sqrt(
            aggregated["hazard_score"] * aggregated["detection_frequency_fraction"]
        )
        aggregated["operational_rank"] = aggregated["operational_score"].rank(method="min", ascending=False, na_option="bottom")
        rows.append(aggregated)

    result = pd.concat(rows, ignore_index=True, sort=False)
    return result, reproduction_error


def prepare_bangladesh(
    p: dict[str, Path], references: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, float]:
    clusters = pd.read_csv(p["bangladesh_clusters"], low_memory=False)
    if clusters["cross_polarity_cluster"].duplicated().any():
        clusters = clusters.sort_values(["cross_polarity_cluster", "cluster_rank_within"]).drop_duplicates("cross_polarity_cluster")
    raw = {task: task for task in TASK_ORDER}
    retained = {task: f"retained__{task}" for task in TASK_ORDER}
    clusters = _add_hazard_percentiles(clusters, raw, references)
    common = clusters[[
        "cross_polarity_cluster", "spectrum_id", "mode", "precursor_mz", "rt_min",
        "sample_detection_fraction", "exposure_component", "confidence_component",
        "cluster_member_n", "modes",
    ]].copy()
    rows = []
    canonical_score: pd.Series | None = None
    for variant, specification in VARIANTS.items():
        score, _, _ = _variant_score(clusters, specification)
        if variant == PRIMARY_VARIANT:
            canonical_score = score
        eligible = _retained_mask(clusters, specification["required"], retained)
        result = common.copy()
        result["case"] = "Bangladesh"
        result["variant"] = variant
        # The frozen Bangladesh design ranks every candidate cluster.  Endpoint
        # rejection is carried by confidence_component; it is not an exclusion
        # rule.  Keep all raw-prediction hazard scores, matching the corrected
        # all-16,768-cluster canonical analysis.
        result["hazard_score"] = score
        result["eligible"] = eligible
        result["hazard_rank"] = result["hazard_score"].rank(method="min", ascending=False, na_option="bottom")
        result["operational_score"] = (
            (0.70 * result["hazard_score"] + 0.30 * result["exposure_component"])
            * result["confidence_component"]
        )
        result["operational_rank"] = result["operational_score"].rank(method="min", ascending=False, na_option="bottom")
        rows.append(result)
    if canonical_score is None:
        raise RuntimeError("Canonical Bangladesh hazard variant was not computed")
    reproduction_error = float(np.nanmax(np.abs(
        canonical_score.to_numpy(float) - clusters["hazard_component"].to_numpy(float)
    )))
    return pd.concat(rows, ignore_index=True, sort=False), reproduction_error


def formula_table() -> pd.DataFrame:
    rows = []
    for name, spec in VARIANTS.items():
        rows.append({
            "variant": name,
            "required_endpoints": ";".join(spec["required"]),
            "toxicity_operator": spec["toxicity"],
            "hazard_aggregation": spec["aggregation"],
            "interpretation": spec["interpretation"],
            "primary_harmonized_variant": name == PRIMARY_VARIANT,
        })
    return pd.DataFrame(rows)


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return np.nan
    return float(spearmanr(x[mask], y[mask]).statistic)


def rank_stability(portugal: pd.DataFrame, bangladesh: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specifications = [
        ("Portugal", portugal, "case_feature_id"),
        ("Bangladesh", bangladesh, "cross_polarity_cluster"),
    ]
    for case, frame, identifier in specifications:
        primary = frame.loc[frame["variant"].eq(PRIMARY_VARIANT), [
            identifier, "hazard_score", "hazard_rank", "operational_score", "operational_rank"
        ]].rename(columns={c: f"primary_{c}" for c in ["hazard_score", "hazard_rank", "operational_score", "operational_rank"]})
        for variant in VARIANTS:
            other = frame.loc[frame["variant"].eq(variant), [
                identifier, "hazard_score", "hazard_rank", "operational_score", "operational_rank"
            ]].rename(columns={c: f"other_{c}" for c in ["hazard_score", "hazard_rank", "operational_score", "operational_rank"]})
            merged = primary.merge(other, on=identifier, how="inner", validate="one_to_one")
            for ranking in ["hazard", "operational"]:
                valid = merged.dropna(
                    subset=[f"primary_{ranking}_score", f"other_{ranking}_score"]
                ).copy()
                score_a = valid[f"primary_{ranking}_score"]
                score_b = valid[f"other_{ranking}_score"]
                rho = _safe_spearman(score_a, score_b)
                cutoffs = [("top_5pct", max(1, math.ceil(0.05 * len(valid)))),
                           ("top_10pct", max(1, math.ceil(0.10 * len(valid)))),
                           ("top_20pct", max(1, math.ceil(0.20 * len(valid))))]
                if case == "Bangladesh":
                    cutoffs.append(("top_100", min(100, len(valid))))
                for label, n_top in cutoffs:
                    top_a = set(valid.nlargest(n_top, f"primary_{ranking}_score")[identifier])
                    top_b = set(valid.nlargest(n_top, f"other_{ranking}_score")[identifier])
                    overlap = len(top_a & top_b)
                    union = len(top_a | top_b)
                    rows.append({
                        "case": case, "primary_variant": PRIMARY_VARIANT, "comparison_variant": variant,
                        "ranking": ranking, "cutoff": label, "universe_n": len(valid), "top_n": n_top,
                        "spearman_score": rho, "overlap_n": overlap,
                        "overlap_fraction_of_primary": overlap / len(top_a) if top_a else np.nan,
                        "jaccard": overlap / union if union else np.nan,
                    })
    return pd.DataFrame(rows)


def portugal_known_target_recovery(p: dict[str, Path], rankings: pd.DataFrame) -> pd.DataFrame:
    recovery = pd.read_csv(p["portugal_recovery"], low_memory=False)
    watchlist_ids = set(
        recovery.loc[
            recovery["published_watchlist_cec"].fillna(False).astype(bool), "target_id"
        ]
    )
    matches = pd.read_csv(p["portugal_group_matches"], low_memory=False)
    matches = matches.loc[
        matches["is_nonambiguous_published_cec_match"].fillna(False).astype(bool)
        & matches["case_feature_id"].notna()
    ].drop_duplicates(["case_feature_id", "target_id"])
    rows = []
    for variant, frame in rankings.groupby("variant", sort=False):
        universe_ids = set(frame.loc[frame["hazard_score"].notna(), "case_feature_id"])
        universe_n = len(universe_ids)
        in_universe = matches.loc[matches["case_feature_id"].isin(universe_ids)].copy()
        scopes = {
            "all_published_cec_matches": in_universe,
            "published_watchlist_matches": in_universe.loc[
                in_universe["target_id"].isin(watchlist_ids)
            ],
        }
        for ranking, rank_col in [("hazard", "hazard_rank"), ("occurrence_assisted", "operational_rank")]:
            for top_fraction in [0.05, 0.10, 0.20]:
                top_n = max(1, math.ceil(top_fraction * universe_n))
                top_ids = set(
                    frame.loc[frame["case_feature_id"].isin(universe_ids)]
                    .nsmallest(top_n, rank_col)["case_feature_id"]
                )
                for scope, scoped in scopes.items():
                    target_total = scoped["target_id"].nunique()
                    known_groups = set(scoped["case_feature_id"])
                    captured = scoped.loc[scoped["case_feature_id"].isin(top_ids), "target_id"].nunique()
                    group_hits = len(top_ids & known_groups)
                    density = group_hits / top_n if top_n else np.nan
                    background = len(known_groups) / universe_n if universe_n else np.nan
                    p_value = float(hypergeom.sf(group_hits - 1, universe_n, len(known_groups), top_n)) if known_groups else np.nan
                    rows.append({
                        "variant": variant, "ranking": ranking, "label_scope": scope,
                        "top_fraction": top_fraction, "universe_group_n": universe_n,
                        "top_group_n": top_n, "known_hit_group_n": group_hits,
                        "known_hit_density": density, "background_known_hit_prevalence": background,
                        "enrichment_fold": density / background if background else np.nan,
                        "hypergeometric_p_value": p_value,
                        "captured_unique_target_n": captured,
                        "recoverable_unique_target_n_in_universe": target_total,
                        "unique_target_capture_fraction": captured / target_total if target_total else np.nan,
                        "interpretation": "Known-target continuity benchmark; not hazard truth.",
                    })
    return pd.DataFrame(rows)


def orthogonality(portugal: pd.DataFrame, bangladesh: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, frame in portugal.groupby("variant", sort=False):
        frame = frame.dropna(subset=["hazard_score", "frequency_baseline_rank"]).copy()
        n = len(frame)
        top_n = max(1, math.ceil(0.20 * n))
        hazard = set(frame.nsmallest(top_n, "hazard_rank")["case_feature_id"])
        occurrence = set(frame.nsmallest(top_n, "frequency_baseline_rank")["case_feature_id"])
        rows.append({
            "case": "Portugal", "variant": variant, "comparison": "hazard_top20_vs_frequency_top20",
            "universe_n": n, "top_n": top_n, "overlap_n": len(hazard & occurrence),
            "hazard_only_n": len(hazard - occurrence), "occurrence_only_n": len(occurrence - hazard),
            "jaccard": len(hazard & occurrence) / len(hazard | occurrence),
        })
    for variant, frame in bangladesh.groupby("variant", sort=False):
        frame = frame.dropna(subset=["hazard_score", "exposure_component"]).copy()
        n = len(frame)
        top_n = min(100, n)
        hazard = set(frame.nsmallest(top_n, "hazard_rank")["cross_polarity_cluster"])
        exposure = set(frame.nlargest(top_n, "exposure_component")["cross_polarity_cluster"])
        rows.append({
            "case": "Bangladesh", "variant": variant, "comparison": "hazard_top100_vs_exposure_top100",
            "universe_n": n, "top_n": top_n, "overlap_n": len(hazard & exposure),
            "hazard_only_n": len(hazard - exposure), "occurrence_only_n": len(exposure - hazard),
            "jaccard": len(hazard & exposure) / len(hazard | exposure),
        })
    return pd.DataFrame(rows)


def candidate_migration(portugal: pd.DataFrame, bangladesh: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case, frame, identifier, comparator, rank_col, top_n in [
        ("Portugal", portugal, "case_feature_id", PORTUGAL_AGGREGATION_COMPARATOR, "operational_rank", math.ceil(0.20 * portugal["case_feature_id"].nunique())),
        ("Bangladesh", bangladesh, "cross_polarity_cluster", BANGLADESH_AGGREGATION_COMPARATOR, "operational_rank", 100),
    ]:
        primary = frame.loc[frame["variant"].eq(PRIMARY_VARIANT)].nsmallest(top_n, rank_col)
        other = frame.loc[frame["variant"].eq(comparator)].nsmallest(top_n, rank_col)
        primary_ids = set(primary[identifier])
        other_ids = set(other[identifier])
        universe = primary[[identifier, "hazard_score", "operational_score", "hazard_rank", "operational_rank"]].merge(
            other[[identifier, "hazard_score", "operational_score", "hazard_rank", "operational_rank"]],
            on=identifier, how="outer", suffixes=("_canonical", "_comparator"),
        )
        universe["case"] = case
        universe["comparison_variant"] = comparator
        universe["identifier"] = universe[identifier]
        universe["migration"] = np.select(
            [universe[identifier].isin(primary_ids & other_ids), universe[identifier].isin(primary_ids - other_ids), universe[identifier].isin(other_ids - primary_ids)],
            ["retained", "entered_canonical_top_set", "left_canonical_top_set"],
            default="outside_both",
        )
        rows.append(universe.drop(columns=[identifier]))
    return pd.concat(rows, ignore_index=True, sort=False)


def make_figure(
    portugal: pd.DataFrame, bangladesh: pd.DataFrame, stability: pd.DataFrame, p: dict[str, Path]
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), constrained_layout=True)
    cases = [
        (axes[0], "Portugal", portugal, "case_feature_id", PORTUGAL_AGGREGATION_COMPARATOR),
        (axes[1], "Bangladesh", bangladesh, "cross_polarity_cluster", BANGLADESH_AGGREGATION_COMPARATOR),
    ]
    for ax, case, frame, identifier, comparator in cases:
        a = frame.loc[frame["variant"].eq(PRIMARY_VARIANT), [identifier, "operational_rank"]].rename(columns={"operational_rank": "canonical_rank"})
        b = frame.loc[frame["variant"].eq(comparator), [identifier, "operational_rank"]].rename(columns={"operational_rank": "comparator_rank"})
        merged = a.merge(b, on=identifier, how="inner")
        ax.scatter(merged["comparator_rank"], merged["canonical_rank"], s=5, alpha=0.25, color="#315da8")
        limit = max(merged["comparator_rank"].max(), merged["canonical_rank"].max())
        ax.plot([1, limit], [1, limit], "--", color="0.5", lw=1)
        ax.set_xlabel(f"{comparator.replace('_', ' ')} rank")
        ax.set_ylabel("Canonical operational rank")
        ax.set_title(f"{'A' if case == 'Portugal' else 'B'}  {case}")

    ax = axes[2]
    subset = stability.loc[
        stability["ranking"].eq("operational")
        & stability["cutoff"].isin(["top_20pct", "top_100"])
        & ~stability["comparison_variant"].eq(PRIMARY_VARIANT)
    ].copy()
    subset["label"] = subset["case"] + "\n" + subset["comparison_variant"].str.replace("_", " ")
    ax.barh(np.arange(len(subset)), subset["overlap_fraction_of_primary"], color="#6b9f78")
    ax.set_yticks(np.arange(len(subset)), subset["label"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Top-set overlap with harmonized primary")
    ax.set_title("C  Ranking sensitivity")
    fig.savefig(p["figure_png"], dpi=300, bbox_inches="tight")
    fig.savefig(p["figure_pdf"], bbox_inches="tight")
    plt.close(fig)


def write_summary(
    stability: pd.DataFrame, recovery: pd.DataFrame, orth: pd.DataFrame,
    reproduction_error: float, p: dict[str, Path],
) -> None:
    lines = [
        "# Notebook 41: harmonized hazard-score and field-ranking sensitivity", "",
        f"Primary harmonized formula: `{PRIMARY_VARIANT}`. All endpoint components are Development-selected-model OOF-referenced hazard percentiles. Toxicity is the maximum of inverse EC50, LC50 and NOEC percentiles, and P/B/T are combined by geometric mean.", "",
        f"Portugal canonical-formula reproduction maximum absolute error: {reproduction_error:.3e}.", "",
        "## Rank stability relative to the harmonized primary score", "",
        "| Case | Comparison | Ranking | Cutoff | Spearman | Top-set overlap |", "|---|---|---|---|---:|---:|",
    ]
    selected = stability.loc[
        stability["comparison_variant"].isin([PORTUGAL_AGGREGATION_COMPARATOR, BANGLADESH_AGGREGATION_COMPARATOR, "novel_supported_b_lc50"])
        & stability["cutoff"].isin(["top_20pct", "top_100"])
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"| {row['case']} | {row['comparison_variant']} | {row['ranking']} | {row['cutoff']} | "
            f"{row['spearman_score']:.3f} | {row['overlap_fraction_of_primary']:.3f} |"
        )
    lines.extend(["", "## Portugal known-target continuity", ""])
    key = recovery.loc[
        recovery["variant"].eq(PRIMARY_VARIANT) & recovery["top_fraction"].eq(0.20)
    ]
    for _, row in key.iterrows():
        lines.append(
            f"- {row['ranking']}, {row['label_scope']}: {int(row['captured_unique_target_n'])}/"
            f"{int(row['recoverable_unique_target_n_in_universe'])} recoverable targets; enrichment {row['enrichment_fold']:.2f}; "
            f"p={row['hypergeometric_p_value']:.3g}."
        )
    lines.extend(["", "## Orthogonality", ""])
    for _, row in orth.loc[orth["variant"].eq(PRIMARY_VARIANT)].iterrows():
        lines.append(
            f"- {row['case']}: overlap={int(row['overlap_n'])}, hazard-only={int(row['hazard_only_n'])}, "
            f"occurrence/exposure-only={int(row['occurrence_only_n'])}, Jaccard={row['jaccard']:.3f}."
        )
    lines.extend([
        "", "## Claim boundary", "",
        "Known-CEC recovery is a continuity benchmark, not a hazard-positive truth set. The harmonized scores quantify how endpoint and aggregation choices change prioritization. They do not validate that unresolved high-ranking features are hazardous, and the B+LC50 sensitivity score must not be called a complete PBT score.",
    ])
    p["summary"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_qc(
    p: dict[str, Path], portugal: pd.DataFrame, bangladesh: pd.DataFrame,
    portugal_reproduction_error: float, bangladesh_reproduction_error: float,
    input_hashes: dict[str, str],
) -> pd.DataFrame:
    portugal_variant_counts = portugal.groupby("variant")["case_feature_id"].nunique()
    bangladesh_variant_counts = bangladesh.groupby("variant")["cross_polarity_cluster"].nunique()
    checks = [
        ("new_outputs_are_isolated", str(p["out"]).endswith(str(OUTPUT_SUBDIR)), str(p["out"])),
        ("portugal_variant_set_complete", set(portugal["variant"]) == set(VARIANTS), sorted(portugal["variant"].unique())),
        ("bangladesh_variant_set_complete", set(bangladesh["variant"]) == set(VARIANTS), sorted(bangladesh["variant"].unique())),
        ("portugal_canonical_formula_reproduced", portugal_reproduction_error < 1e-12, portugal_reproduction_error),
        ("bangladesh_canonical_formula_reproduced", bangladesh_reproduction_error < 1e-12, bangladesh_reproduction_error),
        ("portugal_frozen_common_universe", portugal_variant_counts.eq(16627).all(), portugal_variant_counts.to_dict()),
        ("portugal_common_scores_complete", portugal[["hazard_score", "operational_score"]].notna().all().all(), "all five variants are fully scored in the canonical common universe"),
        ("bangladesh_frozen_all_cluster_universe", bangladesh_variant_counts.eq(16768).all(), bangladesh_variant_counts.to_dict()),
        ("bangladesh_scores_complete", bangladesh[["hazard_score", "operational_score"]].notna().all().all(), "all five variants are fully scored for all canonical clusters"),
        ("harmonized_scores_in_unit_interval", portugal.loc[portugal["variant"].isin(VARIANTS), "hazard_score"].dropna().between(0, 1).all() and bangladesh.loc[bangladesh["variant"].isin(VARIANTS), "hazard_score"].dropna().between(0, 1).all(), "0 to 1"),
        ("no_duplicate_portugal_keys", not portugal.duplicated(["variant", "case_feature_id"]).any(), len(portugal)),
        ("no_duplicate_bangladesh_keys", not bangladesh.duplicated(["variant", "cross_polarity_cluster"]).any(), len(bangladesh)),
        ("upstream_inputs_unchanged", all(sha256_file(p[name]) == digest for name, digest in input_hashes.items()), len(input_hashes)),
    ]
    frame = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    atomic_csv(frame, p["qc"])
    if not frame["passed"].all():
        raise RuntimeError(f"Notebook 41 QC failed: {frame.loc[~frame['passed'], 'check'].tolist()}")
    return frame


def run_analysis(root: str | Path | None = None) -> dict[str, Any]:
    root_path = locate_root(root)
    p = paths(root_path)
    p["out"].mkdir(parents=True, exist_ok=True)
    p["figures"].mkdir(parents=True, exist_ok=True)
    validation = validate_inputs(root_path)
    plan = analysis_plan()
    if p["analysis_plan"].exists():
        existing_plan = _read_json(p["analysis_plan"])
        if existing_plan.get("version") == VERSION and existing_plan != plan:
            raise RuntimeError("Existing Notebook 41 analysis plan differs from the fixed plan in code")
    atomic_json(p["analysis_plan"], plan)
    input_names = validation["input"].tolist()
    input_hashes = {name: sha256_file(p[name]) for name in input_names}

    references = development_references(p["development_oof"])
    portugal, portugal_reproduction_error = prepare_portugal(p, references)
    bangladesh, bangladesh_reproduction_error = prepare_bangladesh(p, references)
    formulas = formula_table()
    stability = rank_stability(portugal, bangladesh)
    recovery = portugal_known_target_recovery(p, portugal)
    orth = orthogonality(portugal, bangladesh)
    migration = candidate_migration(portugal, bangladesh)

    atomic_csv(portugal, p["portugal_rankings"])
    atomic_csv(bangladesh, p["bangladesh_rankings"])
    atomic_csv(stability, p["rank_comparison"])
    atomic_csv(recovery, p["target_recovery"])
    atomic_csv(orth, p["orthogonality"])
    atomic_csv(migration, p["migration"])
    atomic_csv(formulas, p["formula_table"])
    make_figure(portugal, bangladesh, stability, p)
    write_summary(stability, recovery, orth, portugal_reproduction_error, p)
    qc = run_qc(
        p, portugal, bangladesh,
        portugal_reproduction_error, bangladesh_reproduction_error,
        input_hashes,
    )

    outputs = [
        p["analysis_plan"], p["portugal_rankings"], p["bangladesh_rankings"],
        p["rank_comparison"], p["target_recovery"], p["orthogonality"],
        p["migration"], p["formula_table"], p["qc"], p["summary"],
        p["figure_png"], p["figure_pdf"],
    ]
    manifest = {
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "post_freeze_presubmission_analysis": True,
        "changes_frozen_endpoint_predictions": False,
        "primary_harmonized_variant": PRIMARY_VARIANT,
        "field_unknown_hazard_truth_available": False,
        "input_hashes": input_hashes,
        "output_hashes": {path.name: sha256_file(path) for path in outputs},
        "software": {"python": sys.version, "python_executable": sys.executable},
    }
    atomic_json(p["manifest"], manifest)
    return {
        "root": root_path, "output_dir": p["out"], "manifest": manifest,
        "portugal": portugal, "bangladesh": bangladesh, "formulas": formulas,
        "rank_stability": stability, "target_recovery": recovery,
        "orthogonality": orth, "migration": migration, "qc": qc,
        "summary": p["summary"], "figure": p["figure_png"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(validate_inputs(args.root).to_string(index=False))
        return
    result = run_analysis(args.root)
    print(json.dumps({
        "status": result["manifest"]["status"],
        "output_dir": str(result["output_dir"]),
        "summary": str(result["summary"]),
        "figure": str(result["figure"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
