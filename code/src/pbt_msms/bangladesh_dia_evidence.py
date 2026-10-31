"""Final evidence adjudication for the uniformly annotated Bangladesh DIA cohort."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pbt_msms.bangladesh_dia_library import sha256_file
from pbt_msms.bangladesh_dia_sirius import paths as sirius_paths


VERSION = "PBT_MSMS_BANGLADESH_EVIDENCE_35_V2"
COHORT_ORDER = ["TOP_PRIORITY", "MID_PRIORITY", "LOW_PRIORITY", "HIGH_RISK_HIGH_UNCERTAINTY"]
EVIDENCE_ORDER = {
    "STRUCTURE_HIGHER_SUPPORT": 0,
    "STRUCTURE_MODERATE_SUPPORT": 1,
    "STRUCTURE_LOW_SUPPORT": 2,
    "FORMULA_ONLY": 3,
    "EXACT_MASS_ONLY": 4,
}
THRESHOLDS = {
    "higher_tanimoto_min": 0.50,
    "higher_distinct_csi_margin_min": 10.0,
    "higher_formula_score_normalized_min": 0.80,
    "moderate_tanimoto_min": 0.35,
    "moderate_distinct_csi_margin_min": 5.0,
    "moderate_formula_score_normalized_min": 0.50,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(frame: pd.DataFrame, path: Path, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    compression: Any = {"method": "gzip", "mtime": 0} if compressed else None
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def paths(root: Path) -> dict[str, Path]:
    p34 = sirius_paths(root)
    out = p34["out"].parent / "35_final_evidence_v1"
    return {
        **p34,
        "out34": p34["out"],
        "out": out,
        "feature_evidence": out / "35_feature_annotation_evidence.csv.gz",
        "cluster_evidence": out / "35_cluster_final_evidence_table.csv",
        "top_priority": out / "35_top_priority_verification_table.csv",
        "conflicts": out / "35_conflict_review_queue.csv",
        "cohort_audit": out / "35_cohort_final_audit.csv",
        "figure_cohort": out / "35_cohort_evidence_distribution.png",
        "figure_priority": out / "35_priority_annotation_map.png",
        "run_manifest35": out / "35_run_manifest.json",
    }


def _verify_stage34(p: dict[str, Path]) -> dict[str, Any]:
    manifest = json.loads(p["run_manifest34"].read_text(encoding="utf-8"))
    if manifest.get("status") != "SIRIUS_ANNOTATIONS_IMPORTED_AND_FROZEN":
        raise RuntimeError("Notebook 34 SIRIUS annotations are not frozen")
    for name, details in manifest.get("artifacts", {}).items():
        candidate = p["out34"] / name
        if not candidate.is_file() or sha256_file(candidate) != details["sha256"]:
            raise RuntimeError(f"Notebook 34 frozen artifact changed: {name}")
    export_manifest = json.loads(p["export_manifest"].read_text(encoding="utf-8"))
    expected_features = int(export_manifest["n_frozen_features"])
    expected_clusters = int(export_manifest["n_clusters"])
    if manifest.get("n_features") != expected_features or manifest.get("n_clusters") != expected_clusters:
        raise RuntimeError("Notebook 34 frozen dimensions changed")
    return manifest


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _number(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _structure_metrics(structures: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if structures.empty:
        return pd.DataFrame(columns=["mode", "spectrum_id"])
    required = {"mode", "spectrum_id", "structure_rank", "inchikey_2d", "csi_score"}
    if not required.issubset(structures.columns):
        raise RuntimeError(f"Notebook 34 structure table lacks: {sorted(required - set(structures.columns))}")
    if structures.duplicated(["mode", "spectrum_id", "structure_rank"]).any():
        raise RuntimeError("Notebook 34 structure ranks are not unique per feature")
    for (mode, spectrum_id), group in structures.groupby(["mode", "spectrum_id"], sort=False):
        group = group.sort_values(["structure_rank", "csi_score"], ascending=[True, False]).copy()
        top = group.iloc[0]
        top_key = _clean(top.get("inchikey_2d"))
        alternatives = group[group["inchikey_2d"].fillna("").astype(str).ne(top_key)]
        alternative = alternatives.iloc[0] if len(alternatives) else None
        top_score = _number(top.get("csi_score"))
        alternative_score = _number(alternative.get("csi_score")) if alternative is not None else np.nan
        margin = top_score - alternative_score if np.isfinite(top_score) and np.isfinite(alternative_score) else np.nan
        rows.append({
            "mode": str(mode),
            "spectrum_id": str(spectrum_id),
            "structure_candidate_n_returned": int(len(group)),
            "structure_connectivity_n_returned": int(group["inchikey_2d"].fillna("").replace("", np.nan).nunique()),
            "rank1_connectivity_supporting_candidate_n": int(group["inchikey_2d"].fillna("").astype(str).eq(top_key).sum()),
            "best_distinct_alternative_rank": int(alternative["structure_rank"]) if alternative is not None else np.nan,
            "best_distinct_alternative_inchikey_2d": _clean(alternative.get("inchikey_2d")) if alternative is not None else "",
            "best_distinct_alternative_csi_score": alternative_score,
            "rank1_distinct_csi_margin": margin,
        })
    return pd.DataFrame(rows)


def _feature_tier(row: pd.Series) -> str:
    if not bool(row["structure_available"]):
        return "FORMULA_ONLY" if bool(row["formula_available"]) else "EXACT_MASS_ONLY"
    formula_match = bool(row["structure_formula_matches_top_sirius_formula"])
    tanimoto = _number(row.get("tanimoto_similarity"))
    margin = _number(row.get("rank1_distinct_csi_margin"))
    formula_score = _number(row.get("sirius_score_normalized"))
    if (formula_match and tanimoto >= THRESHOLDS["higher_tanimoto_min"]
            and margin >= THRESHOLDS["higher_distinct_csi_margin_min"]
            and formula_score >= THRESHOLDS["higher_formula_score_normalized_min"]):
        return "STRUCTURE_HIGHER_SUPPORT"
    if (formula_match and tanimoto >= THRESHOLDS["moderate_tanimoto_min"]
            and margin >= THRESHOLDS["moderate_distinct_csi_margin_min"]
            and formula_score >= THRESHOLDS["moderate_formula_score_normalized_min"]):
        return "STRUCTURE_MODERATE_SUPPORT"
    return "STRUCTURE_LOW_SUPPORT"


def _caution_flags(row: pd.Series) -> str:
    flags: list[str] = []
    if bool(row["structure_available"]):
        if not bool(row["structure_formula_matches_top_sirius_formula"]):
            flags.append("structure_formula_differs_from_top_sirius_formula")
        if _number(row.get("tanimoto_similarity"), -np.inf) < THRESHOLDS["moderate_tanimoto_min"]:
            flags.append("low_fingerprint_similarity")
        margin = _number(row.get("rank1_distinct_csi_margin"))
        if not np.isfinite(margin):
            flags.append("no_distinct_runner_up_in_top10")
        elif margin < THRESHOLDS["moderate_distinct_csi_margin_min"]:
            flags.append("small_distinct_candidate_margin")
        if not _clean(row.get("structure_name")):
            flags.append("missing_structure_name")
    if bool(row["formula_available"]) and _number(row.get("sirius_score_normalized"), -np.inf) < THRESHOLDS["moderate_formula_score_normalized_min"]:
        flags.append("weak_formula_ranking_support")
    status = _clean(row.get("formula_status"))
    if status == "unsupported_multiply_charged":
        flags.append("sirius_unsupported_multiply_charged")
    elif not bool(row["formula_available"]):
        flags.append("no_formula_candidate")
    return ";".join(flags)


def _build_feature_evidence(features: pd.DataFrame, structures: pd.DataFrame) -> pd.DataFrame:
    if not features["spectrum_id"].is_unique:
        raise RuntimeError("Notebook 34 feature evidence must contain unique spectrum IDs")
    metrics = _structure_metrics(structures)
    result = features.merge(metrics, on=["mode", "spectrum_id"], how="left", validate="one_to_one")
    result["formula_available"] = result["molecular_formula"].fillna("").astype(str).ne("")
    result["structure_available"] = result["inchikey_2d"].fillna("").astype(str).ne("")
    result["structure_formula_matches_top_sirius_formula"] = (
        result["structure_available"]
        & result["candidate_formula"].fillna("").astype(str).eq(result["molecular_formula"].fillna("").astype(str))
    )
    result["structure_adduct_matches_top_sirius_adduct"] = (
        result["structure_available"]
        & result["candidate_adduct"].fillna("").astype(str).eq(result["formula_adduct"].fillna("").astype(str))
    )
    result["operational_evidence_tier"] = result.apply(_feature_tier, axis=1)
    result["operational_evidence_order"] = result["operational_evidence_tier"].map(EVIDENCE_ORDER).astype(int)
    result["identity_caution_flags"] = result.apply(_caution_flags, axis=1)
    result["identity_claim_ceiling"] = np.select(
        [result["structure_available"], result["formula_available"]],
        ["Level 3 tentative structure", "Level 4 formula only"],
        default="Level 5 exact mass only",
    )
    return result.sort_values(["cohort", "cluster_rank", "mode", "spectrum_id"]).reset_index(drop=True)


def _cross_mode_support(group: pd.DataFrame, unique_structure: str, conflict: bool) -> str:
    modes = set(group["mode"].astype(str))
    if conflict:
        return "cross_member_conflict"
    if modes != {"Pos", "Neg"}:
        return "single_mode_cluster"
    if unique_structure:
        supporting = set(group.loc[group["inchikey_2d"].fillna("").astype(str).eq(unique_structure), "mode"].astype(str))
        return "two_mode_same_structure" if supporting == {"Pos", "Neg"} else "single_mode_structure_support"
    return "two_mode_without_structure"


def _follow_up(row: dict[str, Any]) -> str:
    if row["identity_resolution_status"] == "UNRESOLVED_CROSS_MEMBER_CONFLICT":
        return "manual_cross_polarity_review_then_targeted_standard"
    tier = row["operational_evidence_tier"]
    if tier in {"STRUCTURE_HIGHER_SUPPORT", "STRUCTURE_MODERATE_SUPPORT"}:
        return "authentic_standard_rt_and_msms_confirmation"
    if tier == "STRUCTURE_LOW_SUPPORT":
        return "orthogonal_structure_review_then_authentic_standard"
    if tier == "FORMULA_ONLY":
        return "improve_msms_or_targeted_suspect_screening"
    return "reacquire_or_improve_msms_before_identity_claim"


def _build_cluster_evidence(features: pd.DataFrame, clusters34: pd.DataFrame) -> pd.DataFrame:
    risk_columns = [
        "cross_polarity_cluster", "cohort", "selection_order", "selection_reason", "cluster_rank",
        "rank_fraction",
        "priority_score_primary", "priority_score_hazard_heavy", "priority_score_exposure_heavy",
        "persistence_component", "bioaccumulation_component", "toxicity_component", "hazard_component",
        "occurrence_component", "abundance_component", "exposure_component", "confidence_component",
        "P_screen_not_ready_binary", "B_LOGBCF_REGRESSION", "T_EC50_MG_L_REGRESSION",
        "T_LC50_MG_L_REGRESSION", "T_NOEC_MG_L_REGRESSION", "sample_detection_fraction",
        "sample_group_both_fraction", "blank_corrected_max_height", "retained_endpoint_n",
        "all_five_endpoints_retained", "cluster_member_n", "modes", "member_spectrum_ids",
        "cluster_uncertainty_mean", "cluster_uncertainty_max", "library_match_status",
        "has_any_main_match", "is_nonambiguous_library_match",
    ]
    risk_columns = [column for column in risk_columns if column in clusters34.columns]
    risk = clusters34[risk_columns].copy()
    if len(risk) != 400 or not risk["cross_polarity_cluster"].is_unique:
        raise RuntimeError("Notebook 34 cluster table must contain 400 unique clusters")

    rows: list[dict[str, Any]] = []
    for cluster_id, group in features.groupby("cross_polarity_cluster", sort=False):
        structures = sorted(set(group["inchikey_2d"].dropna().astype(str)) - {""})
        formulas = sorted(set(group["molecular_formula"].dropna().astype(str)) - {""})
        conflict = len(structures) > 1 or (not structures and len(formulas) > 1)
        ordered = group.assign(
            _tan=pd.to_numeric(group["tanimoto_similarity"], errors="coerce").fillna(-np.inf),
            _margin=pd.to_numeric(group["rank1_distinct_csi_margin"], errors="coerce").fillna(-np.inf),
            _formula=pd.to_numeric(group["sirius_score_normalized"], errors="coerce").fillna(-np.inf),
        ).sort_values(
            ["operational_evidence_order", "_tan", "_margin", "_formula", "spectrum_id"],
            ascending=[True, False, False, False, True],
        )
        representative = ordered.iloc[0]
        unique_structure = structures[0] if len(structures) == 1 else ""
        unique_formula = formulas[0] if len(formulas) == 1 else ""
        if conflict:
            tier = "CONFLICTING_MEMBER_ANNOTATIONS"
            resolution = "UNRESOLVED_CROSS_MEMBER_CONFLICT"
            reported_name = reported_key = reported_smiles = reported_formula = ""
        elif unique_structure:
            tier = str(representative["operational_evidence_tier"])
            resolution = "TENTATIVE_STRUCTURE"
            reported_name = _clean(representative.get("structure_name"))
            reported_key = unique_structure
            reported_smiles = _clean(representative.get("smiles"))
            reported_formula = _clean(representative.get("candidate_formula"))
        elif unique_formula:
            tier = "FORMULA_ONLY"
            resolution = "FORMULA_ONLY"
            reported_name = reported_key = reported_smiles = ""
            reported_formula = unique_formula
        else:
            tier = "EXACT_MASS_ONLY"
            resolution = "EXACT_MASS_ONLY"
            reported_name = reported_key = reported_smiles = reported_formula = ""
        annotation_level = (
            "3_tentative_structure" if structures else
            "4_formula_only" if formulas else
            "5_exact_mass_only"
        )
        row = {
            "cross_polarity_cluster": str(cluster_id),
            "annotation_level": annotation_level,
            "identity_resolution_status": resolution,
            "operational_evidence_tier": tier,
            "reported_structure_name": reported_name,
            "reported_inchikey_2d": reported_key,
            "reported_smiles": reported_smiles,
            "reported_formula": reported_formula,
            "representative_spectrum_id": str(representative["spectrum_id"]),
            "representative_mode": str(representative["mode"]),
            "representative_structure_name": _clean(representative.get("structure_name")),
            "representative_inchikey_2d": _clean(representative.get("inchikey_2d")),
            "representative_candidate_formula": _clean(representative.get("candidate_formula")),
            "representative_top_sirius_formula": _clean(representative.get("molecular_formula")),
            "representative_tanimoto_similarity": _number(representative.get("tanimoto_similarity")),
            "representative_csi_score": _number(representative.get("csi_score")),
            "representative_distinct_csi_margin": _number(representative.get("rank1_distinct_csi_margin")),
            "representative_sirius_score_normalized": _number(representative.get("sirius_score_normalized")),
            "representative_structure_formula_matches_top_sirius_formula": bool(representative["structure_formula_matches_top_sirius_formula"]),
            "representative_identity_caution_flags": str(representative["identity_caution_flags"]),
            "structure_member_n": int(group["structure_available"].sum()),
            "structure_connectivity_unique_n": len(structures),
            "structure_candidates_across_members": ";".join(structures),
            "formula_member_n": int(group["formula_available"].sum()),
            "formula_unique_n": len(formulas),
            "formula_candidates_across_members": ";".join(formulas),
            "cross_member_annotation_conflict": bool(conflict),
            "cross_mode_identity_support": _cross_mode_support(group, unique_structure, conflict),
        }
        row["recommended_follow_up"] = _follow_up(row)
        rows.append(row)
    evidence = pd.DataFrame(rows)
    result = risk.merge(evidence, on="cross_polarity_cluster", how="left", validate="one_to_one")
    if len(result) != 400 or result["operational_evidence_tier"].isna().any():
        raise RuntimeError("Cluster evidence construction is incomplete")
    stage34_conflict = clusters34.set_index("cross_polarity_cluster")["cross_member_annotation_conflict"].astype(bool)
    observed_conflict = result.set_index("cross_polarity_cluster")["cross_member_annotation_conflict"].astype(bool)
    if not observed_conflict.reindex(stage34_conflict.index).equals(stage34_conflict):
        raise RuntimeError("Notebook 35 conflict status changed relative to frozen Notebook 34")
    return result.sort_values(["cluster_rank", "cross_polarity_cluster"]).reset_index(drop=True)


def _audit(clusters: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort in COHORT_ORDER:
        group = clusters[clusters["cohort"].astype(str).eq(cohort)]
        rows.append({
            "cohort": cohort,
            "n_clusters": len(group),
            "structure_higher_support": int(group["operational_evidence_tier"].eq("STRUCTURE_HIGHER_SUPPORT").sum()),
            "structure_moderate_support": int(group["operational_evidence_tier"].eq("STRUCTURE_MODERATE_SUPPORT").sum()),
            "structure_low_support": int(group["operational_evidence_tier"].eq("STRUCTURE_LOW_SUPPORT").sum()),
            "conflicting_member_annotations": int(group["operational_evidence_tier"].eq("CONFLICTING_MEMBER_ANNOTATIONS").sum()),
            "formula_only": int(group["operational_evidence_tier"].eq("FORMULA_ONLY").sum()),
            "exact_mass_only": int(group["operational_evidence_tier"].eq("EXACT_MASS_ONLY").sum()),
            "two_mode_same_structure": int(group["cross_mode_identity_support"].eq("two_mode_same_structure").sum()),
            "structure_formula_disagreement": int((
                group["representative_inchikey_2d"].fillna("").ne("")
                & ~group["representative_structure_formula_matches_top_sirius_formula"].astype(bool)
            ).sum()),
        })
    audit = pd.DataFrame(rows)
    count_columns = [column for column in audit.columns if column not in {"cohort", "n_clusters", "two_mode_same_structure", "structure_formula_disagreement"}]
    if not audit["n_clusters"].eq(100).all() or not audit[count_columns].sum(axis=1).eq(100).all():
        raise RuntimeError("Cohort evidence audit does not partition every frozen cohort")
    return audit


def _figures(clusters: pd.DataFrame, audit: pd.DataFrame, p: dict[str, Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = [
        "STRUCTURE_HIGHER_SUPPORT", "STRUCTURE_MODERATE_SUPPORT", "STRUCTURE_LOW_SUPPORT",
        "CONFLICTING_MEMBER_ANNOTATIONS", "FORMULA_ONLY", "EXACT_MASS_ONLY",
    ]
    colors = ["#166534", "#65a30d", "#f59e0b", "#dc2626", "#2563eb", "#94a3b8"]
    matrix = []
    for cohort in COHORT_ORDER:
        counts = clusters.loc[clusters["cohort"].astype(str).eq(cohort), "operational_evidence_tier"].value_counts()
        matrix.append([int(counts.get(category, 0)) for category in categories])
    fig, ax = plt.subplots(figsize=(11, 5.8))
    left = np.zeros(len(COHORT_ORDER))
    for index, category in enumerate(categories):
        values = np.asarray(matrix)[:, index]
        ax.barh(COHORT_ORDER, values, left=left, color=colors[index], label=category.replace("_", " ").title())
        left += values
    ax.set_xlabel("Frozen cross-polarity clusters")
    ax.set_xlim(0, 100)
    ax.set_title("Bangladesh DIA annotation evidence by frozen cohort")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(p["figure_cohort"], dpi=220, bbox_inches="tight")
    plt.close(fig)

    plot = clusters.copy()
    plot["cohort"] = pd.Categorical(plot["cohort"], COHORT_ORDER, ordered=True)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for category, color in zip(categories, colors):
        part = plot[plot["operational_evidence_tier"].eq(category)]
        ax.scatter(part["rank_fraction"] * 100.0, part["priority_score_primary"], s=26, alpha=0.72, color=color,
                   label=category.replace("_", " ").title(), edgecolors="none")
    ax.set_xlabel("Frozen model-only rank percentile (lower is higher priority)")
    ax.set_ylabel("Frozen exploratory priority score")
    ax.set_title("Identity evidence over the frozen PBT-priority ranking")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(p["figure_priority"], dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_evidence(root: Path) -> dict[str, Any]:
    """Create the frozen Notebook 35 evidence tables without making identity claims above Level 3."""
    p = paths(root)
    p["out"].mkdir(parents=True, exist_ok=True)
    manifest34 = _verify_stage34(p)
    features34 = pd.read_csv(p["feature_summary34"])
    structures34 = pd.read_csv(p["structure_candidates"])
    clusters34 = pd.read_csv(p["cluster_summary34"])
    features = _build_feature_evidence(features34, structures34)
    if len(features) != int(manifest34["n_features"]):
        raise RuntimeError(
            "Notebook 34 feature count does not match its frozen manifest: "
            f"{len(features)} != {manifest34['n_features']}"
        )
    clusters = _build_cluster_evidence(features, clusters34)
    audit = _audit(clusters)
    top_priority = clusters[clusters["cohort"].astype(str).eq("TOP_PRIORITY")].sort_values("cluster_rank").copy()
    conflicts = clusters[clusters["cross_member_annotation_conflict"].astype(bool)].sort_values("cluster_rank").copy()
    if len(top_priority) != 100 or len(conflicts) != int(manifest34["cross_member_conflict_clusters"]):
        raise RuntimeError("Notebook 35 shortlist or conflict queue is incomplete")

    write_csv(features, p["feature_evidence"], compressed=True)
    write_csv(clusters, p["cluster_evidence"])
    write_csv(top_priority, p["top_priority"])
    write_csv(conflicts, p["conflicts"])
    write_csv(audit, p["cohort_audit"])
    _figures(clusters, audit, p)

    table_artifacts = [p["feature_evidence"], p["cluster_evidence"], p["top_priority"], p["conflicts"], p["cohort_audit"]]
    figure_artifacts = [p["figure_cohort"], p["figure_priority"]]
    artifact_details: dict[str, dict[str, Any]] = {}
    for artifact in table_artifacts:
        artifact_details[artifact.name] = {"sha256": sha256_file(artifact), "rows": int(pd.read_csv(artifact).shape[0])}
    for artifact in figure_artifacts:
        artifact_details[artifact.name] = {"sha256": sha256_file(artifact), "bytes": artifact.stat().st_size}

    manifest = {
        "version": VERSION,
        "stage": 35,
        "status": "FINAL_EVIDENCE_TABLE_FROZEN",
        "generated_at_utc": utc_now(),
        "n_features": len(features),
        "n_clusters": len(clusters),
        "n_top_priority_clusters": len(top_priority),
        "n_conflict_clusters": len(conflicts),
        "annotation_scope": manifest34.get("annotation_scope", "all identity-blind frozen clusters"),
        "feature_formula_success": int(features["formula_available"].sum()),
        "feature_structure_success": int(features["structure_available"].sum()),
        "feature_structure_formula_agreement": int(features["structure_formula_matches_top_sirius_formula"].sum()),
        "cluster_evidence_tier_counts": {key: int(value) for key, value in clusters["operational_evidence_tier"].value_counts().items()},
        "operational_thresholds": THRESHOLDS,
        "threshold_interpretation": "Internal prioritization bands only; not calibrated identification probabilities or regulatory thresholds.",
        "identity_claim_ceiling": "CSI:FingerID candidates remain Level 3 tentative structures until authentic-standard RT and MS/MS confirmation.",
        "pbt_claim_limit": "Frozen model outputs are exploratory prioritization scores, not regulatory PBT determinations.",
        "input_hashes": {
            p["run_manifest34"].name: sha256_file(p["run_manifest34"]),
            p["feature_summary34"].name: sha256_file(p["feature_summary34"]),
            p["structure_candidates"].name: sha256_file(p["structure_candidates"]),
            p["cluster_summary34"].name: sha256_file(p["cluster_summary34"]),
        },
        "artifacts": artifact_details,
    }
    write_json(p["run_manifest35"], manifest)
    return manifest
