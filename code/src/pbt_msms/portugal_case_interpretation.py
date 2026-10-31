"""Portugal wastewater case interpretation after the frozen identity reveal.

This module consumes Notebook 24 and 25 artifacts without changing predictions,
model choices, ranking formulas, or identity calls.  It creates descriptive
site/campaign summaries, library-threshold sensitivity analyses, a ranked list
of library-unmatched high-hazard features, and manuscript-ready figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare


VERSION = "PBT_MSMS_PORTUGAL_CASE_INTERPRETATION_26_V1"
FINAL_STATUS = "PORTUGAL_CASE_COMPLETE_READY_FOR_MANUSCRIPT_INTEGRATION"
TOP_FRACTIONS = (0.05, 0.10, 0.20)
COSINE_THRESHOLDS = (0.60, 0.70, 0.80, 0.85)
PEAK_THRESHOLDS = (4, 6, 8)
PPM_THRESHOLDS = (10.0, 15.0, 20.0)
AMBIGUITY_DELTA = 0.05
SITE_ORDER = ("WWTP In", "WWTP Eff", "CP-VC", "PS-F", "PS-T", "PS-VC", "PVZ-1", "PVZ-2")


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
    case = root / "external_data" / "portugal_wastewater_5830725"
    out25 = case / "06_revealed_evaluation" / "25_identity_reveal"
    out = output_dir or case / "07_case_interpretation" / "26_portugal_case_interpretation"
    manifests = case / "manifests"
    manifest_out = out / "manifests" if output_dir else manifests
    return {
        "root": root,
        "case": case,
        "out25": out25,
        "out": out,
        "figures": out / "figures",
        "manifests": manifests,
        "manifest_out": manifest_out,
        "run25": manifests / "25_identity_reveal_run_manifest.json",
        "match25": manifests / "25_library_match_manifest.json",
        "freeze24": manifests / "24_identity_blind_freeze.json",
        "features24": case / "04_hazard_ranking_24" / "24_feature_predictions_identity_blind.csv.gz",
        "groups24": case / "04_hazard_ranking_24" / "24_case_feature_groups_identity_blind.csv.gz",
        "revealed25": out25 / "25_revealed_priority_table.csv.gz",
        "recovery25": out25 / "25_known_target_recovery.csv",
        "feature_matches25": out25 / "25_feature_library_matches.csv.gz",
        "group_matches25": out25 / "25_case_group_library_matches.csv.gz",
        "site_campaign": out / "26_site_campaign_summary.csv",
        "site_mode_campaign": out / "26_site_mode_campaign_summary.csv",
        "site_summary": out / "26_site_summary.csv",
        "campaign_summary": out / "26_campaign_summary.csv",
        "friedman": out / "26_campaign_friedman_tests.csv",
        "watchlist_site": out / "26_watchlist_site_detection.csv",
        "sensitivity": out / "26_library_threshold_sensitivity.csv",
        "capture": out / "26_rank_capture_summary.csv",
        "unmatched": out / "26_library_unmatched_high_priority_candidates.csv",
        "summary": out / "26_manuscript_result_summary.json",
        "qc": manifest_out / "26_case_interpretation_qc_checks.csv",
        "run_manifest": manifest_out / "26_case_interpretation_run_manifest.json",
    }


def software_versions() -> dict[str, str]:
    import scipy

    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").fillna("").str.lower().isin({"true", "1", "yes"})


def verify_inputs(p: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    run25 = json.loads(p["run25"].read_text(encoding="utf-8"))
    match25 = json.loads(p["match25"].read_text(encoding="utf-8"))
    freeze24 = json.loads(p["freeze24"].read_text(encoding="utf-8"))
    if run25.get("status") != "READY_FOR_NOTEBOOK_26_PORTUGAL_CASE_INTERPRETATION":
        raise AssertionError(run25.get("status"))
    if not run25.get("qc_all_required_pass"):
        raise AssertionError("Notebook 25 required QC did not pass")
    if match25.get("status") != "PASS":
        raise AssertionError("Notebook 25 library matching did not pass")
    if freeze24.get("status") != "IDENTITY_BLIND_RANKING_FROZEN":
        raise AssertionError("Notebook 24 ranking is not frozen")
    for name, expected in run25["output_hashes"].items():
        candidates = [p["out25"] / name, p["out25"] / "figures" / name, p["manifests"] / name]
        artifact = next((item for item in candidates if item.is_file()), None)
        if artifact is None or sha256_file(artifact) != expected:
            raise AssertionError(f"Notebook 25 artifact changed: {name}")
    return run25, match25, freeze24


def assign_common_ranks(revealed: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    low_only = (
        features.assign(_low=as_bool(features["low_ms2_depth_sample"]))
        .groupby("case_feature_id", as_index=False)
        .agg(low_depth_only_group=("_low", "all"))
    )
    frame = revealed.merge(low_only, on="case_feature_id", how="left", validate="one_to_one")
    frame["low_depth_only_group"] = frame["low_depth_only_group"].fillna(False).astype(bool)
    frame["known_published_cec_hit"] = as_bool(frame["known_published_cec_hit"])
    common = frame.loc[frame["median_primary_hazard_score"].notna() & ~frame["low_depth_only_group"]].copy()
    hazard_order = common.sort_values(
        ["median_primary_hazard_score", "case_feature_id"], ascending=[False, True]
    ).index
    occurrence_order = common.sort_values(
        ["occurrence_assisted_score", "case_feature_id"], ascending=[False, True]
    ).index
    common.loc[hazard_order, "common_hazard_rank"] = np.arange(1, len(common) + 1)
    common.loc[occurrence_order, "common_occurrence_rank"] = np.arange(1, len(common) + 1)
    common["common_hazard_rank_fraction"] = common["common_hazard_rank"] / len(common)
    common["common_occurrence_rank_fraction"] = common["common_occurrence_rank"] / len(common)
    for fraction in TOP_FRACTIONS:
        label = int(round(fraction * 100))
        cutoff = int(math.ceil(len(common) * fraction))
        common[f"top_{label}pct_hazard"] = common["common_hazard_rank"].le(cutoff)
        common[f"top_{label}pct_occurrence"] = common["common_occurrence_rank"].le(cutoff)
    return common


def threshold_sensitivity(pairs: pd.DataFrame, watchlist_ids: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ppm in PPM_THRESHOLDS:
        for cosine in COSINE_THRESHOLDS:
            for peaks in PEAK_THRESHOLDS:
                passing = pairs.loc[
                    pairs["precursor_ppm_error"].le(ppm)
                    & pairs["modified_cosine"].ge(cosine)
                    & pairs["matched_peaks"].ge(peaks)
                ].copy()
                group_target = (
                    passing.sort_values(
                        ["case_feature_id", "target_id", "modified_cosine", "matched_peaks", "precursor_ppm_error"],
                        ascending=[True, True, False, False, True],
                    )
                    .drop_duplicates(["case_feature_id", "target_id"])
                )
                winners: list[pd.Series] = []
                for _, group in group_target.groupby("case_feature_id", sort=False):
                    ordered = group.sort_values(
                        ["modified_cosine", "matched_peaks", "precursor_ppm_error", "target_id"],
                        ascending=[False, False, True, True],
                    )
                    best = ordered.iloc[0]
                    ambiguous = bool(
                        len(ordered) > 1
                        and float(best["modified_cosine"]) - float(ordered.iloc[1]["modified_cosine"])
                        <= AMBIGUITY_DELTA
                    )
                    if not ambiguous:
                        winners.append(best)
                winner_frame = pd.DataFrame(winners)
                rows.append({
                    "precursor_ppm_max": ppm,
                    "modified_cosine_min": cosine,
                    "matched_peaks_min": peaks,
                    "feature_target_pair_n": int(len(passing)),
                    "case_group_any_match_n": int(group_target["case_feature_id"].nunique()),
                    "case_group_nonambiguous_n": int(len(winner_frame)),
                    "unique_target_n": int(winner_frame["target_id"].nunique()) if len(winner_frame) else 0,
                    "watchlist_target_n": int(winner_frame.loc[
                        winner_frame["target_id"].isin(watchlist_ids), "target_id"
                    ].nunique()) if len(winner_frame) else 0,
                    "is_notebook25_primary_rule": bool(ppm == 20.0 and cosine == 0.70 and peaks == 6),
                })
    return pd.DataFrame(rows)


def summarize_detection_unit(unit: pd.DataFrame, all_unit: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "eligible_common_group_n": int(unit["case_feature_id"].nunique()),
        "median_primary_hazard_score": float(unit["median_primary_hazard_score"].median()),
    }
    for fraction in TOP_FRACTIONS:
        label = int(round(fraction * 100))
        top = unit.loc[unit[f"top_{label}pct_hazard"]]
        top_n = int(top["case_feature_id"].nunique())
        unmatched_n = int(top.loc[~top["known_published_cec_hit"], "case_feature_id"].nunique())
        result[f"global_top_{label}pct_hazard_group_n"] = top_n
        result[f"global_top_{label}pct_hazard_per_1000_eligible"] = (
            1000.0 * top_n / result["eligible_common_group_n"] if result["eligible_common_group_n"] else np.nan
        )
        result[f"global_top_{label}pct_unmatched_group_n"] = unmatched_n
    known = all_unit.loc[all_unit["known_published_cec_hit"]]
    watch = known.loc[known["published_watchlist_cec"]]
    result.update({
        "known_cec_group_n": int(known["case_feature_id"].nunique()),
        "known_cec_target_n": int(known["target_id"].nunique()),
        "watchlist_group_n": int(watch["case_feature_id"].nunique()),
        "watchlist_target_n": int(watch["target_id"].nunique()),
    })
    return result


def site_campaign_tables(
    features: pd.DataFrame, revealed: pd.DataFrame, common: pd.DataFrame, recovery: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_flags = recovery[["target_id", "published_watchlist_cec"]].drop_duplicates("target_id")
    target_flags["published_watchlist_cec"] = as_bool(target_flags["published_watchlist_cec"])
    annotation = common.copy().merge(target_flags, on="target_id", how="left")
    annotation["published_watchlist_cec"] = as_bool(annotation["published_watchlist_cec"])
    full_annotation = revealed.copy()
    full_annotation["known_published_cec_hit"] = as_bool(full_annotation["known_published_cec_hit"])
    full_annotation = full_annotation.merge(target_flags, on="target_id", how="left")
    full_annotation["published_watchlist_cec"] = as_bool(full_annotation["published_watchlist_cec"])

    detections = features[["sample_id", "site_code", "campaign", "mode", "case_feature_id"]].drop_duplicates()
    revealed_columns = [
        "case_feature_id", "target_id", "compound_name_published", "known_published_cec_hit",
        "median_primary_hazard_score", "common_hazard_rank_fraction", "common_occurrence_rank_fraction",
        "published_watchlist_cec",
    ] + [f"top_{int(round(f * 100))}pct_hazard" for f in TOP_FRACTIONS]
    common_detection = detections.merge(annotation[revealed_columns], on="case_feature_id", how="inner")

    all_annotation = full_annotation[[
        "case_feature_id", "target_id", "compound_name_published",
        "known_published_cec_hit", "published_watchlist_cec",
    ]].copy()
    all_detection = detections.merge(all_annotation, on="case_feature_id", how="inner")

    site_campaign_rows: list[dict[str, Any]] = []
    for (site, campaign), unit in common_detection.groupby(["site_code", "campaign"], sort=True):
        all_unit = all_detection.loc[
            all_detection["site_code"].eq(site) & all_detection["campaign"].eq(campaign)
        ]
        site_campaign_rows.append({"site_code": site, "campaign": int(campaign), **summarize_detection_unit(unit, all_unit)})
    site_campaign = pd.DataFrame(site_campaign_rows)

    mode_rows: list[dict[str, Any]] = []
    for (site, campaign, mode), unit in common_detection.groupby(["site_code", "campaign", "mode"], sort=True):
        all_unit = all_detection.loc[
            all_detection["site_code"].eq(site)
            & all_detection["campaign"].eq(campaign)
            & all_detection["mode"].eq(mode)
        ]
        mode_rows.append({
            "site_code": site, "campaign": int(campaign), "mode": mode,
            **summarize_detection_unit(unit, all_unit),
        })
    site_mode_campaign = pd.DataFrame(mode_rows)

    metrics = [
        "eligible_common_group_n", "median_primary_hazard_score",
        "global_top_5pct_hazard_group_n", "global_top_5pct_hazard_per_1000_eligible",
        "global_top_10pct_hazard_group_n", "global_top_20pct_hazard_group_n",
        "known_cec_target_n", "watchlist_target_n",
    ]
    site_summary = site_campaign.groupby("site_code", as_index=False)[metrics].agg(["mean", "std"])
    site_summary.columns = ["site_code"] + [f"{metric}_{stat}" for metric, stat in site_summary.columns[1:]]
    campaign_summary = site_campaign.groupby("campaign", as_index=False)[metrics].agg(["mean", "std"])
    campaign_summary.columns = ["campaign"] + [f"{metric}_{stat}" for metric, stat in campaign_summary.columns[1:]]

    friedman_rows: list[dict[str, Any]] = []
    for metric in [
        "global_top_5pct_hazard_per_1000_eligible", "median_primary_hazard_score",
        "known_cec_target_n", "watchlist_target_n",
    ]:
        pivot = site_campaign.pivot(index="site_code", columns="campaign", values=metric).dropna()
        columns = sorted(pivot.columns)
        if len(columns) == 3 and len(pivot) >= 3:
            statistic, p_value = friedmanchisquare(*(pivot[column].to_numpy() for column in columns))
            kendalls_w = statistic / (len(pivot) * (len(columns) - 1))
        else:
            statistic = p_value = kendalls_w = np.nan
        friedman_rows.append({
            "metric": metric, "paired_site_n": int(len(pivot)), "campaign_n": int(len(columns)),
            "friedman_chi_square": statistic, "p_value": p_value, "kendalls_w": kendalls_w,
            "interpretation_scope": "exploratory repeated-site campaign comparison; no causal claim",
        })
    friedman = pd.DataFrame(friedman_rows)

    watch_targets = recovery.loc[as_bool(recovery["published_watchlist_cec"]), [
        "target_id", "compound_name_published"
    ]].drop_duplicates()
    combinations = pd.MultiIndex.from_product(
        [SITE_ORDER, (1, 2, 3), watch_targets["target_id"]],
        names=["site_code", "campaign", "target_id"],
    ).to_frame(index=False)
    observed = (
        all_detection.loc[all_detection["published_watchlist_cec"]]
        .groupby(["site_code", "campaign", "target_id"], as_index=False)
        .agg(detected=("case_feature_id", "size"), matched_group_n=("case_feature_id", "nunique"))
    )
    watchlist_site = combinations.merge(observed, on=["site_code", "campaign", "target_id"], how="left")
    watchlist_site = watchlist_site.merge(watch_targets, on="target_id", how="left")
    watchlist_site["detected"] = watchlist_site["detected"].fillna(0).gt(0)
    watchlist_site["matched_group_n"] = watchlist_site["matched_group_n"].fillna(0).astype(int)
    return site_campaign, site_mode_campaign, site_summary, campaign_summary, friedman, watchlist_site


def rank_capture(common: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    methods = {
        "Frozen PBT hazard": "common_hazard_rank",
        "Occurrence-assisted hazard": "common_occurrence_rank",
    }
    for method, rank_column in methods.items():
        ordered = common.sort_values([rank_column, "case_feature_id"])
        for fraction in TOP_FRACTIONS:
            n_top = int(math.ceil(len(ordered) * fraction))
            top = ordered.head(n_top)
            rows.append({
                "method": method,
                "top_fraction": fraction,
                "top_group_n": n_top,
                "known_cec_group_n": int(top["known_published_cec_hit"].sum()),
                "known_cec_share": float(top["known_published_cec_hit"].mean()),
                "library_unmatched_group_n": int((~top["known_published_cec_hit"]).sum()),
                "library_unmatched_share": float((~top["known_published_cec_hit"]).mean()),
            })
    return pd.DataFrame(rows)


def unmatched_candidates(common: pd.DataFrame) -> pd.DataFrame:
    cutoff_5 = int(math.ceil(len(common) * 0.05))
    cutoff_10 = int(math.ceil(len(common) * 0.10))
    cutoff_20 = int(math.ceil(len(common) * 0.20))
    frame = common.loc[
        ~common["known_published_cec_hit"] & common["common_hazard_rank"].le(cutoff_20)
    ].copy()
    recurrent = frame["detected_sample_n"].ge(2) & (
        frame["detected_site_n"].ge(2) | frame["detected_campaign_n"].ge(2)
    )
    frame["evidence_tier"] = np.select(
        [
            frame["common_hazard_rank"].le(cutoff_5) & recurrent,
            frame["common_hazard_rank"].le(cutoff_5)
            | (frame["common_hazard_rank"].le(cutoff_10) & recurrent),
        ],
        ["Tier A: top 5% hazard and recurrent", "Tier B: top 10% and/or less recurrent"],
        default="Tier C: top 20% sensitivity list",
    )
    frame["identity_interpretation"] = "unmatched by the present published-CEC/public-library workflow; structure unknown"
    tier_order = {
        "Tier A: top 5% hazard and recurrent": 1,
        "Tier B: top 10% and/or less recurrent": 2,
        "Tier C: top 20% sensitivity list": 3,
    }
    frame["_tier"] = frame["evidence_tier"].map(tier_order)
    frame = frame.sort_values(
        ["_tier", "common_hazard_rank", "detected_sample_n", "case_feature_id"],
        ascending=[True, True, False, True],
    )
    frame["candidate_priority_rank"] = np.arange(1, len(frame) + 1)
    columns = [
        "candidate_priority_rank", "evidence_tier", "identity_interpretation", "case_feature_id", "mode",
        "median_precursor_mz", "median_rt_seconds", "median_primary_hazard_score",
        "common_hazard_rank", "common_hazard_rank_fraction", "occurrence_assisted_score",
        "common_occurrence_rank", "median_p_probability", "median_b_logbcf",
        "median_t_ec50_log10_mg_l", "median_t_lc50_log10_mg_l", "median_t_noec_log10_mg_l",
        "detected_sample_n", "detected_site_n", "detected_campaign_n", "median_sample_abundance_percentile",
    ]
    return frame[columns]


def annotated_heatmap(ax: plt.Axes, data: pd.DataFrame, fmt: str, cmap: str,
                      title: str, cbar_label: str, vmin: float | None = None,
                      vmax: float | None = None) -> None:
    image = ax.imshow(data.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(data.columns)), [str(item) for item in data.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(data.index)), [str(item) for item in data.index])
    for row in range(len(data.index)):
        for column in range(len(data.columns)):
            value = float(data.iloc[row, column])
            label = format(value, fmt)
            ax.text(column, row, label, ha="center", va="center", fontsize=7,
                    color="white" if value > np.nanmean(data.to_numpy(dtype=float)) else "black")
    ax.set_title(title, loc="left", fontweight="bold")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)


def make_figures(
    p: dict[str, Path], site_campaign: pd.DataFrame, watchlist_site: pd.DataFrame,
    sensitivity: pd.DataFrame, capture: pd.DataFrame, candidates: pd.DataFrame,
) -> list[Path]:
    p["figures"].mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5), constrained_layout=True)
    site_heat = site_campaign.pivot(
        index="site_code", columns="campaign", values="global_top_5pct_hazard_per_1000_eligible"
    ).reindex(SITE_ORDER)
    annotated_heatmap(
        axes[0, 0], site_heat, ".1f", "YlOrRd",
        "A  Globally top-5% PBT-hazard features", "groups per 1,000 eligible",
    )
    axes[0, 0].set_xlabel("Sampling campaign")
    axes[0, 0].set_ylabel("Site")

    watch_fraction = (
        watchlist_site.groupby(["site_code", "compound_name_published"])["detected"].mean()
        .unstack(fill_value=0).reindex(SITE_ORDER)
    )
    annotated_heatmap(
        axes[0, 1], watch_fraction, ".2f", "Blues",
        "B  Watch-List CEC detection consistency", "fraction of 3 campaigns", 0, 1,
    )
    axes[0, 1].set_xlabel("Article-defined Watch-List CEC")
    axes[0, 1].set_ylabel("Site")

    sens = sensitivity.loc[sensitivity["precursor_ppm_max"].eq(20.0)].pivot(
        index="matched_peaks_min", columns="modified_cosine_min", values="unique_target_n"
    )
    annotated_heatmap(
        axes[1, 0], sens, ".0f", "viridis",
        "C  Library-threshold sensitivity (20 ppm)", "unique recovered CEC targets",
    )
    axes[1, 0].set_xlabel("Minimum modified cosine")
    axes[1, 0].set_ylabel("Minimum matched peaks")
    primary_row = list(sens.index).index(6)
    primary_column = list(sens.columns).index(0.70)
    axes[1, 0].add_patch(plt.Rectangle(
        (primary_column - 0.5, primary_row - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2.5
    ))

    fractions = sorted(capture["top_fraction"].unique())
    x = np.arange(len(fractions))
    width = 0.36
    for offset, (method, color) in enumerate([
        ("Frozen PBT hazard", "#3b6fb6"), ("Occurrence-assisted hazard", "#e67e22")
    ]):
        values = capture.loc[capture["method"].eq(method)].sort_values("top_fraction")
        axes[1, 1].bar(
            x + (offset - 0.5) * width, 100 * values["known_cec_share"], width,
            label=method, color=color,
        )
    axes[1, 1].set_xticks(x, [f"Top {int(100 * item)}%" for item in fractions])
    axes[1, 1].set_ylabel("Published-CEC library matches (%)")
    axes[1, 1].set_title("D  Annotation share within prioritized groups", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].spines[["top", "right"]].set_visible(False)
    main_path = p["figures"] / "26_portugal_case_main_figure.png"
    main_pdf = p["figures"] / "26_portugal_case_main_figure.pdf"
    fig.savefig(main_path, dpi=300, bbox_inches="tight")
    fig.savefig(main_pdf, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 7.5), constrained_layout=True)
    sizes = 18 + 13 * np.sqrt(candidates["detected_sample_n"].clip(lower=1))
    markers = {"Pos": "o", "Neg": "^"}
    for mode, marker in markers.items():
        subset = candidates.loc[candidates["mode"].eq(mode)]
        sc = ax.scatter(
            subset["median_precursor_mz"], subset["median_rt_seconds"] / 60.0,
            c=subset["median_primary_hazard_score"], s=sizes.loc[subset.index],
            marker=marker, cmap="magma", vmin=candidates["median_primary_hazard_score"].min(),
            vmax=candidates["median_primary_hazard_score"].max(), alpha=0.72,
            linewidths=np.where(subset["evidence_tier"].str.startswith("Tier A"), 0.9, 0.0),
            edgecolors="cyan", label=mode,
        )
    ax.set_xlabel("Precursor m/z")
    ax.set_ylabel("Retention time (min)")
    ax.set_title(
        "Library-unmatched high-priority features\nCyan outlines: recurrent Tier A candidates",
        loc="left", fontweight="bold",
    )
    ax.legend(title="Ion mode", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.colorbar(sc, ax=ax, label="Frozen primary PBT-hazard score")
    candidate_path = p["figures"] / "26_unmatched_high_priority_feature_map.png"
    candidate_pdf = p["figures"] / "26_unmatched_high_priority_feature_map.pdf"
    fig.savefig(candidate_path, dpi=300, bbox_inches="tight")
    fig.savefig(candidate_pdf, bbox_inches="tight")
    plt.close(fig)
    return [main_path, main_pdf, candidate_path, candidate_pdf]


def add_qc(checks: list[dict[str, Any]], check: str, passed: bool, detail: Any) -> None:
    checks.append({"check": check, "passed": bool(passed), "detail": str(detail)})


def run(root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    p = paths(root, output_dir)
    p["out"].mkdir(parents=True, exist_ok=True)
    run25, match25, freeze24 = verify_inputs(p)

    features = pd.read_csv(p["features24"])
    revealed = pd.read_csv(p["revealed25"], low_memory=False)
    recovery = pd.read_csv(p["recovery25"], low_memory=False)
    pairs = pd.read_csv(p["feature_matches25"], low_memory=False)
    common = assign_common_ranks(revealed, features)
    watchlist_ids = set(recovery.loc[as_bool(recovery["published_watchlist_cec"]), "target_id"])

    sensitivity = threshold_sensitivity(pairs, watchlist_ids)
    capture = rank_capture(common)
    candidates = unmatched_candidates(common)
    site_campaign, site_mode_campaign, site_summary, campaign_summary, friedman, watchlist_site = (
        site_campaign_tables(features, revealed, common, recovery)
    )

    write_csv(site_campaign, p["site_campaign"])
    write_csv(site_mode_campaign, p["site_mode_campaign"])
    write_csv(site_summary, p["site_summary"])
    write_csv(campaign_summary, p["campaign_summary"])
    write_csv(friedman, p["friedman"])
    write_csv(watchlist_site, p["watchlist_site"])
    write_csv(sensitivity, p["sensitivity"])
    write_csv(capture, p["capture"])
    write_csv(candidates, p["unmatched"])
    figure_paths = make_figures(p, site_campaign, watchlist_site, sensitivity, capture, candidates)

    main_sensitivity = sensitivity.loc[sensitivity["is_notebook25_primary_rule"]].iloc[0]
    tier_counts = candidates["evidence_tier"].value_counts().to_dict()
    result_summary = {
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "analysis_scope": "descriptive Portugal site/campaign interpretation after frozen inference and identity reveal",
        "common_evaluation_group_n": int(len(common)),
        "site_n": int(site_campaign["site_code"].nunique()),
        "campaign_n": int(site_campaign["campaign"].nunique()),
        "site_campaign_unit_n": int(len(site_campaign)),
        "library_primary_rule": {
            "precursor_ppm_max": 20.0, "modified_cosine_min": 0.70, "matched_peaks_min": 6,
            "case_group_any_match_n": int(main_sensitivity["case_group_any_match_n"]),
            "case_group_nonambiguous_n": int(main_sensitivity["case_group_nonambiguous_n"]),
            "unique_target_n": int(main_sensitivity["unique_target_n"]),
            "watchlist_target_n": int(main_sensitivity["watchlist_target_n"]),
        },
        "unmatched_high_priority_candidate_n": int(len(candidates)),
        "unmatched_candidate_tier_counts": {str(key): int(value) for key, value in tier_counts.items()},
        "interpretation": [
            "Site/campaign results are descriptive and exploratory because each site-campaign-mode has one sample unit.",
            "Threshold sensitivity evaluates robustness of the fixed Notebook 25 match rule; it is not used to retune identities.",
            "Library-unmatched features are not proven novel compounds and require orthogonal identification work.",
            "The Watch-List set is a regulatory-relevance benchmark, not binary PBT ground truth.",
        ],
    }
    write_json(p["summary"], result_summary)

    checks: list[dict[str, Any]] = []
    add_qc(checks, "notebook25_frozen_and_unchanged", True, run25["status"])
    add_qc(checks, "notebook24_ranking_still_frozen", freeze24["identity_blind"] is True, freeze24["status"])
    add_qc(checks, "common_universe_matches_notebook25", len(common) == run25["counts"]["common_enrichment_universe_groups"], len(common))
    add_qc(checks, "site_campaign_design_complete", len(site_campaign) == 24, len(site_campaign))
    add_qc(checks, "site_mode_campaign_design_complete", len(site_mode_campaign) == 48, len(site_mode_campaign))
    add_qc(checks, "main_rule_group_count_reproduced", int(main_sensitivity["case_group_any_match_n"]) == match25["counts"]["case_groups_with_any_main_match"], int(main_sensitivity["case_group_any_match_n"]))
    add_qc(checks, "main_rule_nonambiguous_count_reproduced", int(main_sensitivity["case_group_nonambiguous_n"]) == match25["counts"]["case_groups_nonambiguous"], int(main_sensitivity["case_group_nonambiguous_n"]))
    add_qc(checks, "main_rule_target_count_reproduced", int(main_sensitivity["unique_target_n"]) == match25["counts"]["unique_published_targets_recovered"], int(main_sensitivity["unique_target_n"]))
    add_qc(checks, "watchlist_all_recovered_at_primary_rule", int(main_sensitivity["watchlist_target_n"]) == 8, int(main_sensitivity["watchlist_target_n"]))
    add_qc(checks, "unmatched_candidate_table_nonempty", len(candidates) > 0, len(candidates))
    add_qc(checks, "unmatched_candidates_have_no_known_hit", True, "Candidates were filtered on known_published_cec_hit == False")
    add_qc(checks, "identity_not_fed_back_into_model", True, "Notebook 26 is read-only with respect to frozen predictions and ranks")
    qc = pd.DataFrame(checks)
    write_csv(qc, p["qc"])
    all_pass = bool(qc["passed"].all())

    artifacts = [
        p["site_campaign"], p["site_mode_campaign"], p["site_summary"], p["campaign_summary"],
        p["friedman"], p["watchlist_site"], p["sensitivity"], p["capture"], p["unmatched"],
        p["summary"], p["qc"], *figure_paths,
    ]
    manifest = {
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "status": FINAL_STATUS if all_pass else "FAILED_QC",
        "qc_all_required_pass": all_pass,
        "scientific_scope": "Portugal wastewater site/campaign interpretation and robustness analyses",
        "counts": {
            "model_scored_features": int(run25["counts"]["model_scored_features"]),
            "common_evaluation_groups": int(len(common)),
            "sites": int(site_campaign["site_code"].nunique()),
            "campaigns": int(site_campaign["campaign"].nunique()),
            "site_campaign_units": int(len(site_campaign)),
            "published_cecs_recovered": int(run25["counts"]["recovered_unique_targets"]),
            "watchlist_cecs_recovered": int(run25["counts"]["recovered_watchlist_targets"]),
            "library_unmatched_top20pct_candidates": int(len(candidates)),
            "tier_a_unmatched_candidates": int(tier_counts.get("Tier A: top 5% hazard and recurrent", 0)),
        },
        "primary_library_rule_unchanged": True,
        "identity_feedback_into_model": False,
        "input_hashes": {
            p["run25"].name: sha256_file(p["run25"]),
            p["match25"].name: sha256_file(p["match25"]),
            p["freeze24"].name: sha256_file(p["freeze24"]),
            p["features24"].name: sha256_file(p["features24"]),
            p["revealed25"].name: sha256_file(p["revealed25"]),
            p["recovery25"].name: sha256_file(p["recovery25"]),
            p["feature_matches25"].name: sha256_file(p["feature_matches25"]),
        },
        "analysis_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "limitations": result_summary["interpretation"],
        "output_hashes": {path.name: sha256_file(path) for path in artifacts},
        "next_step": "Integrate the Portugal case into the manuscript; optionally proceed to the Bangladesh DIA domain-transfer case",
        "software": software_versions(),
    }
    write_json(p["run_manifest"], manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
