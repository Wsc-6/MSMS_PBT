"""Uniform SIRIUS export and import for all frozen Bangladesh DIA cohorts."""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pbt_msms.bangladesh_dia_library import paths as library_paths
from pbt_msms.bangladesh_dia_library import sha256_file


VERSION = "PBT_MSMS_BANGLADESH_SIRIUS_34_V2"


def utc_now() -> str:
    from datetime import datetime, timezone
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
    lp = library_paths(root)
    out = lp["out"].parent / "34_sirius_annotation_v1"
    return {
        **lp,
        "out33": lp["out"],
        "out": out,
        "positive_ms": out / "positive_ms",
        "negative_ms": out / "negative_ms",
        "input_manifest": out / "34_sirius_input_manifest.csv",
        "unsupported_manifest": out / "34_sirius_unsupported_multiply_charged_features.csv",
        "export_manifest": out / "34_export_manifest.json",
        "formula_annotations": out / "34_formula_annotations.csv.gz",
        "structure_candidates": out / "34_structure_candidates_top10.csv.gz",
        "feature_summary34": out / "34_feature_annotation_summary.csv.gz",
        "cluster_summary34": out / "34_cluster_annotation_summary.csv",
        "cohort_audit34": out / "34_cohort_annotation_audit.csv",
        "run_manifest34": out / "34_run_manifest.json",
        "instructions": out / "SIRIUS_运行说明.md",
        "legacy_recalculation_note": out / "34_RECALCULATION_REQUIRED_20261007.md",
    }


def _verify_stage33(p: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    run = json.loads(p["run_manifest"].read_text(encoding="utf-8"))
    prep = json.loads(p["prepare_manifest"].read_text(encoding="utf-8"))
    if run.get("status") != "PUBLIC_LIBRARY_MATCHING_COMPLETE":
        raise RuntimeError("Notebook 33 public-library matching is not complete")
    if prep.get("status") != "LIBRARY_CANDIDATES_PREPARED":
        raise RuntimeError("Notebook 33 preparation is not complete")
    for name, details in run["artifacts"].items():
        candidate = p["out33"] / name
        if not candidate.is_file() or sha256_file(candidate) != details["sha256"]:
            raise RuntimeError(f"Notebook 33 frozen artifact changed: {name}")
    for name, details in prep["artifacts"].items():
        candidate = p["out33"] / name
        if not candidate.is_file() or sha256_file(candidate) != details["sha256"]:
            raise RuntimeError(f"Notebook 33 prepared artifact changed: {name}")
    features = pd.read_csv(p["feature_summary"])
    clusters = pd.read_csv(p["cluster_summary"])
    expected_features = int(prep["counts"]["case_spectra"])
    if len(features) != expected_features or not features["spectrum_id"].is_unique:
        raise RuntimeError(f"Notebook 33 must contain {expected_features} unique feature spectra")
    expected_cluster_n = int(run["counts"]["frozen_clusters"])
    if len(clusters) != expected_cluster_n or not clusters["cross_polarity_cluster"].is_unique:
        raise RuntimeError(f"Notebook 33 must contain {expected_cluster_n} unique frozen clusters")
    expected_clusters = set(clusters["cross_polarity_cluster"].astype(str))
    selected = features[features["cross_polarity_cluster"].astype(str).isin(expected_clusters)].copy()
    if selected["cross_polarity_cluster"].nunique() != len(clusters):
        raise RuntimeError("Notebook 33 feature-to-cluster mapping is incomplete")
    return run, prep, selected, clusters


def export_inputs(root: Path) -> dict[str, Any]:
    """Export every frozen cohort spectrum for a uniform SIRIUS annotation pass."""
    p = paths(root)
    run33, prep33, selected, clusters = _verify_stage33(p)
    if p["legacy_recalculation_note"].is_file():
        p["legacy_recalculation_note"].unlink()
    p["positive_ms"].mkdir(parents=True, exist_ok=True)
    p["negative_ms"].mkdir(parents=True, exist_ok=True)
    for folder in (p["positive_ms"], p["negative_ms"]):
        for stale in folder.glob("*.ms"):
            stale.unlink()
    unsupported_mask = selected["adduct"].astype(str).str.contains(r"\][2-9][+-]$", regex=True)
    unsupported = selected.loc[unsupported_mask].copy()
    supported = selected.loc[~unsupported_mask].copy()
    unsupported["sirius_exclusion_reason"] = "SIRIUS 6 supports singly charged ions only"
    unsupported["annotation_level"] = "5_exact_mass_only"
    write_csv(unsupported, p["unsupported_manifest"])
    case_mz = np.load(p["case_mz"], mmap_mode="r")
    case_intensity = np.load(p["case_intensity"], mmap_mode="r")
    case_offsets = np.load(p["case_offsets"], mmap_mode="r")
    rows: list[dict[str, Any]] = []
    cluster_reason = clusters.set_index("cross_polarity_cluster")["library_match_status"].map(
        lambda status: f"uniform SIRIUS annotation after public-library status: {status}"
    )
    for feature in supported.sort_values(["cohort", "selection_order", "mode", "spectrum_id"]).itertuples(index=False):
        case_row = int(feature.case_row)
        start, end = case_offsets[case_row:case_row + 2]
        mz = np.asarray(case_mz[start:end], dtype=float)
        intensity = np.asarray(case_intensity[start:end], dtype=float)
        folder = p["positive_ms"] if feature.mode == "Pos" else p["negative_ms"]
        target = folder / f"{feature.spectrum_id}.ms"
        lines = [
            f">compound {feature.spectrum_id}",
            f">feature_id {feature.spectrum_id}",
            f">parentmass {float(feature.precursor_mz):.8f}",
            f">ionization {feature.adduct}",
            f">rt {float(feature.rt_min) * 60.0:.4f}s",
            "",
            ">ms2",
        ]
        lines.extend(f"{mass:.8f} {abundance:.8f}" for mass, abundance in zip(mz, intensity))
        target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        rows.append({
            "spectrum_id": str(feature.spectrum_id),
            "case_row": case_row,
            "alignment_id": int(feature.alignment_id),
            "mode": str(feature.mode),
            "adduct": str(feature.adduct),
            "precursor_mz": float(feature.precursor_mz),
            "rt_min": float(feature.rt_min),
            "peak_n": len(mz),
            "cross_polarity_cluster": str(feature.cross_polarity_cluster),
            "cohort": str(feature.cohort),
            "cluster_rank": int(feature.cluster_rank),
            "priority_score_primary": float(feature.priority_score_primary),
            "sirius_annotation_reason": str(cluster_reason.at[str(feature.cross_polarity_cluster)]),
            "ms_file": str(target.relative_to(root)),
            "ms_sha256": sha256_file(target),
        })
    export = pd.DataFrame(rows).sort_values(["mode", "cohort", "cluster_rank", "spectrum_id"]).reset_index(drop=True)
    write_csv(export, p["input_manifest"])
    instructions = f"""# SIRIUS 6 运行说明（Notebook 34）

本批次统一覆盖 Notebook 32 冻结的全部 400 个候选簇，不按 Notebook 33 谱库命中状态选择性送入 SIRIUS。公开实测谱库证据仍单独保留。原始仪器为 Thermo Scientific Q Exactive HF-X。

## 0. 启动前

1. 使用 SIRIUS 6（优先使用与历史分析一致的 6.5.4）并登录可运行 CSI:FingerID 的账号。
2. 确认账号的 feature 限额能够覆盖 {len(export)} 个输入。
3. 不打开或复用任何重跑前的 Bangladesh SIRIUS 项目；项目内容必须与本轮导出的 spectrum_id 完全一致。

## 1. Positive 项目

1. 新建独立项目 `Bangladesh_DIA_34_Pos`。
2. 点击 Import，导入 `positive_ms` 目录中的全部 {int((export['mode'] == 'Pos').sum())} 个 `.ms` 文件。
3. 导入完成后确认 feature 数为 {int((export['mode'] == 'Pos').sum())}，且离子模式为正离子。
4. 点击 Compute All，载入 Default preset。
5. Instrument/Profile 选择 Orbitrap，Precursor mass deviation 设为 10 ppm。
6. Molecular Formula Annotation 选择 De novo + bottom-up。
7. 启用 SIRIUS Molecular Formula Annotation、CSI:FingerID Fingerprint Prediction/CANOPUS、CSI:FingerID Structure Database Search。
8. 关闭 Spectral Library Search、ZODIAC 和 MSNovelist；其余数据库与 element filter 保持 Default preset。
9. 点击 Show JSON 保存参数记录，再把该配置保存为 preset `Bangladesh_DIA_34_CanonicalHazard`。
10. 点击 Compute，等待所有 {int((export['mode'] == 'Pos').sum())} 个 feature 结束，包括失败记录。

## 2. Negative 项目

1. 新建独立项目 `Bangladesh_DIA_34_Neg`。
2. 点击 Import，导入 `negative_ms` 目录中的全部 {int((export['mode'] == 'Neg').sum())} 个 `.ms` 文件。
3. 确认 feature 数为 {int((export['mode'] == 'Neg').sum())}，且离子模式为负离子。
4. 点击 Compute All，直接加载 preset `Bangladesh_DIA_34_CanonicalHazard`，不得改动参数。
5. 点击 Compute，等待所有 {int((export['mode'] == 'Neg').sum())} 个 feature 结束，包括失败记录。

## 3. 计算完成后的检查

1. Positive 显示 {int((export['mode'] == 'Pos').sum())}/{int((export['mode'] == 'Pos').sum())} 已结束，Negative 显示 {int((export['mode'] == 'Neg').sum())}/{int((export['mode'] == 'Neg').sum())} 已结束；失败也算结束，但不能删除。
2. 随机检查 Formulas 和 Structures 页面能够打开；没有候选的 feature 保持原样。
3. 保持 SIRIUS 以及两个项目打开，再运行 Notebook 34 的自动导入单元格。导入器会按全部 spectrum_id 自动匹配项目。

## 固定要求

- 不手工输入已知分子式、名称或结构。
- 不根据 PBT 排名调整单个候选的参数。
- 不删除计算失败的 feature；失败也是结果。
- {len(unsupported)} 条多电荷谱图不会送入 SIRIUS，因为 SIRIUS 6 仅支持单电荷离子；它们已单独登记，不得改写成单电荷。
- 正负项目必须使用同一个已保存 preset。
- CSI:FingerID 候选最多记为 Level 3；仅分子式最多记为 Level 4；不得记为 Level 1。
"""
    p["instructions"].write_text(instructions, encoding="utf-8")
    manifest = {
        "version": VERSION,
        "stage": "export",
        "status": "READY_FOR_SIRIUS_GUI_COMPUTATION",
        "generated_at_utc": utc_now(),
        "stage33_run_manifest_sha256": sha256_file(p["run_manifest"]),
        "stage33_prepare_manifest_sha256": sha256_file(p["prepare_manifest"]),
        "n_clusters": int(selected["cross_polarity_cluster"].nunique()),
        "n_frozen_features": len(selected),
        "n_sirius_features": len(export),
        "n_unsupported_multiply_charged_features": len(unsupported),
        "mode_counts": {key: int(value) for key, value in export["mode"].value_counts().items()},
        "cohort_cluster_counts": {key: int(value) for key, value in clusters["cohort"].value_counts().items()},
        "annotation_scope": "all 400 identity-blind frozen clusters; independent of public-library match status",
        "identity_blinding": "No formula, name, SMILES, or InChIKey supplied to SIRIUS",
        "input_manifest_sha256": sha256_file(p["input_manifest"]),
        "unsupported_manifest_sha256": sha256_file(p["unsupported_manifest"]),
        "instructions_sha256": sha256_file(p["instructions"]),
    }
    write_json(p["export_manifest"], manifest)
    return manifest


def _request(
    base_url: str,
    path: str,
    params: dict[str, Any] | None = None,
    timeout: int = 180,
    method: str = "GET",
):
    query = urllib.parse.urlencode(params or {}, doseq=True)
    url = base_url + path + (("?" + query) if query else "")
    request = urllib.request.Request(url, method=method, headers={"Accept": "application/json, text/plain"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8-sig")
    return json.loads(raw) if raw.strip() else None


def _first(mapping: dict[str, Any], *keys: str):
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _project_features(base_url: str, project_id: str) -> pd.DataFrame:
    key = urllib.parse.quote(project_id, safe="")
    payload = _request(base_url, f"/api/projects/{key}/aligned-features", {"includeMergedMs2": "false"}) or []
    frame = pd.DataFrame(payload)
    if frame.empty:
        return frame
    external = frame.get("externalFeatureId", pd.Series(index=frame.index, dtype=object))
    names = frame.get("name", pd.Series(index=frame.index, dtype=object))
    frame["spectrum_id"] = external.fillna(names).astype(str)
    return frame


def _open_saved_projects(base_url: str, p: dict[str, Path]) -> tuple[set[str], list[Path]]:
    """Open separately saved SIRIUS projects in the current REST session.

    The GUI may expose only its current project after a restart even when both
    positive- and negative-mode project files were saved correctly.  Opening
    every ``.sirius`` file in the frozen stage directory makes the import
    independent of GUI tab/window state.  Feature IDs are still used below to
    determine which project represents each ion mode.
    """
    project_files = sorted(
        (path for path in p["out"].glob("*.sirius") if path.is_file()),
        key=lambda path: path.name.lower(),
    )
    if not project_files:
        return set(), []
    opened = _request(base_url, "/api/projects") or []
    opened_ids = {str(_first(project, "projectId", "id")) for project in opened}
    saved_ids: set[str] = set()
    for project_path in project_files:
        slug = "".join(
            character if character.isascii() and character.isalnum() else "_"
            for character in project_path.stem
        ).strip("_")
        project_id = f"bangladesh_dia_34_saved_{slug}"
        if project_id not in opened_ids:
            _request(
                base_url,
                f"/api/projects/{urllib.parse.quote(project_id, safe='')}",
                {"pathToProject": str(project_path.resolve())},
                timeout=300,
                method="PUT",
            )
            opened_ids.add(project_id)
        saved_ids.add(project_id)
    return saved_ids, project_files


def _match_projects(
    base_url: str,
    inputs: pd.DataFrame,
    preferred_project_ids: set[str] | None = None,
) -> tuple[dict[str, tuple[str, pd.DataFrame]], list[dict[str, Any]]]:
    projects = _request(base_url, "/api/projects") or []
    preferred = preferred_project_ids or set()
    expected = {mode: set(inputs.loc[inputs["mode"].eq(mode), "spectrum_id"].astype(str)) for mode in ("Pos", "Neg")}
    candidates: dict[str, list[tuple[int, int, str, pd.DataFrame]]] = {"Pos": [], "Neg": []}
    for project in projects:
        project_id = str(_first(project, "projectId", "id"))
        features = _project_features(base_url, project_id)
        if features.empty:
            continue
        observed = set(features["spectrum_id"])
        for mode, wanted in expected.items():
            if wanted and wanted.issubset(observed):
                candidates[mode].append((
                    0 if project_id in preferred else 1,
                    len(observed) - len(wanted),
                    project_id,
                    features[features["spectrum_id"].isin(wanted)].copy(),
                ))
    matched: dict[str, tuple[str, pd.DataFrame]] = {}
    for mode in ("Pos", "Neg"):
        if not candidates[mode]:
            raise RuntimeError(f"没有找到包含全部 {mode} feature_id 的 SIRIUS 项目")
        candidates[mode].sort(key=lambda item: (item[0], item[1], item[2]))
        if len(candidates[mode]) > 1 and candidates[mode][0][:2] == candidates[mode][1][:2]:
            raise RuntimeError(f"存在多个同样匹配的 {mode} 项目，请关闭或删除重复项目后重试")
        _, _, project_id, features = candidates[mode][0]
        matched[mode] = (project_id, features)
    return matched, projects


def import_results(root: Path, top_structure_n: int = 10) -> dict[str, Any]:
    """Import completed formula and CSI:FingerID results from the running SIRIUS GUI."""
    p = paths(root)
    run33, prep33, selected, clusters33 = _verify_stage33(p)
    export_manifest = json.loads(p["export_manifest"].read_text(encoding="utf-8"))
    if export_manifest.get("status") != "READY_FOR_SIRIUS_GUI_COMPUTATION":
        raise RuntimeError("Notebook 34 export stage is not complete")
    if sha256_file(p["input_manifest"]) != export_manifest["input_manifest_sha256"]:
        raise RuntimeError("SIRIUS input manifest changed after export")
    if sha256_file(p["unsupported_manifest"]) != export_manifest["unsupported_manifest_sha256"]:
        raise RuntimeError("SIRIUS unsupported-feature manifest changed after export")
    inputs = pd.read_csv(p["input_manifest"])
    unsupported = pd.read_csv(p["unsupported_manifest"])
    port_file = Path.home() / ".sirius" / "sirius-6.port"
    if not port_file.is_file():
        raise RuntimeError("SIRIUS 6 未运行。请启动 SIRIUS、打开并完成正负两个项目后重新运行本单元格。")
    base_url = f"http://127.0.0.1:{int(port_file.read_text(encoding='utf-8').strip())}"
    info = _request(base_url, "/api/info", {"includeConsumables": "false", "includeVersions": "true"}) or {}
    saved_project_ids, saved_project_files = _open_saved_projects(base_url, p)
    matched, projects = _match_projects(base_url, inputs, preferred_project_ids=saved_project_ids)
    formula_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    for mode, (project_id, features) in matched.items():
        project_key = urllib.parse.quote(project_id, safe="")
        for feature in features.itertuples(index=False):
            feature_id = str(getattr(feature, "alignedFeatureId"))
            feature_key = urllib.parse.quote(feature_id, safe="")
            base = f"/api/projects/{project_key}/aligned-features/{feature_key}"
            try:
                formulas = _request(base_url, base + "/formulas") or []
            except Exception:
                formulas = []
            formulas = sorted(formulas, key=lambda x: (int(x.get("rank") or 10**9), -float(x.get("siriusScore") or -1e30)))
            best_formula = formulas[0] if formulas else {}
            formula_rows.append({
                "mode": mode,
                "spectrum_id": str(feature.spectrum_id),
                "sirius_project_id": project_id,
                "aligned_feature_id": feature_id,
                "formula_status": "success" if best_formula else "no_formula_candidate",
                "molecular_formula": _first(best_formula, "molecularFormula", "formula"),
                "formula_adduct": _first(best_formula, "adduct", "ionType"),
                "formula_rank": best_formula.get("rank"),
                "sirius_score": best_formula.get("siriusScore"),
                "sirius_score_normalized": best_formula.get("siriusScoreNormalized"),
                "formula_id": best_formula.get("formulaId"),
                "formula_candidate_n": len(formulas),
            })
            try:
                structures = _request(base_url, base + "/db-structures") or []
            except Exception:
                structures = []
            structures = sorted(structures, key=lambda x: -float(x.get("csiScore") or -1e30))[:top_structure_n]
            for rank, candidate in enumerate(structures, 1):
                full_key = str(_first(candidate, "inchiKey", "inchikey") or "")
                structure_rows.append({
                    "mode": mode,
                    "spectrum_id": str(feature.spectrum_id),
                    "structure_rank": rank,
                    "structure_name": _first(candidate, "structureName", "name"),
                    "inchikey_full": full_key,
                    "inchikey_2d": full_key.split("-")[0] if full_key else "",
                    "smiles": candidate.get("smiles"),
                    "candidate_formula": _first(candidate, "molecularFormula", "formula"),
                    "candidate_adduct": _first(candidate, "adduct", "ionType"),
                    "csi_score": candidate.get("csiScore"),
                    "tanimoto_similarity": candidate.get("tanimotoSimilarity"),
                    "xlogp": _first(candidate, "xlogP", "xlogp"),
                })
    formulas = inputs.merge(pd.DataFrame(formula_rows), on=["mode", "spectrum_id"], how="left", validate="one_to_one")
    unsupported_base = unsupported.copy()
    if "peak_n" not in unsupported_base:
        unsupported_base["peak_n"] = unsupported_base["case_peak_n"]
    for column in inputs.columns:
        if column not in unsupported_base:
            unsupported_base[column] = np.nan
    unsupported_formula = unsupported_base[inputs.columns].copy()
    unsupported_formula["sirius_project_id"] = ""
    unsupported_formula["aligned_feature_id"] = ""
    unsupported_formula["formula_status"] = "unsupported_multiply_charged"
    unsupported_formula["molecular_formula"] = ""
    unsupported_formula["formula_adduct"] = ""
    unsupported_formula["formula_rank"] = np.nan
    unsupported_formula["sirius_score"] = np.nan
    unsupported_formula["sirius_score_normalized"] = np.nan
    unsupported_formula["formula_id"] = ""
    unsupported_formula["formula_candidate_n"] = 0
    formulas = pd.concat([formulas, unsupported_formula[formulas.columns]], ignore_index=True)
    if len(formulas) != export_manifest["n_frozen_features"] or not formulas["spectrum_id"].is_unique:
        raise RuntimeError("Supported plus unsupported SIRIUS feature accounting is incomplete")
    structure_columns = ["mode", "spectrum_id", "structure_rank", "structure_name", "inchikey_full", "inchikey_2d",
                         "smiles", "candidate_formula", "candidate_adduct", "csi_score", "tanimoto_similarity", "xlogp"]
    structures = pd.DataFrame(structure_rows, columns=structure_columns)
    top_structure = structures[structures["structure_rank"].eq(1)].drop(columns="structure_rank") if len(structures) else pd.DataFrame(columns=[c for c in structure_columns if c != "structure_rank"])
    feature_summary = formulas.merge(top_structure, on=["mode", "spectrum_id"], how="left", validate="one_to_one")
    feature_summary["annotation_level"] = np.select(
        [feature_summary["inchikey_2d"].fillna("").ne(""), feature_summary["molecular_formula"].fillna("").ne("")],
        ["3_tentative_structure", "4_formula_only"],
        default="5_exact_mass_only",
    )
    feature_summary["annotation_level_numeric"] = feature_summary["annotation_level"].str[0].astype(int)
    cluster_rows: list[dict[str, Any]] = []
    for cluster_id, group in feature_summary.groupby("cross_polarity_cluster", sort=False):
        formulas_observed = sorted(set(group["molecular_formula"].dropna().astype(str)) - {""})
        structures_observed = sorted(set(group["inchikey_2d"].dropna().astype(str)) - {""})
        ordered = group.assign(
            _csi=pd.to_numeric(group["csi_score"], errors="coerce").fillna(-np.inf),
            _sirius=pd.to_numeric(group["sirius_score"], errors="coerce").fillna(-np.inf),
        ).sort_values(["annotation_level_numeric", "_csi", "_sirius", "spectrum_id"], ascending=[True, False, False, True])
        best = ordered.iloc[0].drop(labels=["_csi", "_sirius"]).to_dict()
        conflict = len(structures_observed) > 1 or (not structures_observed and len(formulas_observed) > 1)
        best.update({
            "cross_polarity_cluster": cluster_id,
            "annotated_member_n": int(group["annotation_level_numeric"].le(4).sum()),
            "formula_member_n": int(group["molecular_formula"].fillna("").ne("").sum()),
            "formula_unique_n": len(formulas_observed),
            "formula_candidates_across_members": ";".join(formulas_observed),
            "structure_member_n": int(group["inchikey_2d"].fillna("").ne("").sum()),
            "structure_connectivity_unique_n": len(structures_observed),
            "structure_candidates_across_members": ";".join(structures_observed),
            "cross_member_annotation_conflict": conflict,
            "cluster_annotation_status": "conflicting_member_annotations" if conflict else str(best["annotation_level"]),
        })
        cluster_rows.append(best)
    cluster_summary = pd.DataFrame(cluster_rows)
    cluster_summary = clusters33.drop(columns=[column for column in clusters33.columns if column in cluster_summary.columns and column != "cross_polarity_cluster"]).merge(
        cluster_summary, on="cross_polarity_cluster", how="left", validate="one_to_one"
    )
    audit = cluster_summary.groupby("cohort", sort=False).agg(
        n_clusters=("cross_polarity_cluster", "size"),
        tentative_structure=("annotation_level", lambda x: int(x.eq("3_tentative_structure").sum())),
        formula_only=("annotation_level", lambda x: int(x.eq("4_formula_only").sum())),
        exact_mass_only=("annotation_level", lambda x: int(x.eq("5_exact_mass_only").sum())),
        cross_member_conflict=("cross_member_annotation_conflict", "sum"),
    ).reset_index()
    write_csv(formulas, p["formula_annotations"], compressed=True)
    write_csv(structures, p["structure_candidates"], compressed=True)
    write_csv(feature_summary, p["feature_summary34"], compressed=True)
    write_csv(cluster_summary, p["cluster_summary34"])
    write_csv(audit, p["cohort_audit34"])
    artifacts = [p["formula_annotations"], p["structure_candidates"], p["feature_summary34"], p["cluster_summary34"], p["cohort_audit34"]]
    manifest = {
        "version": VERSION,
        "stage": 34,
        "status": "SIRIUS_ANNOTATIONS_IMPORTED_AND_FROZEN",
        "generated_at_utc": utc_now(),
        "sirius_version": _first(info, "siriusVersion", "version"),
        "project_ids": {mode: project_id for mode, (project_id, _) in matched.items()},
        "n_features": len(feature_summary),
        "n_sirius_supported_features": len(inputs),
        "n_unsupported_multiply_charged_features": len(unsupported),
        "n_clusters": len(cluster_summary),
        "annotation_scope": export_manifest["annotation_scope"],
        "computation_configuration": {
            "instrument_profile": "Orbitrap",
            "ms2_mass_accuracy_ppm": 10,
            "formula_strategy": "De novo + bottom up",
            "spectral_library_search": False,
            "zodiac": False,
            "fingerprint_and_canopus": True,
            "structure_database_search": True,
            "msnovelist": False,
            "same_preset_for_both_modes": "Bangladesh_DIA_34_CanonicalHazard",
        },
        "formula_success": int(feature_summary["molecular_formula"].fillna("").ne("").sum()),
        "structure_success": int(feature_summary["inchikey_2d"].fillna("").ne("").sum()),
        "cluster_annotation_level_counts": {key: int(value) for key, value in cluster_summary["annotation_level"].value_counts().items()},
        "cross_member_conflict_clusters": int(cluster_summary["cross_member_annotation_conflict"].sum()),
        "saved_project_files": {
            project_path.name: {
                "size_bytes": int(project_path.stat().st_size),
                "modified_time_ns": int(project_path.stat().st_mtime_ns),
                "hash_note": "SHA-256 omitted because SIRIUS locks open project databases on Windows",
            }
            for project_path in saved_project_files
        },
        "claim_limits": [
            "CSI:FingerID rank-1 structures are Level-3 tentative candidates, not confirmed identities",
            "Formula-only results are Level-4 annotations",
            "No Level-1 claim without authentic-standard retention-time and MS/MS confirmation",
            "SIRIUS annotations do not validate the numerical PBT predictions",
            "Public-library status did not determine inclusion in the uniform SIRIUS annotation pass",
        ],
        "input_hashes": {
            p["run_manifest"].name: sha256_file(p["run_manifest"]),
            p["export_manifest"].name: sha256_file(p["export_manifest"]),
            p["input_manifest"].name: sha256_file(p["input_manifest"]),
            p["unsupported_manifest"].name: sha256_file(p["unsupported_manifest"]),
        },
        "artifacts": {path.name: {"sha256": sha256_file(path), "rows": int(pd.read_csv(path).shape[0])} for path in artifacts},
    }
    write_json(p["run_manifest34"], manifest)
    return manifest
