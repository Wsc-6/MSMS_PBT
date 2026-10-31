"""Recompute frozen external comparisons without fitting or selecting a model.

Run with the bundled Python from the project root. All writes stay beside this file.
"""
from pathlib import Path
import hashlib, json, platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent / 'inputs'
OUT = Path(__file__).resolve().parent
PLAN_PATH = OUT / "analysis_plan.json"
PLAN = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
MODELS = PLAN["models"]
CHECKS = []

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check(name, passed, detail):
    CHECKS.append({"check": name, "passed": bool(passed), "detail": str(detail)})
    if not passed:
        raise AssertionError(f"{name}: {detail}")

def read(key):
    spec = PLAN["inputs"][key]
    path = ROOT / spec["path"]
    check("input_hash_" + key, sha(path) == spec["sha256"], spec["path"])
    return pd.read_csv(path, low_memory=False)

def metric(y, score, classification):
    if classification:
        return float(average_precision_score(y, score)) if np.unique(y).size == 2 else np.nan
    return float(np.mean(np.abs(y-score)))

def percentile(a):
    a=np.asarray(a, dtype=float)
    a=a[np.isfinite(a)]
    return (*np.quantile(a, [0.025, 0.975]), len(a)) if len(a) else (np.nan,np.nan,0)

def main():
    long = read("prediction_long")
    final = read("final_external")
    original = read("original_metadata")
    matched = read("matched_unlabeled")
    read("novel_applicability")
    read("final_registry")
    freeze_path=ROOT/PLAN["inputs"]["matched_prediction_freeze"]["path"]
    check("input_hash_matched_prediction_freeze",sha(freeze_path)==PLAN["inputs"]["matched_prediction_freeze"]["sha256"],freeze_path)
    old_freeze=json.loads(freeze_path.read_text(encoding="utf-8"))
    old_hash=old_freeze["artifact"]["40_new_external_predictions_unlabeled.csv.gz"]
    check("historical_matched_prediction_freeze_hash",old_hash==PLAN["inputs"]["matched_unlabeled"]["sha256"],old_hash)
    check("matched_unlabeled_has_no_y_true", "y_true" not in matched.columns, list(matched.columns))
    external=long[long.cohort.isin(["selected_novel","natural_novel","transfer"]) & long.pipeline.isin(MODELS)].copy()
    check("unique_model_task_compound",not external.duplicated(["pipeline","task","inchikey_2d"]).any(),len(external))
    check("unique_final_task_compound",not final.duplicated(["task","inchikey_2d"]).any(),len(final))
    source_frames={
        MODELS[0]: final.rename(columns={"y_score_calibrated":"score_source"}),
        MODELS[1]: original[original.pipeline.eq("Metadata only")].rename(columns={"y_score":"score_source"}),
        MODELS[2]: matched[matched.pipeline.eq("Matched nonfragment HGB")].rename(columns={"y_score":"score_source"}),
    }
    for model, source in source_frames.items():
        actual=external[external.pipeline.eq(model)].merge(source[["task","inchikey_2d","score_source"]],on=["task","inchikey_2d"],validate="one_to_one")
        check("source_score_match_"+model,len(actual)==len(source) and np.allclose(actual.y_score,actual.score_source,atol=1e-12,rtol=0),f"n={len(actual)}, max_diff={np.max(np.abs(actual.y_score-actual.score_source))}")
    label_check=external.merge(final[["task","inchikey_2d","y_true"]].rename(columns={"y_true":"canonical_y"}),on=["task","inchikey_2d"],validate="many_to_one")
    diff=np.abs(label_check.y_true-label_check.canonical_y)
    check("canonical_label_agreement",not (diff>1e-5).any() and label_check.y_true.isna().equals(label_check.canonical_y.isna()),float(diff.max()))
    check("finite_all_external_scores",np.isfinite(external.y_score).all(),len(external))
    absolute, contrasts, missingness, compounds=[],[],[],[]
    for cohort, roles in PLAN["cohorts"].items():
        for task in PLAN["tasks"]:
            canon=final[final.evaluation_role.isin(roles)&final.task.eq(task)][["inchikey_2d","evaluation_role","y_true","retained"]].set_index("inchikey_2d").sort_index()
            part=external[external.cohort.isin(roles)&external.task.eq(task)]
            wide=part.pivot(index="inchikey_2d",columns="pipeline",values="y_score")
            data=canon.join(wide[MODELS],how="left")
            finite_y=np.isfinite(data.y_true)
            finite_all=np.isfinite(data[MODELS]).all(axis=1)
            common=data.loc[finite_y & finite_all].copy()
            classification=task=="P_screen_not_ready_binary"
            common["cohort"]=cohort
            common["task"]=task
            compounds.append(common.reset_index())
            n=len(common)
            y=common.y_true.to_numpy(float)
            scores=common[MODELS].to_numpy(float)
            n_pos=int((y==1).sum()) if classification else None
            n_neg=int((y==0).sum()) if classification else None
            common_info={"cohort":cohort,"task":task,"n":n,"n_positive":n_pos,"n_negative":n_neg,"metric":"AP" if classification else "MAE","score_stage":"all_available_before_rejection"}
            point=np.array([metric(y,scores[:,m],classification) for m in range(3)])
            rng=np.random.default_rng(42)
            samples=np.full((2000,3),np.nan)
            for b in range(2000):
                ix=rng.integers(0,n,n)
                for m in range(3):
                    samples[b,m]=metric(y[ix],scores[ix,m],classification)
            for m,model in enumerate(MODELS):
                lo,hi,valid=percentile(samples[:,m])
                absolute.append({**common_info,"model":model,"value":point[m],"ci_low":lo,"ci_high":hi,"valid_bootstrap":valid})
                exists=canon.index.isin(part.loc[part.pipeline.eq(model),"inchikey_2d"])
                finite_score=np.isfinite(data[model])
                missingness.append({"cohort":cohort,"task":task,"model":model,"cohort_compounds":len(data),"label_available":int(finite_y.sum()),"label_missing":int((~finite_y).sum()),"prediction_rows":int(exists.sum()),"prediction_rows_missing":int((~exists).sum()),"nonfinite_prediction":int((~finite_score).sum()),"label_available_prediction_missing":int((finite_y&~finite_score).sum()),"model_evaluable":int((finite_y&finite_score).sum()),"common_three_model_n":n,"lost_to_common_intersection":int((finite_y&finite_score).sum())-n,"final_model_labeled_retained":int((finite_y&data.retained.astype(bool)).sum()),"final_model_labeled_rejected":int((finite_y&~data.retained.astype(bool)).sum()),"final_model_all_retained":int(data.retained.astype(bool).sum()),"final_model_all_rejected":int((~data.retained.astype(bool)).sum())})
            for m in (1,2):
                sign=1 if classification else -1
                gain=sign*(point[0]-point[m])
                draws=sign*(samples[:,0]-samples[:,m])
                lo,hi,valid=percentile(draws)
                contrasts.append({**common_info,"candidate":MODELS[0],"reference":MODELS[m],"positive_gain_metric":"AP_gain" if classification else "MAE_reduction","gain":gain,"ci_low":lo,"ci_high":hi,"valid_bootstrap":valid,"interval_interpretation":"pointwise_descriptive"})
    abs_df=pd.DataFrame(absolute); contrast_df=pd.DataFrame(contrasts)
    missing_df=pd.DataFrame(missingness); compound_df=pd.concat(compounds,ignore_index=True)
    check("all_twenty_cohort_endpoint_sets",len(abs_df)==60 and len(contrast_df)==40,f"absolute={len(abs_df)}, contrasts={len(contrast_df)}")
    check("no_loss_to_three_model_intersection",missing_df.lost_to_common_intersection.eq(0).all(),missing_df.lost_to_common_intersection.sum())
    check("bootstrap_counts_valid",contrast_df.valid_bootstrap.between(1,2000).all(),contrast_df.valid_bootstrap.min())
    # MAE is additive over disjoint strata on a fixed compound set; AP is not.
    relation=[]
    for task in PLAN["tasks"]:
        for reference in MODELS[1:]:
            sub=contrast_df[(contrast_df.task==task)&(contrast_df.reference==reference)]
            pooled=sub[sub.cohort.eq("novel_all")].iloc[0]
            strata=sub[sub.cohort.isin(["selected_novel","natural_novel"])]
            weighted=float(np.average(strata.gain,weights=strata.n))
            relation.append({"task":task,"reference":reference,"pooled_n":int(pooled.n),"strata_n_sum":int(strata.n.sum()),"pooled_gain":pooled.gain,"n_weighted_stratum_gain":weighted,"difference":pooled.gain-weighted,"weighted_identity_expected":task!="P_screen_not_ready_binary"})
            check("cohort_counts_partition_"+task+reference,int(pooled.n)==int(strata.n.sum()),f"{pooled.n} = {strata.n.sum()}")
            if task!="P_screen_not_ready_binary":
                check("MAE_stratum_weighting_"+task+reference,abs(pooled.gain-weighted)<1e-12,pooled.gain-weighted)
    for filename,frame in {"absolute_metrics.csv":abs_df,"paired_gains.csv":contrast_df,"missingness.csv":missing_df,"common_compound_predictions.csv.gz":compound_df,"pooled_stratified_relation.csv":pd.DataFrame(relation),"checks.csv":pd.DataFrame(CHECKS)}.items():
        frame.to_csv(OUT/filename,index=False)
    table31=[]
    for task in PLAN["tasks"]:
        a=abs_df[abs_df.cohort.eq("novel_all")&abs_df.task.eq(task)]
        g=contrast_df[contrast_df.cohort.eq("novel_all")&contrast_df.task.eq(task)]
        first=a.iloc[0]
        row={"task":task,"metric":first.metric,"n":int(first.n),"n_positive":first.n_positive,"n_negative":first.n_negative}
        for model,prefix in zip(MODELS,["final","original_metadata","matched_nonfragment"]):
            s=a[a.model.eq(model)].iloc[0]
            for field in ["value","ci_low","ci_high"]:row[prefix+"_"+field]=s[field]
        for model,prefix in zip(MODELS[1:],["gain_vs_original_metadata","gain_vs_matched_nonfragment"]):
            s=g[g.reference.eq(model)].iloc[0]
            for field in ["gain","ci_low","ci_high"]:row[prefix+("" if field=="gain" else "_"+field)]=s[field]
        row["interval_interpretation"]="pointwise descriptive 95% paired compound bootstrap, B=2000, seed=42"
        table31.append(row)
    pd.DataFrame(table31).to_csv(OUT/"TableS31_novel_all_three_model_comparison.csv",index=False)
    concise=abs_df[abs_df.cohort.eq("novel_all")].pivot(index="task",columns="model",values="value")
    print(concise.to_string())
    print(contrast_df[contrast_df.cohort.eq("novel_all")][["task","n","reference","gain","ci_low","ci_high"]].to_string(index=False))
    print(f"Checks: {len(CHECKS)} passed")
    manifest={"executed_at_utc":datetime.now(timezone.utc).isoformat(),"analysis_plan_sha256":sha(PLAN_PATH),"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__,"training_performed":False,"source_artifacts_modified":False,"outputs":{p.name:sha(p) for p in sorted(OUT.glob("*.csv*"))},"reanalysis_script_sha256":sha(Path(__file__))}
    (OUT/"run_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
