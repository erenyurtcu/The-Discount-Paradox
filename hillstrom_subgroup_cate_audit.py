"""Hillstrom subgroup heterogeneity audit for DR_Verified_v1.
Run from the project root: py -u .\hillstrom_subgroup_cate_audit.py
Read-only with respect to existing models, predictions, and app pages.
"""
from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parent
VER=ROOT/"Hillstrom_Dataset"/"Processed"/"DR_Verified_v1"
OUT=ROOT/"Hillstrom_Dataset"/"Analysis"/"Subgroup_CATE_Verified_v1"
P=.5

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def main():
    manifest=json.loads((VER/"manifest.json").read_text(encoding="utf-8"))
    for name,digest in manifest["files_sha256"].items():
        path=VER/name
        assert path.exists() and sha(path)==digest, "Artifact checksum failed: "+name
    raw=ROOT/manifest["input_csv"]
    assert sha(raw)==manifest["input_sha256"], "Input CSV changed since verified training"
    df=pd.read_csv(raw)
    df.columns=df.columns.str.strip().str.lower()
    df=df[df.segment.isin(["No E-Mail","Mens E-Mail"])].copy().reset_index(drop=True)
    t=(df.segment=="Mens E-Mail").astype(int).to_numpy()
    ids=np.arange(len(df))
    dev,test=train_test_split(ids,test_size=.2,stratify=t,random_state=42)
    dev,val=train_test_split(dev,test_size=.1875,stratify=t[dev],random_state=43)
    nuisance,cate=train_test_split(dev,test_size=25/65,stratify=t[dev],random_state=44)
    assert len(val)==manifest["split_sizes"]["validation"]
    assert len(test)==manifest["split_sizes"]["test"]
    vp=pd.read_parquet(VER/"hillstrom_dr_validation_predictions.parquet")
    tp=pd.read_parquet(VER/"hillstrom_dr_test_predictions.parquet")
    for label,idx,pred in [("validation",val,vp),("test",test,tp)]:
        assert len(pred)==len(idx),label+" length mismatch"
        assert np.array_equal(pred["treatment"].to_numpy(),t[idx]),label+" treatment row order mismatch"
        assert np.allclose(pred["spend"].to_numpy(),df.spend.to_numpy()[idx],rtol=0,atol=1e-10),label+" spend row order mismatch"
    frames=[]
    for label,idx,pred in [("validation",val,vp),("test",test,tp)]:
        f=df.iloc[idx][["recency","history","newbie","channel","zip_code","mens","womens"]].copy().reset_index(drop=True)
        f["split"]=label
        f["treatment"]=pred.treatment.to_numpy()
        f["spend"]=pred.spend.to_numpy()
        f["predicted_cate"]=pred.predicted_cate.to_numpy()
        frames.append(f)
    all_data=pd.concat(frames,ignore_index=True)
    # All segmentation cutpoints are derived from model-development data only.
    recency_cut=float(df.iloc[dev].recency.median())
    history_cut=float(df.iloc[dev].history.median())
    all_data["recency_group"]=np.where(all_data.recency<=recency_cut,"Lower recency","Higher recency")
    all_data["history_group"]=np.where(all_data.history<=history_cut,"Lower history","Higher history")
    all_data["newbie_group"]=np.where(all_data.newbie==1,"New customer","Existing customer")
    dims=["newbie_group","channel","zip_code","recency_group","history_group"]
    rows=[]
    for split in ("validation","test"):
        f=all_data[all_data.split==split]
        for dim in dims:
            for group,g in f.groupby(dim,dropna=False,sort=True):
                n=len(g)
                if n<100:continue
                a=g[g.treatment==1].spend.to_numpy(float)
                b=g[g.treatment==0].spend.to_numpy(float)
                if min(len(a),len(b))<30:continue
                ate=float(a.mean()-b.mean())
                se=float(np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b)))
                cate=g.predicted_cate.to_numpy(float)
                # Descriptive distribution of predictions; NOT an inferential CI for true CATE.
                rows.append(dict(split=split,feature=dim,group=str(group),n=n,treated_n=len(a),control_n=len(b),
                    predicted_cate_mean=float(cate.mean()),predicted_cate_median=float(np.median(cate)),
                    predicted_cate_sd=float(cate.std(ddof=1)),
                    randomized_group_ate=ate,randomized_group_ate_se=se,
                    randomized_group_ate_ci95_low=ate-1.96*se,
                    randomized_group_ate_ci95_high=ate+1.96*se))
    result=pd.DataFrame(rows)
    assert not result.empty
    # Difference in randomized group ATE between levels, only for binary segmentation dimensions.
    contrasts=[]
    for split in ("validation","test"):
        for dim in ["newbie_group","recency_group","history_group"]:
            s=result[(result.split==split)&(result.feature==dim)].sort_values("group")
            if len(s)!=2:continue
            a,b=s.iloc[0],s.iloc[1]
            delta=float(a.randomized_group_ate-b.randomized_group_ate)
            se=float(np.hypot(a.randomized_group_ate_se,b.randomized_group_ate_se))
            contrasts.append(dict(split=split,feature=dim,group_a=a.group,group_b=b.group,
                difference_in_group_ate=delta,se=se,ci95_low=delta-1.96*se,ci95_high=delta+1.96*se,
                ci_excludes_zero=bool(delta-1.96*se>0 or delta+1.96*se<0)))
    contrast_df=pd.DataFrame(contrasts)
    OUT.mkdir(parents=True,exist_ok=True)
    result.to_csv(OUT/"subgroup_cate_and_randomized_ate.csv",index=False)
    contrast_df.to_csv(OUT/"subgroup_ate_contrasts.csv",index=False)
    (OUT/"subgroup_definition.json").write_text(json.dumps({
        "source":"DR_Verified_v1 with original Hillstrom rows reattached by verified split indices",
        "recency_median_dev":recency_cut,"history_median_dev":history_cut,
        "dimensions":dims,"minimum_group_n":100,"minimum_arm_n":30,
        "confidence_intervals":"Normal-approximation Welch SE for randomized within-group treatment-control spend mean difference; unadjusted exploratory 95% CI",
        "cautions":[
            "Predicted CATE means are descriptive predictions, not causal effect validation or uncertainty intervals.",
            "Randomized subgroup ATEs are independent experimental estimates within pre-treatment groups.",
            "Exploratory multiple comparisons; no multiplicity adjustment.",
            "Previously inspected test data are not a pristine confirmatory holdout.",
            "Hillstrom treatment is Mens E-Mail vs No E-Mail, not a retail discount.",
            "No model refitting or modification of verified artifacts."
        ]
    },indent=2,ensure_ascii=False),encoding="utf-8")
    lines=["# Hillstrom: verified DR-Learner subgroup audit","",
        "Treatment: Mens E-Mail versus No E-Mail; outcome: spend ($).",
        "Subgroup definitions use only baseline variables. Median cutpoints derived from development sample.",
        "Predicted CATE averages are **descriptive model predictions**, not demonstrated treatment effects.",
        "Randomized subgroup ATE is an independent treatment-minus-control spend difference within each group.",
        "95% CIs are exploratory normal approximations; multiple testing not adjusted.",
        "Previously inspected test set is not an untouched confirmatory holdout.","",
        "## Subgroup results",""]
    for split in ("validation","test"):
        lines.append("### "+split)
        s=result[result.split==split]
        lines.extend(["| Feature | Group | N | Mean predicted CATE ($) | Randomized subgroup ATE ($) | Exploratory 95% CI ($) |",
        "|---|---|---:|---:|---:|---|"])
        for r in s.itertuples():
            lines.append(f"| {r.feature} | {r.group} | {r.n} | {r.predicted_cate_mean:+.3f} | {r.randomized_group_ate:+.3f} | [{r.randomized_group_ate_ci95_low:+.3f}, {r.randomized_group_ate_ci95_high:+.3f}] |")
        lines.append("")
    lines+=["## Binary subgroup interaction contrasts (exploratory)","",
        "| Split | Feature | Group A - Group B | ATE difference ($) | 95% CI ($) | CI excludes 0? |",
        "|---|---|---|---:|---|---|"]
    for r in contrast_df.itertuples():
        lines.append(f"| {r.split} | {r.feature} | {r.group_a} - {r.group_b} | {r.difference_in_group_ate:+.3f} | [{r.ci95_low:+.3f}, {r.ci95_high:+.3f}] | {'Yes' if r.ci_excludes_zero else 'No'} |")
    lines+=["","## Interpretation guide","",
        "A higher predicted CATE does not itself establish that the subgroup has a higher true treatment effect.",
        "If a subgroup's randomized ATE CI includes zero, the data do not establish an effect at the nominal 95% level.",
        "If an interaction contrast CI includes zero, the data do not establish a difference in treatment effects between the two groups.",
        "Even if an exploratory CI excludes zero, account for multiple comparisons and the previously inspected test set.",
        "These analyses concern randomized email treatment, not price elasticity or discount causality.",""]
    (OUT/"subgroup_report.md").write_text("\n".join(lines),encoding="utf-8")
    print("PASS verified input/model checksums and row alignment")
    print("PASS subgroup analysis; rows =",len(result),"interaction contrasts =",len(contrast_df))
    print("Development median thresholds: recency =",recency_cut,", history =",history_cut)
    print("\nTEST subgroup results:\n",result[result.split=="test"][["feature","group","n","predicted_cate_mean","randomized_group_ate","randomized_group_ate_ci95_low","randomized_group_ate_ci95_high"]].to_string(index=False))
    print("\nTEST interaction contrasts:\n",contrast_df[contrast_df.split=="test"].to_string(index=False))
    print("\nREPORT:",OUT/"subgroup_report.md")
    print("CSV:",OUT/"subgroup_cate_and_randomized_ate.csv")
    print("Existing model/app artifacts unchanged.")

if __name__=="__main__":
    try:main()
    except Exception as e:
        print("FAIL:",type(e).__name__,str(e),file=sys.stderr)
        raise
