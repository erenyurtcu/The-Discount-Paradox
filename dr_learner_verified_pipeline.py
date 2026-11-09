"""Versioned, reproducible Hillstrom DR-Learner run. Never modifies historical outputs.
Run from project root: py -u .\dr_learner_verified_pipeline.py
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

ROOT=Path.cwd()
DATA=ROOT/'Hillstrom_Dataset'
OLD=DATA/'Processed'
DEST=OLD/'DR_Verified_v1'
COST=.50
P=.5
SEED=42

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def forest(seed):
    return RandomForestRegressor(n_estimators=120,min_samples_leaf=80,
        max_features=.8,n_jobs=1,random_state=seed)

def actions(pred):
    n=len(pred)
    order=np.argsort(-pred,kind='stable')
    result={'None':np.zeros(n,dtype=np.int8),'Everyone':np.ones(n,dtype=np.int8)}
    for fraction in (.2,.4,.6,.8):
        a=np.zeros(n,dtype=np.int8)
        a[order[:round(fraction*n)]]=1
        result[f'Top {round(100*fraction)}%']=a
    return result

def score(idx,pred,label,t,y):
    contrast=t[idx]*y[idx]/P-(1-t[idx])*y[idx]/(1-P)
    acts=actions(pred)
    rows=[]
    for name,a in acts.items():
        contribution=a*(contrast-COST)
        est=float(contribution.mean())
        se=float(contribution.std(ddof=1)/np.sqrt(len(idx)))
        rows.append(dict(split=label,policy=name,target_fraction=float(a.mean()),
            net_value=est,ci95_low=est-1.96*se,ci95_high=est+1.96*se))
    return pd.DataFrame(rows),acts

def train(X,y,t,nuisance,cate,val,test):
    mu={}
    for arm in (0,1):
        ix=nuisance[t[nuisance]==arm]
        model=forest(SEED+arm)
        model.fit(X.iloc[ix],y[ix])
        mu[arm]=model
    m0=mu[0].predict(X.iloc[cate])
    m1=mu[1].predict(X.iloc[cate])
    tc=t[cate]
    yc=y[cate]
    pseudo=m1-m0+tc*(yc-m1)/P-(1-tc)*(yc-m0)/(1-P)
    model=forest(SEED+10)
    model.fit(X.iloc[cate],pseudo)
    return model.predict(X.iloc[val]),model.predict(X.iloc[test])

def main():
    print('=== VERIFIED DR-LEARNER PIPELINE ===',flush=True)
    files=[p for p in DATA.rglob('*.csv') if 'Processed' not in p.parts and 'MineThatData' in p.name]
    if len(files)!=1:raise RuntimeError(f'Expected exactly one original Hillstrom CSV: {files}')
    raw=files[0]
    df=pd.read_csv(raw)
    df.columns=df.columns.str.strip().str.lower()
    df=df[df.segment.isin(['No E-Mail','Mens E-Mail'])].copy().reset_index(drop=True)
    features=['recency','history','mens','womens','newbie','history_segment','zip_code','channel']
    X=pd.get_dummies(df[features],columns=['history_segment','zip_code','channel'],dtype=float).fillna(0)
    y=df.spend.to_numpy(float)
    t=(df.segment=='Mens E-Mail').astype(int).to_numpy()
    ids=np.arange(len(df))
    dev,test=train_test_split(ids,test_size=.2,stratify=t,random_state=42)
    dev,val=train_test_split(dev,test_size=.1875,stratify=t[dev],random_state=43)
    nuisance,cate=train_test_split(dev,test_size=25/65,stratify=t[dev],random_state=44)
    assert not (set(nuisance)&set(cate) or set(nuisance)&set(val) or set(nuisance)&set(test) or set(cate)&set(val) or set(cate)&set(test) or set(val)&set(test))
    print('Rows:',{'nuisance':len(nuisance),'cate':len(cate),'validation':len(val),'test':len(test)},flush=True)
    print('Training first model (n_jobs=1)...',flush=True)
    va,te=train(X,y,t,nuisance,cate,val,test)
    print('Training independent verification model (n_jobs=1)...',flush=True)
    vb,tb=train(X,y,t,nuisance,cate,val,test)
    for name,a,b in [('validation',va,vb),('test',te,tb)]:
        maximum=float(np.max(np.abs(a-b)))
        print(f'{name}: maximum repeat difference = {maximum:.17g}',flush=True)
        if not np.array_equal(a,b):raise RuntimeError(f'FAIL: {name} predictions not exactly repeatable; nothing written')
    vdf,vactions=score(val,va,'validation',t,y)
    winner=str(vdf.loc[vdf.net_value.idxmax(),'policy'])
    tdf,tactions=score(test,te,'final_test',t,y)
    print('\nVALIDATION RESULTS\n'+vdf.to_string(index=False),flush=True)
    print('\nSelected policy ON VALIDATION:',winner,flush=True)
    print('\nFINAL TEST RESULTS\n'+tdf.to_string(index=False),flush=True)
    selected=tactions[winner]
    contrast=t[test]*y[test]/P-(1-t[test])*y[test]/(1-P)
    paired=(selected-tactions['Everyone'])*(contrast-COST)
    delta=float(paired.mean())
    se=float(paired.std(ddof=1)/np.sqrt(len(paired)))
    print(f'\nSelected minus Everyone: {delta:.8f}, 95% CI [{delta-1.96*se:.8f}, {delta+1.96*se:.8f}]',flush=True)
    val_preds=pd.DataFrame({'predicted_cate':va,'treatment':t[val],'spend':y[val]})
    test_preds=pd.DataFrame({'predicted_cate':te,'treatment':t[test],'spend':y[test],'selected_action':selected})
    expected={
        'hillstrom_dr_validation.csv':vdf,
        'hillstrom_dr_final_test.csv':tdf,
        'hillstrom_dr_validation_predictions.parquet':val_preds,
        'hillstrom_dr_test_predictions.parquet':test_preds,
    }
    # Never silently overwrite a prior verified run.
    if DEST.exists() and any(DEST.iterdir()):
        raise FileExistsError(f'{DEST} already contains files. Existing results preserved. Rename/archive it before a new run.')
    DEST.mkdir(parents=True,exist_ok=True)
    for name,data in expected.items():
        path=DEST/name
        if name.endswith('.csv'):data.to_csv(path,index=False)
        else:data.to_parquet(path,index=False)
        check=pd.read_csv(path,keep_default_na=False) if name.endswith('.csv') else pd.read_parquet(path)
        if len(check)!=len(data):raise RuntimeError(f'FAIL: readback row count {name}')
        print('WROTE',path,flush=True)
    manifest={
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'note':'New reproducible development run. Historical final test was previously inspected; not an untouched independent holdout.',
        'input_csv':str(raw.relative_to(ROOT)),'input_sha256':sha(raw),
        'python':platform.python_version(),'sklearn':sklearn.__version__,
        'numpy':np.__version__,'pandas':pd.__version__,
        'random_state':42,'n_jobs':1,'n_estimators':120,'min_samples_leaf':80,'max_features':.8,
        'cost_per_email':COST,'assignment_probability':P,
        'selected_policy_on_validation':winner,
        'paired_test_selected_minus_everyone':delta,
        'paired_test_ci95':[delta-1.96*se,delta+1.96*se],
        'split_sizes':{'nuisance':len(nuisance),'cate':len(cate),'validation':len(val),'test':len(test)},
        'files_sha256':{name:sha(DEST/name) for name in expected},
        'repeatability':'EXACT PASS for validation and test predictions across independent fits'
    }
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nPASS: exact repeatability, validation-selected policy, separate versioned artifacts',flush=True)
    print('OUTPUT:',DEST,flush=True)
    print('Historical files in Processed root were NOT modified.',flush=True)

if __name__=='__main__':main()
