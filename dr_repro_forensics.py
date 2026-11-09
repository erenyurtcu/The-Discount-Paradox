"""Read-only DR-Learner forensic test. Run from Machine_Learning_and_Reasoning root.
Does not overwrite historical predictions or model outputs.
"""
from pathlib import Path
import hashlib, json, platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

ROOT=Path.cwd()
raw=list((ROOT/'Hillstrom_Dataset').rglob('*MineThatData*.csv'))
raw=[p for p in raw if 'Processed' not in p.parts]
if len(raw)!=1: raise RuntimeError(f'Expected one raw Hillstrom CSV; found {raw}')
df=pd.read_csv(raw[0]); df.columns=df.columns.str.strip().str.lower()
df=df[df.segment.isin(['No E-Mail','Mens E-Mail'])].copy().reset_index(drop=True)
features=['recency','history','mens','womens','newbie','history_segment','zip_code','channel']
X=pd.get_dummies(df[features],columns=['history_segment','zip_code','channel'],dtype=float).fillna(0)
y=df.spend.to_numpy(float); t=(df.segment=='Mens E-Mail').astype(int).to_numpy()
ids=np.arange(len(df)); dev,test=train_test_split(ids,test_size=.2,stratify=t,random_state=42)
dev,val=train_test_split(dev,test_size=.1875,stratify=t[dev],random_state=43)
nuisance,cate=train_test_split(dev,test_size=25/65,stratify=t[dev],random_state=44)

def fit_predict(n_jobs):
 def forest(seed):
  return RandomForestRegressor(n_estimators=120,min_samples_leaf=80,max_features=.8,n_jobs=n_jobs,random_state=seed)
 mu=[]
 for arm in (0,1):
  m=forest(42+arm); ix=nuisance[t[nuisance]==arm]; m.fit(X.iloc[ix],y[ix]);mu.append(m)
 m0=mu[0].predict(X.iloc[cate]);m1=mu[1].predict(X.iloc[cate]);tc=t[cate];yc=y[cate]
 pseudo=m1-m0+2*tc*(yc-m1)-2*(1-tc)*(yc-m0)
 m=forest(52);m.fit(X.iloc[cate],pseudo)
 return {'validation':m.predict(X.iloc[val]),'test':m.predict(X.iloc[test])}

def diff(a,b):
 delta=np.abs(a-b)
 return {'exact_1e-8':bool(np.allclose(a,b,rtol=0,atol=1e-8)),
         'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),
         'corr':float(np.corrcoef(a,b)[0,1])}

print('=== DR REPRODUCIBILITY FORENSICS (READ ONLY) ===',flush=True)
print('Versions:',json.dumps({'python':platform.python_version(),'sklearn':sklearn.__version__,
 'numpy':np.__version__,'pandas':pd.__version__}),flush=True)
print('Source SHA256:',hashlib.sha256(raw[0].read_bytes()).hexdigest(),flush=True)
print('Rows:',{'nuisance':len(nuisance),'cate':len(cate),'validation':len(val),'test':len(test)},flush=True)
print('Training A: n_jobs=1',flush=True); a=fit_predict(1)
print('Training B: n_jobs=1',flush=True); b=fit_predict(1)
print('Training C: n_jobs=-1',flush=True); c=fit_predict(-1)
for split,idx,filename in [('validation',val,'hillstrom_dr_validation_predictions.parquet'),('test',test,'hillstrom_dr_test_predictions.parquet')]:
 old=pd.read_parquet(ROOT/'Hillstrom_Dataset'/'Processed'/filename)
 saved=old.predicted_cate.to_numpy(float)
 if len(saved)!=len(idx):raise RuntimeError(f'{split}: row count mismatch')
 if not np.array_equal(old.treatment.to_numpy(int),t[idx]) or not np.allclose(old.spend.to_numpy(float),y[idx],atol=1e-12,rtol=0):
  raise RuntimeError(f'{split}: treatment or spend order mismatch')
 print('\n'+split.upper(),flush=True)
 print('A vs B (repeatability):',json.dumps(diff(a[split],b[split])),flush=True)
 print('A vs C (parallelism):',json.dumps(diff(a[split],c[split])),flush=True)
 print('A vs historical:',json.dumps(diff(a[split],saved)),flush=True)
 print('C vs historical:',json.dumps(diff(c[split],saved)),flush=True)
 print('Saved SHA256:',hashlib.sha256((ROOT/'Hillstrom_Dataset'/'Processed'/filename).read_bytes()).hexdigest(),flush=True)
print('\nNO MODEL OR PREDICTION FILES MODIFIED.',flush=True)
