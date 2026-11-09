"""Read-only three-part forensic audit. Run at the project root: py final_three_checks.py"""
from pathlib import Path
import ast, json, traceback, hashlib
import numpy as np
import pandas as pd
ROOT=Path.cwd(); outcomes={}
def section(name,fn):
 print('\n'+'='*20+' '+name+' '+'='*20,flush=True)
 try: outcomes[name]=fn()
 except Exception as exc:
  outcomes[name]={'status':'ERROR','error':str(exc)}; traceback.print_exc()
 print('RESULT:',json.dumps(outcomes[name],ensure_ascii=False,default=str),flush=True)

def dr():
 from sklearn.model_selection import train_test_split
 from sklearn.ensemble import RandomForestRegressor
 src=list((ROOT/'Hillstrom_Dataset').glob('*MineThatData*.csv'))
 if len(src)!=1: raise RuntimeError('Original Hillstrom CSV not unique: '+str(src))
 df=pd.read_csv(src[0]);df.columns=df.columns.str.strip().str.lower()
 df=df[df.segment.isin(['No E-Mail','Mens E-Mail'])].copy().reset_index(drop=True)
 cols=['recency','history','mens','womens','newbie','history_segment','zip_code','channel']
 X=pd.get_dummies(df[cols],columns=['history_segment','zip_code','channel'],dtype=float).fillna(0)
 t=(df.segment=='Mens E-Mail').astype(int).to_numpy();y=df.spend.to_numpy(float)
 idx=np.arange(len(df));dev,test=train_test_split(idx,test_size=.20,stratify=t,random_state=42)
 dev,val=train_test_split(dev,test_size=.1875,stratify=t[dev],random_state=43)
 nuisance,cate=train_test_split(dev,test_size=25/65,stratify=t[dev],random_state=44)
 def forest(seed):return RandomForestRegressor(n_estimators=120,min_samples_leaf=80,max_features=.8,n_jobs=1,random_state=seed)
 models=[]
 for arm in [0,1]:
  m=forest(42+arm);part=nuisance[t[nuisance]==arm];m.fit(X.iloc[part],y[part]);models.append(m)
  print('Nuisance fitted:',arm,flush=True)
 m0=models[0].predict(X.iloc[cate]);m1=models[1].predict(X.iloc[cate]);tc=t[cate];yc=y[cate]
 pseudo=m1-m0+2*tc*(yc-m1)-2*(1-tc)*(yc-m0)
 m=forest(52);m.fit(X.iloc[cate],pseudo);print('DR model fitted',flush=True)
 res={};overall=True
 for name,ids,filename in [('validation',val,'hillstrom_dr_validation_predictions.parquet'),('test',test,'hillstrom_dr_test_predictions.parquet')]:
  saved=pd.read_parquet(ROOT/'Hillstrom_Dataset'/'Processed'/filename)
  pred=m.predict(X.iloc[ids]);old=saved.predicted_cate.to_numpy(float)
  if len(pred)!=len(old):raise ValueError(name+' row count differs')
  matched_t=np.array_equal(saved.treatment.to_numpy(int),t[ids]);matched_y=np.allclose(saved.spend.to_numpy(float),y[ids],rtol=0,atol=1e-12)
  delta=np.abs(pred-old);exact=bool(np.allclose(pred,old,atol=1e-8,rtol=0))
  overall &= exact and matched_t and matched_y
  res[name]={'n':len(pred),'max_abs_diff':float(delta.max()),'mean_abs_diff':float(delta.mean()),'prediction_match':exact,'treatment_match':matched_t,'spend_match':matched_y,'corr':float(np.corrcoef(pred,old)[0,1])}
 return {'status':'PASS' if overall else 'FAIL_REPRODUCIBILITY','details':res,'interpretation':'Mismatch does not prove its historical cause; saved model provenance is needed.'}

def promo():
 import statsmodels.api as sm
 path=ROOT/'Dominicks_Dataset'/'Analysis'/'sdr_pilot_4900000663.parquet'
 df=pd.read_parquet(path,columns=['store','week','move','price','qty','sale'])
 df=df[(df.price>0)&(df.qty>0)&(df.move>0)].copy().sort_values(['store','week'])
 df['lp']=np.log(df.price/df.qty);df['ls']=np.log(df.move)
 df['code']=df.sale.fillna('').astype(str).str.strip().replace('', 'NONE')
 for c in ['week','lp','ls','code']:df['prev_'+c]=df.groupby('store',sort=False)[c].shift()
 df=df[df.prev_week==df.week-1].copy()
 df['dx']=df.lp-df.prev_lp;df['dy']=df.ls-df.prev_ls
 for c in ['dx','dy']:df[c]-=df.groupby('week')[c].transform('mean')
 X=pd.DataFrame({'price':df.dx},index=df.index)
 for code in ['S','B']:
  z=(df.code==code).astype(float)-(df.prev_code==code).astype(float)
  X['delta_promo_'+code]=z-z.groupby(df.week).transform('mean')
 fit=sm.OLS(df.dy,X).fit(cov_type='cluster',cov_kwds={'groups':df.store})
 old=pd.read_csv(ROOT/'Dominicks_Dataset'/'Analysis'/'diet_coke_promotion_adjusted_results.csv')
 old=old.loc[old.model=='Week + promotion-adjusted'].iloc[0]
 checks={'n':int(len(df)),'stores':int(df.store.nunique()),'coefficient':float(fit.params['price']),'se':float(fit.bse['price']),'ci_low':float(fit.conf_int().loc['price',0]),'ci_high':float(fit.conf_int().loc['price',1]),'promo_S':float(fit.params['delta_promo_S']),'promo_B':float(fit.params['delta_promo_B'])}
 tolerances={'coefficient':1e-8,'se':1e-6,'ci_low':2e-6,'ci_high':2e-6,'promo_S':1e-8,'promo_B':1e-8}
 compare={'coefficient':'price_coefficient','se':'clustered_se','ci_low':'ci_lower','ci_high':'ci_upper','promo_S':'promo_S_coefficient','promo_B':'promo_B_coefficient'}
 diffs={k:abs(checks[k]-float(old[v])) for k,v in compare.items()}
 passed=(len(df)==int(old.observations)) and all(diffs[k]<=tolerances[k] for k in diffs)
 return {'status':'PASS' if passed else 'FAIL','reproduced':checks,'absolute_differences':diffs}

def ui():
 from streamlit.testing.v1 import AppTest
 base=ROOT/'profit_arena_v2'
 pages=sorted(p for p in base.glob('[0-9][0-9]_*.py') if '_before_' not in p.name)
 if len(pages)!=8:raise RuntimeError('Expected 8 pages, found '+str(len(pages)))
 result={};all_pass=True
 for page in pages:
  try:
   at=AppTest.from_file(str(page),default_timeout=60).run()
   errs=[str(x.message) for x in at.exception]
   sliders=list(at.slider)
   interactions=[]
   for i,s in enumerate(sliders[:2]):
    # Streamlit's AppTest supports only certain slider value types.
    old=s.value;vmin=s.min;vmax=s.max
    target=vmax if old!=vmax else vmin
    if target==old:continue
    try:
     at.slider[i].set_value(target).run()
     errors_after=[str(x.message) for x in at.exception]
     interactions.append({'slider':str(s.label),'changed':str(target),'exceptions':errors_after})
     errs.extend(errors_after)
    except Exception as ex:interactions.append({'slider':str(s.label),'error':str(ex)});errs.append(str(ex))
   ok=not errs and bool(at.title or at.header or at.subheader or at.markdown or at.metric or at.plotly_chart)
   all_pass &=ok
   result[page.name]={'status':'PASS' if ok else 'FAIL','sliders':len(sliders),'interactions':interactions,'errors':errs[:4]}
  except Exception as ex:
   all_pass=False;result[page.name]={'status':'ERROR','error':str(ex)}
  print(page.name,result[page.name]['status'],flush=True)
 return {'status':'PASS' if all_pass else 'FAIL','pages':result,'caution':'AppTest checks runtime exceptions and slider reruns, not numerical plot correctness or visual layout.'}

section('DR_LEARNER',dr)
section('DOMINICKS_PROMOTION',promo)
section('STREAMLIT_APPTEST',ui)
print('\n=== FINAL SUMMARY ===')
for k,v in outcomes.items():print(k,v['status'])
print('This audit is read-only. It does not replace the saved model files.')
