"""Final DR version binding and evidence audit. Run from project root."""
from pathlib import Path
import json, hashlib, shutil, sys, ast
import pandas as pd

ROOT=Path(__file__).resolve().parent
PAGES=ROOT/'profit_arena_v2'
BASE=ROOT/'Hillstrom_Dataset'/'Processed'
VER=BASE/'DR_Verified_v1'
TARGETS=['03_model_disagreement.py','08_intervention_simulator.py']
DRFILE='hillstrom_dr_validation_predictions.parquet'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def check(cond,msg):
    if not cond:raise AssertionError(msg)
    print('PASS',msg,flush=True)

def main():
    print('=== FINAL TECHNICAL CLOSURE ===',flush=True)
    manifest=json.loads((VER/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files_sha256'].items():
        check((VER/name).exists() and sha(VER/name)==digest,'verified artifact checksum: '+name)
    check(manifest['repeatability'].startswith('EXACT PASS'),'deterministic training manifest')
    val=pd.read_csv(VER/'hillstrom_dr_validation.csv')
    test=pd.read_csv(VER/'hillstrom_dr_final_test.csv')
    pred=pd.read_parquet(VER/DRFILE)
    check(len(pred)==6392,'validation predictions count = 6392')
    check(set(['predicted_cate','treatment','spend']).issubset(pred.columns),'DR prediction columns compatible')
    selected=manifest['selected_policy_on_validation']
    check(selected==val.loc[val['net_value'].idxmax(),'policy'],'validation-selected policy consistent')
    check(selected in set(test['policy']),'selected policy exists in final test')
    check((PAGES/'03_model_disagreement.py').exists() and (PAGES/'08_intervention_simulator.py').exists(),'target pages present')
    for name in TARGETS:
        path=PAGES/name
        source=path.read_text(encoding='utf-8-sig')
        ast.parse(source,filename=name)
        check(DRFILE in source,'DR reference exists in '+name)
        old='"'+DRFILE+'"'
        new='"DR_Verified_v1/'+DRFILE+'"'
        if new in source:
            print('ALREADY BOUND',name,flush=True)
            continue
        check(old in source,'expected historical DR reference found in '+name)
        backup=path.with_suffix(path.suffix+'.pre_verified_backup')
        if not backup.exists():shutil.copy2(path,backup)
        source=source.replace(old,new)
        # Explicit provenance: other policy/agreement/hurdle artifacts remain historical.
        note='\n# Provenance: DR predictions come from DR_Verified_v1; policy/agreement/hurdle files remain historical exploratory analyses.\n'
        source+=note
        ast.parse(source,filename=name)
        path.write_text(source,encoding='utf-8')
        print('UPDATED',name,'(backup:',backup.name+')',flush=True)
    for name in TARGETS:
        source=(PAGES/name).read_text(encoding='utf-8-sig')
        check('"DR_Verified_v1/'+DRFILE+'"' in source,'verified DR source bound in '+name)
    for name in ['01_experiment_lab.py','02_policy_arena.py','03_model_disagreement.py','04_retail_pricing_lab.py','05_profit_frontier.py','06_uncertainty_lab.py','07_model_evidence.py','08_intervention_simulator.py']:
        ast.parse((PAGES/name).read_text(encoding='utf-8-sig'),filename=name)
    check(True,'all 8 pages syntax')
    print('\nSCIENTIFIC INTERPRETATION:',flush=True)
    print('- Hillstrom: randomized EMAIL treatment; no randomized retail price effects.',flush=True)
    print('- Dominicks: observational association, NOT identified causal elasticity.',flush=True)
    print('- Pricing simulator: assumed constant elasticity and fixed cost.',flush=True)
    print('- DR_Verified_v1: reproducible new run, historical files remain distinct.',flush=True)
    print('- Earlier final test was inspected: not an untouched independent holdout.',flush=True)
    print('- Policy Arena and intervention policy chart use earlier T-Learner evaluations, NOT the new DR policy test.',flush=True)
    print('- Model disagreement uses new DR versus historical Hurdle predictions: diagnostic comparison, not a newly joint-trained benchmark.',flush=True)
    print('- New selected policy:',selected,flush=True)
    print('- Final-test selected net value:',test.loc[test.policy==selected,'net_value'].iloc[0],flush=True)
    print('\nPASS: VERSION BINDING AND TECHNICAL CLOSURE STATIC CHECKS',flush=True)
    print('NEXT: rerun Streamlit numerical audit; it must show failures=0.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('FAIL:',type(e).__name__,str(e),file=sys.stderr,flush=True)
        sys.exit(1)
