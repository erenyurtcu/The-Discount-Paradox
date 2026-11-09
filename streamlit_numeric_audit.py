"""Read-only Streamlit UI numerical and plot reactivity audit."""
from pathlib import Path
import json, math, re, sys
from streamlit.testing.v1 import AppTest

BASE = Path('profit_arena_v2')
PAGES = sorted(p for p in BASE.glob('[0-9][0-9]_*.py') if '_before_' not in p.name)
assert len(PAGES) == 8, f'Expected 8 active pages; got {[p.name for p in PAGES]}'

def metrics(at):
    return {m.label: (m.value, m.delta) for m in at.metric}

def plot_fingerprints(at):
    out=[]
    for el in at.get('plotly_chart'):
        try:
            raw = el.proto.figure.spec
            out.append(raw)
        except Exception:
            try: out.append(str(el.proto))
            except Exception: out.append(str(el))
    return out

def num(v):
    return float(re.sub(r'[^0-9.+-]', '', str(v).replace(',', '')))

def check_page(path):
    print('\n===', path.name, '===', flush=True)
    at=AppTest.from_file(str(path), default_timeout=120).run()
    if at.exception: raise AssertionError('Initial runtime exceptions: '+str([x.message for x in at.exception]))
    initial=metrics(at)
    charts0=plot_fingerprints(at)
    print('initial:',len(initial),'metrics;',len(charts0),'charts;',len(at.slider),'sliders;',len(at.select_slider),'select_sliders',flush=True)
    assert len(initial)+len(charts0)>0, 'No observable metric or chart'
    checked=0
    for index in range(len(at.slider)):
        at=AppTest.from_file(str(path), default_timeout=120).run()
        sl=at.slider[index]
        old=sl.value
        if isinstance(old,tuple):
            # preserve valid ordered range
            low,high=old
            step=sl.proto.step if hasattr(sl.proto,'step') else 0.1
            candidate=(low,high-step) if high-step>low else (low+step,high)
        else:
            lo,hi=sl.min,sl.max
            candidate=hi if old!=hi else lo
        sl.set_value(candidate).run()
        if at.exception: raise AssertionError(f'Slider {index} runtime exceptions: '+str([x.message for x in at.exception]))
        new=metrics(at)
        charts1=plot_fingerprints(at)
        changed_metrics=[k for k in initial if initial[k]!=new.get(k)]
        changed_charts=sum(a!=b for a,b in zip(charts0,charts1))
        print('slider',index,repr(sl.label),'->',repr(candidate),'metric_changes',changed_metrics,'chart_changes',changed_charts,flush=True)
        if not changed_metrics and not changed_charts:
            raise AssertionError(f'No observable numerical or chart response for {sl.label}')
        if path.name=='04_retail_pricing_lab.py':
            discount=next(s.value for s in at.slider if s.label=='Discount (%)')
            elasticity=next(s.value for s in at.slider if s.label=='Assumed price elasticity')
            cost=next(s.value for s in at.slider if s.label=='Assumed unit cost ($)')
            # The displayed new price, quantity, revenue and profit must satisfy their accounting identities.
            p=num(new['New unit price'][0]); q=num(new['Predicted weekly units'][0]); r=num(new['Scenario revenue'][0]); g=num(new['Scenario gross profit'][0]);
            assert math.isclose(r,p*q,abs_tol=max(0.35,0.003*q)),f'Revenue identity mismatch {r} vs {p*q}'
            assert math.isclose(g,(p-cost)*q,abs_tol=max(0.35,0.003*q)),f'Profit identity mismatch {g} vs {(p-cost)*q}'
            print('  pricing accounting identities PASS (display-rounding tolerance)',flush=True)
        checked+=1
    if len(at.slider)==0:
        print('No sliders: static page; runtime and chart presence checked',flush=True)
    print('PASS',path.name,checked,'slider scenarios',flush=True)
    return checked

if __name__=='__main__':
    total=0; fails=[]
    for p in PAGES:
        try: total+=check_page(p)
        except Exception as e:
            print('FAIL',p.name,repr(e),flush=True); fails.append((p.name,str(e)))
    print('\nSUMMARY: pages=',len(PAGES),'slider scenarios=',total,'failures=',len(fails),flush=True)
    for p,e in fails:print('FAIL',p,e,flush=True)
    sys.exit(1 if fails else 0)
