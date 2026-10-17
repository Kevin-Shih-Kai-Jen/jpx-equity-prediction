"""Verify aggregate report consistency without training or loading model binaries."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

D=Path(__file__).resolve().parent
a=json.loads((D/'summary.json').read_text())
f=pd.read_csv(D/'fold_metrics.csv')
d=pd.read_csv(D/'daily_metrics.csv',parse_dates=['date'])
c=pd.read_csv(D/'cost_sensitivity.csv')
m=pd.read_csv(D/'monthly_metrics.csv')
o=a['overall']
assert list(f.fold)==list(range(1,9))
assert a['run_audit']['status']=='passed'
assert a['run_audit']['completed_epochs']==160
assert a['run_audit']['completed_folds']==8
assert f.samples.sum()==o['samples']==1850796
assert f.days.sum()==len(d)==o['days']==958
assert d.date.is_monotonic_increasing and d.date.is_unique
assert d.date.iloc[0].strftime('%Y-%m-%d')==o['start']
assert d.date.iloc[-1].strftime('%Y-%m-%d')==o['end']
assert d.rank_ic.isna().sum()==1
assert d.loc[d.rank_ic.isna(),'date'].iloc[0].strftime('%Y-%m-%d')=='2020-09-29'
def eq(x,y,tol=1e-10): assert np.isclose(x,y,rtol=0,atol=tol),(x,y)
r=d.spread_return.to_numpy()
eq(r.mean()/r.std(ddof=1),o['sharpe_unannualized_pooled'])
eq(d.rank_ic.mean(),o['daily_rank_ic_mean'])
eq(r.sum(),o['daily_double_precision_sum'])
eq(f.earnings.sum(),o['original_total_earnings'])
eq(np.average(f.mae,weights=f.samples),o['weighted_mae'],2e-9)
eq(np.average(f.zero_mae,weights=f.samples),o['weighted_zero_mae'])
assert np.allclose(np.cumsum(r),d.cumulative_spread_sum,rtol=0,atol=1e-10)
for _,row in f.iterrows():
    ds=d.loc[d.fold.eq(row.fold)]
    assert len(ds)==row.days and ds.stocks.sum()==row.samples
    rv=ds.spread_return.to_numpy(dtype='float32')
    eq(float(rv.sum()),row.earnings,1e-9)
    eq(float(rv.mean()/rv.std(ddof=1)),row.sharpe,1e-8)
    eq(ds.rank_ic.mean(),row.rank_IC)
for _,row in c.iterrows():
    net=r-row.cost_bps_per_traded_notional*1e-4*d.target_weight_change_including_final_close.to_numpy()
    eq(net.sum(),row.sum_returns)
    eq(net.mean()/net.std(ddof=1),row.sharpe_unannualized)
eq(r.sum()/d.target_weight_change_including_final_close.sum()*1e4,o['cost_break_even_bps'])
eq(m.spread_sum.sum(),r.sum())
print('PASS: 8 folds, 160 epochs, 958 dates, 1,850,796 samples; aggregate metrics and cost scenarios agree.')
