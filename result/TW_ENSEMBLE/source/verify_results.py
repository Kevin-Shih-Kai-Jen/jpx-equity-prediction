"""Verify publication hashes and recompute the published portfolio/fold results.

Uses only the included data. Add --check-model to compare checkpoint tensors
(requires PyTorch). Full-universe ranking and MAE require the original prediction
Parquet files, which are retained in the original completed Kaggle export.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'data'
parser=argparse.ArgumentParser();parser.add_argument('--check-model',action='store_true');args=parser.parse_args()
read=lambda name:json.loads((DATA/name).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
env=read('environment.json');run=read('backtest_summary.json')
d=pd.read_csv(DATA/'backtest_daily.csv',parse_dates=['Date','realized_date'])
g=d.loc[d.complete].set_index('Date').sort_index()
f=pd.read_csv(DATA/'fold_metrics.csv');late=pd.read_csv(DATA/'fold_metrics_from_2021_H2.csv')
p=pd.read_csv(DATA/'selected_positions.csv.gz',parse_dates=['Date','LabelEntry','LabelEnd'])
p=p.loc[p.CompleteDay]
training=pd.read_csv(DATA/'fold_training_summary.csv')
assert len(f)==40 and len(training)==40 and g.index.is_unique
assert len(g)==2334 and (~d.complete).sum()==17
assert f.days.sum()==2334 and late.days.sum()==1302
assert set(f.fold)==set(g.fold)
assert p.groupby('Date').size().eq(6).all()
np.testing.assert_allclose(p.groupby('Date').Weight.sum(),0.,atol=1e-12)
np.testing.assert_allclose(p.groupby('Date').Weight.apply(lambda x:x.abs().sum()),1.,atol=1e-12)
reconstructed=(p.Weight*p.Target).groupby(p.Date).sum()
np.testing.assert_allclose(reconstructed.loc[g.index],g.gross,rtol=1e-10,atol=1e-12)
c=env['signature']['config'];opening=p.Weight.abs();closing=opening*(1+p.Target)
sales=np.where(p.Weight>0,closing,opening)
borrow=np.where(p.Weight<0,opening*c['short_borrow_annual']*(p.LabelEnd-p.LabelEntry).dt.days/365,0.)
cost=((opening+closing)*(c['commission_one_way']+c['slippage_one_way'])+sales*c['sell_tax']+borrow).groupby(p.Date).sum()
np.testing.assert_allclose(cost.loc[g.index],g.cost_scenario,rtol=1e-10,atol=1e-12)
np.testing.assert_allclose(g.gross-g.cost_scenario,g.net_cost_scenario,atol=1e-12)
for table,frame,name in [(f,g,'all_validation'),(late,g.loc[g.index>='2021-07-01'],'from_2021_H2')]:
    for col,prefix in [('gross','gross'),('net_cost_scenario','net')]:
        expected=run['conclusions'][name]['portfolios'][col]
        logs=np.log1p(frame[col]).sum()
        assert np.isclose(np.expm1(logs),expected['compounded_return'],atol=1e-12)
        assert np.isclose(np.expm1(logs*252/len(frame)),expected['cagr_252'],atol=1e-12)
        assert np.isclose(np.expm1(np.log1p(table[f'{prefix}_return_total']).sum()),expected['compounded_return'],atol=1e-12)
        for row in table.itertuples():
            daily=frame.loc[frame.fold.eq(row.fold),col]
            assert len(daily)==row.days
            logs=np.log1p(daily).sum()
            assert np.isclose(np.expm1(logs),getattr(row,f'{prefix}_return_total'),atol=1e-12)
            assert np.isclose(np.expm1(logs*252/len(daily)),getattr(row,f'{prefix}_annualized_252'),atol=1e-12)
assert f.gross_return_total.gt(0).sum()==19 and f.net_return_total.gt(0).sum()==0
assert late.gross_return_total.gt(0).sum()==11 and late.net_return_total.gt(0).sum()==0
for row in training.itertuples():
    assert pd.Timestamp(row.train_label_last_date)<pd.Timestamp(row.validation_start)
    assert pd.Timestamp(row.validation_label_last_date)<pd.Timestamp(row.test_start)
assert training.loc[training.best_epoch.gt(0),'fold'].tolist()==[3]
for path,key in [('source/train_tw.py','script_sha256'),('source/models.py','models_sha256'),('model/JPX_ENSEMBLE_initial.pt','weights_sha256')]:
    assert sha(BASE/path)==env['signature'][key]
assert sha(BASE/'model/TW_ENSEMBLE_final.pt')==read('completed.json')['final_weight_sha256']
manifest=BASE/'manifest.json'
if manifest.exists():
    for item in json.loads(manifest.read_text())['files']:
        file=BASE/item['path'];assert file.stat().st_size==item['bytes']
        assert sha(file)==item['sha256'],item['path']
if args.check_model:
    import torch
    initial=torch.load(BASE/'model/JPX_ENSEMBLE_initial.pt',map_location='cpu',weights_only=True)
    final=torch.load(BASE/'model/TW_ENSEMBLE_final.pt',map_location='cpu',weights_only=True)
    changed=[key for key,value in final['model_state_dict'].items() if not torch.equal(value,initial['model_state_dict'][key])]
    assert len(changed)==16
print('PASS: both periods, all 40 folds, costs, label purges, checkpoint/source hashes and publication manifest.')
