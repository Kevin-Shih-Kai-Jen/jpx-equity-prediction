"""Post-analysis of frozen JPX predictions; no training or checkpoint mutation."""
from pathlib import Path
import sys, json, hashlib, math
import numpy as np
import pandas as pd

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', required=True, type=Path, help='Frozen repository or restored completed-run archive')
parser.add_argument('--data', required=True, type=Path, help='Original stock_prices.csv with the recorded SHA256')
parser.add_argument('--results', required=True, type=Path, help='Completed run directory containing predictions and audits')
parser.add_argument('--output-dir', required=True, type=Path, help='New directory for recomputed analysis; existing directory rejected')
args = parser.parse_args()
ROOT = args.repo.resolve()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=False)
RUN = args.results.resolve()
manifest = json.loads((Path(__file__).resolve().parent / 'artifacts.json').read_text())
for source, expected in manifest['source_sha256'].items():
    if source.startswith('codes/'):
        with (ROOT / source).open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == expected, source
sys.path.insert(0, str(ROOT / 'codes'))
import data_loader, data_processor, features, utils, evaluation

records = json.loads((RUN / 'fold_results.json').read_text())
audit = json.loads((RUN / 'completion_audit.json').read_text())
assert len(records) == 8 and audit['status'] == 'passed'
data_path = args.data.resolve()
assert hashlib.file_digest(data_path.open('rb'), 'sha256').hexdigest() == 'bf774a86f834e5338bba74f2356ebb750f93a1cc4e89b064ee07251e8ed851db'
df = data_loader.main_loader(data_path, show_basic_data=False)
df['ExpectedDividend'] = df['ExpectedDividend'].fillna(0)
cols = ['SecuritiesCode','Date','Open','High','Low','Close','Volume','AdjustmentFactor','Target']
valid = features.get_valid_rows(df, set(cols)).index
df = data_processor.correct_prices_with_factors(df).loc[valid].copy()
print('Original row validation and price adjustment completed', flush=True)
all_daily, summaries = [], []
previous_weights = pd.Series(dtype='float64')
total_abs_error = total_zero_error = total_samples = 0

for r in records:
    fold = r['fold']
    start, train_end, val_start, end = r['common_process_context']['split_dates']
    train = data_processor.get_time_for_specific_period(df, start, train_end)
    val = data_processor.get_time_for_specific_period(df, val_start, end)
    val = utils.compensate_val_lost_time(train, val, 42)
    predictions = np.load(RUN / f'fold_{fold:02d}_predictions.npy', allow_pickle=False)
    ev = evaluation.get_eval_df(val, predictions, 42)
    assert len(ev) == r['validation_samples']
    assert ev['Date'].min() >= pd.Timestamp(val_start)
    assert ev['Date'].max() < pd.Timestamp(end)
    y = ev['Target'].to_numpy(dtype='float32')
    errors = np.abs(predictions - y).astype('float64')
    zero_errors = np.abs(y).astype('float64')
    mae = float(errors.mean()); zero_mae = float(zero_errors.mean())
    assert np.isclose(mae, r['validation_average_loss'], atol=2e-9, rtol=0)
    total_abs_error += errors.sum(); total_zero_error += zero_errors.sum(); total_samples += len(y)
    days = []
    for date, day in ev.groupby('Date'):
        weighted = evaluation.calculate_weights(day.copy())
        weights = weighted.set_index('SecuritiesCode')['Weight']
        turnover = float(weights.sub(previous_weights, fill_value=0).abs().sum())
        previous_weights = weights[weights != 0]
        ret = float(np.float32((weighted['Weight'] * weighted['Target']).sum()))
        ic = float(day['Prediction'].corr(day['Target'], method='spearman'))
        days.append({'date':str(date.date()),'fold':fold,'spread_return':ret,'rank_ic':ic,
                     'absolute_target_weight_change':turnover,'stocks':len(day),
                     'unique_predictions':int(day['Prediction'].nunique()),
                     'prediction_std':float(day['Prediction'].std()),
                     'prediction_unique_ratio':float(day['Prediction'].nunique()/len(day))})
    daily = pd.DataFrame(days)
    rv = daily['spread_return'].to_numpy(dtype='float32')
    metrics = {'earnings':float(rv.sum()),'sharpe':float(rv.mean()/rv.std(ddof=1)),
               'rank_IC':float(daily['rank_ic'].mean())}
    for k,v in metrics.items():
        assert np.isclose(v, r['original_metrics'][k], atol=1e-9, rtol=0), (fold,k,v,r['original_metrics'][k])
    curve = np.r_[0, np.cumsum(rv.astype('float64'))]
    summary = {'fold':fold,'start':daily['date'].iloc[0],'end':daily['date'].iloc[-1],
        'days':len(daily),'samples':len(y),'mae':r['validation_average_loss'],'zero_mae':zero_mae,
        'mae_improvement_vs_zero':1-mae/zero_mae,**metrics,
        'mean_daily_bps':float(rv.astype('float64').mean()*1e4),
        'daily_std_bps':float(rv.astype('float64').std(ddof=1)*1e4),
        'positive_day_fraction':float((rv>0).mean()),
        'annualized_sharpe_252_iid_reference':metrics['sharpe']*math.sqrt(252),
        'cumulative_additive_max_fall':float((np.maximum.accumulate(curve)-curve).max()),
        'mean_target_weight_change':float(daily['absolute_target_weight_change'].mean()),
        'cost_break_even_bps_on_weight_change':float(rv.astype('float64').sum()/daily['absolute_target_weight_change'].sum()*1e4),
        'undefined_ic_days':int(daily['rank_ic'].isna().sum()),
        'constant_prediction_days':int((daily['unique_predictions']==1).sum()),
        'mean_unique_prediction_ratio':float(daily['prediction_unique_ratio'].mean())}
    summaries.append(summary); all_daily += days
    print(json.dumps(summary),flush=True)

daily = pd.DataFrame(all_daily)
assert not daily['date'].duplicated().any()
assert daily['date'].is_monotonic_increasing
r = daily['spread_return'].to_numpy('float64'); ic = daily['rank_ic'].to_numpy('float64')
w = daily['absolute_target_weight_change'].to_numpy('float64')
# Include closing the final portfolio at the end of the sensitivity scenario.
w_close = w.copy(); w_close[-1] += float(previous_weights.abs().sum())
daily['target_weight_change_including_final_close'] = w_close
daily['cumulative_spread_sum'] = np.cumsum(r)
curve = np.r_[0, np.cumsum(r)]
peak = np.maximum.accumulate(curve)
daily['cumulative_additive_fall'] = (peak-curve)[1:]
costs = []
for bps in [0,1,2,3,5,10]:
    net = r - bps*1e-4*w_close
    costs.append({'cost_bps_per_traded_notional':bps,'sum_returns':float(net.sum()),
        'mean_daily_bps':float(net.mean()*1e4),'sharpe_unannualized':float(net.mean()/net.std(ddof=1))})
monthly = daily.assign(month=pd.to_datetime(daily['date']).dt.to_period('M').astype(str)).groupby('month',sort=True).agg(
    spread_sum=('spread_return','sum'),days=('date','count'),rank_ic=('rank_ic','mean'),
    cumulative_end=('cumulative_spread_sum','last'))
monthly.to_csv(OUT/'monthly_results.csv')
daily.to_csv(OUT/'daily_results.csv',index=False)
pd.DataFrame(summaries).to_csv(OUT/'fold_analysis.csv',index=False)
pd.DataFrame(costs).to_csv(OUT/'cost_sensitivity.csv',index=False)
report = {'method':'Post-analysis of the original saved predictions, using original row validation, date splits, 41-row history compensation, sorting and portfolio weighting. Normalization omitted only because it changes feature columns but not Target, Date, SecuritiesCode or row ordering. Every fold original MAE, earnings, Sharpe and Rank IC is cross-checked. No retraining.',
  'source_commit':audit['repository_commit'],'data_sha256':'bf774a86f834e5338bba74f2356ebb750f93a1cc4e89b064ee07251e8ed851db',
  'run_audit':{k:v for k,v in audit.items() if k != 'final_original_checkpoint'},'folds':summaries,'overall':{'start':daily['date'].iloc[0],'end':daily['date'].iloc[-1],
    'days':len(r),'samples':int(total_samples),'original_total_earnings':sum(s['earnings'] for s in summaries),
    'daily_double_precision_sum':float(r.sum()),'weighted_mae':float(total_abs_error/total_samples),
    'weighted_zero_mae':float(total_zero_error/total_samples),'mae_improvement_vs_zero':float(1-total_abs_error/total_zero_error),
    'daily_mean_bps':float(r.mean()*1e4),'daily_std_bps':float(r.std(ddof=1)*1e4),
    'sharpe_unannualized_pooled':float(r.mean()/r.std(ddof=1)),
    'annualized_sharpe_252_iid_reference':float(r.mean()/r.std(ddof=1)*math.sqrt(252)),
    'daily_rank_ic_mean':float(np.nanmean(ic)),'positive_day_fraction':float((r>0).mean()),
    'additive_max_fall':float((peak-curve).max()),'worst_day':float(r.min()),'best_day':float(r.max()),
    'mean_absolute_weight_change':float(w_close.mean()),
    'cost_break_even_bps':float(r.sum()/w_close.sum()*1e4),
    'positive_months':int((monthly['spread_sum']>0).sum()),'months':len(monthly),
    'lag1_return_autocorrelation':float(pd.Series(r).autocorr(1)),
    'undefined_ic_days':int(np.isnan(ic).sum()),
    'constant_prediction_dates':daily.loc[daily['unique_predictions']==1,['date','fold','spread_return']].to_dict('records'),
    'best_five_day_contribution':float(np.sort(r)[-5:].sum()),
    'sum_excluding_best_five_days':float(r.sum()-np.sort(r)[-5:].sum()),
    'fold_3_5_earnings_share':float((summaries[2]['earnings']+summaries[4]['earnings'])/sum(s['earnings'] for s in summaries))},
  'cost_sensitivity':costs,'cost_assumptions':'Illustrative one-way cost c multiplied by sum(abs(w_t-w_previous)), starting and ending flat. Includes long and short target-weight changes. Ignores drift, actual execution, turnover from marked-to-market weights, liquidity, borrow availability/cost and taxes. These are scenarios, not realized net returns or broker quotes.',
  'annualization_note':'Daily Sharpe times sqrt(252), illustrative IID / no-serial-correlation convention; not adjusted for autocorrelation, not a performance guarantee.',
  'uncertainty_note':'One fixed seed and one historical expanding-window path. The model and Adam continue across folds. Fold results are not eight independent experiments; no significance or future performance claim.',
  'period_note':'Results are indexed by the supplied Target date. No claim of directly tradeable intraday timing.'}
(OUT/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('OVERALL',json.dumps(report['overall']),flush=True)
print('COSTS',json.dumps(costs),flush=True)

# Recheck the original undefined-IC diagnostic from the same supplied data.
undefined = daily.loc[daily['rank_ic'].isna(), ['date','fold','unique_predictions','stocks']].to_dict('records')
report['overall']['undefined_ic_dates'] = undefined
if undefined:
    for item in undefined:
        targets = df.loc[df['Date'].eq(pd.Timestamp(item['date'])), 'Target']
        assert targets.nunique() == 1 and targets.iloc[0] == 0, item
    report['ic_note'] = 'One date has constant supplied Target (all zero), making daily Spearman undefined. Original pandas mean skips this date. No date has constant model predictions.'
(OUT/'analysis.json').write_text(json.dumps(report, ensure_ascii=True, indent=2)+'\n')
