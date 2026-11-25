"""Kaggle-pretrained ENSEMBLE -> Taiwan expanding-window validation/backtest.

Historical research using user-supplied prices. The portfolio is a Target-return
proxy: it does not establish borrow availability, fillability, or total returns.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from models import ENSEMBLE

ROOT = Path(__file__).resolve().parent
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'ExpectedDividend']
EXPECTED_WEIGHT_HASH = '0799f7efc93712944624417d2054a922e36849d5c57df754ddae559b8bd6ff0e'
CONFIG = dict(experiment='TW_ENSEMBLE_transfer_20261125_kaggle_v1', lookback=21,
              hidden_size=128, num_layers=1, dropout=0.2, seed=20261125,
              train_start='2017-01-03', test_start='2017-04-03', step_days=90,
              inner_validation_days=30, max_epochs=50, patience=10,
              min_delta=1e-5, lr=0.001, batch_size=4096, shuffle=False,
              feature_scaler='frozen_original_JPX', precision='float32_tf32', multi_gpu='DataParallel_when_available',
              portfolio='top3_bottom3_linear_2_to_1_gross_1',
              source_notebook='https://www.kaggle.com/code/aiaila101/jpx-ensemble-norelu-top3-gap90',
              source_version=1, source_train_last_date='2021-03-08',
              source_validation_last_date='2021-06-08',
              commission_one_way=0.001425, sell_tax=0.003,
              slippage_one_way=0.0005, short_borrow_annual=0.03)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def clean_json(x):
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean_json(x.tolist())
    if isinstance(x, np.generic):
        return clean_json(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, (Path, pd.Timestamp)):
        return str(x)
    return x


def json_save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(temp, path)


def torch_save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def prepare(data_path, checkpoint, out):
    print('PREPARING_DATA', flush=True)
    if str(data_path).endswith('.zip'):
        with zipfile.ZipFile(data_path) as z:
            assert z.namelist() == ['stock_price_final.csv']
            assert z.testzip() is None
            with z.open('stock_price_final.csv') as f:
                raw = pd.read_csv(f)
    elif str(data_path).endswith('.parquet'):
        raw = pd.read_parquet(data_path)
    else:
        raw = pd.read_csv(data_path)
    raw['Date'] = pd.to_datetime(raw['Date'])
    assert not raw.duplicated(['Date', 'SecuritiesCode']).any()
    raw = raw.sort_values(['SecuritiesCode', 'Date']).reset_index(drop=True)
    assert raw.Date.min() == pd.Timestamp(CONFIG['train_start'])
    assert (raw.AdjustmentFactor > 0).all()
    g = raw.groupby('SecuritiesCode', sort=False)
    calendar = np.sort(raw.Date.unique())
    dnum = np.searchsorted(calendar, raw.Date.to_numpy())
    label_end = g.Date.shift(-2)
    label_entry = g.Date.shift(-1)
    # The data's Target is exactly Close[t+2] / Close[t+1] / AF[t+2] - 1.
    recomputed = g.Close.shift(-2) / g.Close.shift(-1) / g.AdjustmentFactor.shift(-2) - 1
    comparable = raw.Target.notna() & recomputed.notna()
    match = np.isclose(raw.Target, recomputed, rtol=1e-5, atol=1e-8)
    assert match[comparable].all(), 'Target definition differs from audited data'
    label_consecutive = ((np.searchsorted(calendar, label_end.to_numpy()) == dnum + 2)
                         & (np.searchsorted(calendar, label_entry.to_numpy()) == dnum + 1))
    raw['LabelEnd'] = label_end
    raw['LabelEntry'] = label_entry
    raw['LabelConsecutive'] = label_consecutive
    raw['ExpectedDividend'] = raw.ExpectedDividend.fillna(0.)
    values = raw[FEATURES].to_numpy(dtype=np.float64, copy=True)
    feature_valid = (np.isfinite(values).all(axis=1)
                     & (values[:, :4] > 0).all(axis=1) & (values[:, 4] >= 0))
    # Preserve the source model's causal adjustment convention (prior cumprod).
    cumulative = g.AdjustmentFactor.cumprod()
    prior = cumulative.groupby(raw.SecuritiesCode, sort=False).shift(1, fill_value=1.)
    assert np.isfinite(prior).all() and (prior > 0).all()
    values[:, :4] /= prior.to_numpy()[:, None]
    values[:, 4] *= prior.to_numpy()
    mu = np.asarray(checkpoint['scaler']['mean'], dtype=np.float64)
    sd = np.asarray(checkpoint['scaler']['std'], dtype=np.float64)
    values = ((values - mu) / sd).astype(np.float32)
    values[~feature_valid] = 0.  # Such rows can never be in an eligible window.
    n = CONFIG['lookback']
    idx = np.arange(len(raw))
    begin = np.maximum(idx - n + 1, 0)
    bad_prefix = np.r_[0, np.cumsum(~feature_valid)]
    codes = raw.SecuritiesCode.to_numpy()
    window_valid = ((idx >= n - 1) & (codes == codes[begin])
                    & (dnum - dnum[begin] == n - 1)
                    & (bad_prefix[idx + 1] - bad_prefix[begin] == 0))
    # Selection uses only information known at the signal-date close.
    raw['SignalEligible'] = window_valid & (raw.Volume.to_numpy() > 0)
    raw['LearnEligible'] = (raw.SignalEligible & np.isfinite(raw.Target)
                            & raw.LabelConsecutive & raw.LabelEnd.notna())
    anomalies = raw.loc[raw.Target.abs() > .20,
                        ['Date', 'SecuritiesCode', 'Close', 'AdjustmentFactor', 'Target', 'LabelEnd']]
    anomalies.to_csv(out / 'target_anomalies_over_20pct.csv', index=False)
    audit = dict(rows=len(raw), stocks=raw.SecuritiesCode.nunique(), trading_dates=len(calendar),
                 first_date=raw.Date.min(), last_date=raw.Date.max(), duplicate_stock_dates=0,
                 source_data_sha256=sha256(data_path), source_weight_sha256=EXPECTED_WEIGHT_HASH,
                 missing_ohlc_rows=int((~feature_valid).sum()),
                 missing_targets=int(raw.Target.isna().sum()),
                 expected_dividend_policy='All source rows missing; fill with zero as original code does. No dividend total-return claim.',
                 target_formula='Close[t+2]/Close[t+1]/AdjustmentFactor[t+2]-1, per stock row',
                 verified_target_matches=int(comparable.sum()), verified_target_mismatches=0,
                 target_min=raw.Target.min(), target_max=raw.Target.max(),
                 target_abs_over_20pct=len(anomalies),
                 target_abs_over_50pct=int((raw.Target.abs() > .5).sum()),
                 feature_adjustment='OHLC / prior cumulative factor; Volume * prior cumulative factor (original source convention)',
                 signal_samples=int(raw.SignalEligible.sum()), learn_samples=int(raw.LearnEligible.sum()),
                 universe='All supplied codes as observed on each day; completeness of delisted history is unverified',
                 limitations=['Corporate-action completeness and extreme price moves are unverified.',
                              'No historical shortability, limit-lock, auction fill, or delisting cash-flow data.',
                              'Research backtest conditional on supplied data; no claim of live executable performance.'])
    json_save(out / 'data_audit.json', audit)
    return raw, values


def folds_for(last_date):
    start = pd.Timestamp(CONFIG['test_start'])
    stop = pd.Timestamp(last_date) + pd.Timedelta(days=1)
    result = []
    while start < stop:
        end = min(start + pd.Timedelta(days=CONFIG['step_days']), stop)
        result.append(dict(fold=len(result) + 1,
                           validation_start=start - pd.Timedelta(days=CONFIG['inner_validation_days']),
                           test_start=start, test_end_exclusive=end))
        start = end
    return result


def split_indices(raw, fold):
    vs, ts, te = (pd.Timestamp(fold[k]) for k in
                  ('validation_start', 'test_start', 'test_end_exclusive'))
    train = np.flatnonzero((raw.Date < vs) & (raw.LabelEnd < vs) & raw.LearnEligible)
    val = np.flatnonzero((raw.Date >= vs) & (raw.Date < ts) & (raw.LabelEnd < ts) & raw.LearnEligible)
    test = np.flatnonzero((raw.Date >= ts) & (raw.Date < te) & raw.SignalEligible)
    assert len(train) and len(val) and len(test)
    assert raw.iloc[train].LabelEnd.max() < raw.iloc[val].Date.min()
    assert raw.iloc[val].LabelEnd.max() < raw.iloc[test].Date.min()
    assert not np.intersect1d(train, val).size and not np.intersect1d(val, test).size
    return train, val, test


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def run_batches(model, x, y, indices, batch_size, optimizer=None):
    training = optimizer is not None
    model.train(training)
    offsets = torch.arange(1 - CONFIG['lookback'], 1, device=x.device)
    total = torch.zeros((), device=x.device)
    with torch.set_grad_enabled(training):
        for start in range(0, len(indices), batch_size):
            end_idx = indices[start:start + batch_size]
            xb = x[end_idx[:, None] + offsets[None, :]]
            pred = model(xb).squeeze(-1)
            loss = torch.nn.functional.huber_loss(pred, y[end_idx], delta=1., reduction='mean')
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            total += loss.detach() * len(end_idx)
    return float(total.item() / len(indices))


def predict(model, x, indices, batch_size):
    model.eval()
    offsets = torch.arange(1 - CONFIG['lookback'], 1, device=x.device)
    values = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start + batch_size]
            values.append(model(x[ids[:, None] + offsets]).squeeze(-1).cpu().numpy())
    result = np.concatenate(values)
    assert np.isfinite(result).all()
    return result


def metric_series(returns):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if not len(r):
        return dict(days=0)
    sd = np.std(r, ddof=1) if len(r) > 1 else 0.
    can_compound = bool((r > -1).all())
    eq = np.r_[1., np.cumprod(1. + r)] if can_compound else None
    return dict(days=len(r), mean_daily=r.mean(), additive_return=r.sum(),
                annualized_sharpe=float(r.mean() / sd * np.sqrt(252)) if sd > 0 else None,
                positive_day_fraction=float((r > 0).mean()),
                compounded_return=float(eq[-1] - 1) if can_compound else None,
                cagr_252=float(eq[-1] ** (252 / len(r)) - 1) if can_compound else None,
                max_drawdown=float(np.min(eq / np.maximum.accumulate(eq) - 1)) if can_compound else None,
                min_daily=r.min(), max_daily=r.max())


def backtest(predictions):
    """Pick names before looking at targets. Missing selected labels invalidate a day.

    Explicit one-day close-to-close tranches, fully closed/reopened every day.
    Cost sensitivity charges both trades (no turnover netting), including changes
    in closing notional. It is not an execution simulation.
    """
    positions, days, ic = [], [], []
    w = np.linspace(2., 1., 3)
    w = w / w.sum()
    for date, frame in predictions.groupby('Date', sort=True):
        usable = frame.loc[frame.LabelConsecutive & frame.Target.notna()]
        ric = usable.Prediction.rank().corr(usable.Target.rank()) if len(usable) > 2 else np.nan
        ic.append(dict(Date=date, rank_ic=ric, eligible=len(frame), labels=len(usable)))
        if len(frame) < 6:
            days.append(dict(Date=date, complete=False, reason='fewer_than_6_candidates'))
            continue
        ranked = frame.sort_values(['Prediction', 'SecuritiesCode'], ascending=[False, True])
        long = ranked.head(3).copy()
        short = ranked.tail(3).sort_values(['Prediction', 'SecuritiesCode'], ascending=[True, True]).copy()
        long['Weight'], short['Weight'] = .5 * w, -.5 * w
        long['Side'], short['Side'] = 'long', 'short'
        picks = pd.concat([long, short], ignore_index=True)
        complete = bool(picks.Target.notna().all() and picks.LabelConsecutive.all())
        picks['CompleteDay'] = complete
        positions.append(picks)
        rec = dict(Date=date, fold=int(frame.fold.iloc[0]), complete=complete,
                   selected_extreme_labels=int((picks.Target.abs() > .2).sum()),
                   missing_selected_labels=int(picks.Target.isna().sum()),
                   reason='' if complete else 'selected_position_has_missing_or_nonconsecutive_label')
        if complete:
            r = picks.Target.to_numpy(float)
            weights = picks.Weight.to_numpy(float)
            abs_w = np.abs(weights)
            buy = float(abs_w[weights > 0].sum() + np.sum(abs_w[weights < 0] * (1 + r[weights < 0])))
            sell = float(abs_w[weights < 0].sum() + np.sum(abs_w[weights > 0] * (1 + r[weights > 0])))
            days_held = (picks.LabelEnd - picks.LabelEntry).dt.days.to_numpy()
            borrow = float(np.sum(abs_w[weights < 0] * CONFIG['short_borrow_annual'] * days_held[weights < 0] / 365))
            costs = ((buy + sell) * (CONFIG['commission_one_way'] + CONFIG['slippage_one_way'])
                     + sell * CONFIG['sell_tax'] + borrow)
            gross = float(weights @ r)
            rec.update(gross=gross, cost_scenario=costs, net_cost_scenario=gross - costs,
                       long_only_gross=float(w @ long.Target.to_numpy()),
                       clipped_20pct_sensitivity=float(weights @ np.clip(r, -.2, .2)),
                       equal_weight_available_labels=float(usable.Target.mean()),
                       realized_date=picks.LabelEnd.max())
        days.append(rec)
    return pd.DataFrame(days), pd.concat(positions, ignore_index=True), pd.DataFrame(ic)


def write_report(out, complete=False):
    paths = sorted(out.glob('fold_*_predictions.parquet'))
    if not paths:
        return
    pred = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    assert not pred.duplicated(['Date', 'SecuritiesCode']).any()
    daily, positions, ics = backtest(pred)
    daily.to_csv(out / 'backtest_daily.csv', index=False)
    positions.to_csv(out / 'selected_positions.csv.gz', index=False, compression='gzip')
    ics.to_csv(out / 'daily_rank_ic.csv', index=False)
    scored = daily.loc[daily.complete]
    metric_names = ['gross', 'net_cost_scenario', 'long_only_gross', 'clipped_20pct_sensitivity', 'equal_weight_available_labels']
    metrics = {k: metric_series(scored[k]) for k in metric_names}
    late = scored.loc[scored.Date >= pd.Timestamp('2021-07-01')]
    late_pred = pred.loc[(pred.Date >= pd.Timestamp('2021-07-01')) & pred.LabelConsecutive & pred.Target.notna()]
    late_ic = ics.loc[ics.Date >= pd.Timestamp('2021-07-01')]
    late_metrics = {k: metric_series(late[k]) for k in metric_names}
    valid = pred.LabelConsecutive & pred.Target.notna()
    report = dict(status='complete' if complete else 'partial', completed_folds=len(paths),
                  first_signal_date=pred.Date.min(), last_signal_date=pred.Date.max(),
                  signal_dates=daily.Date.nunique(), complete_dates=len(scored),
                  incomplete_dates=int((~daily.complete).sum()),
                  prediction_rows=len(pred), labeled_prediction_rows=int(valid.sum()),
                  mae=float(np.abs(pred.loc[valid, 'Prediction'] - pred.loc[valid, 'Target']).mean()),
                  zero_baseline_mae=float(pred.loc[valid, 'Target'].abs().mean()),
                  mean_daily_rank_ic=ics.rank_ic.mean(), portfolios=metrics,
                  selected_extreme_labels=int(daily.selected_extreme_labels.fillna(0).sum()),
                  conclusions={
                    'all_validation': dict(first_signal_date=pred.Date.min(), last_signal_date=pred.Date.max(),
                       complete_dates=len(scored), incomplete_dates=int((~daily.complete).sum()), portfolios=metrics,
                       mean_daily_rank_ic=ics.rank_ic.mean(),
                       interpretation='Taiwan labels are held out chronologically, but the fixed Japanese pretrained checkpoint contains training/validation information through 2021. Pre-July-2021 results are retrospective transfer diagnostics, not historically available-model performance.'),
                    'from_2021_H2': dict(cutoff='2021-07-01', complete_dates=len(late),
                       incomplete_dates=int(((daily.Date >= pd.Timestamp('2021-07-01')) & ~daily.complete).sum()),
                       portfolios=late_metrics, mean_daily_rank_ic=late_ic.rank_ic.mean(),
                       mae=float((late_pred.Prediction - late_pred.Target).abs().mean()),
                       zero_baseline_mae=float(late_pred.Target.abs().mean()),
                       interpretation='Exact signal-date slice from the same expanding-window run, after known Japanese pretraining/validation dates. Architecture and strategy still selected retrospectively; dataset limitations remain.')},
                  interpretation='Two conclusions: full historical validation and signal dates from 2021-07-01. Japanese pretrained checkpoint has later information than the early historical period.',
                  cost_note='Cost scenario only: commission 0.1425% each side, sell tax 0.3%, slippage 5bp each side, short borrow 3%/year. All daily tranches fully opened/closed; no turnover netting. Borrow and fills unverified.',
                  missing_label_note='Picks are chosen without target availability filtering. Entire days with unresolved selected labels are excluded and explicitly counted; reported Sharpe/CAGR are conditional on complete days.',
                  dividend_note='ExpectedDividend is entirely missing; price-return target excludes cash dividends.',
                  anomaly_note='Extreme source targets are retained in the main result. Clipping at +/-20% is a separately labeled sensitivity, not a data repair.')
    json_save(out / 'backtest_summary.json', report)
    json_save(out / 'conclusion_all_validation.json', report['conclusions']['all_validation'])
    json_save(out / 'conclusion_from_2021_H2.json', report['conclusions']['from_2021_H2'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for col, label in [('gross', '3 long / 3 short: gross'),
                       ('net_cost_scenario', '3 / 3: cost scenario'),
                       ('clipped_20pct_sensitivity', '3 / 3: +/-20% label sensitivity')]:
        r = scored[col].to_numpy()
        axes[0].plot(scored.Date, np.cumsum(r), label=label)
        if (r > -1).all():
            eq = np.cumprod(1 + r)
            axes[1].plot(scored.Date, eq / np.maximum.accumulate(np.r_[1., eq])[1:] - 1, label=label)
    axes[0].set_title('Taiwan ENSEMBLE - expanding-window historical research')
    axes[0].set_ylabel('Cumulative additive return')
    axes[1].set_ylabel('Drawdown (compounded proxy)')
    axes[0].legend(); axes[0].grid(alpha=.2); axes[1].grid(alpha=.2)
    fig.text(.01, .01, f'Complete days only; {int((~daily.complete).sum())} unresolved days excluded. No shortability/fill/dividend validation.', fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(out / 'backtest.png', dpi=160); plt.close(fig)
    rows = [{'period': period, 'portfolio': k, **v} for period, mm in
            [('all_validation', metrics), ('from_2021_H2', late_metrics)] for k, v in mm.items()]
    table = pd.DataFrame(rows)
    table.to_csv(out / 'backtest_metrics.csv', index=False)
    (out / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Taiwan ENSEMBLE</title>'
        '<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:20px}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}img{width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>Taiwan ENSEMBLE expanding-window backtest</h1><p>Status: ' + report['status'] + '</p>'
        '<p>' + report['interpretation'] + '</p><img src="backtest.png">'
        + table.to_html(index=False, float_format=lambda v: f'{v:.6f}')
        + '<pre>' + json.dumps(clean_json(report), indent=2) + '</pre>', encoding='utf-8')
    print('BACKTEST_SUMMARY', json.dumps(clean_json(report)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'stock_price_final.parquet')
    parser.add_argument('--weights', type=Path, default=ROOT / 'JPX_ENSEMBLE_final.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--max-seconds', type=float, default=36000)
    args = parser.parse_args()
    started_at = time.monotonic()
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    assert sha256(args.weights) == EXPECTED_WEIGHT_HASH
    cp = torch.load(args.weights, map_location='cpu', weights_only=True)
    assert cp['config']['features'] == FEATURES and cp['config']['lookback'] == 21
    raw, values = prepare(args.data, cp, out)
    folds = folds_for(raw.Date.max())
    json_save(out / 'fold_schedule.json', folds)
    for fold in folds:
        split_indices(raw, fold)
    if args.audit_only:
        print('AUDIT_AND_ALL_SPLIT_ASSERTIONS_PASSED', flush=True)
        return
    assert args.smoke_test or args.device == 'cuda', 'Full training requires CUDA'
    assert args.device != 'cuda' or torch.cuda.is_available(), 'No CUDA GPU allocated'
    random.seed(CONFIG['seed']); np.random.seed(CONFIG['seed']); torch.manual_seed(CONFIG['seed'])
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG['seed'])
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    model = ENSEMBLE(6, 128, 1, device=str(device))
    model.load_state_dict(cp['model_state_dict'], strict=True)
    assert all(torch.equal(model.state_dict()[k].cpu(), v) for k, v in cp['model_state_dict'].items())
    print('STRICT_PRETRAINED_WEIGHT_LOAD_PASS', EXPECTED_WEIGHT_HASH, flush=True)
    execution_model = torch.nn.DataParallel(model) if device.type == 'cuda' and torch.cuda.device_count() > 1 else model
    print('EXECUTION_GPUS', torch.cuda.device_count() if device.type == 'cuda' else 0, flush=True)
    x = torch.as_tensor(values, device=device)
    y = torch.as_tensor(raw.Target.to_numpy(dtype=np.float32, copy=True), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    signature = dict(config=CONFIG, data_sha256=sha256(args.data), weights_sha256=EXPECTED_WEIGHT_HASH,
                     script_sha256=sha256(__file__), models_sha256=sha256(ROOT / 'models.py'))
    json_save(out / 'environment.json', dict(signature=signature, python=sys.version, torch=str(torch.__version__),
         numpy=np.__version__, pandas=pd.__version__, cuda=torch.version.cuda,
         gpu_count=torch.cuda.device_count() if device.type == 'cuda' else 0,
         gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if device.type == 'cuda' else [],
         gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else str(device),
         gpu_memory_gib=torch.cuda.get_device_properties(0).total_memory / 2**30 if device.type == 'cuda' else 0,
         optimizer_policy='New Adam on Taiwan transfer; model+optimizer best state carried between Taiwan folds.',
         modifications_from_kaggle='Separate early-stop validation and test; batch 4096; fixed source scaler; gradient norm clip 1; TF32; original architecture/loss/lr/max epochs/patience.'))
    if args.smoke_test:
        train, val, test = split_indices(raw, folds[0])
        if isinstance(execution_model, torch.nn.DataParallel):
            model.eval()
            with torch.no_grad():
                probe_ids = torch.as_tensor(train[:128], device=device)
                probe = x[probe_ids[:, None] + torch.arange(-20, 1, device=device)]
                torch.testing.assert_close(execution_model(probe), model(probe), rtol=1e-4, atol=1e-5)
            print('MULTI_GPU_INFERENCE_PARITY_PASS', flush=True)
        before = cpu_state(model)
        train = torch.as_tensor(train[:256], device=device)
        val = torch.as_tensor(val[:64], device=device)
        print('SMOKE_TRAIN_LOSS', run_batches(execution_model, x, y, train, 64, optimizer), flush=True)
        print('SMOKE_VAL_LOSS', run_batches(execution_model, x, y, val, 64), flush=True)
        assert any(not torch.equal(before[k], v.cpu()) for k, v in model.state_dict().items())
        print('SMOKE_PREDICTIONS', predict(execution_model, x, val, 64)[:5], flush=True)
        json_save(out / 'smoke_test.json', dict(passed=True, training_samples=256, validation_samples=64))
        return
    resume_path = out / 'resume_latest.pt'
    state = dict(next_fold=0, epoch=0, best_loss=None, bad_epochs=0, best_model=None, best_optimizer=None)
    if resume_path.exists():
        saved = torch.load(resume_path, map_location='cpu', weights_only=True)
        assert saved['signature'] == signature, 'Resume inputs/config/code differ'
        state = saved['state']
        model.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng_cpu'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['rng_cuda'])
        print('RESUME', state['next_fold'] + 1, state['epoch'], flush=True)

    def save():
        torch_save(resume_path, dict(signature=signature, state=state, model=cpu_state(model),
                    optimizer=copy.deepcopy(optimizer.state_dict()), rng_cpu=torch.get_rng_state(),
                    rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else []))

    for fi in range(state['next_fold'], len(folds)):
        fold = folds[fi]
        tr, va, te = split_indices(raw, fold)
        train, val, test = [torch.as_tensor(a, device=device) for a in (tr, va, te)]
        print('FOLD', fi + 1, '/', len(folds), clean_json(fold), 'samples', len(tr), len(va), len(te), flush=True)
        if state['epoch'] == 0:
            state.update(best_loss=run_batches(execution_model, x, y, val, CONFIG['batch_size']),
                         best_model=cpu_state(model), best_optimizer=copy.deepcopy(optimizer.state_dict()),
                         best_epoch=0, bad_epochs=0)
            save()
        for epoch in range(state['epoch'] + 1, CONFIG['max_epochs'] + 1):
            if state['bad_epochs'] >= CONFIG['patience']:
                break
            if time.monotonic() - started_at >= args.max_seconds:
                save()
                write_report(out, complete=False)
                json_save(out / 'time_budget_reached.json', dict(fold=fi + 1, epoch=state['epoch']))
                print('TIME_BUDGET_REACHED_CHECKPOINT_SAVED', flush=True)
                return
            t0 = time.monotonic()
            tl = run_batches(execution_model, x, y, train, CONFIG['batch_size'], optimizer)
            vl = run_batches(execution_model, x, y, val, CONFIG['batch_size'])
            improved = vl < state['best_loss'] - CONFIG['min_delta']
            if improved:
                state.update(best_loss=vl, best_model=cpu_state(model),
                             best_optimizer=copy.deepcopy(optimizer.state_dict()), best_epoch=epoch, bad_epochs=0)
            else:
                state['bad_epochs'] += 1
            state['epoch'] = epoch
            row = dict(fold=fi + 1, epoch=epoch, train_huber=tl, validation_huber=vl,
                       best_validation_huber=state['best_loss'], best_epoch=state['best_epoch'],
                       bad_epochs=state['bad_epochs'], seconds=time.monotonic() - t0)
            with (out / 'epoch_history.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            json_save(out / 'progress.json', row)
            save()
            print('EPOCH', json.dumps(row), flush=True)
        model.load_state_dict(state['best_model'], strict=True)
        optimizer.load_state_dict(state['best_optimizer'])
        prediction = predict(execution_model, x, test, CONFIG['batch_size'])
        pp = raw.iloc[te][['Date', 'SecuritiesCode', 'Target', 'LabelEnd', 'LabelEntry', 'LabelConsecutive']].copy()
        pp['Prediction'] = prediction; pp['fold'] = fi + 1
        pp.to_parquet(out / f'fold_{fi+1:02d}_predictions.parquet', index=False)
        json_save(out / f'fold_{fi+1:02d}_report.json', dict(**fold,
             train_signal_last_date=raw.iloc[tr].Date.max(), train_label_last_date=raw.iloc[tr].LabelEnd.max(),
             validation_signal_last_date=raw.iloc[va].Date.max(), validation_label_last_date=raw.iloc[va].LabelEnd.max(),
             train_samples=len(tr), validation_samples=len(va), test_samples=len(te),
             epochs=state['epoch'], best_epoch=state['best_epoch'], validation_huber=state['best_loss']))
        torch_save(out / f'fold_{fi+1:02d}_best.pt', dict(model_state_dict=cpu_state(model), scaler=cp['scaler'],
                                                       config=CONFIG, fold=clean_json(fold)))
        state.update(next_fold=fi + 1, epoch=0, best_loss=None, bad_epochs=0, best_model=None, best_optimizer=None)
        save()
        write_report(out, complete=fi + 1 == len(folds))
    torch_save(out / 'TW_ENSEMBLE_final.pt', dict(model_state_dict=cpu_state(model), scaler=cp['scaler'], config=CONFIG))
    json_save(out / 'completed.json', dict(folds_completed=len(folds),
               completed_at_utc=pd.Timestamp.now(tz='UTC'), final_weight_sha256=sha256(out / 'TW_ENSEMBLE_final.pt')))
    print('TRAINING_AND_BACKTEST_COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
