"""Rebuild the English charts and report from the published analysis data.

Run from any working directory. This does not train a model or change the run.
"""
from pathlib import Path
import html
import json
import os
import tempfile

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data'
FIG = BASE / 'figures'
FIG.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'tw_ensemble_mpl'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from markdown_it import MarkdownIt

BLUE, RED, GRAY, GOLD = '#176c80', '#bb493e', '#708090', '#e2b86b'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
    'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
daily=pd.read_csv(DATA/'backtest_daily.csv',parse_dates=['Date','realized_date'])
good=daily.loc[daily.complete].sort_values('Date')
folds=pd.read_csv(DATA/'fold_metrics.csv')
late=pd.read_csv(DATA/'fold_metrics_from_2021_H2.csv')
years=pd.read_csv(DATA/'year_and_legs.csv')
stocks=pd.read_csv(DATA/'stock_selection.csv')
training=pd.read_csv(DATA/'fold_training_summary.csv')
summary=json.loads((DATA/'fold_summary.json').read_text())
run=json.loads((DATA/'backtest_summary.json').read_text())
env=json.loads((DATA/'environment.json').read_text())
audit=json.loads((DATA/'data_audit.json').read_text())
verified=json.loads((DATA/'independent_verification.json').read_text())


def save(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=180,facecolor='white')
    fig.savefig(FIG/f'{name}.svg',facecolor='white')
    plt.close(fig)


def fold_axis(ax):
    ax.set_xlim(.3,40.8);ax.set_xticks(range(1,41));ax.tick_params(axis='x',labelsize=8)
    ax.axhline(0,color='#8b99a3',lw=.8);ax.grid(axis='y',alpha=.16)
    ax.axvspan(33.5,36.5,color=GOLD,alpha=.15)
    ax.axvspan(39.5,40.5,color=GRAY,alpha=.13)


fig,axes=plt.subplots(2,2,figsize=(13,7),sharex='col',gridspec_kw={'height_ratios':[1.65,1]})
for j,(name,title) in enumerate([('all_validation','Full history: 2017-04-05 to 2026-11-19'),('from_2021_H2','From 2021-07-01 to 2026-11-19')]):
    f=good if j==0 else good.loc[good.Date.ge('2021-07-01')]
    for col,label,color in [('gross','3 long / 3 short: gross',BLUE),('net_cost_scenario','Cost scenario',RED)]:
        eq=np.cumprod(1+f[col].to_numpy());peak=np.maximum.accumulate(np.r_[1.,eq])[1:]
        axes[0,j].plot(f.Date,eq,label=label,color=color,lw=1.6)
        axes[1,j].plot(f.Date,eq/peak-1,color=color,lw=1.3)
    axes[0,j].axhline(1,color=GRAY,lw=.7,ls='--');axes[0,j].set_title(title,loc='left',fontsize=11,pad=12)
    axes[0,j].set_ylabel('Equity (initial capital = 1)');axes[0,j].legend(frameon=False,fontsize=9)
    axes[1,j].set_ylabel('Drawdown');axes[1,j].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes[:,j]:ax.grid(alpha=.15)
fig.suptitle('TW_ENSEMBLE | 40 completed expanding windows',x=.07,ha='left',fontsize=17,fontweight='bold')
fig.text(.07,.025,'Complete-label days only: 17 excluded in full history; 10 excluded since 2021-07-01. Costs are an assumed scenario.',fontsize=9,color='#526170')
fig.tight_layout(rect=(0,.055,1,.94));save(fig,'equity_and_drawdown')

fig,axes=plt.subplots(2,1,figsize=(15,9),sharex=True,gridspec_kw={'height_ratios':[1.3,1]})
x=folds.fold
axes[0].plot(x,folds.gross_annualized_252,'-o',color=BLUE,ms=4,lw=1.8,label='Gross, annualized')
axes[0].plot(x,folds.net_annualized_252,'-o',color=RED,ms=3,lw=1.3,label='Cost scenario, annualized')
axes[0].set_ylabel('Annualized return (252 days)');axes[0].set_ylim(-1,2.65)
axes[0].legend(loc='upper left',frameon=False,ncols=2)
for fold,shift in [(35,(-68,14)),(36,(12,12)),(32,(-85,14))]:
    row=folds.loc[folds.fold.eq(fold)].iloc[0]
    axes[0].annotate(f'F{fold}  {row.gross_annualized_252:+.1%}',(fold,row.gross_annualized_252),
        xytext=shift,textcoords='offset points',fontsize=10,color=BLUE,arrowprops=dict(arrowstyle='-',color=BLUE,lw=.7))
axes[1].bar(x-.18,folds.gross_return_total,width=.36,color=BLUE,label='Observed fold compounded return: gross')
axes[1].bar(x+.18,folds.net_return_total,width=.36,color=RED,alpha=.8,label='Observed fold compounded return: cost scenario')
axes[1].set_ylabel('Observed fold return');axes[1].set_ylim(-.55,.45)
axes[1].legend(loc='upper left',frameon=False,ncols=2,fontsize=9)
axes[1].set_xlabel('Fold number (scheduled windows span 90 calendar days; F40 has only 6 scored days)')
for ax in axes:fold_axis(ax);ax.yaxis.set_major_formatter(PercentFormatter(1))
axes[0].tick_params(labelbottom=True)
fig.suptitle('Fold returns | 19 of 40 positive gross; none positive after assumed costs',x=.065,ha='left',fontsize=18,fontweight='bold')
fig.text(.065,.925,'Gold: F34-F36, where gains are concentrated. Gray: F40, an incomplete and very short observation window.',fontsize=11,color='#526170')
fig.text(.065,.022,'Annualized = product(1 + daily return) ** (252 / scored days) - 1. This conversion is not a forecast of next-year performance.',fontsize=10,color='#526170')
fig.tight_layout(rect=(.015,.055,.995,.91));save(fig,'fold_returns')

fig,axes=plt.subplots(3,1,figsize=(15,10),sharex=True)
for ax,col,label in zip(axes,['gross_sharpe','gross_max_drawdown','mean_rank_ic'],['Gross Sharpe\n(annualized)','Within-fold\nmaximum drawdown','Mean daily\nRank IC']):
    v=folds[col];ax.bar(x,v,color=np.where(v.ge(0),BLUE,RED),width=.67);fold_axis(ax);ax.set_ylabel(label)
axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[-1].set_xlabel('Fold number')
fig.suptitle('Fold-level risk and ranking signal',x=.065,ha='left',fontsize=19,fontweight='bold')
fig.text(.065,.92,'Drawdown resets to initial capital 1 in each fold. Rank IC uses all eligible labeled cross-sectional predictions, not just six selected stocks.',fontsize=10,color='#526170')
fig.text(.065,.02,'Sharpe uses sample daily standard deviation and a zero risk-free rate. F40 has only 6 scored portfolio days; its statistics are unstable.',fontsize=10,color='#526170')
fig.tight_layout(rect=(.015,.045,.995,.90));save(fig,'fold_risk_and_ic')

fig,axes=plt.subplots(2,1,figsize=(14,9),sharex=True)
xx=np.arange(len(years));labels=[f'{int(y)}'+('*' if y in [2017,2026] else '') for y in years.year]
axes[0].bar(xx-.18,years.gross_return_total,width=.36,color=BLUE,label='Long/short gross compounded return')
axes[0].bar(xx+.18,years.net_return_total,width=.36,color=RED,label='Cost scenario compounded return')
axes[0].set_ylabel('Observed calendar-year return');axes[0].set_ylim(-1,.65)
axes[1].bar(xx-.18,years.long_additive_contribution,width=.36,color=BLUE,label='Long contribution (sum of daily returns)')
axes[1].bar(xx+.18,years.short_additive_contribution,width=.36,color='#9273a6',label='Short contribution (sum of daily returns)')
axes[1].set_ylabel('Arithmetic contribution to portfolio return')
axes[1].set_ylim(min(years.short_additive_contribution.min(),years.long_additive_contribution.min())-.12,
                 max(years.short_additive_contribution.max(),years.long_additive_contribution.max())+.15)
for ax in axes:
    ax.axhline(0,color=GRAY,lw=.8);ax.grid(axis='y',alpha=.15);ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False,ncols=2,loc='upper left',fontsize=9)
axes[1].set_xticks(xx,labels)
fig.suptitle('Calendar-year returns and long/short contributions',x=.075,ha='left',fontsize=19,fontweight='bold')
fig.text(.075,.925,'* 2017 starts on April 5; 2026 ends on November 19. The top panel does not annualize these partial years.',fontsize=10,color='#526170')
fig.text(.075,.023,'Long exposure +0.5; short exposure -0.5. Daily leg contributions add to gross return; sums of contributions are not compounded returns.',fontsize=10,color='#526170')
fig.tight_layout(rect=(.015,.055,.995,.91));save(fig,'calendar_year_and_legs')

for group in range(4):
    fig,axes=plt.subplots(2,5,figsize=(15,7),sharex=True,sharey=True)
    subset=folds.iloc[group*10:(group+1)*10]
    for ax,row in zip(axes.flat,subset.itertuples()):
        f=good.loc[good.fold.eq(row.fold)]
        for col,color in [('gross',BLUE),('net_cost_scenario',RED)]:
            eq=np.r_[1.,np.cumprod(1+f[col].to_numpy())]
            assert eq.min()>.48 and eq.max()<1.48
            ax.plot(np.arange(len(eq)),eq,color=color,lw=1.4)
        ax.axhline(1,color=GRAY,lw=.6,alpha=.7)
        ax.set_title(f'F{row.fold:02d} | {row.first_signal[:7]}\nGross {row.gross_return_total:+.1%}',loc='left',fontsize=10)
        ax.text(.04,.07,f'n={row.days}',transform=ax.transAxes,fontsize=9,color=GRAY)
        ax.grid(alpha=.12);ax.set_ylim(.48,1.48);ax.set_xlim(0,66)
        if row.fold==40:ax.set_facecolor('#f1f3f5')
    for ax in axes[-1]:ax.set_xlabel('Scored day number',fontsize=9)
    for ax in axes[:,0]:ax.set_ylabel('Equity (start = 1)',fontsize=9)
    fig.suptitle(f'Within-fold equity | F{group*10+1:02d}-F{group*10+10:02d}',x=.06,ha='left',fontsize=20,fontweight='bold')
    fig.text(.06,.92,'Blue: gross. Red: cost scenario. Every fold starts at 1; all panels share the same axes.',fontsize=11,color='#526170')
    fig.tight_layout(rect=(.01,.01,.995,.89),h_pad=2.2);save(fig,f'fold_equity_{group*10+1:02d}_{group*10+10:02d}')


def pct(v):return f'{v:+.2%}'
def number(v):return f'{v:.3f}'
def p(text):return {'type':'paragraph','text':text}
def bullets(items):return {'type':'bullets','items':items}
def table(headers,rows):return {'type':'table','headers':headers,'rows':[[str(v) for v in row] for row in rows]}
def image(name,caption):return {'type':'image','src':f'figures/{name}.png','caption':caption}
def code(text):return {'type':'code','text':text}
sections=[]
def section(title,*blocks,page=False):sections.append({'title':title,'blocks':list(blocks),'page_break':page})

a=summary['all'];b=summary['from_2021_H2'];con=summary['concentration']
run_url='https://www.kaggle.com/code/aiaila101/tw-ensemble-expanding-window-20261125?scriptVersionId=373806445'
repo_url='https://github.com/Kevin-Shih-Kai-Jen/jpx-equity-prediction'
data_url=repo_url+'/tree/a79c2b2efe9a8472f52e61c94f062425e06a0c35/Taiwan_stock_data/dataset'
provenance_url=repo_url+'/blob/a79c2b2efe9a8472f52e61c94f062425e06a0c35/Taiwan_stock_data/README.md'

metric_rows=[]
for key,label in [('all_validation','Full history'),('from_2021_H2','From 2021-07-01')]:
    c=run['conclusions'][key]
    for portfolio,plabel in [('gross','Gross'),('net_cost_scenario','Cost scenario')]:
        m=c['portfolios'][portfolio]
        total=f'{m["compounded_return"]:+.6%}' if portfolio=='net_cost_scenario' else pct(m['compounded_return'])
        metric_rows.append([label,plabel,m['days'],total,pct(m['cagr_252']),number(m['annualized_sharpe']),pct(m['max_drawdown'])])
section('Executive conclusion',
    p('The recorded 3-long / 3-short strategy earns a positive gross compounded return, but it does not cover the assumed transaction costs. Only 19 of 40 folds have a positive gross return; none is positive in the cost scenario. Profits are concentrated in F34-F36, while most training windows retain their incoming weights.'),
    table(['Period','Portfolio','Days','Compounded return','Annualized return','Sharpe','Max drawdown'],metric_rows),
    p('Full-history scored signals run from 2017-04-05 through 2026-11-19, with returns realized through 2026-11-23. There are 2,334 complete dates and 17 excluded dates. The exact signal-date slice from 2021-07-01 contains 1,302 complete dates and 10 excluded dates.'),
    p('All annualized returns use 252 scored days per year. Days without complete selected-position labels are excluded, not assigned zero returns. These are research results conditional on complete-label dates, not a complete executable calendar-time NAV.'),
    p('The Japanese checkpoint used training/validation information through June 2021. The early period is a retrospective transfer diagnostic. The later slice avoids that known checkpoint timing mismatch, but model architecture and strategy choices were still made retrospectively.'))

section('Model, execution and expanding-window design',
    p('TW_ENSEMBLE is the Taiwan transfer run of the JPX no-ReLU ENSEMBLE architecture: GRU, RNN and LSTM final hidden states are averaged, followed by dropout and two linear layers (128 to 32 to 1), with no added ReLU between those dense layers. The exact source is included in source/models.py.'),
    table(['Item','Recorded setting'],[
        ['Completed run','Kaggle version 1; 40 / 40 folds; 2 x Tesla T4'],
        ['Completion time','2026-11-25 16:08:36 Asia/Taipei (08:08:36 UTC)'],
        ['Observed run duration','7,621.4 seconds, approximately 2 hours 7 minutes'],
        ['Lookback and features','21 observations; Open, High, Low, Close, Volume, ExpectedDividend'],
        ['Architecture','Hidden size 128; one recurrent layer per branch; dropout 0.2'],
        ['Preprocessing','Original JPX scaler frozen; missing ExpectedDividend filled with zero'],
        ['Expanding training origin','2017-01-03'],
        ['First outer test window','Starts 2017-04-03; advances by 90 calendar days'],
        ['Inner early-stop validation','30 calendar days immediately before each outer test start'],
        ['Label purging','Training labels finish before inner validation; validation labels finish before outer test'],
        ['Optimizer and loss','New Adam for Taiwan transfer; learning rate 0.001; Huber delta 1'],
        ['Batch and stopping','Global batch 4,096; max 50 epochs; patience 10; min_delta 0.00001'],
        ['State carried between folds','Best selected model and Adam optimizer state'],
        ['GPU execution','DataParallel; float32 with TF32; gradient norm clip 1; shuffle=False'],
        ['Seed','20261125'],
        ['Runtime','Python 3.13.16; PyTorch 2.11.0+cu130; CUDA 13.0; NumPy 2.1.3; pandas 2.3.3']]),
    p('This publication analyzes the completed run; it does not retrain, change hyperparameters or select a replacement model using the outer test returns. GPU kernels and parallel execution can prevent bitwise-identical results in a future rerun.'),page=True)

section('Data audit and evaluation coverage',
    table(['Audit item','Result'],[
        ['Source rows / stock codes',f'{audit["rows"]:,} / {audit["stocks"]:,}'],
        ['Source date range','2017-01-03 through 2026-11-23'],
        ['Source trading dates',f'{audit["trading_dates"]:,}'],
        ['Duplicate stock-date pairs',audit['duplicate_stock_dates']],
        ['Rows with missing OHLC',f'{audit["missing_ohlc_rows"]:,}'],
        ['Missing targets',f'{audit["missing_targets"]:,}'],
        ['Comparable target checks',f'{audit["verified_target_matches"]:,} matches; {audit["verified_target_mismatches"]} mismatches'],
        ['Absolute target > 20% / > 50%',f'{audit["target_abs_over_20pct"]} / {audit["target_abs_over_50pct"]}'],
        ['Prediction rows across outer folds',f'{run["prediction_rows"]:,}'],
        ['Full-history complete / excluded dates','2,334 / 17'],
        ['From 2021-07-01 complete / excluded dates','1,302 / 10']]),
    p('Targets are computed within each stock as Close[t+2] / Close[t+1] / AdjustmentFactor[t+2] - 1. The supplied target was independently recomputed and matched on all 4,240,618 comparable rows. For scored positions, entry and exit labels must correspond to consecutive global trading dates.'),
    p('Selection happens before checking target availability. If any selected position has a missing or nonconsecutive label, the whole date is excluded. The strategy never substitutes a different stock because a selected target is missing.'),
    p('The dataset documentation explicitly states that its stock universe was obtained from the then-current listing universe. Delisted and otherwise removed companies are not fully represented, so survivorship bias remains. Cash dividends, capital reductions and stock splits are not fully adjusted by the supplied construction.'),
    p('Sources: [pinned dataset files]('+data_url+') and [pinned dataset documentation]('+provenance_url+').'),page=True)

section('Portfolio construction and cost assumptions',
    bullets([
        'After signal-day t close, rank predictions and buy the three highest-ranked stocks while shorting the three lowest-ranked stocks.',
        'Within each side, rank weights are proportional to 2 : 1.5 : 1. Long exposure is +0.5, short exposure is -0.5, and total absolute exposure is 1.',
        'The target-return proxy assumes entry at t+1 close and exit at t+2 close. Each daily tranche is fully opened and closed; no cross-day turnover netting is applied.',
        'Costs use opening and adjusted closing notionals. Sell-side tax is charged on sales, and borrow cost uses actual calendar holding days.']),
    table(['Cost parameter','Assumption'],[['Commission per side','0.1425%'],['Sell-side tax','0.3%'],['Slippage per side','0.05% (5 bps)'],['Annualized short borrow fee','3%'],['Average full-history daily cost','0.6915%, approximately 69.15 bps']]),
    p('These are scenario parameters, not verified account-specific fees or a fill simulation. Historical shortability, daily limit constraints and executable auction prices are not established by these data.'),
    p('Net scenario return on each scored date equals the gross portfolio return minus that date\'s modeled cost. The full-history average gross return is only 2.23 bps per day; from 2021-07-01 it is 3.56 bps per day. Both are far below the assumed average daily cost.'),page=True)

section('Equity curves and drawdown',image('equity_and_drawdown','The two periods are independently rebased to initial capital 1. Red curves apply the stated cost scenario. Missing selected-label dates are excluded.'),page=True)

section('How often is a fold profitable?',
    table(['Statistic','All 40 folds','From 2021-07-01: 23 segments'],[
        ['Positive gross return','19 / 40 (47.5%)','11 / 23 (47.8%)'],
        ['Positive cost-scenario return','0 / 40','0 / 23'],
        ['Median fold gross compounded return',pct(a['median_gross_return']),pct(b['median_gross_return'])],
        ['Median fold gross annualized return',pct(a['median_gross_annualized']),pct(b['median_gross_annualized'])],
        ['Median fold gross Sharpe',number(a['median_sharpe']),number(b['median_sharpe'])],
        ['Folds with positive mean daily Rank IC','24 / 40','14 / 23'],
        ['Positive long-only gross fold return','26 / 40','13 / 23'],
        ['Daily gross positive-return frequency',f'{summary["daily"]["all"]["daily_win_rate"]:.2%}',f'{summary["daily"]["from_2021_H2"]["daily_win_rate"]:.2%}']]),
    p('The later group includes only the portion of F18 from July 1 onward, all of F19-F39, and the short F40 tail. It is not a set of 23 full 90-day windows. All 23 segments are recomputed from daily returns after the exact cutoff.'),
    p('The longest gross winning streak is three folds, including F29-F31 and F34-F36. The longest gross losing streak is four folds, F23-F26. A profitable fold and an accepted training update measure different outcomes.'),page=True)

section('Annualized return in every fold',
    image('fold_returns','The upper panel annualizes each fold separately; the lower panel shows its actual observed compounded return.'),
    p('Annualized return = product(1 + daily return) ** (252 / scored days) - 1. Fold annualized returns cannot be averaged to obtain the full-history annualized return. F1-F39 have 53-65 scored dates each. F40 has only six: its -0.60% observed return becomes -22.43% when annualized. This is a conversion, not a forecast.'),page=True)

extreme_rows=[]
for label,key in [('Best','best_three'),('Worst','worst_three')]:
    for r in summary[key]:extreme_rows.append([label,f'F{r["fold"]:02d}',r['first_signal']+' to '+r['last_signal'],r['days'],pct(r['gross_return_total']),pct(r['gross_annualized_252'])])
section('Best and worst folds; concentration of gains',
    table(['Group','Fold','Scored signal dates','Days','Gross return','Annualized'],extreme_rows),
    p(f'F34-F36 together compound to {pct(con["top_three_compounded_return"])} gross. Linking the remaining 37 folds produces {pct(con["remaining_folds_compounded_return"])}. These three folds were identified retrospectively as the best gross-return folds; this diagnostic does not imply that they could have been selected in advance.'),
    p(f'A fixed calendar split tells a similar story: before 2025 the gross compounded return is {pct(con["before_2025"]["return_total"])}, while from 2025 onward it is {pct(con["since_2025"]["return_total"])}. The positive full-history outcome is therefore not evidence of consistently profitable periods.'),
    p('F15 has the worst observed fold loss (-18.46%). F32 has the worst annualized fold return (-56.48%), because its effective observation length differs from F15.'),page=True)

section('Fold-level risk and ranking signal',image('fold_risk_and_ic','Within-fold drawdown resets at the start of each fold. Full-path drawdown can be deeper across fold boundaries.'),
    p('Rank IC is the daily Spearman correlation across eligible stocks with valid labels. Its coverage differs from the six selected positions and from the set of complete portfolio dates. F40 is especially short. Positive mean IC and positive top/bottom-tail portfolio returns need not coincide.'),page=True)

section('What actually changed during continuation training?',
    p('Every fold executed gradient training, but only F3 epoch 1 passed the fixed validation improvement threshold. In the other 39 folds, best_epoch = 0 means the incoming model was retained after training. From F3 onward, the selected weights did not change again. Sixteen parameter tensors in the final checkpoint differ from the original Japanese checkpoint.'),
    p('F3 itself earns -1.66% gross in its outer test window. F34-F36 do not select new weights. Their stronger returns therefore cannot be described as evidence that new updates in those windows improved the model.'),
    table(['Prediction metric','Full history','From 2021-07-01'],[
        ['Mean daily Rank IC',f'{run["mean_daily_rank_ic"]:.6f}',f'{run["conclusions"]["from_2021_H2"]["mean_daily_rank_ic"]:.6f}'],
        ['Model MAE',f'{run["mae"]:.6f}',f'{run["conclusions"]["from_2021_H2"]["mae"]:.6f}'],
        ['Zero-return baseline MAE',f'{run["zero_baseline_mae"]:.6f}',f'{run["conclusions"]["from_2021_H2"]["zero_baseline_mae"]:.6f}']]),
    p('Average rank correlations are close to zero, and MAE is slightly worse than predicting a zero return. These measures assess the broad prediction universe; they are distinct from tail-portfolio returns. This publication reports the original metrics without tuning on outer-test performance.'),page=True)

section('Calendar-year performance and long/short contributions',image('calendar_year_and_legs','The top panel shows actual observed calendar-year returns. The lower panel sums weighted daily leg contributions; these sums are not compounded returns.'),page=True)
legrows=[]
for key,label in [('all','Full history'),('from_2021_H2','From 2021-07-01')]:
    legrows.append([label]+[f'{summary["legs"][key][k]:+.2f}' for k in ['long_mean_bps','short_mean_bps','combined_mean_bps','cost_mean_bps']])
section('Annual results and the cost budget',
    table(['Year','Scored days','Gross return','Cost-scenario return','Gross Sharpe','Within-year drawdown'],[
        [str(int(r.year))+('*' if int(r.year) in [2017,2026] else ''),r.days,pct(r.gross_return_total),pct(r.net_return_total),number(r.gross_sharpe),pct(r.gross_max_drawdown)] for r in years.itertuples()]),
    p('* 2017 begins on April 5 and 2026 ends on November 19. Returns in this table are not annualized. Dates are assigned to years by signal date.'),
    table(['Period','Long bps/day','Short bps/day','Gross bps/day','Cost bps/day'],legrows),
    p('One basis point (bps) equals 0.01%. The short leg has a negative average contribution over both periods. These are arithmetic contributions at long +0.5 / short -0.5 exposure, not standalone compounded performance.'),
    p(f'For a hypothetical identical cost deducted every scored day, the ex-post break-even charge is {summary["constant_daily_break_even_bps"]["all"]:.2f} bps/day over full history and {summary["constant_daily_break_even_bps"]["from_2021_H2"]:.2f} bps/day in the later slice. It is solved from product(1 + gross return - constant cost) = 1, not from arithmetic average return. It is not a live fee quote or a forward-looking cost allowance.'),page=True)

altrows=[]
for key,label in [('all_validation','Full history'),('from_2021_H2','From 2021-07-01')]:
    for portfolio,plabel in [('long_only_gross','Top 3 long-only'),('clipped_20pct_sensitivity','Long/short, targets clipped +/-20%'),('equal_weight_available_labels','Equal weight, available labels')]:
        m=run['conclusions'][key]['portfolios'][portfolio]
        altrows.append([label,plabel,pct(m['compounded_return']),pct(m['cagr_252']),number(m['annualized_sharpe']),pct(m['max_drawdown'])])
section('Other portfolios and extreme-label sensitivity',
    table(['Period','Gross portfolio / sensitivity','Compounded return','Annualized','Sharpe','Max drawdown'],altrows),
    p('The long-only and equal-weight comparisons have different exposure and risk structures from the zero-net-exposure long/short strategy. All figures in this table are before costs; the equal-weight series covers the supplied available-label universe and is not a market index or a complete tradable benchmark.'),
    p('The source contains 422 targets with absolute returns over 20%, and 192 over 50%. One extreme label is selected by the strategy. Clipping individual targets at +/-20% is a sensitivity analysis, not a repair of the main data. It leaves the principal conclusion unchanged.'),page=True)

for group in range(4):
    section(f'Within-fold equity paths: F{group*10+1:02d}-F{group*10+10:02d}',
        image(f'fold_equity_{group*10+1:02d}_{group*10+10:02d}','All panels use the same axes. Each fold starts at 1. Blue is gross return; red applies the cost scenario. Gray F40 is the short terminal window.'),page=True)

fold_headers=['Fold','Scored signal dates','Days','Excluded','Gross','Gross ann.','Net scenario','Net ann.','Sharpe','Max DD','Mean IC','Best epoch']
for group in range(2):
    rows=[]
    for r in folds.iloc[group*20:(group+1)*20].itertuples():
        rows.append([f'F{r.fold:02d}',r.first_signal+' to '+r.last_signal,r.days,r.excluded_dates,pct(r.gross_return_total),pct(r.gross_annualized_252),pct(r.net_return_total),pct(r.net_annualized_252),number(r.gross_sharpe),pct(r.gross_max_drawdown),f'{r.mean_rank_ic:.4f}',r.best_epoch])
    section(f'Complete fold detail: F{group*20+1:02d}-F{group*20+20:02d}',table(fold_headers,rows),
        p('Net denotes the stated cost scenario. Sharpe and maximum drawdown are gross and reset by fold. Best epoch 0 means the incoming weights were retained, not that training was skipped. The exact later-period slice is in data/fold_metrics_from_2021_H2.csv.'),page=True)

section('Selection concentration',
    p('The six positions on each of 2,334 scored dates produce 14,004 selected-position records and 391 distinct stock codes. The following table ranks stock-code / direction pairs by selected-day count.'),
    table(['Stock code','Direction','Selected days','Share of all slots','Arithmetic contribution'],[
        [int(r.SecuritiesCode),r.Side,r.selected_days,f'{r.selection_slot_fraction:.1%}',pct(r.additive_contribution)] for r in stocks.head(10).itertuples()]),
    p('Arithmetic contribution is the sum of position weight times stock return over selected dates. It is not the stock\'s standalone compounded return or cash profit. Long and short appearances of the same code are separate rows. The complete selection table and daily positions are included.'),page=True)

section('Interpretation limits',bullets([
    'Checkpoint timing: source Japanese training ended on 2021-03-08 and validation on 2021-06-08. Pre-July-2021 results do not establish performance of a model that was available at the historical signal time.',
    'Retrospective design: the architecture, strategy and dataset were selected after the historical periods. Even the later slice is not a fully prospective live test.',
    'Survivorship bias: the supplied dataset documentation states that the universe uses companies still listed at extraction time; historical delistings and removals are incomplete.',
    'Corporate actions and income: ExpectedDividend is entirely missing and filled with zero. The target excludes cash dividends and does not fully account for capital reductions, stock splits or all exceptional events.',
    'Execution and borrowing: historical short availability, borrow constraints, price-limit locks, auction fills and delisting cash flows are not verified.',
    'Missing labels: entire unresolved selected-position dates are excluded. Results are conditional on complete dates and can differ from a fully realized portfolio.',
    'Short windows: fold annualization and Sharpe are noisy. F40 contains only six scored days; partial calendar years and the partial F18 cutoff must not be treated as full windows.',
    'Concentration analysis: the best three folds were identified after observing results. Removing or keeping them is descriptive and is not an implementable selection rule.',
    'No tuning claim: no hyperparameters were changed or new models selected while writing this English analysis.']),page=True)

section('Verification and artifact provenance',
    bullets([
        'All 40 outer test windows are disjoint; training and inner-validation label completion dates precede the next evaluation boundary.',
        'Daily selected-position returns and the stated cost formula were independently reconstructed. Rank selection was checked on 40 spread-out dates.',
        'Compounding all fold returns reproduces both original period totals. Complete and excluded date counts reconcile exactly.',
        'Long and short contributions sum to the daily gross portfolio return. Annualized and unannualized fold return signs agree.',
        'The training script, architecture source, source data archive and initial weights match the SHA-256 recorded by the completed Kaggle run.',
        'The final checkpoint hash is verified; 16 parameter tensors differ from the initial checkpoint.']),
    table(['Artifact','SHA-256'],[
        ['Source data ZIP',env['signature']['data_sha256']],['Initial Japanese checkpoint',env['signature']['weights_sha256']],
        ['Executed training script',env['signature']['script_sha256']],['Architecture source',env['signature']['models_sha256']],
        ['Final Taiwan checkpoint',json.loads((DATA/'completed.json').read_text())['final_weight_sha256']],
        ['Original complete run export ZIP','330c3ac26eb66d8d5d58639d6b9269bcc7fe3015b8988a9a45849ef0e8272d41']]),
    p('The publication includes daily portfolio data, selected positions, daily IC, fold and annual metrics, epoch history, audit and configuration files, both initial and final checkpoints, and the exact training source. Per-stock prediction Parquet files, all intermediate fold checkpoints and the resume checkpoint are retained in the original Kaggle export, not duplicated in this analysis directory.'),
    p('The recorded [Kaggle run]('+run_url+') is private and requires authorized account access. The [initial-model notebook]('+env['signature']['config']['source_notebook']+') and pinned dataset links identify the inputs. This is not a Kaggle competition leaderboard result.'),page=True)

section('Files and reproduction',
    table(['Path','Purpose'],[
        ['README.md','Complete English report rendered directly by GitHub'],['TW_ENSEMBLE_Backtest_Report_EN.pdf','Portable English report'],
        ['report.html','English HTML report; keep the figures directory alongside it'],['figures/','English PNG and SVG figures'],
        ['data/','Daily, fold, yearly, selection, audit and provenance tables'],['model/','Original Japanese checkpoint and final Taiwan checkpoint'],
        ['source/train_tw.py; source/models.py','Exact source executed on Kaggle'],['source/TW_ENSEMBLE_Kaggle.ipynb','Prepared Kaggle notebook'],
        ['source/build_analysis.py; source/build_pdf.py','Rebuild English report and figures without retraining'],
        ['source/verify_results.py','Recheck published calculations, provenance and model differences'],['manifest.json','Publication file sizes and SHA-256 checksums']]),
    p('From this result directory, install analysis dependencies and rebuild the English outputs:'),
    code('python -m pip install -r source/requirements-analysis.txt\npython source/verify_results.py\npython source/build_analysis.py\npython source/build_pdf.py'),
    p('To rerun training separately, install compatible GPU dependencies, reassemble the pinned dataset ZIP, and choose a fresh output directory:'),
    code('python -m pip install -r source/requirements-training.txt\npython source/train_tw.py --data /path/to/stock_price_final.zip \\\n  --weights model/JPX_ENSEMBLE_initial.pt --output /path/to/new_run'),
    p('The archived final checkpoint contains model_state_dict, scaler and config. Load it with torch.load(..., weights_only=True), instantiate ENSEMBLE(6, 128, 1), and load model_state_dict strictly. Preserve the frozen scaler and original feature construction for inference; this checkpoint does not by itself supply a live execution system.'),page=True)

document={'title':'TW_ENSEMBLE','subtitle':'Taiwan transfer learning: complete backtest analysis',
          'date':'2026-11-25','sections':sections}
(DATA/'report_content.json').write_text(json.dumps(document,ensure_ascii=False,indent=2)+'\n')


def markdown_block(block):
    kind=block['type']
    if kind=='paragraph':return block['text']+'\n'
    if kind=='bullets':return '\n'.join('- '+x for x in block['items'])+'\n'
    if kind=='code':return '```bash\n'+block['text']+'\n```\n'
    if kind=='image':return f'![{block["caption"]}]({block["src"]})\n\n'+block['caption']+'\n'
    if kind=='table':
        esc=lambda x:str(x).replace('|','\\|').replace('\n',' ')
        return '\n'.join(['| '+' | '.join(map(esc,block['headers']))+' |','| '+' | '.join(['---']*len(block['headers']))+' |']+['| '+' | '.join(map(esc,row))+' |' for row in block['rows']])+'\n'
    raise ValueError(kind)


markdown='# TW_ENSEMBLE\n\nTaiwan transfer learning: complete backtest analysis. Published 2026-11-25.\n\n'
markdown+='[English PDF](TW_ENSEMBLE_Backtest_Report_EN.pdf) | [HTML report](report.html) | [Fold metrics](data/fold_metrics.csv) | [Final model](model/TW_ENSEMBLE_final.pt)\n\n'
for s in sections:
    markdown+='## '+s['title']+'\n\n'+'\n'.join(markdown_block(block) for block in s['blocks'])+'\n'
(BASE/'README.md').write_text(markdown)
body=MarkdownIt('commonmark',{'html':False}).enable('table').render(markdown)
body=body.replace('<table>','<div class="scroll"><table>').replace('</table>','</table></div>')
style='''body{margin:0;background:#f4f6f8;color:#172432;font:16px/1.75 system-ui,-apple-system,Arial,sans-serif}main{max-width:1200px;margin:36px auto;padding:0 28px}h1{font-size:40px;line-height:1.2;color:#176c80}h2{font-size:25px;margin:46px 0 18px;padding-top:24px;border-top:1px solid #ccd7df}p,li{max-width:1100px}.scroll{overflow:auto;max-height:820px;background:white;border:1px solid #dce3e9;border-radius:8px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:11px 12px;border-bottom:1px solid #dfe6ec;white-space:nowrap;text-align:left;font-variant-numeric:tabular-nums}th{background:#eaf1f5;position:sticky;top:0}tr:nth-child(even){background:#f9fbfc}img{width:100%;height:auto;background:white}a{color:#176c80}pre{overflow:auto;background:#e9eef2;padding:18px;border-radius:8px;font-size:13px}code{font-family:ui-monospace,monospace;font-size:.9em}@media(max-width:700px){main{padding:0 14px}h1{font-size:31px}h2{font-size:22px}body{font-size:15px}}@media print{body{background:white}main{margin:0;max-width:none}.scroll{max-height:none;overflow:visible}th{position:static}h2{break-after:avoid}img{break-inside:avoid}}'''
(BASE/'report.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TW_ENSEMBLE - Complete Backtest Analysis</title><style>'+style+'</style></head><body><main>'+body+'</main></body></html>')
(BASE/'model/README.md').write_text('# TW_ENSEMBLE model checkpoints\n\n`JPX_ENSEMBLE_initial.pt` is the exact Japanese starting checkpoint; `TW_ENSEMBLE_final.pt` is the final selected Taiwan checkpoint.\n\nThe final archive contains `model_state_dict`, `scaler` and `config`. Architecture: `ENSEMBLE(6, 128, 1)` in [`../source/models.py`](../source/models.py). Use `torch.load(..., map_location="cpu", weights_only=True)` and a strict state-dictionary load. Preserve the source feature construction and frozen scaler for inference.\n\nOnly F3 epoch 1 was accepted as an update. All later selected weights were carried forward. Neither file is an optimizer-resume checkpoint or a live trading system. See the complete report for temporal, data and execution limitations.\n')
print(f'Rebuilt {len(sections)} report sections and 8 English figures in {BASE}')
