"""Build the English PDF and figures from frozen aggregate results; no training."""
from pathlib import Path
from datetime import datetime
import json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from xml.sax.saxutils import escape

D = Path(__file__).resolve().parent
A = json.loads((D/'summary.json').read_text())
C = json.loads((D/'experiment_config.json').read_text())
O = A['overall']
F = pd.read_csv(D/'fold_metrics.csv')
DA = pd.read_csv(D/'daily_metrics.csv', parse_dates=['date'])
CO = pd.read_csv(D/'cost_sensitivity.csv')
FIG = D/'figures'
FIG.mkdir(exist_ok=True)
NAVY = '#173345'; TEAL = '#087F8C'; AMBER = '#B96B28'; MUTED = '#526775'; GRID = '#DCE5E9'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10, 'axes.labelcolor':MUTED,
                     'text.color':NAVY,'xtick.color':MUTED,'ytick.color':MUTED,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':False,'axes.edgecolor':GRID,
                     'savefig.facecolor':'white','svg.fonttype':'none'})

def finish(fig, name):
    fig.savefig(FIG/(name+'.png'), dpi=210, bbox_inches='tight', pad_inches=.12)
    fig.savefig(FIG/(name+'.svg'), bbox_inches='tight', pad_inches=.12)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(9,3.1))
ax.plot(DA.date, DA.cumulative_spread_sum*100, color=TEAL,lw=2)
ax.fill_between(DA.date,0,DA.cumulative_spread_sum*100,color=TEAL,alpha=.08)
ax.axhline(0, color=GRID, lw=.8)
ax.yaxis.grid(True,color=GRID,alpha=.65)
ax.set_ylabel('Cumulative gross spread sum (pp)')
ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_ylim(-2,27); ax.set_xlim(DA.date.min(), DA.date.max()+pd.Timedelta(days=80))
ax.annotate(f"{O['daily_double_precision_sum']*100:.2f} pp",xy=(DA.date.iloc[-1],DA.cumulative_spread_sum.iloc[-1]*100),xytext=(-6,12),textcoords='offset points',ha='right',color=TEAL,fontweight='bold')
fig.tight_layout(); finish(fig,'cumulative_gross_spread')

fig, axes=plt.subplots(1,2,figsize=(9,2.75))
for ax,col,title in [(axes[0],'sharpe','Sharpe (daily, non-annualized)'),(axes[1],'rank_IC','Mean daily Rank IC')]:
    values=F[col]
    ax.bar(F.fold,values,color=[TEAL if v>=0 else AMBER for v in values],width=.64,zorder=3)
    ax.axhline(0,color=MUTED,lw=.7);ax.yaxis.grid(True,color=GRID,alpha=.6,zorder=0)
    ax.set_xticks(F.fold);ax.set_xlabel('Fold');ax.set_title(title,loc='left',fontsize=11,fontweight='bold',pad=12)
    ax.tick_params(axis='both',length=0)
axes[0].set_ylim(0,.32);axes[1].set_ylim(-.004,.013)
fig.tight_layout(w_pad=3);finish(fig,'fold_sharpe_and_ic')

fig,ax=plt.subplots(figsize=(9,3))
v=F.mae_improvement_vs_zero*100
ax.bar(F.fold,v,color=[TEAL if a>=0 else AMBER for a in v],width=.62,zorder=3)
ax.axhline(0,color=NAVY,lw=.8);ax.yaxis.grid(True,color=GRID,alpha=.6,zorder=0)
ax.set_ylabel('MAE improvement vs. zero (%)');ax.set_xlabel('Fold');ax.set_xticks(F.fold)
ax.set_ylim(-2.65,.6)
for x,y in zip(F.fold,v):ax.text(x,y+(.08 if y>=0 else -.11),f'{y:+.2f}%',ha='center',va='bottom' if y>=0 else 'top',fontsize=9)
fig.tight_layout();finish(fig,'mae_vs_zero')

fig,ax=plt.subplots(figsize=(9,3.15))
v=CO.sum_returns*100
ax.bar([str(int(i)) for i in CO.cost_bps_per_traded_notional],v,color=[TEAL if i>=0 else AMBER for i in v],width=.62,zorder=3)
ax.axhline(0,color=NAVY,lw=.8);ax.yaxis.grid(True,color=GRID,alpha=.65,zorder=0)
ax.set_ylim(-90,35);ax.set_ylabel('Additive scenario spread sum (pp)')
ax.set_xlabel('Illustrative one-way cost (bp per traded target notional)')
for x,y in enumerate(v):ax.text(x,y+(2 if y>=0 else -2),f'{y:+.2f}',ha='center',va='bottom' if y>=0 else 'top',fontsize=10)
fig.tight_layout();finish(fig,'cost_sensitivity')

regular=font_manager.findfont('DejaVu Sans')
bold=font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight='bold'))
pdfmetrics.registerFont(TTFont('Body',regular));pdfmetrics.registerFont(TTFont('Bold',bold))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
W,H=A4;M=43;CW=W-2*M
OUT=D.parent/'CNN_LSTM_Results_EN.pdf'
cv=canvas.Canvas(str(OUT),pagesize=A4,pageCompression=1)
cv.setTitle('JPX CNN-LSTM | Completed Backtest Results')
cv.setAuthor('Kevin-Shih-Kai-Jen | JPX equity prediction')
cv.setSubject('English report of the completed 8-fold CNN-LSTM historical backtest')

def text(s,x,top,width,size=10.1,leading=None,color=NAVY,bold=False,max_h=None):
    sty=ParagraphStyle('p',fontName='Bold' if bold else 'Body',fontSize=size,leading=leading or size*1.45,textColor=HexColor(color))
    p=Paragraph(s,sty);ww,hh=p.wrap(width,H)
    if max_h is not None and hh>max_h+.1:raise ValueError(f'Text overflows: {s[:70]} ({hh}>{max_h})')
    p.drawOn(cv,x,H-top-hh);return hh
def rect(x,top,width,height,color):
    cv.setFillColor(HexColor(color));cv.rect(x,H-top-height,width,height,stroke=0,fill=1)
def rule(top):
    cv.setStrokeColor(HexColor(GRID));cv.setLineWidth(.65);cv.line(M,H-top,W-M,H-top)
def label(s,top):text(s.upper(),M,top,CW,8.4,11,TEAL,True)
def page(n,section,title,subtitle):
    rect(0,0,W,7,TEAL)
    text('JPX / CNN-LSTM',M,26,280,8.5,12,TEAL,True)
    text('RESULTS  |  17 OCT 2026',W-220,26,177,8.2,12,MUTED)
    text(section.upper(),M,65,CW,8.7,12,MUTED,True)
    text(title,M,87,CW,25,30,NAVY,True,max_h=30)
    text(subtitle,M,128,CW,9.3,13.5,MUTED,max_h=41)
    rule(792)
    text('Frozen source b3cd1bfd6172  |  Historical research backtest',M,803,440,7.4,10,MUTED)
    text(f'{n:02d} / 06',W-88,803,45,7.4,10,MUTED)
def end(): cv.showPage()
def card(x,top,width,value,title,note='',accent=TEAL):
    rect(x,top,width,91,'#F1F6F7')
    rect(x,top,3,91,accent)
    text(title,x+12,top+10,width-23,8.3,11,MUTED,True,max_h=22)
    text(value,x+12,top+33,width-23,23,28,accent,True,max_h=28)
    if note:text(note,x+12,top+66,width-23,7.6,10.2,MUTED,max_h=22)
def figure(name,top,height):
    cv.drawImage(str(FIG/(name+'.png')),M,H-top-height,CW,height,preserveAspectRatio=True,anchor='c',mask='auto')
def paragraph(title,body,top):
    text(title,M,top,CW,11.3,15,NAVY,True)
    return text(body,M,top+23,CW,9.8,14.3,max_h=90)
def table(headers,rows,widths,top,row_heights,size=8.8):
    x=M
    rect(M,top,CW,29,NAVY)
    for h,w in zip(headers,widths):text(h,x+7,top+7,w-14,8.2,10.5,'#FFFFFF',True,max_h=22);x+=w
    y=top+29
    for i,(row,rh) in enumerate(zip(rows,row_heights)):
        if i%2==0:rect(M,y,CW,rh,'#F1F6F7')
        x=M
        for cell,w in zip(row,widths):text(str(cell),x+7,y+9,w-14,size,size*1.35,max_h=rh-13);x+=w
        y+=rh
    return y

page(1,'01 / Executive result','Positive spreads, modest signal',
     'Completed experiment: 8 folds x 20 epochs. Validation Target dates: 30 Jun 2017 to 08 Jun 2021.')
gap=12;cw=(CW-2*gap)/3
card(M,182,cw,f"{O['sharpe_unannualized_pooled']:.4f}",'POOLED SHARPE','Daily; non-annualized')
card(M+cw+gap,182,cw,f"{O['daily_rank_ic_mean']:.5f}",'MEAN RANK IC','957 defined daily values')
card(M+2*(cw+gap),182,cw,f"{O['original_total_earnings']*100:.2f} pp",'GROSS SPREAD SUM','Additive; before costs')
label('958 dates / 1,850,796 validation samples',299)
figure('cumulative_gross_spread',323,215)
text('Original daily spreads added over time, using +0.5 long / -0.5 short exposure. This curve is not compounded NAV.',M,546,CW,8.2,11.5,MUTED,max_h=26)
paragraph('What the completed run supports',
          'Every fold has a positive gross spread sum. However, the mean daily rank correlation is small, and folds 3 and 5 provide 50.98% of the total. The ranking signal varies materially across periods.',588)
paragraph('What it does not establish',
          'Validation MAE is 0.0160325 versus 0.0159500 for predicting zero: the model error is 0.52% higher. These results do not establish a robust, cost-adjusted trading advantage or a Kaggle leaderboard score.',679)
end()

page(2,'02 / Method and JPX context','The original experiment, preserved',
     'One model and one Adam state continue through all folds. There is no restart between folds.')
rows=[
 ['Model','Conv1d: 6 to 64 channels, kernel 3; ReLU; max pool 2; 3-layer LSTM, hidden size 100; linear output. Dropout 0.2.'],
 ['Inputs','42 valid observations per security; Open, High, Low, Close, Volume and ExpectedDividend.'],
 ['Training','20 epochs per fold; batch 64; shuffle=False; L1 loss; Adam, learning rate 0.001. Seed 20261016.'],
 ['Validation','8 expanding-window folds with 180-calendar-day intervals. Remove the last two available training dates before fitting the model.'],
 ['Preprocessing','Fit scaling on the pre-purge training partition; reuse it in validation. Add up to 41 earlier input rows for validation windows.'],
 ['Coverage','958 dates and 1,850,796 samples after original filtering. Trailing partial split is not evaluated.'],
 ['Execution','Cloud CPU, 8 PyTorch threads. Completed-epoch recovery preserved model, Adam, random states and saved evaluations.']]
table(['Component','Recorded setup'],rows,[97,CW-97],174,[58,48,48,58,58,46,58],9)
paragraph('Primary JPX comparison',
          'JPX scores the Sharpe ratio of daily rank-weighted spreads from the top 200 and bottom 200 securities. The report therefore leads with the original non-annualized Sharpe; Rank IC and MAE are supporting diagnostics. [1,2]',600)
paragraph('Offline result, not a submitted score',
          'The project uses +0.5 long and -0.5 short weights. Its internal average ranks for ties are not a valid submission column by themselves. Universe coverage, unique zero-based ranks and API row order still need end-to-end submission checks. [2,4]',691)
end()

page(3,'03 / Fold-level evaluation','All eight folds completed',
     'Each row includes 20 completed epochs and its full validation. Sharpe is daily and non-annualized.')
rows=[]
for _,r in F.iterrows():
    rows.append([int(r.fold),f'{r.start}<br/>{r.end}',int(r.days),f'{r.mae:.6f}',f'{r.sharpe:.4f}',f'{r.rank_IC:.5f}',f'{r.earnings*100:.3f}'])
table(['Fold','Target date range','Days','MAE','Sharpe','Rank IC','Gross<br/>sum (pp)'],rows,[35,111,38,71,69,69,CW-393],177,[41]*8,8.7)
text('Gross sum = sum of original daily spread returns, before costs. Fold values preserve the original saved calculations.',M,545,CW,8.2,11.5,MUTED,max_h=25)
figure('fold_sharpe_and_ic',585,152)
text('Strongest fold Sharpe: 5 (0.2744). Weakest: 6 (0.0318), which also has negative Rank IC. The pooled score is computed from all daily returns, not the mean of these eight Sharpes.',M,751,CW,8.3,11.5,MUTED,max_h=35)
end()

page(4,'04 / Diagnostic interpretation','MAE trails the zero baseline',
     'The model can select useful extremes while showing little improvement in all-stock prediction error.')
cw=(CW-12)/2
card(M,180,cw,f"{O['weighted_mae']:.7f}",'MODEL VALIDATION MAE','Sample-weighted across all folds')
card(M+cw+12,180,cw,f"{O['weighted_zero_mae']:.7f}",'ZERO-PREDICTION MAE','Model error is 0.52% higher',AMBER)
figure('mae_vs_zero',305,197)
text('Positive bars mean lower MAE than the zero baseline. The zero baseline is a prediction-error comparator, not a ranking strategy.',M,515,CW,8.3,11.5,MUTED,max_h=25)
paragraph('Performance is concentrated and uneven',
          'Folds 3 and 5 contribute 50.98% of the gross spread sum. The five strongest dates contribute 30.07%. Positive spreads occur on 52.61% of dates and in 35 of 49 calendar months. These are descriptive counts, not significance tests.',560)
paragraph('One missing IC value has a data explanation',
          'On 29 Sep 2020 the supplied Targets are all zero, so Spearman correlation is undefined. The 1,973 evaluated securities have 1,973 distinct predictions. The original mean skips this date; this is not constant model output.',663)
end()

page(5,'05 / Secondary cost analysis','A small cushion under simple costs',
     'Illustrative sensitivity only. Costs were not part of the original reported JPX-style score.')
figure('cost_sensitivity',179,215)
paragraph('Break-even cost: 2.33 bp per traded target notional',
          'Subtract cost x sum of absolute daily target-weight changes, starting and ending flat. At 2 bp the additive spread sum remains +3.35 pp; at 3 bp it becomes -6.77 pp. This simplified result is not realized net performance.',420)
text('The scenario excludes price drift, marked-to-market rebalancing, execution, liquidity limits, short borrow and taxes. The mean absolute target-weight change is 1.0567 per date, including the initial and final trades.',M,502,CW,9,13,MUTED,max_h=52)
rule(572)
label('Read the units correctly',592)
rows=[['bp / bps','Basis point(s): 1 bp = 0.01% = 0.0001 in decimal-return units.'],
      ['pp','Percentage point(s). A gross spread sum of 0.23595 is shown as 23.595 pp, not a compounded investor gain.'],
      ['Gross spread sum','Daily long-short returns added before costs. The additive maximum peak-to-trough fall is 3.23 pp, not NAV drawdown.']]
table(['Term','Meaning in this report'],rows,[112,CW-112],619,[38,49,49],8.6)
end()

page(6,'06 / Evidence and next experiments','Complete, traceable, and bounded',
     'Analysis is based on frozen predictions. Report preparation did not retrain or change the algorithm.')
rows=[['Completed','17 Oct 2026, 03:35:42 UTC (11:35:42 in Taipei).'],
      ['Total wall time','18 h 25 m 53 s, including interruptions and downtime.'],
      ['Recorded training','11 h 00 m 57 s summed over 160 completed epochs; excludes preprocessing, validation, interrupted partial work and downtime.'],
      ['Completion checks','8 fold evaluations; finite predictions; final model/normalization match; continued Adam state with 2,373,700 cumulative steps.']]
table(['Evidence','Recorded result'],rows,[112,CW-112],174,[37,37,51,51],9)
paragraph('Research limits and sensible next tests',
          'This is one seed and one continuous historical training path; the folds are not independent trials. Next tests should use an untouched period, multiple seeds and simple ranking baselines, followed by an exact competition-API evaluation. These tests are not included in the present results.',398)
text('The square-root-of-252 Sharpe reference is 1.5787, under an IID/no-serial-correlation convention. Lag-1 daily return autocorrelation is 0.1273. The primary score remains 0.0994, non-annualized; the reference is not an autocorrelation-adjusted estimate. [3]',M,493,CW,9,13,MUTED,max_h=66)
rule(566)
label('Sources and supporting files',586)
refs=[
 ('[1] Kaggle JPX evaluation overview','https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/overview/evaluation'),
 ('[2] Kaggle JPX code requirements and data description','https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/overview/code-requirements'),
 ('[3] A. W. Lo (2002), The Statistics of Sharpe Ratios','https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios'),
 ('[4] Frozen project source and original evaluation','https://github.com/Kevin-Shih-Kai-Jen/jpx-equity-prediction/tree/b3cd1bfd617251c7eebc42c588e4067a09af1fe3')]
y=610
for title,url in refs:
    text(f'<link href="{escape(url)}" color="{TEAL}">{escape(title)}</link>',M,y,CW,8.8,12.3,max_h=20);y+=24
text('Full URLs and evaluation formulae are in details/evaluation_protocol.md. Exact parameters, per-fold and daily data, original artifact hashes, charts and reproduction scripts are in details/. Public competition descriptions accessed 17 Oct 2026.',M,716,CW,8.5,12.3,MUTED,max_h=57)
end()
cv.save()
print(OUT)
