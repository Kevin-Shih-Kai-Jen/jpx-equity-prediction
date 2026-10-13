from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'pdf' / 'CNN_LSTM_Daily_Update_2026-10-13.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
metrics = {
    'date': '2026-10-13', 'model': 'CNN-LSTM', 'time_span': 42, 'epochs': 10,
    'evaluation_data': 'same data used for training (in-sample)',
    'training_average_loss': 0.015604025316993357,
    'evaluation_average_loss': 0.015570441672566427,
    'sharpe': 0.10057470947504044, 'rank_ic': 0.0042738020891595,
    'zero_baseline_mae_first_report': 0.015564425848424435,
    'prediction_mean': 0.0007700243149884045,
    'prediction_std': 0.0011315844021737576,
    'target_mean': 0.0003973348648287356,
    'target_std': 0.023427896201610565,
    'model_mae': 0.015570441260933876,
    'zero_baseline_mae': 0.015564427711069584,
    'mae_improvement_percent': -0.038636498153209686,
    'prediction_target_correlation': 0.052963461726903915,
    'source': 'User-provided update; no experiment or metric recomputation performed.',
    'original_interpretation': 'Today proves my model is working. With 10 epochs of training data and validate through the training data, I find out that my model can be trained, not just random output.',
    'editorial_note': 'The PDF qualifies the interpretation: in-sample evaluation does not establish generalization or rule out random output.',
    'next_steps': [
        'Use expanding-window validation.', 'Include early stopping.',
        'Save trained models to reuse them when checking the pipeline.',
        'Explore training methods: average loss over 20 time frames and a tentative (total_days / 20)^2 training schedule; design still to be refined.',
        'Try the ensemble model from https://www.mdpi.com/2076-3417/13/7/4644'
    ]
}
(ROOT / 'reported_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')

W, H = A4
M = 45
CW = W - 2*M
navy = HexColor('#172E45')
teal = HexColor('#087E8B')
ink = HexColor('#243648')
muted = HexColor('#647381')
line = HexColor('#DDE5EA')
c = canvas.Canvas(str(OUT), pagesize=A4, pageCompression=1, invariant=1)
c.setTitle('CNN-LSTM Daily Update | 13 October 2026')
c.setAuthor('Quant Learning & Internship 2027')
styles = {
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=10.2, leading=14.6, textColor=ink),
    'small': ParagraphStyle('small', fontName='Helvetica', fontSize=8.3, leading=11.3, textColor=muted),
    'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=9, leading=12, textColor=ink),
}

def para(text, y, style='body', x=M, width=CW):
    p = Paragraph(text, styles[style])
    _, h = p.wrap(width, H)
    p.drawOn(c, x, y-h)
    return y-h

def section(text, y):
    c.setFillColor(teal)
    c.setFont('Helvetica-Bold', 10)
    c.drawString(M, y, text.upper())
    return y-14

c.setFillColor(teal)
c.rect(M, H-49, 28, 4, fill=1, stroke=0)
c.setFont('Helvetica-Bold', 8)
c.drawString(M+38, H-49, 'QUANT LEARNING & INTERNSHIP 2027')
c.setFillColor(navy)
c.setFont('Helvetica-Bold', 25)
c.drawString(M, H-89, 'CNN-LSTM daily update')
c.setFillColor(muted)
c.setFont('Helvetica', 10)
c.drawString(M, H-110, '13 October 2026  |  42-step time span  |  10 training epochs')

y = section('Progress', H-145)
y = para('Completed the evaluation pipeline using JPX-style Sharpe and Rank IC alongside MAE. The 10-epoch run produced nonconstant predictions and was evaluated on the same training data.', y)
y = section('Results / in-sample', y-30)
rows = [
    ['Metric', 'Value', 'Metric', 'Value'],
    ['Training avg. loss', '0.015604', 'Evaluation avg. loss', '0.015570'],
    ['Sharpe', '0.100575', 'Rank IC', '0.004274'],
    ['Model MAE', '0.015570', 'Zero-baseline MAE', '0.015564'],
    ['MAE improvement', '-0.038636%', 'Prediction-target corr.', '0.052963'],
    ['Prediction mean', '0.000770', 'Target mean', '0.000397'],
    ['Prediction std.', '0.001132', 'Target std.', '0.023428'],
]
t = Table(rows, colWidths=[137, 97, 157, CW-391], rowHeights=[25]+[25]*6)
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),navy), ('TEXTCOLOR',(0,0),(-1,0),white),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
    ('FONTNAME',(0,1),(-1,-1),'Helvetica'), ('FONTSIZE',(0,0),(-1,-1),9),
    ('TEXTCOLOR',(0,1),(-1,-1),ink),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[HexColor('#F2F6F8'), white]),
    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ('LEFTPADDING',(0,0),(-1,-1),9), ('RIGHTPADDING',(0,0),(-1,-1),9),
    ('ALIGN',(1,0),(1,-1),'RIGHT'), ('ALIGN',(3,0),(3,-1),'RIGHT'),
    ('LINEBELOW',(0,-1),(-1,-1),0.5,line),
]))
_, th = t.wrap(CW,H)
t.drawOn(c,M,y-th)
y -= th+9
y = para('Values rounded to six decimal places; metrics supplied by the researcher.',y,'small')

y -= 15
note = '<b>Interpretation:</b> The pipeline runs end to end, but MAE is slightly worse than the zero baseline and Rank IC is near zero. These in-sample results do not yet establish performance on unseen data.'
p = Paragraph(note, styles['body'])
_, nh = p.wrap(CW-24,H)
c.setFillColor(HexColor('#ECF5F6'))
c.roundRect(M,y-nh-22,CW,nh+22,5,stroke=0,fill=1)
p.drawOn(c,M+12,y-nh-11)
y -= nh+49
y = section('Next steps',y)
steps = [
    '<b>Validation:</b> Use an expanding window and add early stopping.',
    '<b>Efficiency:</b> Save model checkpoints for pipeline checks without retraining.',
    '<b>Training experiments:</b> Track average loss over 20 time frames and explore a (total_days / 20)<super>2</super> training schedule; the design still needs refinement.',
    '<b>Architecture:</b> Test the ensemble approach in the linked paper [1].',
]
for i, text in enumerate(steps,1):
    c.setFillColor(teal)
    c.setFont('Helvetica-Bold',10)
    c.drawString(M,y-10,str(i)+'.')
    y = para(text,y,x=M+18,width=CW-18)-7
y -= 8
y = para('[1] <link href="https://www.mdpi.com/2076-3417/13/7/4644" color="#087E8B">Applied Sciences (2023), 13(7), 4644 - ensemble-model reference</link>.',y,'small')
assert y > 55, y
c.setStrokeColor(line)
c.line(M,44,W-M,44)
c.setFillColor(muted)
c.setFont('Helvetica',8)
c.drawString(M,30,'RESEARCH LOG  /  Preliminary in-sample check')
c.drawRightString(W-M,30,'1 / 1')
c.showPage()
c.save()
r = PdfReader(OUT)
assert len(r.pages) == 1
text = r.pages[0].extract_text()
for needle in ['0.100575','0.004274','-0.038636%','early stopping','expanding window','0.052963']:
    assert needle in text, needle
print(OUT)
print('Verified: one page; key metrics and next steps present.')
