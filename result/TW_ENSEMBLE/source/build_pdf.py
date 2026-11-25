"""Build the English PDF from the same content used by README.md and report.html."""
from pathlib import Path
import html
import json
import re
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Preformatted

BASE=Path(__file__).resolve().parents[1]
document=json.loads((BASE/'data/report_content.json').read_text())
OUTPUT=BASE/'TW_ENSEMBLE_Backtest_Report_EN.pdf'
PAGE_W,PAGE_H=landscape(A4)
LEFT,RIGHT,TOP,BOTTOM=36,36,42,34
WIDTH=PAGE_W-LEFT-RIGHT
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='ReportTitle',fontName='Helvetica-Bold',fontSize=27,leading=32,textColor=colors.HexColor('#176c80'),spaceAfter=9))
styles.add(ParagraphStyle(name='SectionTitle',fontName='Helvetica-Bold',fontSize=18,leading=22,textColor=colors.HexColor('#172432'),spaceAfter=13,keepWithNext=True))
styles.add(ParagraphStyle(name='ReportBody',fontName='Helvetica',fontSize=10,leading=14.2,textColor=colors.HexColor('#172432'),spaceAfter=10))
styles.add(ParagraphStyle(name='SmallBody',parent=styles['ReportBody'],fontSize=8.7,leading=12,spaceAfter=8))
styles.add(ParagraphStyle(name='Cell',fontName='Helvetica',fontSize=8.3,leading=10.8,textColor=colors.HexColor('#172432')))
styles.add(ParagraphStyle(name='CellSmall',parent=styles['Cell'],fontSize=7.2,leading=9.4))
styles.add(ParagraphStyle(name='CellHead',parent=styles['Cell'],fontName='Helvetica-Bold',textColor=colors.HexColor('#176c80')))
styles.add(ParagraphStyle(name='CodeBox',fontName='Courier',fontSize=8,leading=11,spaceAfter=10,backColor=colors.HexColor('#edf2f5'),borderPadding=8))


def rich(text):
    text=html.escape(str(text))
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:f'<link href="{m.group(2)}" color="#176c80">{m.group(1)}</link>',text)
    return text


def table(block):
    headers=block['headers'];rows=block['rows'];n=len(headers)
    widths=None
    if n==2:
        widths=[WIDTH*.28,WIDTH*.72]
    elif n==12:
        fractions=[.04,.185,.04,.055,.075,.075,.075,.075,.06,.075,.065,.08]
        widths=[WIDTH*x/sum(fractions) for x in fractions]
    elif headers[0]=='Group':
        widths=[45,40,WIDTH-305,45,85,90]
    elif n==7:
        widths=[110,90,45,120,115,80,WIDTH-560]
    elif n==3:
        widths=[WIDTH*.45,WIDTH*.275,WIDTH*.275]
    elif n==6:
        if headers[1]=='Gross portfolio / sensitivity':
            widths=[110,230,110,100,70,WIDTH-620]
        else:
            widths=[WIDTH/n]*n
    else:
        widths=[WIDTH/n]*n
    small=n>=10
    style=styles['CellSmall' if small else 'Cell']
    header_style=ParagraphStyle(name='HeaderTemp',parent=style,fontName='Helvetica-Bold',textColor=colors.HexColor('#176c80'))
    vals=[[Paragraph(rich(x),header_style) for x in headers]]
    for row in rows:
        cells=[]
        for i,v in enumerate(row):
            s=rich(v)
            cells.append(Paragraph(s,style))
        vals.append(cells)
    t=Table(vals,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f4')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7fafb')]),
        ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#c5d5df')),
        ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#dce5ea')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),4 if small else 5),('BOTTOMPADDING',(0,0),(-1,-1),4 if small else 5),
    ]))
    return t


def page_chrome(canvas,doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#d3dfe6'));canvas.setLineWidth(.5)
    canvas.line(LEFT,PAGE_H-27,PAGE_W-RIGHT,PAGE_H-27)
    canvas.setFont('Helvetica-Bold',8);canvas.setFillColor(colors.HexColor('#176c80'))
    canvas.drawString(LEFT,PAGE_H-20,'TW_ENSEMBLE  |  Complete Backtest Analysis')
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#5b6978'))
    canvas.drawRightString(PAGE_W-RIGHT,PAGE_H-20,'2026-11-25  |  English edition')
    canvas.drawString(LEFT,18,'Recorded research run. Cost assumptions and data limitations apply.')
    canvas.drawRightString(PAGE_W-RIGHT,18,str(doc.page))
    canvas.restoreState()


story=[]
for i,section in enumerate(document['sections']):
    if i and section.get('page_break'):story.append(PageBreak())
    if i==0:
        story.extend([Paragraph(document['title'],styles['ReportTitle']),Paragraph(document['subtitle'],styles['ReportBody']),Spacer(1,5)])
    story.append(Paragraph(rich(section['title']),styles['SectionTitle']))
    for block in section['blocks']:
        kind=block['type']
        if kind=='paragraph':story.append(Paragraph(rich(block['text']),styles['ReportBody']))
        elif kind=='bullets':
            for text in block['items']:story.append(Paragraph(rich(text),styles['ReportBody'],bulletText='-'))
        elif kind=='table':story.extend([table(block),Spacer(1,12)])
        elif kind=='code':story.append(Preformatted(block['text'],styles['CodeBox'],maxLineLength=110))
        elif kind=='image':
            img=Image(str(BASE/block['src']))
            scale=min(WIDTH/img.imageWidth,385/img.imageHeight)
            img.drawWidth=img.imageWidth*scale;img.drawHeight=img.imageHeight*scale
            img.hAlign='CENTER';story.append(img)
            story.append(Paragraph(rich(block['caption']),styles['SmallBody']))
        else:raise ValueError(kind)
doc=SimpleDocTemplate(str(OUTPUT),pagesize=(PAGE_W,PAGE_H),leftMargin=LEFT,rightMargin=RIGHT,topMargin=TOP,bottomMargin=BOTTOM,
    title='TW_ENSEMBLE - Complete Backtest Analysis',author='Kevin-Shih-Kai-Jen',subject='Taiwan expanding-window model backtest, 40 folds')
doc.build(story,onFirstPage=page_chrome,onLaterPages=page_chrome)
print(OUTPUT)
