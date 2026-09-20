from pathlib import Path
from textwrap import wrap

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / 'RELATORIO.md'
PDF_PATH = ROOT / 'RELATORIO.pdf'

MARGIN_LEFT = 55
MARGIN_RIGHT = 55
TITLE = 'OUVIDORIA INTELIGENTE'

DEFAULT_TEXT = """# Ouvidoria Inteligente

## Objetivo

Desenvolver uma solução de busca semântica para identificar manifestações públicas semelhantes, mesmo quando a linguagem usada não é idêntica.

## Metodologia

O projeto utiliza embeddings semânticos, cálculo de similaridade de cosseno e análise de duplicatas para comparar registros textuais de forma inteligente.

## Validação

Ao buscar a expressão "Transito na frente da escola", o sistema retornou registros semanticamente relacionados a risco de acidente, infraestrutura escolar e segurança no entorno.

## Resultado

A solução mostrou-se eficaz para encontrar demandas parecidas, organizar a base de dados e apoiar a triagem de manifestações com linguagem natural e variada.

## Conclusão

O sistema atende ao objetivo de classificar e recuperar manifestações públicas com base no significado dos textos, tornando a análise mais rápida e útil para gestão pública.
"""


def parse_blocks(text: str):
    blocks = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith('# '):
            blocks.append(('title', line[2:].upper()))
        elif line.startswith('## '):
            blocks.append(('subtitle', line[3:]))
        elif line.startswith('### '):
            blocks.append(('smalltitle', line[4:]))
        elif line.startswith('- '):
            blocks.append(('bullet', '• ' + line[2:]))
        elif line.startswith('> '):
            blocks.append(('quote', line[2:]))
        elif line.strip():
            blocks.append(('text', line))
        else:
            blocks.append(('blank', ''))
    return blocks


def draw_wrapped_text(c, text, x, y, width, font_name, font_size, left_align=True):
    c.setFont(font_name, font_size)
    wrapped = wrap(text, width=max(20, int(width / (font_size * 0.6))))
    for line in wrapped:
        if y < 60:
            c.showPage()
            y = A4[1] - 80
            c.setFont(font_name, font_size)
        if left_align:
            c.drawString(x, y, line)
        else:
            c.drawCentredString(A4[0] / 2, y, line)
        y -= font_size + 2
    return y


def main():
    text = MD_PATH.read_text(encoding='utf-8') if MD_PATH.exists() else DEFAULT_TEXT
    blocks = parse_blocks(text)

    c = canvas.Canvas(str(PDF_PATH), pagesize=A4)
    c.setFillColor(HexColor('#1f2937'))
    c.setStrokeColor(HexColor('#1f2937'))
    width, height = A4
    y = height - 70

    c.setFont('Helvetica-Bold', 18)
    c.drawCentredString(width / 2, height - 38, TITLE)
    y -= 35

    for kind, value in blocks:
        if kind == 'blank':
            y -= 12
            continue

        if y < 70:
            c.showPage()
            c.setFont('Helvetica-Bold', 18)
            c.drawCentredString(width / 2, height - 38, TITLE)
            y = height - 90

        if kind == 'title':
            c.setFont('Helvetica-Bold', 12)
            c.drawCentredString(width / 2, y, value)
            y -= 18
        elif kind == 'subtitle':
            c.setFont('Helvetica-Bold', 12)
            c.drawString(MARGIN_LEFT, y, value)
            y -= 17
        elif kind == 'smalltitle':
            c.setFont('Helvetica-Bold', 11)
            c.drawString(MARGIN_LEFT + 8, y, value)
            y -= 16
        elif kind == 'bullet':
            c.setFont('Helvetica', 10.5)
            c.drawString(MARGIN_LEFT + 10, y, value)
            y -= 15
        elif kind == 'quote':
            c.setFont('Helvetica-Oblique', 10.5)
            c.drawString(MARGIN_LEFT + 10, y, value)
            y -= 15
        elif kind == 'text':
            paragraph_width = width - MARGIN_LEFT - MARGIN_RIGHT
            y = draw_wrapped_text(
                c,
                value,
                MARGIN_LEFT,
                y,
                paragraph_width,
                'Helvetica',
                10.5,
                left_align=True,
            )
    c.save()
    print(f'PDF gerado em: {PDF_PATH}')


if __name__ == '__main__':
    main()
