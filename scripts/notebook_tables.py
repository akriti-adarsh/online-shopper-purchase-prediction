"""Readable notebook tables with styled HTML and a static rendering for previews."""

from base64 import b64encode
from html import escape
from io import BytesIO
from numbers import Real
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import pandas as pd


INK = "#18253F"
MUTED = "#53647E"
TEAL = "#087F8C"


def _formatted(value, specification):
    if specification is not None and isinstance(value, Real):
        return format(value, specification)
    return str(value)


def table_figure(frame, title, subtitle="", formats=None, widths=None,
                 highlight_columns=(), accent_row=None, index_label="Model"):
    """Render a table without hiding precision or depending on notebook CSS."""
    formats = formats or {}
    headings = [index_label, *map(str, frame.columns)]
    widths = widths or [3.0, *([1.25] * len(frame.columns))]
    if len(widths) != len(headings):
        raise ValueError("Provide one width for the index and each data column.")
    rows = [[str(index), *[_formatted(row[col], formats.get(col)) for col in frame.columns]]
            for index, row in frame.iterrows()]
    wrap = lambda text, width: textwrap.fill(text, max(9, int(width * 13)),
                                            break_long_words=False, break_on_hyphens=False)
    wrapped = [[wrap(value, width) for value, width in zip(row, widths)] for row in rows]
    heights = [max(0.59, 0.27 * max(value.count("\n") + 1 for value in row) + 0.26)
               for row in wrapped]
    header_height = 0.67
    title_height = 1.0 if subtitle else 0.77
    width = sum(widths)
    height = title_height + header_height + sum(heights) + 0.25
    fig = plt.figure(figsize=(width, height), facecolor="#F4F7FC")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.set_axis_off()
    ax.add_patch(Rectangle((0.05, height - 0.48), 0.06, 0.28, color=TEAL))
    ax.text(0.23, height - 0.31, title, fontsize=15, weight="bold", color=INK, va="center")
    if subtitle:
        ax.text(0.23, height - 0.68, subtitle, fontsize=9.5, color=MUTED, va="center")
    top = height - title_height
    ax.add_patch(FancyBboxPatch((0.02, top - header_height), width - 0.04, header_height,
                               boxstyle="round,pad=0,rounding_size=0.06",
                               facecolor=INK, edgecolor="none"))
    x = 0
    for i, (name, col_width) in enumerate(zip(headings, widths)):
        ax.text(x + 0.17 if i == 0 else x + col_width / 2,
                top - header_height / 2, wrap(name, col_width),
                color="white", fontsize=9.5, weight="bold",
                ha="left" if i == 0 else "center", va="center")
        x += col_width
    y = top - header_height
    for row_number, (row, row_height) in enumerate(zip(wrapped, heights)):
        row_name = frame.index[row_number]
        fill = "#E5F4F0" if row_name == accent_row else ("white" if row_number % 2 == 0 else "#EDF2F9")
        ax.add_patch(Rectangle((0.02, y - row_height), width - 0.04, row_height,
                               facecolor=fill, edgecolor="none"))
        if row_name == accent_row:
            ax.add_patch(Rectangle((0.02, y - row_height), 0.06, row_height,
                                   facecolor=TEAL, edgecolor="none"))
        x = 0
        for i, (value, col_width) in enumerate(zip(row, widths)):
            column = frame.columns[i - 1] if i else None
            numeric = i > 0 and pd.api.types.is_numeric_dtype(frame[column])
            winner = column in highlight_columns and frame.loc[row_name, column] == frame[column].max()
            if winner:
                ax.add_patch(FancyBboxPatch((x + 0.09, y - row_height + 0.1),
                                           col_width - 0.18, row_height - 0.2,
                                           boxstyle="round,pad=0,rounding_size=0.07",
                                           facecolor="#C7EDE5", edgecolor="#55BFA8", linewidth=0.8))
            ax.text(x + col_width - 0.19 if numeric else x + 0.17,
                    y - row_height / 2, value,
                    ha="right" if numeric else "left", va="center",
                    fontsize=10.1 if numeric else 9.7,
                    fontfamily="DejaVu Sans Mono" if numeric else "DejaVu Sans",
                    color="#075F54" if winner else INK,
                    weight="bold" if winner or (i == 0 and row_name == accent_row) else "normal",
                    linespacing=1.45)
            x += col_width
        ax.plot([0.12, width - 0.12], [y - row_height, y - row_height], color="#DDE5EF", lw=0.6)
        y -= row_height
    return fig


def display_table(frame, title, subtitle="", formats=None, widths=None,
                  highlight_columns=(), accent_row=None, index_label="Model"):
    """Show the same formatted values as an image and an expandable HTML table."""
    from IPython.display import HTML, display

    fig = table_figure(frame, title, subtitle, formats, widths,
                       highlight_columns, accent_row, index_label)
    stream = BytesIO()
    fig.savefig(stream, format="png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    display({"image/png": b64encode(stream.getvalue()).decode("ascii"),
             "text/plain": title + "\n" + frame.to_string()}, raw=True)

    formats = formats or {}
    head = "".join(f'<th style="padding:12px 14px;text-align:left;background:{INK};color:white">'
                   f'{escape(str(column))}</th>' for column in [index_label, *frame.columns])
    body = []
    for number, (index, row) in enumerate(frame.iterrows()):
        color = "#E5F4F0" if index == accent_row else ("#FFFFFF" if number % 2 == 0 else "#EDF2F9")
        cells = [f'<th scope="row" style="padding:12px 14px;text-align:left">{escape(str(index))}</th>']
        for col in frame.columns:
            value = escape(_formatted(row[col], formats.get(col)))
            align = "right" if pd.api.types.is_numeric_dtype(frame[col]) else "left"
            cells.append(f'<td style="padding:12px 14px;text-align:{align}">{value}</td>')
        body.append(f'<tr style="background:{color};color:{INK}">' + "".join(cells) + "</tr>")
    markup = (
        '<details><summary style="padding:8px;color:#087F8C;cursor:pointer">'
        'View selectable table values</summary><div style="overflow-x:auto">'
        '<table style="border-collapse:collapse;width:100%;font-family:system-ui,sans-serif;font-size:14px">'
        f'<caption style="text-align:left;padding:10px;font-weight:700">{escape(title)}</caption>'
        '<thead><tr>' + head + '</tr></thead><tbody>' + "".join(body) + '</tbody></table></div></details>'
    )
    display(HTML(markup))
