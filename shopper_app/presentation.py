"""Escaped, semantic presentation helpers for the retail ledger interface."""

from base64 import b64encode
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUTCOMES = {
    "Correct non-buyers": "neutral", "Identified buyers": "buyer",
    "Missed buyers": "missed", "False alerts": "false", "Unlabeled": "neutral",
}


def text(value):
    return escape(str(value), quote=True)


def install_styles():
    st.html("<style>" + (ROOT / "assets/app.css").read_text(encoding="utf-8") + "</style>")


def brand():
    encoded = b64encode((ROOT / "assets/brand.svg").read_bytes()).decode("ascii")
    mark = f'<img src="data:image/svg+xml;base64,{encoded}" alt="Shopper Signals receipt mark" width="38" height="38">'
    st.html(f'<div class="brand">{mark}<div><strong>Shopper Signals</strong>'
            '<small>A closer look at purchase decisions</small></div></div>')


def heading(title, number):
    st.html(f'<div class="panel-heading"><h3>{text(title)}</h3><span class="index">{text(number)}</span></div>')


def finding(result, count):
    if not result.get("labeled", True):
        purchases = int(result["predictions"]["PredictedRevenue"].sum())
        sentence = f'The model flagged <strong>{purchases:,} of {count:,} sessions</strong> as purchases.'
        note = "These are predictions. Add the Revenue column to measure how often they are correct."
    else:
        _, _, fn, tp = result["confusion"].ravel()
        accuracy = result["metrics"]["Accuracy"]
        sentence = (f'The model correctly classified <strong>{accuracy:.2%}</strong> of sessions '
                    f'and identified <strong>{tp:,} of the {tp + fn:,} buyers</strong>.')
        note = (f"{fn:,} actual buyers were missed. Select an outcome below to inspect its sessions."
                if tp + fn else "This dataset has no observed buyers. Purchase recall cannot be estimated.")
    st.html(f'<section class="finding" aria-label="Evaluation finding"><p>{sentence}</p><small>{text(note)}</small></section>')


def coverage_bar(title, question, positive, other, other_label, kind):
    total = int(positive + other)
    share = positive / total if total else 0
    description = (f"{positive:,} identified, {other:,} {other_label.lower()}, out of {total:,}."
                   if total else "No sessions in this group.")
    positive_text = f"{positive:,}" if share >= .18 else ""
    other_text = f"{other:,}" if 1 - share >= .18 and total else ""
    pct_a, pct_b = (f"{share:.1%}", f"{1-share:.1%}") if total else ("unavailable", "unavailable")
    st.html(f'''<section class="coverage"><div class="coverage-head"><strong>{text(title)}</strong>
        <span>{total:,} total</span></div><p class="coverage-question">{text(question)}</p>
        <div class="coverage-track" role="img" aria-label="{text(description)}">
        <span class="buyer" style="width:{share*100:.5f}%">{positive_text}</span>
        <span class="{kind}" style="width:{(1-share)*100 if total else 0:.5f}%">{other_text}</span></div>
        <div class="coverage-values"><span><i class="swatch buyer" aria-hidden="true"></i>
        {positive:,} identified · {pct_a}</span><span><i class="swatch {kind}" aria-hidden="true"></i>
        {other:,} {text(other_label.lower())} · {pct_b}</span></div></section>''')


def formatted(value, numeric=False):
    if value is None or value is pd.NA or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return '<span title="Not available for these data">—</span>'
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}" if numeric else text(f"{value:g}")
    return text(value)


def table_html(frame, caption, *, selected=None, numeric_columns=(), index=False):
    columns = list(frame.columns)
    if index:
        frame = frame.reset_index()
        columns = list(frame.columns)
    heads = ''.join(f'<th scope="col" class="{"numeric" if c in numeric_columns else ""}">{text(c)}</th>' for c in columns)
    rows = []
    for _, row in frame.iterrows():
        active = selected is not None and str(row.iloc[0]) == str(selected)
        cells = []
        for i, c in enumerate(columns):
            value = formatted(row[c], c in numeric_columns)
            if active and i == 0:
                value += '<span class="selected-label">Selected model</span>'
            cells.append(f'<td class="{"numeric" if c in numeric_columns else ""}">{value}</td>')
        rows.append(f'<tr class="{"selected-row" if active else ""}">' + ''.join(cells) + '</tr>')
    return (f'<div class="ledger-wrap" role="region" aria-label="{text(caption)}" tabindex="0">'
            f'<table class="ledger-table"><caption>{text(caption)}</caption><thead><tr>{heads}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def ledger_table(frame, caption, **kwargs):
    st.html(table_html(frame, caption, **kwargs))
