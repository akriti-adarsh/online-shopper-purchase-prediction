"""Accessible session rows with keyboard-operable detail and sorting actions."""

from pathlib import Path

import streamlit.components.v2 as components

from shopper_app.presentation import OUTCOMES, formatted, text

_table = components.component(
    "shopper_session_table", html='<div class="sessions-host"></div>',
    js=(Path(__file__).resolve().parents[1] / "assets/session-table.js").read_text(encoding="utf-8"),
    isolate_styles=False,
)


def session_table(rows, score_label, caption, sort_by, ascending):
    columns = {"SessionID": "Session", "Revenue": "Actual", "PredictedRevenue": "Predicted",
               "PredictionScore": score_label, "ResultCategory": "Outcome"}
    headings = []
    for column, label in columns.items():
        if column not in rows.columns:
            headings.append(f'<th scope="col">{text(label)}</th>')
            continue
        direction = "ascending" if ascending else "descending"
        sort = f' aria-sort="{direction}"' if sort_by == column else ''
        arrow = (" ↑" if ascending else " ↓") if sort_by == column else ''
        headings.append(f'<th scope="col"{sort}><button data-sort="{column}">{text(label + arrow)}</button></th>')
    body = []
    for _, row in rows.iterrows():
        identifier = text(row["SessionID"])
        category = str(row["ResultCategory"])
        body.append(f'''<tr><td><button class="session-link" data-session="{identifier}"
            aria-label="Inspect session {identifier}">{identifier}</button></td>
            <td>{("Purchase" if bool(row["Revenue"]) else "No purchase") if "Revenue" in row else "Unavailable"}</td>
            <td>{"Purchase" if bool(row["PredictedRevenue"]) else "No purchase"}</td>
            <td class="numeric">{formatted(row["PredictionScore"], True)}</td>
            <td><span class="category {OUTCOMES.get(category, 'neutral')}">{text(category)}</span></td></tr>''')
    markup = (f'<div class="ledger-wrap" role="region" aria-label="Session results" tabindex="0">'
              f'<table class="ledger-table"><caption>{text(caption)}</caption><thead><tr>{"".join(headings)}</tr></thead>'
              f'<tbody>{"".join(body)}</tbody></table></div>')
    return _table(data={"html": markup}, key="sessions_table", on_selected_change=lambda: None,
                  on_sort_change=lambda: None)
