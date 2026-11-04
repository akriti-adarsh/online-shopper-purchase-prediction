"""Filter and export prediction rows without changing their session identity."""

import pandas as pd

from shopper_app.evaluation import RESULT_CATEGORIES

FILTER_OPTIONS = [
    "All sessions", "Correct predictions", "Incorrect predictions", *RESULT_CATEGORIES,
]


def filter_sessions(
    predictions: pd.DataFrame,
    category: str = "All sessions",
    search: str = "",
    sort_by: str = "OriginalRow",
    ascending: bool = True,
) -> pd.DataFrame:
    """Return all matching rows; text search is literal and case-insensitive.

    Search covers original features and result fields. OriginalRow breaks sort
    ties so paging stays deterministic. Neither row indices nor IDs are reset.
    """
    if category not in FILTER_OPTIONS:
        raise ValueError("Choose a supported session outcome filter.")
    if sort_by not in predictions.columns:
        raise ValueError("Choose a column present in the session table for sorting.")
    rows = predictions
    if category == "Correct predictions":
        rows = rows.loc[rows["Correct"].eq(True).fillna(False)]
    elif category == "Incorrect predictions":
        rows = rows.loc[rows["Correct"].eq(False).fillna(False)]
    elif category != "All sessions":
        rows = rows.loc[rows["ResultCategory"].eq(category)]
    query = str(search).strip()
    if query:
        matches = pd.Series(False, index=rows.index)
        for column in rows.columns:
            matches |= rows[column].astype("string").str.contains(
                query, case=False, regex=False, na=False,
            )
        rows = rows.loc[matches]
    columns = [sort_by] if sort_by == "OriginalRow" else [sort_by, "OriginalRow"]
    directions = [ascending] if sort_by == "OriginalRow" else [ascending, True]
    return rows.sort_values(
        columns, ascending=directions, kind="stable", na_position="last",
    ).copy()


def paginate_sessions(frame: pd.DataFrame, page: int = 1, page_size: int = 25) -> dict:
    """Return a one-based page, clamped after filtering, with inclusive positions.

    Empty filters have count/pages/start/end of zero and page one. The rows
    value is a frame retaining its original indices and stable SessionID values.
    """
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ValueError("Page must be a positive whole number.")
    if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= 500:
        raise ValueError("Page size must be a whole number between 1 and 500.")
    count = len(frame)
    pages = (count + page_size - 1) // page_size
    page = min(page, max(pages, 1))
    offset = (page - 1) * page_size
    return {
        "rows": frame.iloc[offset:offset + page_size].copy(),
        "count": count,
        "pages": pages,
        "page": page,
        "page_size": page_size,
        "start": offset + 1 if count else 0,
        "end": min(offset + page_size, count),
    }


def export_sessions(frame: pd.DataFrame) -> bytes:
    """Export every supplied filtered row, including its original features and ID."""
    summary = [
        "SessionID", "OriginalRow", "Revenue", "PredictedRevenue",
        "PredictionScore", "ResultCategory", "Correct",
    ]
    columns = [column for column in summary if column in frame.columns]
    columns += [column for column in frame.columns if column not in columns]
    return frame.loc[:, columns].to_csv(index=False, lineterminator="\n").encode("utf-8")


def session_detail(predictions: pd.DataFrame, session_id: str) -> pd.Series:
    """Find exactly one session in the complete active prediction result."""
    rows = predictions.loc[predictions["SessionID"].eq(session_id)]
    if rows.empty:
        raise KeyError("That session is no longer present in the active dataset.")
    if len(rows) != 1:
        raise ValueError("Session identifiers must be unique within a dataset.")
    return rows.iloc[0].copy()
