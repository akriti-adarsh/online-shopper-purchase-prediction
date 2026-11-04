"""Bridge browser appearance preferences to the chart palette."""

from pathlib import Path

import streamlit as st

_SCRIPT = Path(__file__).resolve().parents[1] / "assets" / "appearance.js"
_PREFERENCES = {"system", "light", "dark"}
_THEMES = {"light", "dark"}

_appearance_component = st.components.v2.component(
    "shopper_appearance",
    html="""
    <div class="appearance-control">
      <label for="shopper-appearance">Appearance</label>
      <select id="shopper-appearance" aria-label="Appearance">
        <option value="system">System</option>
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
    </div>
    """,
    js=_SCRIPT.read_text(encoding="utf-8"),
    isolate_styles=False,
)


def _validated_appearance(value):
    """Accept only the small state contract sent by the appearance control."""
    if not isinstance(value, dict):
        return None
    preference = value.get("preference")
    effective = value.get("effective")
    if not isinstance(preference, str) or not isinstance(effective, str):
        return None
    if preference not in _PREFERENCES or effective not in _THEMES:
        return None
    if preference != "system" and preference != effective:
        return None
    return {"preference": preference, "effective": effective}


def render_appearance(focus_request=None):
    """Render a persistent preference selector and return its resolved palette.

    Streamlit must mount the component before browser preferences can reach
    Python. The initial server pass therefore uses light until the bridge has
    reported its effective theme; later passes retain the acknowledged theme.
    """
    state = st.session_state.get("appearance_bridge", {})
    previous = _validated_appearance(state.get("appearance")) if isinstance(state, dict) else None
    focus = None
    if isinstance(focus_request, dict):
        token = focus_request.get("token")
        selector = focus_request.get("selector")
        if isinstance(token, str) and token and isinstance(selector, str) and selector:
            focus = {"token": token, "selector": selector}
    result = _appearance_component(
        key="appearance_bridge",
        data={"appearance": previous, "focus_request": focus},
        on_appearance_change=lambda: None,
    )
    current = _validated_appearance(result.appearance) or previous
    return current["effective"] if current else "light"
