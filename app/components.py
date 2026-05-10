"""Shared Streamlit widgets used by every page."""
from pathlib import Path
import streamlit as st

CSS_PATH = Path(__file__).parent / "style.css"


def inject_styles() -> None:
    if CSS_PATH.exists():
        with open(CSS_PATH) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_disclaimer() -> None:
    st.markdown(
        '<div class="disclaimer-banner">⚠ <strong>Disclaimer:</strong> '
        'This tool is for research and decision-support only. '
        'It does not replace clinical judgment or diagnosis.</div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value, color: str = "#0F766E") -> None:
    st.markdown(
        f'<div class="kpi-card"><div class="label">{label}</div>'
        f'<div class="value" style="color:{color}">{value}</div></div>',
        unsafe_allow_html=True,
    )


def risk_badge(tier: str) -> str:
    return f'<span class="risk-badge risk-{tier}">{tier}</span>'


def confidence_bar(value: float) -> None:
    pct = int(value * 100)
    color = "#16A34A" if value >= 0.75 else ("#D97706" if value >= 0.5 else "#DC2626")
    st.markdown(
        f"<div style='background:#E2E8F0; border-radius:8px; overflow:hidden; height:14px;'>"
        f"<div style='width:{pct}%; background:{color}; height:100%;'></div></div>"
        f"<div style='font-size:0.85rem; color:#64748B; margin-top:4px;'>"
        f"Model confidence: <strong>{pct}%</strong></div>",
        unsafe_allow_html=True,
    )
