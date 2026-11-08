"""Production theme: colours, layout components and Plotly styling."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# Model colours (CNN-LSTM is the highlighted hero colour)
MODEL_COLORS = {
    "CNN-LSTM (tuned)": "#00E5C3",
    "LSTM (tuned)": "#4DA8FF",
    "Random Forest": "#FFC04D",
    "Gradient Boosting": "#FF6F91",
    "Decision Tree": "#B388FF",
}
STATUS_COLORS = {"Critical": "#FF4D6D", "Warning": "#FFB547", "Healthy": "#00E5C3"}
ACCENTS = {"teal": "#00E5C3", "blue": "#4DA8FF", "amber": "#FFC04D", "pink": "#FF6F91", "violet": "#B388FF"}


def model_color(name: str) -> str:
    return MODEL_COLORS.get(name, "#9AA4B2")


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame { font-family: 'Inter', system-ui, sans-serif; }
[data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 600px at 10% -10%, rgba(0,229,195,0.10), transparent 60%),
              radial-gradient(900px 500px at 100% 0%, rgba(77,168,255,0.10), transparent 60%),
              linear-gradient(180deg, #0b1220 0%, #0e1a2b 100%);
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px; }

/* sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a1b2e 0%, #0d2338 60%, #0b1f30 100%);
  border-right: 1px solid rgba(255,255,255,0.06); }
[data-testid="stSidebar"] * { color: #e8f7f4; }
.brand { display:flex; align-items:center; gap:10px; margin-bottom: 6px; }
.brand .logo { width:38px; height:38px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg,#00E5C3,#4DA8FF); font-size:20px; box-shadow:0 6px 18px rgba(0,229,195,.35); }
.brand .name { font-weight:800; font-size:1.05rem; line-height:1.1; }
.brand .sub { font-size:.72rem; opacity:.65; letter-spacing:.06em; text-transform:uppercase; }

/* hero */
.hero { border-radius: 20px; padding: 26px 30px; margin-bottom: 18px; position: relative; overflow: hidden;
  background: linear-gradient(120deg, #00bfa6 0%, #1f7ae0 52%, #6a3de8 100%);
  box-shadow: 0 18px 40px rgba(31,122,224,0.25); }
.hero::before { content:""; position:absolute; right:-80px; top:-80px; width:260px; height:260px; border-radius:50%;
  background: radial-gradient(circle, rgba(255,255,255,0.35), transparent 70%); }
.hero h1 { color:#fff; font-size:2rem; font-weight:800; margin:0 0 6px 0; letter-spacing:-0.01em; }
.hero p { color: rgba(255,255,255,0.92); margin:0; font-size:1.02rem; }
.hero .chip { display:inline-block; margin-top:12px; padding:4px 12px; border-radius:999px; font-size:.78rem;
  background: rgba(255,255,255,0.18); color:#fff; font-weight:600; margin-right:6px; }
.live { display:inline-block; width:9px; height:9px; border-radius:50%; background:#fff; margin-right:8px;
  animation: pulse 1.8s infinite; vertical-align: middle; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(255,255,255,.8)} 70%{box-shadow:0 0 0 10px rgba(255,255,255,0)}
  100%{box-shadow:0 0 0 0 rgba(255,255,255,0)} }

/* KPI cards */
.kpi { border-radius: 18px; padding: 18px 20px; margin-bottom: 14px; color:#0b1220; position:relative; overflow:hidden;
  background: linear-gradient(135deg, var(--c1), var(--c2)); box-shadow: 0 12px 28px rgba(0,0,0,.28);
  transition: transform .2s ease, box-shadow .2s ease; min-height: 118px; }
.kpi:hover { transform: translateY(-3px); box-shadow: 0 18px 34px rgba(0,0,0,.35); }
.kpi .label { font-size:.74rem; text-transform:uppercase; letter-spacing:.09em; font-weight:700; opacity:.78; }
.kpi .value { font-size:1.75rem; font-weight:800; margin:6px 0 2px 0; letter-spacing:-0.01em; }
.kpi .sub { font-size:.84rem; font-weight:600; opacity:.85; }
.kpi .icon { position:absolute; right:14px; top:12px; font-size:1.5rem; opacity:.85; }

/* panels */
.panel { border-radius:18px; padding:16px 18px; background: rgba(255,255,255,0.035);
  border: 1px solid rgba(255,255,255,0.07); margin-bottom: 14px; }
.section { font-size:1.12rem; font-weight:700; margin: 18px 0 8px 0; display:flex; align-items:center; gap:10px; }
.section::before { content:""; width:4px; height:18px; border-radius:4px; background: linear-gradient(180deg,#00E5C3,#4DA8FF); }
.banner { border-radius:16px; padding:14px 18px; margin: 4px 0 14px 0; font-weight:600; display:flex; gap:12px; align-items:flex-start; }
.banner.success { background: linear-gradient(90deg, rgba(0,229,195,.18), rgba(77,168,255,.10)); border:1px solid rgba(0,229,195,.45); }
.banner.info { background: rgba(77,168,255,.12); border:1px solid rgba(77,168,255,.4); }
.banner.warn { background: rgba(255,181,71,.12); border:1px solid rgba(255,181,71,.45); }
.banner small { font-weight:500; opacity:.85; display:block; margin-top:3px; }

.badge { display:inline-block; padding:3px 11px; border-radius:999px; font-size:.76rem; font-weight:800; color:#0b1220; }
.pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.74rem; font-weight:700;
  background: rgba(0,229,195,.16); color:#7ff5e1; border:1px solid rgba(0,229,195,.4); }
.footer { text-align:center; opacity:.55; font-size:.78rem; margin-top:30px; }

/* tabs & tables */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 12px 12px 0 0; padding: 10px 18px; font-weight:600; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,0.07); }
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.markdown(
        '<div class="brand"><div class="logo">⚙️</div><div><div class="name">RUL Intelligence</div>'
        '<div class="sub">CNC · CNN-LSTM</div></div></div>', unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips: list[str] | None = None) -> None:
    chip_html = "".join(f'<span class="chip">{c}</span>' for c in (chips or []))
    st.markdown(
        f'<div class="hero"><h1><span class="live"></span>{title}</h1><p>{subtitle}</p>{chip_html}</div>',
        unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", accent: str = "teal", icon: str = "") -> None:
    c = ACCENTS.get(accent, accent)
    st.markdown(
        f'<div class="kpi" style="--c1:{c};--c2:{c}cc"><div class="icon">{icon}</div>'
        f'<div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True)


def banner(kind: str, title: str, detail: str = "") -> None:
    detail_html = f"<small>{detail}</small>" if detail else ""
    icon = {"success": "🏆", "info": "ℹ️", "warn": "⚠️"}.get(kind, "ℹ️")
    st.markdown(f'<div class="banner {kind}"><div style="font-size:1.3rem">{icon}</div>'
                f'<div>{title}{detail_html}</div></div>', unsafe_allow_html=True)


def badge(status: str) -> str:
    return f'<span class="badge" style="background:{STATUS_COLORS.get(status, "#9AA4B2")}">{status}</span>'


def section(title: str) -> None:
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, x=0.01, font=dict(size=15, color="#e8f7f4")) if title else None,
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(family="Inter, sans-serif", color="#cfe3df"),
        margin=dict(l=16, r=16, t=56 if title else 20, b=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        hovermode="x unified", colorway=list(MODEL_COLORS.values()),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig


def show_chart(fig: go.Figure) -> None:
    try:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def show_table(data, **kwargs) -> None:
    try:
        st.dataframe(data, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, hide_index=True, **kwargs)


def footer() -> None:
    st.markdown('<div class="footer">CNC RUL Intelligence · CNN-LSTM dashboard · Results from the executed '
                'notebook run on synthetic run-to-failure data</div>', unsafe_allow_html=True)
