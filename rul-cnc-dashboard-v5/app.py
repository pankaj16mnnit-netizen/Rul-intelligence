"""
CNC Machining Centre — Remaining Useful Life (RUL) Intelligence

Page 1  Model comparison   : all five models, ranked from the executed notebook run
Page 2  CNN-LSTM prediction: upload sensor history and get RUL from the CNN-LSTM model

Run locally:  streamlit run app.py
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.inference import ModelBundle
from src.metrics import (DEFAULT_RESULTS, RANK_OPTIONS, best_model, leaders_by_metric, pred_column,
                         rank_position, regression_metrics, status_for)
from src.ui import (MODEL_COLORS, STATUS_COLORS, apply_theme, banner, badge, footer, hero,
                    kpi, model_color, section, show_chart, show_table, sidebar_brand, style_fig)

ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "results" / "model_comparison.csv"
TEST_PRED_FILE = ROOT / "results" / "test_predictions.csv"
CSV_SAMPLE = ROOT / "data" / "sample_upload.csv"
EXCEL_SAMPLE = ROOT / "data" / "CNC_sample_upload_heldout.xlsx"
TARGET = "CNN-LSTM (tuned)"
PAGE_COMPARE = "📊  Model comparison"
PAGE_CNN = "🧠  CNN-LSTM prediction"

st.set_page_config(page_title="CNC RUL Intelligence", page_icon="⚙️", layout="wide",
                   initial_sidebar_state="expanded")
apply_theme()


# ---------- cached data ----------
@st.cache_data(show_spinner=False)
def load_results() -> tuple[pd.DataFrame, str]:
    if RESULTS_FILE.exists():
        return pd.read_csv(RESULTS_FILE), "results/model_comparison.csv"
    return DEFAULT_RESULTS.copy(), "built-in copy of the executed run"


@st.cache_data(show_spinner=False)
def load_test_predictions() -> pd.DataFrame | None:
    return pd.read_csv(TEST_PRED_FILE) if TEST_PRED_FILE.exists() else None


@st.cache_resource(show_spinner="Loading the CNN-LSTM model…")
def load_cnn_bundle():
    try:
        bundle = ModelBundle(load_tabular=False)
    except FileNotFoundError as exc:
        return None, str(exc)
    bundle.load_deep(only=(TARGET,))
    if TARGET not in bundle.deep:
        return None, ("CNN-LSTM weights could not be loaded. Check that models/ contains "
                      "cnn_lstm_seed*.keras and that TensorFlow is installed.")
    return bundle, None


def read_upload(uploaded) -> pd.DataFrame:
    if uploaded.name.lower().endswith(".xlsx"):
        return pd.read_excel(uploaded, sheet_name=0)
    return pd.read_csv(uploaded)


results, results_src = load_results()

# ---------- sidebar ----------
with st.sidebar:
    sidebar_brand()
    page = st.radio("Navigate", [PAGE_COMPARE, PAGE_CNN], label_visibility="collapsed")
    st.divider()
    st.markdown("**Maintenance guidance**")
    st.caption("Predicted RUL thresholds (cycles). Set these to your maintenance policy.")
    critical_below = st.slider("Critical if RUL below", 0, 200, 50, 5)
    warning_below = st.slider("Warning if RUL below", 0, 300, 100, 5)
    if warning_below <= critical_below:
        st.warning("Warning threshold must be above the critical threshold.")
        warning_below = critical_below + 5
    st.divider()
    st.caption(f"Results source: {results_src}")

# =====================================================================
# PAGE 1 — MODEL COMPARISON
# =====================================================================
if page == PAGE_COMPARE:
    hero("Model comparison",
         "Five prognostic models on 20 held-out machines (8,387 rows). The leader depends on the metric: "
         "LSTM leads the accuracy metrics, CNN-LSTM leads the late-prediction penalty.",
         ["20 held-out machines", "8,387 test rows", "5 models", "3-seed average for deep models"])

    winners = leaders_by_metric(results)
    lbl_r2, lbl_rmse = "R² (higher is better)", "RMSE (lower is better)"
    lbl_mae, lbl_asym = "MAE (lower is better)", "Asymmetric score (lower is better)"
    cnn = results[results["Model"] == TARGET].iloc[0]
    lstm = results[results["Model"] == "LSTM (tuned)"].iloc[0]
    rank_r2 = rank_position(results, TARGET, lbl_r2)
    rank_asym = rank_position(results, TARGET, lbl_asym)

    cnn_wins = [k.split(" (")[0] for k, v in winners.items() if v == TARGET]
    if len(cnn_wins) == len(winners):
        banner("success", "CNN-LSTM leads every metric in this run.")
    elif cnn_wins:
        banner("success", f"CNN-LSTM is the best model on the {' and '.join(cnn_wins)} metric"
               f"{'s' if len(cnn_wins) > 1 else ''}.",
               f"LSTM leads R², RMSE and MAE. The gap between the two sequence models on R² is "
               f"{abs(cnn['R2'] - lstm['R2']):.3f}, which is within seed-to-seed variation.")
    else:
        banner("info", "CNN-LSTM is not the top model on any metric in this run.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Top model on R²", winners[lbl_r2].replace(" (tuned)", ""),
            f"R² {best_model(results, lbl_r2)[1]:.3f}", "blue", "🏅")
    with c2:
        kpi("CNN-LSTM R²", f"{cnn['R2']:.3f}", f"± {cnn['R2_std']:.3f} · rank {rank_r2} of {len(results)}", "teal", "🧠")
    with c3:
        kpi("Top model on asymmetric score", winners[lbl_asym].replace(" (tuned)", ""),
            f"CNN-LSTM rank {rank_asym} of {len(results)} · lower is better", "violet", "🎯")
    with c4:
        kpi("CNN-LSTM vs LSTM (R²)", f"{cnn['R2'] - lstm['R2']:+.3f}",
            f"seed std ≈ {max(cnn['R2_std'], lstm['R2_std']):.3f}", "amber", "⚖️")

    section("Leader by metric")
    lead_rows = [{"Metric": k.split(" (")[0], "Leader": v,
                  "Leader's value": f"{best_model(results, k)[1]:,.3f}" if "R²" in k
                  else f"{best_model(results, k)[1]:,.2f}" if "Asym" not in k
                  else f"{best_model(results, k)[1]:,.0f}"} for k, v in winners.items()]
    show_table(pd.DataFrame(lead_rows))

    section("Performance leaderboard")
    table = results[["Model", "R2", "R2_std", "RMSE", "RMSE_std", "MAE", "AsymScore"]].copy()
    table = table.sort_values("R2", ascending=False).reset_index(drop=True)
    display = pd.DataFrame({
        "Rank": range(1, len(table) + 1),
        "Model": table["Model"],
        "R²": [f"{a:.3f}" + ("" if pd.isna(b) else f" ± {b:.3f}") for a, b in zip(table.R2, table.R2_std)],
        "RMSE (cycles)": [f"{a:.2f}" + ("" if pd.isna(b) else f" ± {b:.2f}") for a, b in zip(table.RMSE, table.RMSE_std)],
        "MAE (cycles)": [f"{a:.2f}" for a in table.MAE],
        "Asymmetric score": [f"{a:,.0f}" for a in table.AsymScore],
    })
    styled = display.style.apply(
        lambda row: ["background-color: rgba(0,229,195,0.20); font-weight: 700; color: #eafffb"] * len(row)
        if row["Model"] == TARGET else [""] * len(row), axis=1)
    show_table(styled)
    st.caption("Leaderboard: deep models show the mean ± standard deviation of the 3 seeds' individual metrics. "
               "Tabular models: single fit.")

    left, right = st.columns([1.2, 1])
    with left:
        section("Ranking by R²")
        ordered = results.sort_values("R2", ascending=False)
        fig = go.Figure(go.Bar(
            x=ordered["Model"], y=ordered["R2"],
            marker=dict(color=[model_color(m) for m in ordered["Model"]], line=dict(width=0)),
            text=[f"{v:.3f}" for v in ordered["R2"]], textposition="outside",
            error_y=dict(type="data", array=ordered["R2_std"].fillna(0), visible=True, color="#cfe3df"),
        ))
        fig.update_yaxes(range=[0.90, 0.97], title="R²")
        show_chart(style_fig(fig, height=400))

    with right:
        section("Strength profile")
        axes = ["R²", "1/RMSE", "1/MAE", "1/Asym"]
        def scaled(col, lower_better):
            lo, hi = results[col].min(), results[col].max()
            s = (results[col] - lo) / (hi - lo + 1e-12)
            return (1 - s) if lower_better else s
        radar = go.Figure()
        for i, r in results.iterrows():
            vals = [scaled("R2", False)[i], scaled("RMSE", True)[i], scaled("MAE", True)[i], scaled("AsymScore", True)[i]]
            radar.add_trace(go.Scatterpolar(
                r=vals + vals[:1], theta=axes + axes[:1], fill="toself", name=r["Model"],
                line=dict(color=model_color(r["Model"]), width=3 if r["Model"] == TARGET else 1.5),
                opacity=0.7 if r["Model"] == TARGET else 0.35))
        radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1]),
                                       bgcolor="rgba(255,255,255,0.02)",
                                       angularaxis=dict(gridcolor="rgba(255,255,255,0.08)")))
        show_chart(style_fig(radar, height=400))
    st.caption("Outer ring = better on that metric, scaled across all five models.")

    tp = load_test_predictions()
    section("Held-out machine: predicted vs actual RUL")
    if tp is None:
        st.info("Add results/test_predictions.csv from the notebook's outputs folder to show the "
                "machine-level comparison here.")
    else:
        machines = sorted(tp["unit_id"].unique().tolist())
        unit = st.selectbox("Held-out machine", machines, index=0)
        sub = tp[tp["unit_id"] == unit].sort_values("cycle")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub["cycle"], y=sub["RUL_clipped"], name="Actual RUL",
                                 line=dict(color="#FFFFFF", width=3.5)))
        for m in MODEL_COLORS:
            c = pred_column(m)
            if c in sub.columns:
                fig.add_trace(go.Scatter(x=sub["cycle"], y=sub[c], name=m,
                                         line=dict(color=model_color(m),
                                                   width=3 if m == TARGET else 1.5,
                                                   dash="solid" if m == TARGET else "dot")))
        fig.update_xaxes(title="Cycle")
        fig.update_yaxes(title="RUL (cycles, capped at 130)")
        show_chart(style_fig(fig, f"Machine #{unit} · 3-seed averaged predictions", height=440))

        rows = []
        for m in MODEL_COLORS:
            c = pred_column(m)
            if c in tp.columns:
                here = regression_metrics(sub["RUL_clipped"], sub[c])
                whole = regression_metrics(tp["RUL_clipped"], tp[c])
                rows.append({"Model": m, "This machine R²": round(here["R2"], 3),
                             "This machine RMSE": round(here["RMSE"], 2),
                             "All test R²": round(whole["R2"], 3), "All test RMSE": round(whole["RMSE"], 2)})
        show_table(pd.DataFrame(rows))
        st.caption("This section uses the 3-seed averaged predictions for each deep model, so its metrics are "
                   "slightly different from the leaderboard, which averages the seeds' individual metrics.")

    footer()

# =====================================================================
# PAGE 2 — CNN-LSTM PREDICTION
# =====================================================================
else:
    hero("CNN-LSTM RUL prediction",
         "Upload your sensor history. The CNN-LSTM model returns the remaining useful life of every machine, "
         "cycle by cycle.",
         ["Seed-averaged CNN-LSTM", "30-cycle causal windows", "CSV or Excel"])

    bundle, err = load_cnn_bundle()
    if err:
        st.error(err)
        st.info("Complete the Colab export and copy outputs/models/ (including preprocessing.json and the "
                "cnn_lstm_seed*.keras files) into this project's models/ folder.")
    else:
        for w in bundle.warnings:
            st.warning(w)

        with st.expander("How to prepare your file", expanded=False):
            st.markdown(
                "- One row per machine cycle, in any order. Include `unit_id` and `cycle` so each machine's "
                "history can be built.\n"
                f"- Required sensor columns ({len(bundle.cfg['sensor_cols'])}): "
                + ", ".join(f"`{c}`" for c in bundle.cfg["sensor_cols"]) + ".\n"
                "- An optional `RUL` column lets the dashboard compare predictions with actual values.\n"
                "- Predictions use each machine's earlier cycles only. Short histories (under about 30 cycles) "
                "are zero-padded, as in training, and are less reliable.")

        s1, s2 = st.columns([1.3, 1])
        with s1:
            choice = st.radio("Data source", ["Upload my file", "Excel sample (held-out machines 5, 34, 81)",
                                              "CSV sample (two machines)"])
        with s2:
            window = next(e["window"] for e in bundle.entries if e["name"] == TARGET)
            st.markdown(f"<div class='panel'><b>Model in use</b><br>CNN-LSTM (tuned)<br>"
                        f"<span class='pill'>window {window} cycles</span> "
                        "<span class='pill'>3-seed average</span></div>", unsafe_allow_html=True)

        raw = None
        if choice == "Upload my file":
            up = st.file_uploader("Sensor data (CSV or Excel)", type=["csv", "xlsx"])
            if up is not None:
                raw = read_upload(up)
        elif choice.startswith("Excel"):
            raw = pd.read_excel(EXCEL_SAMPLE, sheet_name=0) if EXCEL_SAMPLE.exists() else None
        else:
            raw = pd.read_csv(CSV_SAMPLE) if CSV_SAMPLE.exists() else None

        if raw is None:
            st.info("Upload a file or choose a sample to see CNN-LSTM predictions.")
        else:
            with st.spinner("Running the CNN-LSTM model on your data…"):
                try:
                    pred = bundle.predict_model(raw, TARGET)
                    error = None
                except (ValueError, RuntimeError) as exc:
                    pred, error = None, str(exc)

            if error:
                st.error(error)
            else:
                st.toast("CNN-LSTM predictions ready", icon="🧠")
                for note in bundle.quality_notes(pred):
                    st.warning(note)

                latest = pred.sort_values(["unit_id", "cycle"]).groupby("unit_id").tail(1).reset_index(drop=True)
                latest["Status"] = [status_for(v, critical_below, warning_below) for v in latest[TARGET]]

                n_crit = int((latest["Status"] == "Critical").sum())
                n_warn = int((latest["Status"] == "Warning").sum())
                n_ok = int((latest["Status"] == "Healthy").sum())
                section("Summary")
                k1, k2, k3, k4 = st.columns(4)
                with k1: kpi("Machines scored", f"{len(latest)}", f"{len(pred):,} cycles analysed", "teal", "🛠️")
                with k2: kpi("Critical", f"{n_crit}", f"RUL below {critical_below} cycles", "pink", "🔴")
                with k3: kpi("Warning", f"{n_warn}", f"RUL below {warning_below} cycles", "amber", "🟠")
                with k4: kpi("Healthy", f"{n_ok}", "above the warning threshold", "blue", "🟢")

                section("Latest RUL per machine")
                cards = st.columns(min(4, len(latest)))
                for card, (_, r) in zip(cards, latest.head(4).iterrows()):
                    with card:
                        kpi(f"Machine {int(r['unit_id'])}", f"{r[TARGET]:.0f} cycles",
                            f"{badge(r['Status'])} &nbsp; cycle {int(r['cycle'])}",
                            {"Critical": "pink", "Warning": "amber", "Healthy": "teal"}[r["Status"]], "🔧")

                show_cols = ["unit_id", "cycle", "history_rows", TARGET, "Status"]
                if "RUL_actual" in latest.columns:
                    latest["Error (pred − actual)"] = (latest[TARGET] - latest["RUL_actual"]).round(1)
                    show_cols += ["RUL_actual", "Error (pred − actual)"]
                show_table(latest[show_cols].round(1).rename(columns={TARGET: "Predicted RUL (cycles)"}))
                if "RUL_actual" in latest.columns:
                    st.caption("Actual RUL is capped at 130 cycles, as in training. Positive error = the model "
                               "predicts more life than remains (a late prediction).")

                section("Trajectory and gauge")
                unit = st.selectbox("Machine", sorted(pred["unit_id"].unique().tolist()))
                sub = pred[pred["unit_id"] == unit].sort_values("cycle")
                latest_val = float(sub[TARGET].iloc[-1])
                g_col, t_col = st.columns([1, 2])
                with g_col:
                    gmax = float(max(250, np.ceil(sub[TARGET].max() / 50) * 50))
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number", value=latest_val, number=dict(suffix=" cycles", font=dict(color="#e8f7f4")),
                        title=dict(text=f"Machine {unit} · latest RUL", font=dict(color="#cfe3df")),
                        gauge=dict(
                            axis=dict(range=[0, gmax], tickcolor="#cfe3df"),
                            bar=dict(color=STATUS_COLORS[status_for(latest_val, critical_below, warning_below)]),
                            bgcolor="rgba(255,255,255,0.03)",
                            steps=[dict(range=[0, critical_below], color="rgba(255,77,109,0.35)"),
                                   dict(range=[critical_below, warning_below], color="rgba(255,181,71,0.35)"),
                                   dict(range=[warning_below, gmax], color="rgba(0,229,195,0.25)")])))
                    gauge.update_layout(template="plotly_dark", height=340, paper_bgcolor="rgba(0,0,0,0)",
                                        margin=dict(l=16, r=16, t=50, b=10))
                    show_chart(gauge)
                with t_col:
                    traj = go.Figure()
                    traj.add_trace(go.Scatter(x=sub["cycle"], y=sub[TARGET], name="CNN-LSTM predicted RUL",
                                              mode="lines", line=dict(color=MODEL_COLORS[TARGET], width=3),
                                              fill="tozeroy", fillcolor="rgba(0,229,195,0.10)"))
                    if "RUL_actual" in sub.columns:
                        traj.add_trace(go.Scatter(x=sub["cycle"], y=sub["RUL_actual"], name="Actual RUL",
                                                  line=dict(color="#FFFFFF", width=2, dash="dot")))
                    traj.add_hline(y=critical_below, line=dict(color=STATUS_COLORS["Critical"], dash="dash"),
                                   annotation_text="Critical", annotation_position="right")
                    traj.add_hline(y=warning_below, line=dict(color=STATUS_COLORS["Warning"], dash="dash"),
                                   annotation_text="Warning", annotation_position="right")
                    traj.update_xaxes(title="Cycle")
                    traj.update_yaxes(title="Remaining useful life (cycles)")
                    show_chart(style_fig(traj, "Predicted RUL across the machine's life", height=340))

                csv_buf = StringIO()
                pred.to_csv(csv_buf, index=False)
                st.download_button("⬇️ Download CNN-LSTM predictions (CSV)", csv_buf.getvalue(),
                                   file_name="cnn_lstm_rul_predictions.csv", mime="text/csv")

    footer()
