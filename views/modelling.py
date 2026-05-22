"""Modelling page — full suite of credit model diagnostics."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
import streamlit as st

from core.pipeline import train_models, ENGINEERING_DECISIONS
from core.metrics import confusion_matrix_at_threshold
from utils.data_loader import load_loan_book

# ── Palette ────────────────────────────────────────────────────────────────
TEAL   = "#28949C"
DARK   = "#082D2F"
LIGHT  = "#65CED1"
GREEN  = "#10B981"
RED    = "#EF4444"
MUTED  = "#94A3B8"
WHITE  = "#FFFFFF"

st.title("Modelling")
st.caption("Baseline vs improved logistic regression — every engineering decision justified by EDA.")

df_raw  = load_loan_book()
models  = train_models(df_raw)

auc_i   = models["impr_test_auc"]
gini    = 2 * auc_i - 1
pct     = (auc_i - 0.68) / (0.82 - 0.68) * 100
probs   = models["probs_test"]
actual  = models["y_test"]

# ── KPI row ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Given Baseline AUC", "0.68",         help="Competition reference model.")
c2.metric("Our Baseline AUC",   f"{models['base_test_auc']:.4f}", help="Raw numeric features.")
c3.metric("Our Improved AUC",   f"{auc_i:.4f}", delta=f"+{auc_i-0.68:.4f} vs given")
c4.metric("% to LightGBM",      f"{pct:.0f}%")
st.caption(f"Gini **{gini:.4f}**  ·  Train AUC **{models['impr_train_auc']:.4f}**  ·  Train-test gap **{models['impr_train_auc']-auc_i:+.4f}**")
st.divider()

# ══════════════════════════════════════════════════════════════════════════
# 1. ROC CURVE
# ══════════════════════════════════════════════════════════════════════════
st.subheader("ROC Curve")
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=models["fpr"], y=models["tpr"], mode="lines",
    name=f"Improved model (AUC={auc_i:.4f})", line=dict(color=TEAL, width=3)))
fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
    name="Random (0.50)", line=dict(color=MUTED, dash="dot")))
fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
    plot_bgcolor=WHITE, paper_bgcolor=WHITE, height=380, legend=dict(x=0.55, y=0.05))
st.plotly_chart(fig_roc, use_container_width=True)
st.divider()

# ══════════════════════════════════════════════════════════════════════════
# 2. SCORE DISTRIBUTION  (Plotly toggle buttons — client-side, no reload)
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Score distribution")
st.caption("How well does the model separate defaulters from non-defaulters? Toggle traces with the buttons.")

bins_sd = np.linspace(0, 1, 51)
hist_good, _ = np.histogram(probs[actual == 0], bins=bins_sd, density=True)
hist_bad,  _ = np.histogram(probs[actual == 1], bins=bins_sd, density=True)
bin_centres   = (bins_sd[:-1] + bins_sd[1:]) / 2

fig_sd = go.Figure()
fig_sd.add_trace(go.Bar(x=bin_centres, y=hist_good, name="No Default",
    marker_color=GREEN, opacity=0.7, width=0.018))
fig_sd.add_trace(go.Bar(x=bin_centres, y=hist_bad, name="Default",
    marker_color=RED, opacity=0.7, width=0.018))

threshold_sd = st.slider("Threshold line", 0.05, 0.95, 0.20, 0.01,
    key="sd_threshold", help="Show where the approval cut-off falls on the score distribution.")
fig_sd.add_vline(x=threshold_sd, line_dash="dash", line_color=DARK,
    annotation_text=f"Threshold {threshold_sd:.2f}", annotation_position="top right")

fig_sd.update_layout(
    barmode="overlay",
    xaxis_title="Predicted probability of default",
    yaxis_title="Density",
    plot_bgcolor=WHITE, paper_bgcolor=WHITE, height=380,
    legend=dict(x=0.72, y=0.95),
    updatemenus=[dict(
        type="buttons", direction="right", x=0.0, y=1.12,
        buttons=[
            dict(label="Both",       method="update", args=[{"visible": [True, True]}]),
            dict(label="No Default", method="update", args=[{"visible": [True, False]}]),
            dict(label="Default",    method="update", args=[{"visible": [False, True]}]),
        ]
    )]
)
st.plotly_chart(fig_sd, use_container_width=True)
st.divider()

# ══════════════════════════════════════════════════════════════════════════
# 3. GAINS / LIFT CURVE  (Plotly toggle — switch between Gains and Lift)
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Gains & Lift curve")
st.caption("Use the buttons inside the chart to switch between Gains and Lift views.")

n            = len(actual)
total_pos    = actual.sum()
idx_sorted   = np.argsort(-probs)
sorted_actual = actual[idx_sorted]

pop_pct   = np.arange(1, n + 1) / n
gains_pct = np.cumsum(sorted_actual) / total_pos
lift_vals  = gains_pct / pop_pct

random_gains = pop_pct  # diagonal

fig_gl = go.Figure()
# Gains traces
fig_gl.add_trace(go.Scatter(x=pop_pct, y=gains_pct, mode="lines",
    name="Model gains", line=dict(color=TEAL, width=2.5), visible=True))
fig_gl.add_trace(go.Scatter(x=pop_pct, y=random_gains, mode="lines",
    name="Random", line=dict(color=MUTED, dash="dot"), visible=True))
# Lift traces (hidden by default)
fig_gl.add_trace(go.Scatter(x=pop_pct, y=lift_vals, mode="lines",
    name="Lift", line=dict(color=TEAL, width=2.5), visible=False))
fig_gl.add_trace(go.Scatter(x=pop_pct, y=np.ones(n), mode="lines",
    name="Random (lift=1)", line=dict(color=MUTED, dash="dot"), visible=False))

fig_gl.update_layout(
    xaxis_title="% of applicants reviewed (ranked by predicted risk)",
    yaxis_title="% of all defaults captured",
    plot_bgcolor=WHITE, paper_bgcolor=WHITE, height=400,
    legend=dict(x=0.6, y=0.1),
    updatemenus=[dict(
        type="buttons", direction="right", x=0.0, y=1.12,
        buttons=[
            dict(label="Gains curve",
                 method="update",
                 args=[{"visible": [True, True, False, False]},
                       {"yaxis": {"title": "% of all defaults captured"}}]),
            dict(label="Lift curve",
                 method="update",
                 args=[{"visible": [False, False, True, True]},
                       {"yaxis": {"title": "Lift vs random"}}]),
        ]
    )]
)
st.plotly_chart(fig_gl, use_container_width=True)
with st.expander("How to read these charts"):
    st.markdown("""
**Gains curve:** if you reviewed the top X% of applicants ranked by predicted risk, what % of all actual defaults would you catch? The further above the diagonal, the better.

**Lift curve:** at each point, how many times more defaults does the model capture vs random selection? Lift = 2 means the model is twice as efficient as random. Lift naturally declines toward 1 as you review more of the population.
""")
st.divider()

# ══════════════════════════════════════════════════════════════════════════
# 4. MODEL CALIBRATION
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Calibration plot")
st.caption("Are predicted probabilities trustworthy? A well-calibrated model's dots should sit on the diagonal.")

n_cal_bins = st.slider("Calibration bins", 5, 20, 10, key="cal_bins")
frac_pos, mean_pred = calibration_curve(actual, probs, n_bins=n_cal_bins, strategy="uniform")

fig_cal = go.Figure()
fig_cal.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers",
    name="Our model", line=dict(color=TEAL, width=2),
    marker=dict(size=8, color=TEAL)))
fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
    name="Perfect calibration", line=dict(color=MUTED, dash="dot")))
fig_cal.update_layout(
    xaxis_title="Mean predicted probability",
    yaxis_title="Actual default rate in bin",
    xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
    plot_bgcolor=WHITE, paper_bgcolor=WHITE, height=380,
    legend=dict(x=0.05, y=0.95)
)
st.plotly_chart(fig_cal, use_container_width=True)
st.caption(
    "If dots sit above the diagonal the model underestimates risk (overconfident). "
    "Below the diagonal it overestimates risk (conservative). "
    "The further from the diagonal, the less reliable the threshold-based business recommendations."
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════
# 5. THRESHOLD → CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Decision threshold")
threshold = st.slider("Approval threshold", 0.05, 0.95, 0.20, 0.01, key="mod_threshold")

approved     = probs < threshold
bad_approved = int((approved & (actual == 1)).sum())
bad_rejected = int((~approved & (actual == 1)).sum())

ka, kb, kc, kd = st.columns(4)
ka.metric("Approved",          f"{approved.sum():,}",   f"{approved.sum()/len(actual):.1%}")
kb.metric("Portfolio DR",      f"{actual[approved].mean()*100:.1f}%")
kc.metric("Defaults blocked",  f"{bad_rejected:,}",     f"{bad_rejected/max(actual.sum(),1):.1%} of all")
kd.metric("Defaults slipping", f"{bad_approved:,}",     f"{bad_approved/max(actual.sum(),1):.1%} of all")

cm = confusion_matrix_at_threshold(actual, probs, threshold)
fig_cm = go.Figure(go.Heatmap(
    z=cm, x=["Pred: No Default","Pred: Default"], y=["Actual: No Default","Actual: Default"],
    text=cm, texttemplate="%{text:,}", colorscale="Teal", showscale=False,
))
fig_cm.update_layout(title=f"Confusion matrix at threshold {threshold:.2f}",
    height=300, plot_bgcolor=WHITE, paper_bgcolor=WHITE)
st.plotly_chart(fig_cm, use_container_width=True)
st.divider()

# ══════════════════════════════════════════════════════════════════════════
# 6. COEFFICIENTS
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Top 20 feature coefficients")
st.caption("Negative = associated with lower default risk. Positive = higher risk.")
coef_df = models["coef_df"].copy()
colours = [RED if v > 0 else GREEN for v in coef_df["coef"]]
fig_coef = go.Figure(go.Bar(
    x=coef_df["coef"], y=coef_df["feature"], orientation="h",
    marker_color=colours, hovertemplate="%{y}: %{x:.4f}<extra></extra>",
))
fig_coef.update_layout(xaxis_title="Coefficient",
    yaxis={"categoryorder":"total ascending"},
    plot_bgcolor=WHITE, paper_bgcolor=WHITE, height=520)
st.plotly_chart(fig_coef, use_container_width=True)
st.info("**ever_delinquent** and **months_since_last_delinquency_filled** dominate. Both are derived from 49.9% missingness treated as a signal.")
st.divider()

# ══════════════════════════════════════════════════════════════════════════
# 7. ENGINEERING DECISIONS TABLE
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Engineering decisions")
eng_df = pd.DataFrame(ENGINEERING_DECISIONS, columns=["Step","Justification","Impact"])
st.dataframe(eng_df, use_container_width=True, hide_index=True)
