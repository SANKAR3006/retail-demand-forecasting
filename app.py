import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #e94560;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #e94560; }
    .metric-label { font-size: 0.85rem; color: #a8b2d8; margin-top: 4px; }
    .section-header {
        background: linear-gradient(90deg, #1a1a2e, #0f3460);
        padding: 12px 20px;
        border-radius: 8px;
        border-left: 4px solid #e94560;
        color: white;
        font-weight: 600;
        margin-bottom: 16px;
    }
    .insight-box {
        background: #f0f4ff;
        border-left: 4px solid #0f3460;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: #1a1a2e;
        margin-bottom: 8px;
    }
    .alert-red   { background:#fff0f0; border-left:4px solid #e94560; padding:10px 14px; border-radius:0 8px 8px 0; }
    .alert-green { background:#f0fff4; border-left:4px solid #2ecc71; padding:10px 14px; border-radius:0 8px 8px 0; }
</style>
""", unsafe_allow_html=True)

# ── Data paths ─────────────────────────────────────────────────────────────────
DATA_PATH  = '/content/drive/MyDrive/Colab Notebooks/Demand Forecasting Project/'
MODEL_PATH = DATA_PATH + 'models/'

# ── Loaders ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_predictions():
    df = pd.read_parquet(DATA_PATH + 'val_predictions.parquet')
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def load_panel():
    df = pd.read_parquet(DATA_PATH + 'panel_features.parquet')
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH + 'lgbm_model.pkl')

@st.cache_data
def load_shap_importance():
    return pd.read_csv(DATA_PATH + 'shap_importance.csv')

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Demand Forecasting")
    st.markdown("*Kaggle Store Sales · LightGBM + Optuna*")
    st.divider()

    preds   = load_predictions()
    panel   = load_panel()
    model   = load_model()

    families = sorted(preds['family'].unique())
    stores   = sorted(preds['store_nbr'].unique())

    selected_family = st.selectbox("📦 Product Family", families, index=families.index('GROCERY I'))
    selected_store  = st.selectbox("🏪 Store Number",   stores,   index=0)

    st.divider()
    st.markdown("**Model Info**")
    st.markdown(f"- Algorithm: `LightGBM`")
    st.markdown(f"- Tuning: `Optuna (40 trials)`")
    st.markdown(f"- CV: `TimeSeriesSplit (3 folds)`")
    st.markdown(f"- Target: `log1p(sales)`")

    st.divider()
    st.markdown("**Pipeline**")
    steps = ["✅ Data Prep","✅ Features","✅ Train","✅ Evaluate","✅ SHAP","▶️ App"]
    for s in steps:
        st.markdown(f"&nbsp;&nbsp;{s}", unsafe_allow_html=True)

# ── Overall metrics ────────────────────────────────────────────────────────────
def compute_metrics(actual, predicted):
    actual    = np.array(actual)
    predicted = np.clip(np.array(predicted), 0, None)
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae  = np.mean(np.abs(actual - predicted))
    mask = actual > 0
    mape = np.mean(np.abs((actual[mask]-predicted[mask])/actual[mask]))*100 if mask.sum()>0 else np.nan
    wape = np.sum(np.abs(actual-predicted))/np.sum(actual)*100 if np.sum(actual)>0 else np.nan
    return rmse, mae, mape, wape

overall_rmse, overall_mae, overall_mape, overall_wape = compute_metrics(preds['actual'], preds['pred'])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🔍 SHAP Explainability", "⚠️ Business Signals"])

# ══════════════════════════════════════
# TAB 1 — FORECAST
# ══════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📈 Forecast vs Actual</div>', unsafe_allow_html=True)

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    for col, label, value, unit in zip(
        [col1, col2, col3, col4],
        ["RMSE", "MAE", "MAPE", "WAPE"],
        [overall_rmse, overall_mae, overall_mape, overall_wape],
        ["units", "units", "%", "%"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value:,.1f}{unit}</div>
                <div class="metric-label">{label} · All stores & families</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Filter for selected store × family
    subset = preds[
        (preds['store_nbr'] == selected_store) &
        (preds['family']    == selected_family)
    ].sort_values('date')

    if len(subset) == 0:
        st.warning("No data for this store × family combination.")
    else:
        s_rmse, s_mae, s_mape, s_wape = compute_metrics(subset['actual'], subset['pred'])

        st.markdown(f"**Store {selected_store} · {selected_family}** &nbsp;|&nbsp; "
                    f"WAPE: `{s_wape:.1f}%` &nbsp;|&nbsp; MAE: `{s_mae:.1f}` units")

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(subset['date'], subset['actual'],
                label='Actual', color='#0f3460', linewidth=1.8)
        ax.plot(subset['date'], subset['pred'],
                label='Forecast', color='#e94560', linewidth=1.8, linestyle='--')
        ax.fill_between(subset['date'], subset['actual'], subset['pred'],
                        alpha=0.08, color='#e94560')
        ax.set_xlabel('Date')
        ax.set_ylabel('Units Sold')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # WAPE by family heatmap
    st.markdown('<div class="section-header">📊 WAPE by Product Family</div>', unsafe_allow_html=True)

    family_wape = preds.groupby('family').apply(
        lambda g: np.sum(np.abs(g['actual']-g['pred']))/np.sum(g['actual'])*100
        if np.sum(g['actual']) > 0 else np.nan
    ).reset_index()
    family_wape.columns = ['family','WAPE']
    family_wape = family_wape.sort_values('WAPE')

    fig2, ax2 = plt.subplots(figsize=(11, 8))
    colors = ['#2ecc71' if w < 15 else '#e67e22' if w < 30 else '#e74c3c'
              for w in family_wape['WAPE']]
    bars = ax2.barh(family_wape['family'], family_wape['WAPE'],
                    color=colors, edgecolor='none', height=0.7)
    ax2.axvline(x=25, color='#e74c3c', linestyle='--', linewidth=1.2, label='25% benchmark')
    ax2.axvline(x=overall_wape, color='#3498db', linestyle='--', linewidth=1.2,
                label=f'Overall WAPE ({overall_wape:.1f}%)')
    for bar, val in zip(bars, family_wape['WAPE']):
        ax2.text(val+0.3, bar.get_y()+bar.get_height()/2,
                 f'{val:.1f}%', va='center', fontsize=8)
    ax2.set_xlabel('WAPE (%)')
    ax2.set_title('Forecast Error by Product Family', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ══════════════════════════════════════
# TAB 2 — SHAP
# ══════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🔍 SHAP Feature Importance</div>', unsafe_allow_html=True)

    try:
        shap_imp = load_shap_importance()

        col_a, col_b = st.columns([1.2, 1])

        with col_a:
            st.markdown("**Global Feature Importance (Mean |SHAP|)**")
            top_n = st.slider("Show top N features", 5, len(shap_imp), 15)
            plot_df = shap_imp.head(top_n).sort_values('importance')

            fig3, ax3 = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
            colors = ['#e94560' if f in ['sales_lag_7','rolling_mean_7','rolling_mean_28']
                      else '#0f3460' for f in plot_df['feature']]
            ax3.barh(plot_df['feature'], plot_df['importance'],
                     color=colors, edgecolor='none', height=0.7)
            ax3.set_xlabel('Mean |SHAP value|')
            ax3.set_title('Feature Importance', fontweight='bold')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

        with col_b:
            st.markdown("**Key Findings**")
            top3 = shap_imp.head(3)['feature'].tolist()
            st.markdown(f"""
            <div class="insight-box">🥇 <b>{top3[0]}</b> is the strongest demand driver —
            recent sales history strongly predicts future demand.</div>
            <div class="insight-box">🥈 <b>{top3[1]}</b> smooths short-term volatility
            and captures weekly trend.</div>
            <div class="insight-box">🥉 <b>{top3[2]}</b> captures the monthly
            demand pattern across families.</div>
            <div class="insight-box">📦 <b>onpromotion</b> consistently pushes
            forecasts higher — promotional uplift is measurable.</div>
            <div class="insight-box">🛢️ <b>oil_lag1</b> has a measurable macro
            signal — Ecuador's economy is oil-dependent.</div>
            """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown("**Full Importance Table**")
            st.dataframe(
                shap_imp.rename(columns={'feature':'Feature','importance':'Mean |SHAP|'}),
                use_container_width=True,
                height=300
            )

    except FileNotFoundError:
        st.warning("Run `05_shap.ipynb` first to generate SHAP importance scores.")

# ══════════════════════════════════════
# TAB 3 — BUSINESS SIGNALS
# ══════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">⚠️ Business Signals — Stockout & Overstock Risk</div>',
                unsafe_allow_html=True)

    st.markdown("""
    > Translating forecast error into actionable supply chain decisions.
    > Rows where the model **under-forecasted by >20%** indicate likely stockout risk.
    > Rows where the model **over-forecasted by >20%** indicate overstock / excess inventory risk.
    """)

    threshold = st.slider("Error threshold (%)", 10, 40, 20, step=5)

    preds_copy = preds.copy()
    preds_copy['error_pct'] = (preds_copy['pred'] - preds_copy['actual']) / (preds_copy['actual'] + 1) * 100

    stockout  = preds_copy[preds_copy['error_pct'] < -threshold]
    overstock = preds_copy[preds_copy['error_pct'] >  threshold]

    col_s, col_o, col_n = st.columns(3)
    with col_s:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#e74c3c">{len(stockout):,}</div>
            <div class="metric-label">🔴 Stockout Risk Rows</div>
            <div class="metric-label">{len(stockout)/len(preds)*100:.1f}% of validation</div>
        </div>""", unsafe_allow_html=True)
    with col_o:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#e67e22">{len(overstock):,}</div>
            <div class="metric-label">🟠 Overstock Risk Rows</div>
            <div class="metric-label">{len(overstock)/len(preds)*100:.1f}% of validation</div>
        </div>""", unsafe_allow_html=True)
    with col_n:
        normal = len(preds) - len(stockout) - len(overstock)
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#2ecc71">{normal:,}</div>
            <div class="metric-label">🟢 Within Threshold</div>
            <div class="metric-label">{normal/len(preds)*100:.1f}% of validation</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**🔴 Top Stockout Risk — Store × Family**")
        if len(stockout) > 0:
            top_stockout = (
                stockout.groupby(['store_nbr','family'])
                .agg(risk_days=('date','count'), avg_error=('error_pct','mean'))
                .sort_values('risk_days', ascending=False)
                .head(10)
                .reset_index()
            )
            top_stockout['avg_error'] = top_stockout['avg_error'].apply(lambda x: f'{x:.1f}%')
            st.dataframe(top_stockout, use_container_width=True, height=300)
        else:
            st.success("No stockout risk at this threshold.")

    with col_r:
        st.markdown("**🟠 Top Overstock Risk — Store × Family**")
        if len(overstock) > 0:
            top_overstock = (
                overstock.groupby(['store_nbr','family'])
                .agg(risk_days=('date','count'), avg_error=('error_pct','mean'))
                .sort_values('risk_days', ascending=False)
                .head(10)
                .reset_index()
            )
            top_overstock['avg_error'] = top_overstock['avg_error'].apply(lambda x: f'{x:.1f}%')
            st.dataframe(top_overstock, use_container_width=True, height=300)
        else:
            st.success("No overstock risk at this threshold.")

    st.divider()

    # Daily risk trend
    st.markdown("**📅 Daily Risk Trend**")
    daily_risk = preds_copy.groupby('date').apply(lambda g: pd.Series({
        'stockout_rows' : (g['error_pct'] < -threshold).sum(),
        'overstock_rows': (g['error_pct'] >  threshold).sum(),
        'total_rows'    : len(g)
    })).reset_index()

    fig4, ax4 = plt.subplots(figsize=(13, 3.5))
    ax4.fill_between(daily_risk['date'], daily_risk['stockout_rows'],
                     alpha=0.4, color='#e74c3c', label='Stockout risk rows')
    ax4.fill_between(daily_risk['date'], daily_risk['overstock_rows'],
                     alpha=0.4, color='#e67e22', label='Overstock risk rows')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Number of SKUs at risk')
    ax4.set_title(f'Daily Stockout & Overstock Risk (>{threshold}% error threshold)',
                  fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()
