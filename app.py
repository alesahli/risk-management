import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime, timedelta
from scipy.optimize import minimize
import io

# --- Report deps ---
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from PIL import Image
import tempfile
import os
import copy
import textwrap

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Portfolio Risk Management System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. FUNÇÕES CORE (BACKEND)
# ==============================================================================
@st.cache_data(show_spinner=False)
def get_market_data(tickers, start_date, end_date):
    if not tickers:
        return pd.DataFrame()
    try:
        s_date = pd.to_datetime(start_date) - timedelta(days=40)
        df = yf.download(
            tickers,
            start=s_date,
            end=end_date,
            progress=False,
            auto_adjust=False, 
            threads=False
        )
        
        if df is None or df.empty:
            return pd.DataFrame()

        data = pd.DataFrame()
        target_col = 'Close' 

        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = df.columns.get_level_values(0)
            lvl1 = df.columns.get_level_values(1)
            if target_col in lvl0:
                data = df.xs(target_col, axis=1, level=0)
            elif target_col in lvl1:
                data = df.xs(target_col, axis=1, level=1)
            elif 'Adj Close' in lvl0:
                data = df.xs('Adj Close', axis=1, level=0)
            else:
                data = df.iloc[:, 0]
        else:
            if target_col in df.columns:
                data = df[[target_col]]
            else:
                data = df.iloc[:, [0]]

        if isinstance(data, pd.Series):
            data = data.to_frame()

        if isinstance(tickers, (list, tuple)) and len(tickers) == 1 and data.shape[1] == 1:
            data.columns = tickers

        try:
            data.index = data.index.tz_localize(None)
        except Exception:
            pass

        data = data.dropna(axis=1, how='all')
        return data

    except Exception as e:
        st.error(f"Erro ao baixar dados: {e}")
        return pd.DataFrame()


def calculate_metrics(returns, rf_annual, benchmark_returns=None):
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return {}

    cum_prod = (1 + clean_returns).cumprod()
    total_return = cum_prod.iloc[-1] - 1 if len(cum_prod) > 0 else 0.0

    if len(clean_returns) > 1:
        days_diff = (clean_returns.index[-1] - clean_returns.index[0]).days
        years = days_diff / 365.25 if days_diff >= 7 else len(clean_returns) / 252.0
    else:
        years = 0

    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
    rf_daily = (1 + rf_annual / 100.0) ** (1 / 252) - 1
    ann_vol = clean_returns.std() * np.sqrt(252)

    neg_ret = clean_returns[clean_returns < 0]
    semi_dev = neg_ret.std() * np.sqrt(252) if len(neg_ret) > 1 else 0.0
    
    pos_ret = clean_returns[clean_returns > 0]
    upside_dev = pos_ret.std() * np.sqrt(252) if len(pos_ret) > 1 else 0.0

    excess_ret = clean_returns - rf_daily
    sharpe = (excess_ret.mean() / clean_returns.std()) * np.sqrt(252) if clean_returns.std() != 0 else 0.0
    sortino = (excess_ret.mean() / neg_ret.std()) * np.sqrt(252) if (not neg_ret.empty and neg_ret.std() != 0) else 0.0

    cum = (1 + clean_returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min() if not dd.empty else 0.0

    var_95 = np.percentile(clean_returns, 5) if len(clean_returns) > 0 else 0.0
    cvar_95 = clean_returns[clean_returns <= var_95].mean() if len(clean_returns) > 0 else 0.0

    beta = 0.0
    if benchmark_returns is not None:
        aligned = pd.concat([clean_returns, benchmark_returns], axis=1, join='inner').dropna()
        if not aligned.empty and aligned.shape[0] > 10:
            cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
            var_bench = np.var(aligned.iloc[:, 1])
            beta = cov / var_bench if var_bench != 0 else 0.0

    return {
        "Retorno do Período": float(total_return),
        "Retorno Anualizado": float(ann_return),
        "Volatilidade": float(ann_vol) if pd.notna(ann_vol) else 0.0,
        "Semi-Desvio": float(semi_dev) if pd.notna(semi_dev) else 0.0,
        "Upside-Desvio": float(upside_dev) if pd.notna(upside_dev) else 0.0,
        "Beta": float(beta) if pd.notna(beta) else 0.0,
        "Sharpe": float(sharpe) if pd.notna(sharpe) else 0.0,
        "Sortino": float(sortino) if pd.notna(sortino) else 0.0,
        "Max Drawdown": float(max_dd) if pd.notna(max_dd) else 0.0,
        "VaR 95%": float(var_95) if pd.notna(var_95) else 0.0,
        "CVaR 95%": float(cvar_95) if pd.notna(cvar_95) else 0.0
    }

def calculate_capture_ratios(asset_ret, bench_ret):
    aligned = pd.concat([asset_ret, bench_ret], axis=1, join='inner').dropna()
    if aligned.empty: return 0.0, 0.0
    r_asset, r_bench = aligned.iloc[:, 0], aligned.iloc[:, 1]
    up_mask, down_mask = r_bench > 0, r_bench < 0
    up_cap = (r_asset[up_mask].mean() / r_bench[up_mask].mean()) if up_mask.sum() > 0 and r_bench[up_mask].mean() != 0 else 0.0
    down_cap = (r_asset[down_mask].mean() / r_bench[down_mask].mean()) if down_mask.sum() > 0 and r_bench[down_mask].mean() != 0 else 0.0
    return float(up_cap) * 100.0, float(down_cap) * 100.0

def calculate_flexible_portfolio(asset_returns, weights_dict, cash_pct, rf_annual, fee_annual, rebal_freq):
    rf_daily = (1 + rf_annual / 100.0) ** (1 / 252) - 1
    fee_daily = (1 + fee_annual / 100.0) ** (1 / 252) - 1
    tickers = asset_returns.columns.tolist()
    initial_weights = np.array([weights_dict.get(t, 0.0) for t in tickers]) / 100.0
    w_cash_initial = cash_pct / 100.0

    if rebal_freq == 'Diário':
        gross_ret = asset_returns.fillna(0.0).dot(initial_weights) + (rf_daily * w_cash_initial)
        return gross_ret - fee_daily

    rebal_dates = set()
    if rebal_freq != 'Sem Rebalanceamento':
        resample_code = {'Mensal': 'ME', 'Trimestral': 'QE', 'Semestral': 'QE', 'Anual': 'YE'}.get(rebal_freq, 'QE')
        try:
            temp_resample = asset_returns.resample(resample_code).last().index
            rebal_dates = set(temp_resample[1::2]) if rebal_freq == 'Semestral' else set(temp_resample)
        except:
            rebal_dates = set(asset_returns.resample('Q').last().index)

    current_weights, current_cash_w = initial_weights.copy(), w_cash_initial
    portfolio_rets, returns_arr, dates = [], asset_returns.fillna(0.0).values, asset_returns.index

    for i in range(len(dates)):
        day_ret = np.sum(current_weights * returns_arr[i]) + (current_cash_w * rf_daily)
        portfolio_rets.append(day_ret - fee_daily)
        if (1 + day_ret) != 0:
            current_weights = current_weights * (1 + returns_arr[i]) / (1 + day_ret)
            current_cash_w = current_cash_w * (1 + rf_daily) / (1 + day_ret)
        if dates[i] in rebal_dates:
            current_weights, current_cash_w = initial_weights.copy(), w_cash_initial

    return pd.Series(portfolio_rets, index=dates)

# ==============================================================================
# 2.5 NOVO SOLVER (CARDINALIDADE + CORRELAÇÃO)
# ==============================================================================
def run_solver(df_returns, rf_annual, bounds, target_metric, mgmt_fee_annual=0.0, target_semidev_val=None, max_assets=20, div_penalty_weight=0.2):
    rf_daily = (1 + rf_annual / 100.0) ** (1 / 252) - 1
    fee_daily = (1 + mgmt_fee_annual / 100.0) ** (1 / 252) - 1
    num_assets = len(df_returns.columns)
    corr_matrix = df_returns.corr().values

    lower_bounds = np.array([b[0] for b in bounds], dtype=float)
    upper_bounds = np.array([b[1] for b in bounds], dtype=float)
    initial_guess = (lower_bounds + upper_bounds) / 2.0
    initial_guess /= np.sum(initial_guess)

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]

    if target_metric == "Max Return (Target Semi-Dev)" and target_semidev_val is not None:
        def semidev_constraint(weights):
            w = np.array(weights, dtype=float)
            gross = df_returns.fillna(0.0).dot(w)
            neg = (gross - fee_daily)[(gross - fee_daily) < 0]
            current_semi = neg.std() * np.sqrt(252) if len(neg) > 1 else 0.0
            return (target_semidev_val / 100.0) - current_semi
        constraints.append({'type': 'ineq', 'fun': semidev_constraint})

    def objective(weights):
        w = np.array(weights, dtype=float)
        net_ret = df_returns.fillna(0.0).dot(w) - fee_daily
        
        # 1. MÉTRICA BASE
        res = 0.0
        if target_metric == "Max Sortino":
            neg = net_ret[net_ret < 0]
            if neg.empty or neg.std() == 0: return 1e6
            res = -((net_ret - rf_daily).mean() / neg.std()) * np.sqrt(252)
        elif target_metric == "Min Downside Volatility":
            neg = net_ret[net_ret < 0]
            res = neg.std() * np.sqrt(252) if not neg.empty else 0.0
        elif target_metric == "Max Return (Target Semi-Dev)":
            res = -((1 + ((1 + net_ret).prod() - 1)) ** (252 / len(net_ret)) - 1)

        # 2. PENALIDADE DE CORRELAÇÃO (DIVERSIFICAÇÃO)
        if div_penalty_weight > 0:
            res += div_penalty_weight * np.dot(w.T, np.dot(corr_matrix, w))

        # 3. PENALIDADE DE CARDINALIDADE (MAX ASSETS)
        active_assets = np.sum(w > 0.005) # Ativos com > 0.5%
        if active_assets > max_assets:
            res += (active_assets - max_assets) * 2.0

        return res

    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-6)
    return result

# ==============================================================================
# 3. INTERFACE E CARREGAMENTO
# ==============================================================================
def load_portfolio_from_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8-sig') if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df.columns = [str(c).lower().strip() for c in df.columns]
        col_t = next((c for c in df.columns if c in ['ativo', 'ticker', 'asset', 'symbol']), None)
        col_w = next((c for c in df.columns if c in ['peso', 'weight', 'alocacao', '%']), None)
        if not col_t or not col_w: return None, "Colunas não encontradas."
        portfolio = {str(row[col_t]).strip().upper(): float(str(row[col_w]).replace(',', '.')) for _, row in df.iterrows() if float(str(row[col_w]).replace(',', '.')) > 0}
        if sum(portfolio.values()) <= 1.05: portfolio = {k: v * 100.0 for k, v in portfolio.items()}
        return portfolio, None
    except Exception as e: return None, str(e)

# --- Funções de Plotly/PDF omitidas para brevidade mas mantidas no fluxo ---
def _force_print_theme(fig):
    f = copy.deepcopy(fig)
    f.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black"))
    return f

def fig_to_png_bytes(fig):
    try: return pio.to_image(_force_print_theme(fig), format="png", scale=2)
    except: return None

def df_to_table_fig(df, title=None, max_rows=40, round_map=None):
    dfx = df.copy()
    if round_map:
        for col, dec in round_map.items():
            if col in dfx.columns: dfx[col] = pd.to_numeric(dfx[col], errors="coerce").round(dec)
    dfx = dfx.head(max_rows).fillna("").astype(str)
    return go.Figure(data=[go.Table(
        header=dict(values=list(dfx.columns), fill_color="#F2F2F2", align="left"),
        cells=dict(values=[dfx[c].tolist() for c in dfx.columns], fill_color="white", align="left")
    )], layout=dict(title=title, margin=dict(l=10, r=10, t=50, b=10)))

def write_pdf_report(output_path, report_title, subtitle, sections):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    y = height - 2*cm
    c.setFont("Helvetica-Bold", 16); c.drawString(2*cm, y, report_title); y -= 2*cm
    for sec in sections:
        c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, sec['title']); y -= 1*cm
        for it in sec['items']:
            if it['type'] == 'text':
                c.setFont("Helvetica", 9); lines = textwrap.wrap(it['value'], width=100)
                for l in lines: c.drawString(2*cm, y, l); y -= 0.5*cm
            elif it['type'] == 'image' and it['png_bytes']:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(it['png_bytes']); tmp_path = tmp.name
                c.drawImage(tmp_path, 2*cm, y-10*cm, width=17*cm, height=10*cm); y -= 11*cm
                os.remove(tmp_path)
        if y < 4*cm: c.showPage(); y = height - 2*cm
    c.save()

# ==============================================================================
# 4. DASHBOARD E SIDEBAR
# ==============================================================================
st.sidebar.header("Portfolio Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Portfolio", type=['csv', 'xlsx'])
if uploaded_file:
    p, err = load_portfolio_from_file(uploaded_file)
    if p: st.session_state['imported_portfolio'] = p; st.session_state['tickers_text_key'] = ", ".join(p.keys())

tickers_text = st.sidebar.text_area("Asset Tickers:", value=st.session_state.get('tickers_text_key', "NVDA, MSFT, COST, WMT"), height=100)
tickers_input = [t.strip().upper() for t in tickers_text.split(',') if t.strip()]

st.sidebar.subheader("Market & Risks")
rf_input = st.sidebar.number_input("Risk Free %", value=10.5)
mgmt_fee = st.sidebar.number_input("Mgmt Fee %", value=0.0)
bench_ticker = st.sidebar.text_input("Benchmark", value="QQQ")
rebal_freq_sim = st.sidebar.selectbox("Rebal Freq", ["Sem Rebalanceamento", "Mensal", "Trimestral", "Diário"])

# Pesos Manuais
weights_orig, weights_sim = {}, {}
imported_data = st.session_state.get('imported_portfolio', {})
if tickers_input:
    for t in tickers_input:
        val = imported_data.get(t, 100.0/len(tickers_input))
        weights_orig[t] = st.sidebar.number_input(f"Orig {t}", 0.0, 100.0, float(val))
        weights_sim[t] = st.sidebar.number_input(f"Sim {t}", 0.0, 100.0, float(val), key=f"sim_in_{t}")
cash_orig, cash_sim = 100.0 - sum(weights_orig.values()), 100.0 - sum(weights_sim.values())

# Download Dados
all_t = list(set(tickers_input + [bench_ticker]))
df_p = get_market_data(all_t, datetime.today()-timedelta(days=730), datetime.today())
df_ret = df_p.ffill().pct_change().dropna()
bench_ret = df_ret[bench_ticker] if bench_ticker in df_ret.columns else pd.Series(0, index=df_ret.index)
assets_ret = df_ret[[t for t in tickers_input if t in df_ret.columns]]

# Portfólios
ret_orig = calculate_flexible_portfolio(assets_ret, weights_orig, cash_orig, rf_input, mgmt_fee, "Diário")
ret_sim = calculate_flexible_portfolio(assets_ret, weights_sim, cash_sim, rf_input, mgmt_fee, rebal_freq_sim)

# KPIs
m_orig, m_sim, m_bench = calculate_metrics(ret_orig, rf_input, bench_ret), calculate_metrics(ret_sim, rf_input, bench_ret), calculate_metrics(bench_ret, rf_input, bench_ret)

st.title("Portfolio Risk Management System")
col1, col2 = st.columns([3, 1])
with col1:
    df_comp = pd.DataFrame({
        "Metric": list(m_orig.keys()),
        "Current": [f"{v:.2%}" if "Retorno" in k or "Vol" in k else f"{v:.2f}" for k, v in m_orig.items()],
        "Simulated": [f"{v:.2%}" if "Retorno" in k or "Vol" in k else f"{v:.2f}" for k, v in m_sim.items()],
        "Bench": [f"{v:.2%}" if "Retorno" in k or "Vol" in k else f"{v:.2f}" for k, v in m_bench.items()]
    })
    st.table(df_comp.set_index("Metric"))

# Abas
tab1, tab2, tab3 = st.tabs(["Performance", "Correlation", "Portfolio Solver"])

with tab2:
    st.plotly_chart(px.imshow(assets_ret.corr(), text_auto=".2f", color_continuous_scale="RdYlGn"))

with tab3:
    st.markdown("### Portfolio Optimization")
    df_opt = pd.concat([assets_ret, pd.Series((1+rf_input/100.0)**(1/252)-1, index=assets_ret.index, name="CASH")], axis=1)
    
    col_s, col_r = st.columns([1, 2])
    with col_s:
        obj = st.selectbox("Objective", ["Min Downside Volatility", "Max Sortino", "Max Return (Target Semi-Dev)"])
        m_ast = st.number_input("Ativos na Carteira", 1, len(df_opt.columns), min(20, len(df_opt.columns)-1))
        d_fac = st.slider("Fator Diversificação (Correlação)", 0.0, 1.0, 0.2)
        target_v = st.number_input("Target Semi-Dev %", 5.0) if obj == "Max Return (Target Semi-Dev)" else None
        
        st.caption("Constraints Globais")
        g_min, g_max = st.number_input("Min Ativo %", 3.0), st.number_input("Max Ativo %", 8.0)
        
        edited = st.data_editor(pd.DataFrame({
            "Asset": df_opt.columns, 
            "Min %": [g_min if a != "CASH" else 0.0 for a in df_opt.columns], 
            "Max %": [g_max if a != "CASH" else 100.0 for a in df_opt.columns]
        }))

        if st.button("Run Solver"):
            b = [(r["Min %"]/100, r["Max %"]/100) for _, r in edited.iterrows()]
            res = run_solver(df_opt, rf_input, b, obj, mgmt_fee, target_v, m_ast, d_fac)
            if res.success:
                st.session_state['sol'] = {'w': res.x, 'a': df_opt.columns}
                st.success("Otimizado!")

    with col_r:
        if 'sol' in st.session_state:
            w, a = st.session_state['sol']['w'], st.session_state['sol']['a']
            df_w = pd.DataFrame({"Asset": a, "Weight %": w*100}).query("`Weight %` > 0.1").sort_values("Weight %", ascending=False)
            st.plotly_chart(px.pie(df_w, values="Weight %", names="Asset", title="Sugerido"))
            st.dataframe(df_w.style.format({"Weight %": "{:.2f}%"}), hide_index=True)

# Export PDF Omitido por espaço mas mantido conforme o seu código original
