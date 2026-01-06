import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.optimize import minimize
import io

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

@st.cache_data
def get_market_data(tickers, start_date, end_date):
    if not tickers: return pd.DataFrame()
    try:
        s_date = pd.to_datetime(start_date) - timedelta(days=20)
        df = yf.download(tickers, start=s_date, end=end_date, progress=False, auto_adjust=True, threads=False)
        
        if df.empty: return pd.DataFrame()
        
        data = pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns: data = df['Close']
            elif 'Close' in df.columns.get_level_values(0): data = df.xs('Close', axis=1, level=0)
            elif 'Close' in df.columns.get_level_values(1): data = df.xs('Close', axis=1, level=1)
            else: data = df.iloc[:, 0] 
        else:
            if 'Close' in df.columns: data = df[['Close']]
            else: data = df.iloc[:, [0]]

        if isinstance(data, pd.Series): data = data.to_frame()
        if len(tickers) == 1 and data.shape[1] == 1: data.columns = tickers
        
        data.index = data.index.tz_localize(None)
        data = data[data.index >= pd.to_datetime(start_date)]
        data = data.dropna(axis=1, how='all')
        
        return data
    except Exception as e:
        st.error(f"Erro ao baixar dados: {e}")
        return pd.DataFrame()

def calculate_metrics(returns, rf_annual, benchmark_returns=None):
    returns = returns.dropna()
    if returns.empty: return {}
    
    rf_daily = (1 + rf_annual/100)**(1/252) - 1
    days = len(returns)
    
    total_return = (1 + returns).prod() - 1
    if days > 10: ann_return = (1 + total_return)**(252 / days) - 1
    else: ann_return = total_return
    
    ann_vol = returns.std() * np.sqrt(252)
    
    neg_ret = returns[returns < 0]
    semi_dev = neg_ret.std() * np.sqrt(252) if len(neg_ret) > 1 else 0.0
    
    pos_ret = returns[returns > 0]
    upside_dev = pos_ret.std() * np.sqrt(252) if len(pos_ret) > 1 else 0.0
    
    excess_ret = returns - rf_daily
    sharpe = (excess_ret.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    sortino = (excess_ret.mean() / neg_ret.std()) * np.sqrt(252) if (not neg_ret.empty and neg_ret.std() != 0) else 0
    
    cum = (1 + returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    
    beta = 0.0
    if benchmark_returns is not None:
        aligned = pd.concat([returns, benchmark_returns], axis=1, join='inner').dropna()
        if not aligned.empty and aligned.shape[0] > 10:
            cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
            var_bench = np.var(aligned.iloc[:, 1])
            beta = cov / var_bench if var_bench != 0 else 0
            
    return {
        "Retorno do Período": total_return, "Retorno Anualizado": ann_return,
        "Volatilidade": ann_vol, 
        "Semi-Desvio": semi_dev,
        "Upside-Desvio": upside_dev,
        "Beta": beta,
        "Sharpe": sharpe, "Sortino": sortino, "Max Drawdown": max_dd,
        "VaR 95%": var_95, "CVaR 95%": cvar_95
    }

def calculate_capture_ratios(asset_ret, bench_ret):
    aligned = pd.concat([asset_ret, bench_ret], axis=1, join='inner').dropna()
    if aligned.empty: return 0.0, 0.0
    r_asset = aligned.iloc[:, 0]; r_bench = aligned.iloc[:, 1]
    up_mask = r_bench > 0
    up_cap = r_asset[up_mask].mean() / r_bench[up_mask].mean() if up_mask.sum() > 0 else 0
    down_mask = r_bench < 0
    down_cap = r_asset[down_mask].mean() / r_bench[down_mask].mean() if down_mask.sum() > 0 else 0
    return up_cap * 100.0, down_cap * 100.0

def calculate_flexible_portfolio(asset_returns, weights_dict, cash_pct, rf_annual, fee_annual, rebal_freq):
    rf_daily = (1 + rf_annual/100)**(1/252) - 1
    fee_daily = (1 + fee_annual/100)**(1/252) - 1
    tickers = asset_returns.columns.tolist()
    initial_weights = np.array([weights_dict.get(t, 0) for t in tickers]) / 100.0
    w_cash_initial = cash_pct / 100.0
    
    if rebal_freq == 'Diário':
        gross_ret = asset_returns.fillna(0.0).dot(initial_weights) + (rf_daily * w_cash_initial)
        return gross_ret - fee_daily

    rebal_dates = set()
    if rebal_freq != 'Sem Rebalanceamento':
        try:
            if rebal_freq == 'Mensal': resample_code = 'ME'
            elif rebal_freq == 'Trimestral': resample_code = 'QE'
            elif rebal_freq == 'Anual': resample_code = 'YE'
            elif rebal_freq == 'Semestral': resample_code = 'QE' 
            else: resample_code = 'QE' 
            temp_resample = asset_returns.resample(resample_code).last().index
            if rebal_freq == 'Semestral': rebal_dates = set(temp_resample[1::2])
            else: rebal_dates = set(temp_resample)
        except:
            if rebal_freq == 'Mensal': rebal_dates = set(asset_returns.resample('M').last().index)
            else: rebal_dates = set(asset_returns.resample('Q').last().index)

    current_weights = initial_weights.copy()
    current_cash_w = w_cash_initial
    portfolio_rets = []
    returns_arr = asset_returns.fillna(0.0).values
    dates = asset_returns.index
    
    for i in range(len(dates)):
        r_assets = returns_arr[i]
        day_ret = np.sum(current_weights * r_assets) + (current_cash_w * rf_daily)
        net_day_ret = day_ret - fee_daily
        portfolio_rets.append(net_day_ret)
        
        denominator = 1 + day_ret
        if denominator != 0:
            current_weights = current_weights * (1 + r_assets) / denominator
            current_cash_w = current_cash_w * (1 + rf_daily) / denominator
        
        if dates[i] in rebal_dates:
            current_weights = initial_weights.copy(); current_cash_w = w_cash_initial

    return pd.Series(portfolio_rets, index=dates)

def run_solver(df_returns, rf_annual, bounds, target_metric, mgmt_fee_annual=0.0, target_semidev_val=None):
    rf_daily = (1 + rf_annual/100)**(1/252) - 1
    fee_daily = (1 + mgmt_fee_annual/100)**(1/252) - 1
    num_assets = len(df_returns.columns)
    
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    initial_guess = (lower_bounds + upper_bounds) / 2
    sum_guess = np.sum(initial_guess)
    if sum_guess > 0: initial_guess = initial_guess / sum_guess
    else: initial_guess = np.array([1/num_assets] * num_assets)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    if target_metric == "Max Return (Target Semi-Dev)" and target_semidev_val is not None:
        def semidev_constraint(weights):
            w = np.array(weights)
            gross = df_returns.fillna(0).dot(w)
            net = gross - fee_daily
            neg = net[net < 0]
            current_semi = neg.std() * np.sqrt(252)
            return (target_semidev_val/100.0) - current_semi
        constraints.append({'type': 'ineq', 'fun': semidev_constraint})

    def objective(weights):
        w = np.array(weights)
        gross_ret = df_returns.fillna(0).dot(w)
        net_ret = gross_ret - fee_daily
        if abs(np.sum(w) - 1.0) > 0.001: return 1e5
        if target_metric == "Max Sortino":
            neg_ret = net_ret[net_ret < 0]
            if neg_ret.empty or neg_ret.std() == 0: return 1e5
            excess_ret = net_ret - rf_daily
            sortino = (excess_ret.mean() / neg_ret.std()) * np.sqrt(252)
            return -sortino
        elif target_metric == "Min Downside Volatility":
            neg_ret = net_ret[net_ret < 0]
            if neg_ret.empty: return 0
            semi_dev = neg_ret.std() * np.sqrt(252)
            return semi_dev
        elif target_metric == "Max Return (Target Semi-Dev)":
            total_ret = (1 + net_ret).prod() - 1
            ann_ret = (1 + total_ret)**(252 / len(net_ret)) - 1
            return -ann_ret
            
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-6, options={'maxiter': 1000})
    return result

# --- FUNÇÃO DE EXPORTAÇÃO HTML PROFISSIONAL ---
def generate_interactive_report(df_comp_fmt, figures_dict):
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    html_content = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ background-color: #f4f7f9; padding: 40px; font-family: sans-serif; }}
            .header-box {{ background: #1e3c72; color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; text-align: center; }}
            .card {{ border: none; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); margin-bottom: 30px; background: white; padding: 20px; }}
            .table {{ font-size: 0.9rem; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header-box">
                <h1>Portfolio Risk Report</h1>
                <p>Relatório Consolidado de Gestão de Risco</p>
                <small>Gerado em: {now_str}</small>
            </div>
            
            <div class="card">
                <h3>Métricas de Performance</h3>
                {df_comp_fmt.to_html(classes='table table-striped table-hover', index=False)}
            </div>
    """
    for title, fig in figures_dict.items():
        if fig is not None:
            html_content += f"""
            <div class="card">
                <h3>{title}</h3>
                {fig.to_html(full_html=False, include_plotlyjs=False)}
            </div>
            """
    html_content += "</div></body></html>"
    return html_content

# --- FUNÇÃO DE IMPORTAÇÃO ---
def load_portfolio_from_file(uploaded_file):
    try:
        df = pd.DataFrame()
        if uploaded_file.name.endswith('.csv'):
            try: df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8-sig')
            except: pass
            if df.empty or df.shape[1] < 2:
                uploaded_file.seek(0)
                try: df = pd.read_csv(uploaded_file, sep=',', decimal='.')
                except: pass
        else:
            try: df = pd.read_excel(uploaded_file)
            except ImportError: return None, "Servidor sem suporte a .xlsx. Por favor use o Template CSV."
        
        if df.empty: return None, "Não foi possível ler o arquivo. Verifique se é um CSV válido."
        df.columns = [str(c).lower().strip() for c in df.columns]
        col_ticker = next((c for c in df.columns if c in ['ativo', 'ticker', 'asset', 'symbol', 'código']), None)
        col_weight = next((c for c in df.columns if c in ['peso', 'weight', 'alocacao', '%', 'valor']), None)
        
        if not col_ticker or not col_weight: return None, f"Colunas 'Ativo' e 'Peso' não encontradas."
        portfolio = {}
        for _, row in df.iterrows():
            t = str(row[col_ticker]).strip().upper()
            val_raw = str(row[col_weight]).replace(',', '.')
            try: w = float(val_raw)
            except: w = 0.0
            if w > 0: portfolio[t] = w
        total_w = sum(portfolio.values())
        if total_w <= 1.05 and total_w > 0:
             for k in portfolio: portfolio[k] = portfolio[k] * 100.0
        return portfolio, None
    except Exception as e: return None, str(e)

# ==============================================================================
# 3. BARRA LATERAL (INPUTS)
# ==============================================================================
st.sidebar.header("Portfolio Configuration")

with st.sidebar.expander("📂 Import / Export Portfolio", expanded=False):
    df_template = pd.DataFrame({"Ativo": ["PETR4.SA", "VALE3.SA"], "Peso": [50.0, 50.0]})
    csv_template = df_template.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
    st.download_button(label="Download Template (CSV)", data=csv_template, file_name="portfolio_template.csv", mime="text/csv", use_container_width=True)
    
    uploaded_file = st.file_uploader("Upload Portfolio (CSV/XLSX)", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        portfolio_dict, error_msg = load_portfolio_from_file(uploaded_file)
        if portfolio_dict:
            st.session_state['imported_portfolio'] = portfolio_dict
            st.session_state['tickers_text_key'] = ", ".join(portfolio_dict.keys())
            st.success(f"Carregado: {len(portfolio_dict)} ativos.")
        else: st.error(f"Erro: {error_msg}")

default_tickers_text = "VALE3.SA, PETR4.SA, BPAC11.SA"
if 'tickers_text_key' in st.session_state: default_tickers_text = st.session_state['tickers_text_key']

tickers_text = st.sidebar.text_area("Asset Tickers:", value=default_tickers_text, height=100)
tickers_input = [t.strip().upper() for t in tickers_text.split(',') if t.strip()]

periodo_option = st.sidebar.radio("Time Horizon:", ["1 Ano", "2 Anos", "Desde 2020", "Personalizado"], horizontal=True)
end_date = datetime.today()
if periodo_option == "1 Ano": start_date = end_date - timedelta(days=365)
elif periodo_option == "2 Anos": start_date = end_date - timedelta(days=730)
elif periodo_option == "Desde 2020": start_date = datetime(2020, 1, 1)
else:
    c_start, c_end = st.sidebar.columns(2)
    start_date = c_start.date_input("Start Date", value=datetime(2024,1,1))
    end_date = c_end.date_input("End Date", value=datetime.today())

st.sidebar.subheader("Market & Costs")
c_rf, c_fee = st.sidebar.columns(2)
rf_input = c_rf.number_input("Risk Free (% p.a.)", value=10.5, step=0.5)
mgmt_fee = c_fee.number_input("Mgmt Fee (% p.a.)", value=0.0, step=0.1)
bench_ticker = st.sidebar.text_input("Benchmark Ticker", value="^BVSP")

if 'rebal_freq_key' not in st.session_state: st.session_state['rebal_freq_key'] = "Sem Rebalanceamento"
st.sidebar.subheader("Rebalancing (Simulated)")
rebal_freq_sim = st.sidebar.selectbox("Frequency:", ["Sem Rebalanceamento", "Mensal", "Trimestral", "Semestral", "Anual", "Diário"], index=0, key='rebal_freq_key')

st.sidebar.markdown("---")
st.sidebar.subheader("Allocation")
weights_orig, weights_sim = {}, {}
imported_data = st.session_state.get('imported_portfolio', {})

if tickers_input:
    total_orig, total_sim = 0, 0
    def_val_calc = 100.0 / len(tickers_input) if len(tickers_input) > 0 else 0
    c1, c2, c3 = st.sidebar.columns([2, 1.5, 1.5])
    c1.markdown("**Ticker**"); c2.markdown("**Curr %**"); c3.markdown("**Sim %**")

    for t in tickers_input:
        c1, c2, c3 = st.sidebar.columns([2, 1.5, 1.5])
        c1.text(t)
        if imported_data: val_default = imported_data.get(t, 0.0)
        else: val_default = def_val_calc
        
        w_o = c2.number_input(f"o_{t}", 0.0, 100.0, float(val_default), step=5.0, label_visibility="collapsed")
        key_sim = f"sim_{t}"
        if key_sim not in st.session_state: st.session_state[key_sim] = float(val_default)
        w_s = c3.number_input(f"s_{t}", 0.0, 100.0, key=key_sim, step=5.0, label_visibility="collapsed")
        weights_orig[t] = w_o; weights_sim[t] = w_s 
        total_orig += w_o; total_sim += w_s
    
    cash_orig = 100 - total_orig; cash_sim = 100 - total_sim
    st.sidebar.info(f"Cash Position: Current {cash_orig:.0f}% | Simulated {cash_sim:.0f}%")
    
    if total_orig > 0:
        df_export = pd.DataFrame(list(weights_orig.items()), columns=["Ativo", "Peso"])
        csv_exp = df_export.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
        st.sidebar.download_button("Export Current Portfolio (CSV)", data=csv_exp, file_name="my_portfolio.csv", mime="text/csv", use_container_width=True)

# ==============================================================================
# 4. PROCESSAMENTO E CÁLCULOS
# ==============================================================================
all_tickers = list(set(tickers_input + [bench_ticker]))

with st.spinner("Fetching market data..."):
    df_prices = get_market_data(all_tickers, start_date, end_date)

if df_prices.empty: st.error("No data found."); st.stop()

df_ret = df_prices.ffill().pct_change().iloc[1:]

if bench_ticker in df_ret.columns: bench_ret = df_ret[bench_ticker]
else: bench_ret = pd.Series(0, index=df_ret.index)

valid_assets = [t for t in tickers_input if t in df_ret.columns]
assets_ret = df_ret[valid_assets]

if not valid_assets: st.error("Assets not found in data."); st.stop()

# Simulações de Retorno
ret_orig = calculate_flexible_portfolio(assets_ret, weights_orig, cash_orig, rf_input, mgmt_fee, rebal_freq="Diário")
ret_sim = calculate_flexible_portfolio(assets_ret, weights_sim, cash_sim, rf_input, mgmt_fee, rebal_freq=rebal_freq_sim)

asset_stats = {}
for t in valid_assets:
    m = calculate_metrics(assets_ret[t], rf_input, bench_ret)
    if not m: continue
    up, down = calculate_capture_ratios(assets_ret[t], bench_ret)
    asset_stats[t] = {
        "Beta": m.get("Beta", 1.0), "UpCapture": up, "DownCapture": down, 
        "Vol": m.get("Volatilidade", 0.0), "SemiDev": m.get("Semi-Desvio", 0.0), 
        "UpsideDev": m.get("Upside-Desvio", 0.0), "Ret": m.get("Retorno Anualizado", 0.0)
    }

# Métricas Comparativas
m_orig = calculate_metrics(ret_orig, rf_input, bench_ret)
m_sim = calculate_metrics(ret_sim, rf_input, bench_ret)
m_bench = calculate_metrics(bench_ret, rf_input, bench_ret)

# --- NOVA LÓGICA DE FORMATAÇÃO (BETA E SHARPE COMO NÚMEROS) ---
metrics_order = ["Retorno do Período", "Retorno Anualizado", "Volatilidade", "Semi-Desvio", "Beta", "Sharpe", "Sortino", "Max Drawdown", "VaR 95%", "CVaR 95%"]
keys_present = [k for k in metrics_order if k in m_orig]

def format_value(metric_name, val):
    if metric_name in ["Beta", "Sharpe", "Sortino"]: return f"{val:.2f}"
    return f"{val:.2%}"

df_comp_display = pd.DataFrame({
    "Metric": keys_present, 
    "Current (Fixed W)": [format_value(k, m_orig.get(k, 0)) for k in keys_present], 
    f"Simulated ({rebal_freq_sim})": [format_value(k, m_sim.get(k, 0)) for k in keys_present], 
    "Benchmark": [format_value(k, m_bench.get(k, 0)) for k in keys_present]
})

# ==============================================================================
# 5. DASHBOARD
# ==============================================================================
st.title("Portfolio Risk Management System")

col_kpi, col_delta = st.columns([3, 1])
with col_kpi:
    st.markdown(f"#### Performance Metrics (Simulated Rebal: {rebal_freq_sim})")
    st.dataframe(df_comp_display.set_index("Metric"), use_container_width=True)
    if rebal_freq_sim != "Diário": st.info(f"ℹ️ Drift active: '{rebal_freq_sim}' vs 'Fixed Weights'.")

with col_delta:
    st.markdown("##### Performance Delta")
    d_ret = m_sim.get("Retorno do Período", 0) - m_orig.get("Retorno do Período", 0)
    d_beta = m_sim.get("Beta", 0) - m_orig.get("Beta", 0)
    st.metric("Total Period Return", f"{m_sim.get('Retorno do Período', 0):.2%}", delta=f"{d_ret:.2%}")
    st.metric("Portfolio Beta", f"{m_sim.get('Beta', 0):.2f}", delta=f"{d_beta:.2f}", delta_color="inverse")

# --- BLOCO B: STRESS TEST ---
with st.expander("Stress Test Scenarios (Historical)", expanded=False):
    scenario = st.radio("Select Scenario:", ["COVID-19 Crash (2020)", "Hawkish Cycle (2021-2022)"], horizontal=True)
    if scenario == "COVID-19 Crash (2020)": s_start, s_end, period_start, period_end = "2020-01-20", "2020-03-30", "2020-01-23", "2020-03-23"
    else: s_start, s_end, period_start, period_end = "2021-06-01", "2022-07-25", "2021-06-08", "2022-07-18"

    df_bench_stress = get_market_data([bench_ticker], s_start, s_end)
    df_assets_stress = get_market_data(tickers_input, s_start, s_end)

    if not df_bench_stress.empty:
        mask_b = (df_bench_stress.index >= pd.to_datetime(period_start)) & (df_bench_stress.index <= pd.to_datetime(period_end))
        bench_cut = df_bench_stress.loc[mask_b]
        if not bench_cut.empty:
            bench_res = (bench_cut.iloc[-1, 0] / bench_cut.iloc[0, 0]) - 1
            perfs, used_proxy = {}, []
            for t in tickers_input:
                asset_return, has_data = 0.0, False
                if t in df_assets_stress.columns:
                    mask_a = (df_assets_stress.index >= pd.to_datetime(period_start)) & (df_assets_stress.index <= pd.to_datetime(period_end))
                    s_asset = df_assets_stress.loc[mask_a, t].dropna()
                    if not s_asset.empty: asset_return, has_data = (s_asset.iloc[-1] / s_asset.iloc[0]) - 1, True
                if not has_data:
                    beta_proxy = asset_stats.get(t, {}).get("Beta", 1.0)
                    asset_return = beta_proxy * bench_res
                    used_proxy.append(t)
                perfs[t] = asset_return
            s_orig = sum([weights_orig.get(t, 0)/100 * perfs[t] for t in tickers_input])
            s_sim = sum([weights_sim.get(t, 0)/100 * perfs[t] for t in tickers_input])
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Benchmark", f"{bench_res:.2%}")
            c2.metric("Current Portfolio", f"{s_orig:.2%}")
            c3.metric("Simulated Portfolio", f"{s_sim:.2%}")
            if used_proxy: st.caption(f"⚠️ Proxy used for: {', '.join(used_proxy)}")

# --- BLOCO C: ABAS GRÁFICAS ---
st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Risk vs Return", "Volatility Quality", "Capture Ratios", "Correlation Matrix", "History", "Portfolio Solver"])

with tab1:
    scatter_data = []
    for t, s in asset_stats.items(): scatter_data.append({"Label": t, "X": s["Vol"], "Y": s["Ret"], "Type": "Asset", "Size": 8})
    scatter_data.append({"Label": "CURRENT", "X": m_orig.get("Volatilidade", 0), "Y": m_orig.get("Retorno Anualizado", 0), "Type": "Current Portfolio", "Size": 20})
    scatter_data.append({"Label": "SIMULATED", "X": m_sim.get("Volatilidade", 0), "Y": m_sim.get("Retorno Anualizado", 0), "Type": "Simulated Portfolio", "Size": 20})
    fig1 = px.scatter(pd.DataFrame(scatter_data), x="X", y="Y", color="Type", size="Size", text="Label", title="Risk vs Return Analysis")
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    q_data = [{"Label": t, "Vol": s["Vol"], "SemiDev": s["SemiDev"]} for t, s in asset_stats.items()]
    df_q = pd.DataFrame(q_data)
    fig2 = px.scatter(df_q, x="Vol", y="SemiDev", text="Label", title="Vol Quality")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    up_s, down_s = calculate_capture_ratios(ret_sim, bench_ret)
    c_data = [{"Label": t, "Up": s["UpCapture"], "Down": s["DownCapture"], "Type": "Asset"} for t, s in asset_stats.items()]
    c_data.append({"Label": "SIMULATED", "Up": up_s, "Down": down_s, "Type": "Portfolio"})
    fig3 = px.scatter(pd.DataFrame(c_data), x="Down", y="Up", text="Label", color="Type", title="Capture Ratios")
    fig3.add_vline(x=100, line_dash="dash"); fig3.add_hline(y=100, line_dash="dash")
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    fig4 = px.imshow(assets_ret.corr(), text_auto=".2f", color_continuous_scale="RdYlGn", title="Correlation Matrix")
    st.plotly_chart(fig4, use_container_width=True)

with tab5:
    df_cum = pd.DataFrame({"Simulated": (1 + ret_sim).cumprod(), "Benchmark": (1 + bench_ret).cumprod()})
    fig5 = px.line(df_cum, title="Cumulative Return")
    st.plotly_chart(fig5, use_container_width=True)

with tab6:
    st.markdown("### Portfolio Optimization")
    rf_daily = (1 + rf_input/100)**(1/252) - 1
    df_opt = pd.concat([assets_ret, pd.Series(rf_daily, index=assets_ret.index, name="CASH")], axis=1)
    opt_assets = df_opt.columns.tolist()
    col_setup, col_res = st.columns([1, 2])
    with col_setup:
        target_obj = st.selectbox("Objective Function:", ["Max Sortino", "Min Downside Volatility", "Max Return (Target Semi-Dev)"])
        target_v = st.number_input("Target Downside Vol %", value=5.0) if "Target" in target_obj else None
        bounds = [(0.0, 1.0) for _ in opt_assets]
        if st.button("Run Solver", type="primary"):
            res = run_solver(df_opt, rf_input, tuple(bounds), target_obj, mgmt_fee, target_v)
            if res.success:
                st.session_state['solver_res'] = {'weights': res.x, 'assets': opt_assets}
    with col_res:
        if 'solver_res' in st.session_state:
            sw, sa = st.session_state['solver_res']['weights'], st.session_state['solver_res']['assets']
            df_w = pd.DataFrame({"Asset": sa, "Weight": sw * 100}).query("Weight > 0.01")
            st.plotly_chart(px.pie(df_w, values="Weight", names="Asset", title="Optimal Allocation"))
            st.dataframe(df_w.sort_values("Weight", ascending=False))

# ==============================================================================
# 6. RELATÓRIO EXECUTIVO (SIDEBAR)
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Relatórios")
if st.sidebar.button("Gerar Relatório Interativo (.html)", use_container_width=True, type="primary"):
    figs_to_report = {
        "Histórico de Retorno": fig5,
        "Risk vs Return": fig1,
        "Capture Ratios": fig3,
        "Matriz de Correlação": fig4,
        "Qualidade de Volatilidade": fig2
    }
    report_html = generate_interactive_report(df_comp_display, figs_to_report)
    st.sidebar.download_button(
        label="📥 Baixar Agora",
        data=report_html,
        file_name=f"Relatorio_{datetime.now().strftime('%Y%m%d')}.html",
        mime="text/html",
        use_container_width=True
    )
