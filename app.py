import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.optimize import minimize

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
    """Baixa dados do Yahoo Finance com tratamento de erros e formatos."""
    if not tickers: return pd.DataFrame()
    try:
        s_date = pd.to_datetime(start_date) - timedelta(days=10)
        
        # CORREÇÃO: threads=False evita o travamento no Streamlit Cloud
        df = yf.download(tickers, start=s_date, end=end_date, progress=False, auto_adjust=False, threads=False)
        
        if df.empty: return pd.DataFrame()
        
        # Tratamento robusto para MultiIndex (nova versão yfinance) ou Index Simples
        data = pd.DataFrame()
        
        # Verifica se é MultiIndex (vários ativos)
        if isinstance(df.columns, pd.MultiIndex):
            # Tenta extrair 'Adj Close' no nível 0 ou 1
            if 'Adj Close' in df.columns.get_level_values(0):
                data = df.xs('Adj Close', axis=1, level=0)
            elif 'Close' in df.columns.get_level_values(0):
                data = df.xs('Close', axis=1, level=0)
            # Tenta verificar se os níveis estão invertidos (Ticker no nível 0, Price Type no 1)
            elif 'Adj Close' in df.columns.get_level_values(1):
                data = df.xs('Adj Close', axis=1, level=1)
            elif 'Close' in df.columns.get_level_values(1):
                data = df.xs('Close', axis=1, level=1)
            else:
                # Fallback: tenta pegar a primeira coluna de cada par
                data = df.iloc[:, 0] 
        else:
            # Estrutura simples (um ativo ou colunas flat)
            if 'Adj Close' in df.columns: data = df['Adj Close']
            elif 'Close' in df.columns: data = df['Close']
            else: data = df

        # Garante que seja DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame()
            if len(tickers) == 1:
                data.columns = tickers
        
        # Remove fuso horário para evitar erros de comparação
        data.index = data.index.tz_localize(None)
        
        data = data[data.index >= pd.to_datetime(start_date)]
        return data.dropna()
    except Exception as e:
        return pd.DataFrame()

def calculate_metrics(returns, rf_annual, benchmark_returns=None):
    """Calcula todas as métricas de risco e retorno."""
    if returns.empty: return {}
    rf_daily = (1 + rf_annual/100)**(1/252) - 1
    
    total_return = (1 + returns).prod() - 1
    ann_return = (1 + total_return)**(252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    
    neg_ret = returns[returns < 0]
    semi_dev = neg_ret.std() * np.sqrt(252)
    
    excess_ret = returns - rf_daily
    sharpe = (excess_ret.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    sortino = (excess_ret.mean() / neg_ret.std()) * np.sqrt(252) if neg_ret.std() != 0 else 0
    
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
        "Retorno Anual": ann_return,
        "Volatilidade": ann_vol,
        "Semi-Desvio": semi_dev,
        "Beta": beta,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "VaR 95%": var_95,
        "CVaR 95%": cvar_95
    }

def calculate_capture_ratios(asset_ret, bench_ret):
    aligned = pd.concat([asset_ret, bench_ret], axis=1, join='inner').dropna()
    if aligned.empty: return 0.0, 0.0
    r_asset = aligned.iloc[:, 0]
    r_bench = aligned.iloc[:, 1]
    
    up_mask = r_bench > 0
    up_cap = r_asset[up_mask].mean() / r_bench[up_mask].mean() if up_mask.sum() > 0 else 0
    
    down_mask = r_bench < 0
    down_cap = r_asset[down_mask].mean() / r_bench[down_mask].mean() if down_mask.sum() > 0 else 0
    
    return up_cap * 100.0, down_cap * 100.0

def calculate_flexible_portfolio(asset_returns, weights_dict, cash_pct, rf_annual, fee_annual, rebal_freq):
    """
    Simula o portfolio considerando Drift de pesos e Rebalanceamento Periódico.
    """
    rf_daily = (1 + rf_annual/100)**(1/252) - 1
    fee_daily = (1 + fee_annual/100)**(1/252) - 1
    
    tickers = asset_returns.columns.tolist()
    initial_weights = np.array([weights_dict.get(t, 0) for t in tickers]) / 100.0
    w_cash_initial = cash_pct / 100.0
    
    # 1. Modo Simples: Rebalanceamento Diário (Peso Fixo Constante)
    if rebal_freq == 'Diário':
        gross_ret = asset_returns.dot(initial_weights) + (rf_daily * w_cash_initial)
        return gross_ret - fee_daily

    # 2. Modo Avançado: Com Drift e Datas de Rebalanceamento
    rebal_dates = set()
    if rebal_freq != 'Sem Rebalanceamento':
        try:
            if rebal_freq == 'Mensal': resample_code = 'ME'
            elif rebal_freq == 'Trimestral': resample_code = 'QE'
            elif rebal_freq == 'Anual': resample_code = 'YE'
            elif rebal_freq == 'Semestral': resample_code = 'QE' 
            else: resample_code = 'QE' 
            
            temp_resample = asset_returns.resample(resample_code).last().index
            
            if rebal_freq == 'Semestral':
                rebal_dates = set(temp_resample[1::2])
            else:
                rebal_dates = set(temp_resample)
        except:
            if rebal_freq == 'Mensal': rebal_dates = set(asset_returns.resample('M').last().index)
            else: rebal_dates = set(asset_returns.resample('Q').last().index)

    # Simulação Iterativa
    current_weights = initial_weights.copy()
    current_cash_w = w_cash_initial
    
    portfolio_rets = []
    returns_arr = asset_returns.values
    dates = asset_returns.index
    n_days = len(dates)
    
    for i in range(n_days):
        r_assets = returns_arr[i]
        
        # Retorno Bruto do Portfolio
        day_ret = np.sum(current_weights * r_assets) + (current_cash_w * rf_daily)
        
        # Retorno Líquido (após taxa adm)
        net_day_ret = day_ret - fee_daily
        portfolio_rets.append(net_day_ret)
        
        # Atualização dos Pesos (Drift Natural)
        denominator = 1 + day_ret
        if denominator != 0:
            current_weights = current_weights * (1 + r_assets) / denominator
            current_cash_w = current_cash_w * (1 + rf_daily) / denominator
        
        # Checagem de Rebalanceamento
        if dates[i] in rebal_dates:
            current_weights = initial_weights.copy()
            current_cash_w = w_cash_initial

    return pd.Series(portfolio_rets, index=dates)

# --- FUNÇÃO SOLVER ---
def run_solver(df_returns, rf_annual, bounds, target_metric, mgmt_fee_annual=0.0, target_semidev_val=None):
    """
    Roda o otimizador com limite de iterações aumentado e chute inicial inteligente.
    """
    rf_daily = (1 + rf_annual/100)**(1/252) - 1
    fee_daily = (1 + mgmt_fee_annual/100)**(1/252) - 1
    
    num_assets = len(df_returns.columns)
    
    # IMPROVEMENT: Smart Initial Guess based on Bounds
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    
    # Começa no meio do intervalo permitido (média entre min e max)
    initial_guess = (lower_bounds + upper_bounds) / 2
    
    # Normaliza para somar 1.0 (100%)
    sum_guess = np.sum(initial_guess)
    if sum_guess > 0:
        initial_guess = initial_guess / sum_guess
    else:
        initial_guess = np.array([1/num_assets] * num_assets)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    if target_metric == "Max Return (Target Semi-Dev)" and target_semidev_val is not None:
        def semidev_constraint(weights):
            w = np.array(weights)
            gross = df_returns.dot(w)
            net = gross - fee_daily
            neg = net[net < 0]
            current_semi = neg.std() * np.sqrt(252)
            return (target_semidev_val/100.0) - current_semi
        constraints.append({'type': 'ineq', 'fun': semidev_constraint})

    def objective(weights):
        w = np.array(weights)
        gross_ret = df_returns.dot(w)
        net_ret = gross_ret - fee_daily
        
        # Penalidade para soma fugir de 1
        if abs(np.sum(w) - 1.0) > 0.001: return 1e5
        
        if target_metric == "Max Sortino":
            neg_ret = net_ret[net_ret < 0]
            semi_dev = neg_ret.std() * np.sqrt(252)
            if semi_dev == 0: return 1e5
            excess_ret = net_ret - rf_daily
            sortino = (excess_ret.mean() / neg_ret.std()) * np.sqrt(252)
            return -sortino
            
        elif target_metric == "Min Downside Volatility":
            neg_ret = net_ret[net_ret < 0]
            semi_dev = neg_ret.std() * np.sqrt(252)
            return semi_dev
        
        elif target_metric == "Max Return (Target Semi-Dev)":
            total_ret = (1 + net_ret).prod() - 1
            ann_ret = (1 + total_ret)**(252 / len(net_ret)) - 1
            return -ann_ret
            
    # IMPROVEMENT: Increased maxiter
    result = minimize(
        objective, 
        initial_guess, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints, 
        tol=1e-6,
        options={'maxiter': 1000} # Increased from default 100
    )
    return result

# ==============================================================================
# 3. BARRA LATERAL (INPUTS)
# ==============================================================================
st.sidebar.header("Portfolio Configuration")

# Seleção de Ativos
tickers_text = st.sidebar.text_area("Asset Tickers:", value="PETR4.SA, VALE3.SA, BPAC11.SA, ITUB4.SA, WEGE3.SA, IVVB11.SA")
tickers_input = [t.strip().upper() for t in tickers_text.split(',') if t.strip()]

# Seleção de Datas
periodo_option = st.sidebar.radio("Time Horizon:", ["1 Ano", "2 Anos", "Desde 2020", "Personalizado"], horizontal=True)
end_date = datetime.today()
if periodo_option == "1 Ano": start_date = end_date - timedelta(days=365)
elif periodo_option == "2 Anos": start_date = end_date - timedelta(days=730)
elif periodo_option == "Desde 2020": start_date = datetime(2020, 1, 1)
else:
    c_start, c_end = st.sidebar.columns(2)
    start_date = c_start.date_input("Start Date", value=datetime(2022,1,1))
    end_date = c_end.date_input("End Date", value=datetime.today())

# Parâmetros de Mercado e Custos
st.sidebar.subheader("Market & Costs")
c_rf, c_fee = st.sidebar.columns(2)
rf_input = c_rf.number_input("Risk Free (% p.a.)", value=10.5, step=0.5)
mgmt_fee = c_fee.number_input("Mgmt Fee (% p.a.)", value=0.0, step=0.1, help="Taxa de Gestão anual descontada diariamente.")

bench_ticker = st.sidebar.text_input("Benchmark Ticker", value="^BVSP")

# --- OPÇÃO DE REBALANCEAMENTO ---
st.sidebar.subheader("Rebalancing (Simulated)")
rebal_freq_sim = st.sidebar.selectbox(
    "Frequency:", 
    ["Sem Rebalanceamento", "Mensal", "Trimestral", "Semestral", "Anual", "Diário"],
    index=2, # Default Trimestral
    help="'Diário' = Pesos Fixos Constantes. 'Sem Rebalanceamento' = Buy & Hold (pesos variam livremente)."
)

# Pesos
st.sidebar.markdown("---")
st.sidebar.subheader("Allocation: Current vs Simulated")
weights_orig, weights_sim = {}, {}

if tickers_input:
    c1, c2, c3 = st.sidebar.columns([2, 1.5, 1.5])
    c1.markdown("**Ticker**"); c2.markdown("**Curr %**"); c3.markdown("**Sim %**")
    total_orig, total_sim = 0, 0
    
    def_val = float(int(100/len(tickers_input)))
    
    for t in tickers_input:
        c1, c2, c3 = st.sidebar.columns([2, 1.5, 1.5])
        c1.text(t)
        
        # Current
        w_o = c2.number_input(f"o_{t}", 0.0, 100.0, def_val, step=5.0, label_visibility="collapsed")
        
        # Simulated (Session State)
        key_sim = f"sim_{t}"
        if key_sim not in st.session_state:
            st.session_state[key_sim] = def_val
            
        w_s = c3.number_input(f"s_{t}", 0.0, 100.0, key=key_sim, step=5.0, label_visibility="collapsed")
        
        weights_orig[t] = w_o
        weights_sim[t] = w_s 
        
        total_orig += w_o
        total_sim += w_s
    
    cash_orig = 100 - total_orig
    cash_sim = 100 - total_sim
    st.sidebar.info(f"Cash Position: Current {cash_orig:.0f}% | Simulated {cash_sim:.0f}%")

# ==============================================================================
# 4. PROCESSAMENTO DE DADOS E CÁLCULOS
# ==============================================================================
all_tickers = list(set(tickers_input + [bench_ticker]))

with st.spinner("Fetching market data..."):
    df_prices = get_market_data(all_tickers, start_date, end_date)

if df_prices.empty: st.error("No data found. Check tickers or internet connection."); st.stop()
df_ret = df_prices.ffill().pct_change().dropna()

if bench_ticker in df_ret.columns: bench_ret = df_ret[bench_ticker]
else: bench_ret = pd.Series(0, index=df_ret.index)

valid_assets = [t for t in tickers_input if t in df_ret.columns]
assets_ret = df_ret[valid_assets]

# --- CÁLCULO DAS CARTEIRAS ---
ret_orig = calculate_flexible_portfolio(assets_ret, weights_orig, cash_orig, rf_input, mgmt_fee, rebal_freq="Diário")
ret_sim = calculate_flexible_portfolio(assets_ret, weights_sim, cash_sim, rf_input, mgmt_fee, rebal_freq=rebal_freq_sim)

# Estatísticas Individuais
asset_stats = {}
for t in valid_assets:
    m = calculate_metrics(assets_ret[t], rf_input, bench_ret)
    up, down = calculate_capture_ratios(assets_ret[t], bench_ret)
    asset_stats[t] = {
        "Beta": m.get("Beta", 1.0), "UpCapture": up, "DownCapture": down,
        "Vol": m["Volatilidade"], "SemiDev": m["Semi-Desvio"], "Ret": m["Retorno Anual"]
    }

# ==============================================================================
# 5. DASHBOARD PRINCIPAL
# ==============================================================================
st.title("Portfolio Risk Management System")

# --- BLOCO A: KPIs ---
m_orig = calculate_metrics(ret_orig, rf_input, bench_ret)
m_sim = calculate_metrics(ret_sim, rf_input, bench_ret)
m_bench = calculate_metrics(bench_ret, rf_input, bench_ret)

col_kpi, col_delta = st.columns([3, 1])
with col_kpi:
    st.markdown(f"#### Performance Metrics (Simulated Rebal: {rebal_freq_sim})")
    df_comp = pd.DataFrame({
        "Metric": m_orig.keys(), 
        "Current (Fixed W)": m_orig.values(), 
        f"Simulated ({rebal_freq_sim})": m_sim.values(), 
        "Benchmark": m_bench.values()
    })
    for c in df_comp.columns[1:]:
        df_comp[c] = df_comp[c].apply(lambda x: f"{x:.2%}" if abs(x)<5 and x!=0 else f"{x:.2f}")
    st.dataframe(df_comp.set_index("Metric"), use_container_width=True)

with col_delta:
    st.markdown("##### Performance Delta")
    d_ret = m_sim["Retorno Anual"] - m_orig["Retorno Anual"]
    d_beta = m_sim["Beta"] - m_orig["Beta"]
    st.metric("Return (a.a.)", f"{m_sim['Retorno Anual']:.2%}", delta=f"{d_ret:.2%}")
    st.metric("Portfolio Beta", f"{m_sim['Beta']:.2f}", delta=f"{d_beta:.2f}", delta_color="inverse")
    if mgmt_fee > 0: st.caption(f"*Includes annual management fee of {mgmt_fee}%")

# --- BLOCO B: STRESS TEST (COM BETA PROXY) ---
with st.expander("Stress Test Scenarios (Historical)", expanded=False):
    scenario = st.radio("Select Scenario:", ["COVID-19 Crash (2020)", "Hawkish Cycle (2021-2022)"], horizontal=True)
    
    # Definição das Datas
    if scenario == "COVID-19 Crash (2020)":
        s_start, s_end = "2020-01-20", "2020-03-30"
        period_start, period_end = "2020-01-23", "2020-03-23"
    else:
        s_start, s_end = "2021-06-01", "2022-07-25"
        period_start, period_end = "2021-06-08", "2022-07-18"

    # 1. Baixar Benchmark separadamente com CORREÇÃO DE THREADS
    try:
        df_bench_stress = yf.download(bench_ticker, start=s_start, end=s_end, progress=False, auto_adjust=False, threads=False)
        if not df_bench_stress.empty:
            # Tratamento simplificado para benchmark
            if isinstance(df_bench_stress.columns, pd.MultiIndex):
                if 'Adj Close' in df_bench_stress.columns.get_level_values(0):
                    df_bench_stress = df_bench_stress['Adj Close']
                elif 'Close' in df_bench_stress.columns.get_level_values(0):
                    df_bench_stress = df_bench_stress['Close']
                else:
                    df_bench_stress = df_bench_stress.iloc[:, 0]
            else:
                if 'Adj Close' in df_bench_stress.columns: df_bench_stress = df_bench_stress['Adj Close']
                elif 'Close' in df_bench_stress.columns: df_bench_stress = df_bench_stress['Close']
            
            # Garante Série
            if isinstance(df_bench_stress, pd.DataFrame):
                df_bench_stress = df_bench_stress.iloc[:, 0] 
                
            df_bench_stress.index = df_bench_stress.index.tz_localize(None)
    except:
        df_bench_stress = pd.Series()

    # 2. Baixar Ativos com CORREÇÃO DE THREADS
    df_assets_stress = pd.DataFrame()
    if tickers_input:
        try:
            raw_assets = yf.download(tickers_input, start=s_start, end=s_end, progress=False, auto_adjust=False, threads=False)
            if not raw_assets.empty:
                # Tratamento simplificado para ativos
                if isinstance(raw_assets.columns, pd.MultiIndex):
                    if 'Adj Close' in raw_assets.columns.get_level_values(0):
                         df_assets_stress = raw_assets.xs('Adj Close', axis=1, level=0)
                    elif 'Close' in raw_assets.columns.get_level_values(0):
                         df_assets_stress = raw_assets.xs('Close', axis=1, level=0)
                    else:
                         df_assets_stress = raw_assets.iloc[:, 0]
                else:
                    if 'Adj Close' in raw_assets.columns: df_assets_stress = raw_assets['Adj Close']
                    elif 'Close' in raw_assets.columns: df_assets_stress = raw_assets['Close']
                    else: df_assets_stress = raw_assets
                
                if isinstance(df_assets_stress, pd.Series):
                    df_assets_stress = df_assets_stress.to_frame(name=tickers_input[0])
                
                df_assets_stress.index = df_assets_stress.index.tz_localize(None)
        except:
            pass

    # Lógica de Cálculo
    if not df_bench_stress.empty:
        mask_b = (df_bench_stress.index >= pd.to_datetime(period_start)) & (df_bench_stress.index <= pd.to_datetime(period_end))
        bench_cut = df_bench_stress.loc[mask_b]
        
        if not bench_cut.empty:
            bench_res = (bench_cut.iloc[-1] / bench_cut.iloc[0]) - 1
            days_stress = (pd.to_datetime(period_end) - pd.to_datetime(period_start)).days
            fee_factor_stress = (1 + mgmt_fee/100)**(days_stress/365) - 1

            perfs = {}
            used_proxy = []

            for t in tickers_input:
                asset_return = 0.0
                has_data = False
                
                if t in df_assets_stress.columns:
                    mask_a = (df_assets_stress.index >= pd.to_datetime(period_start)) & (df_assets_stress.index <= pd.to_datetime(period_end))
                    s_asset = df_assets_stress.loc[mask_a, t].dropna()
                    
                    if not s_asset.empty:
                        asset_return = (s_asset.iloc[-1] / s_asset.iloc[0]) - 1
                        has_data = True
                
                if not has_data:
                    beta_proxy = asset_stats.get(t, {}).get("Beta", 1.0)
                    asset_return = beta_proxy * bench_res
                    used_proxy.append(t)
                
                perfs[t] = asset_return

            s_orig = sum([weights_orig.get(t, 0)/100 * perfs[t] for t in tickers_input]) - fee_factor_stress
            s_sim = sum([weights_sim.get(t, 0)/100 * perfs[t] for t in tickers_input]) - fee_factor_stress
            
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Benchmark ({scenario.split()[0]})", f"{bench_res:.2%}", delta="Market Move", delta_color="inverse")
            
            msg_help = "Calculated using Beta Proxy for missing assets." if used_proxy else "Full historical data available."
            c2.metric("Current Portfolio", f"{s_orig:.2%}", delta=f"{s_orig-bench_res:.2%} vs Bench", help=msg_help)
            c3.metric("Simulated Portfolio", f"{s_sim:.2%}", delta=f"{s_sim-s_orig:.2%} vs Current", help=msg_help)
            
            if used_proxy:
                st.caption(f"⚠️ Assets estimated via Beta Proxy (did not exist then): {', '.join(used_proxy)}")

        else: st.warning("Benchmark data empty for this scenario range.")
    else: st.warning(f"Insufficient Benchmark data for {scenario}.")

# --- BLOCO C: ABAS GRÁFICAS ---
st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Risk vs Return", 
    "Volatility Quality", 
    "Capture Ratios", 
    "Correlation Matrix", 
    "History",
    "Portfolio Solver"
])

with tab1:
    risk_mode = st.radio("Risk Metric (X-Axis):", ["Total Volatility", "Downside Deviation"], horizontal=True)
    x_key = "Vol" if risk_mode == "Total Volatility" else "SemiDev"
    ratio_name = "Sharpe" if risk_mode == "Total Volatility" else "Sortino"
    
    scatter_data = []
    for t, s in asset_stats.items():
        scatter_data.append({"Label": t, "X": s[x_key], "Y": s["Ret"], "Type": "Asset", "Size": 8})
    
    x_orig = m_orig["Volatilidade" if x_key=="Vol" else "Semi-Desvio"]
    y_orig = m_orig["Retorno Anual"]
    x_sim = m_sim["Volatilidade" if x_key=="Vol" else "Semi-Desvio"]
    y_sim = m_sim["Retorno Anual"]

    scatter_data.append({"Label": "CURRENT", "X": x_orig, "Y": y_orig, "Type": "Current Portfolio", "Size": 20})
    scatter_data.append({"Label": "SIMULATED", "X": x_sim, "Y": y_sim, "Type": "Simulated Portfolio", "Size": 20})
    scatter_data.append({"Label": "BENCHMARK", "X": m_bench["Volatilidade" if x_key=="Vol" else "Semi-Desvio"], "Y": m_bench["Retorno Anual"], "Type": "Benchmark", "Size": 12})
    
    df_s1 = pd.DataFrame(scatter_data)
    color_map = {"Asset": "#636EFA", "Current Portfolio": "#00CC96", "Simulated Portfolio": "#FFD700", "Benchmark": "#7F7F7F"}
    
    fig1 = px.scatter(df_s1, x="X", y="Y", color="Type", size="Size", text="Label", color_discrete_map=color_map)
    fig1.add_annotation(x=x_sim, y=y_sim, ax=x_orig, ay=y_orig, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#FFD700")

    max_x = df_s1["X"].max() * 1.2
    rf_dec = rf_input / 100
    for r in [0.5, 1.0, 1.5, 2.0]:
        fig1.add_trace(go.Scatter(x=[0, max_x], y=[rf_dec, rf_dec + r * max_x], mode='lines', line=dict(color='rgba(150,150,150,0.3)', width=1, dash='dot'), showlegend=False, hoverinfo='skip'))
        fig1.add_annotation(x=max_x, y=rf_dec + r * max_x, text=f"{ratio_name} {r}", showarrow=False, xanchor="left", font=dict(size=10, color="gray"))

    fig1.update_traces(textposition='top center')
    fig1.update_layout(xaxis_title=risk_mode, yaxis_title="Annualized Return (Net)", height=600)
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    q_data = [{"Label": t, "Vol": s["Vol"], "SemiDev": s["SemiDev"]} for t, s in asset_stats.items()]
    df_q = pd.DataFrame(q_data)
    max_v = df_q["Vol"].max() * 1.1 if not df_q.empty else 1
    
    fig2 = px.scatter(df_q, x="Vol", y="SemiDev", text="Label", height=500)
    fig2.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v, line=dict(color="darkred", width=1, dash="dash"))
    fig2.update_layout(xaxis_title="Total Volatility (Std Dev)", yaxis_title="Downside Deviation")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    c_ctrl_1, c_ctrl_2 = st.columns([1, 1])
    up_o, down_o = calculate_capture_ratios(ret_orig, bench_ret)
    up_s, down_s = calculate_capture_ratios(ret_sim, bench_ret)
    
    c_data = [{"Label": t, "Up": s["UpCapture"], "Down": s["DownCapture"], "Type": "Asset"} for t, s in asset_stats.items()]
    c_data.append({"Label": "CURRENT", "Up": up_o, "Down": down_o, "Type": "Portfolio"})
    c_data.append({"Label": "SIMULATED", "Up": up_s, "Down": down_s, "Type": "Portfolio"})
    
    df_c = pd.DataFrame(c_data)
    with c_ctrl_1:
        default_sel = ["CURRENT", "SIMULATED"] + [t for t in tickers_input[:3] if t in df_c["Label"].values]
        selected = st.multiselect("Filter Chart:", df_c["Label"].unique(), default=default_sel)
    with c_ctrl_2:
        zoom = st.checkbox("Auto Zoom", value=False)

    df_plot = df_c[df_c["Label"].isin(selected)] if selected else df_c
    fig3 = px.scatter(df_plot, x="Down", y="Up", text="Label", color="Type", color_discrete_map={"Asset": "#636EFA", "Portfolio": "#00CC96"})
    
    fig3.add_vline(x=100, line_dash="dash", line_color="gray"); fig3.add_hline(y=100, line_dash="dash", line_color="gray")
    fig3.add_shape(type="rect", x0=-50, y0=100, x1=100, y1=250, line_width=0, fillcolor="green", opacity=0.1)
    if not zoom: fig3.update_xaxes(range=[-10, 160]); fig3.update_yaxes(range=[0, 160])
    st.plotly_chart(fig3, use_container_width=True)

    table_rows = []
    for t, s in asset_stats.items():
        table_rows.append({"Name": t, "Upside": s["UpCapture"], "Downside": s["DownCapture"], "Spread": s["UpCapture"] - s["DownCapture"], "Ratio": s["UpCapture"]/s["DownCapture"] if s["DownCapture"]!=0 else 0})
    table_rows.append({"Name": "CURRENT", "Upside": up_o, "Downside": down_o, "Spread": up_o - down_o, "Ratio": up_o/down_o if down_o!=0 else 0})
    table_rows.append({"Name": "SIMULATED", "Upside": up_s, "Downside": down_s, "Spread": up_s - down_s, "Ratio": up_s/down_s if down_s!=0 else 0})
    st.dataframe(pd.DataFrame(table_rows).sort_values("Spread", ascending=False).style.format({"Upside": "{:.1f}%", "Downside": "{:.1f}%", "Spread": "{:.1f}%", "Ratio": "{:.2f}"}).applymap(lambda x: 'color: lightgreen' if x < 0 else '', subset=['Downside']).applymap(lambda x: 'font-weight: bold', subset=['Spread']), use_container_width=True)

with tab4:
    st.plotly_chart(px.imshow(assets_ret.corr(), text_auto=".2f", aspect="auto", color_continuous_scale="RdYlGn", zmin=-1, zmax=1), use_container_width=True)

with tab5:
    cum_orig, cum_sim, cum_bench = (1 + ret_orig).cumprod(), (1 + ret_sim).cumprod(), (1 + bench_ret).cumprod()
    st.line_chart(pd.DataFrame({"Current (Fixed)": cum_orig, f"Simulated ({rebal_freq_sim})": cum_sim, "Benchmark": cum_bench}))
    dd_orig = (cum_orig - cum_orig.cummax()) / cum_orig.cummax()
    dd_sim = (cum_sim - cum_sim.cummax()) / cum_sim.cummax()
    st.area_chart(pd.DataFrame({"Current DD": dd_orig, "Simulated DD": dd_sim}))

with tab6:
    st.markdown("### Portfolio Optimization")
    st.info("Set minimum and maximum constraints. Results are Net of Management Fees.")
    
    rf_daily = (1 + rf_input/100)**(1/252) - 1
    cash_series = pd.Series(rf_daily, index=assets_ret.index, name="CASH")
    
    df_opt = pd.concat([assets_ret, cash_series], axis=1)
    opt_assets = df_opt.columns.tolist()
    
    col_setup, col_res = st.columns([1, 2])
    
    with col_setup:
        # Seletor de Objetivo
        target_obj = st.selectbox("Objective Function:", ["Max Sortino", "Min Downside Volatility", "Max Return (Target Semi-Dev)"])
        
        # Campo condicional para Target Semi-Dev
        target_semidev_input = None
        if target_obj == "Max Return (Target Semi-Dev)":
            target_semidev_input = st.number_input("Target Downside Volatility (%)", value=5.0, step=0.5, min_value=0.1)
        
        st.markdown("##### Weight Constraints (%)")
        
        default_min = [0.0] * len(opt_assets)
        default_max = [100.0] * len(opt_assets)
        
        df_constraints = pd.DataFrame({"Asset": opt_assets, "Min %": default_min, "Max %": default_max})
        edited_df = st.data_editor(
            df_constraints, 
            column_config={
                "Min %": st.column_config.NumberColumn(min_value=0, max_value=100, step=1),
                "Max %": st.column_config.NumberColumn(min_value=0, max_value=100, step=1)
            },
            hide_index=True, use_container_width=True
        )
        
        if st.button("Run Solver", type="primary", use_container_width=True):
            bounds = []
            for index, row in edited_df.iterrows():
                bounds.append((row["Min %"]/100.0, row["Max %"]/100.0))
            
            with st.spinner("Optimizing..."):
                res = run_solver(df_opt, rf_input, tuple(bounds), target_obj, mgmt_fee, target_semidev_input)
            
            st.session_state['solver_result'] = {
                'success': res.success,
                'message': res.message,
                'weights': res.x,
                'opt_assets': opt_assets
            }

    with col_res:
        if 'solver_result' in st.session_state and st.session_state['solver_result']['success']:
            res_data = st.session_state['solver_result']
            opt_weights = res_data['weights']
            opt_assets_saved = res_data['opt_assets']
            
            # Recalcula métricas
            fee_daily = (1 + mgmt_fee/100)**(1/252) - 1
            opt_ret_series = df_opt.dot(opt_weights) - fee_daily
            m_opt = calculate_metrics(opt_ret_series, rf_input, bench_ret)
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Optimized Return (Net)", f"{m_opt['Retorno Anual']:.2%}")
            k2.metric("Sortino Ratio", f"{m_opt['Sortino']:.2f}")
            k3.metric("Downside Vol", f"{m_opt['Semi-Desvio']:.2%}")
            
            c_chart, c_table = st.columns([1.2, 1])
            
            df_w = pd.DataFrame({"Asset": opt_assets_saved, "Weight": opt_weights * 100})
            df_w = df_w[df_w["Weight"] > 0.01] 
            
            with c_chart:
                fig_pie = px.pie(df_w, values="Weight", names="Asset", title="Optimal Allocation", hole=0.4)
                fig_pie.update_traces(textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with c_table:
                st.markdown("##### Detailed Weights")
                st.dataframe(df_w.sort_values("Weight", ascending=False).style.format({"Weight": "{:.2f}%"}), use_container_width=True, hide_index=True)
            
            st.success(f"Solver Status: {res_data['message']}")
            
            def update_weights_callback():
                if 'solver_result' in st.session_state:
                    saved_res = st.session_state['solver_result']
                    for i, asset_name in enumerate(saved_res['opt_assets']):
                        if asset_name != "CASH":
                            key = f"sim_{asset_name}"
                            new_val = saved_res['weights'][i] * 100.0
                            st.session_state[key] = round(new_val, 2)

            st.button("Apply to Simulation", on_click=update_weights_callback)
                
        elif 'solver_result' in st.session_state and not st.session_state['solver_result']['success']:
            st.error(f"Optimization Failed: {st.session_state['solver_result']['message']}")
        else:
            st.info("Adjust constraints and click 'Run Solver'.")
