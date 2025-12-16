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
    """
    Baixa dados do Yahoo Finance com tratamento robusto e ajuste de proventos.
    """
    if not tickers: return pd.DataFrame()
    try:
        # Buffer de dias para garantir que o cálculo de retorno comece exatamente na data pedida
        s_date = pd.to_datetime(start_date) - timedelta(days=20)
        
        # CORREÇÃO CRÍTICA 1: auto_adjust=True
        # Garante que o preço já venha descontado de dividendos e splits.
        # threads=False previne erros no Streamlit Cloud.
        df = yf.download(tickers, start=s_date, end=end_date, progress=False, auto_adjust=True, threads=False)
        
        if df.empty: return pd.DataFrame()
        
        data = pd.DataFrame()
        
        # --- Lógica de Extração da Coluna 'Close' ---
        # O yfinance muda o formato dependendo da versão e qtd de ativos. Isso padroniza:
        if isinstance(df.columns, pd.MultiIndex):
            # Prioridade: Coluna 'Close' no nível superior (formato novo yfinance)
            if 'Close' in df.columns:
                data = df['Close']
            # Formatos antigos/alternativos
            elif 'Close' in df.columns.get_level_values(0):
                data = df.xs('Close', axis=1, level=0)
            elif 'Close' in df.columns.get_level_values(1):
                data = df.xs('Close', axis=1, level=1)
            else:
                # Fallback extremo: pega a primeira coluna de cada ativo
                data = df.iloc[:, 0] 
        else:
            # Apenas 1 ativo ou estrutura flat
            if 'Close' in df.columns:
                data = df[['Close']]
            else:
                data = df.iloc[:, [0]]

        # Garante DataFrame e nomes de colunas corretos
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        if len(tickers) == 1 and data.shape[1] == 1:
            data.columns = tickers
        
        # Limpeza de Fuso Horário e Filtro de Data
        data.index = data.index.tz_localize(None)
        
        # Filtra para iniciar exatamente na data pedida pelo usuário
        data = data[data.index >= pd.to_datetime(start_date)]
        
        # Remove colunas inteiramente vazias, mas NÃO remove linhas ainda (para não perder dados desalinhados)
        data = data.dropna(axis=1, how='all')
        
        return data
    except Exception as e:
        st.error(f"Erro ao baixar dados: {e}")
        return pd.DataFrame()

def calculate_metrics(returns, rf_annual, benchmark_returns=None):
    """
    Calcula métricas de risco e retorno.
    CORREÇÃO 2: Distingue 'Retorno Total' (o que aconteceu) de 'Retorno Anual' (projeção).
    """
    # Remove NaNs apenas para o ativo específico sendo calculado neste momento
    returns = returns.dropna()
    
    if returns.empty: return {}
    
    rf_daily = (1 + rf_annual/100)**(1/252) - 1
    days = len(returns)
    
    # --- CÁLCULOS DE RETORNO ---
    # 1. Retorno Total do Período (Acumulado): É o valor exato que variou entre data X e Y.
    total_return = (1 + returns).prod() - 1
    
    # 2. Retorno Anualizado (CAGR): É a projeção desse ritmo para 1 ano (252 dias úteis).
    # Se o período for curto (ex: 1 mês), isso vai inflar o número (ex: 3% vira 40% a.a.).
    if days > 10:
        ann_return = (1 + total_return)**(252 / days) - 1
    else:
        ann_return = total_return # Para períodos muito curtos, mantém o total para não distorcer
    
    ann_vol = returns.std() * np.sqrt(252)
    
    neg_ret = returns[returns < 0]
    semi_dev = neg_ret.std() * np.sqrt(252)
    
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
        # Alinha as séries apenas onde ambas existem
        aligned = pd.concat([returns, benchmark_returns], axis=1, join='inner').dropna()
        if not aligned.empty and aligned.shape[0] > 10:
            cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
            var_bench = np.var(aligned.iloc[:, 1])
            beta = cov / var_bench if var_bench != 0 else 0
            
    return {
        "Retorno do Período": total_return, # O valor real acumulado (ex: 11%)
        "Retorno Anualizado": ann_return,   # O valor projetado anual (ex: 40%)
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
    """Calcula a carteira dia a dia, tratando dados faltantes (NaN) como retorno zero."""
    rf_daily = (1 + rf_annual/100)**(1/252) - 1
    fee_daily = (1 + fee_annual/100)**(1/252) - 1
    
    tickers = asset_returns.columns.tolist()
    initial_weights = np.array([weights_dict.get(t, 0) for t in tickers]) / 100.0
    w_cash_initial = cash_pct / 100.0
    
    # 1. Rebalanceamento Diário (Peso Constante)
    if rebal_freq == 'Diário':
        # fillna(0) impede que a carteira quebre se 1 ativo não tiver cotação no dia
        gross_ret = asset_returns.fillna(0).dot(initial_weights) + (rf_daily * w_cash_initial)
        return gross_ret - fee_daily

    # 2. Rebalanceamento Periódico (Simulação de Drift)
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
    # Preenche NaNs com 0.0 para cálculo vetorial (dia sem pregão pro ativo não muda o valor financeiro dele)
    returns_arr = asset_returns.fillna(0.0).values
    dates = asset_returns.index
    n_days = len(dates)
    
    for i in range(n_days):
        r_assets = returns_arr[i]
        
        day_ret = np.sum(current_weights * r_assets) + (current_cash_w * rf_daily)
        net_day_ret = day_ret - fee_daily
        portfolio_rets.append(net_day_ret)
        
        denominator = 1 + day_ret
        if denominator != 0:
            current_weights = current_weights * (1 + r_assets) / denominator
            current_cash_w = current_cash_w * (1 + rf_daily) / denominator
        
        if dates[i] in rebal_dates:
            current_weights = initial_weights.copy()
            current_cash_w = w_cash_initial

    return pd.Series(portfolio_rets, index=dates)

# --- FUNÇÃO SOLVER ---
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

# ==============================================================================
# 3. BARRA LATERAL (INPUTS)
# ==============================================================================
st.sidebar.header("Portfolio Configuration")

tickers_text = st.sidebar.text_area("Asset Tickers:", value="VALE3.SA, PETR4.SA, BPAC11.SA")
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

st.sidebar.subheader("Rebalancing (Simulated)")
rebal_freq_sim = st.sidebar.selectbox("Frequency:", ["Sem Rebalanceamento", "Mensal", "Trimestral", "Semestral", "Anual", "Diário"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Allocation")
weights_orig, weights_sim = {}, {}

if tickers_input:
    total_orig, total_sim = 0, 0
    def_val = float(int(100/len(tickers_input)))
    for t in tickers_input:
        c1, c2, c3 = st.sidebar.columns([2, 1.5, 1.5])
        c1.text(t)
        w_o = c2.number_input(f"o_{t}", 0.0, 100.0, def_val, step=5.0, label_visibility="collapsed")
        key_sim = f"sim_{t}"
        if key_sim not in st.session_state: st.session_state[key_sim] = def_val
        w_s = c3.number_input(f"s_{t}", 0.0, 100.0, key=key_sim, step=5.0, label_visibility="collapsed")
        weights_orig[t] = w_o; weights_sim[t] = w_s 
        total_orig += w_o; total_sim += w_s
    cash_orig = 100 - total_orig; cash_sim = 100 - total_sim
    st.sidebar.info(f"Cash Position: Current {cash_orig:.0f}% | Simulated {cash_sim:.0f}%")

# ==============================================================================
# 4. PROCESSAMENTO E CÁLCULOS
# ==============================================================================
all_tickers = list(set(tickers_input + [bench_ticker]))

with st.spinner("Fetching market data..."):
    df_prices = get_market_data(all_tickers, start_date, end_date)

if df_prices.empty: st.error("No data found."); st.stop()

# CORREÇÃO: Não usar dropna() globalmente para não cortar ativos com históricos diferentes
df_ret = df_prices.ffill().pct_change()
# Remove apenas a primeira linha que é sempre NaN devido ao pct_change
df_ret = df_ret.iloc[1:]

if bench_ticker in df_ret.columns: bench_ret = df_ret[bench_ticker]
else: bench_ret = pd.Series(0, index=df_ret.index)

valid_assets = [t for t in tickers_input if t in df_ret.columns]
assets_ret = df_ret[valid_assets]

if not valid_assets: st.error("Assets not found in data."); st.stop()

# Carteiras
ret_orig = calculate_flexible_portfolio(assets_ret, weights_orig, cash_orig, rf_input, mgmt_fee, rebal_freq="Diário")
ret_sim = calculate_flexible_portfolio(assets_ret, weights_sim, cash_sim, rf_input, mgmt_fee, rebal_freq=rebal_freq_sim)

asset_stats = {}
for t in valid_assets:
    m = calculate_metrics(assets_ret[t], rf_input, bench_ret)
    up, down = calculate_capture_ratios(assets_ret[t], bench_ret)
    asset_stats[t] = {
        "Beta": m.get("Beta", 1.0), "UpCapture": up, "DownCapture": down, 
        "Vol": m["Volatilidade"], "SemiDev": m["Semi-Desvio"], 
        "Ret": m["Retorno Anualizado"] # Usa anualizado para o gráfico Scatter
    }

# ==============================================================================
# 5. DASHBOARD
# ==============================================================================
st.title("Portfolio Risk Management System")

# --- BLOCO A: KPIs ---
m_orig = calculate_metrics(ret_orig, rf_input, bench_ret)
m_sim = calculate_metrics(ret_sim, rf_input, bench_ret)
m_bench = calculate_metrics(bench_ret, rf_input, bench_ret)

col_kpi, col_delta = st.columns([3, 1])
with col_kpi:
    st.markdown(f"#### Performance Metrics (Simulated Rebal: {rebal_freq_sim})")
    
    # Ordem das métricas para exibir na tabela
    metrics_order = [
        "Retorno do Período", "Retorno Anualizado", "Volatilidade", "Semi-Desvio", 
        "Beta", "Sharpe", "Sortino", "Max Drawdown", "VaR 95%", "CVaR 95%"
    ]
    
    keys_present = [k for k in metrics_order if k in m_orig]
    
    df_comp = pd.DataFrame({
        "Metric": keys_present, 
        "Current (Fixed W)": [m_orig[k] for k in keys_present], 
        f"Simulated ({rebal_freq_sim})": [m_sim[k] for k in keys_present], 
        "Benchmark": [m_bench.get(k, 0) for k in keys_present]
    })
    
    for c in df_comp.columns[1:]:
        df_comp[c] = df_comp[c].apply(lambda x: f"{x:.2%}" if abs(x)<5 and x!=0 else f"{x:.2f}")
    
    st.dataframe(
        df_comp.set_index("Metric").style.applymap(
            lambda x: "background-color: #f0f2f6; font-weight: bold", 
            subset=(["Retorno do Período"], slice(None))
        ), 
        use_container_width=True
    )

with col_delta:
    st.markdown("##### Performance Delta")
    # Agora destaca o Retorno do Período (o real)
    d_ret = m_sim["Retorno do Período"] - m_orig["Retorno do Período"]
    d_beta = m_sim["Beta"] - m_orig["Beta"]
    
    st.metric("Total Period Return", f"{m_sim['Retorno do Período']:.2%}", delta=f"{d_ret:.2%}")
    st.metric("Annualized Return", f"{m_sim['Retorno Anualizado']:.2%}", help="Taxa anual projetada (CAGR).")
    st.metric("Portfolio Beta", f"{m_sim['Beta']:.2f}", delta=f"{d_beta:.2f}", delta_color="inverse")

# --- BLOCO B: STRESS TEST ---
with st.expander("Stress Test Scenarios (Historical)", expanded=False):
    scenario = st.radio("Select Scenario:", ["COVID-19 Crash (2020)", "Hawkish Cycle (2021-2022)"], horizontal=True)
    
    if scenario == "COVID-19 Crash (2020)":
        s_start, s_end = "2020-01-20", "2020-03-30"
        period_start, period_end = "2020-01-23", "2020-03-23"
    else:
        s_start, s_end = "2021-06-01", "2022-07-25"
        period_start, period_end = "2021-06-08", "2022-07-18"

    # Download dedicado para Stress Test (com auto_adjust=True)
    try:
        df_bench_stress = yf.download(bench_ticker, start=s_start, end=s_end, progress=False, auto_adjust=True, threads=False)
        if not df_bench_stress.empty:
            if isinstance(df_bench_stress.columns, pd.MultiIndex):
                if 'Close' in df_bench_stress.columns: df_bench_stress = df_bench_stress['Close']
                elif 'Close' in df_bench_stress.columns.get_level_values(0): df_bench_stress = df_bench_stress.xs('Close', axis=1, level=0)
                else: df_bench_stress = df_bench_stress.iloc[:, 0]
            elif 'Close' in df_bench_stress.columns:
                 df_bench_stress = df_bench_stress['Close']
            else:
                 df_bench_stress = df_bench_stress.iloc[:, 0]
            if isinstance(df_bench_stress, pd.DataFrame): df_bench_stress = df_bench_stress.iloc[:, 0] 
            df_bench_stress.index = df_bench_stress.index.tz_localize(None)
    except: df_bench_stress = pd.Series()

    df_assets_stress = pd.DataFrame()
    if tickers_input:
        try:
            raw_assets = yf.download(tickers_input, start=s_start, end=s_end, progress=False, auto_adjust=True, threads=False)
            if not raw_assets.empty:
                if isinstance(raw_assets.columns, pd.MultiIndex):
                    if 'Close' in raw_assets.columns: df_assets_stress = raw_assets['Close']
                    elif 'Close' in raw_assets.columns.get_level_values(0): df_assets_stress = raw_assets.xs('Close', axis=1, level=0)
                    elif 'Close' in raw_assets.columns.get_level_values(1): df_assets_stress = raw_assets.xs('Close', axis=1, level=1)
                    else: df_assets_stress = raw_assets.iloc[:, 0]
                elif 'Close' in raw_assets.columns:
                    df_assets_stress = raw_assets[['Close']]
                    if len(tickers_input) == 1: df_assets_stress.columns = tickers_input
                else: df_assets_stress = raw_assets
                
                if isinstance(df_assets_stress, pd.Series): df_assets_stress = df_assets_stress.to_frame(name=tickers_input[0])
                df_assets_stress.index = df_assets_stress.index.tz_localize(None)
        except: pass

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
            c2.metric("Current Portfolio", f"{s_orig:.2%}", delta=f"{s_orig-bench_res:.2%} vs Bench")
            c3.metric("Simulated Portfolio", f"{s_sim:.2%}", delta=f"{s_sim-s_orig:.2%} vs Current")
            if used_proxy: st.caption(f"⚠️ Proxy used for: {', '.join(used_proxy)}")
        else: st.warning("Benchmark data empty for this range.")
    else: st.warning("Insufficient Benchmark data.")

# --- BLOCO C: ABAS GRÁFICAS ---
st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Risk vs Return", "Volatility Quality", "Capture Ratios", "Correlation Matrix", "History", "Portfolio Solver"])

with tab1:
    risk_mode = st.radio("Risk Metric (X-Axis):", ["Total Volatility", "Downside Deviation"], horizontal=True)
    x_key = "Vol" if risk_mode == "Total Volatility" else "SemiDev"
    scatter_data = []
    # Usa Retorno ANUALIZADO para o gráfico scatter ser comparável com Vol Anualizada
    for t, s in asset_stats.items(): scatter_data.append({"Label": t, "X": s[x_key], "Y": s["Ret"], "Type": "Asset", "Size": 8})
    scatter_data.append({"Label": "CURRENT", "X": m_orig["Volatilidade" if x_key=="Vol" else "Semi-Desvio"], "Y": m_orig["Retorno Anualizado"], "Type": "Current Portfolio", "Size": 20})
    scatter_data.append({"Label": "SIMULATED", "X": m_sim["Volatilidade" if x_key=="Vol" else "Semi-Desvio"], "Y": m_sim["Retorno Anualizado"], "Type": "Simulated Portfolio", "Size": 20})
    scatter_data.append({"Label": "BENCHMARK", "X": m_bench.get("Volatilidade" if x_key=="Vol" else "Semi-Desvio",0), "Y": m_bench.get("Retorno Anualizado",0), "Type": "Benchmark", "Size": 12})
    
    fig1 = px.scatter(pd.DataFrame(scatter_data), x="X", y="Y", color="Type", size="Size", text="Label", 
                      color_discrete_map={"Asset": "#636EFA", "Current Portfolio": "#00CC96", "Simulated Portfolio": "#FFD700", "Benchmark": "#7F7F7F"})
    fig1.update_layout(xaxis_title=risk_mode, yaxis_title="Annualized Return")
    fig1.update_traces(textposition='top center')
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    q_data = [{"Label": t, "Vol": s["Vol"], "SemiDev": s["SemiDev"]} for t, s in asset_stats.items()]
    if q_data:
        df_q = pd.DataFrame(q_data)
        max_v = df_q["Vol"].max() * 1.1 if not df_q.empty else 1
        fig2 = px.scatter(df_q, x="Vol", y="SemiDev", text="Label", height=500)
        fig2.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v, line=dict(color="darkred", width=1, dash="dash"))
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    up_o, down_o = calculate_capture_ratios(ret_orig, bench_ret)
    up_s, down_s = calculate_capture_ratios(ret_sim, bench_ret)
    c_data = [{"Label": t, "Up": s["UpCapture"], "Down": s["DownCapture"], "Type": "Asset"} for t, s in asset_stats.items()]
    c_data.append({"Label": "CURRENT", "Up": up_o, "Down": down_o, "Type": "Portfolio"})
    c_data.append({"Label": "SIMULATED", "Up": up_s, "Down": down_s, "Type": "Portfolio"})
    df_c = pd.DataFrame(c_data)
    
    fig3 = px.scatter(df_c, x="Down", y="Up", text="Label", color="Type", color_discrete_map={"Asset": "#636EFA", "Portfolio": "#00CC96"})
    fig3.add_vline(x=100, line_dash="dash", line_color="gray"); fig3.add_hline(y=100, line_dash="dash", line_color="gray")
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.plotly_chart(px.imshow(assets_ret.corr(), text_auto=".2f", aspect="auto", color_continuous_scale="RdYlGn", zmin=-1, zmax=1), use_container_width=True)

with tab5:
    cum_orig, cum_sim, cum_bench = (1 + ret_orig).cumprod(), (1 + ret_sim).cumprod(), (1 + bench_ret).cumprod()
    st.line_chart(pd.DataFrame({"Current (Fixed)": cum_orig, f"Simulated ({rebal_freq_sim})": cum_sim, "Benchmark": cum_bench}))
    dd_orig = (cum_orig - cum_orig.cummax()) / cum_orig.cummax()
    st.area_chart(dd_orig)

with tab6:
    st.markdown("### Portfolio Optimization")
    rf_daily = (1 + rf_input/100)**(1/252) - 1
    cash_series = pd.Series(rf_daily, index=assets_ret.index, name="CASH")
    df_opt = pd.concat([assets_ret, cash_series], axis=1)
    opt_assets = df_opt.columns.tolist()
    
    col_setup, col_res = st.columns([1, 2])
    with col_setup:
        target_obj = st.selectbox("Objective Function:", ["Max Sortino", "Min Downside Volatility", "Max Return (Target Semi-Dev)"])
        target_semidev_input = None
        if target_obj == "Max Return (Target Semi-Dev)":
            target_semidev_input = st.number_input("Target Downside Volatility (%)", value=5.0, step=0.5, min_value=0.1)
        
        default_min = [0.0] * len(opt_assets)
        default_max = [100.0] * len(opt_assets)
        df_constraints = pd.DataFrame({"Asset": opt_assets, "Min %": default_min, "Max %": default_max})
        edited_df = st.data_editor(df_constraints, column_config={"Min %": st.column_config.NumberColumn(min_value=0, max_value=100), "Max %": st.column_config.NumberColumn(min_value=0, max_value=100)}, hide_index=True)
        
        if st.button("Run Solver", type="primary"):
            bounds = [(r["Min %"]/100.0, r["Max %"]/100.0) for i, r in edited_df.iterrows()]
            with st.spinner("Optimizing..."):
                res = run_solver(df_opt, rf_input, tuple(bounds), target_obj, mgmt_fee, target_semidev_input)
            st.session_state['solver_result'] = {'success': res.success, 'message': res.message, 'weights': res.x, 'opt_assets': opt_assets}

    with col_res:
        if 'solver_result' in st.session_state and st.session_state['solver_result']['success']:
            res_data = st.session_state['solver_result']
            opt_weights = res_data['weights']
            opt_assets_saved = res_data['opt_assets']
            fee_daily = (1 + mgmt_fee/100)**(1/252) - 1
            opt_ret_series = df_opt.fillna(0).dot(opt_weights) - fee_daily
            m_opt = calculate_metrics(opt_ret_series, rf_input, bench_ret)
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Optimized Annual Return", f"{m_opt['Retorno Anualizado']:.2%}")
            k2.metric("Sortino Ratio", f"{m_opt['Sortino']:.2f}")
            k3.metric("Downside Vol", f"{m_opt['Semi-Desvio']:.2%}")
            
            df_w = pd.DataFrame({"Asset": opt_assets_saved, "Weight": opt_weights * 100}).query("Weight > 0.01")
            c_chart, c_table = st.columns([1, 1])
            with c_chart: st.plotly_chart(px.pie(df_w, values="Weight", names="Asset", title="Allocation"), use_container_width=True)
            with c_table: st.dataframe(df_w.sort_values("Weight", ascending=False).style.format({"Weight": "{:.2f}%"}), use_container_width=True, hide_index=True)
            
            def update_weights_callback():
                saved_res = st.session_state['solver_result']
                for i, asset_name in enumerate(saved_res['opt_assets']):
                    if asset_name != "CASH": st.session_state[f"sim_{asset_name}"] = round(saved_res['weights'][i] * 100.0, 2)
            st.button("Apply to Simulation", on_click=update_weights_callback)
        elif 'solver_result' in st.session_state:
            st.error(f"Failed: {st.session_state['solver_result']['message']}")
        else:
            st.info("Adjust constraints and click 'Run Solver'.")
