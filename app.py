import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.optimize import minimize
import io

# ##############################################################################
# 1. CONFIGURAÇÃO DA PÁGINA
# ##############################################################################
st.set_page_config(
    page_title="Portfolio Risk Management System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ##############################################################################
# 2. FUNÇÕES CORE (BACKEND)
# ##############################################################################

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
        "Volatilidade": ann_vol, "Semi-Desvio": semi_dev, "Upside-Desvio": upside_dev,
        "Beta": beta, "Sharpe": sharpe, "Sortino": sortino, "Max Drawdown": max_dd,
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
            resample_code = {'Mensal':'ME', 'Trimestral':'QE', 'Anual':'YE', 'Semestral':'QE'}.get(rebal_freq, 'QE')
            temp_resample = asset_returns.resample(resample_code).last().index
            rebal_dates = set(temp_resample[1::2] if rebal_freq == 'Semestral' else temp_resample)
        except:
            rebal_dates = set(asset_returns.resample('M' if rebal_freq=='Mensal' else 'Q').last().index)

    current_weights, current_cash_w = initial_weights.copy(), w_cash_initial
    portfolio_rets = []
    returns_arr, dates = asset_returns.fillna(0.0).values, asset_returns.index

    for i in range(len(dates)):
        r_assets = returns_arr[i]
        day_ret = np.sum(current_weights * r_assets) + (current_cash_w * rf_daily)
        net_day_ret = day_ret - fee_daily
        portfolio_rets.append(net_day_ret)
        denom = 1 + day_ret
        if denom != 0:
            current_weights = current_weights * (1 + r_assets) / denom
            current_cash_w = current_cash_w * (1 + rf_daily) / denom
        if dates[i] in rebal_dates:
            current_weights, current_cash_w = initial_weights.copy(), w_cash_initial
    return pd.Series(portfolio_rets, index=dates)

def run_solver(df_returns, rf_annual, bounds, target_metric, mgmt_fee_annual=0.0, target_semidev_val=None):
    rf_daily = (1 + rf_annual/100)**(1/252) - 1
    fee_daily = (1 + mgmt_fee_annual/100)**(1/252) - 1
    num_assets = len(df_returns.columns)
    
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    initial_guess = (lower_bounds + upper_bounds) / 2
    if np.sum(initial_guess) > 0: initial_guess /= np.sum(initial_guess)
    else: initial_guess = np.array([1/num_assets] * num_assets)

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    if target_metric == "Max Return (Target Semi-Dev)" and target_semidev_val is not None:
        def semidev_constraint(weights):
            net = df_returns.fillna(0).dot(weights) - fee_daily
            neg = net[net < 0]
            return (target_semidev_val/100.0) - (neg.std() * np.sqrt(252))
        constraints.append({'type': 'ineq', 'fun': semidev_constraint})

    def objective(weights):
        net_ret = df_returns.fillna(0).dot(weights) - fee_daily
        if target_metric == "Max Sortino":
            neg = net_ret[net_ret < 0]
            if neg.empty or neg.std() == 0: return 1e5
            return -((net_ret.mean() - rf_daily) / neg.std() * np.sqrt(252))
        elif target_metric == "Min Downside Volatility":
            neg = net_ret[net_ret < 0]
            return neg.std() * np.sqrt(252) if not neg.empty else 0
        elif target_metric == "Max Return (Target Semi-Dev)":
            return -((1 + net_ret).prod()**(252/len(net_ret)) - 1)
            
    return minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-6)

def load_portfolio_from_file(uploaded_file):
    try:
        df = pd.DataFrame()
        if uploaded_file.name.endswith('.csv'):
            try: df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8-sig')
            except: 
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=',', decimal='.')
        else: df = pd.read_excel(uploaded_file)
        
        df.columns = [str(c).lower().strip() for c in df.columns]
        col_ticker = next((c for c in df.columns if c in ['ativo', 'ticker', 'asset', 'symbol', 'código']), None)
        col_weight = next((c for c in df.columns if c in ['peso', 'weight', 'alocacao', '%', 'valor']), None)
        if not col_ticker or not col_weight: return None, "Colunas não encontradas."
        
        portfolio = {str(row[col_ticker]).strip().upper(): float(str(row[col_weight]).replace(',', '.')) for _, row in df.iterrows()}
        if sum(portfolio.values()) <= 1.05: portfolio = {k: v*100 for k,v in portfolio.items()}
        return portfolio, None
    except Exception as e: return None, str(e)

def generate_html_report(df_comp_fmt, figs):
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    html = f"""
    <html><head><meta charset="UTF-8"><link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>body{{background:#f8f9fa;padding:40px}} .card{{border:none;border-radius:15px;box-shadow:0 4px 15px rgba(0,0,0,0.05);margin-bottom:30px;padding:25px;background:white}}
    .header{{background:#1e3c72;color:white;padding:40px;border-radius:15px;margin-bottom:30px;text-align:center}}</style></head>
    <body><div class="container"><div class="header"><h1>Executive Risk Report</h1><p>Relatório Consolidado de Gestão - {now}</p></div>
    <div class="card"><h3>Performance Metrics</h3>{df_comp_fmt.to_html(classes='table table-hover', index=False)}</div>"""
    for title, fig in figs.items():
        if fig: html += f'<div class="card"><h3>{title}</h3>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'
    return html + "</div></body></html>"

# ##############################################################################
# 3. BARRA LATERAL (INPUTS)
# ##############################################################################
st.sidebar.header("Portfolio Configuration")
with st.sidebar.expander("📂 Import / Export Portfolio", expanded=True):
    df_template = pd.DataFrame({"Ativo": ["PETR4.SA", "VALE3.SA"], "Peso": [50.0, 50.0]})
    csv_template = df_template.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
    st.download_button("Download Template (CSV)", csv_template, "template.csv", "text/csv", use_container_width=True)
    
    uploaded_file = st.file_uploader("Upload Portfolio (CSV/XLSX)", type=['csv', 'xlsx'])
    if uploaded_file:
        p_dict, err = load_portfolio_from_file(uploaded_file)
        if p_dict: 
            st.session_state['imported_portfolio'] = p_dict
            st.session_state['tickers_text_key'] = ", ".join(p_dict.keys())
        else: st.error(err)

tickers_text = st.sidebar.text_area("Asset Tickers:", value=st.session_state.get('tickers_text_key', "VALE3.SA, PETR4.SA, BPAC11.SA"), height=100)
tickers_input = [t.strip().upper() for t in tickers_text.split(',') if t.strip()]

periodo_opt = st.sidebar.radio("Time Horizon:", ["1 Ano", "2 Anos", "Desde 2020", "Personalizado"], horizontal=True)
end_date = datetime.today()
if periodo_opt == "1 Ano": start_date = end_date - timedelta(days=365)
elif periodo_opt == "2 Anos": start_date = end_date - timedelta(days=730)
elif periodo_opt == "Desde 2020": start_date = datetime(2020, 1, 1)
else:
    c_s, c_e = st.sidebar.columns(2)
    start_date, end_date = c_s.date_input("Start", datetime(2024,1,1)), c_e.date_input("End", datetime.today())

st.sidebar.subheader("Market & Costs")
c_rf, c_fee = st.sidebar.columns(2)
rf_input = c_rf.number_input("Risk Free %", 10.5)
mgmt_fee = c_fee.number_input("Mgmt Fee %", 0.0)
bench_ticker = st.sidebar.text_input("Benchmark", "^BVSP")
rebal_freq_sim = st.sidebar.selectbox("Frequency:", ["Sem Rebalanceamento", "Mensal", "Trimestral", "Semestral", "Anual", "Diário"])

st.sidebar.markdown("---")
st.sidebar.subheader("Allocation")
weights_orig, weights_sim = {}, {}
imported_data = st.session_state.get('imported_portfolio', {})
if tickers_input:
    total_o, total_s = 0, 0
    def_w = 100.0 / len(tickers_input)
    st.sidebar.columns([2, 1.5, 1.5])[0].markdown("Ticker")
    for t in tickers_input:
        c1, c2, c3 = st.sidebar.columns([2, 1.5, 1.5])
        c1.text(t)
        v_def = imported_data.get(t, def_w)
        w_o = c2.number_input(f"o_{t}", 0.0, 100.0, float(v_def), label_visibility="collapsed")
        w_s = c3.number_input(f"s_{t}", 0.0, 100.0, float(v_def), key=f"sim_{t}", label_visibility="collapsed")
        weights_orig[t], weights_sim[t] = w_o, w_s
        total_o += w_o; total_s += w_s
    st.sidebar.info(f"Cash Position: Current {100-total_o:.0f}% | Simulated {100-total_s:.0f}%")

# ##############################################################################
# 4. PROCESSAMENTO
# ##############################################################################
all_t = list(set(tickers_input + [bench_ticker]))
df_prices = get_market_data(all_t, start_date, end_date)
if df_prices.empty: st.error("No data found."); st.stop()

df_ret = df_prices.ffill().pct_change().iloc[1:]
b_ret = df_ret[bench_ticker] if bench_ticker in df_ret.columns else pd.Series(0, index=df_ret.index)
a_ret = df_ret[[t for t in tickers_input if t in df_ret.columns]]

ret_orig = calculate_flexible_portfolio(a_ret, weights_orig, 100-total_o, rf_input, mgmt_fee, "Diário")
ret_sim = calculate_flexible_portfolio(a_ret, weights_sim, 100-total_s, rf_input, mgmt_fee, rebal_freq_sim)

asset_stats = {t: {**calculate_metrics(a_ret[t], rf_input, b_ret), "Up": calculate_capture_ratios(a_ret[t], b_ret)[0], "Down": calculate_capture_ratios(a_ret[t], b_ret)[1]} for t in a_ret.columns}

# ##############################################################################
# 5. DASHBOARD
# ##############################################################################
st.title("Portfolio Risk Management System")
m_o, m_s, m_b = calculate_metrics(ret_orig, rf_input, b_ret), calculate_metrics(ret_sim, rf_input, b_ret), calculate_metrics(b_ret, rf_input, b_ret)

col_kpi, col_delta = st.columns([3, 1])
with col_kpi:
    m_order = ["Retorno do Período", "Retorno Anualizado", "Volatilidade", "Semi-Desvio", "Beta", "Sharpe", "Sortino", "Max Drawdown"]
    df_comp = pd.DataFrame({"Metric": m_order})
    for col, data in [("Current", m_o), ("Simulated", m_s), ("Benchmark", m_b)]:
        df_comp[col] = df_comp["Metric"].apply(lambda x: f"{data.get(x,0):.2f}" if x in ["Beta", "Sharpe", "Sortino"] else f"{data.get(x,0):.2%}")
    st.dataframe(df_comp.set_index("Metric"), use_container_width=True)

with col_delta:
    st.metric("Total Period Return", f"{m_s.get('Retorno do Período', 0):.2%}", delta=f"{m_s.get('Retorno do Período',0)-m_o.get('Retorno do Período',0):.2%}")
    st.metric("Portfolio Beta", f"{m_s.get('Beta', 0):.2f}", delta=f"{m_s.get('Beta',0)-m_o.get('Beta',0):.2f}", delta_color="inverse")

with st.expander("Stress Test Scenarios (Historical)", expanded=False):
    scen = st.radio("Scenario:", ["COVID-19 Crash (2020)", "Hawkish Cycle (2021-2022)"], horizontal=True)
    s_s, s_e, p_s, p_e = ("2020-01-20", "2020-03-30", "2020-01-23", "2020-03-23") if "COVID" in scen else ("2021-06-01", "2022-07-25", "2021-06-08", "2022-07-18")
    df_st_b = get_market_data([bench_ticker], s_s, s_e)
    df_st_a = get_market_data(tickers_input, s_s, s_e)
    if not df_st_b.empty:
        b_move = (df_st_b.iloc[-1,0] / df_st_b.iloc[0,0]) - 1
        p_o = sum([weights_orig[t]/100 * ((df_st_a[t].iloc[-1]/df_st_a[t].iloc[0])-1 if t in df_st_a else asset_stats[t]['Beta']*b_move) for t in tickers_input])
        p_s = sum([weights_sim[t]/100 * ((df_st_a[t].iloc[-1]/df_st_a[t].iloc[0])-1 if t in df_st_a else asset_stats[t]['Beta']*b_move) for t in tickers_input])
        c1, c2, c3 = st.columns(3)
        c1.metric("Bench Move", f"{b_move:.2%}"); c2.metric("Current Port", f"{p_o:.2%}"); c3.metric("Simulated Port", f"{p_s:.2%}")

st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Risk/Return", "Vol Quality", "Capture", "Correlation", "History", "Solver"])

with tab1:
    fig1 = px.scatter(pd.DataFrame([{"L": t, "X": s["Volatilidade"], "Y": s["Retorno Anualizado"], "T": "Asset"} for t, s in asset_stats.items()] + [{"L": "SIM", "X": m_s["Volatilidade"], "Y": m_s["Retorno Anualizado"], "T": "Port"}]), x="X", y="Y", text="L", color="T", title="Risk vs Return")
    st.plotly_chart(fig1, use_container_width=True)
with tab2:
    v_data = [{"Asset": t, "Up": s["Upside-Desvio"], "Down": s["Semi-Desvio"], "Ratio": s["Upside-Desvio"]/s["Semi-Desvio"] if s["Semi-Desvio"]>0 else 0} for t, s in asset_stats.items()]
    st.dataframe(pd.DataFrame(v_data).sort_values("Ratio", ascending=False), use_container_width=True)
    fig2 = px.scatter(pd.DataFrame(v_data), x="Down", y="Up", text="Asset", title="Volatility Quality")
    st.plotly_chart(fig2, use_container_width=True)
with tab3:
    fig3 = px.scatter(pd.DataFrame([{"L": t, "Up": s["Up"], "Down": s["Down"]} for t, s in asset_stats.items()]), x="Down", y="Up", text="L", title="Market Capture")
    fig3.add_vline(x=100, line_dash="dash"); fig3.add_hline(y=100, line_dash="dash")
    st.plotly_chart(fig3, use_container_width=True)
with tab4:
    fig4 = px.imshow(a_ret.corr(), text_auto=".2f", color_continuous_scale="RdYlGn")
    st.plotly_chart(fig4, use_container_width=True)
with tab5:
    fig5 = px.line(pd.DataFrame({"Sim": (1+ret_sim).cumprod(), "Bench": (1+b_ret).cumprod()}), title="Cumulative Returns")
    st.plotly_chart(fig5, use_container_width=True)
with tab6:
    st.subheader("Optimization")
    obj = st.selectbox("Objective:", ["Max Sortino", "Min Downside Volatility", "Max Return (Target Semi-Dev)"])
    t_sd = st.number_input("Target Semi-Dev %", 5.0) if "Target" in obj else None
    edit_df = st.data_editor(pd.DataFrame({"Asset": tickers_input+["CASH"], "Min %": 0.0, "Max %": 100.0}), hide_index=True)
    if st.button("Run Solver"):
        df_opt = pd.concat([a_ret, pd.Series((1+rf_input/100)**(1/252)-1, index=a_ret.index, name="CASH")], axis=1)
        res = run_solver(df_opt, rf_input, tuple([(r["Min %"]/100, r["Max %"]/100) for _, r in edit_df.iterrows()]), obj, mgmt_fee, t_sd)
        if res.success: st.session_state['opt_w'] = pd.DataFrame({"Asset": df_opt.columns, "W": res.x*100}).query("W>0.01")
    if 'opt_w' in st.session_state: st.plotly_chart(px.pie(st.session_state['opt_w'], values="W", names="Asset"))

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Reporting")
if st.sidebar.button("Generate Executive Report (HTML)", use_container_width=True, type="primary"):
    html_rep = generate_html_report(df_comp, {"Risk Analysis": fig1, "Vol Quality": fig2, "Performance": fig5})
    st.sidebar.download_button("📥 Baixar Relatório", html_rep, f"Relatorio_{datetime.now().strftime('%Y%m%d')}.html", "text/html", use_container_width=True)
