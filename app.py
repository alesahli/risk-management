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
from fpdf import FPDF

# Configuração para exportação de gráficos
pio.kaleido.scope.default_format = "png"

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Portfolio Risk Management System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CLASSE PARA O RELATÓRIO PDF ---
class RiskReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Portfolio Risk Management - Executive Report', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

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
    ann_return = (1 + total_return)**(252 / days) - 1 if days > 10 else total_return
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
    r_asset, r_bench = aligned.iloc[:, 0], aligned.iloc[:, 1]
    up_mask, down_mask = r_bench > 0, r_bench < 0
    up_cap = r_asset[up_mask].mean() / r_bench[up_mask].mean() if up_mask.sum() > 0 else 0
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
            rebal_dates = set(temp_resample[1::2]) if rebal_freq == 'Semestral' else set(temp_resample)
        except:
            rebal_dates = set(asset_returns.resample('Q').last().index)

    current_weights, current_cash_w = initial_weights.copy(), w_cash_initial
    portfolio_rets = []
    returns_arr, dates = asset_returns.fillna(0.0).values, asset_returns.index
    for i in range(len(dates)):
        day_ret = np.sum(current_weights * returns_arr[i]) + (current_cash_w * rf_daily)
        net_day_ret = day_ret - fee_daily
        portfolio_rets.append(net_day_ret)
        denominator = 1 + day_ret
        if denominator != 0:
            current_weights = current_weights * (1 + returns_arr[i]) / denominator
            current_cash_w = current_cash_w * (1 + rf_daily) / denominator
        if dates[i] in rebal_dates:
            current_weights, current_cash_w = initial_weights.copy(), w_cash_initial
    return pd.Series(portfolio_rets, index=dates)

def run_solver(df_returns, rf_annual, bounds, target_metric, mgmt_fee_annual=0.0, target_semidev_val=None):
    rf_daily = (1 + rf_annual/100)**(1/252) - 1
    fee_daily = (1 + mgmt_fee_annual/100)**(1/252) - 1
    num_assets = len(df_returns.columns)
    lb = np.array([b[0] for b in bounds]); ub = np.array([b[1] for b in bounds])
    initial_guess = (lb + ub) / 2
    if np.sum(initial_guess) > 0: initial_guess /= np.sum(initial_guess)
    else: initial_guess = np.array([1/num_assets] * num_assets)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    if target_metric == "Max Return (Target Semi-Dev)" and target_semidev_val is not None:
        def semidev_constraint(weights):
            net = df_returns.fillna(0).dot(weights) - fee_daily
            neg = net[net < 0]
            current_semi = neg.std() * np.sqrt(252) if len(neg) > 1 else 0
            return (target_semidev_val/100.0) - current_semi
        constraints.append({'type': 'ineq', 'fun': semidev_constraint})

    def objective(weights):
        net_ret = df_returns.fillna(0).dot(weights) - fee_daily
        if target_metric == "Max Sortino":
            neg = net_ret[net_ret < 0]
            if neg.empty or neg.std() == 0: return 1e5
            sortino = ((net_ret - rf_daily).mean() / neg.std()) * np.sqrt(252)
            return -sortino
        elif target_metric == "Min Downside Volatility":
            neg = net_ret[net_ret < 0]
            return neg.std() * np.sqrt(252) if not neg.empty else 0
        elif target_metric == "Max Return (Target Semi-Dev)":
            total_ret = (1 + net_ret).prod() - 1
            return -((1 + total_ret)**(252 / len(net_ret)) - 1)
            
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-6)
    return result

def load_portfolio_from_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            try: df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8-sig')
            except: df = pd.read_csv(uploaded_file)
        else: df = pd.read_excel(uploaded_file)
        
        df.columns = [str(c).lower().strip() for c in df.columns]
        col_ticker = next((c for c in df.columns if c in ['ativo', 'ticker', 'asset', 'symbol']), None)
        col_weight = next((c for c in df.columns if c in ['peso', 'weight', 'alocacao', '%']), None)
        
        if not col_ticker or not col_weight: return None, "Colunas não encontradas."
        portfolio = {}
        for _, row in df.iterrows():
            t = str(row[col_ticker]).strip().upper()
            try: w = float(str(row[col_weight]).replace(',', '.'))
            except: w = 0.0
            if w > 0: portfolio[t] = w
        
        total_w = sum(portfolio.values())
        if total_w <= 1.05 and total_w > 0:
            for k in portfolio: portfolio[k] *= 100.0
        return portfolio, None
    except Exception as e: return None, str(e)

# ==============================================================================
# 3. BARRA LATERAL (INPUTS)
# ==============================================================================
st.sidebar.header("Portfolio Configuration")
with st.sidebar.expander("📂 Import / Export Portfolio", expanded=True):
    df_template = pd.DataFrame({"Ativo": ["PETR4.SA", "VALE3.SA"], "Peso": [50.0, 50.0]})
    csv_temp = df_template.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
    st.download_button("Download Template (CSV)", csv_temp, "template.csv", "text/csv", use_container_width=True)
    uploaded_file = st.file_uploader("Upload Portfolio", type=['csv', 'xlsx'])
    if uploaded_file:
        p_dict, err = load_portfolio_from_file(uploaded_file)
        if p_dict:
            st.session_state['imported_portfolio'] = p_dict
            st.session_state['tickers_text_key'] = ", ".join(p_dict.keys())

default_tickers = st.session_state.get('tickers_text_key', "VALE3.SA, PETR4.SA, BPAC11.SA")
tickers_text = st.sidebar.text_area("Asset Tickers:", value=default_tickers, height=100)
tickers_input = [t.strip().upper() for t in tickers_text.split(',') if t.strip()]

periodo_option = st.sidebar.radio("Time Horizon:", ["1 Ano", "2 Anos", "Desde 2020", "Personalizado"], horizontal=True)
end_date = datetime.today()
if periodo_option == "1 Ano": start_date = end_date - timedelta(days=365)
elif periodo_option == "2 Anos": start_date = end_date - timedelta(days=730)
elif periodo_option == "Desde 2020": start_date = datetime(2020, 1, 1)
else:
    c1, c2 = st.sidebar.columns(2)
    start_date = c1.date_input("Start Date", value=datetime(2024,1,1))
    end_date = c2.date_input("End Date", value=datetime.today())

st.sidebar.subheader("Market & Costs")
rf_input = st.sidebar.number_input("Risk Free (% p.a.)", value=10.5)
mgmt_fee = st.sidebar.number_input("Mgmt Fee (% p.a.)", value=0.0)
bench_ticker = st.sidebar.text_input("Benchmark", value="^BVSP")
rebal_freq_sim = st.sidebar.selectbox("Rebalancing Frequency:", ["Sem Rebalanceamento", "Mensal", "Trimestral", "Semestral", "Anual", "Diário"], index=0, key='rebal_freq_key')

st.sidebar.subheader("Allocation")
weights_orig, weights_sim = {}, {}
imported_data = st.session_state.get('imported_portfolio', {})
if tickers_input:
    total_o, total_s = 0, 0
    def_val = 100.0 / len(tickers_input)
    for t in tickers_input:
        c1, c2, c3 = st.sidebar.columns([2, 1.5, 1.5])
        c1.text(t)
        v_def = imported_data.get(t, def_val)
        w_o = c2.number_input(f"o_{t}", 0.0, 100.0, float(v_def), label_visibility="collapsed")
        w_s = c3.number_input(f"s_{t}", 0.0, 100.0, key=f"sim_{t}", value=float(v_def), label_visibility="collapsed")
        weights_orig[t], weights_sim[t] = w_o, w_s
        total_o += w_o; total_s += w_s
    cash_orig, cash_sim = 100 - total_o, 100 - total_s
    st.sidebar.info(f"Cash: {cash_orig:.0f}% | {cash_sim:.0f}%")

# ==============================================================================
# 4. PROCESSAMENTO E CÁLCULOS
# ==============================================================================
all_tickers = list(set(tickers_input + [bench_ticker]))
with st.spinner("Fetching data..."):
    df_prices = get_market_data(all_tickers, start_date, end_date)
    if df_prices.empty: st.stop()
    df_ret = df_prices.ffill().pct_change().dropna()
    bench_ret = df_ret[bench_ticker] if bench_ticker in df_ret.columns else pd.Series(0, index=df_ret.index)
    valid_assets = [t for t in tickers_input if t in df_ret.columns]
    assets_ret = df_ret[valid_assets]

ret_orig = calculate_flexible_portfolio(assets_ret, weights_orig, cash_orig, rf_input, mgmt_fee, "Diário")
ret_sim = calculate_flexible_portfolio(assets_ret, weights_sim, cash_sim, rf_input, mgmt_fee, rebal_freq_sim)

# ==============================================================================
# 5. DASHBOARD
# ==============================================================================
st.title("Portfolio Risk Management System")

m_orig, m_sim, m_bench = calculate_metrics(ret_orig, rf_input, bench_ret), calculate_metrics(ret_sim, rf_input, bench_ret), calculate_metrics(bench_ret, rf_input, bench_ret)

metrics_order = ["Retorno do Período", "Retorno Anualizado", "Volatilidade", "Semi-Desvio", "Beta", "Sharpe", "Sortino", "Max Drawdown", "VaR 95%", "CVaR 95%"]
df_comp = pd.DataFrame({
    "Metric": metrics_order,
    "Current": [m_orig.get(k, 0) for k in metrics_order],
    "Simulated": [m_sim.get(k, 0) for k in metrics_order],
    "Benchmark": [m_bench.get(k, 0) for k in metrics_order]
})

df_comp_fmt = df_comp.copy()
for c in df_comp_fmt.columns[1:]:
    df_comp_fmt[c] = df_comp_fmt[c].apply(lambda x: f"{x:.2%}" if abs(x)<5 and x!=0 else f"{x:.2f}")

st.dataframe(df_comp_fmt.set_index("Metric"), use_container_width=True)

# --- GRÁFICOS PARA O PDF ---
cum_df = pd.DataFrame({"Current": (1+ret_orig).cumprod(), "Simulated": (1+ret_sim).cumprod(), "Benchmark": (1+bench_ret).cumprod()})
fig_hist = px.line(cum_df, title="Cumulative Performance History")

# Stress Test
with st.expander("Stress Test Scenarios"):
    st.write("Stress tests analyze historical crashes...")
    # (Sua lógica de Stress Test original aqui)

# Abas
tab1, tab2, tab3 = st.tabs(["Performance", "Correlation", "Portfolio Solver"])
with tab1:
    st.plotly_chart(fig_hist, use_container_width=True)
with tab2:
    st.plotly_chart(px.imshow(assets_ret.corr(), text_auto=".2f", color_continuous_scale="RdYlGn"), use_container_width=True)
with tab3:
    st.subheader("Optimizer")
    target_obj = st.selectbox("Objective:", ["Max Sortino", "Min Downside Volatility", "Max Return (Target Semi-Dev)"])
    if st.button("Run Optimizer"):
        bounds = [(0, 1.0)] * len(valid_assets)
        res = run_solver(assets_ret, rf_input, bounds, target_obj, mgmt_fee)
        if res.success:
            st.success("Optimization Successful!")
            weights_opt = res.x
            df_w = pd.DataFrame({"Asset": valid_assets, "Weight": weights_opt * 100})
            st.dataframe(df_w)
            st.session_state['opt_weights'] = weights_opt

# ==============================================================================
# 6. FUNCIONALIDADE DE RELATÓRIO PDF
# ==============================================================================
st.sidebar.markdown("---")
if st.sidebar.button("📊 Gerar Relatório PDF", use_container_width=True, type="primary"):
    with st.spinner("Gerando PDF profissional..."):
        try:
            pdf = RiskReport()
            pdf.add_page()
            
            # Tabela de Métricas
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 10, '1. Performance Metrics Summary', 0, 1)
            pdf.set_font('Arial', '', 8)
            col_w = pdf.epw / 4
            for col in df_comp_fmt.columns: pdf.cell(col_w, 8, col, 1)
            pdf.ln()
            for _, row in df_comp_fmt.iterrows():
                for val in row: pdf.cell(col_w, 7, str(val), 1)
                pdf.ln()
            
            # Gráfico
            pdf.ln(10)
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 10, '2. Historical Performance', 0, 1)
            img_bytes = pio.to_image(fig_hist, format='png', width=800, height=400)
            pdf.image(io.BytesIO(img_bytes), x=10, w=190)
            
            output = pdf.output()
            st.sidebar.download_button("📩 Download PDF", output, "Report_Risk.pdf", "application/pdf", use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao gerar PDF: {e}")
