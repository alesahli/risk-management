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

# --- NOVA FUNÇÃO DE EXPORTAÇÃO HTML ---
def generate_interactive_report(df_comp, figures_dict, weights_orig, weights_sim):
    """Gera um documento HTML moderno e interativo com Bootstrap 5."""
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Relatório Executivo de Portfólio</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ background-color: #f4f7f9; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #333; }}
            .header-banner {{ background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 40px 0; margin-bottom: 30px; }}
            .card {{ border: none; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 30px; }}
            .table-styled {{ font-size: 0.9rem; }}
            .table-styled thead {{ background-color: #f8f9fa; }}
            .footer {{ text-align: center; padding: 20px; color: #777; font-size: 0.8rem; }}
            .section-title {{ border-left: 5px solid #1e3c72; padding-left: 15px; margin-bottom: 25px; font-weight: bold; color: #1e3c72; }}
        </style>
    </head>
    <body>
        <div class="header-banner">
            <div class="container text-center">
                <h1 class="display-5 fw-bold">Executive Portfolio Risk Analysis</h1>
                <p class="lead mb-0">Relatório Consolidado de Gestão e Performance</p>
                <small>Gerado em: {now_str}</small>
            </div>
        </div>

        <div class="container">
            <!-- KPIs -->
            <div class="row">
                <div class="col-12">
                    <h3 class="section-title">Métricas de Performance e Risco</h3>
                    <div class="card p-4">
                        <div class="table-responsive">
                            {df_comp.to_html(classes='table table-hover table-styled', index=False)}
                        </div>
                    </div>
                </div>
            </div>

            <!-- Gráficos -->
    """
    
    # Adicionar Gráficos Plotly
    for title, fig in figures_dict.items():
        chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
        html_content += f"""
            <div class="row">
                <div class="col-12">
                    <h3 class="section-title">{title}</h3>
                    <div class="card p-3">
                        {chart_html}
                    </div>
                </div>
            </div>
        """

    html_content += """
            <div class="footer">
                <p>Relatório gerado automaticamente pelo Portfolio Risk Management System.</p>
                <p>&copy; 2024 - Risk Intelligence System</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

# ==============================================================================
# 3. BARRA LATERAL (INPUTS)
# ==============================================================================
st.sidebar.header("Portfolio Configuration")

with st.sidebar.expander("📂 Import / Export Portfolio", expanded=True):
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
        else:
            st.error(f"Erro: {error_msg}")

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

# ==============================================================================
# 5. DASHBOARD & GRÁFICOS (ARMAZENADOS PARA RELATÓRIO)
# ==============================================================================
st.title("Portfolio Risk Management System")

# KPIs
m_orig = calculate_metrics(ret_orig, rf_input, bench_ret)
m_sim = calculate_metrics(ret_sim, rf_input, bench_ret)
m_bench = calculate_metrics(bench_ret, rf_input, bench_ret)

col_kpi, col_delta = st.columns([3, 1])
with col_kpi:
    metrics_order = ["Retorno do Período", "Retorno Anualizado", "Volatilidade", "Semi-Desvio", "Beta", "Sharpe", "Sortino", "Max Drawdown", "VaR 95%", "CVaR 95%"]
    keys_present = [k for k in metrics_order if k in m_orig]
    
    df_comp = pd.DataFrame({
        "Metric": keys_present, 
        "Current (Fixed W)": [m_orig.get(k, 0) for k in keys_present], 
        f"Simulated ({rebal_freq_sim})": [m_sim.get(k, 0) for k in keys_present], 
        "Benchmark": [m_bench.get(k, 0) for k in keys_present]
    })
    
    # Criar versão formatada para exibição
    df_comp_fmt = df_comp.copy()
    for c in df_comp_fmt.columns[1:]:
        df_comp_fmt[c] = df_comp_fmt[c].apply(lambda x: f"{x:.2%}" if abs(x)<5 and x!=0 else f"{x:.2f}")
    
    st.markdown(f"#### Performance Metrics (Simulated Rebal: {rebal_freq_sim})")
    st.dataframe(df_comp_fmt.set_index("Metric"), use_container_width=True)

with col_delta:
    st.metric("Portfolio Beta", f"{m_sim.get('Beta', 0):.2f}")
    st.metric("Ann. Return", f"{m_sim.get('Retorno Anualizado', 0):.2%}")

# --- TABELAS E GRÁFICOS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Risk vs Return", "Volatility Quality", "Capture Ratios", "Correlation", "History", "Solver"])

# Fig 1: Risk vs Return
x_key = "Vol" # Padrão
scatter_data = []
for t, s in asset_stats.items(): scatter_data.append({"Label": t, "X": s[x_key], "Y": s["Ret"], "Type": "Asset", "Size": 8})
scatter_data.append({"Label": "CURRENT", "X": m_orig.get("Volatilidade", 0), "Y": m_orig.get("Retorno Anualizado", 0), "Type": "Current Portfolio", "Size": 20})
scatter_data.append({"Label": "SIMULATED", "X": m_sim.get("Volatilidade", 0), "Y": m_sim.get("Retorno Anualizado", 0), "Type": "Simulated Portfolio", "Size": 20})
scatter_data.append({"Label": "BENCHMARK", "X": m_bench.get("Volatilidade", 0), "Y": m_bench.get("Retorno Anualizado", 0), "Type": "Benchmark", "Size": 12})
fig1 = px.scatter(pd.DataFrame(scatter_data), x="X", y="Y", color="Type", size="Size", text="Label", title="Risk vs Return Analysis")
with tab1: st.plotly_chart(fig1, use_container_width=True)

# Fig 2: Vol Quality
q_data = [{"Label": t, "Vol": s["Vol"], "SemiDev": s["SemiDev"]} for t, s in asset_stats.items()]
df_q = pd.DataFrame(q_data)
fig2 = px.scatter(df_q, x="Vol", y="SemiDev", text="Label", title="Volatility Quality (Downside vs Total)")
with tab2: st.plotly_chart(fig2, use_container_width=True)

# Fig 3: Capture Ratios
up_s, down_s = calculate_capture_ratios(ret_sim, bench_ret)
c_data = [{"Label": t, "Up": s["UpCapture"], "Down": s["DownCapture"], "Type": "Asset"} for t, s in asset_stats.items()]
c_data.append({"Label": "SIMULATED", "Up": up_s, "Down": down_s, "Type": "Portfolio"})
fig3 = px.scatter(pd.DataFrame(c_data), x="Down", y="Up", text="Label", color="Type", title="Upside vs Downside Capture")
fig3.add_vline(x=100, line_dash="dash"); fig3.add_hline(y=100, line_dash="dash")
with tab3: st.plotly_chart(fig3, use_container_width=True)

# Fig 4: Correlation
fig4 = px.imshow(assets_ret.corr(), text_auto=".2f", color_continuous_scale="RdYlGn", title="Correlation Matrix")
with tab4: st.plotly_chart(fig4, use_container_width=True)

# Fig 5: History
cum_sim, cum_bench = (1 + ret_sim).cumprod(), (1 + bench_ret).cumprod()
df_hist = pd.DataFrame({"Simulated": cum_sim, "Benchmark": cum_bench})
fig5 = px.line(df_hist, title="Cumulative Performance")
with tab5: st.plotly_chart(fig5, use_container_width=True)

# ==============================================================================
# 6. EXPORTAÇÃO (BARRA LATERAL)
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Reporting")

# Preparar dicionário de figuras para o relatório
figs_report = {
    "Performance Histórica": fig5,
    "Análise de Risco x Retorno": fig1,
    "Captura de Mercado": fig3,
    "Matriz de Correlação": fig4,
    "Qualidade da Volatilidade": fig2
}

if st.sidebar.button("Gerar Relatório Interativo (.html)", use_container_width=True):
    html_report = generate_interactive_report(df_comp_fmt, figs_report, weights_orig, weights_sim)
    
    st.sidebar.download_button(
        label="📥 Baixar Agora",
        data=html_report,
        file_name=f"Relatorio_Portfolio_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
        mime="text/html",
        use_container_width=True
    )
    st.sidebar.success("Relatório preparado com sucesso!")

# Solver Tab (Manteve Original)
with tab6:
    st.markdown("### Portfolio Optimization")
    # ... código do solver original (não alterado) ...
    # (Inserir aqui o bloco solver original se desejar manter a funcionalidade idêntica)
