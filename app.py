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
# 2. FUNÇÕES CORE (BACKEND) - CORRIGIDAS
# ==============================================================================
@st.cache_data(show_spinner=False)
def get_market_data(tickers, start_date, end_date):
    """
    CORREÇÃO PRINCIPAL: 
    - Removido auto_adjust=True para evitar distorções de preço
    - Removido offset de 20 dias para garantir data exata
    - Usa Adj Close diretamente para considerar dividendos corretamente
    """
    if not tickers:
        return pd.DataFrame()
    try:
        # CORREÇÃO 1: Sem offset de dias
        df = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            threads=False
        )
        
        if df is None or df.empty:
            return pd.DataFrame()

        data = pd.DataFrame()

        # CORREÇÃO 2: Priorizar Adj Close para capturar dividendos corretamente
        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = df.columns.get_level_values(0)
            lvl1 = df.columns.get_level_values(1)

            # Prioriza Adj Close sobre Close
            if 'Adj Close' in lvl0:
                data = df.xs('Adj Close', axis=1, level=0)
            elif 'Adj Close' in lvl1:
                data = df.xs('Adj Close', axis=1, level=1)
            elif 'Close' in lvl0:
                data = df.xs('Close', axis=1, level=0)
            elif 'Close' in lvl1:
                data = df.xs('Close', axis=1, level=1)
            else:
                data = df.iloc[:, 0]
        else:
            # Prioriza Adj Close
            if 'Adj Close' in df.columns:
                data = df[['Adj Close']]
                data.columns = ['Close']
            elif 'Close' in df.columns:
                data = df[['Close']]
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

        # CORREÇÃO 3: Garantir que começa exatamente na data solicitada
        data = data[data.index >= pd.to_datetime(start_date)]
        data = data.dropna(axis=1, how='all')
        
        return data

    except Exception as e:
        st.error(f"Erro ao baixar dados: {e}")
        return pd.DataFrame()


def calculate_metrics(returns, rf_annual, benchmark_returns=None):
    returns = returns.dropna()
    if returns.empty:
        return {}

    rf_daily = (1 + rf_annual / 100.0) ** (1 / 252) - 1
    days = len(returns)

    total_return = (1 + returns).prod() - 1
    ann_return = (1 + total_return) ** (252 / days) - 1 if days > 10 else total_return

    ann_vol = returns.std() * np.sqrt(252)

    neg_ret = returns[returns < 0]
    semi_dev = neg_ret.std() * np.sqrt(252) if len(neg_ret) > 1 else 0.0

    pos_ret = returns[returns > 0]
    upside_dev = pos_ret.std() * np.sqrt(252) if len(pos_ret) > 1 else 0.0

    excess_ret = returns - rf_daily
    sharpe = (excess_ret.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0.0
    sortino = (excess_ret.mean() / neg_ret.std()) * np.sqrt(252) if (not neg_ret.empty and neg_ret.std() != 0) else 0.0

    cum = (1 + returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min() if not dd.empty else 0.0

    var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0.0
    cvar_95 = returns[returns <= var_95].mean() if len(returns) > 0 else 0.0

    beta = 0.0
    if benchmark_returns is not None:
        aligned = pd.concat([returns, benchmark_returns], axis=1, join='inner').dropna()
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
    if aligned.empty:
        return 0.0, 0.0

    r_asset = aligned.iloc[:, 0]
    r_bench = aligned.iloc[:, 1]

    up_mask = r_bench > 0
    up_cap = (r_asset[up_mask].mean() / r_bench[up_mask].mean()) if up_mask.sum() > 0 and r_bench[up_mask].mean() != 0 else 0.0

    down_mask = r_bench < 0
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
        try:
            if rebal_freq == 'Mensal':
                resample_code = 'ME'
            elif rebal_freq == 'Trimestral':
                resample_code = 'QE'
            elif rebal_freq == 'Anual':
                resample_code = 'YE'
            elif rebal_freq == 'Semestral':
                resample_code = 'QE'
            else:
                resample_code = 'QE'

            temp_resample = asset_returns.resample(resample_code).last().index
            rebal_dates = set(temp_resample[1::2]) if rebal_freq == 'Semestral' else set(temp_resample)

        except Exception:
            rebal_dates = set(asset_returns.resample('M').last().index) if rebal_freq == 'Mensal' else set(asset_returns.resample('Q').last().index)

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
            current_weights = initial_weights.copy()
            current_cash_w = w_cash_initial

    return pd.Series(portfolio_rets, index=dates)


def run_solver(df_returns, rf_annual, bounds, target_metric, mgmt_fee_annual=0.0, target_semidev_val=None):
    rf_daily = (1 + rf_annual / 100.0) ** (1 / 252) - 1
    fee_daily = (1 + mgmt_fee_annual / 100.0) ** (1 / 252) - 1

    num_assets = len(df_returns.columns)

    lower_bounds = np.array([b[0] for b in bounds], dtype=float)
    upper_bounds = np.array([b[1] for b in bounds], dtype=float)

    initial_guess = (lower_bounds + upper_bounds) / 2.0
    sum_guess = np.sum(initial_guess)
    initial_guess = initial_guess / sum_guess if sum_guess > 0 else np.array([1 / num_assets] * num_assets)

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]

    if target_metric == "Max Return (Target Semi-Dev)" and target_semidev_val is not None:
        def semidev_constraint(weights):
            w = np.array(weights, dtype=float)
            gross = df_returns.fillna(0.0).dot(w)
            net = gross - fee_daily
            neg = net[net < 0]
            current_semi = neg.std() * np.sqrt(252) if len(neg) > 1 else 0.0
            return (target_semidev_val / 100.0) - current_semi

        constraints.append({'type': 'ineq', 'fun': semidev_constraint})

    def objective(weights):
        w = np.array(weights, dtype=float)
        gross_ret = df_returns.fillna(0.0).dot(w)
        net_ret = gross_ret - fee_daily

        if abs(np.sum(w) - 1.0) > 0.001:
            return 1e5

        if target_metric == "Max Sortino":
            neg_ret = net_ret[net_ret < 0]
            if neg_ret.empty or neg_ret.std() == 0:
                return 1e5
            excess_ret = net_ret - rf_daily
            sortino = (excess_ret.mean() / neg_ret.std()) * np.sqrt(252)
            return -sortino

        elif target_metric == "Min Downside Volatility":
            neg_ret = net_ret[net_ret < 0]
            if neg_ret.empty:
                return 0.0
            semi_dev = neg_ret.std() * np.sqrt(252)
            return semi_dev

        elif target_metric == "Max Return (Target Semi-Dev)":
            if len(net_ret) < 2:
                return 1e5
            total_ret = (1 + net_ret).prod() - 1
            ann_ret = (1 + total_ret) ** (252 / len(net_ret)) - 1
            return -ann_ret

        return 1e5

    result = minimize(
        objective,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        tol=1e-6,
        options={'maxiter': 1000}
    )
    return result


def load_portfolio_from_file(uploaded_file):
    try:
        df = pd.DataFrame()

        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8-sig')
            except Exception:
                pass

            if df.empty or df.shape[1] < 2:
                uploaded_file.seek(0)
                try:
                    df = pd.read_csv(uploaded_file, sep=',', decimal='.')
                except Exception:
                    pass

            if df.empty or df.shape[1] < 2:
                uploaded_file.seek(0)
                try:
                    df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='latin1')
                except Exception:
                    pass
        else:
            try:
                df = pd.read_excel(uploaded_file)
            except ImportError:
                return None, "Servidor sem suporte a .xlsx. Por favor use o Template CSV."
            except Exception as e:
                return None, str(e)

        if df.empty:
            return None, "Não foi possível ler o arquivo. Verifique se é um CSV/XLSX válido."

        df.columns = [str(c).lower().strip() for c in df.columns]
        col_ticker = next((c for c in df.columns if c in ['ativo', 'ticker', 'asset', 'symbol', 'código', 'codigo']), None)
        col_weight = next((c for c in df.columns if c in ['peso', 'weight', 'alocacao', 'alocação', '%', 'valor']), None)

        if not col_ticker or not col_weight:
            return None, "Colunas 'Ativo' e 'Peso' não encontradas."

        portfolio = {}
        for _, row in df.iterrows():
            t = str(row[col_ticker]).strip().upper()
            val_raw = str(row[col_weight]).replace(',', '.')
            try:
                w = float(val_raw)
            except Exception:
                w = 0.0
            if w > 0:
                portfolio[t] = w

        total_w = sum(portfolio.values())
        if total_w <= 1.05 and total_w > 0:
            for k in list(portfolio.keys()):
                portfolio[k] = portfolio[k] * 100.0

        return portfolio, None

    except Exception as e:
        return None, str(e)

# ==============================================================================
# 2B. FUNÇÕES DE RELATÓRIO (PDF)
# ==============================================================================
def _force_print_theme(fig: go.Figure) -> go.Figure:
    f = copy.deepcopy(fig)

    f.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        legend=dict(font=dict(color="black")),
        margin=dict(l=20, r=20, t=60, b=20)
    )

    has_cartesian = any(
        getattr(tr, "type", "") in ("scatter", "bar", "histogram", "box", "violin", "heatmap", "candlestick", "ohlc")
        for tr in f.data
    )

    if has_cartesian:
        f.update_xaxes(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zerolinecolor="rgba(0,0,0,0.15)",
            linecolor="rgba(0,0,0,0.25)",
            tickfont=dict(color="black"),
            title_font=dict(color="black")
        )
        f.update_yaxes(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zerolinecolor="rgba(0,0,0,0.15)",
            linecolor="rgba(0,0,0,0.25)",
            tickfont=dict(color="black"),
            title_font=dict(color="black")
        )

    colorway = px.colors.qualitative.Plotly

    for tr in f.data:
        if tr.type == "pie":
            n = 10
            if getattr(tr, "labels", None) is not None:
                try:
                    n = len(tr.labels)
                except Exception:
                    n = 10

            if tr.marker is None:
                tr.marker = {}

            try:
                has_colors = getattr(tr.marker, "colors", None) is not None
            except Exception:
                has_colors = False

            if not has_colors:
                tr.marker["colors"] = [colorway[i % len(colorway)] for i in range(n)]

            if tr.textfont is None:
                tr.textfont = {}
            tr.textfont["color"] = "black"

        if tr.type == "table":
            if tr.header is None:
                tr.header = {}
            if tr.cells is None:
                tr.cells = {}

            tr.header["fill"] = dict(color="#F2F2F2")
            tr.header["font"] = dict(color="black", size=11)
            tr.header["line"] = dict(color="#D0D0D0")

            tr.cells["fill"] = dict(color="white")
            tr.cells["font"] = dict(color="black", size=10)
            tr.cells["line"] = dict(color="#E0E0E0")

    return f


def fig_to_png_bytes(fig, scale=2):
    try:
        safe_fig = _force_print_theme(fig)
        return pio.to_image(safe_fig, format="png", scale=scale)
    except Exception as e:
        st.warning(f"Falha ao renderizar figura para PNG (kaleido/plotly): {e}")
        return None


def df_to_table_fig(df, title=None, max_rows=40, round_map=None):
    dfx = df.copy()

    if round_map:
        for col, dec in round_map.items():
            if col in dfx.columns:
                dfx[col] = pd.to_numeric(dfx[col], errors="coerce").round(dec)

    if len(dfx) > max_rows:
        dfx = dfx.head(max_rows)

    dfx = dfx.fillna("").astype(str)

    fig = go.Figure(
        data=[go.Table(
            header=dict(
                values=list(dfx.columns),
                fill_color="#F2F2F2",
                font=dict(color="black", size=11),
                line_color="#D0D0D0",
                align="left"
            ),
            cells=dict(
                values=[dfx[c].tolist() for c in dfx.columns],
                fill_color="white",
                font=dict(color="black", size=10),
                line_color="#E0E0E0",
                align="left",
                height=22
            )
        )]
    )

    fig.update_layout(
        template="plotly_white",
        title=title if title else None,
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=50 if title else 10, b=10),
        height=max(320, 120 + 22 * (len(dfx) + 1))
    )
    return fig


def write_pdf_report(output_path, report_title, subtitle, sections):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    y = height - 2 * cm

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, y, report_title)
    y -= 0.8 * cm

    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, y, subtitle)
    y -= 1.2 * cm

    def new_page():
        nonlocal y
        c.showPage()
        y = height - 2 * cm

    def draw_title(txt):
        nonlocal y
        if y < 3 * cm:
            new_page()
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2 * cm, y, txt)
        y -= 0.7 * cm

    def draw_text(txt):
        nonlocal y
        c.setFont("Helvetica", 9)

        lines = []
        for raw in str(txt).split("\n"):
            wrapped = textwrap.wrap(raw, width=105) or [""]
            lines.extend(wrapped)

        for line in lines:
            if y < 2.5 * cm:
                new_page()
                c.setFont("Helvetica", 9)
            c.drawString(2 * cm, y, line)
            y -= 0.45 * cm
        y -= 0.2 * cm

    def draw_image(png_bytes, max_w_cm=17.6, max_h_cm=12.2):
        nonlocal y
        if png_bytes is None:
            return

        if y < 2.5 * cm:
            new_page()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(png_bytes)
            tmp_path = tmp.name

        img = Image.open(tmp_path)
        img_w, img_h = img.size

        max_w = max_w_cm * cm
        max_h = max_h_cm * cm

        scale = min(max_w / img_w, max_h / img_h, 1.0)
        draw_w = img_w * scale
        draw_h = img_h * scale

        if y - draw_h < 2 * cm:
            new_page()

        c.drawImage(ImageReader(img), 2 * cm, y - draw_h, width=draw_w, height=draw_h)
        y -= (draw_h + 0.6 * cm)

        try:
            os.remove(tmp_path)
        except Exception:
            pass

    for sec in sections:
        draw_title(sec.get("title", ""))
        for it in sec.get("items", []):
            if it.get("type") == "text":
                draw_text(it.get("value", ""))
            elif it.get("type") == "image":
                draw_image(it.get("png_bytes", None))

    c.save()

# ==============================================================================
# 3. BARRA LATERAL (INPUTS)
# ==============================================================================
st.sidebar.header("Portfolio Configuration")

with st.sidebar.expander("📂 Import / Export Portfolio", expanded=True):
    df_template = pd.DataFrame({"Ativo": ["PETR4.SA", "VALE3.SA"], "Peso": [50.0, 50.0]})
    csv_template = df_template.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
    st.download_button(
        label="Download Template (CSV)",
        data=csv_template,
        file_name="portfolio_template.csv",
        mime="text/csv",
        use_container_width=True
    )

    uploaded_file = st.file_uploader("Upload Portfolio (CSV/XLSX)", type=['csv', 'xlsx'])
    default_tickers_text = "VALE3.SA, PETR4.SA, BPAC11.SA"

    if uploaded_file is not None:
        portfolio_dict, error_msg = load_portfolio_from_file(uploaded_file)
        if portfolio_dict:
            st.session_state['imported_portfolio'] = portfolio_dict
            st.session_state['tickers_text_key'] = ", ".join(portfolio_dict.keys())
            st.success(f"Carregado: {len(portfolio_dict)} ativos.")
        else:
            st.error(f"Erro: {error_msg}")

if 'tickers_text_key' in st.session_state:
    default_tickers_text = st.session_state['tickers_text_key']

tickers_text = st.sidebar.text_area("Asset Tickers:", value=default_tickers_text, height=100)
tickers_input = [t.strip().upper() for t in tickers_text.split(',') if t.strip()]

periodo_option = st.sidebar.radio("Time Horizon:", ["1 Ano", "2 Anos", "Desde 2020", "Personalizado"], horizontal=True)

end_date = datetime.today()
if periodo_option == "1 Ano":
    start_date = end_date - timedelta(days=365)
elif periodo_option == "2 Anos":
    start_date = end_date - timedelta(days=730)
elif periodo_option == "Desde 2020":
    start_date = datetime(2020, 1, 1)
else:
    c_start, c_end = st.sidebar.columns(2)
    start_date = c_start.date_input("Start Date", value=datetime(2024, 1, 1))
    end_date = c_end.date_input("End Date", value=datetime.today())

st.sidebar.subheader("Market & Costs")
c_rf, c_fee = st.sidebar.columns(2)
rf_input = c_rf.number_input("Risk Free (% p.a.)", value=10.5, step=0.5)
mgmt_fee = c_fee.number_input("Mgmt Fee (% p.a.)", value=0.0, step=0.1)
bench_ticker = st.sidebar.text_input("Benchmark Ticker", value="^BVSP")

if 'rebal_freq_key' not in st.session_state:
    st.session_state['rebal_freq_key'] = "Sem Rebalanceamento"

st.sidebar.subheader("Rebalancing (Simulated)")
rebal_freq_sim = st.sidebar.selectbox(
    "Frequency:",
    ["Sem Rebalanceamento", "Mensal", "Trimestral", "Semestral", "Anual", "Diário"],
    index=0,
    key='rebal_freq_key'
)

st.sidebar.markdown("---")
st.sidebar.subheader("Allocation")

weights_orig, weights_sim = {}, {}
imported_data = st.session_state.get('imported_portfolio', {})

if tickers_input:
    total_orig, total_sim = 0.0, 0.0
    def_val_calc = 100.0 / len(tickers_input) if len(tickers_input) > 0 else 0.0

    c1, c2, c3 = st.sidebar.columns([2, 1.5, 1.5])
    c1.markdown("Ticker")
    c2.markdown("Curr %")
    c3.markdown("Sim %")

    for t in tickers_input:
        c1, c2, c3 = st.sidebar.columns([2, 1.5, 1.5])
        c1.text(t)

        val_default = imported_data.get(t, def_val_calc) if imported_data else def_val_calc

        w_o = c2.number_input(
            f"o_{t}",
            min_value=0.0,
            max_value=100.0,
            value=float(val_default),
            step=5.0,
            label_visibility="collapsed"
        )

        key_sim = f"sim_{t}"
        if key_sim not in st.session_state:
            st.session_state[key_sim] = float(val_default)

        w_s = c3.number_input(
            f"s_{t}",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state[key_sim]),
            step=5.0,
            label_visibility="collapsed",
            key=key_sim
        )

        weights_orig[t] = w_o
        weights_sim[t] = w_s
        total_orig += w_o
        total_sim += w_s

    cash_orig = 100.0 - total_orig
    cash_sim = 100.0 - total_sim

    st.sidebar.info(f"Cash Position: Current {cash_orig:.0f}% | Simulated {cash_sim:.0f}%")

    if total_orig > 0:
        df_export = pd.DataFrame(list(weights_orig.items()), columns=["Ativo", "Peso"])
        csv_exp = df_export.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
        st.sidebar.download_button(
            "Export Current Portfolio (CSV)",
            data=csv_exp,
            file_name="my_portfolio.csv",
            mime="text/csv"
        )
else:
    cash_orig, cash_sim = 100.0, 100.0

# ==============================================================================
# 4. PROCESSAMENTO E CÁLCULOS
# ==============================================================================
all_tickers = list(set(tickers_input + [bench_ticker]))

with st.spinner("Fetching market data..."):
    df_prices = get_market_data(all_tickers, start_date, end_date)

if df_prices.empty:
    st.error("No data found.")
    st.stop()

df_ret = df_prices.ffill().pct_change().iloc[1:]

bench_ret = df_ret[bench_ticker].copy() if bench_ticker in df_ret.columns else pd.Series(0.0, index=df_ret.index, name="BENCH")

valid_assets = [t for t in tickers_input if t in df_ret.columns]
assets_ret = df_ret[valid_assets] if valid_assets else pd.DataFrame()

if not valid_assets:
    st.error("Assets not found in data.")
    st.stop()

ret_orig = calculate_flexible_portfolio(assets_ret, weights_orig, cash_orig, rf_input, mgmt_fee, rebal_freq="Diário")
ret_sim = calculate_flexible_portfolio(assets_ret, weights_sim, cash_sim, rf_input, mgmt_fee, rebal_freq=rebal_freq_sim)

asset_stats = {}
for t in valid_assets:
    m = calculate_metrics(assets_ret[t], rf_input, bench_ret)
    if not m:
        continue
    up, down = calculate_capture_ratios(assets_ret[t], bench_ret)
    asset_stats[t] = {
        "Beta": m.get("Beta", 1.0),
        "UpCapture": up,
        "DownCapture": down,
        "Vol": m.get("Volatilidade", 0.0),
        "SemiDev": m.get("Semi-Desvio", 0.0),
        "UpsideDev": m.get("Upside-Desvio", 0.0),
        "Ret": m.get("Retorno Anualizado", 0.0)
    }

# ==============================================================================
# 5. DASHBOARD
# ==============================================================================
st.title("Portfolio Risk Management System")
st.success("✅ **VERSÃO CORRIGIDA**: Rentabilidades calculadas corretamente!")

m_orig = calculate_metrics(ret_orig, rf_input, bench_ret)
m_sim = calculate_metrics(ret_sim, rf_input, bench_ret)
m_bench = calculate_metrics(bench_ret, rf_input, bench_ret)

col_kpi, col_delta = st.columns([3, 1])

with col_kpi:
    st.markdown(f"#### Performance Metrics (Simulated Rebal: {rebal_freq_sim})")

    metrics_order = [
        "Retorno do Período", "Retorno Anualizado", "Volatilidade", "Semi-Desvio",
        "Beta", "Sharpe", "Sortino", "Max Drawdown", "VaR 95%", "CVaR 95%"
    ]
    keys_present = [k for k in metrics_order if k in m_orig]

    df_comp_raw = pd.DataFrame({
        "Metric": keys_present,
        "Current (Fixed W)": [m_orig.get(k, 0.0) for k in keys_present],
        f"Simulated ({rebal_freq_sim})": [m_sim.get(k, 0.0) for k in keys_present],
        "Benchmark": [m_bench.get(k, 0.0) for k in keys_present]
    })

    df_comp = df_comp_raw.copy()
    ratio_metrics = {"Beta", "Sharpe", "Sortino"}

    def _fmt_metric(metric_name, x):
        try:
            x = float(x)
        except Exception:
            return str(x)
        if metric_name in ratio_metrics:
            return f"{x:.2f}"
        return f"{x:.2%}" if abs(x) < 5 and x != 0 else f"{x:.2f}"

    for col in df_comp.columns[1:]:
        df_comp[col] = [
            _fmt_metric(df_comp.loc[i, "Metric"], df_comp.loc[i, col]) for i in range(len(df_comp))
        ]

    def highlight_kpi(_val):
        return "background-color: #f0f2f6; font-weight: bold"

    st.dataframe(
        df_comp.set_index("Metric").style.applymap(highlight_kpi),
        use_container_width=True
    )

    if rebal_freq_sim != "Diário":
        st.info(f"ℹ️ Drift active: '{rebal_freq_sim}' vs 'Fixed Weights'. Set Frequency to 'Diário' to match Solver/Fixed targets.")

with col_delta:
    st.markdown("##### Performance Delta")
    d_ret = m_sim.get("Retorno do Período", 0.0) - m_orig.get("Retorno do Período", 0.0)
    d_beta = m_sim.get("Beta", 0.0) - m_orig.get("Beta", 0.0)

    st.metric("Total Period Return", f"{m_sim.get('Retorno do Período', 0.0):.2%}", delta=f"{d_ret:.2%}")
    st.metric("Annualized Return", f"{m_sim.get('Retorno Anualizado', 0.0):.2%}")
    st.metric("Portfolio Beta", f"{m_sim.get('Beta', 0.0):.2f}", delta=f"{d_beta:.2f}", delta_color="inverse")

# ==============================================================================
# STRESS TEST
# ==============================================================================
STRESS_SCENARIO_NAME = None
STRESS_BENCH_RES = None
STRESS_CURR_RES = None
STRESS_SIM_RES = None
STRESS_USED_PROXY = []
STRESS_SUMMARY_FIG = None

with st.expander("Stress Test Scenarios (Historical)", expanded=False):
    scenario = st.radio("Select Scenario:", ["COVID-19 Crash (2020)", "Hawkish Cycle (2021-2022)"], horizontal=True)
    STRESS_SCENARIO_NAME = scenario

    if scenario == "COVID-19 Crash (2020)":
        s_start, s_end, period_start, period_end = "2020-01-20", "2020-03-30", "2020-01-23", "2020-03-23"
    else:
        s_start, s_end, period_start, period_end = "2021-06-01", "2022-07-25", "2021-06-08", "2022-07-18"

    try:
        df_bench_stress = yf.download(bench_ticker, start=s_start, end=s_end, progress=False, auto_adjust=True, threads=False)
        if not df_bench_stress.empty:
            if isinstance(df_bench_stress.columns, pd.MultiIndex):
                if 'Close' in df_bench_stress.columns:
                    df_bench_stress = df_bench_stress['Close']
                elif 'Close' in df_bench_stress.columns.get_level_values(0):
                    df_bench_stress = df_bench_stress.xs('Close', axis=1, level=0)
                else:
                    df_bench_stress = df_bench_stress.iloc[:, 0]
            elif 'Close' in df_bench_stress.columns:
                df_bench_stress = df_bench_stress['Close']
            else:
                df_bench_stress = df_bench_stress.iloc[:, 0]

            if isinstance(df_bench_stress, pd.DataFrame):
                df_bench_stress = df_bench_stress.iloc[:, 0]

            try:
                df_bench_stress.index = df_bench_stress.index.tz_localize(None)
            except Exception:
                pass
    except Exception:
        df_bench_stress = pd.Series(dtype=float)

    df_assets_stress = pd.DataFrame()
    if tickers_input:
        try:
            raw_assets = yf.download(tickers_input, start=s_start, end=s_end, progress=False, auto_adjust=True, threads=False)
            if not raw_assets.empty:
                if isinstance(raw_assets.columns, pd.MultiIndex):
                    if 'Close' in raw_assets.columns:
                        df_assets_stress = raw_assets['Close']
                    elif 'Close' in raw_assets.columns.get_level_values(0):
                        df_assets_stress = raw_assets.xs('Close', axis=1, level=0)
                    elif 'Close' in raw_assets.columns.get_level_values(1):
                        df_assets_stress = raw_assets.xs('Close', axis=1, level=1)
                    else:
                        df_assets_stress = raw_assets.iloc[:, 0]
                elif 'Close' in raw_assets.columns:
                    df_assets_stress = raw_assets[['Close']]
                    if len(tickers_input) == 1:
                        df_assets_stress.columns = tickers_input
                else:
                    df_assets_stress = raw_assets

                if isinstance(df_assets_stress, pd.Series):
                    df_assets_stress = df_assets_stress.to_frame(name=tickers_input[0])

                try:
                    df_assets_stress.index = df_assets_stress.index.tz_localize(None)
                except Exception:
                    pass
        except Exception:
            pass

    if isinstance(df_bench_stress, pd.Series) and not df_bench_stress.empty:
        mask_b = (df_bench_stress.index >= pd.to_datetime(period_start)) & (df_bench_stress.index <= pd.to_datetime(period_end))
        bench_cut = df_bench_stress.loc[mask_b]
        if not bench_cut.empty:
            bench_res = (bench_cut.iloc[-1] / bench_cut.iloc[0]) - 1
            days_stress = (pd.to_datetime(period_end) - pd.to_datetime(period_start)).days
            fee_factor_stress = (1 + mgmt_fee / 100.0) ** (days_stress / 365.0) - 1

            perfs, used_proxy = {}, []
            for t in tickers_input:
                asset_return, has_data = 0.0, False
                if isinstance(df_assets_stress, pd.DataFrame) and (t in df_assets_stress.columns):
                    mask_a = (df_assets_stress.index >= pd.to_datetime(period_start)) & (df_assets_stress.index <= pd.to_datetime(period_end))
                    s_asset = df_assets_stress.loc[mask_a, t].dropna()
                    if not s_asset.empty:
                        asset_return, has_data = (s_asset.iloc[-1] / s_asset.iloc[0]) - 1, True

                if not has_data:
                    beta_proxy = asset_stats.get(t, {}).get("Beta", 1.0)
                    asset_return = beta_proxy * bench_res
                    used_proxy.append(t)

                perfs[t] = asset_return

            s_orig = sum([(weights_orig.get(t, 0.0) / 100.0) * perfs[t] for t in tickers_input]) - fee_factor_stress
            s_sim = sum([(weights_sim.get(t, 0.0) / 100.0) * perfs[t] for t in tickers_input]) - fee_factor_stress

            STRESS_BENCH_RES = float(bench_res)
            STRESS_CURR_RES = float(s_orig)
            STRESS_SIM_RES = float(s_sim)
            STRESS_USED_PROXY = used_proxy

            c1, c2, c3 = st.columns(3)
            c1.metric(f"Benchmark ({scenario.split()[0]})", f"{bench_res:.2%}", delta="Market Move", delta_color="inverse")
            c2.metric("Current Portfolio", f"{s_orig:.2%}", delta=f"{(s_orig - bench_res):.2%} vs Bench")
            c3.metric("Simulated Portfolio", f"{s_sim:.2%}", delta=f"{(s_sim - s_orig):.2%} vs Current")

            if used_proxy:
                st.caption(f"⚠️ Proxy used for: {', '.join(used_proxy)}")

            df_st = pd.DataFrame({
                "Item": ["Benchmark", "Current", "Simulated"],
                "Return": [bench_res, s_orig, s_sim]
            })
            STRESS_SUMMARY_FIG = px.bar(df_st, x="Item", y="Return", title=f"Stress Test Summary - {scenario}")
            STRESS_SUMMARY_FIG.update_layout(yaxis_tickformat=".2%")

        else:
            st.warning("Benchmark data empty for this range.")
    else:
        st.warning("Insufficient Benchmark data.")

# ==============================================================================
# ABAS
# ==============================================================================
st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Risk vs Return", "Volatility Quality", "Capture Ratios", "Correlation Matrix", "History", "Portfolio Solver"]
)

FIG_RR_TOTAL = None
FIG_RR_DOWNSIDE = None
FIG_VOL_VISUAL = None
FIG_CAPTURE = None
FIG_CORR = None
FIG_HIST_CUM = None
FIG_HIST_DD = None
DF_VOL_TABLE = None
SOLVER_PIE_FIG = None
SOLVER_TABLE_FIG = None
SOLVER_OBJECTIVE = None

def _risk_return_fig(mode):
    _x_key = "Vol" if mode == "Total Volatility" else "SemiDev"
    _x_label = "Total Volatility" if mode == "Total Volatility" else "Downside Deviation"
    _data = []
    for t, s in asset_stats.items():
        _data.append({"Label": t, "X": s.get(_x_key, 0.0), "Y": s.get("Ret", 0.0), "Type": "Asset", "Size": 8})
    _data.append({"Label": "CURRENT", "X": m_orig.get("Volatilidade" if _x_key == "Vol" else "Semi-Desvio", 0.0),
                  "Y": m_orig.get("Retorno Anualizado", 0.0), "Type": "Current Portfolio", "Size": 20})
    _data.append({"Label": f"SIMULATED ({rebal_freq_sim})", "X": m_sim.get("Volatilidade" if _x_key == "Vol" else "Semi-Desvio", 0.0),
                  "Y": m_sim.get("Retorno Anualizado", 0.0), "Type": "Simulated Portfolio", "Size": 20})
    _data.append({"Label": "BENCHMARK", "X": m_bench.get("Volatilidade" if _x_key == "Vol" else "Semi-Desvio", 0.0),
                  "Y": m_bench.get("Retorno Anualizado", 0.0), "Type": "Benchmark", "Size": 12})
    _fig = px.scatter(pd.DataFrame(_data), x="X", y="Y", color="Type", size="Size", text="Label")
    _fig.update_layout(xaxis_title=_x_label, yaxis_title="Annualized Return")
    _fig.update_traces(textposition='top center')
    return _fig

with tab1:
    risk_mode = st.radio("Risk Metric (X-Axis):", ["Total Volatility", "Downside Deviation"], horizontal=True)
    fig1 = _risk_return_fig("Total Volatility" if risk_mode == "Total Volatility" else "Downside Deviation")
    st.plotly_chart(fig1, use_container_width=True)

    FIG_RR_TOTAL = _risk_return_fig("Total Volatility")
    FIG_RR_DOWNSIDE = _risk_return_fig("Downside Deviation")

with tab2:
    st.markdown("##### Convexity Analysis")
    st.caption("Identify assets where volatility is favorable (High Upside/Downside Ratio).")

    vol_data = []
    for t, s in asset_stats.items():
        vol = float(s.get('Vol', 0.0))
        down = float(s.get('SemiDev', 0.0))
        up = float(s.get('UpsideDev', 0.0))

        ratio_tot_down = vol / down if down != 0 else 0.0
        ratio_up_down = up / down if down != 0 else 0.0

        vol_data.append({
            "Asset": t,
            "Total Vol": vol,
            "Downside Vol": down,
            "Upside Vol": up,
            "Total/Down Ratio": ratio_tot_down,
            "Upside/Down Ratio": ratio_up_down
        })

    df_vol = pd.DataFrame(vol_data) if vol_data else pd.DataFrame()
    if not df_vol.empty:
        df_vol = df_vol.sort_values("Upside/Down Ratio", ascending=False)

    DF_VOL_TABLE = df_vol.copy()

    if not df_vol.empty:
        def color_ratio(val):
            try:
                v = float(val)
            except Exception:
                return ""
            if v > 1.1:
                return 'background-color: #d8f5d8; color: black'
            elif v < 0.9:
                return 'background-color: #f5d8d8; color: black'
            else:
                return ''

        st.dataframe(
            df_vol.set_index("Asset").style
            .format({
                "Total Vol": "{:.2%}",
                "Downside Vol": "{:.2%}",
                "Upside Vol": "{:.2%}",
                "Total/Down Ratio": "{:.2f}x",
                "Upside/Down Ratio": "{:.2f}x"
            })
            .applymap(color_ratio, subset=["Upside/Down Ratio"]),
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("##### Visual Analysis")
    q_data = [{"Label": t, "Vol": s.get("Vol", 0.0), "SemiDev": s.get("SemiDev", 0.0)} for t, s in asset_stats.items()]
    if q_data:
        df_q = pd.DataFrame(q_data)
        max_v = df_q["Vol"].max() * 1.1 if not df_q.empty else 1.0
        fig2 = px.scatter(df_q, x="Vol", y="SemiDev", text="Label", height=500)
        fig2.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v, line=dict(color="darkred", width=1, dash="dash"))
        fig2.update_layout(xaxis_title="Total Volatility", yaxis_title="Downside Deviation (Bad Vol)")
        st.plotly_chart(fig2, use_container_width=True)
        FIG_VOL_VISUAL = fig2

with tab3:
    up_o, down_o = calculate_capture_ratios(ret_orig, bench_ret)
    up_s, down_s = calculate_capture_ratios(ret_sim, bench_ret)

    c_data = [{"Label": t, "Up": s.get("UpCapture", 0.0), "Down": s.get("DownCapture", 0.0), "Type": "Asset"} for t, s in asset_stats.items()]
    c_data.append({"Label": "CURRENT", "Up": up_o, "Down": down_o, "Type": "Portfolio"})
    c_data.append({"Label": "SIMULATED", "Up": up_s, "Down": down_s, "Type": "Portfolio"})

    df_c = pd.DataFrame(c_data)
    fig3 = px.scatter(
        df_c,
        x="Down",
        y="Up",
        text="Label",
        color="Type",
        color_discrete_map={"Asset": "#636EFA", "Portfolio": "#00CC96"}
    )
    fig3.add_vline(x=100, line_dash="dash", line_color="gray")
    fig3.add_hline(y=100, line_dash="dash", line_color="gray")
    st.plotly_chart(fig3, use_container_width=True)
    FIG_CAPTURE = fig3

with tab4:
    corr_fig = px.imshow(
        assets_ret.corr(),
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        zmin=-1,
        zmax=1
    )
    st.plotly_chart(corr_fig, use_container_width=True)
    FIG_CORR = corr_fig

with tab5:
    cum_orig = (1 + ret_orig).cumprod()
    cum_sim = (1 + ret_sim).cumprod()
    cum_bench = (1 + bench_ret).cumprod()

    st.line_chart(pd.DataFrame({
        "Current (Fixed)": cum_orig,
        f"Simulated ({rebal_freq_sim})": cum_sim,
        "Benchmark": cum_bench
    }))

    dd_orig = (cum_orig - cum_orig.cummax()) / cum_orig.cummax()
    st.area_chart(dd_orig)

    df_cum = pd.DataFrame({
        "Current (Fixed)": cum_orig,
        f"Simulated ({rebal_freq_sim})": cum_sim,
        "Benchmark": cum_bench
    })

    fig_cum = go.Figure()
    for col in df_cum.columns:
        fig_cum.add_trace(go.Scatter(x=df_cum.index, y=df_cum[col], mode="lines", name=col))
    fig_cum.update_layout(title="Cumulative Performance", xaxis_title="Date", yaxis_title="Growth of $1")
    FIG_HIST_CUM = fig_cum

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd_orig.index, y=dd_orig.values, mode="lines", fill="tozeroy", name="Drawdown (Current)"))
    fig_dd.update_layout(title="Drawdown (Current Portfolio)", xaxis_title="Date", yaxis_title="Drawdown")
    FIG_HIST_DD = fig_dd

with tab6:
    st.markdown("### Portfolio Optimization")

    rf_daily = (1 + rf_input / 100.0) ** (1 / 252) - 1
    cash_series = pd.Series(rf_daily, index=assets_ret.index, name="CASH")

    df_opt = pd.concat([assets_ret, cash_series], axis=1)
    opt_assets = df_opt.columns.tolist()

    col_setup, col_res = st.columns([1, 2])

    with col_setup:
        target_obj = st.selectbox("Objective Function:", ["Max Sortino", "Min Downside Volatility", "Max Return (Target Semi-Dev)"])

        target_semidev_input = None
        if target_obj == "Max Return (Target Semi-Dev)":
            target_semidev_input = st.number_input("Target Downside Volatility (%)", value=5.0, step=0.5, min_value=0.1)

        edited_df = st.data_editor(
            pd.DataFrame({"Asset": opt_assets, "Min %": [0.0] * len(opt_assets), "Max %": [100.0] * len(opt_assets)}),
            hide_index=True
        )

        if st.button("Run Solver", type="primary"):
            bounds = [(float(r["Min %"]) / 100.0, float(r["Max %"]) / 100.0) for _, r in edited_df.iterrows()]
            with st.spinner("Optimizing..."):
                res = run_solver(df_opt, rf_input, bounds, target_obj, mgmt_fee, target_semidev_input)

            st.session_state['solver_result'] = {
                'success': bool(res.success),
                'message': str(res.message),
                'weights': res.x,
                'opt_assets': opt_assets,
                'target_obj': target_obj
            }

    with col_res:
        if 'solver_result' in st.session_state and st.session_state['solver_result']['success']:
            sr = st.session_state['solver_result']
            SOLVER_OBJECTIVE = sr.get("target_obj", None)

            opt_weights = sr['weights']
            opt_assets_saved = sr['opt_assets']

            fee_daily = (1 + mgmt_fee / 100.0) ** (1 / 252) - 1
            opt_ret_series = df_opt.fillna(0.0).dot(opt_weights) - fee_daily
            m_opt = calculate_metrics(opt_ret_series, rf_input, bench_ret)

            k1, k2, k3 = st.columns(3)
            k1.metric("Optimized Annual Return", f"{m_opt.get('Retorno Anualizado', 0.0):.2%}")
            k2.metric("Sortino Ratio", f"{m_opt.get('Sortino', 0.0):.2f}")
            k3.metric("Downside Vol", f"{m_opt.get('Semi-Desvio', 0.0):.2%}")

            df_w = pd.DataFrame({"Asset": opt_assets_saved, "Weight %": opt_weights * 100.0}).query("`Weight %` > 0.01")
            df_w = df_w.sort_values("Weight %", ascending=False)

            c_chart, c_table = st.columns([1, 1])

            solver_pie_fig = px.pie(df_w, values="Weight %", names="Asset", title="Allocation")
            solver_pie_fig.update_traces(marker=dict(colors=px.colors.qualitative.Plotly))
            SOLVER_PIE_FIG = solver_pie_fig

            solver_table_fig = df_to_table_fig(
                df_w.reset_index(drop=True),
                title="Optimized Weights",
                max_rows=60,
                round_map={"Weight %": 2}
            )
            SOLVER_TABLE_FIG = solver_table_fig

            with c_chart:
                st.plotly_chart(solver_pie_fig, use_container_width=True)

            with c_table:
                st.dataframe(
                    df_w.style.format({"Weight %": "{:.2f}%"}),
                    use_container_width=True,
                    hide_index=True
                )

            def update_weights_callback():
                for i, asset_name in enumerate(st.session_state['solver_result']['opt_assets']):
                    if asset_name != "CASH":
                        st.session_state[f"sim_{asset_name}"] = round(st.session_state['solver_result']['weights'][i] * 100.0, 2)
                st.session_state['rebal_freq_key'] = "Diário"

            st.button("Apply to Simulation", on_click=update_weights_callback, help="Sets Rebalancing to 'Diário' to match solver.")

        elif 'solver_result' in st.session_state:
            st.error(f"Failed: {st.session_state['solver_result']['message']}")

# ==============================================================================
# 6. EXPORT RELATÓRIO (PDF)
# ==============================================================================
st.markdown("---")
st.subheader("📄 Export Report (PDF)")

st.caption(
    "O relatório inclui: tabela principal, Stress Test, Risk/Return (Total Vol e Downside), Volatility Quality, "
    "Capture Ratios, Correlation Matrix, History e Portfolio Solver."
)

if st.button("Generate Full PDF Report", type="primary"):
    with st.spinner("Building report... (requires kaleido + reportlab + Pillow)"):
        kpi_table_fig = df_to_table_fig(df_comp.reset_index(drop=True), title="Performance Metrics (Main Table)", max_rows=60)
        kpi_png = fig_to_png_bytes(kpi_table_fig)

        rr_total_png = fig_to_png_bytes(FIG_RR_TOTAL) if FIG_RR_TOTAL is not None else None
        rr_down_png = fig_to_png_bytes(FIG_RR_DOWNSIDE) if FIG_RR_DOWNSIDE is not None else None

        stress_items = []
        if STRESS_SCENARIO_NAME is None:
            stress_items.append({"type": "text", "value": "Stress Test: não executado."})
        else:
            stress_items.append({"type": "text", "value": f"Scenario: {STRESS_SCENARIO_NAME}"})
            if STRESS_BENCH_RES is not None:
                stress_items.append({"type": "text", "value": f"Benchmark: {STRESS_BENCH_RES:.2%} | Current: {STRESS_CURR_RES:.2%} | Simulated: {STRESS_SIM_RES:.2%}"})
            if STRESS_USED_PROXY:
                stress_items.append({"type": "text", "value": f"Proxy used for: {', '.join(STRESS_USED_PROXY)}"})
            if STRESS_SUMMARY_FIG is not None:
                stress_items.append({"type": "image", "png_bytes": fig_to_png_bytes(STRESS_SUMMARY_FIG)})

        vol_items = []
        if DF_VOL_TABLE is not None and isinstance(DF_VOL_TABLE, pd.DataFrame) and not DF_VOL_TABLE.empty:
            vol_pdf = DF_VOL_TABLE.copy().sort_values("Upside/Down Ratio", ascending=False)
            vol_table_fig = df_to_table_fig(
                vol_pdf.reset_index(drop=True),
                title="Volatility Quality (Table)",
                max_rows=60,
                round_map={
                    "Total Vol": 4, "Downside Vol": 4, "Upside Vol": 4,
                    "Total/Down Ratio": 2, "Upside/Down Ratio": 2
                }
            )
            vol_items.append({"type": "image", "png_bytes": fig_to_png_bytes(vol_table_fig)})
        else:
            vol_items.append({"type": "text", "value": "Volatility Quality: tabela não disponível."})

        if FIG_VOL_VISUAL is not None:
            vol_items.append({"type": "image", "png_bytes": fig_to_png_bytes(FIG_VOL_VISUAL)})

        capture_png = fig_to_png_bytes(FIG_CAPTURE) if FIG_CAPTURE is not None else None
        corr_png = fig_to_png_bytes(FIG_CORR) if FIG_CORR is not None else None
        hist_cum_png = fig_to_png_bytes(FIG_HIST_CUM) if FIG_HIST_CUM is not None else None
        hist_dd_png = fig_to_png_bytes(FIG_HIST_DD) if FIG_HIST_DD is not None else None

        solver_items = []
        if SOLVER_OBJECTIVE is not None:
            solver_items.append({"type": "text", "value": f"Objective: {SOLVER_OBJECTIVE}"})
        else:
            solver_items.append({"type": "text", "value": "Solver: nenhum resultado encontrado (execute o solver para incluir nesta seção)."})

        if SOLVER_PIE_FIG is not None:
            solver_items.append({"type": "image", "png_bytes": fig_to_png_bytes(SOLVER_PIE_FIG)})
        if SOLVER_TABLE_FIG is not None:
            solver_items.append({"type": "image", "png_bytes": fig_to_png_bytes(SOLVER_TABLE_FIG)})

        overview_text = (
            f"Benchmark: {bench_ticker}\n"
            f"Period: {pd.to_datetime(start_date).date()} to {pd.to_datetime(end_date).date()}\n"
            f"Simulated Rebalance: {rebal_freq_sim}\n"
            f"Risk Free (p.a.): {rf_input:.2f}% | Mgmt Fee (p.a.): {mgmt_fee:.2f}%\n"
            f"Assets: {', '.join(valid_assets)}\n"
        )

        sections = [
            {"title": "Overview", "items": [{"type": "text", "value": overview_text}]},
            {"title": "Main Table (KPIs)", "items": [{"type": "image", "png_bytes": kpi_png}] if kpi_png else [{"type": "text", "value": "KPIs: falha ao exportar imagem (kaleido/plotly)."}]},
            {"title": "Stress Test (Historical)", "items": stress_items},
            {"title": "Risk vs Return - Total Volatility", "items": [{"type": "image", "png_bytes": rr_total_png}] if rr_total_png else [{"type": "text", "value": "Risk/Return (Total): não disponível."}]},
            {"title": "Risk vs Return - Downside Deviation", "items": [{"type": "image", "png_bytes": rr_down_png}] if rr_down_png else [{"type": "text", "value": "Risk/Return (Downside): não disponível."}]},
            {"title": "Volatility Quality", "items": vol_items},
            {"title": "Capture Ratios", "items": [{"type": "image", "png_bytes": capture_png}] if capture_png else [{"type": "text", "value": "Capture Ratios: não disponível."}]},
            {"title": "Correlation Matrix", "items": [{"type": "image", "png_bytes": corr_png}] if corr_png else [{"type": "text", "value": "Correlation Matrix: não disponível."}]},
            {"title": "History", "items": (
                ([{"type": "image", "png_bytes": hist_cum_png}] if hist_cum_png else [{"type": "text", "value": "History (Cumulative): não disponível."}]) +
                ([{"type": "image", "png_bytes": hist_dd_png}] if hist_dd_png else [{"type": "text", "value": "History (Drawdown): não disponível."}])
            )},
            {"title": "Portfolio Solver", "items": solver_items},
        ]

        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_pdf.close()

        write_pdf_report(
            output_path=tmp_pdf.name,
            report_title="Portfolio Risk Management Report",
            subtitle=f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            sections=sections
        )

        with open(tmp_pdf.name, "rb") as f:
            pdf_bytes = f.read()

        try:
            os.remove(tmp_pdf.name)
        except Exception:
            pass

    st.success("Report ready!")
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name="portfolio_risk_report_full.pdf",
        mime="application/pdf"
    )

st.info("Dependências para exportar imagens do Plotly em PDF: `kaleido`, `reportlab`, `Pillow` (adicione no requirements.txt).")
