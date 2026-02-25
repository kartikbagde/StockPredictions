import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="TradePro Quant", layout="wide")
st.title("📈 TradePro Quant - AI Trading Dashboard")

# -------------------------------
# ==========================
# CUSTOM RSI FUNCTION
# ==========================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# ==========================
# CUSTOM MACD FUNCTION
# ==========================
def calculate_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    return macd, signal
# --------------------------------------------

# ==========================
# SIDEBAR FILTERS
# ==========================
st.sidebar.header("🔎 Stock Filters")

ticker_input = st.sidebar.text_input("Enter Stock Symbol", "RELIANCE")
period = st.sidebar.selectbox("Select Time Period", ["6mo", "1y", "2y"])
compare_input = st.sidebar.text_input(
    "Compare Multiple Stocks (comma separated)",
    "RELIANCE,TCS,INFY"
)

# Ensure NSE format
ticker = ticker_input.strip().upper()
if "." not in ticker:
    ticker = ticker + ".NS"

# ==========================
# DOWNLOAD DATA
# ==========================
data = yf.download(ticker, period=period)

if data.empty:
    st.error("❌ Invalid stock symbol or no data available.")
    st.stop()

data = data.dropna()

latest_price = float(data["Close"].iloc[-1])

st.metric("💰 Live Price", f"₹ {round(latest_price,2)}")

# ==========================
# TECHNICAL INDICATORS
# ==========================

close_series = data["Close"].astype(float)

data["MA50"] = close_series.rolling(50).mean()
data["MA200"] = close_series.rolling(200).mean()

# Custom RSI
data["RSI"] = calculate_rsi(close_series)

# Custom MACD
data["MACD"], data["MACD_SIGNAL"] = calculate_macd(close_series)

# ==========================
# CANDLESTICK CHART
# ==========================
st.subheader("📊 Candlestick Chart")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Candlestick"
))

fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="50 DMA"))
fig.add_trace(go.Scatter(x=data.index, y=data["MA200"], name="200 DMA"))

fig.update_layout(height=600, xaxis_rangeslider_visible=False)

st.plotly_chart(fig, use_container_width=True)

# ==========================
# RSI
# ==========================
st.subheader("📉 RSI")

rsi_fig = go.Figure()
rsi_fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI"))
rsi_fig.add_hline(y=70)
rsi_fig.add_hline(y=30)

st.plotly_chart(rsi_fig, use_container_width=True)

# ==========================
# MACD
# ==========================
st.subheader("📊 MACD")

macd_fig = go.Figure()
macd_fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD"))
macd_fig.add_trace(go.Scatter(x=data.index, y=data["MACD_SIGNAL"], name="Signal"))

st.plotly_chart(macd_fig, use_container_width=True)

# ==========================
# SIGNAL GENERATOR
# ==========================
st.subheader("📢 Trading Signal")

data["Signal"] = 0

buy_condition = (
    (data["RSI"] < 30) &
    (data["MACD"] > data["MACD_SIGNAL"]) &
    (data["MA50"] > data["MA200"])
)

sell_condition = (
    (data["RSI"] > 70) &
    (data["MACD"] < data["MACD_SIGNAL"]) &
    (data["MA50"] < data["MA200"])
)

data.loc[buy_condition, "Signal"] = 1
data.loc[sell_condition, "Signal"] = -1

latest_signal = data["Signal"].iloc[-1]

if latest_signal == 1:
    st.success("🟢 BUY Signal")
elif latest_signal == -1:
    st.error("🔴 SELL Signal")
else:
    st.info("⚪ HOLD")

# ==========================
# BACKTESTING
# ==========================
st.subheader("📈 Backtesting Strategy")

initial_capital = 100000
cash = initial_capital
shares = 0
portfolio_values = []

for i in range(len(data)):
    if data["Signal"].iloc[i] == 1 and cash > 0:
        shares = cash / data["Close"].iloc[i]
        cash = 0
    elif data["Signal"].iloc[i] == -1 and shares > 0:
        cash = shares * data["Close"].iloc[i]
        shares = 0

    portfolio_values.append(cash + shares * data["Close"].iloc[i])

data["Portfolio"] = portfolio_values

final_value = portfolio_values[-1]
profit = final_value - initial_capital
roi = (profit / initial_capital) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Final Portfolio Value", f"₹ {round(final_value,2)}")
col2.metric("Total Profit", f"₹ {round(profit,2)}")
col3.metric("ROI", f"{round(roi,2)} %")

backtest_fig = go.Figure()
backtest_fig.add_trace(go.Scatter(x=data.index, y=data["Portfolio"], name="Strategy Growth"))
st.plotly_chart(backtest_fig, use_container_width=True)

# ==========================
# 7-DAY FORECAST (Lightweight)
# ==========================
st.subheader("🤖 7-Day AI Forecast")

df = data[["Close"]].dropna().reset_index()
df["Day"] = np.arange(len(df))

X = df[["Day"]]
y = df["Close"]

model = LinearRegression()
model.fit(X, y)

future_days = np.arange(len(df), len(df) + 7).reshape(-1, 1)
future_preds = model.predict(future_days)

future_dates = pd.date_range(
    start=df["Date"].iloc[-1] + pd.Timedelta(days=1),
    periods=7
)

forecast_fig = go.Figure()
forecast_fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Historical"))
forecast_fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_preds,
    name="Forecast",
    line=dict(dash="dash")
))

st.plotly_chart(forecast_fig, use_container_width=True)

# ==========================
# PORTFOLIO SIMULATOR
# ==========================
st.subheader("💼 Portfolio Simulator")

investment = st.number_input("Enter Investment Amount", min_value=0.0)

if investment > 0:
    shares = investment / latest_price
    projected_value = shares * future_preds[-1]
    profit = projected_value - investment

    st.metric("Projected Value (7 Days)", f"₹ {round(projected_value,2)}")

    if profit > 0:
        st.success(f"🟢 Expected Profit: ₹ {round(profit,2)}")
    else:
        st.error(f"🔴 Expected Loss: ₹ {round(profit,2)}")

# ==========================
# MULTI-STOCK COMPARISON
# ==========================
st.subheader("📊 Multi-Stock 7-Day Forecast Comparison")

stocks = [s.strip().upper() for s in compare_input.split(",")]
comparison_fig = go.Figure()

for stock in stocks:
    if "." not in stock:
        stock = stock + ".NS"

    stock_data = yf.download(stock, period="6mo")

    if stock_data.empty:
        continue

    df_stock = stock_data[["Close"]].dropna().reset_index()
    df_stock["Day"] = np.arange(len(df_stock))

    X_stock = df_stock[["Day"]]
    y_stock = df_stock["Close"]

    model_stock = LinearRegression()
    model_stock.fit(X_stock, y_stock)

    future_days_stock = np.arange(len(df_stock), len(df_stock) + 7).reshape(-1, 1)
    future_preds_stock = model_stock.predict(future_days_stock)

    comparison_fig.add_trace(
        go.Scatter(y=future_preds_stock, name=stock)
    )

st.plotly_chart(comparison_fig, use_container_width=True)


