import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

st.set_page_config(page_title="TradePro Quant", layout="wide")
st.title("📈 TradePro Quant - AI Trading Platform")

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("Main Stock")
ticker = st.sidebar.text_input("Enter Stock Symbol", "RELIANCE")
if "." not in ticker:
    ticker = ticker.upper() + ".NS"
period = st.sidebar.selectbox("Select Time Period", ["6mo", "1y", "2y"])

st.sidebar.header("Multi-Stock LSTM Comparison")
compare_input = st.sidebar.text_input(
    "Enter stocks separated by comma", "RELIANCE,TCS,INFY"
)

# ==========================
# DOWNLOAD DATA
# ==========================
data = yf.download(ticker, period=period)
if data.empty:
    st.error("Invalid Stock Symbol ❌")
    st.stop()

latest_price = float(data["Close"].iloc[-1])
st.subheader(f"💰 Live Price: ₹ {round(latest_price,2)}")

# ==========================
# INDICATORS
# ==========================

# Ensure Close column is 1D numeric Series
if isinstance(data["Close"], pd.DataFrame):
    close_series = data["Close"].iloc[:,0]  # take first column
else:
    close_series = data["Close"]

close_series = pd.to_numeric(close_series, errors="coerce").dropna()

data["MA50"] = close_series.rolling(50).mean()
data["MA200"] = close_series.rolling(200).mean()

# RSI & MACD
rsi_indicator = RSIIndicator(close=close_series, window=14)
data["RSI"] = rsi_indicator.rsi()

macd_indicator = MACD(close=close_series)
data["MACD"] = macd_indicator.macd()
data["MACD_SIGNAL"] = macd_indicator.macd_signal()

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
rsi_fig.update_layout(height=300)
st.plotly_chart(rsi_fig, use_container_width=True)

# ==========================
# MACD
# ==========================
st.subheader("📊 MACD")
macd_fig = go.Figure()
macd_fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD"))
macd_fig.add_trace(go.Scatter(x=data.index, y=data["MACD_SIGNAL"], name="Signal"))
macd_fig.update_layout(height=300)
st.plotly_chart(macd_fig, use_container_width=True)

# ==========================
# SIGNAL GENERATOR
# ==========================
st.subheader("📢 Trading Signal")
data["Signal"] = 0
buy_condition = (data["RSI"] < 30) & (data["MACD"] > data["MACD_SIGNAL"]) & (data["MA50"] > data["MA200"])
sell_condition = (data["RSI"] > 70) & (data["MACD"] < data["MACD_SIGNAL"]) & (data["MA50"] < data["MA200"])
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
profit = portfolio_values[-1] - initial_capital
roi = (profit / initial_capital) * 100
st.write(f"Final Portfolio Value: ₹ {round(portfolio_values[-1],2)}")
st.write(f"Total Profit: ₹ {round(profit,2)}")
st.write(f"ROI: {round(roi,2)} %")

backtest_fig = go.Figure()
backtest_fig.add_trace(go.Scatter(x=data.index, y=data["Portfolio"], name="Strategy Growth"))
st.plotly_chart(backtest_fig, use_container_width=True)

# ==========================
# LSTM FORECAST
# ==========================
st.subheader("🤖 7-Day LSTM Forecast")
df = data[["Close"]].dropna()

if len(df) > 60:
    close_values = df["Close"].values.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(close_values)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i,0])
        y.append(scaled_data[i,0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60,1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, verbose=0)

    last_60 = scaled_data[-60:].reshape(1,60,1)
    future_scaled = []

    for _ in range(7):
        pred = model.predict(last_60, verbose=0)[0][0]
        future_scaled.append(pred)
        last_60 = np.append(last_60[:,1:,:], [[[pred]]], axis=1)

    future_scaled = np.array(future_scaled).reshape(-1,1)
    future_preds = scaler.inverse_transform(future_scaled)
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=7)

    lstm_fig = go.Figure()
    lstm_fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Historical"))
    lstm_fig.add_trace(go.Scatter(x=future_dates, y=future_preds.flatten(), name="LSTM Forecast", line=dict(dash="dash")))
    st.plotly_chart(lstm_fig, use_container_width=True)

    # Portfolio Simulator
    st.subheader("💼 Portfolio Simulator")
    investment = st.number_input("Enter Investment Amount", min_value=0.0)
    if investment > 0:
        shares = investment / latest_price
        projected_value = shares * future_preds[-1][0]
        profit = projected_value - investment
        st.success(f"Projected Value After 7 Days: ₹ {round(projected_value,2)}")
        if profit > 0:
            st.markdown(f"🟢 Expected Profit: ₹ {round(profit,2)}")
        else:
            st.markdown(f"🔴 Expected Loss: ₹ {round(profit,2)}")

# ==========================
# MULTI-STOCK LSTM COMPARISON
# ==========================
st.subheader("📊 Multi-Stock 7-Day LSTM Comparison")
stocks = [s.strip().upper() for s in compare_input.split(",")]
comparison_fig = go.Figure()

for stock in stocks:
    if "." not in stock:
        stock = stock + ".NS"
    stock_data = yf.download(stock, period="6mo")
    if len(stock_data) < 60:
        continue
    df_stock = stock_data[["Close"]].dropna()
    close_values = df_stock["Close"].values.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(close_values)

    X_stock, y_stock = [], []
    for i in range(60, len(scaled)):
        X_stock.append(scaled[i-60:i,0])
        y_stock.append(scaled[i,0])
    X_stock, y_stock = np.array(X_stock), np.array(y_stock)
    X_stock = X_stock.reshape(X_stock.shape[0], X_stock.shape[1], 1)

    model_stock = Sequential()
    model_stock.add(LSTM(50, return_sequences=True, input_shape=(60,1)))
    model_stock.add(LSTM(50))
    model_stock.add(Dense(1))
    model_stock.compile(optimizer='adam', loss='mse')
    model_stock.fit(X_stock, y_stock, epochs=3, verbose=0)

    last_60 = scaled[-60:].reshape(1,60,1)
    future_scaled = []
    for _ in range(7):
        pred = model_stock.predict(last_60, verbose=0)[0][0]
        future_scaled.append(pred)
        last_60 = np.append(last_60[:,1:,:], [[[pred]]], axis=1)

    future_scaled = np.array(future_scaled).reshape(-1,1)
    future_preds = scaler.inverse_transform(future_scaled)

    comparison_fig.add_trace(go.Scatter(y=future_preds.flatten(), name=stock))


st.plotly_chart(comparison_fig, use_container_width=True)
