import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import math


def fetch_stock_data(symbol="TCS.NS", period="5y"):
    data = yf.download(symbol, period=period)
    if data.empty:
        raise ValueError("No data downloaded. Check stock symbol.")
    return data


def scale_data(data):
    close_data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)
    return scaled_data, scaler


def create_sequences(scaled_data, sequence_length=60):
    X, y = [], []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])

    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future_days(model, scaled_data, scaler, days=7, sequence_length=60):
    future_predictions = []

    last_sequence = scaled_data[-sequence_length:]
    current_sequence = last_sequence.reshape(1, sequence_length, 1)

    for _ in range(days):
        next_day = model.predict(current_sequence, verbose=0)
        
        future_predictions.append(next_day[0][0])

        next_day_reshaped = next_day.reshape(1, 1, 1)
        current_sequence = np.append(current_sequence[:, 1:, :], 
                                     next_day_reshaped, 
                                     axis=1)

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)

    return future_predictions

def main():
    stock_list = ["TCS.NS", "INFY.NS", "RELIANCE.NS"]
    rmse_results = {}

    for stock_symbol in stock_list:
        try:
            print("\n====================================")
            print(f"Processing Stock: {stock_symbol}")
            print("====================================")

            data = fetch_stock_data(stock_symbol)

            scaled_data, scaler = scale_data(data)

            X, y = create_sequences(scaled_data)

            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = build_lstm_model((X_train.shape[1], 1))
            print("Training model...")
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

            predictions = model.predict(X_test, verbose=0)
            predictions = scaler.inverse_transform(predictions)
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

            rmse = math.sqrt(mean_squared_error(y_test_actual, predictions))
            rmse_results[stock_symbol] = rmse
            print(f"\nRMSE for {stock_symbol}: ₹ {rmse:.2f}")

            plt.figure(figsize=(10, 5))
            plt.plot(y_test_actual, label="Actual Price")
            plt.plot(predictions, label="Predicted Price")
            plt.title(f"{stock_symbol} - Actual vs Predicted")
            plt.legend()
            plt.grid(True)
            plt.show()

            future_prices = predict_future_days(model, scaled_data, scaler, days=7)

            print(f"\nNext 7 Day Predictions for {stock_symbol}:")
            for i, price in enumerate(future_prices, 1):
                print(f"Day {i}: ₹ {price[0]:.2f}")

            plt.figure(figsize=(8, 4))
            plt.plot(range(1, 8), future_prices, marker='o')
            plt.title(f"{stock_symbol} - Next 7 Day Prediction")
            plt.grid(True)
            plt.show()

        except Exception as e:
            print(f"Error processing {stock_symbol}: {e}")

    print("\n================ RMSE Comparison ================")
    for stock, value in rmse_results.items():
        print(f"{stock} → ₹ {value:.2f}")

    plt.figure(figsize=(8, 4))
    plt.bar(rmse_results.keys(), rmse_results.values())
    plt.title("RMSE Comparison Across Stocks")
    plt.ylabel("RMSE (Lower is Better)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()