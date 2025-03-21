import os
import requests
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from flask import redirect, render_template, request, session
from functools import wraps

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def apology(message, code=400):
    """Render message as an apology to user."""
    def escape(s):
        for old, new in [("-", "--"), (" ", "-"), ("_", "__"), ("?", "~q"),
                         ("%", "~p"), ("#", "~h"), ("/", "~s"), ("\"", "''")]:
            s = s.replace(old, new)
        return s
    return render_template("apology.html", top=code, bottom=escape(message)), code

def login_required(f):
    """Decorate routes to require login."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated_function

def lookup(symbol):
    """Look up quote for symbol using Alpha Vantage API."""
    try:
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            logging.error("ALPHA_VANTAGE_API_KEY not set in environment")
            return None

        # Append .NS for Indian NSE stocks
        symbol_with_exchange = f"{symbol.upper()}.NS"
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol_with_exchange}&interval=1min&apikey={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        logging.debug(f"Alpha Vantage quote response for {symbol_with_exchange}: {data}")

        if "Time Series (1min)" not in data:
            logging.warning(f"No quote data returned for {symbol_with_exchange}: {data}")
            return None

        latest_time = list(data["Time Series (1min)"].keys())[0]
        latest_data = data["Time Series (1min)"][latest_time]
        price = latest_data["4. close"]

        return {
            "symbol": symbol.upper(),
            "name": data.get("Meta Data", {}).get("2. Symbol", symbol.upper()),
            "price": float(price)
        }
    except requests.RequestException as e:
        logging.error(f"API request failed for {symbol}: {e}")
        return None
    except (KeyError, ValueError) as e:
        logging.error(f"Invalid data format for {symbol}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error in lookup for {symbol}: {e}")
        return None

def predict_price(symbol):
    """Predict next day's stock price using Alpha Vantage API with improved accuracy."""
    try:
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            logging.error("ALPHA_VANTAGE_API_KEY not set in environment")
            return None

        # Fetch 200 days of daily data
        symbol_with_exchange = f"{symbol.upper()}.NS"
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol_with_exchange}&outputsize=full&apikey={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        logging.debug(f"Alpha Vantage time_series response for {symbol_with_exchange}: {data}")

        if "Time Series (Daily)" not in data:
            logging.warning(f"No historical data returned for {symbol_with_exchange}: {data}")
            return None

        # Convert to DataFrame and sort by date
        df = pd.DataFrame(data["Time Series (Daily)"]).T
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().tail(200)  # Last 200 days
        df = df.astype(float)

        # Feature engineering: closing price and 20-day SMA
        prices = df["4. close"].values
        df["SMA_20"] = df["4. close"].rolling(window=20).mean().fillna(method="bfill")
        features = df[["4. close", "SMA_20"]].values
        dates = df.index.strftime("%Y-%m-%d").tolist()

        if len(prices) < 60:  # Minimum data for meaningful prediction
            logging.warning(f"Insufficient data for {symbol_with_exchange}: {len(prices)} days")
            return None

        # Normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)

        # Prepare sequences (look-back of 20 days)
        look_back = 20
        X, y = [], []
        for i in range(look_back, len(scaled_features)):
            X.append(scaled_features[i-look_back:i])  # 2 features per time step
            y.append(scaled_features[i, 0])  # Predict closing price
        X = np.array(X)
        y = np.array(y)

        # Split into train (80%) and validation (20%)
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        # Build improved LSTM model
        model = Sequential()
        model.add(Input(shape=(look_back, 2)))  # 2 features: close, SMA
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.2))  # Prevent overfitting
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))  # Output: next day's close
        model.compile(optimizer="adam", loss="mse")

        # Train with early stopping
        early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
                  callbacks=[early_stopping], verbose=0)

        # Predict next day's price
        last_sequence = scaled_features[-look_back:]
        last_sequence = np.reshape(last_sequence, (1, look_back, 2))
        predicted_scaled = model.predict(last_sequence, verbose=0)

        # Inverse transform prediction (only for closing price)
        predicted_array = np.zeros((1, 2))  # Dummy array for inverse transform
        predicted_array[:, 0] = predicted_scaled[:, 0]  # Set predicted close
        predicted_price = scaler.inverse_transform(predicted_array)[0, 0]

        # Prepare chart data
        historical_prices = prices.tolist()
        chart_prices = historical_prices + [predicted_price]

        return {
            "predicted_price": round(predicted_price, 2),
            "dates": dates,
            "chart_prices": chart_prices
        }
    except requests.RequestException as e:
        logging.error(f"API request failed for {symbol} in predict_price: {e}")
        return None
    except Exception as e:
        logging.error(f"Error in predict_price for {symbol}: {e}")
        return None

def inr(value):
    """Format value as INR (₹) for Indian stocks."""
    return f"₹{value:,.2f}"