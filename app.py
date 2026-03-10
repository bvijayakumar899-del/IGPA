import os
# Set TensorFlow environment variable to suppress oneDNN warnings at the very beginning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
from bs4 import BeautifulSoup
import re
from textblob import TextBlob
import json
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import pickle
import feedparser
import urllib.parse
import time
import warnings
from flask_cors import CORS
import logging
import threading
from functools import wraps
import secrets
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import schedule
import atexit

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generate a random secret key
app.permanent_session_lifetime = timedelta(minutes=30)  # Set session timeout to 30 minutes
CORS(app)

# Add template filters
@app.template_filter('absolute')
def absolute_filter(value):
    """Return the absolute value of a number."""
    return abs(value)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# User data file
USERS_FILE = 'users.json'

# Global constant for lookback period
LOOKBACK_PERIOD = 60  # Reduced for faster processing

# Global cache for models and data
model_cache = {}
cache_lock = threading.Lock()
training_in_progress = set()

# Global variables for market indices
nifty_data = {
    'current_value': 0,
    'change': 0,
    'change_percent': 0,
    'last_updated': ''
}
sensex_data = {
    'current_value': 0,
    'change': 0,
    'change_percent': 0,
    'last_updated': ''
}

# --- User Management Functions ---
def load_users():
    """Load users from JSON file."""
    if not os.path.exists(USERS_FILE):
        return {}
    
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading users: {e}")
        return {}

def save_users(users):
    """Save users to JSON file."""
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=4)
        return True
    except IOError as e:
        logger.error(f"Error saving users: {e}")
        return False

def add_user(username, email, password):
    """Add a new user to the JSON file."""
    users = load_users()
    
    # Convert to lowercase for case-insensitive comparison
    username_lower = username.lower()
    email_lower = email.lower()
    
    # Check if username already exists
    for key in users:
        if key.lower() == username_lower:
            return False, "Username already exists"
    
    # Check if email already exists
    for user in users.values():
        if user.get('email', '').lower() == email_lower:
            return False, "Email already registered"
    
    # Hash the password
    try:
        hashed_password = generate_password_hash(password)
    except Exception as e:
        logger.error(f"Error hashing password: {e}")
        return False, "Error processing password"
    
    # Add new user with lowercase username as key
    users[username_lower] = {
        'email': email,
        'password': hashed_password,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'display_name': username  # Store original case for display
    }
    
    # Save updated users
    if save_users(users):
        return True, "Registration successful"
    else:
        return False, "Failed to save user data"

def verify_user(login_input, password):
    """Verify user credentials using either username or email."""
    users = load_users()
    
    # Convert to lowercase for case-insensitive lookup
    login_input_lower = login_input.lower()
    
    # First, try to find by username (which is the key)
    if login_input_lower in users:
        user_data = users[login_input_lower]
        try:
            password_match = check_password_hash(user_data['password'], password)
            if password_match:
                return login_input_lower, user_data  # Return username key and user data
            else:
                logger.warning(f"Password mismatch for username: {login_input}")
                return None, None
        except Exception as e:
            logger.error(f"Error verifying password for {login_input}: {e}")
            return None, None
    
    # If not found by username, try to find by email
    for username_key, user_data in users.items():
        if user_data.get('email', '').lower() == login_input_lower:
            try:
                password_match = check_password_hash(user_data['password'], password)
                if password_match:
                    return username_key, user_data  # Return username key and user data
                else:
                    logger.warning(f"Password mismatch for email: {login_input}")
                    return None, None
            except Exception as e:
                logger.error(f"Error verifying password for email {login_input}: {e}")
                return None, None
    
    # If we get here, the user was not found by either username or email
    logger.warning(f"Login attempt with non-existent username/email: {login_input}")
    return None, None

# --- Authentication Decorator ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            flash('You need to login first.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- Utility Functions ---
def normalize_symbol(symbol):
    """Normalize stock symbol to uppercase and append .NS if not present."""
    if not symbol:
        return symbol
    
    symbol = symbol.strip().upper()
    if not symbol.endswith('.NS'):
        symbol = symbol + '.NS'
    
    return symbol

def filter_last_5_years(df):
    try:
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        if not pd.api.types.is_datetime64_any_dtype(df.columns):
            df.columns = pd.to_datetime(df.columns)
    except Exception as e:
        logger.warning(f"Warning: Could not convert columns or index to datetime objects. Error: {e}")
        return df
    
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=5)
    if pd.api.types.is_datetime64_any_dtype(df.index):
        return df.loc[df.index >= cutoff]
    elif pd.api.types.is_datetime64_any_dtype(df.columns):
        return df.loc[:, df.columns >= cutoff]
    return df

def format_in_crores(value):
    """Format value as crores with 2 decimal places."""
    if value is None or pd.isna(value):
        return "N/A"
    
    try:
        value = float(value)
        # Convert to crores (1 crore = 10,000,000)
        value_in_crores = value / 10000000
        return f"{value_in_crores:.2f}"
    except (ValueError, TypeError):
        return "N/A"

def format_indian_currency(value):
    """Format value as Indian currency with rupees symbol and commas."""
    if value is None or pd.isna(value):
        return "N/A"
    
    try:
        value = float(value)
        if value >= 10000000:  # Crore
            return f"₹{value/10000000:.2f} Cr"
        elif value >= 100000:  # Lakh
            return f"₹{value/100000:.2f} L"
        elif value >= 1000:  # Thousand
            return f"₹{value/1000:.2f} K"
        else:
            return f"₹{value:.2f}"
    except (ValueError, TypeError):
        return "N/A"

def format_percentage(value):
    """Format value as percentage with 2 decimal places."""
    if value is None or pd.isna(value):
        return "N/A"
    
    try:
        return f"{float(value):.2f}%"
    except (ValueError, TypeError):
        return "N/A"

def format_two_decimals(value):
    """Format value with 2 decimal places."""
    if value is None or pd.isna(value):
        return "N/A"
    
    try:
        return f"{float(value):.2f}"
    except (ValueError, TypeError):
        return "N/A"

def dataframe_to_dict(df):
    if df.empty:
        return {"headers": [], "rows": []}
    return {
        "headers": df.columns.astype(str).tolist(),
        "rows": df.astype(str).values.tolist()
    }

def format_news_date(date_str):
    """Format news date to a more readable format."""
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        return date_obj.strftime('%b %d, %Y %I:%M %p')
    except (ValueError, TypeError):
        return date_str

# --- Market Indices Functions ---
def get_nifty_data():
    """Get current Nifty 50 data from Yahoo Finance."""
    try:
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="2d")
        
        if len(hist) < 2:
            logger.warning("Not enough historical data for Nifty")
            return None
        
        current_value = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        change = current_value - prev_close
        change_percent = (change / prev_close) * 100
        
        return {
            'current_value': current_value,
            'change': change,
            'change_percent': change_percent,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        logger.error(f"Error fetching Nifty data: {e}")
        return None

def get_sensex_data():
    """Get current Sensex data from Yahoo Finance."""
    try:
        sensex = yf.Ticker("^BSESN")
        hist = sensex.history(period="2d")
        
        if len(hist) < 2:
            logger.warning("Not enough historical data for Sensex")
            return None
        
        current_value = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        change = current_value - prev_close
        change_percent = (change / prev_close) * 100
        
        return {
            'current_value': current_value,
            'change': change,
            'change_percent': change_percent,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        logger.error(f"Error fetching Sensex data: {e}")
        return None

def update_market_indices():
    """Update global market indices data."""
    global nifty_data, sensex_data
    
    nifty_info = get_nifty_data()
    if nifty_info:
        nifty_data = nifty_info
    
    sensex_info = get_sensex_data()
    if sensex_info:
        sensex_data = sensex_info
    
    logger.info("Market indices updated")

def scheduled_update():
    """Scheduled function to update market indices."""
    update_market_indices()

# --- Stock Data Functions ---
def get_stock_data(symbol: str):
    try:
        logger.info(f"Fetching stock data for {symbol}")
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")
        
        if hist.empty or len(hist) < 22:
            logger.warning(f"No sufficient historical data found for {symbol}")
            return None
        
        try:
            daily_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100 if len(hist) >= 2 else 0.0
            weekly_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6]) * 100 if len(hist) >= 6 else 0.0
            monthly_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-22]) / hist['Close'].iloc[-22]) * 100 if len(hist) >= 22 else 0.0
        except Exception as e:
            logger.error(f"Error calculating price changes for {symbol}: {e}")
            return None
        
        info = stock.info
        latest_price = hist.iloc[-1]
        current_price = info.get("currentPrice", latest_price["Close"])
        
        stock_data = {
            'symbol': symbol,
            'companyName': info.get("longName", "N/A"),
            'currentPrice': current_price,
            'dailyChange': daily_change,
            'weeklyChange': weekly_change,
            'monthlyChange': monthly_change,
            'marketCap': info.get('marketCap', 0),
            'peRatio': info.get('trailingPE', 0),
            'pbRatio': info.get('priceToBook', 0),
            'dividendYield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0),
            'week52High': info.get('fiftyTwoWeekHigh', 0),
            'week52Low': info.get('fiftyTwoWeekLow', 0),
            'avgVolume': info.get('averageVolume', 0),
            'revenue': info.get('totalRevenue', 0),
            'netIncome': info.get('netIncomeToCommon', 0),
            'eps': info.get('trailingEps', 0),
            'roe': info.get('returnOnEquity', 0),
            'debtToEquity': info.get('debtToEquity', 0),
            'currentRatio': info.get('currentRatio', 0),
            'bookValue': info.get('bookValue', 0),
            'profitMargin': info.get('profitMargins', 0),
            'operatingMargin': info.get('operatingMargins', 0),
            'quarterlyRevenueGrowth': info.get('quarterlyRevenueGrowth', 0),
            # Additional fields
            'summary': info.get('longBusinessSummary', "N/A"),
            'website': info.get('website', "N/A"),
            'sector': info.get('sector', "N/A"),
            'industry': info.get('industry', "N/A"),
            'forwardPE': info.get('forwardPE', 0),
            'pegRatio': info.get('pegRatio', 0),
            'priceToBook': info.get('priceToBook', 0),
            'enterpriseToEbitda': info.get('enterpriseToEbitda', 0),
            'profitMargins': info.get('profitMargins', 0),
            'operatingMargins': info.get('operatingMargins', 0),
            'returnOnEquity': info.get('returnOnEquity', 0),
            'returnOnAssets': info.get('returnOnAssets', 0),
            'revenuePerShare': info.get('revenuePerShare', 0),
            'quarterlyEarningsGrowth': info.get('quarterlyEarningsGrowth', 0),
            'grossProfits': info.get('grossProfits', 0),
            'ebitda': info.get('ebitda', 0),
            'totalCash': info.get('totalCash', 0),
            'totalDebt': info.get('totalDebt', 0),
            'dividendRate': info.get('dividendRate', 0),
            'payoutRatio': info.get('payoutRatio', 0),
            'fiftyDayAverage': info.get('fiftyDayAverage', 0),
            'twoHundredDayAverage': info.get('twoHundredDayAverage', 0),
            'day200MovingAvg': info.get('twoHundredDayAverage', 0),
            'day50MovingAvg': info.get('fiftyDayAverage', 0),
            'freeCashflow': info.get('freeCashflow', 0),
            'tangibleBookValue': info.get('tangibleBookValue', 0),
            'netDebt': info.get('netDebt', 0)
        }
        
        logger.info(f"Successfully fetched stock data for {symbol}")
        return stock_data
    except Exception as e:
        logger.error(f"Error in get_stock_data for {symbol}: {e}")
        return None

def get_stock_news(ticker_symbol, num_articles=5):
    try:
        logger.info(f"Fetching news for {ticker_symbol}")
        query = urllib.parse.quote(f"{ticker_symbol} stock India")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        articles = []
        
        for entry in feed.entries[:num_articles]:
            articles.append({
                'title': entry.title,
                'link': entry.link,
                'publisher': entry.source.title if 'source' in entry else "Google News",
                'published': format_news_date(time.strftime('%Y-%m-%d %H:%M:%S', entry.published_parsed)) if 'published_parsed' in entry else "N/A"
            })
        
        logger.info(f"Successfully fetched {len(articles)} news articles for {ticker_symbol}")
        return {"articles": articles}
    except Exception as e:
        logger.error(f"Error in get_stock_news for {ticker_symbol}: {e}")
        return {"articles": []}

def get_financial_data(symbol):
    try:
        logger.info(f"Fetching financial data for {symbol}")
        ticker = yf.Ticker(symbol)
        
        # Annual financials
        annual_income = ticker.financials
        annual_balance = ticker.balance_sheet
        annual_cashflow = ticker.cashflow
        
        # Quarterly financials
        quarterly_income = ticker.quarterly_financials
        quarterly_balance = ticker.quarterly_balance_sheet
        quarterly_cashflow = ticker.quarterly_cashflow
        
        # Format data for display - convert to crores and format with 2 decimal places
        def format_financial_dataframe(df):
            if df.empty:
                return df
            # Create a copy to avoid modifying the original
            df_formatted = df.copy()
            # Apply formatting only to numeric columns
            for col in df_formatted.columns:
                if pd.api.types.is_numeric_dtype(df_formatted[col]):
                    df_formatted[col] = df_formatted[col].apply(format_in_crores)
            return df_formatted
        
        financial_data = {
            "annualIncome": dataframe_to_dict(format_financial_dataframe(filter_last_5_years(annual_income))),
            "annualBalance": dataframe_to_dict(format_financial_dataframe(filter_last_5_years(annual_balance))),
            "annualCashflow": dataframe_to_dict(format_financial_dataframe(filter_last_5_years(annual_cashflow))),
            "quarterlyIncome": dataframe_to_dict(format_financial_dataframe(filter_last_5_years(quarterly_income))),
            "quarterlyBalance": dataframe_to_dict(format_financial_dataframe(filter_last_5_years(quarterly_balance))),
            "quarterlyCashflow": dataframe_to_dict(format_financial_dataframe(filter_last_5_years(quarterly_cashflow)))
        }
        
        logger.info(f"Successfully fetched financial data for {symbol}")
        return financial_data
    except Exception as e:
        logger.error(f"Error in get_financial_data for {symbol}: {e}")
        return {
            "annualIncome": {"headers": [], "rows": []},
            "annualBalance": {"headers": [], "rows": []},
            "annualCashflow": {"headers": [], "rows": []},
            "quarterlyIncome": {"headers": [], "rows": []},
            "quarterlyBalance": {"headers": [], "rows": []},
            "quarterlyCashflow": {"headers": [], "rows": []}
        }

def get_price_history(symbol, period="1y"):
    try:
        logger.info(f"Fetching price history for {symbol} with period {period}")
        hist = yf.download(symbol, period=period, interval='1d')
        
        if hist.empty:
            logger.warning(f"No price history found for {symbol} with period {period}")
            return None
        
        # Convert DataFrame columns to Series if needed
        close_prices = hist['Close'].squeeze().tolist() if isinstance(hist['Close'], pd.DataFrame) else hist['Close'].tolist()
        dates = hist.index.strftime('%Y-%m-%d').tolist()
        
        # Get OHLC data if available
        ohlc = None
        if all(col in hist.columns for col in ['Open', 'High', 'Low', 'Close']):
            ohlc = {
                'open': hist['Open'].squeeze().tolist() if isinstance(hist['Open'], pd.DataFrame) else hist['Open'].tolist(),
                'high': hist['High'].squeeze().tolist() if isinstance(hist['High'], pd.DataFrame) else hist['High'].tolist(),
                'low': hist['Low'].squeeze().tolist() if isinstance(hist['Low'], pd.DataFrame) else hist['Low'].tolist(),
                'close': close_prices
            }
        
        price_data = {
            'dates': dates,
            'prices': close_prices,
            'ohlc': ohlc
        }
        
        logger.info(f"Successfully fetched price history for {symbol}")
        return price_data
    except Exception as e:
        logger.error(f"Error in get_price_history for {symbol}: {e}")
        return None

def get_moving_averages(symbol, start_date="2025-01-01"):
    """Get moving averages data for a stock symbol starting from a specific date."""
    try:
        logger.info(f"Fetching moving averages for {symbol} from {start_date}")
        
        # Set end date to today
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Download historical data
        hist = yf.download(symbol, start=start_date, end=end_date, interval='1d')
        
        if hist.empty:
            logger.warning(f"No data found for {symbol} from {start_date} to {end_date}")
            return None
        
        # Calculate SMAs
        sma100 = None
        sma200 = None
        
        if len(hist) >= 100:
            sma100_values = hist['Close'].rolling(window=100).mean().dropna()
            sma100 = {
                'values': sma100_values.squeeze().tolist() if isinstance(sma100_values, pd.DataFrame) else sma100_values.tolist(),
                'offset': len(hist) - len(sma100_values)
            }
        
        if len(hist) >= 200:
            sma200_values = hist['Close'].rolling(window=200).mean().dropna()
            sma200 = {
                'values': sma200_values.squeeze().tolist() if isinstance(sma200_values, pd.DataFrame) else sma200_values.tolist(),
                'offset': len(hist) - len(sma200_values)
            }
        
        sma = {
            'sma100': sma100,
            'sma200': sma200
        }
        
        # Calculate EMAs
        ema100 = None
        ema200 = None
        
        if len(hist) >= 100:
            ema100_values = hist['Close'].ewm(span=100).mean().dropna()
            ema100 = {
                'values': ema100_values.squeeze().tolist() if isinstance(ema100_values, pd.DataFrame) else ema100_values.tolist(),
                'offset': len(hist) - len(ema100_values)
            }
        
        if len(hist) >= 200:
            ema200_values = hist['Close'].ewm(span=200).mean().dropna()
            ema200 = {
                'values': ema200_values.squeeze().tolist() if isinstance(ema200_values, pd.DataFrame) else ema200_values.tolist(),
                'offset': len(hist) - len(ema200_values)
            }
        
        ema = {
            'ema100': ema100,
            'ema200': ema200
        }
        
        # Also return the dates and close prices for the same period
        dates = hist.index.strftime('%Y-%m-%d').tolist()
        close_prices = hist['Close'].squeeze().tolist() if isinstance(hist['Close'], pd.DataFrame) else hist['Close'].tolist()
        
        moving_averages_data = {
            'dates': dates,
            'close_prices': close_prices,
            'sma': sma,
            'ema': ema
        }
        
        logger.info(f"Successfully fetched moving averages for {symbol}")
        return moving_averages_data
    except Exception as e:
        logger.error(f"Error in get_moving_averages for {symbol}: {e}")
        return None

def prepare_data_for_models(stock_symbol, lookback_period):
    try:
        logger.info(f"Preparing data for models for {stock_symbol}")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5 * 365)
        df_full_history = yf.download(stock_symbol, start=start_date, end=end_date)
        
        if df_full_history.empty:
            logger.warning(f"No data downloaded for {stock_symbol}")
            return None, None, None, None, None, None, None, None
        
        df_close = df_full_history[['Close']]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_close)
        
        X, y = [], []
        for i in range(lookback_period, len(scaled_data)):
            X.append(scaled_data[i - lookback_period:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        if len(X) < 2:
            logger.warning(f"Not enough data points after applying lookback period of {lookback_period}")
            return None, None, None, None, None, None, None, None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        logger.info(f"Successfully prepared data for {stock_symbol}. Train size: {len(X_train)}, Test size: {len(X_test)}")
        return df_full_history, scaler, X_train, X_test, y_train, y_test, X_train_lstm, X_test_lstm
    except Exception as e:
        logger.error(f"Error in prepare_data_for_models for {stock_symbol}: {e}")
        return None, None, None, None, None, None, None, None

def train_lstm_model(X_train_lstm, y_train, lookback_period):
    try:
        logger.info("Training LSTM model")
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(lookback_period, 1)),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train_lstm, y_train, epochs=5, batch_size=32, verbose=0)
        logger.info("Successfully trained LSTM model")
        return model
    except Exception as e:
        logger.error(f"Error training LSTM model: {e}")
        return None

def train_ann(X_train, y_train, lookback_period):
    try:
        logger.info("Training ANN model")
        model = Sequential([
            Dense(32, activation='relu', input_shape=(lookback_period,)),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
        logger.info("Successfully trained ANN model")
        return model
    except Exception as e:
        logger.error(f"Error training ANN model: {e}")
        return None

def train_svm(X_train, y_train):
    try:
        logger.info("Training SVM model")
        model = SVR(kernel='linear', C=1.0)
        model.fit(X_train, y_train)
        logger.info("Successfully trained SVM model")
        return model
    except Exception as e:
        logger.error(f"Error training SVM model: {e}")
        return None

def train_tree_models(X_train, y_train):
    try:
        logger.info("Training tree models")
        dtree = DecisionTreeRegressor(max_depth=5)
        dtree.fit(X_train, y_train)
        
        rf = RandomForestRegressor(n_estimators=10, max_depth=5, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        logger.info("Successfully trained tree models")
        return dtree, rf
    except Exception as e:
        logger.error(f"Error training tree models: {e}")
        return None, None

def train_all_models_for_symbol(symbol):
    """Train all prediction models for a given stock symbol and cache them."""
    logger.info(f"Training all models for {symbol}")
    
    # Check if already training to avoid duplicate training
    with cache_lock:
        if symbol in training_in_progress:
            logger.info(f"Already training models for {symbol}")
            return False
        training_in_progress.add(symbol)
    
    try:
        # Prepare data
        df_full_history, scaler, X_train, X_test, y_train, y_test, X_train_lstm, X_test_lstm = \
            prepare_data_for_models(symbol, LOOKBACK_PERIOD)
        
        if df_full_history is None:
            logger.warning(f"Failed to prepare data for {symbol}")
            return False
        
        # Train all models
        lstm_model = train_lstm_model(X_train_lstm, y_train, LOOKBACK_PERIOD)
        ann_model = train_ann(X_train, y_train, LOOKBACK_PERIOD)
        svm_model = train_svm(X_train, y_train)
        dtree_model, rf_model = train_tree_models(X_train, y_train)
        
        # Check if all models trained successfully
        if lstm_model is None or ann_model is None or svm_model is None or dtree_model is None or rf_model is None:
            logger.error(f"One or more models failed to train for {symbol}")
            return False
        
        # Cache the models and data
        with cache_lock:
            model_cache[symbol] = {
                'models': {
                    'lstm': lstm_model,
                    'ann': ann_model,
                    'svm': svm_model,
                    'dt': dtree_model,
                    'rf': rf_model
                },
                'data': {
                    'df_full_history': df_full_history,
                    'scaler': scaler,
                    'X_test': X_test,
                    'y_test': y_test,
                    'X_test_lstm': X_test_lstm
                },
                'timestamp': datetime.now()
            }
            training_in_progress.discard(symbol)
        
        logger.info(f"Successfully trained and cached all models for {symbol}")
        return True
    except Exception as e:
        logger.error(f"Error training models for {symbol}: {e}")
        with cache_lock:
            training_in_progress.discard(symbol)
        return False

def handle_errors(f):
    """Decorator to handle errors in API endpoints."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {e}")
            return jsonify({"error": str(e)}), 500
    return wrapper

def get_predictions(symbol, model_type):
    try:
        logger.info(f"Getting predictions for {symbol} using {model_type} model")
        
        # Validate symbol
        if not symbol or symbol.strip() == "" or symbol == ".NS" or len(symbol) < 4:
            logger.error(f"Invalid symbol received: '{symbol}'")
            return None
        
        # Check if models are cached for this symbol
        with cache_lock:
            if symbol not in model_cache:
                logger.info(f"No cached models found for {symbol}, training now...")
                # Release lock before training to avoid blocking other requests
                cache_lock.release()
                try:
                    train_all_models_for_symbol(symbol)
                finally:
                    cache_lock.acquire()
            
            if symbol not in model_cache:
                logger.error(f"Failed to train models for {symbol}")
                return None
            
            cached_data = model_cache[symbol]
            models = cached_data['models']
            data = cached_data['data']
            model = models.get(model_type)
            
            if model is None:
                logger.error(f"Model {model_type} not available for {symbol}")
                return None
        
        # Use the cached model to make predictions
        if model_type == 'lstm':
            predicted = model.predict(data['X_test_lstm'])
        else:
            predicted = model.predict(data['X_test'])
        
        # Inverse transform the predictions
        predicted = data['scaler'].inverse_transform(predicted.reshape(-1, 1))
        actual = data['scaler'].inverse_transform(data['y_test'].reshape(-1, 1))
        
        # Generate future dates for prediction (next 30 days)
        last_date = data['df_full_history'].index[-1]
        future_dates = [(last_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 31)]
        
        # Calculate metrics
        mae = np.mean(np.abs(predicted - actual))
        mse = np.mean((predicted - actual) ** 2)
        rmse = np.sqrt(mse)
        
        # Calculate R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Convert DataFrame to Series if needed before calling tolist()
        actual_prices = data['df_full_history']['Close'].squeeze().tolist() if isinstance(data['df_full_history']['Close'], pd.DataFrame) else data['df_full_history']['Close'].tolist()
        
        # Convert numpy float32 to Python float for JSON serialization
        predicted_prices = [None] * (len(data['df_full_history']) - len(predicted)) + [float(p) for p in predicted.flatten()]
        future_predictions = [float(predicted[-1][0])] * 30
        
        # Create prediction data structure
        prediction_data = {
            'dates': data['df_full_history'].index.strftime('%Y-%m-%d').tolist() + future_dates,
            'actualPrices': actual_prices,
            'predictedPrices': predicted_prices,
            'futurePredictions': future_predictions,
            'metrics': {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'r2': float(r2)
            }
        }
        
        logger.info(f"Successfully generated predictions for {symbol} using {model_type}")
        return prediction_data
        
    except Exception as e:
        logger.error(f"Error in get_predictions for {symbol} with model {model_type}: {e}")
        return None

# Function to create a plot
def create_plot(hist_data, predictions=None):
    plt.figure(figsize=(12, 6))
    plt.plot(hist_data.index, hist_data['Close'], label='Historical Price')
    
    if predictions:
        # Convert string dates to datetime objects
        pred_dates = [datetime.strptime(d, '%Y-%m-%d') for d in predictions['dates']]
        plt.plot(pred_dates, predictions['prices'], 'r--', label='Predicted Price')
    
    plt.title('Stock Price History and Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert to base64 string
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Clear any existing error messages when first loading the page
    if request.method == 'GET':
        session.pop('_flashes', None)
    
    error = None
    if request.method == 'POST':
        login_input = request.form.get('login_input', '').strip()
        password = request.form.get('password', '').strip()
        
        # Basic validation
        if not login_input or not password:
            error = 'Username/Email and password are required'
            return render_template('login.html', error=error)
        
        # Check if user exists and password is correct
        username_key, user_data = verify_user(login_input, password)
        if username_key and user_data:
            session['logged_in'] = True
            session['username'] = username_key  # This is the lowercase username (key)
            session['display_name'] = user_data.get('display_name', username_key)  # Use the stored display_name or fallback to username_key
            
            logger.info(f"User {username_key} logged in successfully")
            flash('You have been logged in successfully!', 'success')
            return redirect(url_for('search'))
        else:
            error = 'Invalid credentials. Please try again.'
            logger.warning(f"Failed login attempt for username/email: {login_input}")
    
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    session.pop('display_name', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    # Clear any existing error messages when first loading the page
    if request.method == 'GET':
        session.pop('_flashes', None)
        
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Basic validation
        if not username or not email or not password or not confirm_password:
            flash('All fields are required.', 'danger')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')
        
        # Add user to JSON file
        success, message = add_user(username, email, password)
        
        if success:
            flash(message, 'success')
            # Clear session to ensure clean state for login
            session.pop('_flashes', None)
            return redirect(url_for('login'))
        else:
            flash(message, 'danger')
            return render_template('register.html')
    
    return render_template('register.html')

@app.route('/search')
@login_required
def search():
    return render_template('search.html')

@app.route('/overview')
@login_required
def overview_page():
    symbol = request.args.get('symbol')
    if not symbol:
        flash('Please enter a stock symbol', 'warning')
        return redirect(url_for('search'))
    
    # Normalize the symbol
    symbol = normalize_symbol(symbol)
    
    # Fetch all the data for the stock
    stock_data = get_stock_data(symbol)
    news_data = get_stock_news(symbol)
    financial_data = get_financial_data(symbol)
    price_data = get_price_history(symbol)
    
    # Format specific fields to 2 decimal places
    if stock_data:
        stock_data['peRatio'] = format_two_decimals(stock_data['peRatio'])
        stock_data['forwardPE'] = format_two_decimals(stock_data['forwardPE'])
        stock_data['priceToBook'] = format_two_decimals(stock_data['priceToBook'])
    
    # Check if any data is missing
    if not stock_data or not news_data or not financial_data or not price_data:
        flash('Some data could not be loaded for this stock.', 'warning')
    
    # Render the overview template with all the data
    return render_template(
        'overview.html',
        symbol=symbol,
        stock_data=stock_data,
        news_data=news_data,
        financial_data=financial_data,
        price_data=price_data,
        format_indian_currency=format_indian_currency,
        format_percentage=format_percentage,
        format_two_decimals=format_two_decimals
    )

@app.route('/overview/<symbol>')
@login_required
def overview(symbol):
    # Normalize the symbol
    symbol = normalize_symbol(symbol)
    return redirect(url_for('overview_page', symbol=symbol))

@app.route('/price-history')
@login_required
def price_history_page():
    symbol = request.args.get('symbol')
    if not symbol:
        flash('Please enter a stock symbol', 'warning')
        return redirect(url_for('search'))
    
    # Normalize the symbol
    symbol = normalize_symbol(symbol)
    
    price_data = get_price_history(symbol)
    return render_template(
        'price_history.html', 
        symbol=symbol, 
        price_data=price_data,
        format_indian_currency=format_indian_currency,
        format_percentage=format_percentage
    )

@app.route('/news')
@login_required
def news_page():
    symbol = request.args.get('symbol')
    if not symbol:
        flash('Please enter a stock symbol', 'warning')
        return redirect(url_for('search'))
    
    # Normalize the symbol
    symbol = normalize_symbol(symbol)
    
    news_data = get_stock_news(symbol)
    return render_template('news.html', symbol=symbol, news_data=news_data)

@app.route('/financials')
@login_required
def financials_page():
    symbol = request.args.get('symbol')
    if not symbol:
        flash('Please enter a stock symbol', 'warning')
        return redirect(url_for('search'))
    
    # Normalize the symbol
    symbol = normalize_symbol(symbol)
    
    financial_data = get_financial_data(symbol)
    return render_template(
        'financials.html', 
        symbol=symbol, 
        financial_data=financial_data,
        format_indian_currency=format_indian_currency,
        format_two_decimals=format_two_decimals
    )

@app.route('/price-prediction')
@login_required
def price_prediction_page():
    symbol = request.args.get('symbol')
    if not symbol:
        flash('Please enter a stock symbol', 'warning')
        return redirect(url_for('search'))
    
    # Normalize the symbol
    symbol = normalize_symbol(symbol)
    
    # Get prediction data
    prediction_data = get_predictions(symbol, 'lstm')
    return render_template(
        'price_prediction.html', 
        symbol=symbol, 
        prediction_data=prediction_data,
        format_indian_currency=format_indian_currency,
        format_percentage=format_percentage
    )

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/copyright')
def copyright():
    return render_template('copyright.html')

@app.route('/market-indices')
def market_indices():
    return render_template('market_indices.html')

@app.route('/api/market-indices')
def api_market_indices():
    """API endpoint to get current market indices data."""
    return jsonify({
        'nifty': nifty_data,
        'sensex': sensex_data
    })

# --- API Endpoints ---
@app.route('/api/stock/<symbol>', methods=['GET'])
@handle_errors
def api_stock(symbol):
    # Normalize the symbol
    symbol = normalize_symbol(symbol)
    
    logger.info(f"API request for stock data: {symbol}")
    stock_data = get_stock_data(symbol)
    if stock_data is None:
        logger.error(f"Stock data not found for {symbol}")
        return jsonify({"error": "Stock data not found"}), 404
    return jsonify(stock_data)

@app.route('/api/news/<symbol>', methods=['GET'])
@handle_errors
def api_news(symbol):
    # Normalize the symbol
    symbol = normalize_symbol(symbol)
    
    logger.info(f"API request for news: {symbol}")
    news_data = get_stock_news(symbol)
    return jsonify(news_data)

@app.route('/api/financials/<symbol>', methods=['GET'])
@handle_errors
def api_financials(symbol):
    # Normalize the symbol
    symbol = normalize_symbol(symbol)
    
    logger.info(f"API request for financials: {symbol}")
    financial_data = get_financial_data(symbol)
    return jsonify(financial_data)

@app.route('/api/price-history/<symbol>', methods=['GET'])
@handle_errors
def api_price_history(symbol):
    # Normalize the symbol
    symbol = normalize_symbol(symbol)
    
    period = request.args.get('period', '1y')
    logger.info(f"API request for price history: {symbol} with period {period}")
    price_data = get_price_history(symbol, period)
    if price_data is None:
        logger.error(f"Price history not found for {symbol}")
        return jsonify({"error": "Price history not found"}), 404
    return jsonify(price_data)

@app.route('/api/moving-averages/<symbol>', methods=['GET'])
@handle_errors
def api_moving_averages(symbol):
    # Normalize the symbol
    symbol = normalize_symbol(symbol)
    
    start_date = request.args.get('start_date', '2025-01-01')
    logger.info(f"API request for moving averages: {symbol} from {start_date}")
    moving_averages_data = get_moving_averages(symbol, start_date)
    if moving_averages_data is None:
        logger.error(f"Moving averages data not found for {symbol}")
        return jsonify({"error": "Moving averages data not found"}), 404
    return jsonify(moving_averages_data)

@app.route('/api/prediction/<symbol>', methods=['GET'])
@handle_errors
def api_prediction(symbol):
    # Normalize the symbol
    symbol = normalize_symbol(symbol)
    
    model = request.args.get('model', 'lstm')
    logger.info(f"API request for prediction: {symbol} using {model} model")
    
    # Validate symbol
    if not symbol or symbol.strip() == "" or symbol == ".NS" or len(symbol) < 4:
        logger.error(f"Invalid symbol received: '{symbol}'")
        return jsonify({"error": "Invalid stock symbol provided"}), 400
    
    # Validate model type
    valid_models = ['lstm', 'ann', 'svm', 'dt', 'rf']
    if model not in valid_models:
        logger.error(f"Invalid model type: {model}")
        return jsonify({
            "error": f"Invalid model type. Valid models are: {', '.join(valid_models)}"
        }), 400
    
    # Check if we're already training models for this symbol
    with cache_lock:
        is_training = symbol in training_in_progress
    
    if is_training:
        return jsonify({
            "status": "training",
            "message": f"Models are currently being trained for {symbol}. Please try again in a few moments."
        }), 202
    
    prediction_data = get_predictions(symbol, model)
    if prediction_data is None:
        logger.error(f"Failed to generate prediction for '{symbol}' using {model}")
        return jsonify({"error": "Prediction data not found"}), 404
    
    return jsonify(prediction_data)

# Background thread for running scheduled tasks
def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)

# Initialize the application
def initialize_app():
    logger.info("Application initialized. Models will be trained on-demand.")
    
    # Create users.json file if it doesn't exist
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump({}, f)
        logger.info(f"Created empty {USERS_FILE} file")
    
    # Check if admin user exists, create if not
    users = load_users()
    if 'admin' not in users:
        hashed_password = generate_password_hash('password')
        users['admin'] = {
            'email': 'admin@example.com',
            'password': hashed_password,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'display_name': 'Admin'
        }
        save_users(users)
        logger.info("Created default admin user (admin/password)")
    
    # Initialize market indices data
    update_market_indices()
    
    # Schedule market indices update every 5 seconds
    schedule.every(5).seconds.do(scheduled_update)
    
    # Start the background thread
    thread = threading.Thread(target=run_schedule)
    thread.daemon = True
    thread.start()
    
    logger.info("Market indices update scheduled every 5 seconds")

if __name__ == '__main__':
    initialize_app()
    app.run(debug=True)