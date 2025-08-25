import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
import warnings
import os
import sys
import traceback
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define SBERTTransformer class BEFORE loading the model
try:
    from sentence_transformers import SentenceTransformer


    class SBERTTransformer:
        def __init__(self, model_name='all-MiniLM-L6-v2'):
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"SBERT model {model_name} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load SBERT model: {str(e)}")
                raise

        def transform(self, sentences):
            try:
                if not sentences:
                    return np.array([])
                return self.model.encode(sentences)
            except Exception as e:
                logger.error(f"Error in SBERT transform: {str(e)}")
                raise

        def fit(self, X, y=None):
            return self


    SBERT_AVAILABLE = True
    logger.info("sentence-transformers available")
except ImportError:
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")
    SBERT_AVAILABLE = False


    class SBERTTransformer:
        def __init__(self, model_name='all-MiniLM-L6-v2'):
            self.model_name = model_name
            logger.warning(f"SBERTTransformer created but sentence-transformers not available")

        def transform(self, sentences):
            raise ImportError("sentence-transformers not installed")

        def fit(self, X, y=None):
            return self


class EnhancedSwingTradingSystem:
    """Enhanced Swing Trading System for Indian Markets with Budget & Risk Management"""

    def __init__(self, model_path="D:/Python_files/models/sentiment_pipeline.joblib", news_api_key=None):
        """Initialize the swing trading system with comprehensive error handling"""
        try:
            self.sentiment_pipeline = None
            self.vectorizer = None
            self.model = None
            self.label_encoder = None
            self.news_api_key = news_api_key or os.getenv("NEWS_API_KEY") or "dd33ebe105ea4b02a3b7e77bc4a93d01"

            # Model status tracking
            self.model_loaded = False
            self.model_type = "None"

            # Trading parameters with validation
            self.swing_trading_params = {
                'min_holding_period': 3,  # days
                'max_holding_period': 30,  # days
                'risk_per_trade': 0.02,  # 2% risk per trade
                'max_portfolio_risk': 0.10,  # 10% max portfolio risk
                'profit_target_multiplier': 2.5,  # Risk-reward ratio
            }

            # Validate trading parameters
            self._validate_trading_params()

            if not self.news_api_key:
                logger.warning("NEWS_API_KEY not provided. Using sample news for sentiment analysis.")
            else:
                logger.info("News API key available. Will fetch real news articles.")

            # Load sentiment model
            self.load_sbert_model(model_path)

            # Initialize comprehensive stock database
            self.initialize_stock_database()

            logger.info("EnhancedSwingTradingSystem initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing EnhancedSwingTradingSystem: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _validate_trading_params(self):
        """Validate trading parameters"""
        try:
            required_params = ['min_holding_period', 'max_holding_period', 'risk_per_trade',
                               'max_portfolio_risk', 'profit_target_multiplier']

            for param in required_params:
                if param not in self.swing_trading_params:
                    raise ValueError(f"Missing required trading parameter: {param}")

                value = self.swing_trading_params[param]
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(f"Invalid trading parameter {param}: {value}")

            # Additional validation
            if self.swing_trading_params['min_holding_period'] >= self.swing_trading_params['max_holding_period']:
                raise ValueError("min_holding_period must be less than max_holding_period")

            if self.swing_trading_params['risk_per_trade'] > 0.1:  # 10% max risk per trade
                raise ValueError("risk_per_trade cannot exceed 10%")

            logger.info("Trading parameters validated successfully")

        except Exception as e:
            logger.error(f"Error validating trading parameters: {str(e)}")
            raise

    def initialize_stock_database(self):
        """Initialize comprehensive Indian stock database (BSE + NSE) with error handling"""
        try:
            self.indian_stocks = {
                # NIFTY 50 Stocks
                "RELIANCE": {"name": "Reliance Industries", "sector": "Oil & Gas"},
                "TCS": {"name": "Tata Consultancy Services", "sector": "Information Technology"},
                "HDFCBANK": {"name": "HDFC Bank", "sector": "Banking"},
                "INFY": {"name": "Infosys", "sector": "Information Technology"},
                "HINDUNILVR": {"name": "Hindustan Unilever", "sector": "Consumer Goods"},
                "ICICIBANK": {"name": "ICICI Bank", "sector": "Banking"},
                "KOTAKBANK": {"name": "Kotak Mahindra Bank", "sector": "Banking"},
                "BAJFINANCE": {"name": "Bajaj Finance", "sector": "Financial Services"},
                "LT": {"name": "Larsen & Toubro", "sector": "Construction"},
                "SBIN": {"name": "State Bank of India", "sector": "Banking"},
                "BHARTIARTL": {"name": "Bharti Airtel", "sector": "Telecommunications"},
                "ASIANPAINT": {"name": "Asian Paints", "sector": "Consumer Goods"},
                "MARUTI": {"name": "Maruti Suzuki", "sector": "Automobile"},
                "TITAN": {"name": "Titan Company", "sector": "Consumer Goods"},
                "SUNPHARMA": {"name": "Sun Pharmaceutical", "sector": "Pharmaceuticals"},
                "ULTRACEMCO": {"name": "UltraTech Cement", "sector": "Cement"},
                "NESTLEIND": {"name": "Nestle India", "sector": "Consumer Goods"},
                "HCLTECH": {"name": "HCL Technologies", "sector": "Information Technology"},
                "AXISBANK": {"name": "Axis Bank", "sector": "Banking"},
                "WIPRO": {"name": "Wipro", "sector": "Information Technology"},
                "NTPC": {"name": "NTPC", "sector": "Power"},
                "POWERGRID": {"name": "Power Grid Corporation", "sector": "Power"},
                "ONGC": {"name": "Oil & Natural Gas Corporation", "sector": "Oil & Gas"},
                "TECHM": {"name": "Tech Mahindra", "sector": "Information Technology"},
                "TATASTEEL": {"name": "Tata Steel", "sector": "Steel"},
                "ADANIENT": {"name": "Adani Enterprises", "sector": "Conglomerate"},
                "COALINDIA": {"name": "Coal India", "sector": "Mining"},
                "HINDALCO": {"name": "Hindalco Industries", "sector": "Metals"},
                "JSWSTEEL": {"name": "JSW Steel", "sector": "Steel"},
                "BAJAJ-AUTO": {"name": "Bajaj Auto", "sector": "Automobile"},
                "M&M": {"name": "Mahindra & Mahindra", "sector": "Automobile"},
                "HEROMOTOCO": {"name": "Hero MotoCorp", "sector": "Automobile"},
                "GRASIM": {"name": "Grasim Industries", "sector": "Cement"},
                "SHREECEM": {"name": "Shree Cement", "sector": "Cement"},
                "EICHERMOT": {"name": "Eicher Motors", "sector": "Automobile"},
                "UPL": {"name": "UPL Limited", "sector": "Chemicals"},
                "BPCL": {"name": "Bharat Petroleum", "sector": "Oil & Gas"},
                "DIVISLAB": {"name": "Divi's Laboratories", "sector": "Pharmaceuticals"},
                "DRREDDY": {"name": "Dr. Reddy's Laboratories", "sector": "Pharmaceuticals"},
                "CIPLA": {"name": "Cipla", "sector": "Pharmaceuticals"},
                "BRITANNIA": {"name": "Britannia Industries", "sector": "Consumer Goods"},
                "TATACONSUM": {"name": "Tata Consumer Products", "sector": "Consumer Goods"},
                "IOC": {"name": "Indian Oil Corporation", "sector": "Oil & Gas"},
                "APOLLOHOSP": {"name": "Apollo Hospitals", "sector": "Healthcare"},
                "BAJAJFINSV": {"name": "Bajaj Finserv", "sector": "Financial Services"},
                "HDFCLIFE": {"name": "HDFC Life Insurance", "sector": "Insurance"},
                "SBILIFE": {"name": "SBI Life Insurance", "sector": "Insurance"},
                "INDUSINDBK": {"name": "IndusInd Bank", "sector": "Banking"},
                "ADANIPORTS": {"name": "Adani Ports", "sector": "Infrastructure"},
                "TATAMOTORS": {"name": "Tata Motors", "sector": "Automobile"},
                "ITC": {"name": "ITC Limited", "sector": "Consumer Goods"},

                # Additional Mid & Small Cap Stocks
                "GODREJCP": {"name": "Godrej Consumer Products", "sector": "Consumer Goods"},
                "COLPAL": {"name": "Colgate-Palmolive India", "sector": "Consumer Goods"},
                "PIDILITIND": {"name": "Pidilite Industries", "sector": "Chemicals"},
                "BAJAJHLDNG": {"name": "Bajaj Holdings", "sector": "Financial Services"},
                "MARICO": {"name": "Marico Limited", "sector": "Consumer Goods"},
                "DABUR": {"name": "Dabur India", "sector": "Consumer Goods"},
                "LUPIN": {"name": "Lupin Limited", "sector": "Pharmaceuticals"},
                "CADILAHC": {"name": "Cadila Healthcare", "sector": "Pharmaceuticals"},
                "BIOCON": {"name": "Biocon Limited", "sector": "Pharmaceuticals"},
                "ALKEM": {"name": "Alkem Laboratories", "sector": "Pharmaceuticals"},
                "TORNTPHARM": {"name": "Torrent Pharmaceuticals", "sector": "Pharmaceuticals"},
                "AUROPHARMA": {"name": "Aurobindo Pharma", "sector": "Pharmaceuticals"},
                "MOTHERSUMI": {"name": "Motherson Sumi Systems", "sector": "Automobile"},
                "BOSCHLTD": {"name": "Bosch Limited", "sector": "Automobile"},
                "EXIDEIND": {"name": "Exide Industries", "sector": "Automobile"},
                "ASHOKLEY": {"name": "Ashok Leyland", "sector": "Automobile"},
                "TVSMOTOR": {"name": "TVS Motor Company", "sector": "Automobile"},
                "BALKRISIND": {"name": "Balkrishna Industries", "sector": "Automobile"},
                "MRF": {"name": "MRF Limited", "sector": "Automobile"},
                "APOLLOTYRE": {"name": "Apollo Tyres", "sector": "Automobile"},
                "BHARATFORG": {"name": "Bharat Forge", "sector": "Automobile"},

                # Banking & Financial Services
                "FEDERALBNK": {"name": "Federal Bank", "sector": "Banking"},
                "BANDHANBNK": {"name": "Bandhan Bank", "sector": "Banking"},
                "IDFCFIRSTB": {"name": "IDFC First Bank", "sector": "Banking"},
                "RBLBANK": {"name": "RBL Bank", "sector": "Banking"},
                "YESBANK": {"name": "Yes Bank", "sector": "Banking"},
                "PNB": {"name": "Punjab National Bank", "sector": "Banking"},
                "BANKBARODA": {"name": "Bank of Baroda", "sector": "Banking"},
                "CANBK": {"name": "Canara Bank", "sector": "Banking"},
                "UNIONBANK": {"name": "Union Bank of India", "sector": "Banking"},
                "CHOLAFIN": {"name": "Cholamandalam Investment", "sector": "Financial Services"},
                "LICHSGFIN": {"name": "LIC Housing Finance", "sector": "Financial Services"},
                "MANAPPURAM": {"name": "Manappuram Finance", "sector": "Financial Services"},
                "MMFIN": {"name": "Mahindra & Mahindra Financial", "sector": "Financial Services"},
                "SRTRANSFIN": {"name": "Shriram Transport Finance", "sector": "Financial Services"},

                # Information Technology
                "MINDTREE": {"name": "Mindtree Limited", "sector": "Information Technology"},
                "LTTS": {"name": "L&T Technology Services", "sector": "Information Technology"},
                "PERSISTENT": {"name": "Persistent Systems", "sector": "Information Technology"},
                "CYIENT": {"name": "Cyient Limited", "sector": "Information Technology"},
                "NIITTECH": {"name": "NIIT Technologies", "sector": "Information Technology"},
                "ROLTA": {"name": "Rolta India", "sector": "Information Technology"},
                "HEXATECHNO": {"name": "Hexa Technologies", "sector": "Information Technology"},
                "COFORGE": {"name": "Coforge Limited", "sector": "Information Technology"},

                # Additional sectors continue...
                "DMART": {"name": "Avenue Supermarts", "sector": "Retail"},
                "TRENT": {"name": "Trent Limited", "sector": "Retail"},
                "PAGEIND": {"name": "Page Industries", "sector": "Textiles"},
                "RAYMOND": {"name": "Raymond Limited", "sector": "Textiles"},
                "VBL": {"name": "Varun Beverages", "sector": "Consumer Goods"},
                "EMAMILTD": {"name": "Emami Limited", "sector": "Consumer Goods"},
                "JUBLFOOD": {"name": "Jubilant FoodWorks", "sector": "Consumer Goods"},
            }

            if not self.indian_stocks:
                raise ValueError("Stock database initialization failed - empty database")

            logger.info(f"Initialized database with {len(self.indian_stocks)} Indian stocks")

        except Exception as e:
            logger.error(f"Error initializing stock database: {str(e)}")
            # Fallback to minimal database
            self.indian_stocks = {
                "RELIANCE": {"name": "Reliance Industries", "sector": "Oil & Gas"},
                "TCS": {"name": "Tata Consultancy Services", "sector": "Information Technology"},
                "HDFCBANK": {"name": "HDFC Bank", "sector": "Banking"},
            }
            logger.warning(f"Using fallback database with {len(self.indian_stocks)} stocks")

    def get_all_stock_symbols(self):
        """Get all stock symbols for analysis with error handling"""
        try:
            if not self.indian_stocks:
                raise ValueError("Stock database is empty")
            return list(self.indian_stocks.keys())
        except Exception as e:
            logger.error(f"Error getting stock symbols: {str(e)}")
            return ["RELIANCE", "TCS", "HDFCBANK"]  # Fallback symbols

    def get_stock_info_from_db(self, symbol):
        """Get stock information from internal database with error handling"""
        try:
            if not symbol:
                raise ValueError("Empty symbol provided")

            base_symbol = str(symbol).split('.')[0].upper().strip()
            if not base_symbol:
                raise ValueError("Invalid symbol format")

            return self.indian_stocks.get(base_symbol, {"name": symbol, "sector": "Unknown"})
        except Exception as e:
            logger.error(f"Error getting stock info for {symbol}: {str(e)}")
            return {"name": str(symbol), "sector": "Unknown"}

    def load_sbert_model(self, model_path):
        """Load trained SBERT sentiment model with comprehensive error handling"""
        try:
            if not SBERT_AVAILABLE:
                logger.warning("sentence-transformers not available, using TextBlob fallback")
                self.model_type = "TextBlob"
                return

            if not model_path:
                logger.warning("No model path provided, using TextBlob fallback")
                self.model_type = "TextBlob"
                return

            if not os.path.exists(model_path):
                logger.warning(f"SBERT model not found at {model_path}")
                logger.info("Using TextBlob as fallback for sentiment analysis")
                self.model_type = "TextBlob"
                return

            logger.info(f"Loading SBERT sentiment model from {model_path}...")

            # Load with timeout protection
            self.sentiment_pipeline = joblib.load(model_path)

            if not isinstance(self.sentiment_pipeline, dict):
                raise ValueError("Invalid model format - expected dictionary")

            self.vectorizer = self.sentiment_pipeline.get("vectorizer")
            self.model = self.sentiment_pipeline.get("model")
            self.label_encoder = self.sentiment_pipeline.get("label_encoder")

            if all([self.vectorizer, self.model, self.label_encoder]):
                # Validate model components
                if not hasattr(self.vectorizer, 'transform'):
                    raise ValueError("Invalid vectorizer - missing transform method")
                if not hasattr(self.model, 'predict'):
                    raise ValueError("Invalid model - missing predict method")
                if not hasattr(self.label_encoder, 'classes_'):
                    raise ValueError("Invalid label encoder - missing classes_")

                logger.info("SBERT sentiment model loaded successfully!")
                logger.info(f"Model classes: {list(self.label_encoder.classes_)}")
                self.model_loaded = True
                self.model_type = "SBERT + RandomForest"
            else:
                logger.warning("Model components incomplete, using TextBlob fallback")
                self.model_type = "TextBlob"
                self.sentiment_pipeline = None

        except Exception as e:
            logger.error(f"Error loading SBERT model: {str(e)}")
            logger.error(traceback.format_exc())
            logger.info("Using TextBlob as fallback for sentiment analysis")
            self.model_type = "TextBlob"
            self.sentiment_pipeline = None

    def get_sector_weights(self, sector):
        """Get dynamic weights based on sector for swing trading with error handling"""
        try:
            if not sector:
                logger.warning("Empty sector provided, using default weights")
                return 0.55, 0.45

            sector = str(sector).lower().strip()

            # Swing trading weights (balanced approach)
            tech_weight, sentiment_weight = 0.55, 0.45

            weights_map = {
                "technology": (0.45, 0.55),  # Tech more sentiment driven
                "information technology": (0.45, 0.55),
                "tech": (0.45, 0.55),
                "it": (0.45, 0.55),
                "financial": (0.60, 0.40),  # Finance more technical
                "financial services": (0.60, 0.40),
                "banking": (0.60, 0.40),
                "finance": (0.60, 0.40),
                "consumer staples": (0.65, 0.35),
                "staples": (0.65, 0.35),
                "consumer goods": (0.65, 0.35),
                "food & staples retailing": (0.65, 0.35),
                "energy": (0.55, 0.45),
                "oil & gas": (0.55, 0.45),
                "utilities": (0.70, 0.30),
                "electric": (0.70, 0.30),
                "power": (0.70, 0.30),
                "healthcare": (0.50, 0.50),
                "pharmaceuticals": (0.50, 0.50),
                "health care": (0.50, 0.50),
                "pharma": (0.50, 0.50),
                "consumer discretionary": (0.45, 0.55),
                "consumer cyclicals": (0.45, 0.55),
                "retail": (0.45, 0.55),
                "automobile": (0.45, 0.55),
                "auto": (0.45, 0.55),
            }

            for key, weights in weights_map.items():
                if key in sector:
                    tech_weight, sentiment_weight = weights
                    break

            # Validate weights
            if tech_weight + sentiment_weight != 1.0:
                logger.warning(f"Invalid weights for sector {sector}, using defaults")
                return 0.55, 0.45

            return tech_weight, sentiment_weight

        except Exception as e:
            logger.error(f"Error getting sector weights for {sector}: {str(e)}")
            return 0.55, 0.45  # Default weights

    def get_indian_stock_data(self, symbol, period="6mo"):
        """Get Indian stock data with extended period for swing trading and comprehensive error handling"""
        try:
            if not symbol:
                raise ValueError("Empty symbol provided")

            symbol = str(symbol).upper().replace(" ", "").replace(".", "")
            if not symbol:
                raise ValueError("Invalid symbol after cleaning")

            # Enhanced Indian stocks mapping
            symbol_mappings = {
                "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "INFY": "INFY.NS",
                "HDFCBANK": "HDFCBANK.NS", "BAJFINANCE": "BAJFINANCE.NS",
                "HINDUNILVR": "HINDUNILVR.NS", "ICICIBANK": "ICICIBANK.NS",
                "KOTAKBANK": "KOTAKBANK.NS", "SBIN": "SBIN.NS",
                "BHARTIARTL": "BHARTIARTL.NS", "LT": "LT.NS", "MARUTI": "MARUTI.NS",
                "ASIANPAINT": "ASIANPAINT.NS", "HCLTECH": "HCLTECH.NS",
                "TITAN": "TITAN.NS", "SUNPHARMA": "SUNPHARMA.NS",
                "NTPC": "NTPC.NS", "ONGC": "ONGC.NS", "ADANIENT": "ADANIENT.NS",
                "WIPRO": "WIPRO.NS", "TECHM": "TECHM.NS", "POWERGRID": "POWERGRID.NS",
            }

            symbols_to_try = [f"{symbol}.NS", f"{symbol}.BO", symbol]
            if symbol in symbol_mappings:
                symbols_to_try.insert(0, symbol_mappings[symbol])

            for sym in symbols_to_try:
                try:
                    logger.info(f"Trying to fetch data for {sym}")
                    ticker = yf.Ticker(sym)

                    # Add timeout and retry logic
                    data = ticker.history(period=period, timeout=30)

                    if data is None or data.empty:
                        logger.warning(f"No data returned for {sym}")
                        continue

                    if len(data) < 30:  # Need more data for swing trading
                        logger.warning(f"Insufficient data for {sym}: {len(data)} days")
                        continue

                    # Validate data quality
                    if data['Close'].isna().all():
                        logger.warning(f"All Close prices are NaN for {sym}")
                        continue

                    # Try to get info (optional, might fail for some stocks)
                    info = {}
                    try:
                        info = ticker.info
                    except Exception as info_error:
                        logger.warning(f"Could not fetch info for {sym}: {str(info_error)}")
                        info = {}

                    logger.info(f"Successfully fetched data for {sym}: {len(data)} days")
                    return data, info, sym

                except Exception as e:
                    logger.warning(f"Failed to fetch data for {sym}: {str(e)}")
                    continue

            logger.error(f"Failed to fetch data for all variations of {symbol}")
            return None, None, None

        except Exception as e:
            logger.error(f"Error in get_indian_stock_data for {symbol}: {str(e)}")
            return None, None, None

    def safe_rolling_calculation(self, data, window, operation='mean'):
        """Safely perform rolling calculations with error handling"""
        try:
            if data is None or data.empty:
                return pd.Series(dtype=float)

            if len(data) < window:
                return pd.Series([np.nan] * len(data), index=data.index)

            if operation == 'mean':
                return data.rolling(window=window, min_periods=1).mean()
            elif operation == 'std':
                return data.rolling(window=window, min_periods=1).std()
            elif operation == 'min':
                return data.rolling(window=window, min_periods=1).min()
            elif operation == 'max':
                return data.rolling(window=window, min_periods=1).max()
            else:
                logger.error(f"Unknown rolling operation: {operation}")
                return pd.Series([np.nan] * len(data), index=data.index)

        except Exception as e:
            logger.error(f"Error in safe_rolling_calculation: {str(e)}")
            return pd.Series([np.nan] * len(data), index=data.index if hasattr(data, 'index') else range(len(data)))

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands with comprehensive error handling"""
        try:
            if prices is None or prices.empty:
                empty_series = pd.Series(dtype=float)
                return empty_series, empty_series, empty_series

            if len(prices) < period:
                nan_series = pd.Series([np.nan] * len(prices), index=prices.index)
                return nan_series, nan_series, nan_series

            sma = self.safe_rolling_calculation(prices, period, 'mean')
            std = self.safe_rolling_calculation(prices, period, 'std')

            if sma.empty or std.empty:
                nan_series = pd.Series([np.nan] * len(prices), index=prices.index)
                return nan_series, nan_series, nan_series

            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)

            return upper_band, sma, lower_band

        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            nan_series = pd.Series([np.nan] * len(prices),
                                   index=prices.index if hasattr(prices, 'index') else range(len(prices)))
            return nan_series, nan_series, nan_series

    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator with error handling"""
        try:
            if any(x is None or x.empty for x in [high, low, close]):
                empty_series = pd.Series(dtype=float)
                return empty_series, empty_series

            if len(close) < k_period:
                nan_series = pd.Series([np.nan] * len(close), index=close.index)
                return nan_series, nan_series

            lowest_low = self.safe_rolling_calculation(low, k_period, 'min')
            highest_high = self.safe_rolling_calculation(high, k_period, 'max')

            if lowest_low.empty or highest_high.empty:
                nan_series = pd.Series([np.nan] * len(close), index=close.index)
                return nan_series, nan_series

            # Avoid division by zero
            denominator = highest_high - lowest_low
            denominator = denominator.replace(0, np.nan)

            k_percent = 100 * ((close - lowest_low) / denominator)
            d_percent = self.safe_rolling_calculation(k_percent, d_period, 'mean')

            return k_percent, d_percent

        except Exception as e:
            logger.error(f"Error calculating Stochastic: {str(e)}")
            nan_series = pd.Series([np.nan] * len(close),
                                   index=close.index if hasattr(close, 'index') else range(len(close)))
            return nan_series, nan_series

    def calculate_support_resistance(self, data, window=20):
        """Calculate support and resistance levels with error handling"""
        try:
            if data is None or data.empty:
                return None, None

            if 'High' not in data.columns or 'Low' not in data.columns:
                logger.error("Missing High/Low columns for support/resistance calculation")
                return None, None

            if len(data) < window:
                return data['Low'].min(), data['High'].max()

            highs = self.safe_rolling_calculation(data['High'], window, 'max')
            lows = self.safe_rolling_calculation(data['Low'], window, 'min')

            if highs.empty or lows.empty:
                return data['Low'].min(), data['High'].max()

            # Find significant levels
            resistance_levels = []
            support_levels = []

            for i in range(window, len(data)):
                try:
                    # Check if current high is a local maximum
                    if not pd.isna(highs.iloc[i]) and data['High'].iloc[i] == highs.iloc[i]:
                        resistance_levels.append(data['High'].iloc[i])

                    # Check if current low is a local minimum
                    if not pd.isna(lows.iloc[i]) and data['Low'].iloc[i] == lows.iloc[i]:
                        support_levels.append(data['Low'].iloc[i])
                except Exception as e:
                    logger.warning(f"Error processing level at index {i}: {str(e)}")
                    continue

            # Get most recent levels
            if len(resistance_levels) >= 3:
                current_resistance = max(resistance_levels[-3:])
            else:
                current_resistance = data['High'].max()

            if len(support_levels) >= 3:
                current_support = min(support_levels[-3:])
            else:
                current_support = data['Low'].min()

            return current_support, current_resistance

        except Exception as e:
            logger.error(f"Error calculating support/resistance: {str(e)}")
            try:
                return data['Low'].min(), data['High'].max()
            except:
                return None, None

    def calculate_volume_profile(self, data, bins=20):
        """Calculate Volume Profile with error handling"""
        try:
            if data is None or data.empty or 'Volume' not in data.columns:
                return None, None

            if 'High' not in data.columns or 'Low' not in data.columns:
                return None, None

            price_range = data['High'].max() - data['Low'].min()
            if price_range <= 0:
                return None, None

            bin_size = price_range / bins
            volume_profile = {}

            for i in range(len(data)):
                try:
                    price = (data['High'].iloc[i] + data['Low'].iloc[i]) / 2
                    volume = data['Volume'].iloc[i]

                    if pd.isna(price) or pd.isna(volume) or volume <= 0:
                        continue

                    bin_level = int((price - data['Low'].min()) / bin_size)
                    bin_level = min(bin_level, bins - 1)
                    bin_level = max(bin_level, 0)

                    price_level = data['Low'].min() + (bin_level * bin_size)

                    if price_level not in volume_profile:
                        volume_profile[price_level] = 0
                    volume_profile[price_level] += volume
                except Exception as e:
                    logger.warning(f"Error processing volume at index {i}: {str(e)}")
                    continue

            if not volume_profile:
                return None, None

            # Find Point of Control (POC) - highest volume level
            poc_price = max(volume_profile.keys(), key=lambda x: volume_profile[x])
            poc_volume = volume_profile[poc_price]

            return volume_profile, poc_price

        except Exception as e:
            logger.error(f"Error calculating volume profile: {str(e)}")
            return None, None

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI with comprehensive error handling"""
        try:
            if prices is None or prices.empty:
                return pd.Series(dtype=float)

            if len(prices) < period:
                return pd.Series([50] * len(prices), index=prices.index)

            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = self.safe_rolling_calculation(gain, period, 'mean')
            avg_loss = self.safe_rolling_calculation(loss, period, 'mean')

            if avg_gain.empty or avg_loss.empty:
                return pd.Series([50] * len(prices), index=prices.index)

            # Avoid division by zero
            avg_loss = avg_loss.replace(0, np.nan)
            rs = avg_gain / avg_loss

            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(50)  # Fill NaN with neutral RSI

            return rsi

        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series([50] * len(prices), index=prices.index if hasattr(prices, 'index') else range(len(prices)))

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD with error handling"""
        try:
            if prices is None or prices.empty:
                empty_series = pd.Series(dtype=float)
                return empty_series, empty_series, empty_series

            if len(prices) < slow:
                zeros = pd.Series([0] * len(prices), index=prices.index)
                return zeros, zeros, zeros

            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()

            if exp1.empty or exp2.empty:
                zeros = pd.Series([0] * len(prices), index=prices.index)
                return zeros, zeros, zeros

            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line

            return macd_line, signal_line, histogram

        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            zeros = pd.Series([0] * len(prices), index=prices.index if hasattr(prices, 'index') else range(len(prices)))
            return zeros, zeros, zeros

    def calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range with error handling"""
        try:
            if any(x is None or x.empty for x in [high, low, close]):
                return pd.Series(dtype=float)

            if len(close) < period:
                return pd.Series([np.nan] * len(close), index=close.index)

            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())

            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = self.safe_rolling_calculation(tr, period, 'mean')

            return atr

        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series([np.nan] * len(close), index=close.index if hasattr(close, 'index') else range(len(close)))

    def fetch_indian_news(self, symbol, num_articles=15):
        """Fetch news for Indian companies with error handling"""
        try:
            if not self.news_api_key:
                return None

            base_symbol = str(symbol).split('.')[0].upper()
            stock_info = self.get_stock_info_from_db(base_symbol)
            company_name = stock_info.get("name", base_symbol)

            url = f"https://newsapi.org/v2/everything?q={company_name}+India+stock&apiKey={self.news_api_key}&pageSize={num_articles}&language=en&sortBy=publishedAt"

            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = []
                for article in data.get('articles', []):
                    if article.get('title'):
                        articles.append(article['title'])
                return articles if articles else None
            else:
                logger.warning(f"News API returned status code: {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            logger.warning("News API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"News API request failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return None

    def get_sample_news(self, symbol):
        """Generate sample news for demonstration with error handling"""
        try:
            base_symbol = str(symbol).split('.')[0]
            stock_info = self.get_stock_info_from_db(base_symbol)
            company_name = stock_info.get("name", base_symbol)

            return [
                f"{company_name} reports strong quarterly earnings beating estimates",
                f"Analysts upgrade {company_name} target price citing strong fundamentals",
                f"{company_name} announces major expansion plans and new product launches",
                f"Regulatory approval boosts {company_name} market position",
                f"{company_name} forms strategic partnership with global leader",
                f"Market volatility creates buying opportunity in {company_name}",
                f"{company_name} invests heavily in R&D and digital transformation",
                f"Industry experts bullish on {company_name} long-term prospects",
                f"Competitive pressure intensifies for {company_name} in key markets",
                f"Strong domestic demand drives {company_name} revenue growth",
                f"{company_name} management provides optimistic guidance for next quarter",
                f"Foreign institutional investors increase stake in {company_name}",
                f"Technical breakout signals potential upside for {company_name}",
                f"{company_name} benefits from favorable government policy changes",
                f"Sector rotation favors {company_name} business model"
            ]
        except Exception as e:
            logger.error(f"Error generating sample news for {symbol}: {str(e)}")
            return [f"Market analysis for {symbol}", f"Investment opportunity in {symbol}"]

    def analyze_sentiment_with_sbert(self, articles):
        """Analyze sentiment using trained SBERT model with error handling"""
        try:
            if not articles or not self.model_loaded:
                return self.analyze_sentiment_with_textblob(articles)

            if not all([self.vectorizer, self.model, self.label_encoder]):
                logger.error("SBERT model components missing")
                return self.analyze_sentiment_with_textblob(articles)

            embeddings = self.vectorizer.transform(articles)

            if embeddings is None or len(embeddings) == 0:
                logger.error("Failed to generate embeddings")
                return self.analyze_sentiment_with_textblob(articles)

            predictions = self.model.predict(embeddings)
            probabilities = self.model.predict_proba(embeddings)

            sentiment_labels = self.label_encoder.inverse_transform(predictions)
            confidence_scores = np.max(probabilities, axis=1)

            return sentiment_labels.tolist(), confidence_scores.tolist()

        except Exception as e:
            logger.error(f"Error in SBERT sentiment analysis: {str(e)}")
            return self.analyze_sentiment_with_textblob(articles)

    def analyze_sentiment_with_textblob(self, articles):
        """Fallback sentiment analysis using TextBlob with error handling"""
        sentiments = []
        confidences = []

        if not articles:
            return sentiments, confidences

        for article in articles:
            try:
                if not article or not isinstance(article, str):
                    sentiments.append('neutral')
                    confidences.append(0.3)
                    continue

                blob = TextBlob(article)
                polarity = blob.sentiment.polarity

                if polarity > 0.1:
                    sentiments.append('positive')
                    confidences.append(min(abs(polarity), 0.8))
                elif polarity < -0.1:
                    sentiments.append('negative')
                    confidences.append(min(abs(polarity), 0.8))
                else:
                    sentiments.append('neutral')
                    confidences.append(0.5)
            except Exception as e:
                logger.warning(f"Error analyzing sentiment for article: {str(e)}")
                sentiments.append('neutral')
                confidences.append(0.3)

        return sentiments, confidences

    def analyze_news_sentiment(self, symbol, num_articles=15):
        """Main sentiment analysis function with comprehensive error handling"""
        try:
            articles = self.fetch_indian_news(symbol, num_articles)
            news_source = "Real news (NewsAPI)" if articles else "Sample news"

            if not articles:
                articles = self.get_sample_news(symbol)

            if not articles:
                logger.error(f"No articles available for {symbol}")
                return [], [], [], "No Analysis", "No Source"

            if self.model_loaded:
                sentiments, confidences = self.analyze_sentiment_with_sbert(articles)
                analysis_method = f"SBERT Model ({self.model_type})"
            else:
                sentiments, confidences = self.analyze_sentiment_with_textblob(articles)
                analysis_method = "TextBlob Fallback"

            return sentiments, articles, confidences, analysis_method, news_source

        except Exception as e:
            logger.error(f"Error in news sentiment analysis for {symbol}: {str(e)}")
            return [], [], [], "Error", "Error"

    def calculate_swing_trading_score(self, data, sentiment_data, sector):
        """Calculate comprehensive swing trading score with error handling"""
        try:
            tech_weight, sentiment_weight = self.get_sector_weights(sector)

            # Initialize components
            technical_score = 0
            sentiment_score = 50  # Default neutral sentiment

            if data is None or data.empty:
                logger.error("No data provided for scoring")
                return 0

            # ===== TECHNICAL ANALYSIS (Enhanced for Swing Trading) =====
            try:
                current_price = data['Close'].iloc[-1]
                if pd.isna(current_price) or current_price <= 0:
                    logger.error("Invalid current price")
                    return 0
            except Exception as e:
                logger.error(f"Error getting current price: {str(e)}")
                return 0

            # RSI Analysis (20 points)
            try:
                rsi = self.calculate_rsi(data['Close'])
                if not rsi.empty and not pd.isna(rsi.iloc[-1]):
                    current_rsi = rsi.iloc[-1]
                    if 30 <= current_rsi <= 70:  # Good for swing trading
                        technical_score += 20
                    elif current_rsi < 30:  # Oversold - potential reversal
                        technical_score += 15
                    elif current_rsi > 70:  # Overbought - potential reversal
                        technical_score += 10
            except Exception as e:
                logger.warning(f"Error calculating RSI: {str(e)}")

            # Bollinger Bands Analysis (15 points)
            try:
                bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data['Close'])
                if not bb_upper.empty and not any(pd.isna([bb_upper.iloc[-1], bb_lower.iloc[-1]])):
                    bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                    if 0.2 <= bb_position <= 0.8:  # Good swing trading zone
                        technical_score += 15
                    elif bb_position < 0.2:  # Near lower band - potential buy
                        technical_score += 12
                    elif bb_position > 0.8:  # Near upper band - potential sell
                        technical_score += 8
            except Exception as e:
                logger.warning(f"Error calculating Bollinger Bands: {str(e)}")

            # Stochastic Analysis (15 points)
            try:
                stoch_k, stoch_d = self.calculate_stochastic(data['High'], data['Low'], data['Close'])
                if not stoch_k.empty and not any(pd.isna([stoch_k.iloc[-1], stoch_d.iloc[-1]])):
                    k_val = stoch_k.iloc[-1]
                    d_val = stoch_d.iloc[-1]
                    if k_val > d_val and k_val < 80:  # Bullish crossover
                        technical_score += 15
                    elif 20 <= k_val <= 80:  # Good swing range
                        technical_score += 10
            except Exception as e:
                logger.warning(f"Error calculating Stochastic: {str(e)}")

            # MACD Analysis (15 points)
            try:
                macd_line, signal_line, histogram = self.calculate_macd(data['Close'])
                if not macd_line.empty and not any(pd.isna([macd_line.iloc[-1], signal_line.iloc[-1]])):
                    if macd_line.iloc[-1] > signal_line.iloc[-1]:  # Bullish
                        technical_score += 15
                    if len(histogram) > 1 and not any(pd.isna([histogram.iloc[-1], histogram.iloc[-2]])):
                        if histogram.iloc[-1] > histogram.iloc[-2]:  # Increasing momentum
                            technical_score += 5
            except Exception as e:
                logger.warning(f"Error calculating MACD: {str(e)}")

            # Volume Analysis (10 points)
            try:
                if 'Volume' in data.columns:
                    avg_volume = self.safe_rolling_calculation(data['Volume'], 20, 'mean').iloc[-1]
                    current_volume = data['Volume'].iloc[-1]
                    if not pd.isna(avg_volume) and not pd.isna(current_volume) and avg_volume > 0:
                        if current_volume > avg_volume * 1.2:  # Above average volume
                            technical_score += 10
                        elif current_volume > avg_volume:
                            technical_score += 5
            except Exception as e:
                logger.warning(f"Error calculating volume: {str(e)}")

            # Support/Resistance Analysis (10 points)
            try:
                support, resistance = self.calculate_support_resistance(data)
                if support and resistance and not any(pd.isna([support, resistance])):
                    distance_to_support = (current_price - support) / support
                    distance_to_resistance = (resistance - current_price) / current_price

                    if distance_to_support < 0.05:  # Near support
                        technical_score += 8
                    elif distance_to_resistance < 0.05:  # Near resistance
                        technical_score += 5
                    elif 0.05 <= distance_to_support <= 0.15:  # Good swing zone
                        technical_score += 10
            except Exception as e:
                logger.warning(f"Error calculating support/resistance: {str(e)}")

            # Moving Average Analysis (15 points)
            try:
                if len(data) >= 50:
                    ma_20 = self.safe_rolling_calculation(data['Close'], 20, 'mean').iloc[-1]
                    ma_50 = self.safe_rolling_calculation(data['Close'], 50, 'mean').iloc[-1]
                    if not any(pd.isna([ma_20, ma_50])):
                        if current_price > ma_20 > ma_50:  # Strong uptrend
                            technical_score += 15
                        elif current_price > ma_20:  # Above short-term MA
                            technical_score += 10
                        elif ma_20 > ma_50:  # MA alignment positive
                            technical_score += 5
            except Exception as e:
                logger.warning(f"Error calculating moving averages: {str(e)}")

            # Normalize technical score to 0-100
            technical_score = min(100, max(0, technical_score))

            # ===== SENTIMENT ANALYSIS =====
            try:
                if sentiment_data and len(sentiment_data) >= 3:
                    sentiments, _, confidences, _, _ = sentiment_data
                    if sentiments and confidences:
                        sentiment_value = 0
                        total_weight = 0

                        for sentiment, confidence in zip(sentiments, confidences):
                            weight = confidence if not pd.isna(confidence) else 0.5
                            if sentiment == 'positive':
                                sentiment_value += weight
                            elif sentiment == 'negative':
                                sentiment_value -= weight
                            total_weight += weight

                        if total_weight > 0:
                            normalized_sentiment = sentiment_value / total_weight
                            sentiment_score = 50 + (normalized_sentiment * 50)
                        else:
                            sentiment_score = 50
                    else:
                        sentiment_score = 50
                else:
                    sentiment_score = 50
            except Exception as e:
                logger.warning(f"Error calculating sentiment score: {str(e)}")
                sentiment_score = 50

            # ===== COMBINE SCORES =====
            sentiment_score = min(100, max(0, sentiment_score))
            final_score = (technical_score * tech_weight) + (sentiment_score * sentiment_weight)
            final_score = min(100, max(0, final_score))

            return final_score

        except Exception as e:
            logger.error(f"Error calculating swing trading score: {str(e)}")
            return 0

    def calculate_risk_metrics(self, data):
        """Calculate risk management metrics with comprehensive error handling"""
        default_metrics = {
            'volatility': 0.3,
            'var_95': -0.05,
            'max_drawdown': -0.2,
            'sharpe_ratio': 0,
            'atr': 0,
            'risk_level': 'HIGH'
        }

        try:
            if data is None or data.empty or 'Close' not in data.columns:
                logger.error("Invalid data for risk metrics calculation")
                return default_metrics

            returns = data['Close'].pct_change().dropna()

            if returns.empty or len(returns) < 2:
                logger.warning("Insufficient returns data for risk metrics")
                return default_metrics

            # Volatility (annualized)
            try:
                volatility = returns.std() * np.sqrt(252)
                if pd.isna(volatility) or volatility < 0:
                    volatility = 0.3
            except Exception:
                volatility = 0.3

            # Value at Risk (95% confidence)
            try:
                var_95 = np.percentile(returns.dropna(), 5)
                if pd.isna(var_95):
                    var_95 = -0.05
            except Exception:
                var_95 = -0.05

            # Maximum Drawdown
            try:
                rolling_max = data['Close'].expanding().max()
                drawdown = (data['Close'] - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
                if pd.isna(max_drawdown):
                    max_drawdown = -0.2
            except Exception:
                max_drawdown = -0.2

            # Sharpe Ratio (assuming 6% risk-free rate)
            try:
                risk_free_rate = 0.06
                excess_returns = returns.mean() * 252 - risk_free_rate
                sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
                if pd.isna(sharpe_ratio):
                    sharpe_ratio = 0
            except Exception:
                sharpe_ratio = 0

            # ATR for position sizing
            try:
                atr = self.calculate_atr(data['High'], data['Low'], data['Close'])
                current_atr = atr.iloc[-1] if not atr.empty and not pd.isna(atr.iloc[-1]) else data['Close'].iloc[
                                                                                                   -1] * 0.02
            except Exception:
                current_atr = data['Close'].iloc[-1] * 0.02 if not data['Close'].empty else 0

            # Risk level determination
            try:
                if volatility > 0.4:
                    risk_level = 'HIGH'
                elif volatility > 0.25:
                    risk_level = 'MEDIUM'
                else:
                    risk_level = 'LOW'
            except Exception:
                risk_level = 'HIGH'

            return {
                'volatility': volatility,
                'var_95': var_95,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'atr': current_atr,
                'risk_level': risk_level
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return default_metrics

    def generate_trading_plan(self, data, score, risk_metrics):
        """Generate complete trading plan with comprehensive error handling"""
        default_plan = {
            'entry_signal': "HOLD/WATCH",
            'entry_strategy': "Wait for clearer signals",
            'position_size_multiplier': 0.5,
            'stop_loss': 0,
            'targets': {'target_1': 0, 'target_2': 0, 'target_3': 0},
            'support': 0,
            'resistance': 0,
            'holding_period': f"{self.swing_trading_params['min_holding_period']}-{self.swing_trading_params['max_holding_period']} days"
        }

        try:
            if data is None or data.empty or 'Close' not in data.columns:
                logger.error("Invalid data for trading plan")
                return default_plan

            current_price = data['Close'].iloc[-1]
            if pd.isna(current_price) or current_price <= 0:
                logger.error("Invalid current price for trading plan")
                return default_plan

            atr = risk_metrics.get('atr', current_price * 0.02)
            if pd.isna(atr) or atr <= 0:
                atr = current_price * 0.02

            # Entry Strategy
            try:
                if score >= 75:
                    entry_signal = "STRONG BUY"
                    entry_strategy = "Enter aggressively on any dip"
                elif score >= 60:
                    entry_signal = "BUY"
                    entry_strategy = "Enter on pullbacks or breakouts"
                elif score >= 45:
                    entry_signal = "HOLD/WATCH"
                    entry_strategy = "Wait for clearer signals"
                elif score >= 30:
                    entry_signal = "SELL"
                    entry_strategy = "Exit longs, consider shorts"
                else:
                    entry_signal = "STRONG SELL"
                    entry_strategy = "Exit all positions"
            except Exception:
                entry_signal = "HOLD/WATCH"
                entry_strategy = "Wait for clearer signals"

            # Position Sizing (based on 2% risk per trade)
            try:
                risk_per_trade = self.swing_trading_params['risk_per_trade']
                stop_loss_distance = atr * 2  # 2 ATR stop loss
                position_size_multiplier = risk_per_trade / (stop_loss_distance / current_price)
                position_size_multiplier = min(position_size_multiplier, 2.0)  # Cap at 2x
            except Exception:
                position_size_multiplier = 0.5

            # Price Targets
            try:
                stop_loss_distance = atr * 2
                stop_loss = max(current_price - stop_loss_distance, 0)
                target_1 = current_price + (stop_loss_distance * 1.5)  # 1.5:1 RR
                target_2 = current_price + (stop_loss_distance * 2.5)  # 2.5:1 RR
                target_3 = current_price + (stop_loss_distance * 4.0)  # 4:1 RR
            except Exception:
                stop_loss = current_price * 0.95
                target_1 = current_price * 1.05
                target_2 = current_price * 1.10
                target_3 = current_price * 1.15

            # Support and Resistance
            try:
                support, resistance = self.calculate_support_resistance(data)
                if not support or pd.isna(support):
                    support = current_price * 0.95
                if not resistance or pd.isna(resistance):
                    resistance = current_price * 1.05
            except Exception:
                support = current_price * 0.95
                resistance = current_price * 1.05

            return {
                'entry_signal': entry_signal,
                'entry_strategy': entry_strategy,
                'position_size_multiplier': position_size_multiplier,
                'stop_loss': stop_loss,
                'targets': {
                    'target_1': target_1,
                    'target_2': target_2,
                    'target_3': target_3
                },
                'support': support,
                'resistance': resistance,
                'holding_period': f"{self.swing_trading_params['min_holding_period']}-{self.swing_trading_params['max_holding_period']} days"
            }

        except Exception as e:
            logger.error(f"Error generating trading plan: {str(e)}")
            return default_plan

    def analyze_swing_trading_stock(self, symbol, period="6mo"):
        """Comprehensive swing trading analysis for a single stock with full error handling"""
        try:
            if not symbol:
                logger.error("Empty symbol provided")
                return None

            logger.info(f"Starting analysis for {symbol}")

            # Get stock data
            data, info, final_symbol = self.get_indian_stock_data(symbol, period)
            if data is None or data.empty:
                logger.error(f"No data available for {symbol}")
                return None

            # Extract information
            stock_info = self.get_stock_info_from_db(symbol)
            sector = stock_info.get('sector', 'Unknown')
            company_name = stock_info.get('name', symbol)

            # Current market data
            try:
                current_price = data['Close'].iloc[-1]
                if len(data) >= 2:
                    price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                    price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
                else:
                    price_change = 0
                    price_change_pct = 0
            except Exception as e:
                logger.error(f"Error calculating price changes: {str(e)}")
                return None

            # Technical indicators
            try:
                rsi = self.calculate_rsi(data['Close'])
                bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data['Close'])
                stoch_k, stoch_d = self.calculate_stochastic(data['High'], data['Low'], data['Close'])
                macd_line, signal_line, histogram = self.calculate_macd(data['Close'])
                support, resistance = self.calculate_support_resistance(data)
                volume_profile, poc_price = self.calculate_volume_profile(data)
            except Exception as e:
                logger.error(f"Error calculating technical indicators: {str(e)}")
                # Continue with None values for failed indicators

            # Sentiment analysis
            try:
                sentiment_results = self.analyze_news_sentiment(final_symbol)
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {str(e)}")
                sentiment_results = ([], [], [], "Error", "Error")

            # Risk metrics
            try:
                risk_metrics = self.calculate_risk_metrics(data)
            except Exception as e:
                logger.error(f"Error calculating risk metrics: {str(e)}")
                risk_metrics = {'volatility': 0.3, 'var_95': -0.05, 'max_drawdown': -0.2,
                                'sharpe_ratio': 0, 'atr': current_price * 0.02, 'risk_level': 'HIGH'}

            # Swing trading score
            try:
                swing_score = self.calculate_swing_trading_score(data, sentiment_results, sector)
            except Exception as e:
                logger.error(f"Error calculating swing score: {str(e)}")
                swing_score = 0

            # Trading plan
            try:
                trading_plan = self.generate_trading_plan(data, swing_score, risk_metrics)
            except Exception as e:
                logger.error(f"Error generating trading plan: {str(e)}")
                trading_plan = {'entry_signal': 'ERROR', 'entry_strategy': 'Analysis failed'}

            # Safe value extraction with error handling
            try:
                rsi_val = rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else None
            except:
                rsi_val = None

            try:
                bb_upper_val = bb_upper.iloc[-1] if not bb_upper.empty and not pd.isna(bb_upper.iloc[-1]) else None
                bb_middle_val = bb_middle.iloc[-1] if not bb_middle.empty and not pd.isna(bb_middle.iloc[-1]) else None
                bb_lower_val = bb_lower.iloc[-1] if not bb_lower.empty and not pd.isna(bb_lower.iloc[-1]) else None
                bb_position = None
                if all(x is not None for x in [bb_upper_val, bb_lower_val]) and bb_upper_val != bb_lower_val:
                    bb_position = (current_price - bb_lower_val) / (bb_upper_val - bb_lower_val)
            except:
                bb_upper_val = bb_middle_val = bb_lower_val = bb_position = None

            try:
                stoch_k_val = stoch_k.iloc[-1] if not stoch_k.empty and not pd.isna(stoch_k.iloc[-1]) else None
                stoch_d_val = stoch_d.iloc[-1] if not stoch_d.empty and not pd.isna(stoch_d.iloc[-1]) else None
            except:
                stoch_k_val = stoch_d_val = None

            try:
                macd_line_val = macd_line.iloc[-1] if not macd_line.empty and not pd.isna(macd_line.iloc[-1]) else None
                signal_line_val = signal_line.iloc[-1] if not signal_line.empty and not pd.isna(
                    signal_line.iloc[-1]) else None
                histogram_val = histogram.iloc[-1] if not histogram.empty and not pd.isna(histogram.iloc[-1]) else None
            except:
                macd_line_val = signal_line_val = histogram_val = None

            try:
                support_resistance_data = {
                    'support': support if support and not pd.isna(support) else None,
                    'resistance': resistance if resistance and not pd.isna(resistance) else None,
                    'distance_to_support': ((current_price - support) / support * 100) if support and not pd.isna(
                        support) else None,
                    'distance_to_resistance': (
                                (resistance - current_price) / current_price * 100) if resistance and not pd.isna(
                        resistance) else None
                }
            except:
                support_resistance_data = {'support': None, 'resistance': None, 'distance_to_support': None,
                                           'distance_to_resistance': None}

            try:
                volume_profile_data = {
                    'poc_price': poc_price if poc_price and not pd.isna(poc_price) else None,
                    'current_vs_poc': ((current_price - poc_price) / poc_price * 100) if poc_price and not pd.isna(
                        poc_price) else None
                }
            except:
                volume_profile_data = {'poc_price': None, 'current_vs_poc': None}

            # Safe sentiment data extraction
            try:
                if sentiment_results and len(sentiment_results) >= 5:
                    sentiment_scores, sentiment_articles, sentiment_confidence, sentiment_method, sentiment_source = sentiment_results
                    sentiment_summary = {
                        'positive': sentiment_scores.count('positive') if sentiment_scores else 0,
                        'negative': sentiment_scores.count('negative') if sentiment_scores else 0,
                        'neutral': sentiment_scores.count('neutral') if sentiment_scores else 0
                    }
                else:
                    sentiment_scores = sentiment_articles = sentiment_confidence = []
                    sentiment_method = sentiment_source = "Error"
                    sentiment_summary = {'positive': 0, 'negative': 0, 'neutral': 0}
            except:
                sentiment_scores = sentiment_articles = sentiment_confidence = []
                sentiment_method = sentiment_source = "Error"
                sentiment_summary = {'positive': 0, 'negative': 0, 'neutral': 0}

            # Compile results
            result = {
                'symbol': final_symbol,
                'company_name': company_name,
                'sector': sector,
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'rsi': rsi_val,
                'bollinger_bands': {
                    'upper': bb_upper_val,
                    'middle': bb_middle_val,
                    'lower': bb_lower_val,
                    'position': bb_position
                },
                'stochastic': {
                    'k': stoch_k_val,
                    'd': stoch_d_val
                },
                'macd': {
                    'line': macd_line_val,
                    'signal': signal_line_val,
                    'histogram': histogram_val
                },
                'support_resistance': support_resistance_data,
                'volume_profile': volume_profile_data,
                'sentiment': {
                    'scores': sentiment_scores,
                    'articles': sentiment_articles,
                    'confidence': sentiment_confidence,
                    'method': sentiment_method,
                    'source': sentiment_source,
                    'sentiment_summary': sentiment_summary
                },
                'risk_metrics': risk_metrics,
                'swing_score': swing_score,
                'trading_plan': trading_plan,
                'model_type': self.model_type,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            logger.info(f"Successfully analyzed {symbol} with score {swing_score}")
            return result

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def analyze_multiple_stocks(self, symbols, period="6mo", max_concurrent=10):
        """Analyze multiple stocks with progress tracking and comprehensive error handling"""
        results = []
        total_stocks = len(symbols) if symbols else 0

        if total_stocks == 0:
            logger.error("No symbols provided for analysis")
            return results

        print(f"Analyzing {total_stocks} stocks...")
        logger.info(f"Starting analysis of {total_stocks} stocks")

        successful_analyses = 0
        failed_analyses = 0

        for i, symbol in enumerate(symbols, 1):
            try:
                if i % 10 == 0:
                    print(f"Progress: {i}/{total_stocks} stocks analyzed ({i / total_stocks * 100:.0f}%)")
                    logger.info(f"Progress: {i}/{total_stocks} stocks analyzed")

                if not symbol or not isinstance(symbol, str):
                    logger.warning(f"Invalid symbol at position {i}: {symbol}")
                    failed_analyses += 1
                    continue

                analysis = self.analyze_swing_trading_stock(symbol.strip(), period)
                if analysis and analysis.get('swing_score', 0) > 0:
                    results.append(analysis)
                    successful_analyses += 1
                else:
                    failed_analyses += 1
                    logger.warning(f"Analysis failed or returned zero score for {symbol}")

            except KeyboardInterrupt:
                logger.info("Analysis interrupted by user")
                print(f"\nAnalysis interrupted. Processed {i - 1}/{total_stocks} stocks.")
                break
            except Exception as e:
                logger.error(f"Unexpected error analyzing {symbol}: {str(e)}")
                failed_analyses += 1
                continue

        # Sort by swing trading score
        try:
            results.sort(key=lambda x: x.get('swing_score', 0), reverse=True)
        except Exception as e:
            logger.error(f"Error sorting results: {str(e)}")

        print(f"Analysis completed: {successful_analyses} successful, {failed_analyses} failed out of {total_stocks}")
        logger.info(f"Analysis completed: {successful_analyses} successful, {failed_analyses} failed")

        return results

    def filter_stocks_by_risk_appetite(self, results, risk_appetite):
        """Filter stocks based on user's risk appetite with error handling"""
        try:
            if not results:
                logger.warning("No results to filter")
                return []

            if not risk_appetite:
                logger.warning("No risk appetite specified, using MEDIUM")
                risk_appetite = "MEDIUM"

            risk_thresholds = {
                'LOW': 0.25,  # <=25% volatility
                'MEDIUM': 0.40,  # <=40% volatility
                'HIGH': 1.0  # <=100% volatility (all stocks)
            }

            max_volatility = risk_thresholds.get(risk_appetite.upper(), 0.40)

            filtered_stocks = []
            for stock in results:
                try:
                    if not isinstance(stock, dict):
                        continue

                    risk_metrics = stock.get('risk_metrics', {})
                    trading_plan = stock.get('trading_plan', {})

                    volatility = risk_metrics.get('volatility', 1.0)  # Default high volatility
                    entry_signal = trading_plan.get('entry_signal', 'HOLD/WATCH')

                    if (volatility <= max_volatility and
                            entry_signal in ['BUY', 'STRONG BUY']):
                        filtered_stocks.append(stock)

                except Exception as e:
                    logger.warning(f"Error filtering stock {stock.get('symbol', 'Unknown')}: {str(e)}")
                    continue

            logger.info(
                f"Filtered {len(filtered_stocks)} stocks from {len(results)} based on {risk_appetite} risk appetite")
            return filtered_stocks

        except Exception as e:
            logger.error(f"Error filtering stocks by risk appetite: {str(e)}")
            return []

    def generate_portfolio_allocation(self, results, total_capital, risk_appetite):
        """Generate risk-adjusted portfolio allocation with comprehensive error handling"""
        try:
            if not results or not isinstance(results, list):
                print("Error: No suitable stocks found for portfolio creation")
                return None

            if total_capital <= 0:
                print("Error: Invalid total capital amount")
                return None

            print(f"\nPORTFOLIO ALLOCATION (Rs.{total_capital:,})")
            print("=" * 80)

            # Calculate total score for normalization
            total_score = sum(r.get('swing_score', 0) for r in results)

            if total_score <= 0:
                print("Error: Total score is zero, cannot create portfolio")
                return None

            portfolio_data = []

            print(
                f"{'Rank':<4} {'Symbol':<12} {'Company':<25} {'Score':<6} {'Risk':<8} {'Allocation':<12} {'Amount':<15}")
            print("-" * 88)

            total_allocated = 0

            for i, result in enumerate(results, 1):
                try:
                    score = result.get('swing_score', 0)
                    if score <= 0:
                        continue

                    allocation_pct = (score / total_score) * 100
                    allocation_amount = int(total_capital * (allocation_pct / 100))

                    # Adjust for position size multiplier
                    trading_plan = result.get('trading_plan', {})
                    position_multiplier = trading_plan.get('position_size_multiplier', 1.0)
                    adjusted_amount = int(allocation_amount * min(position_multiplier, 2.0))  # Cap at 2x

                    company_name = result.get('company_name', result.get('symbol', 'Unknown'))
                    company_short = company_name[:23] + "..." if len(company_name) > 25 else company_name

                    risk_metrics = result.get('risk_metrics', {})
                    risk_level = risk_metrics.get('risk_level', 'UNKNOWN')

                    symbol = result.get('symbol', 'Unknown')
                    sector = result.get('sector', 'Unknown')

                    print(
                        f"{i:<4} {symbol:<12} {company_short:<25} {score:<6.0f} {risk_level:<8} {allocation_pct:<11.1f}% Rs.{adjusted_amount:<14,}")

                    portfolio_data.append({
                        'symbol': symbol,
                        'company': company_name,
                        'score': score,
                        'allocation_pct': allocation_pct,
                        'amount': adjusted_amount,
                        'risk_level': risk_level,
                        'sector': sector
                    })

                    total_allocated += adjusted_amount

                except Exception as e:
                    logger.error(f"Error processing stock {i} in portfolio allocation: {str(e)}")
                    continue

            # Portfolio summary
            try:
                avg_volatility = sum(r.get('risk_metrics', {}).get('volatility', 0) for r in results) / len(results)
                avg_score = sum(r.get('swing_score', 0) for r in results) / len(results)
            except:
                avg_volatility = 0
                avg_score = 0

            # Sector diversification
            sector_allocation = {}
            for stock in portfolio_data:
                try:
                    sector = stock.get('sector', 'Unknown')
                    if sector not in sector_allocation:
                        sector_allocation[sector] = 0
                    sector_allocation[sector] += stock.get('allocation_pct', 0)
                except Exception as e:
                    logger.warning(f"Error calculating sector allocation: {str(e)}")

            print(f"\nPORTFOLIO SUMMARY")
            print("-" * 50)
            print(f"Total Budget: Rs.{total_capital:,}")
            print(f"Total Allocated: Rs.{total_allocated:,} ({total_allocated / total_capital * 100:.1f}%)")
            print(f"Number of Stocks: {len(results)}")
            print(f"Average Score: {avg_score:.1f}/100")
            print(f"Average Volatility: {avg_volatility * 100:.1f}%")
            print(f"Portfolio Risk Level: {risk_appetite}")

            if sector_allocation:
                print(f"\nSECTOR DIVERSIFICATION")
                print("-" * 30)
                for sector, allocation in sorted(sector_allocation.items(), key=lambda x: x[1], reverse=True):
                    print(f"{sector}: {allocation:.1f}%")

            return portfolio_data

        except Exception as e:
            logger.error(f"Error generating portfolio allocation: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"Error generating portfolio allocation: {str(e)}")
            return None

    def get_single_best_recommendation(self, results):
        """Get detailed recommendation for the single best stock with error handling"""
        try:
            if not results or not isinstance(results, list):
                logger.warning("No results available for recommendation")
                return None

            best_stock = results[0]  # Highest scoring stock
            if not isinstance(best_stock, dict):
                logger.error("Invalid best stock data format")
                return None

            print(f"\nSINGLE BEST STOCK RECOMMENDATION")
            print("=" * 70)

            # Safe data extraction
            company_name = best_stock.get('company_name', 'Unknown')
            symbol = best_stock.get('symbol', 'Unknown')
            sector = best_stock.get('sector', 'Unknown')
            swing_score = best_stock.get('swing_score', 0)
            current_price = best_stock.get('current_price', 0)
            price_change = best_stock.get('price_change', 0)
            price_change_pct = best_stock.get('price_change_pct', 0)

            risk_metrics = best_stock.get('risk_metrics', {})
            risk_level = risk_metrics.get('risk_level', 'Unknown')

            print(f"Company: {company_name}")
            print(f"Symbol: {symbol}")
            print(f"Sector: {sector}")
            print(f"Swing Score: {swing_score:.0f}/100")
            print(f"Current Price: Rs.{current_price:.2f}")
            print(f"Price Change: Rs.{price_change:.2f} ({price_change_pct:.2f}%)")
            print(f"Risk Level: {risk_level}")

            # Trading recommendation
            trading_plan = best_stock.get('trading_plan', {})
            print(f"\nTRADING RECOMMENDATION")
            print("-" * 30)
            print(f"Signal: {trading_plan.get('entry_signal', 'Unknown')}")
            print(f"Strategy: {trading_plan.get('entry_strategy', 'Unknown')}")

            targets = trading_plan.get('targets', {})
            print(f"Stop Loss: Rs.{trading_plan.get('stop_loss', 0):.2f}")
            print(f"Target 1: Rs.{targets.get('target_1', 0):.2f}")
            print(f"Target 2: Rs.{targets.get('target_2', 0):.2f}")
            print(f"Target 3: Rs.{targets.get('target_3', 0):.2f}")
            print(f"Holding Period: {trading_plan.get('holding_period', 'Unknown')}")

            # Key technical levels
            print(f"\nKEY LEVELS")
            print("-" * 15)
            print(f"Support: Rs.{trading_plan.get('support', 0):.2f}")
            print(f"Resistance: Rs.{trading_plan.get('resistance', 0):.2f}")

            rsi_val = best_stock.get('rsi')
            if rsi_val is not None:
                print(f"RSI: {rsi_val:.1f}")

            # Sentiment summary
            sentiment = best_stock.get('sentiment', {}).get('sentiment_summary', {})
            print(f"\nSENTIMENT OVERVIEW")
            print("-" * 20)
            print(
                f"Positive: {sentiment.get('positive', 0)}, Negative: {sentiment.get('negative', 0)}, Neutral: {sentiment.get('neutral', 0)}")

            return best_stock

        except Exception as e:
            logger.error(f"Error getting single best recommendation: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"Error generating recommendation: {str(e)}")
            return None

    def analyze_multiple_stocks(self, symbols: List[str]) -> List[Dict]:
        """Analyzes multiple stocks and returns a list of results."""
        results = []
        total_stocks = len(symbols)
        logger.info(f"Starting analysis of {total_stocks} stocks for position trading.")

        for i, symbol in enumerate(symbols, 1):
            # Simple progress printout
            print(f"\rAnalyzing: [{i}/{total_stocks}] {symbol}...", end="")
            try:
                analysis = self.analyze_position_trading_stock(symbol)
                if analysis:
                    results.append(analysis)
            except Exception as e:
                logger.error(f"Failed to analyze {symbol} in batch mode: {e}")
        print("\nAnalysis complete.")
        # Sort results by the final position score
        results.sort(key=lambda x: x.get('position_score', 0), reverse=True)
        return results

    def filter_by_risk_profile(self, results: List[Dict], risk_profile: str) -> List[Dict]:
        """Filters stock analysis results based on the user's risk profile."""
        if not results:
            return []

        # Define thresholds for volatility and minimum required score based on risk
        risk_thresholds = {
            'CONSERVATIVE': {'max_vol': 0.30, 'min_score': 70},  # Low volatility, high conviction
            'BALANCED': {'max_vol': 0.45, 'min_score': 60},  # Medium volatility, good conviction
            'AGGRESSIVE': {'max_vol': 0.65, 'min_score': 55},  # Higher volatility, acceptable score
        }

        params = risk_thresholds.get(risk_profile.upper(), risk_thresholds['BALANCED'])
        max_vol = params['max_vol']
        min_score = params['min_score']

        filtered_stocks = []
        for stock in results:
            # Check if the stock meets the criteria
            meets_volatility = stock.get('risk_metrics', {}).get('volatility', 1.0) <= max_vol
            meets_score = stock.get('position_score', 0) >= min_score
            is_buy_signal = stock.get('trading_plan', {}).get('entry_signal') in ['BUY', 'STRONG BUY']

            if meets_volatility and meets_score and is_buy_signal:
                filtered_stocks.append(stock)

        logger.info(f"Filtered {len(filtered_stocks)} stocks from {len(results)} for {risk_profile} profile.")
        return filtered_stocks

    def generate_portfolio_allocation(self, results: List[Dict], total_capital: float, risk_profile: str):
        """Generates and displays a risk-adjusted portfolio allocation for long-term investment."""
        if not results:
            print("\nNo suitable stocks found to generate a portfolio.")
            return

        print(f"\n{'-' * 25}\n📈 PORTFOLIO ALLOCATION\n{'-' * 25}")
        print(f"Total Budget: ₹{total_capital:,.2f} | Risk Profile: {risk_profile}")

        # Use position score for weighting
        total_score = sum(r.get('position_score', 0) for r in results)
        if total_score == 0:
            print("Cannot generate allocation as total score of suitable stocks is zero.")
            return

        print("\n" + "=" * 80)
        print(f"{'Symbol':<12} {'Company':<25} {'Score':<7} {'Allocation':<12} {'Amount (₹)':<15}")
        print("-" * 80)

        total_allocated = 0
        for result in results:
            score = result.get('position_score', 0)
            allocation_pct = score / total_score
            amount = total_capital * allocation_pct
            total_allocated += amount

            company_name = result.get('company_name', 'N/A')
            company_short = (company_name[:22] + '...') if len(company_name) > 25 else company_name

            print(f"{result['symbol']:<12} {company_short:<25} {score:<7.1f} {allocation_pct:<11.2%} {amount:<15,.2f}")

        print("-" * 80)
        print(f"TOTAL ALLOCATED: {total_allocated:,.2f} ({(total_allocated / total_capital):.1%})")
        print("=" * 80)

    def print_analysis_summary(self, all_results, filtered_results, risk_appetite, total_budget):
        """Print comprehensive analysis summary with error handling"""
        try:
            print(f"\nMARKET ANALYSIS SUMMARY")
            print("=" * 50)
            print(f"Total Stocks Analyzed: {len(all_results) if all_results else 0}")
            print(f"Risk Appetite: {risk_appetite}")
            print(f"Budget: Rs.{total_budget:,}")
            print(f"Suitable Stocks Found: {len(filtered_results) if filtered_results else 0}")

            if all_results and len(all_results) > 0:
                try:
                    avg_market_score = sum(r.get('swing_score', 0) for r in all_results) / len(all_results)
                    print(f"Average Market Score: {avg_market_score:.1f}/100")
                except:
                    print("Average Market Score: Unable to calculate")

                # Risk distribution
                risk_distribution = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'UNKNOWN': 0}
                for result in all_results:
                    try:
                        risk_level = result.get('risk_metrics', {}).get('risk_level', 'UNKNOWN')
                        if risk_level in risk_distribution:
                            risk_distribution[risk_level] += 1
                        else:
                            risk_distribution['UNKNOWN'] += 1
                    except:
                        risk_distribution['UNKNOWN'] += 1

                print(f"\nMARKET RISK DISTRIBUTION")
                print("-" * 25)
                for risk, count in risk_distribution.items():
                    if count > 0:
                        percentage = (count / len(all_results)) * 100
                        print(f"{risk} Risk: {count} stocks ({percentage:.1f}%)")

        except Exception as e:
            logger.error(f"Error printing analysis summary: {str(e)}")
            print(f"Error generating analysis summary: {str(e)}")


# ========================= MAIN EXECUTION =========================

def safe_input_int(prompt, default=None, min_val=None, max_val=None):
    """Safe integer input with validation"""
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input and default is not None:
                return default

            value = int(user_input)

            if min_val is not None and value < min_val:
                print(f"Error: Value must be at least {min_val}")
                continue

            if max_val is not None and value > max_val:
                print(f"Error: Value must be at most {max_val}")
                continue

            return value

        except ValueError:
            print("Error: Please enter a valid integer")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return None


def safe_input_float(prompt, default=None, min_val=None, max_val=None):
    """Safe float input with validation"""
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input and default is not None:
                return default

            value = float(user_input)

            if min_val is not None and value < min_val:
                print(f"Error: Value must be at least {min_val}")
                continue

            if max_val is not None and value > max_val:
                print(f"Error: Value must be at most {max_val}")
                continue

            return value

        except ValueError:
            print("Error: Please enter a valid number")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return None


if __name__ == "__main__":
    try:
        # Initialize the enhanced swing trading system
        print("Initializing Enhanced Swing Trading System...")
        swing_trader = EnhancedSwingTradingSystem(
            model_path="D:/Python_files/models/sentiment_pipeline.joblib",
            news_api_key=os.getenv("NEWS_API_KEY")
        )

        print("ENHANCED SWING TRADING SYSTEM")
        print("Advanced Portfolio Creation with Budget & Risk Management")
        print("=" * 70)

        # ===== USER INPUT COLLECTION =====

        # Get user budget
        total_budget = safe_input_float(
            "\nEnter your total investment budget in INR (e.g., 500000): ",
            min_val=1000
        )

        if total_budget is None:
            print("Exiting...")
            sys.exit(0)

        # Get user risk appetite
        risk_levels = ['LOW', 'MEDIUM', 'HIGH']
        print(f"\nRisk Appetite Options:")
        print("• LOW: Conservative (<=25% volatility) - Blue chip stocks")
        print("• MEDIUM: Balanced (<=40% volatility) - Mixed portfolio")
        print("• HIGH: Aggressive (<=100% volatility) - All opportunities")

        while True:
            try:
                risk_appetite = input("\nEnter your risk appetite (LOW/MEDIUM/HIGH): ").upper().strip()
                if risk_appetite not in risk_levels:
                    print("Error: Invalid risk level. Please enter LOW, MEDIUM, or HIGH.")
                else:
                    break
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)

        print(f"\nConfiguration Set:")
        print(f"Budget: Rs.{total_budget:,.0f}")
        print(f"Risk Appetite: {risk_appetite}")

        # ===== COMPREHENSIVE MARKET ANALYSIS =====

        print(f"\nAnalyzing Indian Stock Market...")
        print(f"Scanning {len(swing_trader.get_all_stock_symbols())} stocks across BSE & NSE")
        print("This may take several minutes...")

        # Get all stock symbols from database
        all_symbols = swing_trader.get_all_stock_symbols()

        # Analyze stocks
        start_time = datetime.now()
        all_results = swing_trader.analyze_multiple_stocks(all_symbols)
        analysis_time = datetime.now() - start_time

        print(f"Analysis completed in {analysis_time.total_seconds():.0f} seconds")

        # ===== RISK-BASED FILTERING =====

        print(f"\nFiltering stocks by risk appetite...")
        filtered_results = swing_trader.filter_stocks_by_risk_appetite(all_results, risk_appetite)

        if not filtered_results:
            print(f"\nNo suitable stocks found matching your criteria:")
            print(f"• Risk Appetite: {risk_appetite}")
            print(f"• Minimum Signal: BUY or STRONG BUY")
            print("\nSuggestions:")
            print("• Consider increasing your risk tolerance")
            print("• Try a different time period")
            print("• Check market conditions")
        else:
            # ===== PORTFOLIO CREATION =====

            print(f"\nFound {len(filtered_results)} suitable investment opportunities")

            # Generate portfolio allocation
            portfolio = swing_trader.generate_portfolio_allocation(
                filtered_results,
                int(total_budget),
                risk_appetite
            )

            # ===== SINGLE BEST RECOMMENDATION =====

            best_stock = swing_trader.get_single_best_recommendation(filtered_results)

            # ===== COMPREHENSIVE SUMMARY =====

            swing_trader.print_analysis_summary(all_results, filtered_results, risk_appetite, total_budget)

            # ===== DETAILED RANKINGS =====

            print(f"\nTOP 10 STOCK RANKINGS")
            print("=" * 70)
            print(f"{'Rank':<4} {'Symbol':<12} {'Company':<20} {'Score':<6} {'Signal':<12} {'Risk':<8}")
            print("-" * 70)

            for i, result in enumerate(filtered_results[:10], 1):
                try:
                    company_name = result.get('company_name', 'Unknown')
                    company_short = company_name[:18] + "..." if len(company_name) > 20 else company_name
                    symbol = result.get('symbol', 'Unknown')
                    score = result.get('swing_score', 0)
                    signal = result.get('trading_plan', {}).get('entry_signal', 'Unknown')
                    risk = result.get('risk_metrics', {}).get('risk_level', 'Unknown')

                    print(f"{i:<4} {symbol:<12} {company_short:<20} {score:<6.0f} {signal:<12} {risk:<8}")
                except Exception as e:
                    logger.error(f"Error displaying ranking for position {i}: {str(e)}")

        # ===== INTERACTIVE MODE =====

        print(f"\nINTERACTIVE MODE")
        print("Available commands:")
        print("• Enter stock symbol for detailed analysis")
        print("• Type 'portfolio' for custom portfolio analysis")
        print("• Type 'settings' to change budget/risk settings")
        print("• Type 'quit' to exit")

        while True:
            try:
                user_input = input("\n> Enter command: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Thank you for using Enhanced Swing Trading System!")
                    print("Happy Trading!")
                    break

                elif user_input.lower() == 'portfolio':
                    symbols = input("Enter stock symbols (comma-separated) or 'sample' for sample analysis: ").strip()

                    if symbols.lower() == 'sample':
                        symbols_list = swing_trader.get_all_stock_symbols()[:20]  # Sample 20 stocks
                        print(f"Analyzing sample portfolio of {len(symbols_list)} stocks...")
                    else:
                        symbols_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]

                    if symbols_list:
                        interactive_results = swing_trader.analyze_multiple_stocks(symbols_list)
                        if interactive_results:
                            filtered_interactive = swing_trader.filter_stocks_by_risk_appetite(
                                interactive_results, risk_appetite)
                            if filtered_interactive:
                                swing_trader.generate_portfolio_allocation(
                                    filtered_interactive, int(total_budget), risk_appetite)
                            else:
                                print("No suitable stocks found in your selection")
                        else:
                            print("No valid analysis results obtained")
                    else:
                        print("No valid symbols provided")

                elif user_input.lower() == 'settings':
                    print("\nCurrent settings:")
                    print(f"Budget: Rs.{total_budget:,.0f}")
                    print(f"Risk Appetite: {risk_appetite}")

                    change = input("Change settings? (y/n): ").lower()
                    if change == 'y':
                        new_budget = safe_input_float(
                            f"Enter new budget (current: Rs.{total_budget:,.0f}): ",
                            default=total_budget, min_val=1000
                        )
                        if new_budget:
                            total_budget = new_budget

                        print("Risk levels: LOW, MEDIUM, HIGH")
                        new_risk = input(f"Enter new risk appetite (current: {risk_appetite}): ").upper().strip()
                        if new_risk in risk_levels:
                            risk_appetite = new_risk

                        print(f"Settings updated - Budget: Rs.{total_budget:,.0f}, Risk: {risk_appetite}")

                elif user_input.upper() in swing_trader.get_all_stock_symbols():
                    # User entered a valid stock symbol
                    symbol = user_input.upper()
                    print(f"\nAnalyzing {symbol}...")

                    # Analyze the single stock
                    analysis = swing_trader.analyze_swing_trading_stock(symbol)

                    if analysis:
                        # Print detailed analysis
                        print(f"\nDETAILED ANALYSIS: {analysis['symbol']} ({analysis['company_name']})")
                        print("=" * 70)
                        print(f"Sector: {analysis['sector']}")
                        print(f"Analysis Date: {analysis['analysis_date']}")
                        print(f"Current Price: Rs.{analysis['current_price']:.2f}")
                        print(f"Price Change: Rs.{analysis['price_change']:.2f} ({analysis['price_change_pct']:.2f}%)")
                        print(f"Swing Score: {analysis['swing_score']:.0f}/100")
                        print(f"Risk Level: {analysis['risk_metrics']['risk_level']}")

                        # Technical Indicators
                        print("\nTECHNICAL INDICATORS")
                        print("-" * 30)
                        if analysis['rsi']:
                            print(f"• RSI: {analysis['rsi']:.1f} (14-day)")

                        if analysis['bollinger_bands']['position']:
                            bb_pos = analysis['bollinger_bands']['position'] * 100
                            print(f"• Bollinger Bands Position: {bb_pos:.1f}% (0% = lower band, 100% = upper band)")

                        if analysis['stochastic']['k'] and analysis['stochastic']['d']:
                            print(
                                f"• Stochastic: K={analysis['stochastic']['k']:.1f}, D={analysis['stochastic']['d']:.1f}")

                        if analysis['macd']['line']:
                            print(
                                f"• MACD: Line={analysis['macd']['line']:.2f}, Signal={analysis['macd']['signal']:.2f}, Histogram={analysis['macd']['histogram']:.2f}")

                        if analysis['support_resistance']['support']:
                            print(f"• Support: Rs.{analysis['support_resistance']['support']:.2f}")

                        if analysis['support_resistance']['resistance']:
                            print(f"• Resistance: Rs.{analysis['support_resistance']['resistance']:.2f}")

                        # Sentiment Summary
                        sentiment = analysis['sentiment']['sentiment_summary']
                        print("\nSENTIMENT OVERVIEW")
                        print("-" * 20)
                        print(
                            f"• Positive: {sentiment['positive']}, Negative: {sentiment['negative']}, Neutral: {sentiment['neutral']}")
                        print(
                            f"• Method: {analysis['sentiment']['method']} | Source: {analysis['sentiment']['source']}")

                        # Sample News Headlines
                        if analysis['sentiment']['articles']:
                            print("\nSAMPLE NEWS HEADLINES")
                            print("-" * 25)
                            for i, article in enumerate(analysis['sentiment']['articles'][:3], 1):
                                print(f"{i}. {article}")

                        # Trading Plan
                        tp = analysis['trading_plan']
                        print("\nTRADING PLAN")
                        print("-" * 20)
                        print(f"• Signal: {tp['entry_signal']}")
                        print(f"• Strategy: {tp['entry_strategy']}")
                        print(f"• Stop Loss: Rs.{tp['stop_loss']:.2f}")
                        print(
                            f"• Targets: Rs.{tp['targets']['target_1']:.2f} | Rs.{tp['targets']['target_2']:.2f} | Rs.{tp['targets']['target_3']:.2f}")
                        print(f"• Holding Period: {tp['holding_period']}")

                        # Risk Metrics
                        rm = analysis['risk_metrics']
                        print("\nRISK METRICS")
                        print("-" * 15)
                        print(f"• Volatility: {rm['volatility'] * 100:.1f}% (annualized)")
                        print(f"• Max Drawdown: {rm['max_drawdown'] * 100:.1f}%")
                        print(f"• Sharpe Ratio: {rm['sharpe_ratio']:.2f}")
                        print(f"• Value at Risk (95%): {rm['var_95'] * 100:.1f}%")

                    else:
                        print(f"Could not analyze {symbol}. Please try another symbol.")

                else:
                    # Try to analyze as custom symbol
                    if len(user_input) > 0:
                        print(f"Attempting to analyze {user_input.upper()}...")
                        analysis = swing_trader.analyze_swing_trading_stock(user_input.upper())

                        if analysis:
                            print(f"Analysis completed for {user_input.upper()}")
                            print(f"Score: {analysis['swing_score']:.0f}/100")
                            print(f"Signal: {analysis['trading_plan']['entry_signal']}")
                            print(f"Risk: {analysis['risk_metrics']['risk_level']}")
                        else:
                            print(f"Could not analyze {user_input.upper()}. Symbol may not be available.")
                    else:
                        print("Invalid command. Type 'quit' to exit.")

            except KeyboardInterrupt:
                print("\nThank you for using Enhanced Swing Trading System!")
                print("Happy Trading!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {str(e)}")
                print(f"Error: {str(e)}")
                print("Please try again or type 'quit' to exit.")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"Critical error occurred: {str(e)}")
        print("Please check the logs for more details.")
        sys.exit(1)
    finally:
        print("System shutdown complete.")
