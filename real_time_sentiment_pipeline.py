import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
import warnings
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# Define SBERTTransformer class BEFORE loading the model
try:
    from sentence_transformers import SentenceTransformer


    class SBERTTransformer:
        def __init__(self, model_name='all-MiniLM-L6-v2'):
            self.model = SentenceTransformer(model_name)

        def transform(self, sentences):
            return self.model.encode(sentences)

        def fit(self, X, y=None):
            return self


    SBERT_AVAILABLE = True
except ImportError:
    print("⚠️ sentence-transformers not available. Install with: pip install sentence-transformers")
    SBERT_AVAILABLE = False


    class SBERTTransformer:
        def __init__(self, model_name='all-MiniLM-L6-v2'):
            self.model_name = model_name
            print(f"⚠️ SBERTTransformer created but sentence-transformers not available")

        def transform(self, sentences):
            raise ImportError("sentence-transformers not installed")

        def fit(self, X, y=None):
            return self


class EnhancedSwingTradingSystem:
    """🚀 Enhanced Swing Trading System for Indian Markets with Budget & Risk Management"""

    def __init__(self, model_path="D:/Python_files/models/sentiment_pipeline.joblib", news_api_key=None):
        self.sentiment_pipeline = None
        self.vectorizer = None
        self.model = None
        self.label_encoder = None
        self.news_api_key = news_api_key or os.getenv("NEWS_API_KEY") or "dd33ebe105ea4b02a3b7e77bc4a93d01"

        # Model status tracking
        self.model_loaded = False
        self.model_type = "None"

        # Trading parameters
        self.swing_trading_params = {
            'min_holding_period': 3,  # days
            'max_holding_period': 30,  # days
            'risk_per_trade': 0.02,  # 2% risk per trade
            'max_portfolio_risk': 0.10,  # 10% max portfolio risk
            'profit_target_multiplier': 2.5,  # Risk-reward ratio
        }

        if not self.news_api_key:
            print("⚠️ NEWS_API_KEY not provided. Using sample news for sentiment analysis.")
        else:
            print("✅ News API key available. Will fetch real news articles.")

        # Load sentiment model
        self.load_sbert_model(model_path)

        # Initialize comprehensive stock database
        self.initialize_stock_database()

    def initialize_stock_database(self):
        """Initialize comprehensive Indian stock database (BSE + NSE)"""
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
            "M&MFIN": {"name": "Mahindra & Mahindra Financial", "sector": "Financial Services"},
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

            # Pharmaceuticals & Healthcare
            "REDDY": {"name": "Dr. Reddy's Labs", "sector": "Pharmaceuticals"},
            "GLENMARK": {"name": "Glenmark Pharmaceuticals", "sector": "Pharmaceuticals"},
            "NATCOPHAR": {"name": "Natco Pharma", "sector": "Pharmaceuticals"},
            "STRIDES": {"name": "Strides Pharma Science", "sector": "Pharmaceuticals"},
            "LALPATHLAB": {"name": "Dr. Lal PathLabs", "sector": "Healthcare"},
            "THYROCARE": {"name": "Thyrocare Technologies", "sector": "Healthcare"},
            "FORTIS": {"name": "Fortis Healthcare", "sector": "Healthcare"},
            "MAXHEALTH": {"name": "Max Healthcare", "sector": "Healthcare"},
            "NARAYANHRD": {"name": "Narayana Hrudayalaya", "sector": "Healthcare"},

            # Metals & Mining
            "SAIL": {"name": "Steel Authority of India", "sector": "Steel"},
            "JINDALSTEL": {"name": "Jindal Steel & Power", "sector": "Steel"},
            "NMDC": {"name": "NMDC Limited", "sector": "Mining"},
            "MOIL": {"name": "MOIL Limited", "sector": "Mining"},
            "VEDL": {"name": "Vedanta Limited", "sector": "Metals"},
            "HINDZINC": {"name": "Hindustan Zinc", "sector": "Metals"},
            "NATIONALUM": {"name": "National Aluminium", "sector": "Metals"},
            "RATNAMANI": {"name": "Ratnamani Metals", "sector": "Metals"},

            # Consumer & Retail
            "DMART": {"name": "Avenue Supermarts", "sector": "Retail"},
            "TRENT": {"name": "Trent Limited", "sector": "Retail"},
            "SHOPERSTOP": {"name": "Shoppers Stop", "sector": "Retail"},
            "PAGEIND": {"name": "Page Industries", "sector": "Textiles"},
            "RAYMOND": {"name": "Raymond Limited", "sector": "Textiles"},
            "ADITYANB": {"name": "Aditya Birla Nuvo", "sector": "Consumer Goods"},
            "VBL": {"name": "Varun Beverages", "sector": "Consumer Goods"},
            "EMAMILTD": {"name": "Emami Limited", "sector": "Consumer Goods"},
            "JUBLFOOD": {"name": "Jubilant FoodWorks", "sector": "Consumer Goods"},
            "WESTLIFE": {"name": "Westlife Development", "sector": "Consumer Goods"},

            # Infrastructure & Construction
            "LTTS": {"name": "L&T Technology Services", "sector": "Construction"},
            "IRB": {"name": "IRB Infrastructure", "sector": "Infrastructure"},
            "GMRINFRA": {"name": "GMR Infrastructure", "sector": "Infrastructure"},
            "GVK": {"name": "GVK Power & Infrastructure", "sector": "Infrastructure"},
            "ASHOKA": {"name": "Ashoka Buildcon", "sector": "Construction"},
            "NCC": {"name": "NCC Limited", "sector": "Construction"},
            "SOBHA": {"name": "Sobha Limited", "sector": "Real Estate"},
            "DLF": {"name": "DLF Limited", "sector": "Real Estate"},
            "GODREJPROP": {"name": "Godrej Properties", "sector": "Real Estate"},
            "PRESTIGE": {"name": "Prestige Estates", "sector": "Real Estate"},
            "BRIGADE": {"name": "Brigade Enterprises", "sector": "Real Estate"},

            # Telecommunications & Media
            "RCOM": {"name": "Reliance Communications", "sector": "Telecommunications"},
            "IDEA": {"name": "Idea Cellular", "sector": "Telecommunications"},
            "HATHWAY": {"name": "Hathway Cable", "sector": "Media"},
            "SITI": {"name": "Siti Networks", "sector": "Media"},
            "ZEEL": {"name": "Zee Entertainment", "sector": "Media"},
            "PVRINOX": {"name": "PVR INOX", "sector": "Entertainment"},
            "TIPS": {"name": "Tips Industries", "sector": "Media"},

            # Power & Energy
            "TATAPOWER": {"name": "Tata Power", "sector": "Power"},
            "ADANIPOWER": {"name": "Adani Power", "sector": "Power"},
            "RPOWER": {"name": "Reliance Power", "sector": "Power"},
            "TORNTPOWER": {"name": "Torrent Power", "sector": "Power"},
            "CESC": {"name": "CESC Limited", "sector": "Power"},
            "NHPC": {"name": "NHPC Limited", "sector": "Power"},
            "SJVN": {"name": "SJVN Limited", "sector": "Power"},
            "THERMAX": {"name": "Thermax Limited", "sector": "Power"},

            # Chemicals & Fertilizers
            "GUJALKALI": {"name": "Gujarat Alkalies", "sector": "Chemicals"},
            "DEEPAKNTR": {"name": "Deepak Nitrite", "sector": "Chemicals"},
            "AARTI": {"name": "Aarti Industries", "sector": "Chemicals"},
            "BALRAMCHIN": {"name": "Balrampur Chini Mills", "sector": "Chemicals"},
            "GNFC": {"name": "Gujarat Narmada Valley", "sector": "Fertilizers"},
            "CHAMBAL": {"name": "Chambal Fertilizers", "sector": "Fertilizers"},
            "COROMANDEL": {"name": "Coromandel International", "sector": "Fertilizers"},
            "KRIBHCO": {"name": "Krishak Bharati Cooperative", "sector": "Fertilizers"},

            # Aviation & Transportation
            "INDIGO": {"name": "InterGlobe Aviation", "sector": "Aviation"},
            "SPICEJET": {"name": "SpiceJet Limited", "sector": "Aviation"},
            "CONCOR": {"name": "Container Corporation", "sector": "Transportation"},
            "GESHIP": {"name": "Great Eastern Shipping", "sector": "Transportation"},
            "ESCORTS": {"name": "Escorts Limited", "sector": "Transportation"},

            # Food & Agriculture
            "KRBL": {"name": "KRBL Limited", "sector": "Food Processing"},
            "LTFH": {"name": "L&T Finance Holdings", "sector": "Food Processing"},
            "ADVENZYMES": {"name": "Advanced Enzymes", "sector": "Food Processing"},
            "AVANTIFEED": {"name": "Avanti Feeds", "sector": "Food Processing"},
            "GODREJAGRO": {"name": "Godrej Agrovet", "sector": "Agriculture"},

            # Emerging Sectors
            "ZOMATO": {"name": "Zomato Limited", "sector": "Technology"},
            "NYKAA": {"name": "FSN E-Commerce Ventures", "sector": "E-commerce"},
            "PAYTM": {"name": "One97 Communications", "sector": "Fintech"},
            "POLICYBZR": {"name": "PB Fintech", "sector": "Fintech"},
            "CARTRADE": {"name": "CarTrade Tech", "sector": "Technology"},
            "EASEMYTRIP": {"name": "Easy Trip Planners", "sector": "Travel"},
            "CLEAN": {"name": "Clean Science Technology", "sector": "Chemicals"},
            "LICI": {"name": "Life Insurance Corporation", "sector": "Insurance"},
            "NEWGEN": {"name": "Newgen Software", "sector": "Information Technology"},
            "ROUTE": {"name": "Route Mobile", "sector": "Technology"}
        }

        print(f"✅ Initialized database with {len(self.indian_stocks)} Indian stocks")

    def get_all_stock_symbols(self):
        """Get all stock symbols for analysis"""
        return list(self.indian_stocks.keys())

    def get_stock_info_from_db(self, symbol):
        """Get stock information from internal database"""
        base_symbol = symbol.split('.')[0].upper()
        return self.indian_stocks.get(base_symbol, {"name": symbol, "sector": "Unknown"})

    def load_sbert_model(self, model_path):
        """Load trained SBERT sentiment model"""
        if not SBERT_AVAILABLE:
            print("⚠️ sentence-transformers not available, using TextBlob fallback")
            self.model_type = "TextBlob"
            return

        if not os.path.exists(model_path):
            print(f"⚠️ SBERT model not found at {model_path}")
            print("🔄 Using TextBlob as fallback for sentiment analysis")
            self.model_type = "TextBlob"
            return

        try:
            print(f"🔄 Loading SBERT sentiment model from {model_path}...")
            self.sentiment_pipeline = joblib.load(model_path)
            self.vectorizer = self.sentiment_pipeline.get("vectorizer")
            self.model = self.sentiment_pipeline.get("model")
            self.label_encoder = self.sentiment_pipeline.get("label_encoder")

            if all([self.vectorizer, self.model, self.label_encoder]):
                print("✅ SBERT sentiment model loaded successfully!")
                print(f"📊 Model classes: {list(self.label_encoder.classes_)}")
                self.model_loaded = True
                self.model_type = "SBERT + RandomForest"
            else:
                print("⚠️ Model components incomplete, using TextBlob fallback")
                self.model_type = "TextBlob"
                self.sentiment_pipeline = None

        except Exception as e:
            print(f"❌ Error loading SBERT model: {str(e)}")
            print("🔄 Using TextBlob as fallback for sentiment analysis")
            self.model_type = "TextBlob"
            self.sentiment_pipeline = None

    def get_sector_weights(self, sector):
        """Get dynamic weights based on sector for swing trading"""
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

        return tech_weight, sentiment_weight

    def get_indian_stock_data(self, symbol, period="6mo"):
        """Get Indian stock data with extended period for swing trading"""
        symbol = symbol.upper().replace(" ", "").replace(".", "")

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
            "DIVISLAB": "DIVISLAB.NS", "DRREDDY": "DRREDDY.NS", "CIPLA": "CIPLA.NS",
            "GRASIM": "GRASIM.NS", "JSWSTEEL": "JSWSTEEL.NS", "TATASTEEL": "TATASTEEL.NS",
            "COALINDIA": "COALINDIA.NS", "BPCL": "BPCL.NS", "IOC": "IOC.NS",
            "HEROMOTOCO": "HEROMOTOCO.NS", "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
            "M&M": "M&M.NS", "EICHERMOT": "EICHERMOT.NS", "TATACONSUM": "TATACONSUM.NS",
            "BRITANNIA": "BRITANNIA.NS", "NESTLEIND": "NESTLEIND.NS", "ITC": "ITC.NS"
        }

        symbols_to_try = [f"{symbol}.NS", f"{symbol}.BO", symbol]
        if symbol in symbol_mappings:
            symbols_to_try.insert(0, symbol_mappings[symbol])

        for sym in symbols_to_try:
            try:
                ticker = yf.Ticker(sym)
                data = ticker.history(period=period)
                if not data.empty and len(data) > 30:  # Need more data for swing trading
                    info = ticker.info
                    return data, info, sym
            except Exception as e:
                continue

        return None, None, None

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return pd.Series([np.nan] * len(prices), index=prices.index), \
                pd.Series([np.nan] * len(prices), index=prices.index), \
                pd.Series([np.nan] * len(prices), index=prices.index)

        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return upper_band, sma, lower_band

    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        if len(close) < k_period:
            return pd.Series([np.nan] * len(close), index=close.index), \
                pd.Series([np.nan] * len(close), index=close.index)

        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()

        return k_percent, d_percent

    def calculate_support_resistance(self, data, window=20):
        """Calculate support and resistance levels"""
        highs = data['High'].rolling(window=window).max()
        lows = data['Low'].rolling(window=window).min()

        # Find significant levels
        resistance_levels = []
        support_levels = []

        for i in range(window, len(data)):
            # Check if current high is a local maximum
            if data['High'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(data['High'].iloc[i])

            # Check if current low is a local minimum
            if data['Low'].iloc[i] == lows.iloc[i]:
                support_levels.append(data['Low'].iloc[i])

        # Get most recent levels
        current_resistance = max(resistance_levels[-3:]) if len(resistance_levels) >= 3 else data['High'].iloc[-1]
        current_support = min(support_levels[-3:]) if len(support_levels) >= 3 else data['Low'].iloc[-1]

        return current_support, current_resistance

    def calculate_volume_profile(self, data, bins=20):
        """Calculate Volume Profile"""
        if 'Volume' not in data.columns:
            return None, None

        price_range = data['High'].max() - data['Low'].min()
        bin_size = price_range / bins
        volume_profile = {}

        for i in range(len(data)):
            price = (data['High'].iloc[i] + data['Low'].iloc[i]) / 2
            volume = data['Volume'].iloc[i]
            bin_level = int((price - data['Low'].min()) / bin_size)
            bin_level = min(bin_level, bins - 1)
            price_level = data['Low'].min() + (bin_level * bin_size)

            if price_level not in volume_profile:
                volume_profile[price_level] = 0
            volume_profile[price_level] += volume

        # Find Point of Control (POC) - highest volume level
        poc_price = max(volume_profile.keys(), key=lambda x: volume_profile[x])
        poc_volume = volume_profile[poc_price]

        return volume_profile, poc_price

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        if len(prices) < period:
            return pd.Series([50] * len(prices), index=prices.index)

        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        if len(prices) < slow:
            zeros = pd.Series([0] * len(prices), index=prices.index)
            return zeros, zeros, zeros

        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()

        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        return macd_line, signal_line, macd_line - signal_line

    def calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range"""
        if len(close) < period:
            return pd.Series([np.nan] * len(close), index=close.index)

        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def fetch_indian_news(self, symbol, num_articles=15):
        """Fetch news for Indian companies"""
        if not self.news_api_key:
            return None

        base_symbol = symbol.split('.')[0].upper()
        stock_info = self.get_stock_info_from_db(base_symbol)
        company_name = stock_info.get("name", base_symbol)

        url = f"https://newsapi.org/v2/everything?q={company_name}+India+stock&apiKey={self.news_api_key}&pageSize={num_articles}&language=en&sortBy=publishedAt"

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = [article['title'] for article in data.get('articles', [])]
                return articles
        except Exception as e:
            pass

        return None

    def get_sample_news(self, symbol):
        """Generate sample news for demonstration"""
        base_symbol = symbol.split('.')[0]
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

    def analyze_sentiment_with_sbert(self, articles):
        """Analyze sentiment using trained SBERT model"""
        try:
            embeddings = self.vectorizer.transform(articles)
            predictions = self.model.predict(embeddings)
            probabilities = self.model.predict_proba(embeddings)

            sentiment_labels = self.label_encoder.inverse_transform(predictions)
            confidence_scores = np.max(probabilities, axis=1)

            return sentiment_labels.tolist(), confidence_scores.tolist()
        except Exception as e:
            return self.analyze_sentiment_with_textblob(articles)

    def analyze_sentiment_with_textblob(self, articles):
        """Fallback sentiment analysis using TextBlob"""
        sentiments = []
        confidences = []

        for article in articles:
            try:
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
            except Exception:
                sentiments.append('neutral')
                confidences.append(0.3)

        return sentiments, confidences

    def analyze_news_sentiment(self, symbol, num_articles=15):
        """Main sentiment analysis function"""
        articles = self.fetch_indian_news(symbol, num_articles)
        news_source = "Real news (NewsAPI)" if articles else "Sample news"

        if not articles:
            articles = self.get_sample_news(symbol)

        if self.model_loaded:
            sentiments, confidences = self.analyze_sentiment_with_sbert(articles)
            analysis_method = f"SBERT Model ({self.model_type})"
        else:
            sentiments, confidences = self.analyze_sentiment_with_textblob(articles)
            analysis_method = "TextBlob Fallback"

        return sentiments, articles, confidences, analysis_method, news_source

    def calculate_swing_trading_score(self, data, sentiment_data, sector):
        """Calculate comprehensive swing trading score"""
        tech_weight, sentiment_weight = self.get_sector_weights(sector)

        # Initialize components
        technical_score = 0
        sentiment_score = 0

        # ===== TECHNICAL ANALYSIS (Enhanced for Swing Trading) =====
        current_price = data['Close'].iloc[-1]

        # RSI Analysis (20 points)
        rsi = self.calculate_rsi(data['Close'])
        if not rsi.empty:
            current_rsi = rsi.iloc[-1]
            if 30 <= current_rsi <= 70:  # Good for swing trading
                technical_score += 20
            elif current_rsi < 30:  # Oversold - potential reversal
                technical_score += 15
            elif current_rsi > 70:  # Overbought - potential reversal
                technical_score += 10

        # Bollinger Bands Analysis (15 points)
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data['Close'])
        if not bb_upper.empty:
            bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            if 0.2 <= bb_position <= 0.8:  # Good swing trading zone
                technical_score += 15
            elif bb_position < 0.2:  # Near lower band - potential buy
                technical_score += 12
            elif bb_position > 0.8:  # Near upper band - potential sell
                technical_score += 8

        # Stochastic Analysis (15 points)
        stoch_k, stoch_d = self.calculate_stochastic(data['High'], data['Low'], data['Close'])
        if not stoch_k.empty:
            k_val = stoch_k.iloc[-1]
            d_val = stoch_d.iloc[-1]
            if k_val > d_val and k_val < 80:  # Bullish crossover
                technical_score += 15
            elif 20 <= k_val <= 80:  # Good swing range
                technical_score += 10

        # MACD Analysis (15 points)
        macd_line, signal_line, histogram = self.calculate_macd(data['Close'])
        if not macd_line.empty:
            if macd_line.iloc[-1] > signal_line.iloc[-1]:  # Bullish
                technical_score += 15
            if len(histogram) > 1 and histogram.iloc[-1] > histogram.iloc[-2]:  # Increasing momentum
                technical_score += 5

        # Volume Analysis (10 points)
        if 'Volume' in data.columns:
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            if current_volume > avg_volume * 1.2:  # Above average volume
                technical_score += 10
            elif current_volume > avg_volume:
                technical_score += 5

        # Support/Resistance Analysis (10 points)
        support, resistance = self.calculate_support_resistance(data)
        distance_to_support = (current_price - support) / support
        distance_to_resistance = (resistance - current_price) / current_price

        if distance_to_support < 0.05:  # Near support
            technical_score += 8
        elif distance_to_resistance < 0.05:  # Near resistance
            technical_score += 5
        elif 0.05 <= distance_to_support <= 0.15:  # Good swing zone
            technical_score += 10

        # Moving Average Analysis (15 points)
        if len(data) >= 50:
            ma_20 = data['Close'].rolling(20).mean().iloc[-1]
            ma_50 = data['Close'].rolling(50).mean().iloc[-1]
            if current_price > ma_20 > ma_50:  # Strong uptrend
                technical_score += 15
            elif current_price > ma_20:  # Above short-term MA
                technical_score += 10
            elif ma_20 > ma_50:  # MA alignment positive
                technical_score += 5

        # Normalize technical score to 0-100
        technical_score = min(100, technical_score)

        # ===== SENTIMENT ANALYSIS =====
        if sentiment_data:
            sentiments, _, confidences, _, _ = sentiment_data
            sentiment_value = 0
            total_weight = 0

            for sentiment, confidence in zip(sentiments, confidences):
                weight = confidence
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

        # ===== COMBINE SCORES =====
        final_score = (technical_score * tech_weight) + (sentiment_score * sentiment_weight)

        return max(0, min(100, final_score))

    def calculate_risk_metrics(self, data):
        """Calculate risk management metrics"""
        returns = data['Close'].pct_change().dropna()

        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)

        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)

        # Maximum Drawdown
        rolling_max = data['Close'].expanding().max()
        drawdown = (data['Close'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Sharpe Ratio (assuming 6% risk-free rate)
        risk_free_rate = 0.06
        excess_returns = returns.mean() * 252 - risk_free_rate
        sharpe_ratio = excess_returns / volatility if volatility > 0 else 0

        # ATR for position sizing
        atr = self.calculate_atr(data['High'], data['Low'], data['Close'])
        current_atr = atr.iloc[-1] if not atr.empty else data['Close'].iloc[-1] * 0.02

        return {
            'volatility': volatility,
            'var_95': var_95,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'atr': current_atr,
            'risk_level': 'HIGH' if volatility > 0.4 else 'MEDIUM' if volatility > 0.25 else 'LOW'
        }

    def generate_trading_plan(self, data, score, risk_metrics):
        """Generate complete trading plan"""
        current_price = data['Close'].iloc[-1]
        atr = risk_metrics['atr']

        # Entry Strategy
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

        # Position Sizing (based on 2% risk per trade)
        risk_per_trade = self.swing_trading_params['risk_per_trade']
        stop_loss_distance = atr * 2  # 2 ATR stop loss
        position_size_multiplier = risk_per_trade / (stop_loss_distance / current_price)

        # Price Targets
        stop_loss = current_price - stop_loss_distance
        target_1 = current_price + (stop_loss_distance * 1.5)  # 1.5:1 RR
        target_2 = current_price + (stop_loss_distance * 2.5)  # 2.5:1 RR
        target_3 = current_price + (stop_loss_distance * 4.0)  # 4:1 RR

        # Support and Resistance
        support, resistance = self.calculate_support_resistance(data)

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

    def analyze_swing_trading_stock(self, symbol, period="6mo"):
        """Comprehensive swing trading analysis for a single stock"""
        try:
            # Get stock data
            data, info, final_symbol = self.get_indian_stock_data(symbol, period)
            if data is None:
                return None

            # Extract information
            stock_info = self.get_stock_info_from_db(symbol)
            sector = stock_info.get('sector', 'Unknown')
            company_name = stock_info.get('name', symbol)

            # Current market data
            current_price = data['Close'].iloc[-1]
            price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            price_change_pct = (price_change / data['Close'].iloc[-2]) * 100

            # Technical indicators
            rsi = self.calculate_rsi(data['Close'])
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data['Close'])
            stoch_k, stoch_d = self.calculate_stochastic(data['High'], data['Low'], data['Close'])
            macd_line, signal_line, histogram = self.calculate_macd(data['Close'])
            support, resistance = self.calculate_support_resistance(data)
            volume_profile, poc_price = self.calculate_volume_profile(data)

            # Sentiment analysis
            sentiment_results = self.analyze_news_sentiment(final_symbol)

            # Risk metrics
            risk_metrics = self.calculate_risk_metrics(data)

            # Swing trading score
            swing_score = self.calculate_swing_trading_score(data, sentiment_results, sector)

            # Trading plan
            trading_plan = self.generate_trading_plan(data, swing_score, risk_metrics)

            # Compile results
            return {
                'symbol': final_symbol,
                'company_name': company_name,
                'sector': sector,
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'rsi': rsi.iloc[-1] if not rsi.empty else None,
                'bollinger_bands': {
                    'upper': bb_upper.iloc[-1] if not bb_upper.empty else None,
                    'middle': bb_middle.iloc[-1] if not bb_middle.empty else None,
                    'lower': bb_lower.iloc[-1] if not bb_lower.empty else None,
                    'position': ((current_price - bb_lower.iloc[-1]) / (
                                bb_upper.iloc[-1] - bb_lower.iloc[-1])) if not bb_upper.empty else None
                },
                'stochastic': {
                    'k': stoch_k.iloc[-1] if not stoch_k.empty else None,
                    'd': stoch_d.iloc[-1] if not stoch_d.empty else None
                },
                'macd': {
                    'line': macd_line.iloc[-1] if not macd_line.empty else None,
                    'signal': signal_line.iloc[-1] if not signal_line.empty else None,
                    'histogram': histogram.iloc[-1] if not histogram.empty else None
                },
                'support_resistance': {
                    'support': support,
                    'resistance': resistance,
                    'distance_to_support': ((current_price - support) / support * 100) if support else None,
                    'distance_to_resistance': (
                                (resistance - current_price) / current_price * 100) if resistance else None
                },
                'volume_profile': {
                    'poc_price': poc_price,
                    'current_vs_poc': ((current_price - poc_price) / poc_price * 100) if poc_price else None
                },
                'sentiment': {
                    'scores': sentiment_results[0],
                    'articles': sentiment_results[1],
                    'confidence': sentiment_results[2],
                    'method': sentiment_results[3],
                    'source': sentiment_results[4],
                    'sentiment_summary': {
                        'positive': sentiment_results[0].count('positive'),
                        'negative': sentiment_results[0].count('negative'),
                        'neutral': sentiment_results[0].count('neutral')
                    }
                },
                'risk_metrics': risk_metrics,
                'swing_score': swing_score,
                'trading_plan': trading_plan,
                'model_type': self.model_type,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {str(e)}")
            return None

    def analyze_multiple_stocks(self, symbols, period="6mo", max_concurrent=10):
        """Analyze multiple stocks with progress tracking"""
        results = []
        total_stocks = len(symbols)

        print(f"🔍 Analyzing {total_stocks} stocks...")

        for i, symbol in enumerate(symbols, 1):
            if i % 10 == 0:
                print(f"Progress: {i}/{total_stocks} stocks analyzed ({i / total_stocks * 100:.0f}%)")

            try:
                analysis = self.analyze_swing_trading_stock(symbol, period)
                if analysis and analysis['swing_score'] > 0:
                    results.append(analysis)
            except Exception as e:
                continue

        # Sort by swing trading score
        results.sort(key=lambda x: x['swing_score'], reverse=True)

        print(f"✅ Successfully analyzed {len(results)} stocks out of {total_stocks}")
        return results

    def filter_stocks_by_risk_appetite(self, results, risk_appetite):
        """Filter stocks based on user's risk appetite"""
        risk_thresholds = {
            'LOW': 0.25,  # ≤25% volatility
            'MEDIUM': 0.40,  # ≤40% volatility
            'HIGH': 1.0  # ≤100% volatility (all stocks)
        }

        max_volatility = risk_thresholds.get(risk_appetite.upper(), 0.40)

        filtered_stocks = [
            stock for stock in results
            if stock['risk_metrics']['volatility'] <= max_volatility and
               stock['trading_plan']['entry_signal'] in ['BUY', 'STRONG BUY']
        ]

        return filtered_stocks

    def generate_portfolio_allocation(self, results, total_capital, risk_appetite):
        """Generate risk-adjusted portfolio allocation"""
        if not results:
            print("❌ No suitable stocks found for portfolio creation")
            return None

        print(f"\n💰 PORTFOLIO ALLOCATION (₹{total_capital:,})")
        print("=" * 80)

        # Calculate total score for normalization
        total_score = sum(r['swing_score'] for r in results)

        portfolio_data = []

        print(f"{'Rank':<4} {'Symbol':<12} {'Company':<25} {'Score':<6} {'Risk':<8} {'Allocation':<12} {'Amount':<15}")
        print("-" * 88)

        for i, result in enumerate(results, 1):
            score = result['swing_score']
            allocation_pct = (score / total_score) * 100
            allocation_amount = int(total_capital * (allocation_pct / 100))

            # Adjust for position size multiplier
            position_multiplier = result['trading_plan']['position_size_multiplier']
            adjusted_amount = int(allocation_amount * min(position_multiplier, 2.0))  # Cap at 2x

            company_short = result['company_name'][:23] + "..." if len(result['company_name']) > 25 else result[
                'company_name']
            risk_level = result['risk_metrics']['risk_level']

            print(
                f"{i:<4} {result['symbol']:<12} {company_short:<25} {score:<6.0f} {risk_level:<8} {allocation_pct:<11.1f}% ₹{adjusted_amount:<14,}")

            portfolio_data.append({
                'symbol': result['symbol'],
                'company': result['company_name'],
                'score': score,
                'allocation_pct': allocation_pct,
                'amount': adjusted_amount,
                'risk_level': risk_level,
                'sector': result['sector']
            })

        # Portfolio summary
        total_allocated = sum([stock['amount'] for stock in portfolio_data])
        avg_volatility = sum(r['risk_metrics']['volatility'] for r in results) / len(results)
        avg_score = sum(r['swing_score'] for r in results) / len(results)

        # Sector diversification
        sector_allocation = {}
        for stock in portfolio_data:
            sector = stock['sector']
            if sector not in sector_allocation:
                sector_allocation[sector] = 0
            sector_allocation[sector] += stock['allocation_pct']

        print(f"\n📊 PORTFOLIO SUMMARY")
        print("-" * 50)
        print(f"Total Budget: ₹{total_capital:,}")
        print(f"Total Allocated: ₹{total_allocated:,} ({total_allocated / total_capital * 100:.1f}%)")
        print(f"Number of Stocks: {len(results)}")
        print(f"Average Score: {avg_score:.1f}/100")
        print(f"Average Volatility: {avg_volatility * 100:.1f}%")
        print(f"Portfolio Risk Level: {risk_appetite}")

        print(f"\n🏭 SECTOR DIVERSIFICATION")
        print("-" * 30)
        for sector, allocation in sorted(sector_allocation.items(), key=lambda x: x[1], reverse=True):
            print(f"{sector}: {allocation:.1f}%")

        return portfolio_data

    def get_single_best_recommendation(self, results):
        """Get detailed recommendation for the single best stock"""
        if not results:
            return None

        best_stock = results[0]  # Highest scoring stock

        print(f"\n🏆 SINGLE BEST STOCK RECOMMENDATION")
        print("=" * 70)

        print(f"🏢 Company: {best_stock['company_name']}")
        print(f"📊 Symbol: {best_stock['symbol']}")
        print(f"🏭 Sector: {best_stock['sector']}")
        print(f"🎯 Swing Score: {best_stock['swing_score']:.0f}/100")
        print(f"💰 Current Price: ₹{best_stock['current_price']:.2f}")
        print(f"📈 Price Change: ₹{best_stock['price_change']:.2f} ({best_stock['price_change_pct']:.2f}%)")
        print(f"⚠️ Risk Level: {best_stock['risk_metrics']['risk_level']}")

        # Trading recommendation
        tp = best_stock['trading_plan']
        print(f"\n🎯 TRADING RECOMMENDATION")
        print("-" * 30)
        print(f"Signal: {tp['entry_signal']}")
        print(f"Strategy: {tp['entry_strategy']}")
        print(f"Stop Loss: ₹{tp['stop_loss']:.2f}")
        print(f"Target 1: ₹{tp['targets']['target_1']:.2f}")
        print(f"Target 2: ₹{tp['targets']['target_2']:.2f}")
        print(f"Target 3: ₹{tp['targets']['target_3']:.2f}")
        print(f"Holding Period: {tp['holding_period']}")

        # Key technical levels
        print(f"\n📊 KEY LEVELS")
        print("-" * 15)
        print(f"Support: ₹{tp['support']:.2f}")
        print(f"Resistance: ₹{tp['resistance']:.2f}")
        if best_stock['rsi']:
            print(f"RSI: {best_stock['rsi']:.1f}")

        # Sentiment summary
        sentiment = best_stock['sentiment']['sentiment_summary']
        print(f"\n💭 SENTIMENT OVERVIEW")
        print("-" * 20)
        print(f"Positive: {sentiment['positive']}, Negative: {sentiment['negative']}, Neutral: {sentiment['neutral']}")

        return best_stock

    def print_analysis_summary(self, all_results, filtered_results, risk_appetite, total_budget):
        """Print comprehensive analysis summary"""
        print(f"\n📈 MARKET ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total Stocks Analyzed: {len(all_results)}")
        print(f"Risk Appetite: {risk_appetite}")
        print(f"Budget: ₹{total_budget:,}")
        print(f"Suitable Stocks Found: {len(filtered_results)}")

        if len(all_results) > 0:
            avg_market_score = sum(r['swing_score'] for r in all_results) / len(all_results)
            print(f"Average Market Score: {avg_market_score:.1f}/100")

            # Risk distribution
            risk_distribution = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
            for result in all_results:
                risk_level = result['risk_metrics']['risk_level']
                risk_distribution[risk_level] += 1

            print(f"\n⚠️ MARKET RISK DISTRIBUTION")
            print("-" * 25)
            for risk, count in risk_distribution.items():
                percentage = (count / len(all_results)) * 100
                print(f"{risk} Risk: {count} stocks ({percentage:.1f}%)")


# ========================= MAIN EXECUTION =========================

if __name__ == "__main__":
    # Initialize the enhanced swing trading system
    swing_trader = EnhancedSwingTradingSystem(
        model_path="D:/Python_files/models/sentiment_pipeline.joblib",
        news_api_key=os.getenv("NEWS_API_KEY")
    )

    print("🚀 ENHANCED SWING TRADING SYSTEM")
    print("Advanced Portfolio Creation with Budget & Risk Management")
    print("=" * 70)

    # ===== USER INPUT COLLECTION =====

    # Get user budget
    while True:
        try:
            budget_input = input("\n💰 Enter your total investment budget in INR (e.g., 500000): ").strip()
            total_budget = float(budget_input)
            if total_budget <= 0:
                print("❌ Please enter a positive number for budget.")
                continue
            break
        except ValueError:
            print("❌ Invalid input. Please enter a numeric value.")

    # Get user risk appetite
    risk_levels = ['LOW', 'MEDIUM', 'HIGH']
    print(f"\n⚠️ Risk Appetite Options:")
    print("• LOW: Conservative (≤25% volatility) - Blue chip stocks")
    print("• MEDIUM: Balanced (≤40% volatility) - Mixed portfolio")
    print("• HIGH: Aggressive (≤100% volatility) - All opportunities")

    while True:
        risk_appetite = input("\nEnter your risk appetite (LOW/MEDIUM/HIGH): ").upper().strip()
        if risk_appetite not in risk_levels:
            print("❌ Invalid risk level. Please enter LOW, MEDIUM, or HIGH.")
        else:
            break

    print(f"\n✅ Configuration Set:")
    print(f"Budget: ₹{total_budget:,.0f}")
    print(f"Risk Appetite: {risk_appetite}")

    # ===== COMPREHENSIVE MARKET ANALYSIS =====

    print(f"\n🔍 ANALYZING INDIAN STOCK MARKET...")
    print(f"Scanning {len(swing_trader.get_all_stock_symbols())} stocks across BSE & NSE")
    print("This may take several minutes...")

    # Get all stock symbols from database
    all_symbols = swing_trader.get_all_stock_symbols()

    # For demo purposes, you can limit the analysis to top stocks
    # Remove this limitation for full market analysis
    # top_symbols = all_symbols[:50]  # Analyze top 50 stocks for demo
    # print(f"Demo mode: Analyzing top {len(top_symbols)} stocks")

    # Analyze all stocks (or top_symbols for demo)
    start_time = datetime.now()
    all_results = swing_trader.analyze_multiple_stocks(all_symbols)  # Use all_symbols for full analysis
    analysis_time = datetime.now() - start_time

    print(f"⏱️ Analysis completed in {analysis_time.total_seconds():.0f} seconds")

    # ===== RISK-BASED FILTERING =====

    print(f"\n🎯 FILTERING STOCKS BY RISK APPETITE...")
    filtered_results = swing_trader.filter_stocks_by_risk_appetite(all_results, risk_appetite)

    if not filtered_results:
        print(f"\n❌ No suitable stocks found matching your criteria:")
        print(f"• Risk Appetite: {risk_appetite}")
        print(f"• Minimum Signal: BUY or STRONG BUY")
        print("\n💡 Suggestions:")
        print("• Consider increasing your risk tolerance")
        print("• Try a different time period")
        print("• Check market conditions")
    else:
        # ===== PORTFOLIO CREATION =====

        print(f"\n✅ Found {len(filtered_results)} suitable investment opportunities")

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

        print(f"\n🏆 TOP 10 STOCK RANKINGS")
        print("=" * 70)
        print(f"{'Rank':<4} {'Symbol':<12} {'Company':<20} {'Score':<6} {'Signal':<12} {'Risk':<8}")
        print("-" * 70)

        for i, result in enumerate(filtered_results[:10], 1):
            company_short = result['company_name'][:18] + "..." if len(result['company_name']) > 20 else result[
                'company_name']
            print(
                f"{i:<4} {result['symbol']:<12} {company_short:<20} {result['swing_score']:<6.0f} {result['trading_plan']['entry_signal']:<12} {result['risk_metrics']['risk_level']:<8}")

    # ===== INTERACTIVE MODE =====

    # ===== INTERACTIVE MODE ===== (Replace your existing interactive section)

    print(f"\n🎮 INTERACTIVE MODE")
    print("Available commands:")
    print("• Enter stock symbol for detailed analysis")
    print("• Type 'portfolio' for portfolio analysis")
    print("• Type 'full' for complete 174-stock analysis")
    print("• Type 'settings' to change budget/risk settings")
    print("• Type 'quit' to exit")

    while True:
        try:
            user_input = input("\n> Enter command: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using Enhanced Swing Trading System!")
                print("Happy Trading! 📈")
                break

            elif user_input.lower() == 'full':
                # Complete analysis of all 174 stocks
                print(f"🚀 COMPLETE MARKET ANALYSIS")
                print(f"Analyzing all {len(swing_trader.get_all_stock_symbols())} stocks...")
                print("⏱️ Estimated time: 5-10 minutes")

                confirm = input("Continue with full analysis? (y/n): ")
                if confirm.lower() == 'y':
                    all_symbols = swing_trader.get_all_stock_symbols()
                    complete_results = swing_trader.analyze_multiple_stocks(all_symbols)
                    complete_filtered = swing_trader.filter_stocks_by_risk_appetite(complete_results, risk_appetite)

                    if complete_filtered:
                        print(f"\n🎯 COMPLETE MARKET PORTFOLIO")
                        swing_trader.generate_portfolio_allocation(complete_filtered, int(total_budget), risk_appetite)
                        swing_trader.get_single_best_recommendation(complete_filtered)
                    else:
                        print("No suitable stocks found matching your criteria")

            elif user_input.lower() == 'portfolio':
                symbols = input("Enter stock symbols (comma-separated) or 'all' for analysis: ").strip()

                if symbols.lower() == 'all':
                    print(f"\n📊 Analysis Options:")
                    print(f"1. Quick (30 stocks) - 1-2 minutes")
                    print(f"2. Extended (100 stocks) - 3-5 minutes")
                    print(f"3. Complete ({len(swing_trader.get_all_stock_symbols())} stocks) - 5-10 minutes")

                    choice = input("Select option (1/2/3): ").strip()

                    if choice == '1':
                        symbols_list = swing_trader.get_all_stock_symbols()[:30]
                    elif choice == '2':
                        symbols_list = swing_trader.get_all_stock_symbols()[:100]
                    else:  # choice == '3' or default
                        symbols_list = swing_trader.get_all_stock_symbols()
                else:
                    symbols_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]

                if symbols_list:
                    interactive_results = swing_trader.analyze_multiple_stocks(symbols_list)
                    filtered_interactive = swing_trader.filter_stocks_by_risk_appetite(interactive_results,
                                                                                       risk_appetite)
                    if filtered_interactive:
                        swing_trader.generate_portfolio_allocation(filtered_interactive, int(total_budget),
                                                                   risk_appetite)
                    else:
                        print("No suitable stocks found in your selection")

            # ... rest of your interactive mode code remains the same

        except KeyboardInterrupt:
            print("\n👋 Thank you for using Enhanced Swing Trading System!")
            print("Happy Trading! 📈")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
