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
    print("⚠️  sentence-transformers not available. Install with: pip install sentence-transformers")
    SBERT_AVAILABLE = False


    class SBERTTransformer:
        def __init__(self, model_name='all-MiniLM-L6-v2'):
            self.model_name = model_name
            print(f"⚠️  SBERTTransformer created but sentence-transformers not available")

        def transform(self, sentences):
            raise ImportError("sentence-transformers not installed")

        def fit(self, X, y=None):
            return self


class EnhancedSwingTradingSystem:
    """🚀 Enhanced Swing Trading System for Indian Markets"""

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
            print("⚠️  NEWS_API_KEY not provided. Using sample news for sentiment analysis.")
        else:
            print("✅ News API key available. Will fetch real news articles.")

        # Load sentiment model
        self.load_sbert_model(model_path)

    def load_sbert_model(self, model_path):
        """Load trained SBERT sentiment model"""
        if not SBERT_AVAILABLE:
            print("⚠️  sentence-transformers not available, using TextBlob fallback")
            self.model_type = "TextBlob"
            return

        if not os.path.exists(model_path):
            print(f"⚠️  SBERT model not found at {model_path}")
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
                print("⚠️  Model components incomplete, using TextBlob fallback")
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
                print(
                    f"⚖️  Swing Trading Weights: {tech_weight * 100:.0f}% Technical, {sentiment_weight * 100:.0f}% Sentiment")
                return tech_weight, sentiment_weight

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
                    print(f"✅ Successfully fetched data for {sym}")
                    return data, info, sym
            except Exception as e:
                continue

        print(f"❌ No valid data found for {symbol}")
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

        company_names = {
            "RELIANCE": "Reliance Industries", "TCS": "Tata Consultancy Services",
            "INFY": "Infosys", "HDFCBANK": "HDFC Bank", "BAJFINANCE": "Bajaj Finance",
            "HINDUNILVR": "Hindustan Unilever", "ICICIBANK": "ICICI Bank",
            "KOTAKBANK": "Kotak Mahindra Bank", "SBIN": "State Bank of India",
            "BHARTIARTL": "Bharti Airtel", "LT": "Larsen & Toubro",
            "MARUTI": "Maruti Suzuki", "ASIANPAINT": "Asian Paints",
            "HCLTECH": "HCL Technologies", "TITAN": "Titan Company",
            "SUNPHARMA": "Sun Pharmaceutical", "NTPC": "NTPC Limited",
            "ONGC": "Oil and Natural Gas Corporation", "ADANIENT": "Adani Enterprises"
        }

        query = company_names.get(base_symbol, base_symbol)
        url = f"https://newsapi.org/v2/everything?q={query}+India+stock&apiKey={self.news_api_key}&pageSize={num_articles}&language=en&sortBy=publishedAt"

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = [article['title'] for article in data.get('articles', [])]
                print(f"✅ Fetched {len(articles)} real news articles")
                return articles
        except Exception as e:
            print(f"❌ Error fetching news: {str(e)[:50]}...")
        return None

    def get_sample_news(self, symbol):
        """Generate sample news for demonstration"""
        company_name = symbol.split('.')[0]
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
            print(f"❌ Error in SBERT sentiment analysis: {str(e)}")
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

        # Market Timing
        rsi = self.calculate_rsi(data['Close'])
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data['Close'])

        timing_advice = []
        if not rsi.empty:
            current_rsi = rsi.iloc[-1]
            if current_rsi < 35:
                timing_advice.append("RSI oversold - good entry opportunity")
            elif current_rsi > 65:
                timing_advice.append("RSI overbought - wait for pullback")

        if not bb_upper.empty:
            bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            if bb_position < 0.3:
                timing_advice.append("Near BB lower band - potential reversal")
            elif bb_position > 0.7:
                timing_advice.append("Near BB upper band - potential resistance")

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
            'timing_advice': timing_advice,
            'holding_period': f"{self.swing_trading_params['min_holding_period']}-{self.swing_trading_params['max_holding_period']} days"
        }

    def get_market_timing_signals(self, data):
        """Generate market timing signals for swing trading"""
        signals = []

        # RSI Timing
        rsi = self.calculate_rsi(data['Close'])
        if not rsi.empty:
            current_rsi = rsi.iloc[-1]
            prev_rsi = rsi.iloc[-2] if len(rsi) > 1 else current_rsi

            if current_rsi < 30 and prev_rsi >= 30:
                signals.append("🟢 RSI just entered oversold - BUY signal")
            elif current_rsi > 70 and prev_rsi <= 70:
                signals.append("🔴 RSI just entered overbought - SELL signal")
            elif 30 <= current_rsi <= 70:
                signals.append("🟡 RSI in neutral zone - HOLD")

        # Bollinger Band Timing
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data['Close'])
        if not bb_upper.empty:
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price

            if prev_price <= bb_lower.iloc[-2] and current_price > bb_lower.iloc[-1]:
                signals.append("🟢 BB Bounce from lower band - BUY signal")
            elif prev_price >= bb_upper.iloc[-2] and current_price < bb_upper.iloc[-1]:
                signals.append("🔴 BB Rejection from upper band - SELL signal")
            elif bb_lower.iloc[-1] < current_price < bb_upper.iloc[-1]:
                signals.append("🟡 Price within BB range - NEUTRAL")

        # Stochastic Timing
        stoch_k, stoch_d = self.calculate_stochastic(data['High'], data['Low'], data['Close'])
        if not stoch_k.empty and len(stoch_k) > 1:
            k_curr, k_prev = stoch_k.iloc[-1], stoch_k.iloc[-2]
            d_curr, d_prev = stoch_d.iloc[-1], stoch_d.iloc[-2]

            if k_prev <= d_prev and k_curr > d_curr and k_curr < 80:
                signals.append("🟢 Stochastic bullish crossover - BUY signal")
            elif k_prev >= d_prev and k_curr < d_curr and k_curr > 20:
                signals.append("🔴 Stochastic bearish crossover - SELL signal")

        # MACD Timing
        macd_line, signal_line, histogram = self.calculate_macd(data['Close'])
        if not macd_line.empty and len(macd_line) > 1:
            macd_curr, macd_prev = macd_line.iloc[-1], macd_line.iloc[-2]
            signal_curr, signal_prev = signal_line.iloc[-1], signal_line.iloc[-2]

            if macd_prev <= signal_prev and macd_curr > signal_curr:
                signals.append("🟢 MACD bullish crossover - BUY signal")
            elif macd_prev >= signal_prev and macd_curr < signal_curr:
                signals.append("🔴 MACD bearish crossover - SELL signal")

        # Volume Confirmation
        if 'Volume' in data.columns and len(data) > 20:
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            current_volume = data['Volume'].iloc[-1]

            if current_volume > avg_volume * 1.5:
                signals.append("🟢 High volume confirms move")
            elif current_volume < avg_volume * 0.7:
                signals.append("🟡 Low volume - weak confirmation")

        return signals

    def analyze_swing_trading_stock(self, symbol, period="6mo"):
        """Comprehensive swing trading analysis"""
        print(f"\n🚀 SWING TRADING ANALYSIS: {symbol.upper()}")
        print("=" * 70)

        # Get stock data
        data, info, final_symbol = self.get_indian_stock_data(symbol, period)
        if data is None:
            print(f"❌ Could not fetch data for {symbol}")
            return None

        # Extract information
        sector = info.get('sector', 'Unknown') if info else 'Unknown'
        company_name = info.get('shortName', symbol) if info else symbol

        print(f"🏢 Company: {company_name}")
        print(f"📊 Symbol: {final_symbol}")
        print(f"🏭 Sector: {sector}")

        # Current market data
        current_price = data['Close'].iloc[-1]
        price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
        price_change_pct = (price_change / data['Close'].iloc[-2]) * 100

        print(f"💰 Current Price: ₹{current_price:.2f}")
        print(f"📈 Price Change: ₹{price_change:.2f} ({price_change_pct:.2f}%)")

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

        # Market timing signals
        timing_signals = self.get_market_timing_signals(data)

        # Compile comprehensive results
        results = {
            'symbol': final_symbol,
            'company_name': company_name,
            'sector': sector,
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,

            # Technical Indicators
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
                'distance_to_resistance': ((resistance - current_price) / current_price * 100) if resistance else None
            },
            'volume_profile': {
                'poc_price': poc_price,
                'current_vs_poc': ((current_price - poc_price) / poc_price * 100) if poc_price else None
            },

            # Sentiment Analysis
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

            # Risk Metrics
            'risk_metrics': risk_metrics,

            # Swing Trading Score
            'swing_score': swing_score,

            # Trading Plan
            'trading_plan': trading_plan,

            # Market Timing
            'timing_signals': timing_signals,

            # Model Info
            'model_type': self.model_type,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return results

    def print_comprehensive_analysis(self, results):
        """Print detailed swing trading analysis"""
        if not results:
            print("❌ Analysis failed")
            return

        print(f"\n🎯 SWING TRADING SCORE: {results['swing_score']:.0f}/100")
        print(f"📊 Risk Level: {results['risk_metrics']['risk_level']}")
        print(f"🤖 Sentiment Model: {results['model_type']}")

        # Technical Analysis Section
        print(f"\n📈 TECHNICAL ANALYSIS")
        print("-" * 50)

        if results['rsi']:
            rsi_status = "OVERSOLD" if results['rsi'] < 30 else "OVERBOUGHT" if results['rsi'] > 70 else "NEUTRAL"
            print(f"RSI (14): {results['rsi']:.1f} - {rsi_status}")

        if results['bollinger_bands']['upper']:
            bb = results['bollinger_bands']
            bb_status = "NEAR LOWER" if bb['position'] < 0.3 else "NEAR UPPER" if bb[
                                                                                      'position'] > 0.7 else "MIDDLE RANGE"
            print(f"Bollinger Bands: Upper ₹{bb['upper']:.2f}, Middle ₹{bb['middle']:.2f}, Lower ₹{bb['lower']:.2f}")
            print(f"BB Position: {bb['position']:.2f} - {bb_status}")

        if results['stochastic']['k']:
            stoch = results['stochastic']
            stoch_status = "OVERSOLD" if stoch['k'] < 20 else "OVERBOUGHT" if stoch['k'] > 80 else "NEUTRAL"
            print(f"Stochastic: K={stoch['k']:.1f}, D={stoch['d']:.1f} - {stoch_status}")

        if results['macd']['line']:
            macd = results['macd']
            macd_status = "BULLISH" if macd['line'] > macd['signal'] else "BEARISH"
            print(f"MACD: Line={macd['line']:.4f}, Signal={macd['signal']:.4f} - {macd_status}")

        # Support/Resistance
        sr = results['support_resistance']
        print(f"Support: ₹{sr['support']:.2f} (Distance: {sr['distance_to_support']:.1f}%)")
        print(f"Resistance: ₹{sr['resistance']:.2f} (Distance: {sr['distance_to_resistance']:.1f}%)")

        # Volume Profile
        vp = results['volume_profile']
        if vp['poc_price']:
            print(f"Volume POC: ₹{vp['poc_price']:.2f} (Current vs POC: {vp['current_vs_poc']:.1f}%)")

        # Sentiment Analysis
        print(f"\n💭 SENTIMENT ANALYSIS ({results['sentiment']['method']})")
        print("-" * 50)
        sentiment_summary = results['sentiment']['sentiment_summary']
        print(
            f"Positive: {sentiment_summary['positive']}, Negative: {sentiment_summary['negative']}, Neutral: {sentiment_summary['neutral']}")
        print(f"News Source: {results['sentiment']['source']}")

        # Risk Metrics
        print(f"\n⚠️  RISK METRICS")
        print("-" * 50)
        rm = results['risk_metrics']
        print(f"Volatility: {rm['volatility'] * 100:.1f}% (Annualized)")
        print(f"Max Drawdown: {rm['max_drawdown'] * 100:.1f}%")
        print(f"Sharpe Ratio: {rm['sharpe_ratio']:.2f}")
        print(f"Value at Risk (95%): {rm['var_95'] * 100:.1f}%")
        print(f"ATR: ₹{rm['atr']:.2f}")

        # Trading Plan
        print(f"\n🎯 TRADING PLAN")
        print("-" * 50)
        tp = results['trading_plan']
        print(f"Signal: {tp['entry_signal']}")
        print(f"Strategy: {tp['entry_strategy']}")
        print(f"Position Size: {tp['position_size_multiplier']:.2f}x of normal allocation")
        print(f"Stop Loss: ₹{tp['stop_loss']:.2f}")
        print(f"Target 1: ₹{tp['targets']['target_1']:.2f} (1.5:1 RR)")
        print(f"Target 2: ₹{tp['targets']['target_2']:.2f} (2.5:1 RR)")
        print(f"Target 3: ₹{tp['targets']['target_3']:.2f} (4.0:1 RR)")
        print(f"Holding Period: {tp['holding_period']}")

        # Market Timing Signals
        print(f"\n⏰ MARKET TIMING SIGNALS")
        print("-" * 50)
        for signal in results['timing_signals']:
            print(f"  {signal}")

        if not results['timing_signals']:
            print("  No clear timing signals at the moment")

        # Recent News Headlines
        print(f"\n📰 RECENT NEWS SENTIMENT")
        print("-" * 50)
        for i, article in enumerate(results['sentiment']['articles'][:8], 1):
            sentiment = results['sentiment']['scores'][i - 1]
            confidence = results['sentiment']['confidence'][i - 1]
            emoji = "🟢" if sentiment == 'positive' else "🔴" if sentiment == 'negative' else "🟡"
            print(f"  {i}. {emoji} [{sentiment.upper()} {confidence:.2f}] {article[:80]}...")

        # Investment Recommendation
        print(f"\n🏆 FINAL RECOMMENDATION")
        print("=" * 50)

        if results['swing_score'] >= 75:
            print("🟢 STRONG BUY - Excellent swing trading opportunity")
            print("   Multiple technical indicators align with positive sentiment")
        elif results['swing_score'] >= 60:
            print("🟢 BUY - Good swing trading setup")
            print("   Favorable technical setup with decent risk-reward ratio")
        elif results['swing_score'] >= 45:
            print("🟡 HOLD/WATCH - Wait for better entry")
            print("   Mixed signals, monitor for clearer direction")
        elif results['swing_score'] >= 30:
            print("🔴 SELL - Negative outlook")
            print("   Technical and sentiment indicators suggest downside risk")
        else:
            print("🔴 STRONG SELL - Avoid or exit positions")
            print("   Multiple negative indicators, high risk scenario")

    def analyze_multiple_stocks(self, symbols, period="6mo"):
        """Analyze multiple stocks and rank them for swing trading"""
        print(f"\n🎯 SWING TRADING PORTFOLIO ANALYSIS")
        print("=" * 70)

        results = []

        for symbol in symbols:
            print(f"\n🔍 Analyzing {symbol}...")
            try:
                analysis = self.analyze_swing_trading_stock(symbol, period)
                if analysis:
                    results.append(analysis)
            except Exception as e:
                print(f"❌ Error analyzing {symbol}: {e}")
                continue

        # Sort by swing trading score
        results.sort(key=lambda x: x['swing_score'], reverse=True)

        # Print ranking
        print(f"\n🏆 SWING TRADING RANKINGS")
        print("=" * 70)
        print(f"{'Rank':<4} {'Symbol':<12} {'Company':<20} {'Score':<6} {'Signal':<12} {'Risk':<8}")
        print("-" * 70)

        for i, result in enumerate(results, 1):
            symbol = result['symbol']
            company = result['company_name'][:18] + "..." if len(result['company_name']) > 20 else result[
                'company_name']
            score = result['swing_score']
            signal = result['trading_plan']['entry_signal']
            risk = result['risk_metrics']['risk_level']

            print(f"{i:<4} {symbol:<12} {company:<20} {score:<6.0f} {signal:<12} {risk:<8}")

        return results

    def generate_portfolio_allocation(self, results, total_capital="100000"):
        """Generate portfolio allocation based on swing trading scores"""
        print(f"\n💰 PORTFOLIO ALLOCATION (₹{total_capital:,})")
        print("=" * 70)

        # Filter only BUY signals
        buy_candidates = [r for r in results if r['trading_plan']['entry_signal'] in ['BUY', 'STRONG BUY']]

        if not buy_candidates:
            print("❌ No BUY signals found in current analysis")
            return

        # Calculate total score for normalization
        total_score = sum(r['swing_score'] for r in buy_candidates)

        print(f"{'Symbol':<12} {'Score':<6} {'Allocation':<12} {'Amount':<12} {'Position Size':<12}")
        print("-" * 70)

        for result in buy_candidates:
            score = result['swing_score']
            allocation_pct = (score / total_score) * 100
            allocation_amount = total_capital * (allocation_pct / 100)

            # Adjust for position size multiplier
            position_multiplier = result['trading_plan']['position_size_multiplier']
            adjusted_amount = allocation_amount * position_multiplier

            print(
                f"{result['symbol']:<12} {score:<6.0f} {allocation_pct:<12.1f}% ₹{allocation_amount:<11,.0f} {position_multiplier:<12.2f}x")

        # Risk summary
        total_risk = sum(r['risk_metrics']['volatility'] for r in buy_candidates) / len(buy_candidates)
        print(f"\n📊 Portfolio Risk Summary:")
        print(f"   Average Volatility: {total_risk * 100:.1f}%")
        print(f"   Number of Positions: {len(buy_candidates)}")
        print(f"   Diversification Score: {min(100, len(buy_candidates) * 20)}/100")


# Example usage and testing
if __name__ == "__main__":
    # Initialize the enhanced swing trading system
    swing_trader = EnhancedSwingTradingSystem(
        model_path="D:/Python_files/models/sentiment_pipeline.joblib",
        news_api_key=os.getenv("NEWS_API_KEY")
    )

    print("🚀 Enhanced Swing Trading System Demo")
    print("This system adds:")
    print("• Bollinger Bands analysis")
    print("• Stochastic oscillator")
    print("• Support/Resistance levels")
    print("• Volume profile analysis")
    print("• Risk management metrics")
    print("• Complete trading plans")
    print("• Market timing advice")

    # Test with popular Indian stocks
    test_stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "BAJFINANCE"]

    print(f"\n🎯 Demo: Single Stock Analysis")
    print("=" * 50)

    # Single stock detailed analysis
    demo_stock = "RELIANCE"
    results = swing_trader.analyze_swing_trading_stock(demo_stock)
    if results:
        swing_trader.print_comprehensive_analysis(results)

    print(f"\n🎯 Demo: Multiple Stock Ranking")
    print("=" * 50)

    # Multiple stock analysis and ranking
    portfolio_results = swing_trader.analyze_multiple_stocks(test_stocks)

    # Generate portfolio allocation
    swing_trader.generate_portfolio_allocation(portfolio_results, total_capital=500000)

    # Interactive mode
    print(f"\n🎮 Interactive Mode")
    print("Commands:")
    print("• Enter stock symbol for detailed analysis")
    print("• Type 'portfolio' to analyze multiple stocks")
    print("• Type 'quit' to exit")

    while True:
        try:
            user_input = input("\nEnter command: ").strip()

            if user_input.lower() in ['quit', 'exit']:
                print("👋 Happy Trading!")
                break
            elif user_input.lower() == 'portfolio':
                symbols = input("Enter stock symbols (comma-separated): ").strip().split(',')
                symbols = [s.strip().upper() for s in symbols if s.strip()]
                if symbols:
                    portfolio_results = swing_trader.analyze_multiple_stocks(symbols)
                    swing_trader.generate_portfolio_allocation(portfolio_results)
            elif user_input:
                results = swing_trader.analyze_swing_trading_stock(user_input)
                if results:
                    swing_trader.print_comprehensive_analysis(results)

        except KeyboardInterrupt:
            print("\n👋 Happy Trading!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
