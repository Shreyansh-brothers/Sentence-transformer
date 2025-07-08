import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import random
import time
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import re
from textblob import TextBlob
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


class EnhancedSentimentAnalyzer:
    """Enhanced sentiment analyzer with multiple methods"""

    def __init__(self):
        self.positive_keywords = {
            'growth': 2, 'profit': 2, 'revenue': 1.5, 'earnings': 1.5,
            'expansion': 1.5, 'acquisition': 1, 'merger': 1, 'partnership': 1,
            'breakthrough': 2, 'outperform': 2, 'beat': 1.5, 'exceed': 1.5,
            'strong': 1, 'robust': 1.5, 'solid': 1, 'positive': 1,
            'bullish': 2, 'rally': 1.5, 'surge': 1.5, 'jump': 1.5,
            'upgrade': 2, 'buy': 1.5, 'recommendation': 1, 'upside': 1.5,
            'target': 1, 'optimistic': 1.5, 'confidence': 1
        }

        self.negative_keywords = {
            'loss': -2, 'decline': -1.5, 'fall': -1.5, 'drop': -1.5,
            'crash': -2, 'plunge': -2, 'slump': -1.5, 'weak': -1,
            'poor': -1.5, 'disappointing': -1.5, 'miss': -1.5,
            'bearish': -2, 'sell': -1.5, 'downgrade': -2, 'avoid': -1.5,
            'concern': -1, 'worry': -1, 'trouble': -1.5, 'challenge': -1,
            'debt': -1, 'lawsuit': -1.5, 'investigation': -1.5, 'fraud': -2,
            'risk': -1, 'volatile': -1, 'uncertainty': -1
        }

    def preprocess_text(self, text):
        """Clean and preprocess text for analysis"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\!\?\-\%\$\₹]', ' ', text)

        # Normalize financial numbers
        text = re.sub(r'\d+\.\d+', 'NUMBER', text)
        text = re.sub(r'\d+', 'NUMBER', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def extract_financial_keywords(self, text):
        """Extract financial sentiment keywords with weights"""
        text_lower = text.lower()
        keyword_score = 0

        for word, weight in self.positive_keywords.items():
            if word in text_lower:
                keyword_score += weight

        for word, weight in self.negative_keywords.items():
            if word in text_lower:
                keyword_score += weight

        return keyword_score

    def analyze_sentiment_multiple_methods(self, text):
        """Combine multiple sentiment analysis methods"""
        results = {}

        # Method 1: TextBlob sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            results['textblob'] = 'positive'
        elif polarity < -0.1:
            results['textblob'] = 'negative'
        else:
            results['textblob'] = 'neutral'

        # Method 2: Financial keyword analysis
        keyword_score = self.extract_financial_keywords(text)
        if keyword_score > 1:
            results['keywords'] = 'positive'
        elif keyword_score < -1:
            results['keywords'] = 'negative'
        else:
            results['keywords'] = 'neutral'

        # Method 3: Financial context analysis
        financial_context = self.analyze_financial_context(text)
        results['context'] = financial_context

        return results

    def analyze_financial_context(self, text):
        """Analyze financial context and numbers"""
        text_lower = text.lower()

        # Check for financial performance indicators
        if any(word in text_lower for word in ['q4', 'quarterly', 'annual', 'fy']):
            if any(word in text_lower for word in ['beat', 'exceed', 'above', 'better than']):
                return 'positive'
            elif any(word in text_lower for word in ['miss', 'below', 'disappointing']):
                return 'negative'

        # Check for guidance and outlook
        if any(word in text_lower for word in ['guidance', 'outlook', 'forecast']):
            if any(word in text_lower for word in ['raised', 'increased', 'positive', 'optimistic']):
                return 'positive'
            elif any(word in text_lower for word in ['lowered', 'reduced', 'negative', 'cautious']):
                return 'negative'

        # Check for analyst recommendations
        if any(word in text_lower for word in ['analyst', 'brokerage', 'recommendation']):
            if any(word in text_lower for word in ['buy', 'upgrade', 'outperform']):
                return 'positive'
            elif any(word in text_lower for word in ['sell', 'downgrade', 'underperform']):
                return 'negative'

        return 'neutral'

    def ensemble_prediction(self, text):
        """Combine multiple methods for final prediction"""
        methods = self.analyze_sentiment_multiple_methods(text)

        # Weighted voting
        weights = {
            'textblob': 0.3,
            'keywords': 0.4,
            'context': 0.3
        }

        scores = {'positive': 0, 'negative': 0, 'neutral': 0}

        for method, sentiment in methods.items():
            if method in weights:
                scores[sentiment] += weights[method]

        # Return prediction with confidence
        max_score = max(scores.values())
        prediction = max(scores, key=scores.get)
        confidence = max_score

        return prediction, confidence, methods


class NewsScraperService:
    """Enhanced news scraper for financial articles"""

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def get_articles_enhanced(self, company_name, symbol, max_articles=5):
        """Get enhanced articles from multiple sources"""
        all_articles = []

        # Search queries
        search_queries = [
            f"{company_name} stock news",
            f"{company_name} earnings",
            f"{company_name} financial results",
            f"{symbol} NSE news",
            f"{company_name} market update"
        ]

        for query in search_queries:
            try:
                articles = self.search_google_news(query, max_results=2)
                all_articles.extend(articles)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"⚠️ Error searching for {query}: {e}")
                continue

        # Remove duplicates and get unique articles
        unique_articles = []
        seen_titles = set()

        for article in all_articles:
            title = article.get('title', '').lower()
            if title not in seen_titles and len(title) > 10:
                seen_titles.add(title)
                unique_articles.append(article)

        return unique_articles[:max_articles]

    def search_google_news(self, query, max_results=5):
        """Search Google News for articles"""
        articles = []

        try:
            encoded_query = quote(query)
            url = f"https://news.google.com/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN%3Aen"

            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Find article elements
                article_elements = soup.find_all('article', {'class': 'IBr9hb'})

                for element in article_elements[:max_results]:
                    try:
                        title_elem = element.find('h3')
                        title = title_elem.get_text(strip=True) if title_elem else None

                        # Get the link
                        link_elem = element.find('a')
                        link = link_elem.get('href') if link_elem else None

                        if link and link.startswith('./'):
                            link = f"https://news.google.com{link[1:]}"

                        # Get source and time
                        source_elem = element.find('div', {'class': 'vr1PYe'})
                        source = source_elem.get_text(strip=True) if source_elem else "Unknown"

                        if title and len(title) > 10:
                            # Try to extract content
                            content = self.extract_article_content(link) if link else title

                            articles.append({
                                'title': title,
                                'content': content or title,
                                'source': source,
                                'url': link,
                                'query': query
                            })

                    except Exception as e:
                        print(f"⚠️ Error parsing article element: {e}")
                        continue

        except Exception as e:
            print(f"⚠️ Error searching Google News: {e}")

        return articles

    def extract_article_content(self, url):
        """Extract article content from URL"""
        try:
            if not url or 'google.com' in url:
                return None

            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'advertisement']):
                element.decompose()

            # Try different selectors for content
            selectors = [
                'article p', '.article-content p', '.story-content p',
                '.article-body p', '.content p', '.story p',
                '.post-content p', '.entry-content p', 'p'
            ]

            best_content = None
            max_length = 0

            for selector in selectors:
                paragraphs = soup.select(selector)
                if paragraphs:
                    text = " ".join([p.get_text(strip=True) for p in paragraphs])
                    text = " ".join(text.split())  # Clean whitespace

                    if len(text) > max_length and len(text) > 100:
                        max_length = len(text)
                        best_content = text

            return best_content[:3000] if best_content else None

        except Exception as e:
            print(f"⚠️ Error extracting content from {url}: {e}")
            return None


class TechnicalAnalyzer:
    """Enhanced technical analysis with multiple indicators"""

    def calculate_technical_indicators(self, symbol, period="6mo"):
        """Calculate comprehensive technical indicators"""
        try:
            stock = yf.Ticker(f"{symbol}.NS")
            hist = stock.history(period=period)

            if len(hist) < 50:
                return None

            indicators = {}

            # RSI Calculation
            delta = hist['Close'].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = -delta.clip(upper=0).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['RSI'] = rsi.iloc[-1]

            # MACD Calculation
            ema_12 = hist['Close'].ewm(span=12).mean()
            ema_26 = hist['Close'].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line

            indicators['MACD'] = macd_line.iloc[-1]
            indicators['MACD_Signal'] = signal_line.iloc[-1]
            indicators['MACD_Histogram'] = histogram.iloc[-1]
            indicators['MACD_Prev_Histogram'] = histogram.iloc[-2] if len(histogram) > 1 else 0

            # Stochastic Oscillator
            high_14 = hist['High'].rolling(14).max()
            low_14 = hist['Low'].rolling(14).min()
            k_percent = 100 * ((hist['Close'] - low_14) / (high_14 - low_14))
            d_percent = k_percent.rolling(3).mean()
            indicators['Stoch_K'] = k_percent.iloc[-1]
            indicators['Stoch_D'] = d_percent.iloc[-1]

            # Moving Averages
            indicators['EMA_20'] = hist['Close'].ewm(span=20).mean().iloc[-1]
            indicators['EMA_50'] = hist['Close'].ewm(span=50).mean().iloc[-1]
            indicators['SMA_20'] = hist['Close'].rolling(20).mean().iloc[-1]
            indicators['SMA_50'] = hist['Close'].rolling(50).mean().iloc[-1]

            # Bollinger Bands
            sma_20 = hist['Close'].rolling(20).mean()
            std_20 = hist['Close'].rolling(20).std()
            indicators['BB_Upper'] = (sma_20 + 2 * std_20).iloc[-1]
            indicators['BB_Lower'] = (sma_20 - 2 * std_20).iloc[-1]
            indicators['BB_Middle'] = sma_20.iloc[-1]

            # Current Price and Price Action
            indicators['Current_Price'] = hist['Close'].iloc[-1]
            indicators['Price_Change_1D'] = (hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100
            indicators['Price_Change_5D'] = (hist['Close'].iloc[-1] / hist['Close'].iloc[-6] - 1) * 100 if len(
                hist) >= 6 else 0

            # Volume Analysis
            indicators['Volume'] = hist['Volume'].iloc[-1]
            indicators['Avg_Volume'] = hist['Volume'].rolling(20).mean().iloc[-1]
            indicators['Volume_Ratio'] = indicators['Volume'] / indicators['Avg_Volume']

            return indicators

        except Exception as e:
            print(f"⚠️ Technical indicators calculation failed for {symbol}: {e}")
            return None

    def analyze_technical_signals(self, indicators):
        """Analyze technical signals and generate score"""
        if not indicators:
            return None, 0

        signals = {}
        score = 0

        # RSI Analysis
        rsi = indicators['RSI']
        if rsi < 30:
            signals['RSI'] = "OVERSOLD - Buy Signal"
            score += 2
        elif rsi > 70:
            signals['RSI'] = "OVERBOUGHT - Sell Signal"
            score -= 2
        elif 30 <= rsi <= 45:
            signals['RSI'] = "BULLISH ZONE"
            score += 1
        elif 55 <= rsi <= 70:
            signals['RSI'] = "BEARISH ZONE"
            score -= 1
        else:
            signals['RSI'] = "NEUTRAL"

        # MACD Analysis
        macd = indicators['MACD']
        signal = indicators['MACD_Signal']
        histogram = indicators['MACD_Histogram']
        prev_histogram = indicators['MACD_Prev_Histogram']

        if macd > signal and histogram > prev_histogram:
            signals['MACD'] = "BULLISH CROSSOVER"
            score += 2
        elif macd < signal and histogram < prev_histogram:
            signals['MACD'] = "BEARISH CROSSOVER"
            score -= 2
        elif macd > signal:
            signals['MACD'] = "BULLISH MOMENTUM"
            score += 1
        elif macd < signal:
            signals['MACD'] = "BEARISH MOMENTUM"
            score -= 1
        else:
            signals['MACD'] = "NEUTRAL"

        # Moving Average Analysis
        current_price = indicators['Current_Price']
        ema_20 = indicators['EMA_20']
        ema_50 = indicators['EMA_50']

        if current_price > ema_20 > ema_50:
            signals['MA_Trend'] = "STRONG UPTREND"
            score += 2
        elif current_price < ema_20 < ema_50:
            signals['MA_Trend'] = "STRONG DOWNTREND"
            score -= 2
        elif current_price > ema_20:
            signals['MA_Trend'] = "SHORT TERM BULLISH"
            score += 1
        elif current_price < ema_20:
            signals['MA_Trend'] = "SHORT TERM BEARISH"
            score -= 1
        else:
            signals['MA_Trend'] = "SIDEWAYS"

        # Bollinger Bands Analysis
        bb_upper = indicators['BB_Upper']
        bb_lower = indicators['BB_Lower']
        bb_middle = indicators['BB_Middle']

        if current_price <= bb_lower:
            signals['BB'] = "OVERSOLD - Potential Buy"
            score += 1.5
        elif current_price >= bb_upper:
            signals['BB'] = "OVERBOUGHT - Potential Sell"
            score -= 1.5
        elif current_price > bb_middle:
            signals['BB'] = "ABOVE MIDDLE - Bullish"
            score += 0.5
        else:
            signals['BB'] = "BELOW MIDDLE - Bearish"
            score -= 0.5

        # Volume Analysis
        volume_ratio = indicators['Volume_Ratio']
        if volume_ratio > 1.5:
            signals['Volume'] = "HIGH VOLUME - Strong Move"
            score += 1 if score > 0 else -1
        elif volume_ratio < 0.5:
            signals['Volume'] = "LOW VOLUME - Weak Move"
            score *= 0.5
        else:
            signals['Volume'] = "NORMAL VOLUME"

        return signals, score


class TradingSignalAnalyzer:
    """Main trading signal analyzer class"""

    def __init__(self):
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        self.news_scraper = NewsScraperService()
        self.technical_analyzer = TechnicalAnalyzer()

    def generate_recommendation(self, sentiment_summary, technical_signals, technical_score, indicators):
        """Generate enhanced recommendation"""

        sentiment_score = sentiment_summary["score"]
        overall_sentiment = sentiment_summary["overall"]
        confidence = sentiment_summary.get("confidence", 0.5)

        # Calculate combined score
        combined_score = (sentiment_score * 1.2) + (technical_score * 1.0)

        # Enhanced recommendation logic
        if combined_score >= 3.0:
            recommendation = "STRONG BUY"
            confidence_level = "HIGH"
        elif combined_score >= 1.5:
            recommendation = "BUY"
            confidence_level = "MEDIUM"
        elif combined_score >= 0.5:
            recommendation = "WEAK BUY"
            confidence_level = "LOW"
        elif combined_score <= -3.0:
            recommendation = "STRONG SELL"
            confidence_level = "HIGH"
        elif combined_score <= -1.5:
            recommendation = "SELL"
            confidence_level = "MEDIUM"
        elif combined_score <= -0.5:
            recommendation = "WEAK SELL"
            confidence_level = "LOW"
        else:
            recommendation = "HOLD"
            confidence_level = "MEDIUM"

        # Risk assessment
        risk_level = "LOW"
        if abs(sentiment_score - technical_score) > 2:
            risk_level = "HIGH"
        elif abs(sentiment_score - technical_score) > 1:
            risk_level = "MEDIUM"

        return {
            "recommendation": recommendation,
            "confidence": confidence_level,
            "combined_score": combined_score,
            "sentiment_score": sentiment_score,
            "technical_score": technical_score,
            "risk_level": risk_level
        }

    def analyze_stock(self, company_name, symbol):
        """Analyze a single stock"""
        print(f"\n{'=' * 60}")
        print(f"🔍 Analyzing {company_name} ({symbol})")
        print(f"{'=' * 60}")

        results = {
            'company': company_name,
            'symbol': symbol,
            'sentiment': None,
            'technical': None,
            'recommendation': None,
            'articles': []
        }

        try:
            # Get news articles
            print("📰 Fetching news articles...")
            articles = self.news_scraper.get_articles_enhanced(company_name, symbol)
            results['articles'] = articles

            if articles:
                print(f"✅ Found {len(articles)} articles")

                # Analyze sentiment
                sentiments = []
                confidences = []

                for i, article in enumerate(articles):
                    content = article.get('content', article.get('title', ''))
                    prediction, confidence, methods = self.sentiment_analyzer.ensemble_prediction(content)
                    sentiments.append(prediction)
                    confidences.append(confidence)

                    print(f"📄 Article {i + 1}: {prediction.upper()} (Confidence: {confidence:.2f})")
                    print(f"   Title: {article.get('title', 'N/A')[:80]}...")
                    print(f"   Methods: {methods}")
                    print()

                # Calculate sentiment summary
                sentiment_summary = {
                    "overall": max(set(sentiments), key=sentiments.count),
                    "score": sum(1 if s == 'positive' else -1 if s == 'negative' else 0 for s in sentiments) / len(
                        sentiments),
                    "confidence": sum(confidences) / len(confidences),
                    "distribution": dict(Counter(sentiments))
                }

                results['sentiment'] = sentiment_summary

                print(f"📊 SENTIMENT ANALYSIS:")
                print(f"   Overall: {sentiment_summary['overall'].upper()}")
                print(f"   Score: {sentiment_summary['score']:.2f}")
                print(f"   Confidence: {sentiment_summary['confidence']:.2f}")
                print(f"   Distribution: {sentiment_summary['distribution']}")

            else:
                print("⚠️ No articles found, using neutral sentiment")
                sentiment_summary = {
                    "overall": "neutral",
                    "score": 0,
                    "confidence": 0.5,
                    "distribution": {"neutral": 1}
                }
                results['sentiment'] = sentiment_summary

            # Technical analysis
            print("\n📈 Calculating technical indicators...")
            indicators = self.technical_analyzer.calculate_technical_indicators(symbol)

            if indicators:
                technical_signals, technical_score = self.technical_analyzer.analyze_technical_signals(indicators)
                results['technical'] = {'signals': technical_signals, 'score': technical_score}

                print(f"📈 TECHNICAL ANALYSIS:")
                for indicator, signal in technical_signals.items():
                    print(f"   {indicator}: {signal}")
                print(f"   Technical Score: {technical_score:.1f}")

                # Generate recommendation
                recommendation = self.generate_recommendation(
                    sentiment_summary, technical_signals, technical_score, indicators
                )
                results['recommendation'] = recommendation

                print(f"\n🎯 RECOMMENDATION:")
                print(f"   Action: {recommendation['recommendation']}")
                print(f"   Confidence: {recommendation['confidence']}")
                print(f"   Risk Level: {recommendation['risk_level']}")
                print(f"   Combined Score: {recommendation['combined_score']:.2f}")

                # Display key metrics
                print(f"\n📊 KEY METRICS:")
                print(f"   Current Price: ₹{indicators['Current_Price']:.2f}")
                print(f"   1D Change: {indicators['Price_Change_1D']:.2f}%")
                print(f"   5D Change: {indicators['Price_Change_5D']:.2f}%")
                print(f"   RSI: {indicators['RSI']:.1f}")
                print(f"   Volume Ratio: {indicators['Volume_Ratio']:.2f}")

            else:
                print("⚠️ Could not calculate technical indicators")

        except Exception as e:
            print(f"⚠️ Error analyzing {company_name}: {e}")

        return results


def main():
    """Main function to run the analysis"""
    print("🚀 Starting Enhanced Trading Signal Analysis")
    print("=" * 80)

    # Define companies to analyze
    companies = {
        "Infosys": "INFY",
        "TCS": "TCS",
        "Reliance": "RELIANCE",
        "HDFC Bank": "HDFCBANK",
        "ITC": "ITC",
        "Wipro": "WIPRO",
        "HCL Technologies": "HCLTECH",
        "Bharti Airtel": "BHARTIARTL",
        "ICICI Bank": "ICICIBANK",
        "Hindustan Unilever": "HINDUNILVR"
    }

    analyzer = TradingSignalAnalyzer()
    all_results = []

    # Analyze each company
    for company_name, symbol in companies.items():
        try:
            result = analyzer.analyze_stock(company_name, symbol)
            all_results.append(result)
            time.sleep(2)  # Rate limiting between requests
        except Exception as e:
            print(f"⚠️ Error analyzing {company_name}: {e}")
            continue

    # Generate summary report
    print(f"\n{'=' * 80}")
    print("📋 ANALYSIS SUMMARY REPORT")
    print(f"{'=' * 80}")

    # Filter results with recommendations
    results_with_recommendations = [r for r in all_results if r.get('recommendation')]

    if results_with_recommendations:
        # Sort by combined score
        sorted_results = sorted(results_with_recommendations,
                                key=lambda x: x['recommendation']['combined_score'],
                                reverse=True)

        print("\n🏆 TOP RECOMMENDATIONS:")
        for i, result in enumerate(sorted_results[:5], 1):
            rec = result['recommendation']
            print(f"{i}. {result['company']} ({result['symbol']})")
            print(f"   Recommendation: {rec['recommendation']}")
            print(f"   Score: {rec['combined_score']:.2f}")
            print(f"   Confidence: {rec['confidence']}")
            print(f"   Risk: {rec['risk_level']}")
            print()

        # Buy signals
        buy_signals = [r for r in results_with_recommendations
                       if r['recommendation']['recommendation'] in ['BUY', 'STRONG BUY']]

        if buy_signals:
            print("\n🟢 BUY SIGNALS:")
            for result in buy_signals:
                rec = result['recommendation']
                print(f"   {result['company']} ({result['symbol']}): {rec['recommendation']}")

        # Sell signals
        sell_signals = [r for r in results_with_recommendations
                        if r['recommendation']['recommendation'] in ['SELL', 'STRONG SELL']]

        if sell_signals:
            print("\n🔴 SELL SIGNALS:")
            for result in sell_signals:
                rec = result['recommendation']
                print(f"   {result['company']} ({result['symbol']}): {rec['recommendation']}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trading_analysis_{timestamp}.json"

    # Prepare results for JSON serialization
    json_results = []
    for result in all_results:
        json_result = {
            'company': result['company'],
            'symbol': result['symbol'],
            'sentiment': result.get('sentiment'),
            'technical_score': result.get('technical', {}).get('score'),
            'recommendation': result.get('recommendation'),
            'articles_count': len(result.get('articles', []))
        }
        json_results.append(json_result)

    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n💾 Results saved to {filename}")
    print(f"✅ Analysis complete! Processed {len(all_results)} companies")


if __name__ == "__main__":
    main()