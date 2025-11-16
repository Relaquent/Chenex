from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_caching import Cache
import requests
import time
import os
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import threading

# === Flask Application ===
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# === Advanced Cache Configuration ===
if os.environ.get("REDIS_URL"):
    cache = Cache(app, config={
        "CACHE_TYPE": "RedisCache",
        "CACHE_REDIS_URL": os.environ.get("REDIS_URL"),
        "CACHE_DEFAULT_TIMEOUT": 180
    })
    print("[INFO] RedisCache active 🚀")
else:
    cache = Cache(app, config={
        "CACHE_TYPE": "SimpleCache",
        "CACHE_DEFAULT_TIMEOUT": 180
    })
    print("[INFO] SimpleCache (RAM) active ⚙️")

# === API Configuration ===
# Using CoinGecko free public API - no authentication required
COINGECKO_API = "https://api.coingecko.com/api/v3"
HEADERS = {"User-Agent": "ChenexCryptoDashboard/2.0", "Accept": "application/json"}

# === Advanced Rate Limiting with Token Bucket ===
class TokenBucket:
    def __init__(self, capacity=10, refill_rate=1):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens=1):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def wait_time(self):
        with self.lock:
            if self.tokens >= 1:
                return 0
            return (1 - self.tokens) / self.refill_rate

# Create separate buckets for different endpoints
rate_limiters = {
    "markets": TokenBucket(capacity=5, refill_rate=0.5),
    "coin_detail": TokenBucket(capacity=10, refill_rate=1),
    "chart": TokenBucket(capacity=8, refill_rate=0.8),
    "global": TokenBucket(capacity=5, refill_rate=0.5)
}

def rate_limit_wait(bucket_name):
    bucket = rate_limiters.get(bucket_name, rate_limiters["global"])
    while not bucket.consume():
        wait = bucket.wait_time()
        print(f"[RATE LIMIT] {bucket_name} - waiting {wait:.2f}s")
        time.sleep(min(wait + 0.1, 2))

# === Exponential Backoff Request Handler ===
def safe_get(url, params=None, retries=5, bucket="global"):
    rate_limit_wait(bucket)
    
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=15)
            
            if r.status_code == 429:
                wait_time = min(2 ** attempt * 5, 60)
                print(f"[429 RATE LIMIT] Attempt {attempt+1}/{retries} - waiting {wait_time}s")
                time.sleep(wait_time)
                continue
            
            if r.status_code == 200:
                return r
            
            if r.status_code >= 500:
                wait_time = 2 ** attempt
                print(f"[SERVER ERROR {r.status_code}] Attempt {attempt+1}/{retries} - waiting {wait_time}s")
                time.sleep(wait_time)
                continue
                
        except requests.exceptions.Timeout:
            print(f"[TIMEOUT] Attempt {attempt+1}/{retries}")
            time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Attempt {attempt+1}/{retries}: {e}")
            time.sleep(2 ** attempt)
    
    print(f"[FATAL] All retries failed for: {url}")
    return None

# === Advanced Prediction Models ===
class AdvancedPredictor:
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices):
        """Moving Average Convergence Divergence"""
        if len(prices) < 26:
            return 0, 0
        
        prices_arr = np.array(prices)
        ema_12 = AdvancedPredictor.ema(prices_arr, 12)
        ema_26 = AdvancedPredictor.ema(prices_arr, 26)
        macd_line = ema_12 - ema_26
        signal_line = AdvancedPredictor.ema(macd_line, 9)
        
        return macd_line[-1], signal_line[-1]
    
    @staticmethod
    def ema(data, period):
        """Exponential Moving Average"""
        return np.array([np.mean(data[:i+1]) if i < period else 
                        data[i] * (2/(period+1)) + 
                        np.mean(data[max(0,i-period):i]) * (1 - 2/(period+1))
                        for i in range(len(data))])
    
    @staticmethod
    def calculate_volatility(prices, window=30):
        """Calculate price volatility"""
        if len(prices) < 2:
            return 0
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns[-window:]) * 100
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        """Bollinger Bands"""
        if len(prices) < period:
            return prices[-1], prices[-1], prices[-1]
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def predict_price(prices, volumes, days_ahead):
        """Advanced multi-factor prediction"""
        if len(prices) < 30:
            return prices[-1], 50, {"rsi": 50, "macd": 0, "signal": 0, "volatility": 0, "trend_slope": 0, "bb_position": 50}
        
        prices_arr = np.array(prices)
        current = prices_arr[-1]
        
        # Technical indicators
        rsi = AdvancedPredictor.calculate_rsi(prices_arr)
        macd, signal = AdvancedPredictor.calculate_macd(prices_arr)
        volatility = AdvancedPredictor.calculate_volatility(prices_arr)
        upper_bb, middle_bb, lower_bb = AdvancedPredictor.calculate_bollinger_bands(prices_arr)
        
        # Trend analysis
        sma_7 = np.mean(prices_arr[-7:])
        sma_30 = np.mean(prices_arr[-30:])
        sma_90 = np.mean(prices_arr[-90:]) if len(prices_arr) >= 90 else sma_30
        
        # Linear regression trend
        x = np.arange(len(prices_arr[-30:]))
        y = prices_arr[-30:]
        z = np.polyfit(x, y, 1)
        trend_slope = z[0]
        
        # Volume analysis
        vol_trend = np.mean(volumes[-7:]) / np.mean(volumes[-30:]) if len(volumes) >= 30 else 1
        
        # Weighted prediction factors
        trend_factor = (sma_7 - sma_30) / sma_30
        momentum_factor = (current - sma_90) / sma_90 if len(prices_arr) >= 90 else trend_factor
        rsi_factor = (rsi - 50) / 50
        macd_factor = 1 if macd > signal else -1
        volatility_factor = min(volatility / 10, 1)
        volume_factor = (vol_trend - 1) * 0.5
        
        # Position within Bollinger Bands
        bb_position = (current - lower_bb) / (upper_bb - lower_bb) if upper_bb != lower_bb else 0.5
        bb_factor = (bb_position - 0.5) * 2
        
        # Combined prediction with confidence weighting
        prediction_change = (
            trend_factor * 0.30 +
            momentum_factor * 0.20 +
            rsi_factor * 0.15 +
            macd_factor * 0.10 +
            bb_factor * 0.15 +
            volume_factor * 0.10
        )
        
        # Time decay and volatility adjustment
        time_factor = days_ahead / 30
        prediction_change *= time_factor
        prediction_change *= (1 + volatility_factor * 0.2)
        
        predicted_price = current * (1 + prediction_change)
        
        # Confidence calculation
        confidence = max(20, min(95, 
            70 - (volatility * 2) + 
            (abs(trend_slope) * 1000) +
            (10 if 40 < rsi < 60 else -5)
        ))
        
        return predicted_price, confidence, {
            "rsi": round(rsi, 2),
            "macd": round(macd, 4),
            "signal": round(signal, 4),
            "volatility": round(volatility, 2),
            "trend_slope": round(trend_slope, 6),
            "bb_position": round(bb_position * 100, 2)
        }

# === ROOT ===
@app.route('/')
def index():
    return render_template('index.html')

# === GLOBAL MARKET STATS ===
@app.route('/api/global')
@cache.cached(timeout=300, query_string=True)
def get_global_stats():
    r = safe_get(f"{COINGECKO_API}/global", bucket="global")
    if not r or r.status_code != 200:
        return jsonify({"success": False, "error": "Global data unavailable"}), 500
    
    data = r.json()["data"]
    return jsonify({
        "success": True,
        "data": {
            "total_market_cap": data.get("total_market_cap", {}).get("usd", 0),
            "total_volume": data.get("total_volume", {}).get("usd", 0),
            "btc_dominance": data.get("market_cap_percentage", {}).get("btc", 0),
            "eth_dominance": data.get("market_cap_percentage", {}).get("eth", 0),
            "active_cryptocurrencies": data.get("active_cryptocurrencies", 0),
            "markets": data.get("markets", 0),
            "market_cap_change_24h": data.get("market_cap_change_percentage_24h_usd", 0)
        }
    })

# === ENHANCED PRICE LIST ===
@app.route('/api/prices')
@cache.cached(timeout=180, query_string=True)
def get_prices():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    
    r = safe_get(f"{COINGECKO_API}/coins/markets", {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": min(per_page, 100),
        "page": page,
        "sparkline": True,
        "price_change_percentage": "1h,24h,7d,30d"
    }, bucket="markets")
    
    if not r or r.status_code != 200:
        return jsonify({"success": False, "error": "API unavailable"}), 500

    coins = r.json()
    data = []
    for c in coins:
        data.append({
            "id": c.get("id", ""),
            "symbol": c.get("symbol", "").upper(),
            "name": c.get("name", ""),
            "image": c.get("image", ""),
            "current_price": c.get("current_price", 0),
            "price_change_1h": c.get("price_change_percentage_1h_in_currency", 0),
            "price_change_24h": c.get("price_change_percentage_24h", 0),
            "price_change_7d": c.get("price_change_percentage_7d_in_currency", 0),
            "price_change_30d": c.get("price_change_percentage_30d_in_currency", 0),
            "market_cap": c.get("market_cap", 0),
            "market_cap_rank": c.get("market_cap_rank", 0),
            "fully_diluted_valuation": c.get("fully_diluted_valuation", 0),
            "total_volume": c.get("total_volume", 0),
            "high_24h": c.get("high_24h", 0),
            "low_24h": c.get("low_24h", 0),
            "circulating_supply": c.get("circulating_supply", 0),
            "total_supply": c.get("total_supply", 0),
            "max_supply": c.get("max_supply", 0),
            "ath": c.get("ath", 0),
            "ath_change_percentage": c.get("ath_change_percentage", 0),
            "ath_date": c.get("ath_date", ""),
            "atl": c.get("atl", 0),
            "atl_change_percentage": c.get("atl_change_percentage", 0),
            "sparkline": c.get("sparkline_in_7d", {}).get("price", [])
        })
    return jsonify({"success": True, "data": data})

# === DETAILED COIN INFO ===
@app.route('/api/coin/<coin_id>')
@cache.cached(timeout=300, query_string=True)
def get_coin_details(coin_id):
    r = safe_get(f"{COINGECKO_API}/coins/{coin_id}", {
        "localization": False,
        "tickers": False,
        "community_data": True,
        "developer_data": True
    }, bucket="coin_detail")
    
    if not r or r.status_code != 200:
        return jsonify({"success": False, "error": "Coin data unavailable"}), 500

    d = r.json()
    md = d.get("market_data", {})

    return jsonify({
        "success": True,
        "data": {
            "id": d.get("id", coin_id),
            "symbol": d.get("symbol", "").upper(),
            "name": d.get("name", ""),
            "description": d.get("description", {}).get("en", "")[:800],
            "categories": d.get("categories", []),
            "links": {
                "homepage": (d.get("links", {}).get("homepage", []) or [""])[0],
                "blockchain_site": (d.get("links", {}).get("blockchain_site", []) or [""])[0],
                "twitter": d.get("links", {}).get("twitter_screen_name", ""),
                "telegram": d.get("links", {}).get("telegram_channel_identifier", "")
            },
            "current_price": md.get("current_price", {}).get("usd", 0),
            "market_cap": md.get("market_cap", {}).get("usd", 0),
            "market_cap_rank": d.get("market_cap_rank", 0),
            "volume": md.get("total_volume", {}).get("usd", 0),
            "high_24h": md.get("high_24h", {}).get("usd", 0),
            "low_24h": md.get("low_24h", {}).get("usd", 0),
            "price_change_24h": md.get("price_change_24h", 0),
            "price_change_percentage_24h": md.get("price_change_percentage_24h", 0),
            "ath": md.get("ath", {}).get("usd", 0),
            "ath_change_percentage": md.get("ath_change_percentage", {}).get("usd", 0),
            "ath_date": md.get("ath_date", {}).get("usd", ""),
            "atl": md.get("atl", {}).get("usd", 0),
            "atl_change_percentage": md.get("atl_change_percentage", {}).get("usd", 0),
            "atl_date": md.get("atl_date", {}).get("usd", ""),
            "circulating_supply": md.get("circulating_supply", 0),
            "total_supply": md.get("total_supply", 0),
            "max_supply": md.get("max_supply", 0),
            "community_data": d.get("community_data", {}),
            "developer_data": d.get("developer_data", {})
        }
    })

# === ADVANCED PRICE PREDICTION ===
@app.route('/api/predict/<coin_id>')
@cache.cached(timeout=180, query_string=True)
def predict_price(coin_id):
    # Get 90 days of data for better prediction
    r = safe_get(f"{COINGECKO_API}/coins/{coin_id}/market_chart", {
        "vs_currency": "usd",
        "days": 90
    }, bucket="chart")
    
    if not r or r.status_code != 200:
        return jsonify({"success": False, "error": "Prediction data unavailable"}), 500

    chart_data = r.json()
    prices = [x[1] for x in chart_data.get("prices", [])]
    volumes = [x[1] for x in chart_data.get("total_volumes", [])]
    
    if not prices or len(prices) < 30:
        return jsonify({"success": False, "error": "Insufficient data"}), 500

    current = prices[-1]
    predictor = AdvancedPredictor()
    
    # Generate predictions
    pred_1d, conf_1d, indicators_1d = predictor.predict_price(prices, volumes, 1)
    pred_7d, conf_7d, indicators_7d = predictor.predict_price(prices, volumes, 7)
    pred_30d, conf_30d, indicators_30d = predictor.predict_price(prices, volumes, 30)
    
    # Average confidence
    avg_confidence = (conf_1d + conf_7d + conf_30d) / 3
    
    # Determine sentiment
    price_change = ((pred_7d - current) / current) * 100
    if price_change > 5:
        sentiment = "Strong Bullish"
    elif price_change > 2:
        sentiment = "Bullish"
    elif price_change < -5:
        sentiment = "Strong Bearish"
    elif price_change < -2:
        sentiment = "Bearish"
    else:
        sentiment = "Neutral"
    
    # Calculate support and resistance levels
    recent_prices = prices[-30:]
    support = min(recent_prices)
    resistance = max(recent_prices)
    
    return jsonify({
        "success": True,
        "data": {
            "current_price": round(current, 2),
            "predictions": {
                "1_day": round(pred_1d, 2),
                "7_day": round(pred_7d, 2),
                "30_day": round(pred_30d, 2)
            },
            "price_changes": {
                "1_day": round(((pred_1d - current) / current) * 100, 2),
                "7_day": round(((pred_7d - current) / current) * 100, 2),
                "30_day": round(((pred_30d - current) / current) * 100, 2)
            },
            "confidence": round(avg_confidence, 2),
            "sentiment": sentiment,
            "technical_indicators": indicators_7d,
            "support_resistance": {
                "support": round(support, 2),
                "resistance": round(resistance, 2),
                "current_position": round(((current - support) / (resistance - support)) * 100, 2)
            }
        }
    })

# === CHART DATA ===
@app.route('/api/chart/<coin_id>')
@cache.cached(timeout=180, query_string=True)
def chart(coin_id):
    days = request.args.get("days", 30, type=int)
    
    r = safe_get(f"{COINGECKO_API}/coins/{coin_id}/market_chart", {
        "vs_currency": "usd",
        "days": min(days, 365)
    }, bucket="chart")
    
    if not r or r.status_code != 200:
        return jsonify({"success": False, "error": "Chart data unavailable"}), 500

    j = r.json()
    return jsonify({
        "success": True,
        "data": {
            "prices": j.get("prices", []),
            "market_caps": j.get("market_caps", []),
            "volumes": j.get("total_volumes", [])
        }
    })

# === MAIN ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n✅ Chenex Advanced Server v2.0 running on http://0.0.0.0:{port}")
    print(f"📊 Dashboard available at http://localhost:{port}")
    print(f"🔗 API endpoints:")
    print(f"   - GET /api/global")
    print(f"   - GET /api/prices")
    print(f"   - GET /api/predict/<coin_id>")
    print(f"   - GET /api/chart/<coin_id>")
    print(f"\n🚀 Server starting...\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
