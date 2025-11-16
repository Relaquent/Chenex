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
        "CACHE_DEFAULT_TIMEOUT": 180  # 3 minutes for faster updates
    })
    print("[INFO] RedisCache active 🚀")
else:
    cache = Cache(app, config={
        "CACHE_TYPE": "SimpleCache",
        "CACHE_DEFAULT_TIMEOUT": 180
    })
    print("[INFO] SimpleCache (RAM) active ⚙️")

# === API Configuration ===
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
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
    "markets": TokenBucket(capacity=5, refill_rate=0.5),  # 30 req/min
    "coin_detail": TokenBucket(capacity=10, refill_rate=1),  # 60 req/min
    "chart": TokenBucket(capacity=8, refill_rate=0.8),  # 48 req/min
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
                wait_time = min(2 ** attempt * 5, 60)  # Exponential backoff, max 60s
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
            return prices[-1]
        
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
        rsi_factor = (rsi - 50) / 50  # Normalize RSI
        macd_factor = 1 if macd > signal else -1
        volatility_factor = min(volatility / 10, 1)  # Cap at 1
        volume_factor = (vol_trend - 1) * 0.5
        
        # Position within Bollinger Bands
        bb_position = (current - lower_bb) / (upper_bb - lower_bb) if upper_bb != lower_bb else 0.5
        bb_factor = (bb_position - 0.5) * 2  # -1 to 1
        
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
        prediction_change *= (1 + volatility_factor * 0.2)  # Increase uncertainty with volatility
        
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

# === Warm-up Cache ===
@app.before_first_request
def warm_up():
    try:
        print("[INIT] Warming up cache...")
        safe_get(f"{COINGECKO_BASE}/coins/markets", {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 50,
            "page": 1,
            "sparkline": False
        }, bucket="markets")
        print("[INIT] Cache warmed ✅")
    except Exception as e:
        print(f"[WARN] Warm-up error: {e}")

# === ROOT ===
@app.route('/')
def index():
    return render_template('index.html')

# === GLOBAL MARKET STATS ===
@app.route('/api/global')
@cache.cached(timeout=300, query_string=True)
def get_global_stats():
    r = safe_get(f"{COINGECKO_BASE}/global", bucket="global")
    if not r or r.status_code != 200:
        return jsonify({"success": False, "error": "Global data unavailable"}), 500
    
    data = r.json()["data"]
    return jsonify({
        "success": True,
        "data": {
            "total_market_cap": data["total_market_cap"]["usd"],
            "total_volume": data["total_volume"]["usd"],
            "btc_dominance": data["market_cap_percentage"]["btc"],
            "eth_dominance": data["market_cap_percentage"].get("eth", 0),
            "active_cryptocurrencies": data["active_cryptocurrencies"],
            "markets": data["markets"],
            "market_cap_change_24h": data["market_cap_change_percentage_24h_usd"]
        }
    })

# === ENHANCED PRICE LIST ===
@app.route('/api/prices')
@cache.cached(timeout=180, query_string=True)
def get_prices():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    
    r = safe_get(f"{COINGECKO_BASE}/coins/markets", {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": min(per_page, 100),
        "page": page,
        "sparkline": True,
        "price_change_percentage": "1h,24h,7d,30d"
    }, bucket="markets")
    
    if not r or r.status_code != 200:
        return jsonify({"success": False, "error": "CoinGecko unavailable"}), 500

    coins = r.json()
    data = []
    for c in coins:
        data.append({
            "id": c["id"],
            "symbol": c["symbol"].upper(),
            "name": c["name"],
            "image": c["image"],
            "current_price": c["current_price"],
            "price_change_1h": c.get("price_change_percentage_1h_in_currency", 0),
            "price_change_24h": c.get("price_change_percentage_24h", 0),
            "price_change_7d": c.get("price_change_percentage_7d_in_currency", 0),
            "price_change_30d": c.get("price_change_percentage_30d_in_currency", 0),
            "market_cap": c["market_cap"],
            "market_cap_rank": c["market_cap_rank"],
            "fully_diluted_valuation": c.get("fully_diluted_valuation"),
            "total_volume": c["total_volume"],
            "high_24h": c["high_24h"],
            "low_24h": c["low_24h"],
            "circulating_supply": c.get("circulating_supply"),
            "total_supply": c.get("total_supply"),
            "max_supply": c.get("max_supply"),
            "ath": c["ath"],
            "ath_change_percentage": c["ath_change_percentage"],
            "ath_date": c["ath_date"],
            "atl": c["atl"],
            "atl_change_percentage": c["atl_change_percentage"],
            "sparkline": c.get("sparkline_in_7d", {}).get("price", [])
        })
    return jsonify({"success": True, "data": data})

# === DETAILED COIN INFO ===
@app.route('/api/coin/<coin_id>')
@cache.cached(timeout=300, query_string=True)
def get_coin_details(coin_id):
    r = safe_get(f"{COINGECKO_BASE}/coins/{coin_id}", {
        "localization": False,
        "tickers": False,
        "community_data": True,
        "developer_data": True
    }, bucket="coin_detail")
    
    if not r or r.status_code != 200:
        return jsonify({"success": False, "error": "Coin data unavailable"}), 500

    d = r.json()
    md = d["market_data"]

    return jsonify({
        "success": True,
        "data": {
            "id": d["id"],
            "symbol": d["symbol"].upper(),
            "name": d["name"],
            "description": d["description"]["en"][:800] if d["description"]["en"] else "",
            "categories": d.get("categories", []),
            "links": {
                "homepage": d["links"]["homepage"][0] if d["links"]["homepage"] else "",
                "blockchain_site": d["links"]["blockchain_site"][0] if d["links"]["blockchain_site"] else "",
                "twitter": d["links"]["twitter_screen_name"],
                "telegram": d["links"]["telegram_channel_identifier"]
            },
            "current_price": md["current_price"]["usd"],
            "market_cap": md["market_cap"]["usd"],
            "market_cap_rank": d["market_cap_rank"],
            "volume": md["total_volume"]["usd"],
            "high_24h": md["high_24h"]["usd"],
            "low_24h": md["low_24h"]["usd"],
            "price_change_24h": md["price_change_24h"],
            "price_change_percentage_24h": md["price_change_percentage_24h"],
            "ath": md["ath"]["usd"],
            "ath_change_percentage": md["ath_change_percentage"]["usd"],
            "ath_date": md["ath_date"]["usd"],
            "atl": md["atl"]["usd"],
            "atl_change_percentage": md["atl_change_percentage"]["usd"],
            "atl_date": md["atl_date"]["usd"],
            "circulating_supply": md.get("circulating_supply"),
            "total_supply": md.get("total_supply"),
            "max_supply": md.get("max_supply"),
            "community_data": d.get("community_data", {}),
            "developer_data": d.get("developer_data", {})
        }
    })

# === ADVANCED PRICE PREDICTION ===
@app.route('/api/predict/<coin_id>')
@cache.cached(timeout=180, query_string=True)
def predict_price(coin_id):
    # Get 90 days of data for better prediction
    r = safe_get(f"{COINGECKO_BASE}/coins/{coin_id}/market_chart", {
        "vs_currency": "usd",
        "days": 90
    }, bucket="chart")
    
    if not r or r.status_code != 200:
        return jsonify({"success": False, "error": "Prediction data unavailable"}), 500

    chart_data = r.json()
    prices = [x[1] for x in chart_data["prices"]]
    volumes = [x[1] for x in chart_data["total_volumes"]]
    
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
    
    r = safe_get(f"{COINGECKO_BASE}/coins/{coin_id}/market_chart", {
        "vs_currency": "usd",
        "days": min(days, 365)
    }, bucket="chart")
    
    if not r or r.status_code != 200:
        return jsonify({"success": False, "error": "Chart data unavailable"}), 500

    j = r.json()
    return jsonify({
        "success": True,
        "data": {
            "prices": j["prices"],
            "market_caps": j["market_caps"],
            "volumes": j["total_volumes"]
        }
    })

# === MAIN ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n✅ Chenex Advanced Server v2.0 running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
