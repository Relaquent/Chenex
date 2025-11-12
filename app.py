from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_caching import Cache
import requests
import time
import os

app = Flask(__name__)
CORS(app)

# Cache (60 saniye)
cache = Cache(app, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 60})

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
HEADERS = {"User-Agent": "ChenexCryptoDashboard/1.0"}

last_request_time = {}

def rate_limit(key, seconds=1):
    now = time.time()
    if key in last_request_time:
        wait = seconds - (now - last_request_time[key])
        if wait > 0:
            time.sleep(wait)
    last_request_time[key] = time.time()

def safe_get(url, params=None, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if r.status_code == 429:
                time.sleep(5 * (i + 1))
                continue
            return r
        except requests.exceptions.RequestException:
            time.sleep(2)
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/prices')
@cache.cached(timeout=60)
def get_prices():
    rate_limit("prices", 2)

    r = safe_get(f"{COINGECKO_BASE}/coins/markets", {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 20,
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "1h,24h,7d"
    })

    if not r or r.status_code != 200:
        return jsonify({"success": False, "error": "CoinGecko unavailable or rate limited"}), 500

    coins = r.json()
    result = []

    for c in coins:
        result.append({
            "id": c["id"],
            "symbol": c["symbol"].upper(),
            "name": c["name"],
            "image": c["image"],
            "current_price": c["current_price"],
            "price_change_1h": c.get("price_change_percentage_1h_in_currency", 0),
            "price_change_24h": c.get("price_change_percentage_24h", 0),
            "price_change_7d": c.get("price_change_percentage_7d_in_currency", 0),
            "market_cap": c["market_cap"],
            "volume": c["total_volume"]
        })

    return jsonify({"success": True, "data": result})


@app.route('/api/coin/<coin_id>')
@cache.cached(timeout=60)
def get_coin_details(coin_id):
    rate_limit("coin", 2)

    r = safe_get(f"{COINGECKO_BASE}/coins/{coin_id}")
    if not r or r.status_code != 200:
        return jsonify({"success": False, "error": "CoinGecko unavailable"}), 500

    d = r.json()
    md = d["market_data"]

    return jsonify({
        "success": True,
        "data": {
            "id": d["id"],
            "symbol": d["symbol"].upper(),
            "name": d["name"],
            "description": d["description"]["en"][:400],
            "current_price": md["current_price"]["usd"],
            "market_cap": md["market_cap"]["usd"],
            "volume": md["total_volume"]["usd"],
            "high_24h": md["high_24h"]["usd"],
            "low_24h": md["low_24h"]["usd"],
            "ath": md["ath"]["usd"],
            "atl": md["atl"]["usd"]
        }
    })


@app.route('/api/predict/<coin_id>')
def predict_price(coin_id):
    rate_limit("predict", 2)

    r = safe_get(f"{COINGECKO_BASE}/coins/{coin_id}/market_chart", {
        "vs_currency": "usd",
        "days": 30
    })

    if not r or r.status_code != 200:
        return jsonify({"success": False, "error": "Prediction data unavailable"}), 500

    prices = [x[1] for x in r.json()["prices"]]
    current = prices[-1]

    sma7 = sum(prices[-7:]) / 7
    sma30 = sum(prices) / len(prices)
    trend = ((sma7 - sma30) / sma30) * 100

    return jsonify({
        "success": True,
        "data": {
            "current_price": current,
            "predictions": {
                "1_day": round(current * (1 + trend/100 * 0.3), 2),
                "7_day": round(current * (1 + trend/100 * 0.7), 2),
                "30_day": round(current * (1 + trend/100 * 1.2), 2)
            },
            "trend": round(trend, 2),
            "sentiment": "Bullish" if trend > 2 else "Bearish" if trend < -2 else "Neutral",
            "confidence": min(abs(trend) * 10, 85)
        }
    })


@app.route('/api/chart/<coin_id>')
@cache.cached(timeout=120)
def chart(coin_id):
    days = request.args.get("days", 7, int)
    rate_limit("chart", 2)

    r = safe_get(f"{COINGECKO_BASE}/coins/{coin_id}/market_chart", {
        "vs_currency": "usd",
        "days": days
    })

    if not r or r.status_code != 200:
        return jsonify({"success": False, "error": "Chart data unavailable"}), 500

    j = r.json()
    return jsonify({"success": True, "data": {"prices": j["prices"], "volumes": j["total_volumes"]}})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n✅ Server running on port", port)
    app.run(host="0.0.0.0", port=port)
