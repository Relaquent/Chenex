from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import time
import os

app = Flask(__name__)
CORS(app)

# Base URL for CoinGecko API
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Non-blocking rate limit storage
last_request_time = {}

def rate_limit(key, seconds=1.5):
    """Render-safe non-blocking rate limiter"""
    now = time.time()
    last = last_request_time.get(key, 0)
    if now - last < seconds:
        return False  # reject instead of sleep (Render safe)
    last_request_time[key] = now
    return True

def safe_get(url, params=None, timeout=15):
    """Safe request with retry + error protection"""
    try:
        for _ in range(2):
            res = requests.get(url, params=params, timeout=timeout)
            if res.status_code == 200:
                return res.json(), None
            time.sleep(0.8)
        return None, f"CoinGecko API returned status {res.status_code}"
    except Exception as e:
        return None, str(e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/prices", methods=["GET"])
def get_prices():
    if not rate_limit("prices"):
        return jsonify({"success": False, "error": "Too many requests"}), 429

    data, error = safe_get(
        f"{COINGECKO_BASE}/coins/markets",
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 20,
            "page": 1,
            "sparkline": True,
            "price_change_percentage": "1h,24h,7d",
        }
    )

    if error:
        return jsonify({"success": False, "error": error}), 200

    formatted_data = []
    for coin in data:
        formatted_data.append({
            "id": coin.get("id"),
            "symbol": coin.get("symbol", "").upper(),
            "name": coin.get("name", ""),
            "image": coin.get("image", ""),
            "current_price": coin.get("current_price", 0),
            "price_change_1h": coin.get("price_change_percentage_1h_in_currency", 0),
            "price_change_24h": coin.get("price_change_percentage_24h_in_currency", 0),
            "price_change_7d": coin.get("price_change_percentage_7d_in_currency", 0),
            "market_cap": coin.get("market_cap", 0),
            "volume": coin.get("total_volume", 0),
            "sparkline": coin.get("sparkline_in_7d", {}).get("price", []),
        })

    return jsonify({"success": True, "data": formatted_data})


@app.route("/api/coin/<coin_id>", methods=["GET"])
def get_coin_details(coin_id):
    if not rate_limit("coin_details"):
        return jsonify({"success": False, "error": "Too many requests"}), 429

    data, error = safe_get(f"{COINGECKO_BASE}/coins/{coin_id}")
    if error:
        return jsonify({"success": False, "error": error}), 200

    market_data = data.get("market_data", {})

    return jsonify({
        "success": True,
        "data": {
            "id": data.get("id"),
            "symbol": data.get("symbol", "").upper(),
            "name": data.get("name", ""),
            "description": (data.get("description", {}).get("en") or "")[:500],
            "current_price": market_data.get("current_price", {}).get("usd", 0),
            "market_cap": market_data.get("market_cap", {}).get("usd", 0),
            "volume": market_data.get("total_volume", {}).get("usd", 0),
            "high_24h": market_data.get("high_24h", {}).get("usd", 0),
            "low_24h": market_data.get("low_24h", {}).get("usd", 0),
            "ath": market_data.get("ath", {}).get("usd", 0),
            "atl": market_data.get("atl", {}).get("usd", 0),
        }
    })


@app.route("/api/predict/<coin_id>", methods=["GET"])
def predict_price(coin_id):
    if not rate_limit("predict"):
        return jsonify({"success": False, "error": "Too many requests"}), 429

    data, error = safe_get(
        f"{COINGECKO_BASE}/coins/{coin_id}/market_chart",
        params={"vs_currency": "usd", "days": 30}
    )

    if error:
        return jsonify({"success": False, "error": error}), 200

    prices = [p[1] for p in data.get("prices", []) if len(p) > 1]
    if len(prices) < 7:
        return jsonify({"success": False, "error": "Insufficient data for prediction"}), 200

    current_price = prices[-1]
    sma_7 = sum(prices[-7:]) / 7
    sma_30 = sum(prices) / len(prices)
    trend = ((sma_7 - sma_30) / sma_30) * 100 if sma_30 else 0

    prediction_1d = current_price * (1 + (trend / 100) * 0.3)
    prediction_7d = current_price * (1 + (trend / 100) * 0.7)
    prediction_30d = current_price * (1 + (trend / 100) * 1.2)
    sentiment = "Bullish" if trend > 2 else "Bearish" if trend < -2 else "Neutral"

    return jsonify({
        "success": True,
        "data": {
            "current_price": current_price,
            "predictions": {
                "1_day": round(prediction_1d, 2),
                "7_day": round(prediction_7d, 2),
                "30_day": round(prediction_30d, 2),
            },
            "trend": round(trend, 2),
            "sentiment": sentiment,
            "confidence": min(abs(trend) * 10, 85),
        },
    })


@app.route("/api/chart/<coin_id>", methods=["GET"])
def get_chart_data(coin_id):
    days = request.args.get("days", 7, type=int)

    if not rate_limit("chart"):
        return jsonify({"success": False, "error": "Too many requests"}), 429

    data, error = safe_get(
        f"{COINGECKO_BASE}/coins/{coin_id}/market_chart",
        params={"vs_currency": "usd", "days": days}
    )

    if error:
        return jsonify({"success": False, "error": error}), 200

    return jsonify({
        "success": True,
        "data": {
            "prices": data.get("prices", []),
            "volumes": data.get("total_volumes", []),
        },
    })


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🚀 Chenex v1.1.3 Backend Server (Render Stable Release)")
    print("=" * 50)
    print("✓ Running in production mode (Debug OFF)")
    print("✓ Auto PORT enabled for Render")
    print("=" * 50 + "\n")

    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
