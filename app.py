from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import time
import traceback

app = Flask(__name__)
CORS(app)

# CoinGecko API ve Proxy
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
PROXY_BASE = "https://api.allorigins.win/raw?url="

# Basit rate limit
last_request_time = {}


def rate_limit(key, seconds=2):
    current_time = time.time()
    if key in last_request_time:
        elapsed = current_time - last_request_time[key]
        if elapsed < seconds:
            time.sleep(seconds - elapsed)
    last_request_time[key] = time.time()


def safe_get(url, params=None, timeout=15):
    """Render üzerinde bağlantı hatası durumunda proxy fallback"""
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"[WARN] Direct request failed, trying proxy → {e}")
        try:
            # Proxy fallback
            proxied_url = PROXY_BASE + requests.utils.quote(url, safe=":/?&=")
            response = requests.get(proxied_url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as e2:
            print(f"[ERROR] Proxy request also failed → {e2}")
            traceback.print_exc()
            raise


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/prices", methods=["GET"])
def get_prices():
    """Get current prices for top cryptocurrencies"""
    try:
        rate_limit("prices")

        response = safe_get(
            f"{COINGECKO_BASE}/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 20,
                "page": 1,
                "sparkline": True,
                "price_change_percentage": "1h,24h,7d",
            },
        )

        data = response.json()
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

    except Exception as e:
        print("[ERROR] /api/prices:", e)
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/coin/<coin_id>", methods=["GET"])
def get_coin_details(coin_id):
    """Get detailed info about a specific cryptocurrency"""
    try:
        rate_limit("coin_details")

        response = safe_get(f"{COINGECKO_BASE}/coins/{coin_id}")
        data = response.json()
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

    except Exception as e:
        print("[ERROR] /api/coin:", e)
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predict/<coin_id>", methods=["GET"])
def predict_price(coin_id):
    """Simple trend-based prediction"""
    try:
        rate_limit("predict")

        response = safe_get(
            f"{COINGECKO_BASE}/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": 30},
        )

        data = response.json()
        prices = [p[1] for p in data.get("prices", []) if len(p) > 1]

        if len(prices) < 7:
            return jsonify({
                "success": False,
                "error": "Insufficient data for prediction"
            }), 400

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

    except Exception as e:
        print("[ERROR] /api/predict:", e)
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chart/<coin_id>", methods=["GET"])
def get_chart_data(coin_id):
    """Get historical chart data"""
    days = request.args.get("days", 7, type=int)
    try:
        rate_limit("chart")

        response = safe_get(
            f"{COINGECKO_BASE}/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": days},
        )

        data = response.json()

        return jsonify({
            "success": True,
            "data": {
                "prices": data.get("prices", []),
                "volumes": data.get("total_volumes", []),
            },
        })

    except Exception as e:
        print("[ERROR] /api/chart:", e)
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🚀 Chenex v1.1.4 Backend Server (Render-Optimized)")
    print("=" * 50)
    print("✓ Running at: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000, host="0.0.0.0")
