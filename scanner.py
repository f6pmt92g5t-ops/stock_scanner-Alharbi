"""
=====================================================
Stock Scanner - فحص أسهم السوق الأمريكي
يحلل الأسهم بمؤشرات فنية شاملة + تلخيص بواسطة Claude
ثم يرسل التقرير على تلقرام
=====================================================
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# ============ الإعدادات (من GitHub Secrets) ============
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY")

CLAUDE_MODEL = "claude-sonnet-4-5"

# قائمة الأسهم المراد فحصها - عدّلها كما تحب
WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META",
    "AMD", "NFLX", "PLTR", "SMCI", "AVGO", "COIN", "MSTR"
]


# ============ جلب البيانات من Twelve Data ============
def fetch_stock_data(ticker, outputsize=250, interval="1day"):
    if not TWELVE_DATA_API_KEY:
        print("⚠️ لم يتم ضبط مفتاح Twelve Data API")
        return None

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": ticker,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
        data = resp.json()

        if data.get("status") == "error" or "values" not in data:
            print(f"⚠️ خطأ في جلب بيانات {ticker}: {data.get('message', data)}")
            return None

        values = data["values"]
        if len(values) < 30:
            return None

        df = pd.DataFrame(values)
        df = df.rename(columns={
            "close": "Close", "high": "High", "low": "Low",
            "open": "Open", "volume": "Volume",
        })
        for col in ["Close", "High", "Low", "Open", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Twelve Data يرجع الأحدث أولاً، نعكسها عشان تصير الأقدم أولاً
        df = df.iloc[::-1].reset_index(drop=True)
        df.dropna(inplace=True)

        return df
    except Exception as e:
        print(f"⚠️ خطأ في جلب بيانات {ticker}: {e}")
        return None


# ============ حساب المؤشرات الفنية الشاملة ============
def calculate_indicators(df):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    ind = {}

    # المتوسطات المتحركة
    ind["MA20"] = close.rolling(20).mean().iloc[-1]
    ind["MA50"] = close.rolling(50).mean().iloc[-1]
    ind["MA200"] = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    ind["RSI"] = (100 - (100 / (1 + rs))).iloc[-1]

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    ind["MACD"] = macd_line.iloc[-1]
    ind["MACD_signal"] = signal_line.iloc[-1]
    ind["MACD_hist"] = (macd_line - signal_line).iloc[-1]

    # Bollinger Bands
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    ind["BB_upper"] = (ma20 + 2 * std20).iloc[-1]
    ind["BB_lower"] = (ma20 - 2 * std20).iloc[-1]

    # Stochastic Oscillator
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    k = 100 * (close - low14) / (high14 - low14)
    ind["Stoch_K"] = k.iloc[-1]
    ind["Stoch_D"] = k.rolling(3).mean().iloc[-1]

    # ATR (تذبذب)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    ind["ATR"] = tr.rolling(14).mean().iloc[-1]

    # ADX (قوة الترند)
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr14)
    minus_di = 100 * (minus_dm.rolling(14).sum() / tr14)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    ind["ADX"] = dx.rolling(14).mean().iloc[-1]

    # حجم التداول
    ind["Volume"] = volume.iloc[-1]
    ind["Avg_Volume_20"] = volume.rolling(20).mean().iloc[-1]
    ind["Volume_ratio"] = ind["Volume"] / ind["Avg_Volume_20"] if ind["Avg_Volume_20"] else 0

    # السعر والتغير
    ind["Price"] = close.iloc[-1]
    ind["Change_pct"] = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100

    # الدعم والمقاومة
    ind["Resistance"] = high.rolling(20).max().iloc[-1]
    ind["Support"] = low.rolling(20).min().iloc[-1]

    return ind


# ============ منطق اختيار الفرص الجيدة ============
def is_good_opportunity(ind):
    if ind["MA200"] is None:
        return False

    conditions = [
        ind["Price"] > ind["MA50"],     # فوق المتوسط 50
        ind["MA50"] > ind["MA200"],     # ترند صاعد عام
        30 < ind["RSI"] < 70,           # ليس متطرف
        ind["MACD_hist"] > 0,           # زخم إيجابي
        ind["Volume_ratio"] > 1.2,      # حجم أعلى من المعتاد
        ind["ADX"] > 20,                # قوة اتجاه
    ]
    return sum(conditions) >= 4


# ============ تلخيص بواسطة Claude ============
def get_claude_summary(ticker, ind):
    if not CLAUDE_API_KEY:
        return "⚠️ لم يتم ضبط مفتاح Claude API
