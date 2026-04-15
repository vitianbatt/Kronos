# -*- coding: utf-8 -*-
"""
prediction_cn_markets_day.py

Description:
    Predicts future daily K-line (1D) data for A-share markets using Kronos model and akshare.
    The script automatically downloads the latest historical data, cleans it, and runs model inference.

Usage:
    python prediction_cn_markets_day.py --symbol 000001

Arguments:
    --symbol     Stock code (e.g. 002594 for BYD, 000001 for SSE Index)

Output:
    - Saves the prediction results to ./outputs/pred_<symbol>_data.csv and ./outputs/pred_<symbol>_chart.png
    - Logs and progress are printed to console

Example:
    bash> python prediction_cn_markets_day.py --symbol 000001
    python3 prediction_cn_markets_day.py --symbol 002594

Notes (personal):
    - Increased LOOKBACK from 400 to 480 to give the model more historical context;
      this seems to improve trend continuity on volatile small-cap stocks.
    - Set SAMPLE_COUNT to 5 so we can average multiple stochastic samples and
      get a smoother/more stable prediction curve.
"""

import os
import argparse
import time
import pandas as pd
import akshare as ak
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor

save_dir = "./outputs"
os.makedirs(save_dir, exist_ok=True)

# Setting
TOKENIZER_PRETRAINED = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_PRETRAINED = "NeoQuasar/Kronos-base"
DEVICE = "cpu"  # "cuda:0"
MAX_CONTEXT = 512
LOOKBACK = 480   # increased from 400 — more context helps on trending/volatile names
PRED_LEN = 120
T = 1.0
TOP_P = 0.9
SAMPLE_COUNT = 5  # average over 5 samples for a smoother prediction

def load_data(symbol: str) -> pd.DataFrame:
    print(f"📥 Fetching {symbol} daily data from akshare ...")

    max_retries = 3
    df = None

    # Retry mechanism
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="")
            if df is not None and not df.empty:
                break
        except Exception as e:
            print(f"⚠️ Attempt {attempt}/{max_retries} failed: {e}")
        time.sleep(1.5)

    # If still empty after retries
    if df is None or df.empty:
        print(f"❌ Failed to fetch data for {symbol} after {max_retries} attempts. Exiting.")
        sys.exit(1)
    
    df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount"
    }, inplace=True)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace({"--": None, "": None})
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix invalid open values
    open_bad = (df["open"] == 0) | (df["open"].isna())
    if open_bad.any():
        print(f"⚠️  Fixed {open_bad.sum()} invalid open values.")
        df.loc[open_bad, "open"] = df["close"].shift(1)
        df["open"].fillna(df["close"], inplace=True)

    # Fix missing amount
    if df["amount"].isna().all() or (df["amount"] == 0).all():
        df["amount"] = df["close"] * df["volume"]

    print(f"✅ Data loaded: {l