# =========================
# app-AI.py  (ANTI-CRASH VERSION)
# =========================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import time
from datetime import datetime, timedelta, timezone
from io import StringIO
import calendar
import re
import numpy as np
import google.generativeai as genai
from openai import OpenAI

st.set_page_config(layout="wide", page_title="台指期權戰情室｜防炸版")
TW_TZ = timezone(timedelta(hours=8))

# =========================
# Safe helpers
# =========================

def safe_df(df: pd.DataFrame) -> bool:
    return df is not None and isinstance(df, pd.DataFrame) and not df.empty

def empty_fig(msg="⚠️ 資料不足"):
    fig = go.Figure()
    fig.update_layout(title=msg, height=300)
    return fig

def safe_plot(func):
    def wrapper(*args, **kwargs):
        try:
            fig = func(*args, **kwargs)
            if fig is None:
                return empty_fig()
            return fig
        except Exception as e:
            return empty_fig(f"⚠️ 圖表錯誤：{type(e).__name__}")
    return wrapper

# =========================
# 現貨
# =========================
@st.cache_data(ttl=60)
def get_spot():
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII"
        r = requests.get(url, timeout=5).json()
        meta = r["chart"]["result"][0]["meta"]
        return float(meta.get("regularMarketPrice"))
    except:
        return None

# =========================
# 選擇權資料（穩定）
# =========================
@st.cache_data(ttl=300)
def get_option_data(days=3):
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": url,
        "Origin": "https://www.taifex.com.tw"
    }

    results = []

    now = datetime.now(tz=TW_TZ)

    for i in range(days + 3):
        d = now - timedelta(days=i)
        if d.weekday() >= 5:
            continue
        if i == 0 and now.hour < 15:
            continue

        qdate = d.strftime("%Y/%m/%d")
        payload = {
            "queryType": "2",
            "MarketCode": "0",
            "marketCode": "0",
            "commodity_id": "TXO",
            "commodity_idt": "TXO",
            "queryDate": qdate,
        }

        try:
            r = requests.post(url, data=payload, headers=headers, timeout=10)
            if "查無資料" in r.text or len(r.text) < 800:
                continue

            dfs = pd.read_html(StringIO(r.text))
            df = dfs[0]

            df.columns = [str(c).replace(" ", "").replace("*", "") for c in df.columns]

            col_map = {
                "Month": next((c for c in df.columns if "月" in c or "週" in c), None),
                "Strike": next((c for c in df.columns if "履約" in c), None),
                "Type": next((c for c in df.columns if "買" in c), None),
                "OI": next((c for c in df.columns if "未沖銷" in c), None),
                "IV": next((c for c in df.columns if "隱含" in c), None),
            }

            if None in col_map.values():
                continue

            df = df.rename(columns=col_map)[list(col_map.keys())]
            df["Strike"] = pd.to_numeric(df["Strike"], errors="coerce")
            df["OI"] = pd.to_numeric(df["OI"], errors="coerce").fillna(0)
            if "IV" in df:
                df["IV"] = pd.to_numeric(df["IV"], errors="coerce")

            df["Date"] = qdate
            results.append(df)

            if len(results) >= days:
                break

        except:
            continue

    if not results:
        return None

    return pd.concat(results, ignore_index=True)

# =========================
# Gamma Exposure（簡化）
# =========================
def calc_gamma_exposure(df):
    if not safe_df(df):
        return None
    try:
        g = df.groupby("Strike")["OI"].sum()
        gex = g * -1  # 簡化假設 dealer short gamma
        return gex.reset_index(name="GEX")
    except:
        return None

# =========================
# 圖表
# =========================
@safe_plot
def plot_oi(df):
    if not safe_df(df):
        return empty_fig("無 OI 資料")

    pivot = df.pivot_table(
        index="Strike", columns="Type", values="OI", aggfunc="sum", fill_value=0
    )

    fig = go.Figure()
    for c in pivot.columns:
        fig.add_bar(y=pivot.index, x=pivot[c], orientation="h", name=str(c))

    fig.update_layout(title="全履約價 Call / Put OI", barmode="overlay", height=700)
    return fig

@safe_plot
def plot_oi_change(df):
    if not safe_df(df) or df["Date"].nunique() < 2:
        return empty_fig("近 3 日 OI 資料不足")

    latest = df[df["Date"] == df["Date"].max()]
    prev = df[df["Date"] != df["Date"].max()]

    chg = (
        latest.groupby("Strike")["OI"].sum()
        - prev.groupby("Strike")["OI"].sum()
    ).dropna()

    fig = go.Figure(go.Bar(x=chg.index, y=chg.values))
    fig.update_layout(title="近 3 日 OI 變化", height=300)
    return fig

@safe_plot
def plot_gex(gex):
    if gex is None or gex.empty:
        return empty_fig("Gamma Exposure 無資料")
    fig = go.Figure(go.Bar(x=gex["Strike"], y=gex["GEX"]))
    fig.update_layout(title="Dealer Gamma Exposure（簡化）", height=300)
    return fig

# =========================
# MAIN
# =========================
def main():
    st.title("🛡️ 台指期權戰情室（防炸版）")

    with st.spinner("抓取資料中…"):
        spot = get_spot()
        opt = get_option_data()

    if spot:
        st.metric("現貨指數", int(spot))
    else:
        st.warning("⚠️ 現貨資料抓取失敗")

    st.markdown("### 📊 選擇權 OI")
    st.plotly_chart(plot_oi(opt), use_container_width=True)

    st.markdown("### 🔄 近 3 日 OI 變化")
    st.plotly_chart(plot_oi_change(opt), use_container_width=True)

    st.markdown("### 🧲 Dealer Gamma Exposure")
    gex = calc_gamma_exposure(opt)
    st.plotly_chart(plot_gex(gex), use_container_width=True)

if __name__ == "__main__":
    main()
