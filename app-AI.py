import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
import urllib3
from datetime import datetime, timedelta, timezone
from io import StringIO
import calendar
import re
import google.generativeai as genai
from openai import OpenAI
from scipy.stats import norm
from scipy.optimize import brentq

# 忽略 SSL 警告 (提升雲端爬蟲成功率)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="台指期權戰情室 (週一修正版)")
TW_TZ = timezone(timedelta(hours=8))

# ==========================================
# 🔑 API 金鑰
# ==========================================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = ""

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except:
    OPENAI_API_KEY = ""

def get_ai_response(prompt, model_type="gemini"):
    if model_type == "gemini":
        if not GEMINI_API_KEY: return "⚠️ 未設定 GEMINI_API_KEY"
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash") 
            return model.generate_content(prompt).text
        except Exception as e: return f"Gemini Error: {e}"
    elif model_type == "openai":
        if not OPENAI_API_KEY: return "⚠️ 未設定 OPENAI_API_KEY"
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user", "content":prompt}]
            )
            return res.choices[0].message.content
        except Exception as e: return f"OpenAI Error: {e}"
    return "Unknown Model"

# ==========================================
# 🧮 寬客核心 (Greeks)
# ==========================================
class QuantLib:
    def __init__(self, r=0.015):
        self.r = r

    def implied_volatility(self, price, S, K, T, type_='Call'):
        if price <= 0.1 or T <= 0: return np.nan
        try:
            def bs_price(sigma):
                d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                if type_ == 'Call':
                    return S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
                else:
                    return K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            def objective(sigma):
                return bs_price(sigma) - price
            return brentq(objective, 0.01, 3.0)
        except: return np.nan

    def get_greeks(self, S, K, T, sigma, type_='Call'):
        if T <= 0 or sigma <= 0: return 0, 0
        try:
            d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            return 0, gamma 
        except: return 0, 0

ql = QuantLib()

# ==========================================
# 🕸️ 數據抓取模組
# ==========================================

@st.cache_data(ttl=60)
def fetch_basic_market_data():
    """只抓現貨 (Yahoo)"""
    data = {"Spot": 0, "Msg": "無數據"}
    ts = int(time.time())
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII?interval=1d&range=1d&_={ts}"
        res = requests.get(url, headers=headers, timeout=4)
        meta = res.json()["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice") or meta.get("chartPreviousClose")
        if price: data["Spot"] = float(price)
    except: pass

    if data["Spot"] > 0:
        data["Msg"] = "✅ 現貨行情更新成功"
    else:
        data["Msg"] = "⚠️ 無法抓取行情，請手動輸入"
    return data

@st.cache_data(ttl=300)
def fetch_option_data_best_effort():
    """盡力抓取選擇權資料 (修正回溯天數問題)"""
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    now = datetime.now(tz=TW_TZ)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Origin": "https://www.taifex.com.tw",
        "Referer": "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    }

    # 🔥 關鍵修正：將回溯天數從 3 改為 10
    # 這樣週一執行時 (回溯0,1,2=一,日,六) 也能繼續找 (3=五)
    for i in range(10):
        d = now - timedelta(days=i)
        d_str = d.strftime("%Y/%m/%d")
        payload = {
            "queryType": "2", "marketCode": "0", "commodity_id": "TXO", 
            "queryDate": d_str, "MarketCode": "0", "commodity_idt": "TXO"
        }
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=6, verify=False)
            if "查無資料" in res.text or len(res.text) < 500: continue
            
            df = pd.read_html(StringIO(res.text))[0]
            df.columns = [str(c).replace(" ","").replace("*","").replace("契約","").strip() for c in df.columns]
            
            col_map = {}
            for c in df.columns:
                if "月" in c: col_map["Month"] = c
                elif "履約" in c: col_map["Strike"] = c
                elif "買賣" in c: col_map["Type"] = c
                elif "OI" in c or "未沖銷" in c: col_map["OI"] = c
                elif "Price" in c or "結算" in c or "收盤" in c: col_map["Price"] = c
            
            if len(col_map) < 5: continue
            
            df = df.rename(columns=col_map)
            df = df[["Month","Strike","Type","OI","Price"]].dropna(subset=["Type"]).copy()
            
            df["Type"] = df["Type"].astype(str).str.strip()
            df["Strike"] = pd.to_numeric(df["Strike"].astype(str).str.replace(",",""), errors="coerce")
            df["OI"] = pd.to_numeric(df["OI"].astype(str).str.replace(",",""), errors="coerce").fillna(0)
            df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",","").replace("-","0"), errors="coerce").fillna(0)
            df["Amount"] = df["OI"] * df["Price"] * 50
            
            return df, d_str
            
        except: continue
            
    return None, None

def process_uploaded_csv(uploaded_file):
    try:
        try:
            df = pd.read_csv(uploaded_file, encoding='big5', header=0)
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', header=0)
        
        df.columns = [str(c).replace(" ","").replace("*","").replace("契約","").strip() for c in df.columns]
        col_map = {}
        for c in df.columns:
            if "月" in c: col_map["Month"] = c
            elif "履約" in c: col_map["Strike"] = c
            elif "買賣" in c: col_map["Type"] = c
            elif "OI" in c or "未沖銷" in c: col_map["OI"] = c
            elif "Price" in c: col_map["Price"] = c
        
        if len(col_map) < 5: return None, None
        df = df.rename(columns=col_map)
        df = df[["Month","Strike","Type","OI","Price"]].dropna(subset=["Type"]).copy()
        df["Strike"] = pd.to_numeric(df["Strike"].astype(str).str.replace(",",""), errors="coerce")
        df["OI"] = pd.to_numeric(df["OI"].astype(str).str.replace(",",""), errors="coerce").fillna(0)
        df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",","").replace("-","0"), errors="coerce").fillna(0)
        df["Amount"] = df["OI"] * df["Price"] * 50
        
        return df, "手動上傳"
    except: return None, None

# ==========================================
# 📊 繪圖函式
# ==========================================
def calculate_gex(df, spot_price):
    gex_list = []
    T = 5/365.0
    for _, row in df.iterrows():
        K, price, oi, cp = row["Strike"], row["Price"], row["OI"], row["Type"]
        iv = 0.2
        if price > 0.5 and oi > 0:
            calc_iv = ql.implied_volatility(price, spot_price, K, T, 'Call' if 'Call' in cp else 'Put')
            if not np.isnan(calc_iv): iv = calc_iv
        _, gamma = ql.get_greeks(spot_price, K, T, iv)
        val = gamma * oi * spot_price * 100
        if 'Put' in cp or '賣' in cp: val = -val
        gex_list.append(val)
    df["GEX"] = gex_list
    return df

def plot_tornado(df, spot_price, title):
    df_c = df[df["Type"].str.contains("Call|買")].groupby("Strike")[["OI","Amount"]].sum().reset_index()
    df_p = df[df["Type"].str.contains("Put|賣")].groupby("Strike")[["OI","Amount"]].sum().reset_index()
    data = pd.merge(df_c, df_p, on="Strike", suffixes=("_C", "_P"), how="outer").fillna(0).sort_values("Strike")
    
    if spot_price > 0:
        base = round(spot_price/100)*100
        data = data[(data["Strike"] >= base-1000) & (data["Strike"] <= base+1000)]
    else:
        max_idx = data["OI_P"].idxmax()
        center = data.loc[max_idx, "Strike"]
        data = data[(data["Strike"] >= center-1000) & (data["Strike"] <= center+1000)]

    fig = go.Figure()
    fig.add_trace(go.Bar(y=data["Strike"], x=-data["OI_P"], orientation='h', name="Put(支撐)", marker_color="green", 
                         customdata=data["Amount_P"]/1e8, hovertemplate="<b>%{y}</b><br>Put: %{x}<br>Amt: %{customdata:.1f}億"))
    fig.add_trace(go.Bar(y=data["Strike"], x=data["OI_C"], orientation='h', name="Call(壓力)", marker_color="red", 
                         customdata=data["Amount_C"]/1e8, hovertemplate="<b>%{y}</b><br>Call: %{x}<br>Amt: %{customdata:.1f}億"))
    
    if spot_price > 0:
        fig.add_hline(y=spot_price, line_dash="dash", line_color="orange")
        fig.add_annotation(x=0, y=spot_price, text=f"現貨 {int(spot_price)}", showarrow=False, bgcolor="orange", font=dict(color="white"))

    fig.update_layout(title=title, barmode='overlay', yaxis=dict(dtick=50, tickformat='d'), height=700)
    return fig

# ==========================================
# 🚀 主程式
# ==========================================
def main():
    st.title("🦅 台指期權戰情室 (週一修正版)")
    
    if st.sidebar.button("🔄 重新掃描數據"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("正在掃描即時行情..."):
        basic_data = fetch_basic_market_data()
        spot = basic_data["Spot"]

    with st.spinner("嘗試抓取選擇權籌碼 (回溯最近交易日)..."):
        df_opt, date_str = fetch_option_data_best_effort()

    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 2])
        c1.metric("加權指數 (Spot)", f"{spot:.0f}" if spot > 0 else "N/A", basic_data["Msg"])
        manual_spot = c3.number_input("🛠️ 手動輸入/校正點位", value=spot if spot > 0 else 0.0, step=1.0)
    
    final_price = manual_spot if manual_spot > 0 else spot

    if df_opt is None:
        st.warning("⚠️ 自動抓取失敗 (已嘗試回溯10天)。")
        st.info("💡 建議手動上傳 CSV 以解鎖圖表。")
        uploaded_file = st.file_uploader("📂 拖入期交所 CSV 檔 (選填)", type=["csv"])
        if uploaded_file:
            df_opt, date_str = process_uploaded_csv(uploaded_file)

    if df_opt is not None:
        st.success(f"✅ 成功載入選擇權籌碼！資料日期: {date_str}")
        
        all_codes = sorted(df_opt["Month"].unique())
        def_idx = 0
        for i, c in enumerate(all_codes):
            if len(c) == 6 and c.isdigit(): def_idx = i; break
        sel_code = st.sidebar.selectbox("🎯 分析合約", all_codes, index=def_idx)
        
        df_target = df_opt[df_opt["Month"] == sel_code].copy()
        
        df_calc = calculate_gex(df_target, final_price)
        
        tab1, tab2 = st.tabs(["🌪️ 籌碼龍捲風", "⚡ GEX Gamma 曝險"])
        with tab1:
            st.plotly_chart(plot_tornado(df_calc, final_price, f"OI 分布: {sel_code}"), use_container_width=True)
        with tab2:
            gex = df_calc.groupby("Strike")["GEX"].sum().reset_index()
            if final_price > 0:
                base = round(final_price/100)*100
                gex = gex[(gex["Strike"] >= base-800) & (gex["Strike"] <= base+800)]
            colors = ['red' if v >= 0 else 'green' for v in gex["GEX"]]
            fig = go.Figure(go.Bar(x=gex["Strike"], y=gex["GEX"]/1e6, marker_color=colors))
            if final_price > 0: fig.add_vline(x=final_price, line_dash="dash", line_color="orange")
            fig.update_layout(title="Dealer Gamma Exposure", yaxis_title="GEX (M)", xaxis_title="Strike")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("紅色=黏滯/阻力 | 綠色=加速/滑價")

    st.markdown("---")
    if st.button("🤖 啟動 AI 莊家分析", type="primary"):
        if df_opt is not None:
            prompt = f"""
            你現在是台指期權的冷血莊家。
            【市場資訊】
            - 資料日期: {date_str}
            - 觀察合約: {sel_code if 'sel_code' in locals() else 'N/A'}
            - 關鍵點位(現貨): {final_price}
            
            請根據以上數據(若有圖表數據請結合GEX觀點)，分析目前市場的支撐壓力，並給出莊家視角的結算劇本。
            """
        else:
            prompt = f"""
            你現在是台指期權的冷血莊家。
            目前因為數據連線限制，我只能告訴你：
            - **目前大盤現貨點位**: {final_price}
            
            請你根據這個點位，結合你資料庫中對近期台股的盤感，
            推測外資與主力的可能心態。
            """
            
        with st.spinner("AI 運算中..."):
            res = get_ai_response(prompt, "gemini")
            if "未設定" in res: res = get_ai_response(prompt, "openai")
            st.info(res)

if __name__ == "__main__":
    main()
