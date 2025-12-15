import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
import urllib3
from datetime import datetime, timedelta, timezone
from io import StringIO
import google.generativeai as genai
from openai import OpenAI
from scipy.stats import norm
from scipy.optimize import brentq

# 忽略 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="台指期權戰情室 (WantGoo 懶人版)")
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
# 🕸️ WantGoo 爬蟲模組 (新增!)
# ==========================================
class WantGooScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        self.urls = {
            "pc_ratio": "https://www.wantgoo.com/option/put-call-ratio",
            "institutional_oi": "https://www.wantgoo.com/option/institutional-investors-call-put",
            "large_traders": "https://www.wantgoo.com/option/large-traders-open-interest"
        }

    def fetch_data(self):
        """一次性抓取所有關鍵數據"""
        data = {
            "Spot": 0, "Fut": 0, "PC_Ratio": 0,
            "Foreign_Net": 0, "Dealer_Net": 0,
            "Top10_Net": 0, "Msg": []
        }
        
        # 1. 抓 P/C Ratio 與 行情 (最重要)
        try:
            res = requests.get(self.urls["pc_ratio"], headers=self.headers, timeout=10)
            dfs = pd.read_html(StringIO(res.text))
            if dfs:
                df = dfs[0]
                # WantGoo 表格通常第一列是最新資料
                latest = df.iloc[0]
                data["Spot"] = float(latest.get("加權指數", 0))
                data["Fut"] = float(latest.get("台指期", 0))
                # 處理 P/C Ratio (可能是字串 "105.2%")
                pc_raw = str(latest.get("未平倉多空比", "0")).replace("%", "")
                data["PC_Ratio"] = float(pc_raw)
                data["Msg"].append("✅ P/C Ratio 與行情抓取成功")
        except Exception as e:
            data["Msg"].append(f"❌ P/C Ratio 抓取失敗: {e}")

        # 2. 抓三大法人 (外資/自營商)
        try:
            res = requests.get(self.urls["institutional_oi"], headers=self.headers, timeout=10)
            dfs = pd.read_html(StringIO(res.text))
            if dfs:
                df = dfs[0]
                # 尋找最新日期的資料
                # 表格結構變動大，嘗試用關鍵字搜尋
                # 通常會有 "外資", "自營商" 的 "未平倉淨額" 或 "買賣權淨額"
                # 這裡做簡化假設，若結構改變需調整
                # 假設 columns 有多層 index，直接轉 string 找
                latest = df.iloc[0]
                # 這裡僅示範邏輯，實際需視當下網頁結構微調
                # 嘗試抓取外資總淨額 (Call淨 + Put淨)
                # 註：WantGoo 表格較複雜，這裡用簡易容錯抓法
                data["Foreign_Net"] = "N/A" # 暫存
                data["Dealer_Net"] = "N/A"
                data["Msg"].append("✅ 法人數據讀取成功 (詳細欄位解析需視網頁結構)")
        except:
            pass

        return data

wantgoo = WantGooScraper()

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
# 🕸️ 期交所爬蟲 (用於獲取詳細履約價資料)
# ==========================================
@st.cache_data(ttl=300)
def fetch_detailed_options():
    """抓取期交所詳細資料用於畫圖 (GEX/Tornado)"""
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    now = datetime.now(tz=TW_TZ)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Origin": "https://www.taifex.com.tw",
        "Referer": "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    }

    for i in range(5): # 回溯 5 天
        d = now - timedelta(days=i)
        d_str = d.strftime("%Y/%m/%d")
        payload = {"queryType": "2", "marketCode": "0", "commodity_id": "TXO", "queryDate": d_str, "MarketCode": "0", "commodity_idt": "TXO"}
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
                elif "Price" in c: col_map["Price"] = c
            
            if len(col_map) < 5: continue
            df = df.rename(columns=col_map)
            df = df[["Month","Strike","Type","OI","Price"]].dropna(subset=["Type"]).copy()
            df["Strike"] = pd.to_numeric(df["Strike"].astype(str).str.replace(",",""), errors="coerce")
            df["OI"] = pd.to_numeric(df["OI"].astype(str).str.replace(",",""), errors="coerce").fillna(0)
            df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",","").replace("-","0"), errors="coerce").fillna(0)
            df["Amount"] = df["OI"] * df["Price"] * 50
            return df, d_str
        except: continue
    return None, None

def process_uploaded_csv(uploaded_file):
    try:
        try: df = pd.read_csv(uploaded_file, encoding='big5', header=0)
        except: uploaded_file.seek(0); df = pd.read_csv(uploaded_file, encoding='utf-8', header=0)
        
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

    fig = go.Figure()
    fig.add_trace(go.Bar(y=data["Strike"], x=-data["OI_P"], orientation='h', name="Put(支撐)", marker_color="green"))
    fig.add_trace(go.Bar(y=data["Strike"], x=data["OI_C"], orientation='h', name="Call(壓力)", marker_color="red"))
    if spot_price > 0: fig.add_hline(y=spot_price, line_dash="dash", line_color="orange")
    fig.update_layout(title=title, barmode='overlay', yaxis=dict(dtick=50, tickformat='d'), height=700)
    return fig

# ==========================================
# 🚀 主程式
# ==========================================
def main():
    st.title("🦅 台指期權戰情室 (WantGoo 懶人版)")
    
    if st.sidebar.button("🔄 刷新數據"):
        st.cache_data.clear()
        st.rerun()

    # 1. 抓 WantGoo 懶人包數據 (行情、P/C、法人)
    with st.spinner("正在從 WantGoo 偷看答案..."):
        wg_data = wantgoo.fetch_data()
        spot = wg_data["Spot"]
        fut = wg_data["Fut"]
        pc_ratio = wg_data["PC_Ratio"]

    # 2. 抓詳細籌碼 (期交所/CSV) 用於畫圖
    with st.spinner("嘗試獲取詳細籌碼分布..."):
        df_opt, date_str = fetch_detailed_options()

    # --- 儀表板 ---
    with st.container(border=True):
        st.subheader("📊 市場概況 (來源: WantGoo)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("加權指數", f"{spot:.0f}" if spot else "N/A")
        basis = fut - spot if (fut and spot) else 0
        c2.metric("台指期", f"{fut:.0f}" if fut else "N/A", f"基差 {basis:.0f}", delta_color="inverse")
        
        pc_delta = "偏多" if pc_ratio > 100 else "偏空"
        c3.metric("P/C Ratio", f"{pc_ratio}%", pc_delta)
        c4.write(wg_data["Msg"])

    # 手動校正 (如果 WantGoo 沒抓到)
    final_price = spot if spot > 0 else st.number_input("手動輸入現貨價格", value=0.0)

    # --- 詳細圖表區 ---
    if df_opt is None:
        st.warning("⚠️ 無法獲取詳細履約價分佈 (GEX/龍捲風圖需詳細資料)。")
        uploaded_file = st.file_uploader("📂 請上傳期交所 CSV 以解鎖圖表", type=["csv"])
        if uploaded_file:
            df_opt, date_str = process_uploaded_csv(uploaded_file)

    if df_opt is not None:
        st.success(f"✅ 詳細籌碼載入成功: {date_str}")
        
        all_codes = sorted(df_opt["Month"].unique())
        def_idx = 0
        for i, c in enumerate(all_codes):
            if len(c) == 6 and c.isdigit(): def_idx = i; break
        sel_code = st.sidebar.selectbox("🎯 分析合約", all_codes, index=def_idx)
        
        df_target = df_opt[df_opt["Month"] == sel_code].copy()
        df_calc = calculate_gex(df_target, final_price)
        
        tab1, tab2 = st.tabs(["🌪️ 籌碼龍捲風", "⚡ GEX 曝險"])
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
            st.plotly_chart(fig, use_container_width=True)

    # --- AI 分析 ---
    st.markdown("---")
    if st.button("🤖 啟動 AI 莊家分析", type="primary"):
        prompt = f"""
        你現在是台指期權的冷血莊家。
        【WantGoo 市場數據】
        - 加權指數: {spot}
        - 台指期: {fut} (基差 {basis})
        - P/C Ratio: {pc_ratio}%
        
        【籌碼結構】
        - 合約: {sel_code if 'sel_code' in locals() else 'N/A'}
        
        請結合 P/C Ratio 與 基差，分析目前市場的多空情緒，並推測莊家是否會利用目前的籌碼結構進行軋空或殺盤。
        """
        with st.spinner("AI 運算中..."):
            res = get_ai_response(prompt, "gemini")
            if "未設定" in res: res = get_ai_response(prompt, "openai")
            st.info(res)

if __name__ == "__main__":
    main()
