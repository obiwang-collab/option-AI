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

# 忽略 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="台指期籌碼戰情室 (WantGoo + 龍捲風合體版)")
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
# 🕸️ WantGoo 爬蟲模組 (新增：抓取宏觀數據)
# ==========================================
class WantGooScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        self.urls = {
            "pc_ratio": "https://www.wantgoo.com/option/put-call-ratio",
            "institutional": "https://www.wantgoo.com/option/institutional-investors-call-put",
            "large_traders": "https://www.wantgoo.com/option/large-traders-open-interest"
        }

    def fetch_summary(self):
        """從玩股網抓取 P/C Ratio、外資、大戶動向"""
        data = {
            "Spot": 0, "Fut": 0, "PC_Ratio": 0,
            "Foreign_Option_Net": 0, "Top10_Trader_Net": 0,
            "Msg": []
        }
        
        # 1. 抓 P/C Ratio 與 即時行情
        try:
            res = requests.get(self.urls["pc_ratio"], headers=self.headers, timeout=10)
            dfs = pd.read_html(StringIO(res.text))
            if dfs:
                df = dfs[0]
                latest = df.iloc[0]
                # 嘗試抓取欄位 (玩股網欄位名稱可能會變，做容錯)
                data["Spot"] = float(latest.get("加權指數", 0))
                data["Fut"] = float(latest.get("台指期", 0))
                pc_str = str(latest.get("成交量多空比", "0")).replace("%", "") # 注意：玩股網有成交量跟未平倉，這裡抓未平倉較準
                if "未平倉多空比" in latest:
                    pc_str = str(latest["未平倉多空比"]).replace("%", "")
                data["PC_Ratio"] = float(pc_str)
                data["Msg"].append("✅ P/C Ratio 更新成功")
        except: pass

        # 2. 抓三大法人 (外資選擇權淨額)
        try:
            res = requests.get(self.urls["institutional"], headers=self.headers, timeout=10)
            dfs = pd.read_html(StringIO(res.text))
            if dfs:
                # 簡單邏輯：抓最新的外資淨額 (需要看網頁結構，這裡假設第一列是最新)
                # 這裡僅做示範，實際需解析多層欄位
                data["Msg"].append("✅ 法人數據連線成功")
        except: pass

        # 3. 抓十大交易人 (大戶)
        try:
            res = requests.get(self.urls["large_traders"], headers=self.headers, timeout=10)
            data["Msg"].append("✅ 大戶數據連線成功")
        except: pass

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
# 🗓️ 日期處理
# ==========================================
MANUAL_SETTLEMENT_FIX = {"202501W1": "2025/01/02"}

def get_settlement_date(contract_code: str) -> str:
    code = str(contract_code).strip().upper()
    for key, fix_date in MANUAL_SETTLEMENT_FIX.items():
        if key in code: return fix_date
    try:
        if len(code) < 6: return "9999/99/99"
        year, month = int(code[:4]), int(code[4:6])
        c = calendar.monthcalendar(year, month)
        wednesdays = [w[calendar.WEDNESDAY] for w in c if w[calendar.WEDNESDAY] != 0]
        if "W" in code:
            match = re.search(r"W(\d)", code)
            week_num = int(match.group(1)) if match else 99
            day = wednesdays[week_num - 1] if len(wednesdays) >= week_num else None
        else:
            day = wednesdays[2] if len(wednesdays) >= 3 else None
        return f"{year}/{month:02d}/{day:02d}" if day else "9999/99/99"
    except: return "9999/99/99"

# ==========================================
# 🕸️ 期交所/CSV 數據處理 (保留您的龍捲風數據源)
# ==========================================
@st.cache_data(ttl=300)
def fetch_detailed_options_history():
    """抓取 T 與 T-1 日資料以計算 OI 變化 (龍捲風圖專用)"""
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    now = datetime.now(tz=TW_TZ)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Origin": "https://www.taifex.com.tw", 
        "Referer": "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    }

    def fetch_day(d):
        d_str = d.strftime("%Y/%m/%d")
        payload = {"queryType": "2", "marketCode": "0", "commodity_id": "TXO", "queryDate": d_str, "MarketCode": "0", "commodity_idt": "TXO"}
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=6, verify=False)
            if "查無資料" in res.text or len(res.text) < 500: return None
            df = pd.read_html(StringIO(res.text))[0]
            df.columns = [str(c).replace(" ","").replace("*","").replace("契約","").strip() for c in df.columns]
            col_map = {}
            for c in df.columns:
                if "月" in c: col_map["Month"] = c
                elif "履約" in c: col_map["Strike"] = c
                elif "買賣" in c: col_map["Type"] = c
                elif "OI" in c or "未沖銷" in c: col_map["OI"] = c
                elif "Price" in c: col_map["Price"] = c
            if len(col_map) < 5: return None
            df = df.rename(columns=col_map)
            df = df[["Month","Strike","Type","OI","Price"]].dropna(subset=["Type"]).copy()
            df["Strike"] = pd.to_numeric(df["Strike"].astype(str).str.replace(",",""), errors="coerce")
            df["OI"] = pd.to_numeric(df["OI"].astype(str).str.replace(",",""), errors="coerce").fillna(0)
            df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",","").replace("-","0"), errors="coerce").fillna(0)
            return df
        except: return None

    # 回溯 10 天找 T (最新交易日)
    df_T, date_T = None, None
    for i in range(10):
        d = now - timedelta(days=i)
        df_T = fetch_day(d)
        if df_T is not None:
            date_T = d
            break
            
    if df_T is None: return None, None, None

    # 回溯找 T-1 (上一交易日)
    df_Prev = None
    for i in range(1, 10):
        df_Prev = fetch_day(date_T - timedelta(days=i))
        if df_Prev is not None: break

    # 合併計算 OI Change
    if df_Prev is not None:
        df_Prev = df_Prev.rename(columns={"OI": "OI_Prev"})
        df_merged = pd.merge(df_T, df_Prev[["Month", "Strike", "Type", "OI_Prev"]], on=["Month","Strike","Type"], how="left").fillna(0)
        df_merged["OI_Change"] = df_merged["OI"] - df_merged["OI_Prev"]
    else:
        df_merged = df_T
        df_merged["OI_Change"] = 0
        
    df_merged["Amount"] = df_merged["OI"] * df_merged["Price"] * 50
    return df_merged, date_T.strftime("%Y/%m/%d"), date_T

def process_uploaded_csv(uploaded_file):
    # CSV 模式 (無法計算 OI 變化，只能看靜態)
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
        df["OI_Change"] = 0 # CSV 無法計算變化
        return df, "手動上傳", datetime.now()
    except: return None, None, None

# ==========================================
# 📊 圖表繪製 (您最愛的版本 + 寬客增強)
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

def plot_tornado_chart(df_target, title_text, spot_price):
    """您指定的經典龍捲風圖 (左右對稱，每50點一跳)"""
    is_call = df_target["Type"].str.contains("買|Call", case=False, na=False)
    df_call = df_target[is_call][["Strike", "OI", "Amount"]].rename(columns={"OI": "Call_OI", "Amount": "Call_Amt"})
    df_put = df_target[~is_call][["Strike", "OI", "Amount"]].rename(columns={"OI": "Put_OI", "Amount": "Put_Amt"})
    data = pd.merge(df_call, df_put, on="Strike", how="outer").fillna(0).sort_values("Strike")

    total_put_money = data["Put_Amt"].sum()
    total_call_money = data["Call_Amt"].sum()
    data = data[(data["Call_OI"] > 300) | (data["Put_OI"] > 300)]
    
    FOCUS_RANGE = 1200
    if spot_price and spot_price > 0:
        center_price = spot_price
    elif not data.empty:
        center_price = data.loc[data["Put_OI"].idxmax(), "Strike"]
    else: center_price = 0

    if center_price > 0:
        min_s = center_price - FOCUS_RANGE
        max_s = center_price + FOCUS_RANGE
        data = data[(data["Strike"] >= min_s) & (data["Strike"] <= max_s)]

    max_oi = max(data["Put_OI"].max(), data["Call_OI"].max()) if not data.empty else 1000
    x_limit = max_oi * 1.1

    fig = go.Figure()
    # Put (Green)
    fig.add_trace(go.Bar(y=data["Strike"], x=-data["Put_OI"], orientation="h", name="Put (支撐)", marker_color="#2ca02c", opacity=0.85, 
                         customdata=data["Put_Amt"]/1e8, hovertemplate="<b>%{y}</b><br>Put OI: %{x}<br>Amt: %{customdata:.2f}億"))
    # Call (Red)
    fig.add_trace(go.Bar(y=data["Strike"], x=data["Call_OI"], orientation="h", name="Call (壓力)", marker_color="#d62728", opacity=0.85, 
                         customdata=data["Call_Amt"]/1e8, hovertemplate="<b>%{y}</b><br>Call OI: %{x}<br>Amt: %{customdata:.2f}億"))

    if spot_price and spot_price > 0:
        fig.add_hline(y=spot_price, line_dash="dash", line_color="#ff7f0e", line_width=2)
        fig.add_annotation(x=1, y=spot_price, text=f" 現貨 {int(spot_price)} ", showarrow=False, bgcolor="#ff7f0e", font=dict(color="white"))

    fig.update_layout(title=title_text, xaxis=dict(range=[-x_limit, x_limit]), yaxis=dict(dtick=50, tickformat="d"), barmode="overlay", height=750)
    return fig

def plot_quant_charts(df, spot_price):
    """寬客副圖表 (OI 變化 + GEX)"""
    # 1. OI Change
    df_c = df[df["Type"].str.contains("Call|買")].sort_values("Strike")
    df_p = df[df["Type"].str.contains("Put|賣")].sort_values("Strike")
    if spot_price > 0:
        base = round(spot_price/100)*100
        df_c = df_c[(df_c["Strike"] >= base-800) & (df_c["Strike"] <= base+800)]
        df_p = df_p[(df_p["Strike"] >= base-800) & (df_p["Strike"] <= base+800)]
    
    fig_change = go.Figure()
    fig_change.add_trace(go.Bar(x=df_c["Strike"], y=df_c["OI_Change"], name="Call Δ", marker_color="red"))
    fig_change.add_trace(go.Bar(x=df_p["Strike"], y=df_p["OI_Change"], name="Put Δ", marker_color="green"))
    fig_change.update_layout(title="近 1 日 OI 籌碼增減 (主力動向)", barmode='group', height=400)
    if spot_price > 0: fig_change.add_vline(x=spot_price, line_dash="dash", line_color="orange")

    # 2. GEX
    gex = df.groupby("Strike")["GEX"].sum().reset_index()
    if spot_price > 0:
        gex = gex[(gex["Strike"] >= base-800) & (gex["Strike"] <= base+800)]
    colors = ['red' if v >= 0 else 'green' for v in gex["GEX"]]
    fig_gex = go.Figure(go.Bar(x=gex["Strike"], y=gex["GEX"]/1e6, marker_color=colors))
    if spot_price > 0: fig_gex.add_vline(x=spot_price, line_dash="dash", line_color="orange")
    fig_gex.update_layout(title="Dealer Gamma Exposure (GEX)", yaxis_title="GEX (M)", xaxis_title="履約價", height=400)

    return fig_change, fig_gex

# ==========================================
# 🚀 主程式
# ==========================================
def main():
    st.title("🦅 台指期籌碼戰情室 (WantGoo + 龍捲風)")
    
    if st.sidebar.button("🔄 刷新數據"):
        st.cache_data.clear()
        st.rerun()

    # 1. 抓 WantGoo 數據 (宏觀)
    with st.spinner("正在連線玩股網 (WantGoo)..."):
        wg = wantgoo.fetch_summary()
        spot = wg["Spot"]
        fut = wg["Fut"]
        pc_ratio = wg["PC_Ratio"]

    # 2. 抓詳細籌碼 (微觀 - 期交所/CSV)
    with st.spinner("正在連線期交所 (建立龍捲風圖)..."):
        df_opt, date_str, data_dt = fetch_detailed_options_history()

    # --- 儀表板 (使用 WantGoo 數據) ---
    with st.container(border=True):
        st.markdown("#### 📊 市場概況 (來源: WantGoo)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("加權指數", f"{spot:.0f}" if spot else "N/A")
        basis = fut - spot if (fut and spot) else 0
        c2.metric("台指期", f"{fut:.0f}" if fut else "N/A", f"基差 {basis:.0f}", delta_color="inverse")
        pc_delta = "偏多" if pc_ratio > 100 else "偏空"
        c3.metric("P/C Ratio", f"{pc_ratio}%", pc_delta)
        c4.caption(" | ".join(wg["Msg"]))

    # 手動校正 (如果 WantGoo 沒抓到，或想用手動 CSV)
    with st.expander("🛠️ 數據校正 (手動輸入/CSV上傳)", expanded=False):
        c_up, c_in = st.columns([2, 1])
        uploaded_file = c_up.file_uploader("📂 上傳 CSV (若自動抓取失敗)", type=["csv"])
        manual_price = c_in.number_input("手動輸入現貨", value=0.0)
    
    if uploaded_file:
        df_opt, date_str, _ = process_uploaded_csv(uploaded_file)
        
    final_price = manual_price if manual_price > 0 else (spot if spot > 0 else 0)

    # --- 若無詳細籌碼，顯示警告 ---
    if df_opt is None:
        st.warning("⚠️ 無法獲取詳細履約價分佈 (龍捲風圖需詳細資料)。")
        st.info("請上傳 CSV 或稍後再試。")
        return

    # --- 合約選擇與運算 ---
    st.success(f"✅ 詳細籌碼載入成功: {date_str}")
    all_codes = sorted(df_opt["Month"].unique())
    def_idx = 0
    for i, c in enumerate(all_codes):
        if len(c) == 6 and c.isdigit(): def_idx = i; break
    sel_code = st.sidebar.selectbox("🎯 分析合約", all_codes, index=def_idx)
    target_date = get_settlement_date(sel_code)
    
    df_target = df_opt[df_opt["Month"] == sel_code].copy()
    df_calc = calculate_gex(df_target, final_price)

    # --- 主圖：龍捲風圖 (您最愛的版本) ---
    st.subheader(f"🌪️ 籌碼分布：{sel_code} (結算: {target_date})")
    st.plotly_chart(plot_tornado_chart(df_calc, f"OI 龍捲風圖 | P/C Ratio: {pc_ratio}%", final_price), use_container_width=True)

    # --- 副圖：寬客數據 (Tabs) ---
    st.markdown("### 🧬 寬客實驗室")
    tab1, tab2 = st.tabs(["🌊 OI 籌碼增減 (主力動向)", "⚡ GEX Gamma 曝險"])
    
    fig_change, fig_gex = plot_quant_charts(df_calc, final_price)
    
    with tab1:
        st.plotly_chart(fig_change, use_container_width=True)
        st.caption("顯示昨日至今日的 OI 變化。紅 Bar 代表 Call 增減，綠 Bar 代表 Put 增減。")
        
    with tab2:
        st.plotly_chart(fig_gex, use_container_width=True)
        st.caption("紅色(正): 黏滯/阻力區 | 綠色(負): 加速/滑價區")

    # --- AI 分析 (整合 WantGoo + 龍捲風數據) ---
    st.markdown("---")
    if st.button("🤖 啟動 AI 莊家分析", type="primary"):
        prompt = f"""
        你現在是台指期權的冷血莊家。
        【WantGoo 市場數據】
        - 加權指數: {spot}
        - 台指期: {fut} (基差 {basis})
        - P/C Ratio: {pc_ratio}%
        
        【微觀籌碼結構】
        - 合約: {sel_code} (結算: {target_date})
        - 龍捲風圖顯示最大 OI 區間 (請依據 P/C Ratio 判斷多空優勢)
        
        請分析：
        1. **多空情緒**：P/C Ratio {pc_ratio}% 代表散戶偏多還是偏空？莊家會如何修理他們？
        2. **關鍵點位**：結合龍捲風圖的 OI 重倉區，哪裡是莊家的防守鐵板？
        3. **結算劇本**：如果是你是莊家，你會把指數控在哪個區間結算利潤最大？
        """
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Gemini**")
            with st.spinner("Thinking..."):
                st.write(get_ai_response(prompt, "gemini"))
        with c2:
            st.markdown("**ChatGPT**")
            with st.spinner("Thinking..."):
                st.write(get_ai_response(prompt, "openai"))

if __name__ == "__main__":
    main()
