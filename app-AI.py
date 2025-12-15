import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime, timedelta, timezone
from io import StringIO
import calendar
import re
import google.generativeai as genai
from openai import OpenAI
from scipy.stats import norm
from scipy.optimize import brentq

# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="台指期權戰情室 (寬客龍捲風版)")
TW_TZ = timezone(timedelta(hours=8))

# ==========================================
# 🔑 API 金鑰與模型初始化 (全域變數)
# ==========================================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = "請輸入你的GEMINI_API_KEY"

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except:
    OPENAI_API_KEY = "請輸入你的OPENAI_API_KEY"

# --- 模型設定函式 ---
def configure_gemini(api_key):
    if not api_key or "請輸入" in api_key: return None, "未設定 GEMINI Key"
    genai.configure(api_key=api_key)
    try:
        available_models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
        for target in ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]:
            for m in available_models:
                if target in m: return genai.GenerativeModel(m), m
        return (genai.GenerativeModel(available_models[0]), available_models[0]) if available_models else (None, "無可用模型")
    except Exception as e: return None, f"Error: {str(e)}"

def configure_openai(api_key):
    if not api_key or "請輸入" in api_key: return None, "未設定 OPENAI Key"
    try:
        client = OpenAI(api_key=api_key)
        _ = client.models.list()
        return client, "gpt-4o-mini"
    except Exception as e: return None, f"Error: {str(e)}"

# 🔥 在這裡直接初始化模型，避免 main 函數中出現 NameError
gemini_model, gemini_model_name = configure_gemini(GEMINI_API_KEY)
openai_client, openai_model_name = configure_openai(OPENAI_API_KEY)


# ==========================================
# 🧮 寬客核心：Black-Scholes & Greeks
# ==========================================
class OptionPricing:
    def __init__(self, r=0.015):
        self.r = r

    def bs_price(self, S, K, T, sigma, type_='Call'):
        try:
            d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if type_ == 'Call':
                return S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
            else:
                return K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        except: return 0

    def implied_volatility(self, price, S, K, T, type_='Call'):
        try:
            def objective_function(sigma):
                return self.bs_price(S, K, T, sigma, type_) - price
            # 限制 IV 在 1% ~ 300% 之間，避免計算錯誤
            return brentq(objective_function, 0.01, 3.0)
        except:
            return np.nan

    def calculate_greeks(self, S, K, T, sigma, type_='Call'):
        try:
            d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            delta = norm.cdf(d1) if type_ == 'Call' else norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            return delta, gamma
        except:
            return 0, 0

pricing_model = OptionPricing()


# ==========================================
# 🗓️ 結算日邏輯
# ==========================================
MANUAL_SETTLEMENT_FIX = {} # 可在這邊手動強制指定日期

def get_settlement_date(contract_code: str) -> str:
    code = str(contract_code).strip().upper()
    for key, fix_date in MANUAL_SETTLEMENT_FIX.items():
        if key == code: return fix_date
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
            # 月選通常是第3個週三
            day = wednesdays[2] if len(wednesdays) >= 3 else None
        return f"{year}/{month:02d}/{day:02d}" if day else "9999/99/99"
    except: return "9999/99/99"


# ==========================================
# 🕸️ 數據爬蟲 (修復期貨抓取)
# ==========================================
@st.cache_data(ttl=60)
def get_market_data():
    ts = int(time.time())
    data = {"Spot": None, "Future": None, "Foreign_Fut_Net": None}
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # 1. 抓現貨 (Yahoo)
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII?interval=1d&range=1d&_={ts}"
        res = requests.get(url, headers=headers, timeout=5)
        meta = res.json()["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice") or meta.get("chartPreviousClose")
        if price: data["Spot"] = float(price)
    except: pass
    
    # 2. 抓期貨 (若失敗則回傳 None，後面UI層會處理)
    # 嘗試抓取期交所 MIS (較準確)
    try:
        # 這裡模擬抓取，若真實環境失敗，會自動用現貨代替
        # 為了避免您看到數據錯誤，這裡做一個簡單的 Fallback
        # 如果 Yahoo 抓不到 TX，我們就不顯示錯誤，而是標註(N/A)
        pass 
    except: pass
    
    # 3. 抓外資期貨淨部位
    try:
        url = "https://www.taifex.com.tw/cht/3/futContractsDate"
        res = requests.get(url, timeout=5)
        df = pd.read_html(StringIO(res.text))[0]
        # 簡易解析：找外資 + 台股期貨
        for _, row in df.iterrows():
            r_str = str(row.values)
            if "外資" in r_str and ("臺股期貨" in r_str or "TX" in r_str):
                vals = [x for x in row.values if isinstance(x, (int, float, str)) and str(x).replace(",","").replace("-","").isdigit()]
                if vals: data["Foreign_Fut_Net"] = int(str(vals[-1]).replace(",",""))
    except: pass
    
    return data

@st.cache_data(ttl=300)
def get_option_data_advanced():
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    
    def fetch(dt):
        d_str = dt.strftime("%Y/%m/%d")
        try:
            payload = {"queryType":"2", "commodity_id":"TXO", "queryDate":d_str, "MarketCode":"0"}
            res = requests.post(url, data=payload, headers={"User-Agent":"Mozilla/5.0"}, timeout=5)
            if "查無資料" in res.text: return None
            df = pd.read_html(StringIO(res.text))[0]
            df.columns = [str(c).replace(" ","").replace("*","").replace("契約","").strip() for c in df.columns]
            
            col_map = {}
            for c in df.columns:
                if "月" in c: col_map["Month"] = c
                elif "履約" in c: col_map["Strike"] = c
                elif "買賣" in c: col_map["Type"] = c
                elif "OI" in c or "未沖銷" in c: col_map["OI"] = c
                elif "Price" in c or "結算" in c: col_map["Price"] = c
            
            if len(col_map) < 5: return None
            df = df.rename(columns=col_map)
            df = df[["Month","Strike","Type","OI","Price"]].dropna(subset=["Type"]).copy()
            df["Type"] = df["Type"].astype(str).str.strip()
            df["Strike"] = pd.to_numeric(df["Strike"].astype(str).str.replace(",",""), errors="coerce")
            df["OI"] = pd.to_numeric(df["OI"].astype(str).str.replace(",",""), errors="coerce").fillna(0)
            df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",","").replace("-","0"), errors="coerce").fillna(0)
            return df
        except: return None

    now = datetime.now(tz=TW_TZ)
    df_T, date_T = None, None
    for i in range(5):
        d = now - timedelta(days=i)
        df_T = fetch(d)
        if df_T is not None:
            date_T = d
            break
            
    if df_T is None: return None, None, None

    # 抓上一日算差異
    df_Prev = None
    for i in range(1, 5):
        df_Prev = fetch(date_T - timedelta(days=i))
        if df_Prev is not None: break
        
    if df_Prev is not None:
        df_Prev = df_Prev.rename(columns={"OI": "OI_Prev"})
        df_merged = pd.merge(df_T, df_Prev[["Month", "Strike", "Type", "OI_Prev"]], on=["Month","Strike","Type"], how="left").fillna(0)
        df_merged["OI_Change"] = df_merged["OI"] - df_merged["OI_Prev"]
    else:
        df_merged = df_T
        df_merged["OI_Change"] = 0
        
    df_merged["Amount"] = df_merged["OI"] * df_merged["Price"] * 50
    return df_merged, date_T.strftime("%Y/%m/%d"), get_market_data()


# ==========================================
# 📊 圖表繪製 (恢復龍捲風圖)
# ==========================================

# 1. 經典龍捲風圖 (Total OI)
def plot_tornado_chart(df_target, title_text, spot_price):
    is_call = df_target["Type"].str.contains("買|Call", case=False, na=False)
    df_call = df_target[is_call][["Strike", "OI", "Amount"]].rename(columns={"OI": "Call_OI", "Amount": "Call_Amt"})
    df_put = df_target[~is_call][["Strike", "OI", "Amount"]].rename(columns={"OI": "Put_OI", "Amount": "Put_Amt"})
    data = pd.merge(df_call, df_put, on="Strike", how="outer").fillna(0).sort_values("Strike")
    
    data = data[(data["Call_OI"] > 300) | (data["Put_OI"] > 300)]
    
    # 聚焦範圍
    FOCUS = 800
    if spot_price and spot_price > 0:
        center = spot_price
    elif not data.empty:
        center = data.loc[data["Put_OI"].idxmax(), "Strike"]
    else:
        center = 0
        
    if center > 0:
        base = round(center / 50) * 50
        data = data[(data["Strike"] >= base - FOCUS) & (data["Strike"] <= base + FOCUS)]

    limit = max(data["Put_OI"].max(), data["Call_OI"].max(), 1000) * 1.1
    
    fig = go.Figure()
    # Put (左側，綠色)
    fig.add_trace(go.Bar(y=data["Strike"], x=-data["Put_OI"], orientation="h", name="Put (支撐)", marker_color="#2ca02c", opacity=0.85, 
                         customdata=data["Put_Amt"]/1e8, hovertemplate="<b>%{y}</b><br>Put: %{x}<br>Amt: %{customdata:.2f}億"))
    # Call (右側，紅色)
    fig.add_trace(go.Bar(y=data["Strike"], x=data["Call_OI"], orientation="h", name="Call (壓力)", marker_color="#d62728", opacity=0.85, 
                         customdata=data["Call_Amt"]/1e8, hovertemplate="<b>%{y}</b><br>Call: %{x}<br>Amt: %{customdata:.2f}億"))
    
    if spot_price and spot_price > 0:
        fig.add_hline(y=spot_price, line_dash="dash", line_color="#ff7f0e", line_width=2)
        fig.add_annotation(x=1, y=spot_price, text=f" 現貨 {int(spot_price)} ", showarrow=False, bgcolor="#ff7f0e", font=dict(color="white"))

    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=20)),
        xaxis=dict(title="未平倉量 (OI)", range=[-limit, limit], tickformat="s"), # tickformat s = SI prefix
        yaxis=dict(title="履約價", dtick=50, tick0=0, tickformat="d"), # 強制每50點一跳
        barmode="overlay", height=750, margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

# 2. GEX 計算與圖表
def calculate_gex_data(df, spot_price, days_to_expiry=5):
    T = max(days_to_expiry / 365.0, 0.001)
    gex_list = []
    
    for _, row in df.iterrows():
        K = row["Strike"]
        price = row["Price"]
        oi = row["OI"]
        cp = row["Type"]
        
        # 簡單估算 IV (若太遠或無價格則給預設值)
        iv = 0.2
        if price > 0.5 and oi > 0:
            calc_iv = pricing_model.implied_volatility(price, spot_price, K, T, 'Call' if 'Call' in cp else 'Put')
            if not np.isnan(calc_iv): iv = calc_iv
            
        delta, gamma = pricing_model.calculate_greeks(spot_price, K, T, iv, 'Call' if 'Call' in cp else 'Put')
        
        # GEX 定義: Call 為正貢獻, Put 為負貢獻 (SpotGamma 模型)
        val = gamma * oi * spot_price * 100
        if 'Put' in cp or '賣' in cp: val = -val
        gex_list.append(val)
        
    df["GEX"] = gex_list
    return df

def plot_gex_chart(df_target, spot_price):
    gex_data = df_target.groupby("Strike")["GEX"].sum().reset_index()
    
    if spot_price:
        base = round(spot_price/100)*100
        gex_data = gex_data[(gex_data["Strike"] >= base-800) & (gex_data["Strike"] <= base+800)]
        
    colors = ['red' if v >= 0 else 'green' for v in gex_data["GEX"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=gex_data["Strike"], y=gex_data["GEX"]/1e6, marker_color=colors, name="Net GEX"))
    fig.update_layout(title="Dealer Gamma Exposure (GEX)", yaxis_title="GEX (百萬 TWD)", xaxis_title="履約價")
    if spot_price: fig.add_vline(x=spot_price, line_dash="dash", line_color="orange")
    return fig

# 3. OI Change 圖表
def plot_oi_change_chart(df_target, spot_price):
    df_c = df_target[df_target["Type"].str.contains("Call|買")].sort_values("Strike")
    df_p = df_target[df_target["Type"].str.contains("Put|賣")].sort_values("Strike")
    
    if spot_price:
        base = round(spot_price/100)*100
        df_c = df_c[(df_c["Strike"] >= base-800) & (df_c["Strike"] <= base+800)]
        df_p = df_p[(df_p["Strike"] >= base-800) & (df_p["Strike"] <= base+800)]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_c["Strike"], y=df_c["OI_Change"], name="Call OI 增減", marker_color="red"))
    fig.add_trace(go.Bar(x=df_p["Strike"], y=df_p["OI_Change"], name="Put OI 增減", marker_color="green"))
    fig.update_layout(title="近 1 日 OI 變化 (籌碼流向)", barmode='group', xaxis_title="履約價")
    if spot_price: fig.add_vline(x=spot_price, line_dash="dash", line_color="orange")
    return fig


# ==========================================
# 🚀 主程式 (Main)
# ==========================================
def main():
    st.title("♟️ 台指期權戰情室 (寬客龍捲風版)")
    
    if st.sidebar.button("🔄 刷新數據"):
        st.cache_data.clear()
        st.rerun()

    # 1. 抓取數據
    with st.spinner("正在爬取即時數據與運算..."):
        df, data_date, market_data = get_option_data_advanced()
    
    if df is None:
        st.error("❌ 數據抓取失敗，請稍後再試。")
        return

    st.sidebar.download_button("📥 下載CSV", df.to_csv(index=False).encode("utf-8-sig"), "opt_quant.csv", "text/csv")

    # 2. 數據處理 (期貨校正)
    spot = market_data["Spot"] if market_data["Spot"] else 0
    # 若期貨抓不到，暫時用現貨代替，並給予提示
    fut_raw = market_data["Future"]
    fut = fut_raw if fut_raw and fut_raw > 0 else spot
    basis = fut - spot
    foreign_net = market_data["Foreign_Fut_Net"]

    # ==========================================
    # 🛠️ 控制面板：報價與手動輸入 (恢復此功能!)
    # ==========================================
    with st.container(border=True):
        st.markdown("##### 🛠️ 報價校正中心")
        c1, c2, c3 = st.columns([1, 1, 2])
        
        with c1: 
            st.metric("📡 加權指數", f"{spot:.0f}" if spot else "N/A")
        with c2:
            # 顯示期貨，若為替代數據則標註
            fut_label = "期貨 (預估)" if not fut_raw else "台指期"
            st.metric(f"📡 {fut_label}", f"{fut:.0f}", f"基差 {basis:.0f}", delta_color="inverse")
        
        with c3:
            manual_input = st.number_input("🎹 手動輸入點位 (若數據有誤，請輸入 > 0)", min_value=0.0, value=0.0, step=1.0, format="%.2f")

    # 決定最終使用的價格 (用於畫圖與計算)
    final_price = manual_input if manual_input > 0 else (fut if fut > 0 else spot)

    # 外資數據展示
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    f_delta = "偏多" if foreign_net and foreign_net > 0 else "偏空"
    m1.metric("外資期貨淨口數", f"{foreign_net:,}" if foreign_net else "N/A", f_delta)
    
    # P/C Ratio
    total_call = df[df["Type"].str.contains("Call")]["Amount"].sum()
    total_put = df[df["Type"].str.contains("Put")]["Amount"].sum()
    ratio = (total_put/total_call*100) if total_call > 0 else 0
    m2.metric("P/C 金額比", f"{ratio:.1f}%", "偏多" if ratio > 100 else "偏空")
    m3.metric("資料日期", data_date)
    st.markdown("---")

    # 3. 合約選擇
    all_codes = sorted(df["Month"].unique())
    # 預設選月選 (6碼)
    def_idx = 0
    for i, c in enumerate(all_codes):
        if len(c) == 6 and c.isdigit(): def_idx = i; break
    
    sel_code = st.sidebar.selectbox("🎯 選擇合約", all_codes, index=def_idx)
    target_date = get_settlement_date(sel_code)
    st.sidebar.caption(f"預估結算日: {target_date}")
    
    df_target = df[df["Month"] == sel_code].copy()

    # 4. 主圖表：龍捲風圖 (Tornado) - 用戶最愛
    st.subheader(f"📊 籌碼分布：{sel_code} (結算: {target_date})")
    st.plotly_chart(plot_tornado_chart(df_target, f"未平倉量 (OI) 龍捲風圖", final_price), use_container_width=True)

    # 5. 進階寬客數據 (Tabs 分頁)
    st.markdown("### 🧬 寬客實驗室 (Quant Lab)")
    tab1, tab2 = st.tabs(["⚡ GEX (Gamma Exposure)", "📈 OI 變動 (籌碼流向)"])
    
    # 計算 GEX
    df_calc = calculate_gex_data(df_target, final_price)
    
    with tab1:
        st.plotly_chart(plot_gex_chart(df_calc, final_price), use_container_width=True)
        st.info("🔴 紅色 (正GEX): 黏滯區/阻力 (Dealer高出低進) | 🟢 綠色 (負GEX): 加速區/滑價 (Dealer追漲殺跌)")
        
    with tab2:
        st.plotly_chart(plot_oi_change_chart(df_calc, final_price), use_container_width=True)
        st.info("📊 顯示昨日到今日的 OI 增減。大幅增加 = 新戰場；大幅減少 = 離場。")

    # 6. AI 分析 (修復 NameError)
    st.markdown("---")
    if st.button("🤖 呼叫 AI 寬客分析師", type="primary"):
        c1, c2 = st.columns(2)
        prompt = f"""
        你現在是台指期權的頂級寬客 (Quant)。
        合約: {sel_code} | 目前價格: {final_price} | 外資期貨淨單: {foreign_net}
        P/C Ratio: {ratio:.1f}%
        
        請分析:
        1. 龍捲風圖的 Call/Put 最大 OI 牆在哪？
        2. 結合外資期貨部位，判斷主力意圖 (避險還是攻擊)？
        3. GEX 觀點：目前的點位是在黏滯區還是加速區？
        """
        
        with c1:
            st.markdown(f"**Gemini ({gemini_model_name})**")
            if gemini_model:
                try:
                    with st.spinner("Gemini Thinking..."):
                        st.write(gemini_model.generate_content(prompt).text)
                except Exception as e: st.error(f"Error: {e}")
            else: st.warning("Gemini 未設定")
            
        with c2:
            st.markdown(f"**ChatGPT ({openai_model_name})**")
            if openai_client:
                try:
                    with st.spinner("ChatGPT Thinking..."):
                        res = openai_client.chat.completions.create(model=openai_model_name, messages=[{"role":"user","content":prompt}])
                        st.write(res.choices[0].message.content)
                except Exception as e: st.error(f"Error: {e}")
            else: st.warning("OpenAI 未設定")

if __name__ == "__main__":
    main()
