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

# 忽略 SSL 警告 (提升爬蟲成功率)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="台指期籌碼戰情室 (Quant 寬客合體版)")
TW_TZ = timezone(timedelta(hours=8))

# ==========================================
# 🔑 金鑰設定
# ==========================================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = ""

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except:
    OPENAI_API_KEY = ""

# --- 模型設定 ---
def configure_gemini(api_key):
    if not api_key: return None, "無 Key"
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model, "gemini-1.5-flash"
    except: return None, "Error"

def configure_openai(api_key):
    if not api_key: return None, "無 Key"
    try:
        client = OpenAI(api_key=api_key)
        return client, "gpt-4o-mini"
    except: return None, "Error"

gemini_model, gemini_model_name = configure_gemini(GEMINI_API_KEY)
openai_client, openai_model_name = configure_openai(OPENAI_API_KEY)


# ==========================================
# 🧮 寬客數學核心 (Black-Scholes & Greeks)
# ==========================================
class QuantEngine:
    def __init__(self, r=0.015):
        self.r = r # 無風險利率 1.5%

    def bs_price(self, S, K, T, sigma, type_='Call'):
        if T <= 0 or sigma <= 0: return 0
        try:
            d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if type_ == 'Call':
                return S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
            else:
                return K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        except: return 0

    def implied_volatility(self, price, S, K, T, type_='Call'):
        if price <= 0.1 or T <= 0: return np.nan
        try:
            def objective(sigma):
                return self.bs_price(S, K, T, sigma, type_) - price
            # IV 限制在 0.1% ~ 300%
            return brentq(objective, 0.001, 3.0)
        except: return np.nan

    def get_greeks(self, S, K, T, sigma, type_='Call'):
        if T <= 0 or sigma <= 0: return 0, 0
        try:
            d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            delta = norm.cdf(d1) if type_ == 'Call' else norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            return delta, gamma
        except: return 0, 0

quant = QuantEngine()


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
# 🕸️ 數據爬蟲 (整合外資、期貨、選擇權歷史)
# ==========================================
@st.cache_data(ttl=60)
def get_market_context():
    """抓取 現貨、期貨(盡量)、外資期貨淨額"""
    data = {"Spot": 0, "Fut": 0, "Foreign_Net": 0, "Basis": 0}
    ts = int(time.time())
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # 1. 現貨 (Yahoo)
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII?interval=1d&range=1d&_={ts}"
        res = requests.get(url, headers=headers, timeout=4)
        meta = res.json()["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice") or meta.get("chartPreviousClose")
        if price: data["Spot"] = float(price)
    except: pass
    
    # 2. 外資期貨淨額 (期交所)
    try:
        url = "https://www.taifex.com.tw/cht/3/futContractsDate"
        # 增加 verify=False 避免雲端擋 SSL
        res = requests.get(url, headers=headers, timeout=6, verify=False)
        df = pd.read_html(StringIO(res.text))[0]
        for _, row in df.iterrows():
            r_str = str(row.values)
            # 尋找 "外資" 且 "臺股期貨"
            if "外資" in r_str and ("臺股期貨" in r_str or "TX" in r_str):
                # 取出所有數字，最後一個通常是未平倉淨額
                vals = [x for x in row.values if isinstance(x, (int, float, str)) and str(x).replace(",","").replace("-","").isdigit()]
                if vals: data["Foreign_Net"] = int(str(vals[-1]).replace(",",""))
    except: pass

    # 3. 期貨價格 (如果抓不到就用現貨暫代)
    # 這裡簡化，直接用 Spot 計算基差 (假設 Fut ~ Spot)
    # 實務上 Yahoo WTX 常常抓不到，若需精確需接期貨商 API
    data["Fut"] = data["Spot"] # 預設
    
    return data

@st.cache_data(ttl=300)
def get_option_data_history():
    """抓取今天(T)與上一交易日(T-1)的選擇權資料，計算 OI Change"""
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    now = datetime.now(tz=TW_TZ)
    headers = {"User-Agent": "Mozilla/5.0"}

    def fetch_day(date_obj):
        d_str = date_obj.strftime("%Y/%m/%d")
        payload = {
            "queryType": "2", "marketCode": "0", "commodity_id": "TXO", 
            "queryDate": d_str, "MarketCode": "0", "commodity_idt": "TXO"
        }
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=6, verify=False)
            if "查無資料" in res.text or len(res.text) < 500: return None
            df = pd.read_html(StringIO(res.text))[0]
            # 清洗
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
            df["Type"] = df["Type"].astype(str).str.strip()
            df["Strike"] = pd.to_numeric(df["Strike"].astype(str).str.replace(",",""), errors="coerce")
            df["OI"] = pd.to_numeric(df["OI"].astype(str).str.replace(",",""), errors="coerce").fillna(0)
            df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",","").replace("-","0"), errors="coerce").fillna(0)
            return df
        except: return None

    # 1. 找最近的一個交易日 (T) - 回溯 10 天避免連假/週一問題
    df_T, date_T = None, None
    for i in range(10):
        d = now - timedelta(days=i)
        df_T = fetch_day(d)
        if df_T is not None:
            date_T = d
            break
            
    if df_T is None: return None, None, None

    # 2. 找上一個交易日 (T-1)
    df_Prev = None
    for i in range(1, 10):
        df_Prev = fetch_day(date_T - timedelta(days=i))
        if df_Prev is not None: break

    # 3. 合併計算 OI Change
    if df_Prev is not None:
        df_Prev = df_Prev.rename(columns={"OI": "OI_Prev"})
        df_merged = pd.merge(df_T, df_Prev[["Month", "Strike", "Type", "OI_Prev"]], 
                             on=["Month","Strike","Type"], how="left").fillna(0)
        df_merged["OI_Change"] = df_merged["OI"] - df_merged["OI_Prev"]
    else:
        df_merged = df_T
        df_merged["OI_Change"] = 0
        
    df_merged["Amount"] = df_merged["OI"] * df_merged["Price"] * 50
    return df_merged, date_T.strftime("%Y/%m/%d")

# --- 寬客數據計算 (IV, Skew, GEX) ---
def calculate_quant_metrics(df, spot_price):
    gex_list = []
    iv_list = []
    T = 5/365.0 # 簡化假設剩 5 天
    
    for _, row in df.iterrows():
        K, price, oi, cp = row["Strike"], row["Price"], row["OI"], row["Type"]
        
        # IV 計算
        iv = np.nan
        if price > 0.5 and oi > 0:
            iv = quant.implied_volatility(price, spot_price, K, T, 'Call' if 'Call' in cp else 'Put')
        iv_list.append(iv * 100 if not np.isnan(iv) else 0)
        
        # GEX 計算
        # 使用預設 IV=0.2 若無法計算，避免 GEX 為 0
        use_iv = iv if not np.isnan(iv) else 0.2
        _, gamma = quant.get_greeks(spot_price, K, T, use_iv)
        
        # GEX = Gamma * OI * Spot * 100
        # Call GEX (Dealer Short Call -> Long Hedge -> Resistance -> Positive GEX in SpotGamma notation)
        # Put GEX (Dealer Short Put -> Short Hedge -> Support/Accel -> Negative GEX)
        val = gamma * oi * spot_price * 100
        if 'Put' in cp or '賣' in cp: val = -val
        gex_list.append(val)
        
    df["GEX"] = gex_list
    df["IV"] = iv_list
    return df


# ==========================================
# 📊 圖表繪製
# ==========================================
def plot_tornado_chart(df_target, title_text, spot_price):
    # 這是您最愛的「莊家獵殺版」原版圖表
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
    fig.add_trace(go.Bar(y=data["Strike"], x=-data["Put_OI"], orientation="h", name="Put (支撐)", marker_color="#2ca02c", opacity=0.85, 
                         customdata=data["Put_Amt"]/1e8, hovertemplate="<b>%{y}</b><br>Put OI: %{x}<br>Amt: %{customdata:.2f}億"))
    fig.add_trace(go.Bar(y=data["Strike"], x=data["Call_OI"], orientation="h", name="Call (壓力)", marker_color="#d62728", opacity=0.85, 
                         customdata=data["Call_Amt"]/1e8, hovertemplate="<b>%{y}</b><br>Call OI: %{x}<br>Amt: %{customdata:.2f}億"))

    if spot_price and spot_price > 0:
        fig.add_hline(y=spot_price, line_dash="dash", line_color="#ff7f0e", line_width=2)
        fig.add_annotation(x=1, y=spot_price, text=f" 現貨 {int(spot_price)} ", showarrow=False, bgcolor="#ff7f0e", font=dict(color="white"))

    fig.update_layout(title=title_text, xaxis=dict(range=[-x_limit, x_limit]), yaxis=dict(dtick=50, tickformat="d"), barmode="overlay", height=750)
    return fig

def plot_quant_charts(df, spot_price):
    """繪製 GEX 與 OI Change 的副圖表"""
    # 1. GEX Chart
    gex = df.groupby("Strike")["GEX"].sum().reset_index()
    if spot_price > 0:
        base = round(spot_price/100)*100
        gex = gex[(gex["Strike"] >= base-800) & (gex["Strike"] <= base+800)]
    colors = ['red' if v >= 0 else 'green' for v in gex["GEX"]]
    fig_gex = go.Figure(go.Bar(x=gex["Strike"], y=gex["GEX"]/1e6, marker_color=colors))
    if spot_price > 0: fig_gex.add_vline(x=spot_price, line_dash="dash", line_color="orange")
    fig_gex.update_layout(title="Dealer Gamma Exposure (GEX)", yaxis_title="GEX (百萬)", xaxis_title="履約價", height=400)

    # 2. OI Change Chart
    df_c = df[df["Type"].str.contains("Call|買")].sort_values("Strike")
    df_p = df[df["Type"].str.contains("Put|賣")].sort_values("Strike")
    if spot_price > 0:
        df_c = df_c[(df_c["Strike"] >= base-800) & (df_c["Strike"] <= base+800)]
        df_p = df_p[(df_p["Strike"] >= base-800) & (df_p["Strike"] <= base+800)]
    
    fig_change = go.Figure()
    fig_change.add_trace(go.Bar(x=df_c["Strike"], y=df_c["OI_Change"], name="Call Δ", marker_color="red"))
    fig_change.add_trace(go.Bar(x=df_p["Strike"], y=df_p["OI_Change"], name="Put Δ", marker_color="green"))
    fig_change.update_layout(title="近 1 日 OI 籌碼變化", barmode='group', height=400)
    if spot_price > 0: fig_change.add_vline(x=spot_price, line_dash="dash", line_color="orange")

    return fig_gex, fig_change

# ==========================================
# 🚀 主程式
# ==========================================
def main():
    st.title("🦅 台指期籌碼戰情室 (莊家獵殺 + 寬客合體版)")

    if st.sidebar.button("🔄 重新整理"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("正在爬取市場全數據 (含外資、期貨、選擇權)..."):
        # 1. 抓選擇權 (含歷史 OI)
        df, data_date = get_option_data_history()
        # 2. 抓現貨與外資
        context = get_market_context()
        auto_taiex = context["Spot"]
        f_net = context["Foreign_Net"]

    if df is None:
        st.error("❌ 查無選擇權資料 (可能為期交所連線限制)。")
        return

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.sidebar.download_button("📥 下載完整數據", csv, "option_quant.csv", "text/csv")

    # ==========================================
    # 🛠️ 數據校正 (手動輸入)
    # ==========================================
    with st.expander("🛠️ 數據校正與市場概況", expanded=False):
        c1, c2, c3 = st.columns([1, 1, 2])
        c1.metric("加權指數", f"{auto_taiex:.0f}")
        c2.metric("外資期貨淨單", f"{f_net:,}", "偏多" if f_net>0 else "偏空")
        manual_price_input = c3.number_input("手動輸入點位 (Greeks計算基準)", value=0.0, step=1.0)
    
    final_taiex = manual_price_input if manual_price_input > 0 else auto_taiex

    # 計算 P/C Ratio
    total_call_amt = df[df["Type"].str.contains("Call|買")]["Amount"].sum()
    total_put_amt = df[df["Type"].str.contains("Put|賣")]["Amount"].sum()
    pc_ratio = (total_put_amt/total_call_amt*100) if total_call_amt > 0 else 0

    # 頂部儀表板
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    col1.metric("分析基準價", f"{final_taiex:.0f}", "手動" if manual_price_input>0 else "自動")
    col2.metric("P/C 金額比", f"{pc_ratio:.1f}%", "偏多" if pc_ratio>100 else "偏空")
    col3.metric("外資動向", "多方" if f_net>2000 else ("空方" if f_net<-2000 else "中性"), f"{f_net}口")
    col4.metric("資料日期", data_date)
    st.markdown("---")

    # ==========================================
    # 合約選擇與資料處理
    # ==========================================
    all_codes = sorted(df["Month"].unique())
    def_idx = 0
    for i, c in enumerate(all_codes):
        if len(c) == 6 and c.isdigit(): def_idx = i; break
    
    # 讓使用者選合約
    sel_code = st.sidebar.selectbox("🎯 選擇合約", all_codes, index=def_idx)
    target_date = get_settlement_date(sel_code)
    
    # 鎖定合約數據
    df_target = df[df["Month"] == sel_code].copy()
    
    # 🔥 執行寬客運算 (IV, GEX)
    df_calc = calculate_quant_metrics(df_target, final_taiex)
    
    # 計算 ATM Skew
    try:
        atm_row = df_calc.iloc[(df_calc['Strike'] - final_taiex).abs().argsort()[:1]]
        atm_strike = atm_row['Strike'].values[0]
        iv_c = df_calc[(df_calc['Strike']==atm_strike) & (df_calc['Type'].str.contains('Call'))]['IV'].values[0]
        iv_p = df_calc[(df_calc['Strike']==atm_strike) & (df_calc['Type'].str.contains('Put'))]['IV'].values[0]
        skew = iv_p - iv_c
    except: skew = 0

    # ==========================================
    # 📊 主圖表：龍捲風圖 (保持原樣)
    # ==========================================
    title_text = f"<b>【{sel_code}】 結算: {target_date} | P/C: {pc_ratio:.1f}%</b>"
    st.plotly_chart(plot_tornado_chart(df_calc, title_text, final_taiex), use_container_width=True)

    # ==========================================
    # 🧬 寬客副圖表 (Tabs)
    # ==========================================
    st.markdown("### 🧬 進階寬客數據")
    tab1, tab2, tab3 = st.tabs(["⚡ GEX 曝險分佈", "🌊 OI 籌碼變化", "📈 波動率 Skew"])
    
    fig_gex, fig_change = plot_quant_charts(df_calc, final_taiex)
    
    with tab1:
        st.plotly_chart(fig_gex, use_container_width=True)
        st.info("🔴 紅色 (正GEX): 黏滯區/阻力 | 🟢 綠色 (負GEX): 加速區/滑價")
    with tab2:
        st.plotly_chart(fig_change, use_container_width=True)
        st.info("顯示昨日至今日的 OI 增減 (紅Call/綠Put)，觀察主力建倉或撤退方向。")
    with tab3:
        st.metric("ATM Skew (Put IV - Call IV)", f"{skew:.2f}%", "避險情緒高" if skew > 3 else "看多情緒高")
        st.caption("若 Skew > 0 代表 Put IV 較高 (市場怕跌)；Skew < 0 代表 Call IV 較高 (市場看漲)。")

    # ==========================================
    # 🤖 AI 分析 (整合寬客數據)
    # ==========================================
    st.markdown("---")
    if st.button("🚀 啟動莊家獵殺分析", type="primary"):
        c1, c2 = st.columns(2)
        
        prompt = f"""
        你現在是台指期權的冷血莊家 (Quant)。
        【市場參數】
        - 合約: {sel_code} (結算: {target_date})
        - 基準價: {final_taiex}
        - 外資期貨淨單: {f_net} 口
        - ATM Skew: {skew:.2f}% (正值代表怕跌)
        
        【任務】
        請根據「龍捲風圖 (OI Wall)」與「GEX (Gamma Exposure)」進行分析：
        1. **肥羊與雷區**：散戶重倉區在哪？GEX 顯示哪裡是加速區(滑價)？
        2. **外資意圖**：結合期貨淨單 {f_net}，外資是想利用 GEX 加速殺盤，還是利用 OI Wall 區間盤整？
        3. **劇本**：給出一個具體的結算操作劇本。
        """
        
        with c1:
            st.markdown(f"**Gemini ({gemini_model_name})**")
            if gemini_model:
                with st.spinner("Gemini 分析中..."):
                    st.write(gemini_model.generate_content(prompt).text)
            else: st.warning("未設定 Gemini Key")

        with c2:
            st.markdown(f"**ChatGPT ({openai_model_name})**")
            if openai_client:
                with st.spinner("ChatGPT 分析中..."):
                    res = openai_client.chat.completions.create(model=openai_model_name, messages=[{"role":"user","content":prompt}])
                    st.write(res.choices[0].message.content)
            else: st.warning("未設定 OpenAI Key")

if __name__ == "__main__":
    main()
