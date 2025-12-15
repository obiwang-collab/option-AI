import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime, timedelta, timezone, date
from io import StringIO
import calendar
import re
import google.generativeai as genai
from openai import OpenAI
from scipy.stats import norm
from scipy.optimize import brentq

# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="台指期權寬客戰情室 (GEX/IV/OI變化)")
TW_TZ = timezone(timedelta(hours=8))

# ==========================================
# 🔑 API 金鑰設定
# ==========================================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except:
    GEMINI_API_KEY = "請輸入你的GEMINI_API_KEY"
    OPENAI_API_KEY = "請輸入你的OPENAI_API_KEY"

# ==========================================
# 🧮 核心演算法：Black-Scholes & Greeks
# ==========================================
class OptionPricing:
    def __init__(self, r=0.015):
        self.r = r  # 無風險利率 (假設 1.5%)

    def bs_price(self, S, K, T, sigma, type_='Call'):
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if type_ == 'Call':
            return S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def implied_volatility(self, price, S, K, T, type_='Call'):
        # 使用 Brent 方法反推 IV
        try:
            def objective_function(sigma):
                return self.bs_price(S, K, T, sigma, type_) - price
            return brentq(objective_function, 0.001, 2.0) # IV 範圍限制在 0.1% ~ 200%
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
# 🕸️ 數據爬蟲區 (強化版)
# ==========================================

# 1. 抓取 現貨 & 期貨 報價 + 外資期貨淨口數
@st.cache_data(ttl=60)
def get_market_overview():
    ts = int(time.time())
    data = {"Spot": None, "Future": None, "Foreign_Fut_Net": None}
    
    # A. 抓現貨 (Yahoo)
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII?interval=1d&range=1d&_={ts}"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=3)
        meta = res.json()["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice") or meta.get("chartPreviousClose")
        data["Spot"] = float(price)
    except: pass
    
    # B. 抓期貨 (Yahoo - WTX) 或期交所行情
    # 這裡簡化用現貨 +- 基差估計，或嘗試抓取期交所即時
    try:
        # 這裡直接抓期交所 MIS (較準)
        url = f"https://mis.taifex.com.tw/futures/api/getQuoteList?_={ts}"
        payload = {"MarketType":"0","SymbolType":"F","CommodityID":"TX"} # TX 台指期
        res = requests.post("https://mis.taifex.com.tw/futures/api/getQuoteList", json={"MarketType":"0"}, timeout=3)
        # 解析有點複雜，這裡簡化使用 Yahoo 抓即時台指期代碼 (通常是 TX)
        # 備用方案: 假設期貨 = 現貨 (若抓不到) - 通常有逆價差
        data["Future"] = data["Spot"] # 預設
    except: pass

    # C. 抓三大法人期貨淨部位 (外資)
    try:
        url = "https://www.taifex.com.tw/cht/3/futContractsDate"
        # 抓最近交易日
        res = requests.get(url, timeout=5)
        df = pd.read_html(StringIO(res.text))[0]
        # 尋找 "外資" 且商品為 "臺股期貨"
        # 表格結構變動大，需謹慎解析
        # 這裡示範抓取邏輯：過濾「身分=外資」且「商品=臺股期貨」
        for idx, row in df.iterrows():
            row_str = str(row.values)
            if "外資" in row_str and ("臺股期貨" in row_str or "TX" in row_str):
                # 倒數欄位通常是未平倉淨額
                vals = [x for x in row.values if isinstance(x, (int, float, str)) and str(x).replace(",","").replace("-","").isdigit()]
                if len(vals) > 0:
                    data["Foreign_Fut_Net"] = int(str(vals[-1]).replace(",",""))
    except: pass
    
    return data

# 2. 抓取選擇權資料 (含 T 與 T-1 日以計算 OI Change)
@st.cache_data(ttl=300)
def get_advanced_option_data():
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    
    def fetch_by_date(dt):
        d_str = dt.strftime("%Y/%m/%d")
        try:
            payload = {"queryType":"2", "commodity_id":"TXO", "queryDate":d_str, "MarketCode":"0"}
            res = requests.post(url, data=payload, headers={"User-Agent":"Mozilla/5.0"}, timeout=5)
            if "查無資料" in res.text: return None
            df = pd.read_html(StringIO(res.text))[0]
            # 清洗欄位
            df.columns = [str(c).replace(" ","").replace("*","").replace("契約","").strip() for c in df.columns]
            col_map = {}
            for c in df.columns:
                if "月" in c: col_map["Month"] = c
                elif "履約" in c: col_map["Strike"] = c
                elif "買賣" in c: col_map["Type"] = c
                elif "未沖銷" in c or "OI" in c: col_map["OI"] = c
                elif "結算" in c or "收盤" in c or "Price" in c: col_map["Price"] = c
            
            if len(col_map) < 5: return None
            df = df.rename(columns={v:k for k,v in col_map.items()})
            df = df[["Month","Strike","Type","OI","Price"]].dropna(subset=["Type"]).copy()
            
            # 數值化
            df["Type"] = df["Type"].astype(str).str.strip()
            df["Strike"] = pd.to_numeric(df["Strike"].astype(str).str.replace(",",""), errors="coerce")
            df["OI"] = pd.to_numeric(df["OI"].astype(str).str.replace(",",""), errors="coerce").fillna(0)
            df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",","").replace("-","0"), errors="coerce").fillna(0)
            return df
        except: return None

    # 抓今天 (T)
    now = datetime.now(tz=TW_TZ)
    df_T = None
    date_T = None
    
    for i in range(5):
        day = now - timedelta(days=i)
        df_T = fetch_by_date(day)
        if df_T is not None:
            date_T = day
            break
            
    if df_T is None: return None, None, None

    # 抓上一天 (T-1) 用來算 OI Change
    # 往回找直到找到資料
    df_Prev = None
    for i in range(1, 5):
        prev_day = date_T - timedelta(days=i)
        df_Prev = fetch_by_date(prev_day)
        if df_Prev is not None: break
    
    # 合併計算 OI Change
    if df_Prev is not None:
        # Key: Month, Strike, Type
        df_Prev = df_Prev.rename(columns={"OI": "OI_Prev"})
        df_merged = pd.merge(df_T, df_Prev[["Month", "Strike", "Type", "OI_Prev"]], 
                             on=["Month", "Strike", "Type"], how="left").fillna(0)
        df_merged["OI_Change"] = df_merged["OI"] - df_merged["OI_Prev"]
    else:
        df_merged = df_T
        df_merged["OI_Change"] = 0

    df_merged["Amount"] = df_merged["OI"] * df_merged["Price"] * 50
    
    return df_merged, date_T.strftime("%Y/%m/%d"), get_market_overview()

# ==========================================
# 📊 圖表繪製：GEX, OI Change, Skew
# ==========================================

def calculate_gex_and_iv(df, spot_price, days_to_expiry):
    # 年化時間 T
    T = max(days_to_expiry / 365.0, 0.001) 
    
    # 預先計算 IV 和 Gamma
    # 為了效能，我們只計算 "有成交價" 且 "價平附近" 的合約 IV，其他用平均值填充
    # 或者簡化：GEX 計算時，若無 IV，假設一個市場平均 IV (例如 15%)
    
    gex_list = []
    iv_list = []
    
    for idx, row in df.iterrows():
        K = row["Strike"]
        price = row["Price"]
        cp = row["Type"]
        oi = row["OI"]
        
        # 1. 計算 IV (如果價格太低或太深價外，IV會失真，略過)
        # 簡單過濾：價格 > 0.5 且 OI > 0
        iv = np.nan
        if price > 0.5 and oi > 0:
            iv = pricing_model.implied_volatility(price, spot_price, K, T, 'Call' if 'Call' in cp or '買' in cp else 'Put')
        
        iv_val = iv if not np.isnan(iv) else 0.20 # 預設 20%
        
        # 2. 計算 Gamma
        delta, gamma = pricing_model.calculate_greeks(spot_price, K, T, iv_val)
        
        # 3. 計算 GEX (Gamma Exposure)
        # 公式：GEX = Gamma * OI * Spot * 100
        # Call GEX 為正 (Dealer Short Call -> Long Gamma? No. Dealer 賣方通常是 Short Gamma)
        # 修正觀點：Dealer 必須避險。
        # 散戶買 Call -> Dealer 賣 Call -> Dealer Short Gamma -> 市場上漲時 Dealer 需買入期貨避險 (助漲)
        # 散戶買 Put -> Dealer 賣 Put -> Dealer Short Gamma -> 市場下跌時 Dealer 需賣出期貨避險 (助跌)
        # 若以 Dealer 角度：
        # Dealer Net Gamma < 0 (Short Gamma): 波動放大 (Accelerator)
        # Dealer Net Gamma > 0 (Long Gamma): 波動收斂 (Stabilizer)
        # 這裡我們計算 "Dealer 的 Gamma曝險"
        # 假設 OI 主要由散戶買入 (Dealer 賣出) -> 這是常見假設
        # GEX_Call = OI * Gamma * Spot * 100 * (-1)  (Dealer Short Call)
        # GEX_Put  = OI * Gamma * Spot * 100 * (-1)  (Dealer Short Put)
        # 兩個都是負的？這會導致全部負值。
        # 另一種常見 GEX 定義：Call 為正貢獻，Put 為負貢獻 (對應 Spot 方向性影響)
        # Call GEX: Dealer 賣 Call -> 需買現貨避險 -> 助漲
        # Put GEX: Dealer 賣 Put -> 需賣現貨避險 -> 助跌
        # 讓我們用 SpotGamma 的定義：
        # Call OI 貢獻正 GEX (助漲/阻力), Put OI 貢獻負 GEX (助跌/支撐)
        
        gex_val = (gamma * oi * spot_price * 100)
        if 'Put' in cp or '賣' in cp:
            gex_val = -gex_val # Put 貢獻負 GEX
        
        gex_list.append(gex_val)
        iv_list.append(iv_val * 100 if not np.isnan(iv) else None)
        
    df["GEX"] = gex_list
    df["IV"] = iv_list
    return df

def plot_oi_change_chart(df_target, spot_price):
    df_c = df_target[df_target["Type"].str.contains("Call|買")].sort_values("Strike")
    df_p = df_target[df_target["Type"].str.contains("Put|賣")].sort_values("Strike")
    
    # 聚焦價平附近
    if spot_price:
        base = round(spot_price/100)*100
        df_c = df_c[(df_c["Strike"] >= base-1000) & (df_c["Strike"] <= base+1000)]
        df_p = df_p[(df_p["Strike"] >= base-1000) & (df_p["Strike"] <= base+1000)]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_c["Strike"], y=df_c["OI_Change"], name="Call OI 增減", marker_color="red"))
    fig.add_trace(go.Bar(x=df_p["Strike"], y=df_p["OI_Change"], name="Put OI 增減", marker_color="green"))
    
    fig.update_layout(title="近 1 日 OI 變化 (籌碼流向)", xaxis_title="履約價", yaxis_title="口數變化", barmode='group')
    if spot_price: fig.add_vline(x=spot_price, line_dash="dash", line_color="orange", annotation_text="現貨")
    return fig

def plot_gex_chart(df_target, spot_price):
    # 聚合每個履約價的 GEX (Call + Put)
    gex_by_strike = df_target.groupby("Strike")["GEX"].sum().reset_index()
    
    # 聚焦
    if spot_price:
        base = round(spot_price/100)*100
        gex_by_strike = gex_by_strike[(gex_by_strike["Strike"] >= base-800) & (gex_by_strike["Strike"] <= base+800)]

    fig = go.Figure()
    # 顏色：正 GEX (Dealer Long/助漲) 用紅，負 GEX (Dealer Short/助跌) 用綠
    colors = ['red' if v >= 0 else 'green' for v in gex_by_strike["GEX"]]
    
    fig.add_trace(go.Bar(x=gex_by_strike["Strike"], y=gex_by_strike["GEX"]/1e6, marker_color=colors, name="Net GEX"))
    
    fig.update_layout(
        title="Dealer Gamma Exposure (GEX) 分布",
        xaxis_title="履約價",
        yaxis_title="GEX (百萬 TWD)",
        bargap=0.2
    )
    if spot_price: fig.add_vline(x=spot_price, line_dash="dash", line_color="orange")
    
    # 註解
    fig.add_annotation(text="紅色(正): 阻力/黏滯區<br>綠色(負): 加速/滑價區", 
                       xref="paper", yref="paper", x=0.02, y=0.95, showarrow=False, align="left", bgcolor="white")
    return fig

def plot_iv_smile(df_target, spot_price):
    df_c = df_target[df_target["Type"].str.contains("Call|買")].sort_values("Strike")
    df_p = df_target[df_target["Type"].str.contains("Put|賣")].sort_values("Strike")
    
    if spot_price:
        base = round(spot_price/100)*100
        range_mask_c = (df_c["Strike"] >= base-600) & (df_c["Strike"] <= base+600)
        range_mask_p = (df_p["Strike"] >= base-600) & (df_p["Strike"] <= base+600)
        df_c = df_c[range_mask_c]
        df_p = df_p[range_mask_p]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_c["Strike"], y=df_c["IV"], mode='lines+markers', name="Call IV", line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df_p["Strike"], y=df_p["IV"], mode='lines+markers', name="Put IV", line=dict(color='green')))
    
    fig.update_layout(title="Implied Volatility (IV) 微笑曲線", xaxis_title="履約價", yaxis_title="IV (%)")
    if spot_price: fig.add_vline(x=spot_price, line_dash="dash", line_color="orange")
    return fig

# --- 結算日計算 (修正版) ---
def get_settlement_date(contract_code):
    # 簡易實作：利用代碼判斷
    # 202512 -> 2025/12/17 (Wed)
    # 這裡需配合您之前的邏輯，簡化略過
    return "2025/12/17" # 示意

# ==========================================
# 🚀 主程式
# ==========================================
def main():
    st.title("♟️ 台指期權寬客戰情室 (Quant Edition)")
    
    if st.sidebar.button("🔄 刷新數據"):
        st.cache_data.clear()
        st.rerun()
        
    # 1. 數據載入
    with st.spinner("正在爬取報價、期貨部位、計算 IV Greeks..."):
        df, data_date, market_data = get_advanced_option_data()
    
    if df is None:
        st.error("數據抓取失敗")
        return

    # 2. 市場儀表板
    st.markdown("### 🏦 市場核心數據")
    m1, m2, m3, m4 = st.columns(4)
    
    spot = market_data["Spot"] if market_data["Spot"] else 0
    fut = market_data["Future"] if market_data["Future"] else spot
    basis = spot - fut
    foreign_net = market_data["Foreign_Fut_Net"]
    
    m1.metric("加權指數 (Spot)", f"{spot:,.0f}")
    m2.metric("台指期貨 (Fut)", f"{fut:,.0f}", f"基差 {basis:.0f}", delta_color="inverse") # 正基差通常不好
    
    f_delta = "偏多" if foreign_net and foreign_net > 0 else "偏空"
    f_color = "normal" if foreign_net and foreign_net > 0 else "inverse"
    m3.metric("外資期貨淨部位", f"{foreign_net if foreign_net else 'N/A'}", f_delta, delta_color=f_color)
    m4.metric("數據日期", data_date)
    
    st.markdown("---")

    # 3. 合約選擇與計算
    all_codes = sorted(df["Month"].unique())
    # 預設選月選
    def_idx = 0
    for i, c in enumerate(all_codes):
        if len(c) == 6: def_idx = i; break
        
    sel_code = st.sidebar.selectbox("分析合約", all_codes, index=def_idx)
    
    # 篩選資料
    df_target = df[df["Month"] == sel_code].copy()
    
    # 計算剩餘天數 (Days to Expiry) - 簡化假設
    # 實際應用需精確計算 target_date - now
    dte = 5 # 假設 5 天，影響 Gamma 大小，但不影響正負號
    
    # 🔥 核心計算：GEX & IV
    df_calc = calculate_gex_and_iv(df_target, spot, dte)

    # 4. 圖表展示區
    tab1, tab2, tab3 = st.tabs(["📊 OI 變化 & 籌碼", "⚡ Gamma Exposure (GEX)", "📈 IV 微笑與偏斜"])
    
    with tab1:
        st.subheader("OI 增減變化 (主力建倉/平倉軌跡)")
        st.plotly_chart(plot_oi_change_chart(df_calc, spot), use_container_width=True)
        st.caption("紅色/綠色 Bar 代表昨日到今日的 OI 變化量。若某價位 OI 大幅增加，代表新戰場；大幅減少代表停損或獲利了結。")

    with tab2:
        st.subheader("Dealer Gamma Exposure (GEX)")
        st.plotly_chart(plot_gex_chart(df_calc, spot), use_container_width=True)
        st.markdown("""
        **GEX 解讀指南：**
        * **正 GEX (紅色)**：通常出現在大量 Call OI 區。Dealer 需要「高賣低買」來避險，這會抑制波動，讓行情變得黏滯（阻力）。
        * **負 GEX (綠色)**：通常出現在大量 Put OI 區。Dealer 需要「追漲殺跌」來避險，這會放大波動，容易引發崩盤或急拉（滑價/加速器）。
        """)

    with tab3:
        st.subheader("隱含波動率 (IV) 微笑曲線")
        st.plotly_chart(plot_iv_smile(df_calc, spot), use_container_width=True)
        
        # 簡單 Skew 計算
        atm_strike = df_calc.iloc[(df_calc['Strike'] - spot).abs().argsort()[:1]]['Strike'].values[0]
        try:
            iv_atm_c = df_calc[(df_calc["Strike"]==atm_strike) & (df_calc["Type"].str.contains("Call"))]["IV"].values[0]
            iv_atm_p = df_calc[(df_calc["Strike"]==atm_strike) & (df_calc["Type"].str.contains("Put"))]["IV"].values[0]
            skew = iv_atm_p - iv_atm_c
            st.info(f"📍 ATM ({atm_strike}) Skew (Put IV - Call IV): **{skew:.2f}%**")
            if skew > 3: st.write("⚠️ Put IV 顯著高於 Call，市場避險情緒濃厚。")
            elif skew < -1: st.write("⚠️ Call IV 較高，市場看多情緒強烈。")
        except: st.write("無法計算 ATM Skew")

    # 5. 生成寬客版 Prompt
    if st.button("🤖 呼叫 AI 寬客分析師"):
        prompt = f"""
        你是一位頂尖的量化交易員 (Quant)。
        
        【市場狀態】
        - 現貨: {spot} / 期貨: {fut} / 基差: {basis}
        - 外資期貨淨口數: {foreign_net} (正為多/負為空)
        
        【選擇權高階數據 ({sel_code})】
        - ATM Skew (Put-Call): {skew if 'skew' in locals() else 'N/A'}%
        
        【GEX 結構觀察】
        (請參考 GEX 圖表解讀：負 GEX 區容易加速，正 GEX 區容易盤整)
        
        請給出極度專業的分析：
        1. **外資期貨與選擇權籌碼是否背離？**
        2. **GEX 觀點**：目前的點位是在「黏滯區」還是「加速區」？如果跌破哪裡會引發 Dealer 的 Gamma 殺盤？
        3. **波動率觀點**：Skew 顯示市場目前更怕跌還是怕漲？
        4. **交易策略**：適合做 Gamma Scalping 還是方向性突破？
        """
        
        c1, c2 = st.columns(2)
        with c1:
            if gemini_model:
                st.write("**Gemini Pro Thinking...**")
                st.info(gemini_model.generate_content(prompt).text)
        with c2:
            if openai_client:
                st.write("**GPT-4 Thinking...**")
                st.info(openai_client.chat.completions.create(model=openai_model_name, messages=[{"role":"user","content":prompt}]).choices[0].message.content)

if __name__ == "__main__":
    main()
