import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import time
from datetime import datetime, timedelta, timezone
from io import StringIO
import calendar
import re
import google.generativeai as genai
from openai import OpenAI
import os

# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="台指期籌碼戰情室 (莊家獵殺版)")
TW_TZ = timezone(timedelta(hours=8))

# ==========================================
# 🔑 金鑰設定區 (雲端安全版)
# ==========================================

# 1. 讀取 GEMINI API Key
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = "請輸入你的GEMINI_API_KEY"

# 2. 讀取 OPENAI API Key
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = "請輸入你的OPENAI_API_KEY"

# --- 智慧模型設定：Gemini ---
def configure_gemini(api_key):
    if not api_key or "請輸入" in api_key:
        return None, "尚未設定 GEMINI Key"

    genai.configure(api_key=api_key)
    try:
        available_models = [
            m.name for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]

        # 優先使用 2.5 Flash -> 1.5 Flash -> Pro
        for target in [
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-pro",
        ]:
            for m in available_models:
                if target in m:
                    return genai.GenerativeModel(m), m

        if available_models:
            return genai.GenerativeModel(available_models[0]), available_models[0]
        return None, "無可用模型"
    except Exception as e:
        return None, f"連線錯誤: {str(e)}"


# --- 智慧模型設定：OpenAI ---
def configure_openai(api_key):
    if not api_key or "請輸入" in api_key:
        return None, "尚未設定 OPENAI Key"

    try:
        client = OpenAI(api_key=api_key)
        # 試探呼叫確認 Key 有效
        _ = client.models.list()
        return client, "gpt-4o-mini"  # 建議改用 4o-mini 或 3.5-turbo
    except Exception as e:
        return None, f"連線錯誤: {str(e)}"


# 初始化模型
gemini_model, gemini_model_name = configure_gemini(GEMINI_API_KEY)
openai_client, openai_model_name = configure_openai(OPENAI_API_KEY)

# 手動修正結算日：個別特例可放這裡
MANUAL_SETTLEMENT_FIX = {
    "202501W1": "2025/01/02",
}


# --- 結算日計算 ---
def get_settlement_date(contract_code: str) -> str:
    code = str(contract_code).strip().upper()
    for key, fix_date in MANUAL_SETTLEMENT_FIX.items():
        if key in code:
            return fix_date

    try:
        if len(code) < 6:
            return "9999/99/99"

        year = int(code[:4])
        month = int(code[4:6])

        c = calendar.monthcalendar(year, month)
        wednesdays = [
            week[calendar.WEDNESDAY] for week in c if week[calendar.WEDNESDAY] != 0
        ]
        fridays = [week[calendar.FRIDAY] for week in c if week[calendar.FRIDAY] != 0]
        day = None

        if "W" in code:
            match = re.search(r"W(\d)", code)
            if match:
                week_num = int(match.group(1))
                if len(wednesdays) >= week_num:
                    day = wednesdays[week_num - 1]
        elif "F" in code:
            match = re.search(r"F(\d)", code)
            if match:
                week_num = int(match.group(1))
                if len(fridays) >= week_num:
                    day = fridays[week_num - 1]
        else:
            if len(wednesdays) >= 3:
                day = wednesdays[2]

        if day:
            return f"{year}/{month:02d}/{day:02d}"
        else:
            return "9999/99/99"
    except Exception:
        return "9999/99/99"


# --- 現貨即時價 (強化版) ---
@st.cache_data(ttl=60)
def get_realtime_data():
    taiex = None
    ts = int(time.time())
    
    # 偽裝 Headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }

    # 1) 優先嘗試 Yahoo Finance
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII?interval=1d&range=1d&_={ts}"
        res = requests.get(url, headers=headers, timeout=5)
        data = res.json()
        meta = data["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice")
        if price is None:
            price = meta.get("chartPreviousClose")
        if price:
            taiex = float(price)
    except Exception:
        pass

    # 2) 備援：證交所 MIS
    if taiex is None:
        try:
            url = f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_t00.tw&json=1&delay=0&_={ts}000"
            res = requests.get(url, timeout=3)
            data = res.json()
            if "msgArray" in data and len(data["msgArray"]) > 0:
                val = data["msgArray"][0].get("z", "-")
                if val == "-": val = data["msgArray"][0].get("o", "-")
                if val == "-": val = data["msgArray"][0].get("y", "-")
                if val != "-": taiex = float(val)
        except Exception:
            pass

    return taiex

# --- 期交所選擇權資料 ---
@st.cache_data(ttl=300)
def get_option_data():
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    headers = {"User-Agent": "Mozilla/5.0"}

    for i in range(5):
        query_date = (
            datetime.now(tz=TW_TZ) - timedelta(days=i)
        ).strftime("%Y/%m/%d")
        payload = {
            "queryType": "2",
            "marketCode": "0",
            "dateaddcnt": "",
            "commodity_id": "TXO",
            "commodity_id2": "",
            "queryDate": query_date,
            "MarketCode": "0",
            "commodity_idt": "TXO",
        }
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=5)
            if "查無資料" in res.text or len(res.text) < 500:
                continue

            dfs = pd.read_html(StringIO(res.text))
            df = dfs[0]

            df.columns = [
                str(c).replace(" ", "").replace("*", "").replace("契約", "").strip()
                for c in df.columns
            ]
            
            # 動態抓取欄位
            month_col = next((c for c in df.columns if "月" in c or "週" in c), None)
            strike_col = next((c for c in df.columns if "履約" in c), None)
            type_col = next((c for c in df.columns if "買賣" in c), None)
            oi_col = next((c for c in df.columns if "未沖銷" in c or "OI" in c), None)
            price_col = next((c for c in df.columns if "結算" in c or "收盤" in c or "Price" in c), None)

            if not all([month_col, strike_col, type_col, oi_col, price_col]):
                continue

            df = df.rename(columns={
                month_col: "Month",
                strike_col: "Strike",
                type_col: "Type",
                oi_col: "OI",
                price_col: "Price",
            })

            cols_to_keep = ["Month", "Strike", "Type", "OI", "Price"]
            df = df[cols_to_keep].copy()
            df = df.dropna(subset=["Type"])
            
            # 資料清洗
            df["Type"] = df["Type"].astype(str).str.strip()
            df["Strike"] = pd.to_numeric(df["Strike"].astype(str).str.replace(",", ""), errors="coerce")
            df["OI"] = pd.to_numeric(df["OI"].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
            df["Price"] = df["Price"].astype(str).str.replace(",", "").replace("-", "0")
            df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)

            # 計算金額 (OI * Price * 50)
            df["Amount"] = df["OI"] * df["Price"] * 50

            if df["OI"].sum() == 0:
                continue

            return df, query_date
        except Exception:
            continue

    return None, None

# --- Tornado 圖 ---
def plot_tornado_chart(df_target, title_text, spot_price):
    is_call = df_target["Type"].str.contains("買|Call", case=False, na=False)
    df_call = df_target[is_call][["Strike", "OI", "Amount"]].rename(
        columns={"OI": "Call_OI", "Amount": "Call_Amt"}
    )
    df_put = df_target[~is_call][["Strike", "OI", "Amount"]].rename(
        columns={"OI": "Put_OI", "Amount": "Put_Amt"}
    )
    data = (
        pd.merge(df_call, df_put, on="Strike", how="outer")
        .fillna(0)
        .sort_values("Strike")
    )

    total_put_money = data["Put_Amt"].sum()
    total_call_money = data["Call_Amt"].sum()

    data = data[(data["Call_OI"] > 300) or (data["Put_OI"] > 300)]
    
    # 聚焦
    FOCUS_RANGE = 1200
    if spot_price and spot_price > 0:
        center_price = spot_price
    elif not data.empty:
        center_price = data.loc[data["Put_OI"].idxmax(), "Strike"]
    else:
        center_price = 0

    if center_price > 0:
        min_s = center_price - FOCUS_RANGE
        max_s = center_price + FOCUS_RANGE
        data = data[(data["Strike"] >= min_s) & (data["Strike"] <= max_s)]

    max_oi = max(data["Put_OI"].max(), data["Call_OI"].max()) if not data.empty else 1000
    x_limit = max_oi * 1.1

    fig = go.Figure()
    # Put
    fig.add_trace(go.Bar(y=data["Strike"], x=-data["Put_OI"], orientation="h", name="Put (支撐)", marker_color="#2ca02c", opacity=0.85, customdata=data["Put_Amt"] / 100000000, hovertemplate="<b>履約價: %{y}</b><br>Put OI: %{x} 口<br>Put 市值: %{customdata:.2f}億<extra></extra>"))
    # Call
    fig.add_trace(go.Bar(y=data["Strike"], x=data["Call_OI"], orientation="h", name="Call (壓力)", marker_color="#d62728", opacity=0.85, customdata=data["Call_Amt"] / 100000000, hovertemplate="<b>履約價: %{y}</b><br>Call OI: %{x} 口<br>Call 市值: %{customdata:.2f}億<extra></extra>"))

    annotations = []
    if spot_price and spot_price > 0 and not data.empty:
        if data["Strike"].min() <= spot_price <= data["Strike"].max():
            fig.add_hline(y=spot_price, line_dash="dash", line_color="#ff7f0e", line_width=2)
            annotations.append(dict(x=1, y=spot_price, xref="paper", yref="y", text=f" 現貨 {int(spot_price)} ", showarrow=False, xanchor="left", align="center", font=dict(color="white", size=12), bgcolor="#ff7f0e", bordercolor="#ff7f0e", borderpad=4))

    annotations.append(dict(x=0.02, y=1.05, xref="paper", yref="paper", text=f"<b>Put 總金額</b><br>{total_put_money/100000000:.1f} 億", showarrow=False, align="left", font=dict(size=14, color="#2ca02c"), bgcolor="white", bordercolor="#2ca02c", borderwidth=2, borderpad=6))
    annotations.append(dict(x=0.98, y=1.05, xref="paper", yref="paper", text=f"<b>Call 總金額</b><br>{total_call_money/100000000:.1f} 億", showarrow=False, align="right", font=dict(size=14, color="#d62728"), bgcolor="white", bordercolor="#d62728", borderwidth=2, borderpad=6))

    fig.update_layout(title=dict(text=title_text, y=0.95, x=0.5, xanchor="center", yanchor="top", font=dict(size=20, color="black")), xaxis=dict(title="未平倉量 (OI)", range=[-x_limit, x_limit], showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor="black", tickmode="array", tickvals=[-x_limit * 0.75, -x_limit * 0.5, -x_limit * 0.25, 0, x_limit * 0.25, x_limit * 0.5, x_limit * 0.75], ticktext=[f"{int(x_limit*0.75)}", f"{int(x_limit*0.5)}", f"{int(x_limit*0.25)}", "0", f"{int(x_limit*0.25)}", f"{int(x_limit*0.5)}", f"{int(x_limit*0.75)}"]), yaxis=dict(title="履約價", tickmode="linear", dtick=100, tickformat="d"), barmode="overlay", legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"), height=750, margin=dict(l=40, r=80, t=140, b=60), annotations=annotations, paper_bgcolor="white", plot_bgcolor="white")
    return fig

# --- AI 分析函式 (Gemini - 莊家獵殺版) ---
def ask_gemini_brief(df_recent, taiex_price, contract_code, settlement_date):
    if not gemini_model:
        return f"⚠️ {gemini_model_name}"

    try:
        df_ai = df_recent.copy()
        if "Amount" in df_ai.columns:
            df_ai = df_ai.nlargest(80, "Amount")

        data_str = df_ai.to_csv(index=False)

        # 🚀 提示詞修改重點：莊家思維、無風險提示、詳細劇本
        prompt = f"""
你現在是台指選擇權市場的主力莊家。你的目標只有一個：**在結算日吃掉最多散戶的權利金，讓自己的利潤最大化**。

市場現況
- 結算合約：{contract_code} (結算日: {settlement_date})
- 現貨指數（即時運算）：{taiex_price}
  (請以此價格為目前的控盤基準，忽略過時數據)

任務
請根據 CSV 籌碼數據（OI 與 Amount），進行深度的控盤推演：

1. **肥羊與雷區分析**：
   - **肥羊區**：散戶在哪個價位重倉（高 OI/金額）？那是你的獵殺目標。
   - **雷區**：你的防守底線（莊家賣出最多會賠錢的價位）在哪？絕對不能讓指數過去。

2. **操盤劇本 (Script)**：
   - 請寫出未來幾天的「畫線」劇本。例如：「週一先殺破 xxxx 引發多頭停損，週二再急拉至 xxxx 軋空」。
   - 如果指數現在卡在中間，你會如何利用「雙巴」來消耗時間價值？

3. **最佳結算目標**：
   - 給出一個具體的「點位」或「窄區間」，這是能讓 Call/Put 雙邊通殺的完美結局。

4. **莊家指令**：
   - 給出簡短有力的指令（例如：Sell Call @ 28200, Buy Put @ 28000 避險）。

回答要求
- 使用第一人稱（本莊、我）。
- 語氣：**自信、冷血、貪婪**。
- **不要**給出任何風險警語或教育性廢話。
- 分析要詳細，字數約 300-500 字。

數據：
{data_str}
"""
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"分析忙碌中 ({str(e)})"

# --- AI 分析函式 (ChatGPT - 莊家獵殺版) ---
def ask_openai_brief(df_recent, taiex_price, contract_code, settlement_date):
    if not openai_client:
        return f"⚠️ {openai_model_name}"

    try:
        df_ai = df_recent.copy()
        if "Amount" in df_ai.columns:
            df_ai = df_ai.nlargest(80, "Amount")

        data_str = df_ai.to_csv(index=False)

        # 🚀 提示詞修改重點：莊家思維、無風險提示、詳細劇本
        user_prompt = f"""
你現在是台指選擇權市場的主力莊家。你的目標只有一個：**在結算日吃掉最多散戶的權利金，讓自己的利潤最大化**。

市場現況
- 結算合約：{contract_code} (結算日: {settlement_date})
- 現貨指數（即時運算）：{taiex_price}
  (請以此價格為目前的控盤基準，忽略過時數據)

任務
請根據 CSV 籌碼數據（OI 與 Amount），進行深度的控盤推演：

1. **肥羊與雷區分析**：
   - **肥羊區**：散戶在哪個價位重倉（高 OI/金額）？那是你的獵殺目標。
   - **雷區**：你的防守底線（莊家賣出最多會賠錢的價位）在哪？絕對不能讓指數過去。

2. **操盤劇本 (Script)**：
   - 請寫出未來幾天的「畫線」劇本。例如：「週一先殺破 xxxx 引發多頭停損，週二再急拉至 xxxx 軋空」。
   - 如果指數現在卡在中間，你會如何利用「雙巴」來消耗時間價值？

3. **最佳結算目標**：
   - 給出一個具體的「點位」或「窄區間」，這是能讓 Call/Put 雙邊通殺的完美結局。

4. **莊家指令**：
   - 給出簡短有力的指令（例如：Sell Call @ 28200, Buy Put @ 28000 避險）。

回答要求
- 使用第一人稱（本莊、我）。
- 語氣：**自信、冷血、貪婪**。
- **不要**給出任何風險警語或教育性廢話。
- 分析要詳細，字數約 300-500 字。

數據：
{data_str}
"""

        response = openai_client.chat.completions.create(
            model=openai_model_name,
            messages=[
                {
                    "role": "system",
                    "content": "你是一個冷血、唯利是圖的期貨市場莊家。你不關心散戶死活，只在乎如何讓選擇權歸零。",
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,  # 提高創造力，讓劇本更豐富
            max_tokens=800,   # 增加 token 限制，允許更長的分析
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"分析忙碌中 ({str(e)})"


# --- 主程式 ---
def main():
    st.title("🤖 台指期籌碼戰情室 (莊家獵殺版)")

    # 側邊欄重新整理 (Key 修正)
    if st.sidebar.button("🔄 重新整理", key="refresh_main"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("連線期交所中..."):
        df, data_date = get_option_data()
        auto_taiex = get_realtime_data()  # 改名為 auto_taiex 以示區別

    if df is None:
        st.error("查無資料，請稍後再試。")
        return

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.sidebar.download_button(
        "📥 下載完整數據",
        csv,
        f"option_{data_date.replace('/','')}.csv",
        "text/csv",
    )

    # ==========================================
    # 🆕 新增功能：手動校正現貨價格 (不破壞原版面，增加在數據列上方)
    # ==========================================
    with st.expander("🛠️ 數據校正設定 (若現貨/期貨價格延遲，請點此展開輸入)", expanded=False):
        mc1, mc2 = st.columns([1, 2])
        with mc1:
            st.info(f"系統自動抓取: {auto_taiex}")
        with mc2:
            manual_price_input = st.number_input(
                "請輸入看盤軟體最新價格 (輸入 0 代表使用系統自動數據):",
                min_value=0.0,
                value=0.0,
                step=1.0,
                format="%.2f"
            )
    
    # --- 核心邏輯判定 ---
    if manual_price_input > 0:
        final_taiex = manual_price_input
        price_source_msg = "⚠️ 手動校正"
    else:
        final_taiex = auto_taiex if auto_taiex else 0
        price_source_msg = "系統自動"

    # ==========================================

    total_call_amt = df[df["Type"].str.contains("買|Call", case=False, na=False)]["Amount"].sum()
    total_put_amt = df[df["Type"].str.contains("賣|Put", case=False, na=False)]["Amount"].sum()
    pc_ratio_amt = ((total_put_amt / total_call_amt) * 100 if total_call_amt > 0 else 0)

    c1, c2, c3, c4 = st.columns([1.2, 0.8, 1, 1])
    c1.markdown(f"<div style='text-align: left;'><span style='font-size: 14px; color: #555;'>製圖時間</span><br><span style='font-size: 18px; font-weight: bold;'>{datetime.now(tz=TW_TZ).strftime('%Y/%m/%d %H:%M:%S')}</span></div>", unsafe_allow_html=True)
    
    # 這裡使用 final_taiex 顯示
    c2.metric(f"大盤/期貨 ({price_source_msg})", f"{int(final_taiex) if final_taiex else 'N/A'}")
    
    trend = "偏多" if pc_ratio_amt > 100 else "偏空"
    c3.metric("全市場 P/C 金額比", f"{pc_ratio_amt:.1f}%", f"{trend}格局", delta_color="normal" if pc_ratio_amt > 100 else "inverse")
    c4.metric("資料來源日期", data_date)
    st.markdown("---")

    # ==========================================
    # 選出「距離現在最近的結算合約」
    # ==========================================
    nearest_code = None
    nearest_date = None
    nearest_df = None
    plot_targets = []

    unique_codes = df["Month"].unique()
    all_contracts = []

    for code in unique_codes:
        s_date_str = get_settlement_date(code)
        if s_date_str == "9999/99/99" or s_date_str <= data_date:
            continue
        all_contracts.append({"code": code, "date": s_date_str})

    all_contracts.sort(key=lambda x: x["date"])

    if all_contracts:
        nearest = all_contracts[0]
        nearest_code = nearest["code"]
        nearest_date = nearest["date"]
        nearest_df = df[df["Month"] == nearest_code]

        plot_targets.append({"title": "最近結算", "info": nearest})
        monthly = next((c for c in all_contracts if len(c["code"]) == 6), None)
        if monthly:
            if monthly["code"] != nearest_code:
                plot_targets.append({"title": "當月月選", "info": monthly})
            else:
                plot_targets[0]["title"] = "最近結算 (同月選)"

    # 🆕 Call/Put OI 全履約價分布
    st.markdown("### Call/Put OI 全履約價分布")
    # 將全市場所有履約價的 Call 與 Put 未平倉量彙總
    df_all_call = df[df["Type"].str.contains("買|Call", case=False, na=False)]
    df_all_put = df[df["Type"].str.contains("賣|Put", case=False, na=False)]
    call_oi_by_strike = df_all_call.groupby("Strike")["OI"].sum()
    put_oi_by_strike = df_all_put.groupby("Strike")["OI"].sum()
    df_oi = pd.DataFrame({"Strike": call_oi_by_strike.index.union(put_oi_by_strike.index)})
    df_oi["Call_OI"] = df_oi["Strike"].map(call_oi_by_strike).fillna(0)
    df_oi["Put_OI"] = df_oi["Strike"].map(put_oi_by_strike).fillna(0)
    df_oi = df_oi.sort_values("Strike")
    # 使用 Plotly 繪製 OI 分布圖
    fig_oi = go.Figure()
    fig_oi.add_trace(go.Scatter(x=df_oi["Strike"], y=df_oi["Call_OI"], mode="lines", name="Call OI"))
    fig_oi.add_trace(go.Scatter(x=df_oi["Strike"], y=df_oi["Put_OI"], mode="lines", name="Put OI"))
    fig_oi.update_layout(title="全履約價 Call/Put 未平倉量分布", xaxis_title="履約價", yaxis_title="未平倉量 (口)")
    st.plotly_chart(fig_oi, use_container_width=True)

    # 🆕 近三日 OI 變化分析
    st.markdown("### 近三日 OI 變化分析")
    # 定義函數取得指定日期的選擇權資料 DataFrame
    def fetch_option_data_for_date(date_str):
        url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
        headers = {"User-Agent": "Mozilla/5.0"}
        payload = {
            "queryType": "2",
            "marketCode": "0",
            "dateaddcnt": "",
            "commodity_id": "TXO",
            "commodity_id2": "",
            "queryDate": date_str,
            "MarketCode": "0",
            "commodity_idt": "TXO"
        }
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=5)
            if "查無資料" in res.text or len(res.text) < 1000:
                return None
            dfs = pd.read_html(StringIO(res.text))
            df_temp = dfs[0]
            df_temp.columns = [str(c).replace(" ", "").replace("*", "").replace("契約", "").strip() for c in df_temp.columns]
            month_col = next((c for c in df_temp.columns if "月" in c or "週" in c), None)
            strike_col = next((c for c in df_temp.columns if "履約" in c), None)
            type_col = next((c for c in df_temp.columns if "買賣" in c), None)
            oi_col = next((c for c in df_temp.columns if "未沖銷" in c or "OI" in c), None)
            price_col = next((c for c in df_temp.columns if "結算" in c or "收盤" in c or "Price" in c), None)
            if not all([month_col, strike_col, type_col, oi_col, price_col]):
                return None
            df_temp = df_temp.rename(columns={
                month_col: "Month",
                strike_col: "Strike",
                type_col: "Type",
                oi_col: "OI",
                price_col: "Price"
            })
            df_temp = df_temp[["Month", "Strike", "Type", "OI", "Price"]].copy()
            df_temp = df_temp.dropna(subset=["Type"])
            df_temp["Type"] = df_temp["Type"].astype(str).str.strip()
            df_temp["Strike"] = pd.to_numeric(df_temp["Strike"].astype(str).replace(",", "", regex=True), errors="coerce")
            df_temp["OI"] = pd.to_numeric(df_temp["OI"].astype(str).replace(",", "", regex=True), errors="coerce").fillna(0)
            return df_temp
        except Exception:
            return None

    # 抓取連續三個交易日的 OI 資料
    dates_data = []
    for i in range(0, 7):
        date_iter = (datetime.now(tz=TW_TZ) - timedelta(days=i)).strftime("%Y/%m/%d")
        if date_iter == data_date:
            dates_data.append((date_iter, df))
        else:
            df_iter = fetch_option_data_for_date(date_iter)
            if df_iter is not None:
                dates_data.append((date_iter, df_iter))
        if len(dates_data) >= 3:
            break
    dates_data.sort(key=lambda x: x[0])
    if len(dates_data) < 3:
        st.error("無法取得近三日資料，請稍後再試。")
    else:
        date_old, df_old = dates_data[0]
        date_mid, df_mid = dates_data[1]
        date_new, df_new = dates_data[2]
        # 按 Strike+Type 聚合每日期 OI 總和
        df_old_tot = df_old.groupby(["Strike", "Type"])["OI"].sum().reset_index()
        df_new_tot = df_new.groupby(["Strike", "Type"])["OI"].sum().reset_index()
        # 合併新舊OI資料
        df_merge = pd.merge(df_new_tot, df_old_tot, on=["Strike", "Type"], how="outer", suffixes=("_new", "_old")).fillna(0)
        df_merge["OI_change"] = df_merge["OI_new"] - df_merge["OI_old"]
        # 切分 Call/Put
        call_changes = df_merge[df_merge["Type"].str.contains("買|Call", case=False, na=False)].set_index("Strike")["OI_change"]
        put_changes = df_merge[df_merge["Type"].str.contains("賣|Put", case=False, na=False)].set_index("Strike")["OI_change"]
        strikes_all = sorted(set(call_changes.index).union(set(put_changes.index)))
        call_vals = [call_changes.get(x, 0) for x in strikes_all]
        put_vals = [put_changes.get(x, 0) for x in strikes_all]
        fig_change = go.Figure()
        fig_change.add_trace(go.Bar(x=strikes_all, y=call_vals, name="Call OI 增減"))
        fig_change.add_trace(go.Bar(x=strikes_all, y=put_vals, name="Put OI 增減"))
        fig_change.update_layout(barmode='group', title=f"{date_old} → {date_new} 各履約價OI增減", xaxis_title="履約價", yaxis_title="OI 增減 (口)")
        st.plotly_chart(fig_change, use_container_width=True)

    # 🆕 IV 與 Skew 分析
    st.markdown("### IV 與 Skew 分析")
    # 定義 Black-Scholes定價與隱含波動率計算函數
    from math import log, sqrt, exp
    from statistics import NormalDist
    def bs_price(S, K, T, r, sigma, option_type="call"):
        if T <= 0 or sigma <= 0:
            return max(S-K, 0) if option_type == "call" else max(K-S, 0)
        d1 = (log(S/K) + (0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        N = NormalDist(0, 1)
        if option_type == "call":
            return S * N.cdf(d1) - K * exp(-r*T) * N.cdf(d2)
        else:
            return K * exp(-r*T) * N.cdf(-d2) - S * N.cdf(-d1)
    def implied_vol(S, K, T, r, market_price, option_type="call"):
        tol = 1e-4
        max_iter = 50
        low, high = 1e-6, 5.0
        vol = None
        for _ in range(max_iter):
            mid = 0.5 * (low + high)
            price = bs_price(S, K, T, r, mid, option_type)
            if abs(price - market_price) < tol:
                vol = mid
                break
            if price > market_price:
                high = mid
            else:
                low = mid
            vol = mid
        return vol
    # 選擇近月合約 (當月月選優先)
    if 'monthly' in locals() and monthly:
        skew_code = monthly["code"]
        skew_date = monthly["date"]
    else:
        skew_code = nearest_code
        skew_date = nearest_date
    df_skew = df[df["Month"] == skew_code] if skew_code else pd.DataFrame()
    if df_skew.empty:
        st.info("找不到近月合約資料，無法計算 IV 與 Skew。")
    else:
        spot = final_taiex if final_taiex else 0
        # 計算距離到期的年化時間
        try:
            exp_date = datetime.strptime(skew_date, "%Y/%m/%d")
            days_to_exp = (exp_date - datetime.now(tz=TW_TZ)).days
        except:
            days_to_exp = 0
        T = max(days_to_exp, 0) / 252
        strikes_range = sorted(x for x in df_skew["Strike"].unique() if spot-300 <= x <= spot+300)
        iv_points = {"Strike": [], "IV": []}
        for K in strikes_range:
            price_call = None
            price_put = None
            call_rows = df_skew[(df_skew["Strike"] == K) & (df_skew["Type"].str.contains("買|Call"))]
            put_rows = df_skew[(df_skew["Strike"] == K) & (df_skew["Type"].str.contains("賣|Put"))]
            if not call_rows.empty:
                price_call = call_rows["Price"].iloc[0]
            if not put_rows.empty:
                price_put = put_rows["Price"].iloc[0]
            iv_val = None
            if price_call is not None and price_call > 0:
                iv_val = implied_vol(spot, K, T, 0.0, price_call, "call")
            elif price_put is not None and price_put > 0:
                iv_val = implied_vol(spot, K, T, 0.0, price_put, "put")
            if iv_val:
                iv_points["Strike"].append(K)
                iv_points["IV"].append(iv_val * 100)
        if not iv_points["Strike"]:
            st.info("無法取得隱含波動率資料。")
        else:
            fig_skew = go.Figure()
            fig_skew.add_trace(go.Scatter(x=iv_points["Strike"], y=iv_points["IV"], mode="lines+markers", name="IV(%)"))
            # 計算 25Δ Risk Reversal
            call25_iv = None
            put25_iv = None
            for K, iv in zip(iv_points["Strike"], iv_points["IV"]):
                sigma = iv / 100.0
                if T > 0 and sigma > 0:
                    d1 = (log(spot/K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
                    delta_call = NormalDist(0, 1).cdf(d1)
                else:
                    delta_call = 0.0
                if call25_iv is None and delta_call <= 0.25:
                    call25_iv = iv
                if put25_iv is None and delta_call <= 0.75:
                    put25_iv = iv
            rr_text = ""
            if call25_iv is not None and put25_iv is not None:
                rr_val = call25_iv - put25_iv
                rr_text = f" (25Δ RR: {rr_val:.2f}%)"
            fig_skew.update_layout(title=f"{skew_code} IV Skew 曲線{rr_text}", xaxis_title="履約價", yaxis_title="隱含波動率(%)")
            st.plotly_chart(fig_skew, use_container_width=True)

    # 🆕 現貨/期貨/基差資料
    st.markdown("### 現貨/期貨/基差資料")
    fut_price = None
    basis_val = None
    foreign_net = None
    try:
        today_str = data_date
        url_fut = "https://www.taifex.com.tw/cht/3/futDailyMarketReport"
        headers = {"User-Agent": "Mozilla/5.0"}
        payload_fut = {"queryType": "2", "marketcode": "0", "commodity_id": "TX", "queryDate": today_str, "MarketCode": "0", "commodity_idt": "TX"}
        res_fut = requests.post(url_fut, data=payload_fut, headers=headers, timeout=5)
        text = res_fut.text
        front_code = (monthly["code"] if 'monthly' in locals() and monthly else nearest_code)
        if front_code:
            front_code = front_code[:6]  # 年月
            idx = text.find(front_code)
            if idx != -1:
                lines = text[idx:].splitlines()
                if len(lines) > 1:
                    data_line = lines[1]
                    parts = re.split(r"\s+", data_line.strip())
                    if len(parts) > 4:
                        try:
                            fut_price = float(parts[3]) if parts[3] != '-' else None
                        except:
                            fut_price = None
        if fut_price and final_taiex:
            basis_val = fut_price - final_taiex
    except Exception:
        fut_price = None
    # 抓取外資期貨淨部位
    try:
        url_inst = "https://www.taifex.com.tw/cht/3/futContractsDate"
        res_inst = requests.get(url_inst, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        text_inst = res_inst.text
        net_total = 0
        for code_num in ["1", "4", "5"]:
            idx = text_inst.find(f">{code_num}<")
            if idx != -1:
                segment = text_inst[idx:text_inst.find("</tr>", idx)]
                foreign_idx = segment.find("外資")
                if foreign_idx != -1:
                    sub_seg = segment[foreign_idx:]
                    cols = re.findall(r">(.*?)<", sub_seg)
                    cols = [c.strip() for c in cols if c.strip()]
                    if cols:
                        try:
                            net_val = int(cols[-1].replace(",", ""))
                        except:
                            net_val = 0
                        net_total += net_val
        foreign_net = net_total
    except Exception:
        foreign_net = None
    col_spot, col_fut, col_basis, col_foreign = st.columns(4)
    col_spot.metric("現貨指數", f"{int(final_taiex) if final_taiex else 'N/A'}")
    col_fut.metric("期貨價格", f"{int(fut_price) if fut_price else 'N/A'}")
    col_basis.metric("基差 (期貨-現貨)", f"{basis_val:+.0f}" if basis_val is not None else "N/A")
    col_foreign.metric("外資期貨淨部位", f"{foreign_net:+,} 口" if foreign_net is not None else "N/A")

    # 🆕 Dealer Gamma Exposure
    st.markdown("### Dealer Gamma Exposure")
    gamma_code = monthly["code"] if 'monthly' in locals() and monthly else nearest_code
    df_gamma = df[df["Month"] == gamma_code] if gamma_code else pd.DataFrame()
    if df_gamma.empty:
        st.info("無法計算 Gamma Exposure (缺少近月合約資料)。")
    else:
        gamma_settle_date = get_settlement_date(gamma_code)
        try:
            exp_date = datetime.strptime(gamma_settle_date, "%Y/%m/%d")
            days_to_exp_g = (exp_date - datetime.now(tz=TW_TZ)).days
        except:
            days_to_exp_g = 0
        T_g = max(days_to_exp_g, 0) / 252
        S0 = final_taiex if final_taiex else 0
        # 定義期權 Gamma 計算
        def bs_gamma(S, K, T, sigma):
            if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
                return 0.0
            d1 = (log(S/K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
            return NormalDist(0, 1).pdf(d1) / (S * sigma * sqrt(T))
        gamma_data = []
        for strike in sorted(df_gamma["Strike"].unique()):
            OI_call = df_gamma[(df_gamma["Strike"] == strike) & (df_gamma["Type"].str.contains("買|Call"))]["OI"].sum()
            OI_put = df_gamma[(df_gamma["Strike"] == strike) & (df_gamma["Type"].str.contains("賣|Put"))]["OI"].sum()
            if OI_call == 0 and OI_put == 0:
                continue
            # 取隱含波動率
            vol_call = None
            vol_put = None
            call_row = df_gamma[(df_gamma["Strike"] == strike) & (df_gamma["Type"].str.contains("買|Call"))]
            put_row = df_gamma[(df_gamma["Strike"] == strike) & (df_gamma["Type"].str.contains("賣|Put"))]
            if not call_row.empty and call_row["Price"].iloc[0] > 0:
                vol_call = implied_vol(S0, strike, T_g, 0.0, call_row["Price"].iloc[0], "call")
            if not put_row.empty and put_row["Price"].iloc[0] > 0:
                vol_put = implied_vol(S0, strike, T_g, 0.0, put_row["Price"].iloc[0], "put")
            sigma_use = vol_call if vol_call is not None else vol_put
            if sigma_use is None or sigma_use <= 0:
                continue
            gamma_val = bs_gamma(S0, strike, T_g, sigma_use)
            total_gamma = gamma_val * (OI_call + OI_put) * 50
            gamma_data.append((strike, total_gamma))
        if not gamma_data:
            st.info("無 Gamma 資料。")
        else:
            gamma_data.sort(key=lambda x: x[0])
            strikes_list = [x[0] for x in gamma_data]
            gamma_list = [x[1] for x in gamma_data]
            fig_gamma = go.Figure()
            fig_gamma.add_trace(go.Bar(x=strikes_list, y=gamma_list, name="Gamma Exposure"))
            fig_gamma.update_layout(title=f"{gamma_code} Dealer Gamma Exposure", xaxis_title="履約價", yaxis_title="Gamma 暴露值")
            st.plotly_chart(fig_gamma, use_container_width=True)

    # 🌟 雙 AI 分析區塊 🌟
    st.markdown("### 💡 雙 AI 莊家控盤室")

    if nearest_code and nearest_df is not None and not nearest_df.empty:
        st.caption(f"本次獵殺目標合約：**{nearest_code}**，結算日 **{nearest_date}**。")
        target_df_for_ai = nearest_df
        target_code = nearest_code
        target_date = nearest_date
    else:
        st.caption("⚠ 找不到合約，使用全市場資料。")
        target_df_for_ai = df
        target_code = "全市場"
        target_date = data_date

    if st.button("🚀 啟動莊家思維推演", type="primary"):
        ai_col1, ai_col2 = st.columns(2)

        # 注意：這裡傳入的是 final_taiex，確保 AI 吃到的是您校正後的價格
        with ai_col1:
            st.markdown(f"#### 💎 Gemini 莊家 ({gemini_model_name})")
            with st.spinner("Gemini 正在計算最大痛點..."):
                gemini_advice = ask_gemini_brief(target_df_for_ai, final_taiex, target_code, target_date)
            st.info(gemini_advice)

        with ai_col2:
            st.markdown(f"#### 💬 ChatGPT 莊家 ({openai_model_name})")
            with st.spinner("ChatGPT 正在擬定獵殺劇本..."):
                openai_advice = ask_openai_brief(target_df_for_ai, final_taiex, target_code, target_date)
            st.info(openai_advice)

    st.markdown("---")

    # ==========================================
    # 圖表 (圖表中的基準線也使用 final_taiex)
    # ==========================================
    if plot_targets:
        cols = st.columns(len(plot_targets))
        for i, target in enumerate(plot_targets):
            with cols[i]:
                m_code = target["info"]["code"]
                s_date = target["info"]["date"]
                df_target = df[df["Month"] == m_code]

                sub_call = df_target[df_target["Type"].str.contains("Call|買", case=False, na=False)]["Amount"].sum()
                sub_put = df_target[df_target["Type"].str.contains("Put|賣", case=False, na=False)]["Amount"].sum()
                sub_ratio = (sub_put / sub_call * 100) if sub_call > 0 else 0

                title_text = (
                    f"<b>{target['title']} {m_code}</b>"
                    f"<br><span style='font-size: 14px;'>結算: {s_date}</span>"
                    f"<br><span style='font-size: 14px;'>P/C金額比: {sub_ratio:.1f}% ({'偏多' if sub_ratio > 100 else '偏空'})</span>"
                )
                
                # 這裡傳入 final_taiex
                st.plotly_chart(plot_tornado_chart(df_target, title_text, final_taiex), use_container_width=True)
    else:
        st.info("目前無可識別的未來結算合約。")

if __name__ == "__main__":
    main()
