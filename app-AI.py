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
st.set_page_config(layout="wide", page_title="台指期籌碼戰情室 (雙 AI 決策版)")
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
        # 使用 gpt-3.5-turbo 作為預設模型名稱
        return client, "gpt-3.5-turbo"
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
    """
    將期權月份/週別代碼轉換成實際結算日期字串 (YYYY/MM/DD)。
    支援一般月選、W 週選、F 週選。
    """
    code = str(contract_code).strip().upper()

    # 先看是否有手動覆寫
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

        # 週選結算規則
        if "W" in code:
            match = re.search(r"W(\d)", code)
            if match:
                week_num = int(match.group(1))
                if len(wednesdays) >= week_num:
                    day = wednesdays[week_num - 1]

        # F 可能代表週五選
        elif "F" in code:
            match = re.search(r"F(\d)", code)
            if match:
                week_num = int(match.group(1))
                if len(fridays) >= week_num:
                    day = fridays[week_num - 1]

        # 一般月選：第三個星期三
        else:
            if len(wednesdays) >= 3:
                day = wednesdays[2]

        if day:
            return f"{year}/{month:02d}/{day:02d}"
        else:
            return "9999/99/99"
    except Exception:
        return "9999/99/99"


# --- [修正] 現貨即時價 ---
@st.cache_data(ttl=60)
def get_realtime_data():
    """
    修正版：使用更完整的 Headers 避免被擋，並優化 Yahoo Finance 解析邏輯
    """
    taiex = None
    ts = int(time.time())
    
    # 偽裝成一般瀏覽器的 Headers (重要修正)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive"
    }

    # 1) 優先嘗試 Yahoo Finance (通常比 MIS 穩定且即時)
    try:
        # 使用 v8 chart API
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII?interval=1d&range=1d&_={ts}"
        res = requests.get(url, headers=headers, timeout=5)
        data = res.json()
        
        # 解析資料
        meta = data["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice")
        
        # 如果是收盤後，有些欄位可能會變，嘗試讀取 closing price
        if price is None:
            price = meta.get("chartPreviousClose")
            
        if price:
            taiex = float(price)
            # print(f"Yahoo Data Success: {taiex}") # Debug用
    except Exception as e:
        # print(f"Yahoo Failed: {e}")
        pass

    # 2) 如果 Yahoo 失敗，嘗試證交所 MIS
    if taiex is None:
        try:
            url = f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_t00.tw&json=1&delay=0&_={ts}000"
            res = requests.get(url, timeout=3) # 這裡通常不需要太複雜的 header，但有時會慢
            data = res.json()
            if "msgArray" in data and len(data["msgArray"]) > 0:
                val = data["msgArray"][0].get("z", "-") # 最近成交
                if val == "-":
                    val = data["msgArray"][0].get("o", "-") # 開盤
                if val == "-":
                     val = data["msgArray"][0].get("y", "-") # 昨收
                
                if val != "-":
                    taiex = float(val)
        except Exception:
            pass

    return taiex


# --- 期交所選擇權資料 ---
@st.cache_data(ttl=300)
def get_option_data():
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    headers = {"User-Agent": "Mozilla/5.0"}

    # 往回最多 5 天尋找有資料的一天
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

            month_col = next((c for c in df.columns if "月" in c or "週" in c), None)
            strike_col = next((c for c in df.columns if "履約" in c), None)
            type_col = next((c for c in df.columns if "買賣" in c), None)
            oi_col = next(
                (c for c in df.columns if "未沖銷" in c or "OI" in c), None
            )
            price_col = next(
                (c for c in df.columns if "結算" in c or "收盤" in c or "Price" in c),
                None,
            )
            vol_col = next(
                (c for c in df.columns if "成交量" in c or "Volume" in c), None
            )

            if not all([month_col, strike_col, type_col, oi_col, price_col]):
                continue

            rename_dict = {
                month_col: "Month",
                strike_col: "Strike",
                type_col: "Type",
                oi_col: "OI",
                price_col: "Price",
            }
            if vol_col:
                rename_dict[vol_col] = "Volume"

            df = df.rename(columns=rename_dict)

            cols_to_keep = ["Month", "Strike", "Type", "OI", "Price"]
            if "Volume" in df.columns:
                cols_to_keep.append("Volume")
            df = df[cols_to_keep].copy()

            df = df.dropna(subset=["Type"])
            df["Type"] = df["Type"].astype(str).str.strip()
            df["Strike"] = pd.to_numeric(
                df["Strike"].astype(str).str.replace(",", ""), errors="coerce"
            )
            df["OI"] = (
                pd.to_numeric(
                    df["OI"].astype(str).str.replace(",", ""), errors="coerce"
                )
                .fillna(0)
            )
            df["Price"] = (
                df["Price"].astype(str).str.replace(",", "").replace("-", "0")
            )
            df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)

            if "Volume" in df.columns:
                df["Volume"] = (
                    pd.to_numeric(
                        df["Volume"].astype(str).str.replace(",", ""),
                        errors="coerce",
                    )
                    .fillna(0)
                )

            # 金額：OI * 價格 * 50
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

    # 過濾掉 OI 太小的履約價
    data = data[(data["Call_OI"] > 300) | (data["Put_OI"] > 300)]

    # 聚焦區間
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

    max_oi = (
        max(data["Put_OI"].max(), data["Call_OI"].max()) if not data.empty else 1000
    )
    x_limit = max_oi * 1.1

    fig = go.Figure()

    # Put
    fig.add_trace(
        go.Bar(
            y=data["Strike"],
            x=-data["Put_OI"],
            orientation="h",
            name="Put (支撐)",
            marker_color="#2ca02c",
            opacity=0.85,
            customdata=data["Put_Amt"] / 100000000,
            hovertemplate="<b>履約價: %{y}</b><br>Put OI: %{x} 口<br>Put 市值: %{customdata:.2f}億<extra></extra>",
        )
    )

    # Call
    fig.add_trace(
        go.Bar(
            y=data["Strike"],
            x=data["Call_OI"],
            orientation="h",
            name="Call (壓力)",
            marker_color="#d62728",
            opacity=0.85,
            customdata=data["Call_Amt"] / 100000000,
            hovertemplate="<b>履約價: %{y}</b><br>Call OI: %{x} 口<br>Call 市值: %{customdata:.2f}億<extra></extra>",
        )
    )

    annotations = []

    # 現貨價線
    if spot_price and spot_price > 0 and not data.empty:
        if data["Strike"].min() <= spot_price <= data["Strike"].max():
            fig.add_hline(
                y=spot_price,
                line_dash="dash",
                line_color="#ff7f0e",
                line_width=2,
            )
            annotations.append(
                dict(
                    x=1,
                    y=spot_price,
                    xref="paper",
                    yref="y",
                    text=f" 現貨 {int(spot_price)} ",
                    showarrow=False,
                    xanchor="left",
                    align="center",
                    font=dict(color="white", size=12),
                    bgcolor="#ff7f0e",
                    bordercolor="#ff7f0e",
                    borderpad=4,
                )
            )

    # Put / Call 總金額說明
    annotations.append(
        dict(
            x=0.02,
            y=1.05,
            xref="paper",
            yref="paper",
            text=f"<b>Put 總金額</b><br>{total_put_money/100000000:.1f} 億",
            showarrow=False,
            align="left",
            font=dict(size=14, color="#2ca02c"),
            bgcolor="white",
            bordercolor="#2ca02c",
            borderwidth=2,
            borderpad=6,
        )
    )
    annotations.append(
        dict(
            x=0.98,
            y=1.05,
            xref="paper",
            yref="paper",
            text=f"<b>Call 總金額</b><br>{total_call_money/100000000:.1f} 億",
            showarrow=False,
            align="right",
            font=dict(size=14, color="#d62728"),
            bgcolor="white",
            bordercolor="#d62728",
            borderwidth=2,
            borderpad=6,
        )
    )

    fig.update_layout(
        title=dict(
            text=title_text,
            y=0.95,
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(size=20, color="black"),
        ),
        xaxis=dict(
            title="未平倉量 (OI)",
            range=[-x_limit, x_limit],
            showgrid=True,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="black",
            tickmode="array",
            tickvals=[
                -x_limit * 0.75,
                -x_limit * 0.5,
                -x_limit * 0.25,
                0,
                x_limit * 0.25,
                x_limit * 0.5,
                x_limit * 0.75,
            ],
            ticktext=[
                f"{int(x_limit*0.75)}",
                f"{int(x_limit*0.5)}",
                f"{int(x_limit*0.25)}",
                "0",
                f"{int(x_limit*0.25)}",
                f"{int(x_limit*0.5)}",
                f"{int(x_limit*0.75)}",
            ],
        ),
        yaxis=dict(title="履約價", tickmode="linear", dtick=100, tickformat="d"),
        barmode="overlay",
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
        height=750,
        margin=dict(l=40, r=80, t=140, b=60),
        annotations=annotations,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig


# --- AI 分析函式 (Gemini) ---
def ask_gemini_brief(df_recent, taiex_price, contract_code, settlement_date):
    """
    只針對「距離現在最近的結算合約」的 df_recent 做分析。
    """
    if not gemini_model:
        return f"⚠️ {gemini_model_name}"

    try:
        df_ai = df_recent.copy()
        if "Amount" in df_ai.columns:
            df_ai = df_ai.nlargest(80, "Amount")

        data_str = df_ai.to_csv(index=False)

        prompt = f"""
你現在是台指選擇權市場的主要控盤者（莊家），目標是在「最近即將結算」的合約上，讓自己的選擇權部位在結算時獲得最大利益、減少賠付。

分析標的：
- 合約代碼：{contract_code}
- 結算日：{settlement_date}
- 現貨指數：約 {taiex_price}

CSV 欄位包含：Month, Strike, Type, OI, Amount, Price。

請依下列邏輯思考：
1. 只根據本合約的資料，觀察各履約價 Call / Put 金額 (Amount) 集中區。
2. 推估你最希望指數結算落在哪一兩個價位區間，能讓整體賠付最小。
3. 給出具體結算目標區間，例如「28400–28600」，必要時可再補一個備用結算區間。
4. 說明控盤方法：例如「先壓在區間下緣洗籌，再拉回區間中上緣結算」或「維持在區間內來回震盪」。
5. 最後用一句話，給一般交易者簡單的操作建議（偏多、偏空或區間操作）。

請嚴格使用以下輸出格式（繁體中文，且務必包含具體數字區間）：
- 結算目標區間：xxxx–xxxx（如有第二個備用區間可寫「或 xxxx–xxxx」）
- 控盤策略：一句話簡要說明你會如何控盤
- 交易建議：一句話給一般交易者的具體操作建議

限制：
- 不要出現任何風險聲明、心理安慰或教化語句。
- 不要解釋你是誰，也不要解釋計算過程。
- 內容以實務控盤與價位區間為主，總字數控制在 150 字以內。

以下是本次分析用的選擇權籌碼 CSV 資料：
{data_str}
"""

        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"分析忙碌中 ({str(e)})"


# --- AI 分析函式 (ChatGPT / OpenAI) ---
def ask_openai_brief(df_recent, taiex_price, contract_code, settlement_date):
    """
    只針對「最近即將結算」的 df_recent 做分析。
    """
    if not openai_client:
        return f"⚠️ {openai_model_name}"

    try:
        df_ai = df_recent.copy()
        if "Amount" in df_ai.columns:
            df_ai = df_ai.nlargest(80, "Amount")

        data_str = df_ai.to_csv(index=False)

        user_prompt = f"""
你現在是台指選擇權市場的主要控盤者（莊家），目標是在「最近即將結算」的合約上，讓自己的選擇權部位在結算時獲得最大利益、減少賠付。

分析標的：
- 合約代碼：{contract_code}
- 結算日：{settlement_date}
- 現貨指數：約 {taiex_price}

CSV 欄位包含：Month, Strike, Type, OI, Amount, Price。

請依下列邏輯思考：
1. 只根據本合約的資料，觀察各履約價 Call / Put 金額 (Amount) 集中區。
2. 推估你最希望指數結算落在哪一兩個價位區間，能讓整體賠付最小。
3. 給出具體結算目標區間，例如「28400–28600」，必要時可再補一個備用結算區間。
4. 說明控盤方法：例如「先壓在區間下緣洗籌，再拉回區間中上緣結算」或「維持在區間內來回震盪」。
5. 最後用一句話，給一般交易者簡單的操作建議（偏多、偏空或區間操作）。

請嚴格使用以下輸出格式（繁體中文，且務必包含具體數字區間）：
- 結算目標區間：xxxx–xxxx（如有第二個備用區間可寫「或 xxxx–xxxx」）
- 控盤策略：一句話簡要說明你會如何控盤
- 交易建議：一句話給一般交易者的具體操作建議

限制：
- 不要出現任何風險聲明、心理安慰或教化語句。
- 不要解釋你是誰，也不要解釋計算過程。
- 內容以實務控盤與價位區間為主，總字數控制在 150 字以內。

以下是本次分析用的選擇權籌碼 CSV 資料：
{data_str}
"""

        response = openai_client.chat.completions.create(
            model=openai_model_name,
            messages=[
                {
                    "role": "system",
                    "content": "你是一位只重視損益與價位區間的台指期選擇權莊家，不說教、不安撫情緒，只給實務控盤價位與簡短建議。",
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"分析忙碌中 ({str(e)})"


# --- 主程式 ---
def main():
    st.title("🤖 台指期籌碼戰情室 (雙 AI 決策版)")

    # [修正] 側邊欄重新整理：加入 key 參數避免 DuplicateElementId 錯誤
    if st.sidebar.button("🔄 重新整理", key="refresh_main"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("連線期交所中..."):
        df, data_date = get_option_data()
        taiex_now = get_realtime_data()

    if df is None:
        st.error("查無資料，請稍後再試。")
        return

    # 下載完整 CSV
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.sidebar.download_button(
        "📥 下載完整數據",
        csv,
        f"option_{data_date.replace('/','')}.csv",
        "text/csv",
    )

    # 市場整體 P/C 金額比
    total_call_amt = df[df["Type"].str.contains("買|Call", case=False, na=False)][
        "Amount"
    ].sum()
    total_put_amt = df[df["Type"].str.contains("賣|Put", case=False, na=False)][
        "Amount"
    ].sum()
    pc_ratio_amt = (
        (total_put_amt / total_call_amt) * 100 if total_call_amt > 0 else 0
    )

    # 上方摘要資訊
    c1, c2, c3, c4 = st.columns([1.2, 0.8, 1, 1])
    c1.markdown(
        "<div style='text-align: left;'><span style='font-size: 14px; color: #555;'>製圖時間</span><br>"
        f"<span style='font-size: 18px; font-weight: bold;'>{datetime.now(tz=TW_TZ).strftime('%Y/%m/%d %H:%M:%S')}</span></div>",
        unsafe_allow_html=True,
    )
    c2.metric("大盤現貨", f"{int(taiex_now) if taiex_now else 'N/A'}")
    trend = "偏多" if pc_ratio_amt > 100 else "偏空"
    c3.metric(
        "全市場 P/C 金額比",
        f"{pc_ratio_amt:.1f}%",
        f"{trend}格局",
        delta_color="normal" if pc_ratio_amt > 100 else "inverse",
    )
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
        # 結算日需晚於資料日期，才算未結算
        if s_date_str == "9999/99/99" or s_date_str <= data_date:
            continue
        all_contracts.append({"code": code, "date": s_date_str})

    # 依結算日排序，最前面就是「最近即將結算」的合約
    all_contracts.sort(key=lambda x: x["date"])

    if all_contracts:
        nearest = all_contracts[0]
        nearest_code = nearest["code"]
        nearest_date = nearest["date"]
        nearest_df = df[df["Month"] == nearest_code]

        plot_targets.append({"title": "最近結算", "info": nearest})

        # 再找同一個月份的「月選」合約（例如純 202512）
        monthly = next((c for c in all_contracts if len(c["code"]) == 6), None)
        if monthly:
            if monthly["code"] != nearest_code:
                plot_targets.append({"title": "當月月選", "info": monthly})
            else:
                plot_targets[0]["title"] = "最近結算 (同月選)"

    # ==========================================
    # 🌟 雙 AI 分析區塊 🌟 — 僅用「最近結算合約」做分析
    # ==========================================
    st.markdown("### 💡 雙 AI 控盤錦囊 (距離結算最近的合約)")

    if nearest_code and nearest_df is not None and not nearest_df.empty:
        st.caption(
            f"本次 AI 分析標的：**{nearest_code}**，結算日 **{nearest_date}**，以此合約的選擇權籌碼為主。"
        )
        target_df_for_ai = nearest_df
        target_code = nearest_code
        target_date = nearest_date
    else:
        st.caption(
            "⚠ 找不到未來結算的合約，暫以全市場資料作為 AI 分析依據（不建議長期如此使用）。"
        )
        target_df_for_ai = df
        target_code = "全市場"
        target_date = data_date

    if st.button("🚀 啟動雙 AI 策略分析", type="primary"):
        ai_col1, ai_col2 = st.columns(2)

        # --- Gemini 分析 (左欄) ---
        with ai_col1:
            st.markdown(f"#### 💎 Gemini 控盤建議 ({gemini_model_name})")
            with st.spinner("Gemini 正在以莊家視角擬定策略..."):
                gemini_advice = ask_gemini_brief(
                    target_df_for_ai, taiex_now, target_code, target_date
                )
            st.info(gemini_advice)

        # --- ChatGPT 分析 (右欄) ---
        with ai_col2:
            st.markdown(f"#### 💬 ChatGPT 控盤建議 ({openai_model_name})")
            with st.spinner("ChatGPT 正在以莊家視角擬定策略..."):
                openai_advice = ask_openai_brief(
                    target_df_for_ai, taiex_now, target_code, target_date
                )
            st.info(openai_advice)

    st.markdown("---")

    # ==========================================
    # 圖表：最近結算 + 月選
    # ==========================================
    if plot_targets:
        cols = st.columns(len(plot_targets))
        for i, target in enumerate(plot_targets):
            with cols[i]:
                m_code = target["info"]["code"]
                s_date = target["info"]["date"]
                df_target = df[df["Month"] == m_code]

                sub_call = df_target[
                    df_target["Type"].str.contains("Call|買", case=False, na=False)
                ]["Amount"].sum()
                sub_put = df_target[
                    df_target["Type"].str.contains("Put|賣", case=False, na=False)
                ]["Amount"].sum()
                sub_ratio = (sub_put / sub_call * 100) if sub_call > 0 else 0

                title_text = (
                    f"<b>【{target['title']}】 {m_code}</b>"
                    f"<br><span style='font-size: 14px;'>結算: {s_date}</span>"
                    f"<br><span style='font-size: 14px;'>P/C金額比: {sub_ratio:.1f}% "
                    f"({'偏多' if sub_ratio > 100 else '偏空'})</span>"
                )

                st.plotly_chart(
                    plot_tornado_chart(df_target, title_text, taiex_now),
                    use_container_width=True,
                )
    else:
        st.info("目前無可識別的未來結算合約，僅能檢視整體 OI 分布。")


if __name__ == "__main__":
    main()
