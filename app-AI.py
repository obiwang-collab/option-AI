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

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = "請輸入你的GEMINI_API_KEY"

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
        for target in ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]:
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
        _ = client.models.list()
        return client, "gpt-4o-mini"
    except Exception as e:
        return None, f"連線錯誤: {str(e)}"

# 初始化模型
gemini_model, gemini_model_name = configure_gemini(GEMINI_API_KEY)
openai_client, openai_model_name = configure_openai(OPENAI_API_KEY)

# 手動修正結算日
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
        wednesdays = [week[calendar.WEDNESDAY] for week in c if week[calendar.WEDNESDAY] != 0]
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

# --- 現貨即時價 ---
@st.cache_data(ttl=60)
def get_realtime_data():
    taiex = None
    ts = int(time.time())
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    }
    # 1. Yahoo Finance
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII?interval=1d&range=1d&_={ts}"
        res = requests.get(url, headers=headers, timeout=5)
        data = res.json()
        meta = data["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice") or meta.get("chartPreviousClose")
        if price:
            taiex = float(price)
    except Exception:
        pass

    # 2. TWSE MIS
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
        query_date = (datetime.now(tz=TW_TZ) - timedelta(days=i)).strftime("%Y/%m/%d")
        payload = {
            "queryType": "2", "marketCode": "0", "commodity_id": "TXO", "queryDate": query_date, "MarketCode": "0", "commodity_idt": "TXO"
        }
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=5)
            if "查無資料" in res.text or len(res.text) < 500:
                continue
            dfs = pd.read_html(StringIO(res.text))
            df = dfs[0]
            df.columns = [str(c).replace(" ", "").replace("*", "").replace("契約", "").strip() for c in df.columns]
            
            month_col = next((c for c in df.columns if "月" in c or "週" in c), None)
            strike_col = next((c for c in df.columns if "履約" in c), None)
            type_col = next((c for c in df.columns if "買賣" in c), None)
            oi_col = next((c for c in df.columns if "未沖銷" in c or "OI" in c), None)
            price_col = next((c for c in df.columns if "結算" in c or "收盤" in c or "Price" in c), None)

            if not all([month_col, strike_col, type_col, oi_col, price_col]):
                continue

            df = df.rename(columns={month_col: "Month", strike_col: "Strike", type_col: "Type", oi_col: "OI", price_col: "Price"})
            df = df[["Month", "Strike", "Type", "OI", "Price"]].dropna(subset=["Type"]).copy()
            
            df["Type"] = df["Type"].astype(str).str.strip()
            df["Strike"] = pd.to_numeric(df["Strike"].astype(str).str.replace(",", ""), errors="coerce")
            df["OI"] = pd.to_numeric(df["OI"].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
            df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",", "").replace("-", "0"), errors="coerce").fillna(0)
            df["Amount"] = df["OI"] * df["Price"] * 50

            if df["OI"].sum() == 0: continue
            return df, query_date
        except Exception:
            continue
    return None, None

# --- Tornado 圖 (核心外觀修改處) ---
def plot_tornado_chart(df_target, title_text, spot_price):
    is_call = df_target["Type"].str.contains("買|Call", case=False, na=False)
    df_call = df_target[is_call][["Strike", "OI", "Amount"]].rename(columns={"OI": "Call_OI", "Amount": "Call_Amt"})
    df_put = df_target[~is_call][["Strike", "OI", "Amount"]].rename(columns={"OI": "Put_OI", "Amount": "Put_Amt"})
    data = pd.merge(df_call, df_put, on="Strike", how="outer").fillna(0).sort_values("Strike")

    total_put_money = data["Put_Amt"].sum()
    total_call_money = data["Call_Amt"].sum()
    data = data[(data["Call_OI"] > 300) | (data["Put_OI"] > 300)]
    
    # 聚焦範圍 (略為縮小以確保 50 點刻度清楚)
    FOCUS_RANGE = 800  
    if spot_price and spot_price > 0:
        center_price = spot_price
    elif not data.empty:
        center_price = data.loc[data["Put_OI"].idxmax(), "Strike"]
    else:
        center_price = 0

    if center_price > 0:
        # 強制讓中心點對齊 50 的倍數，視覺更整齊
        base_center = round(center_price / 50) * 50
        min_s = base_center - FOCUS_RANGE
        max_s = base_center + FOCUS_RANGE
        data = data[(data["Strike"] >= min_s) & (data["Strike"] <= max_s)]

    max_oi = max(data["Put_OI"].max(), data["Call_OI"].max()) if not data.empty else 1000
    x_limit = max_oi * 1.1

    fig = go.Figure()
    # Put (Green)
    fig.add_trace(go.Bar(y=data["Strike"], x=-data["Put_OI"], orientation="h", name="Put (支撐)", marker_color="#2ca02c", opacity=0.85, 
                         customdata=data["Put_Amt"] / 100000000, 
                         hovertemplate="<b>%{y}</b><br>Put OI: %{x}<br>Amt: %{customdata:.2f}億<extra></extra>"))
    # Call (Red)
    fig.add_trace(go.Bar(y=data["Strike"], x=data["Call_OI"], orientation="h", name="Call (壓力)", marker_color="#d62728", opacity=0.85, 
                         customdata=data["Call_Amt"] / 100000000, 
                         hovertemplate="<b>%{y}</b><br>Call OI: %{x}<br>Amt: %{customdata:.2f}億<extra></extra>"))

    annotations = []
    # 繪製現貨線
    if spot_price and spot_price > 0 and not data.empty:
        if data["Strike"].min() <= spot_price <= data["Strike"].max():
            fig.add_hline(y=spot_price, line_dash="dash", line_color="#ff7f0e", line_width=2)
            annotations.append(dict(x=1, y=spot_price, xref="paper", yref="y", text=f" {int(spot_price)} ", 
                                    showarrow=False, xanchor="left", align="center", font=dict(color="white", size=12), 
                                    bgcolor="#ff7f0e", bordercolor="#ff7f0e", borderpad=2))

    annotations.append(dict(x=0.02, y=1.08, xref="paper", yref="paper", text=f"Put ${total_put_money/100000000:.1f}億", 
                            showarrow=False, align="left", font=dict(size=14, color="#2ca02c")))
    annotations.append(dict(x=0.98, y=1.08, xref="paper", yref="paper", text=f"Call ${total_call_money/100000000:.1f}億", 
                            showarrow=False, align="right", font=dict(size=14, color="#d62728")))

    fig.update_layout(
        title=dict(text=title_text, y=0.98, x=0.5, xanchor="center", yanchor="top", font=dict(size=18)),
        xaxis=dict(title="OI", range=[-x_limit, x_limit], showgrid=True, zeroline=True, zerolinecolor="black"),
        yaxis=dict(
            title="履約價",
            # =========================================
            # 🔥 關鍵修改：強制 Y 軸每 50 點一格 🔥
            # =========================================
            tickmode="linear",
            dtick=50,  # 強制間距 50
            tick0=0,   # 從 0 開始起算 (確保是 50 的倍數)
            tickformat="d"
        ),
        barmode="overlay",
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
        height=750,
        margin=dict(l=50, r=50, t=100, b=50),
        annotations=annotations,
        paper_bgcolor="white", plot_bgcolor="white"
    )
    return fig

# --- AI 分析函式 (Prompt 保持您要的莊家風格) ---
def get_ai_prompt(contract_code, settlement_date, taiex_price, data_str):
    return f"""
你現在是台指選擇權市場的【主力莊家】。目標：**在結算日吃掉最多權利金**。

【市場現況】
- 結算合約：{contract_code} (結算: {settlement_date})
- **目前控盤基準點位：{taiex_price}** (此為最新確認價格)

【任務】
根據 CSV 籌碼 (OI/Amount) 進行控盤推演：
1. **肥羊與雷區**：散戶重倉區(肥羊)與你的防守底線(雷區)。
2. **操盤劇本**：未來幾小時或幾天的畫線劇本(誘多/養空/軋空)。若卡在區間，如何雙巴收租？
3. **最佳結算目標**：給出一個讓 Call/Put 雙殺的具體點位或區間。
4. **莊家指令**：簡短指令 (e.g. Sell Call @ 23500)。

【要求】
- 第一人稱(本莊)。語氣：**自信、冷血、貪婪**。
- 嚴禁風險警語。
- 字數 300-500。

數據：
{data_str}
"""

def ask_gemini_brief(df_recent, taiex_price, contract_code, settlement_date):
    if not gemini_model: return f"⚠️ {gemini_model_name}"
    try:
        df_ai = df_recent.nlargest(80, "Amount") if "Amount" in df_recent.columns else df_recent.copy()
        prompt = get_ai_prompt(contract_code, settlement_date, taiex_price, df_ai.to_csv(index=False))
        return gemini_model.generate_content(prompt).text
    except Exception as e: return f"忙碌中 ({str(e)})"

def ask_openai_brief(df_recent, taiex_price, contract_code, settlement_date):
    if not openai_client: return f"⚠️ {openai_model_name}"
    try:
        df_ai = df_recent.nlargest(80, "Amount") if "Amount" in df_recent.columns else df_recent.copy()
        prompt = get_ai_prompt(contract_code, settlement_date, taiex_price, df_ai.to_csv(index=False))
        return openai_client.chat.completions.create(
            model=openai_model_name,
            messages=[
                {"role": "system", "content": "你是一個冷血期貨莊家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7, max_tokens=800
        ).choices[0].message.content
    except Exception as e: return f"忙碌中 ({str(e)})"

# --- 主程式 ---
def main():
    st.title("🤖 台指期籌碼戰情室 (莊家獵殺版)")
    if st.sidebar.button("🔄 重新整理"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("連線期交所中..."):
        df, data_date = get_option_data()
        auto_taiex = get_realtime_data()

    if df is None:
        st.error("查無資料，請稍後再試。")
        return

    st.sidebar.download_button("📥 下載數據", df.to_csv(index=False).encode("utf-8-sig"), f"option_{data_date.replace('/','')}.csv", "text/csv")

    # ==========================================
    # 🛠️ 數據校正區 (UI 優化版)
    # ==========================================
    # 使用 Container 把這個區塊框起來，更有儀表板的感覺
    with st.container(border=True):
        st.markdown("##### 🛠️ 報價校正中心")
        col_u1, col_u2 = st.columns([1, 2])
        
        with col_u1:
            # 顯示系統自動抓取的值，作為參考
            st.metric("📡 系統自動抓取", f"{auto_taiex if auto_taiex else 'N/A'}", help="來自 Yahoo Finance 或證交所的延遲報價")
            
        with col_u2:
            # 手動輸入框：Step=1 (現貨是連續的)，但視覺上暗示這是為了對齊選擇權
            manual_price_input = st.number_input(
                "🎹 手動輸入現貨/期貨點位 (若輸入 > 0，AI 將以此為準)",
                min_value=0.0,
                value=0.0,
                step=1.0, 
                format="%.2f",
                help="因為選擇權履約價為每 50 點一檔，請輸入您在看盤軟體上看到的精確點位，讓 AI 判斷目前價格位於哪兩個履約價之間。"
            )

    # 決定最終使用的價格
    if manual_price_input > 0:
        final_taiex = manual_price_input
        status_color = "orange" # 手動模式用橘色
        status_text = "手動校正模式"
    else:
        final_taiex = auto_taiex if auto_taiex else 0
        status_color = "green" # 自動模式用綠色
        status_text = "系統自動模式"

    # 顯示 P/C Ratio 區塊
    total_call_amt = df[df["Type"].str.contains("買|Call", case=False, na=False)]["Amount"].sum()
    total_put_amt = df[df["Type"].str.contains("賣|Put", case=False, na=False)]["Amount"].sum()
    pc_ratio_amt = ((total_put_amt / total_call_amt) * 100 if total_call_amt > 0 else 0)

    st.markdown("---")
    m1, m2, m3 = st.columns([1, 1, 1])
    # 這裡顯示最終判定價格，並標註來源
    m1.markdown(f"**📊 分析基準價格**")
    m1.markdown(f"<h2 style='color: {status_color}; margin:0;'>{final_taiex:.2f}</h2>", unsafe_allow_html=True)
    m1.caption(f"目前狀態：{status_text}")

    trend = "偏多" if pc_ratio_amt > 100 else "偏空"
    m2.metric("P/C 金額比 (大盤氣氛)", f"{pc_ratio_amt:.1f}%", f"{trend}")
    m3.metric("資料日期", data_date)
    st.markdown("---")

    # ==========================================
    # 合約邏輯
    # ==========================================
    nearest_code = None
    nearest_date = None
    nearest_df = None
    plot_targets = []
    
    unique_codes = df["Month"].unique()
    all_contracts = []
    for code in unique_codes:
        s_date = get_settlement_date(code)
        if s_date > data_date: all_contracts.append({"code": code, "date": s_date})
    all_contracts.sort(key=lambda x: x["date"])

    if all_contracts:
        nearest = all_contracts[0]
        nearest_code = nearest["code"]
        nearest_date = nearest["date"]
        nearest_df = df[df["Month"] == nearest_code]
        plot_targets.append({"title": "最近結算", "info": nearest})
        
        monthly = next((c for c in all_contracts if len(c["code"]) == 6), None)
        if monthly and monthly["code"] != nearest_code:
            plot_targets.append({"title": "當月月選", "info": monthly})
        elif monthly:
            plot_targets[0]["title"] = "最近結算 (同月選)"

    # ==========================================
    # AI 分析區
    # ==========================================
    st.subheader("💡 雙 AI 莊家控盤室")
    target_df = nearest_df if nearest_code else df
    t_code = nearest_code if nearest_code else "全市場"
    t_date = nearest_date if nearest_date else data_date

    if st.button("🚀 啟動莊家思維推演 (使用上方基準價格)", type="primary"):
        c_ai1, c_ai2 = st.columns(2)
        with c_ai1:
            st.markdown(f"**Gemini ({gemini_model_name})**")
            with st.spinner("Gemini 思考中..."):
                st.info(ask_gemini_brief(target_df, final_taiex, t_code, t_date))
        with c_ai2:
            st.markdown(f"**ChatGPT ({openai_model_name})**")
            with st.spinner("ChatGPT 思考中..."):
                st.info(ask_openai_brief(target_df, final_taiex, t_code, t_date))
    
    st.markdown("---")

    # ==========================================
    # 圖表區 (已套用 dtick=50)
    # ==========================================
    if plot_targets:
        cols = st.columns(len(plot_targets))
        for i, target in enumerate(plot_targets):
            with cols[i]:
                d_t = df[df["Month"] == target["info"]["code"]]
                # 計算該合約的 P/C Ratio
                s_c = d_t[d_t["Type"].str.contains("Call|買")]["Amount"].sum()
                s_p = d_t[d_t["Type"].str.contains("Put|賣")]["Amount"].sum()
                ratio = (s_p/s_c*100) if s_c > 0 else 0
                
                title = f"<b>【{target['title']}】 {target['info']['code']}</b><br><span style='font-size:14px'>結算: {target['info']['date']} | P/C: {ratio:.1f}%</span>"
                # 傳入 final_taiex 畫圖
                st.plotly_chart(plot_tornado_chart(d_t, title, final_taiex), use_container_width=True)
    else:
        st.info("無合約資料")

if __name__ == "__main__":
    main()
