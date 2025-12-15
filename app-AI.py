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

# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="台指期籌碼戰情室 (精準控盤版)")
TW_TZ = timezone(timedelta(hours=8))

# ==========================================
# 🔑 金鑰設定區
# ==========================================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = "請輸入你的GEMINI_API_KEY"

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = "請輸入你的OPENAI_API_KEY"

# --- 模型設定 ---
def configure_gemini(api_key):
    if not api_key or "請輸入" in api_key: return None, "未設定 GEMINI Key"
    genai.configure(api_key=api_key)
    try:
        available_models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
        # 優先順序
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

gemini_model, gemini_model_name = configure_gemini(GEMINI_API_KEY)
openai_client, openai_model_name = configure_openai(OPENAI_API_KEY)

# ==========================================
# 🗓️ 結算日計算 (已修正為 F3/月選優先)
# ==========================================
# 這裡提供手動強制對應表，萬一程式算錯，您可以在這裡寫死
MANUAL_SETTLEMENT_FIX = {
    # 範例: "202512": "2025/12/17",
}

def get_settlement_date(contract_code: str) -> str:
    code = str(contract_code).strip().upper()
    
    # 1. 優先查表
    for key, fix_date in MANUAL_SETTLEMENT_FIX.items():
        if key == code: return fix_date

    try:
        if len(code) < 6: return "9999/99/99"
        year, month = int(code[:4]), int(code[4:6])
        c = calendar.monthcalendar(year, month)
        wednesdays = [w[calendar.WEDNESDAY] for w in c if w[calendar.WEDNESDAY] != 0]
        
        # 2. 自動判斷邏輯
        if "W" in code:
            # 週選邏輯 (W1, W2, W4...)
            match = re.search(r"W(\d)", code)
            week_num = int(match.group(1)) if match else 99
            day = wednesdays[week_num - 1] if len(wednesdays) >= week_num else None
        else:
            # 月選邏輯 (例如 202512) -> 通常是第 3 個週三
            # 這就是您提到的 F3 (Week 3)
            day = wednesdays[2] if len(wednesdays) >= 3 else None

        return f"{year}/{month:02d}/{day:02d}" if day else "9999/99/99"
    except:
        return "9999/99/99"

# --- 1. 現貨即時價 ---
@st.cache_data(ttl=60)
def get_realtime_data():
    ts = int(time.time())
    headers = {"User-Agent": "Mozilla/5.0"}
    # Yahoo
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII?interval=1d&range=1d&_={ts}"
        res = requests.get(url, headers=headers, timeout=5)
        meta = res.json()["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice") or meta.get("chartPreviousClose")
        if price: return float(price)
    except: pass
    # MIS
    try:
        url = f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_t00.tw&json=1&delay=0&_={ts}000"
        res = requests.get(url, timeout=3)
        d = res.json()
        if "msgArray" in d and d["msgArray"]:
            v = d["msgArray"][0].get("z", "-")
            if v == "-": v = d["msgArray"][0].get("o", "-")
            if v != "-": return float(v)
    except: pass
    return None

# --- 2. 選擇權籌碼 ---
@st.cache_data(ttl=300)
def get_option_data():
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    # 嘗試抓取最近 5 天 (避免假日無資料)
    for i in range(5):
        query_date = (datetime.now(tz=TW_TZ) - timedelta(days=i)).strftime("%Y/%m/%d")
        try:
            payload = {"queryType": "2", "marketCode": "0", "commodity_id": "TXO", "queryDate": query_date, "MarketCode": "0", "commodity_idt": "TXO"}
            res = requests.post(url, data=payload, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            if "查無資料" in res.text or len(res.text) < 500: continue
            
            df = pd.read_html(StringIO(res.text))[0]
            # 欄位正規化
            df.columns = [str(c).replace(" ", "").replace("*", "").replace("契約", "").strip() for c in df.columns]
            
            # 智慧欄位對應
            col_map = {}
            for c in df.columns:
                if "月" in c or "週" in c: col_map["Month"] = c
                elif "履約" in c: col_map["Strike"] = c
                elif "買賣" in c: col_map["Type"] = c
                elif "未沖銷" in c or "OI" in c: col_map["OI"] = c
                elif "結算" in c or "收盤" in c or "Price" in c: col_map["Price"] = c
            
            if len(col_map) < 5: continue
            df = df.rename(columns={v: k for k, v in col_map.items()})
            
            df = df[["Month", "Strike", "Type", "OI", "Price"]].dropna(subset=["Type"]).copy()
            df["Type"] = df["Type"].astype(str).str.strip()
            df["Strike"] = pd.to_numeric(df["Strike"].astype(str).str.replace(",", ""), errors="coerce")
            df["OI"] = pd.to_numeric(df["OI"].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
            df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",", "").replace("-", "0"), errors="coerce").fillna(0)
            df["Amount"] = df["OI"] * df["Price"] * 50
            
            if df["OI"].sum() > 0: return df, query_date
        except: continue
    return None, None

# --- 3. 三大法人籌碼 ---
@st.cache_data(ttl=3600)
def get_institutional_data(ref_date_str):
    url = "https://www.taifex.com.tw/cht/3/callsAndPutsDate"
    try_dates = [ref_date_str]
    dt_obj = datetime.strptime(ref_date_str, "%Y/%m/%d")
    try_dates.append((dt_obj - timedelta(days=1)).strftime("%Y/%m/%d"))
    
    for d in try_dates:
        try:
            payload = {"queryType": "1", "goDay": "", "doQuery": "1", "queryDate": d, "commodityId": "TXO"}
            res = requests.post(url, data=payload, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            if "查無資料" in res.text: continue
            
            df = pd.read_html(StringIO(res.text))[0]
            result = {"Date": d, "Foreign_Call_OI_Net": 0, "Foreign_Put_OI_Net": 0, "Dealer_Call_OI_Net": 0, "Dealer_Put_OI_Net": 0}
            
            for _, row in df.iterrows():
                vals = [int(str(x).replace(",", "")) for x in row.values if str(x).replace(",", "").replace("-","").isdigit()]
                if "外資" in str(row.values) and len(vals) >= 12:
                    result["Foreign_Call_OI_Net"] = vals[8]
                    result["Foreign_Put_OI_Net"] = vals[11]
                elif "自營商" in str(row.values) and len(vals) >= 12:
                    result["Dealer_Call_OI_Net"] = vals[8]
                    result["Dealer_Put_OI_Net"] = vals[11]
            return result
        except: continue
    return None

# --- Tornado Chart ---
def plot_tornado_chart(df_target, title_text, spot_price):
    is_call = df_target["Type"].str.contains("買|Call", case=False, na=False)
    df_call = df_target[is_call][["Strike", "OI", "Amount"]].rename(columns={"OI": "Call_OI", "Amount": "Call_Amt"})
    df_put = df_target[~is_call][["Strike", "OI", "Amount"]].rename(columns={"OI": "Put_OI", "Amount": "Put_Amt"})
    data = pd.merge(df_call, df_put, on="Strike", how="outer").fillna(0).sort_values("Strike")
    
    # Filter
    data = data[(data["Call_OI"] > 300) | (data["Put_OI"] > 300)]
    
    FOCUS = 800
    center = spot_price if spot_price and spot_price > 0 else (data.loc[data["Put_OI"].idxmax(), "Strike"] if not data.empty else 0)
    if center > 0:
        base = round(center / 50) * 50
        data = data[(data["Strike"] >= base - FOCUS) & (data["Strike"] <= base + FOCUS)]

    limit = max(data["Put_OI"].max(), data["Call_OI"].max(), 1000) * 1.1
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=data["Strike"], x=-data["Put_OI"], orientation="h", name="Put (支撐)", marker_color="#2ca02c", opacity=0.85, 
                         customdata=data["Put_Amt"]/1e8, hovertemplate="<b>%{y}</b><br>Put: %{x}<br>Amt: %{customdata:.2f}億"))
    fig.add_trace(go.Bar(y=data["Strike"], x=data["Call_OI"], orientation="h", name="Call (壓力)", marker_color="#d62728", opacity=0.85, 
                         customdata=data["Call_Amt"]/1e8, hovertemplate="<b>%{y}</b><br>Call: %{x}<br>Amt: %{customdata:.2f}億"))
    
    if spot_price and spot_price > 0:
        fig.add_hline(y=spot_price, line_dash="dash", line_color="#ff7f0e", line_width=2)
        fig.add_annotation(x=1, y=spot_price, text=f" {int(spot_price)} ", showarrow=False, bgcolor="#ff7f0e", font=dict(color="white"))

    fig.update_layout(title=dict(text=title_text, x=0.5, font=dict(size=18)), xaxis=dict(range=[-limit, limit]), 
                      yaxis=dict(dtick=50, tick0=0, tickformat="d"), barmode="overlay", height=700, margin=dict(l=50, r=50, t=80, b=50))
    return fig

# --- AI Prompt ---
def get_ai_prompt(contract_code, settlement_date, taiex_price, data_str, inst_data):
    inst_text = "目前無法人數據"
    if inst_data:
        inst_text = f"""
【三大法人 (Smart Money)】
* 外資: Call淨OI {inst_data['Foreign_Call_OI_Net']} / Put淨OI {inst_data['Foreign_Put_OI_Net']}
* 自營: Call淨OI {inst_data['Dealer_Call_OI_Net']} / Put淨OI {inst_data['Dealer_Put_OI_Net']}
(註: 若外資OI為正且市場賣壓大 -> 軋空機率高)
"""
    return f"""
你現在是台指選擇權市場的【頂級莊家】。
合約: {contract_code} (結算: {settlement_date}) | 基準價: {taiex_price}
{inst_text}

【任務】
根據 CSV 籌碼，判斷這週(F3/月選)的結算劇本：
1. **肥羊分析**：市場上最大量的 OI 是散戶的停損單，還是法人的鐵板？(結合法人數據判斷)
2. **劇本推演**：主力會如何「上下刷洗」來收割權利金？
3. **結算目標**：給出一個最痛的結算價位。

數據(前80大):
{data_str}
"""

def ask_ai(model_type, df_recent, taiex_price, contract_code, settlement_date, inst_data):
    try:
        df_ai = df_recent.nlargest(80, "Amount") if "Amount" in df_recent.columns else df_recent.copy()
        prompt = get_ai_prompt(contract_code, settlement_date, taiex_price, df_ai.to_csv(index=False), inst_data)
        if model_type == "gemini":
            if not gemini_model: return "⚠️ 未設定 Gemini"
            return gemini_model.generate_content(prompt).text
        elif model_type == "openai":
            if not openai_client: return "⚠️ 未設定 OpenAI"
            return openai_client.chat.completions.create(model=openai_model_name, messages=[{"role": "user", "content": prompt}]).choices[0].message.content
    except Exception as e: return f"忙碌中 ({str(e)})"

# --- 主程式 ---
def main():
    st.title("🦅 台指期籌碼戰情室 (精準控盤版)")
    if st.sidebar.button("🔄 重新整理"): st.cache_data.clear(); st.rerun()

    with st.spinner("抓取期交所即時籌碼..."):
        df, data_date = get_option_data()
        auto_taiex = get_realtime_data()
        inst_data = get_institutional_data(data_date) if data_date else None

    if df is None: st.error("查無資料"); return

    st.sidebar.download_button("📥 下載數據 CSV", df.to_csv(index=False).encode("utf-8-sig"), "option_data.csv", "text/csv")

    # ==========================================
    # 🎯 關鍵升級：合約選擇器 (解決抓錯合約的問題)
    # ==========================================
    st.sidebar.markdown("### 🎯 合約鎖定")
    # 找出所有合約代碼
    all_codes = sorted(df["Month"].unique())
    
    # 嘗試預設選取邏輯：優先找月選 (6碼數字) 且日期最近的
    default_idx = 0
    now_str = datetime.now().strftime("%Y%m")
    for i, code in enumerate(all_codes):
        # 如果是月選 (如 202512)
        if len(code) == 6 and code.isdigit():
             default_idx = i
             break
    
    # 讓使用者自己選！
    selected_code = st.sidebar.selectbox(
        "請選擇你要分析的合約 (月選通常為6碼數字)", 
        all_codes, 
        index=default_idx,
        help="如果系統自動判斷錯誤，請手動選擇正確的合約代碼 (例如本週三結算選 202512)"
    )
    
    # 鎖定合約
    target_df = df[df["Month"] == selected_code]
    target_date = get_settlement_date(selected_code)
    
    st.sidebar.info(f"目前鎖定：**{selected_code}**\n\n預估結算日：{target_date}")

    # ==========================================
    # 儀表板與手動校正
    # ==========================================
    with st.container(border=True):
        st.markdown("##### 🛠️ 控盤數據中心")
        c1, c2 = st.columns([1, 2])
        with c1: st.metric("📡 系統報價", f"{auto_taiex if auto_taiex else 'N/A'}")
        with c2: manual_input = st.number_input("🎹 手動校正點位 (輸入 > 0 即生效)", min_value=0.0, value=0.0, step=1.0, format="%.2f")

    final_taiex = manual_input if manual_input > 0 else (auto_taiex if auto_taiex else 0)

    # 法人看板
    if inst_data:
        st.markdown("### 🏦 法人籌碼結構")
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("外資 Call淨OI", f"{inst_data.get('Foreign_Call_OI_Net',0):,}")
        i2.metric("外資 Put淨OI", f"{inst_data.get('Foreign_Put_OI_Net',0):,}")
        i3.metric("自營 Call淨OI", f"{inst_data.get('Dealer_Call_OI_Net',0):,}")
        i4.metric("自營 Put淨OI", f"{inst_data.get('Dealer_Put_OI_Net',0):,}")
        st.caption(f"資料日期: {inst_data['Date']} | 用於判斷籌碼是「鐵板」還是「燃料」")
    
    st.markdown("---")

    # ==========================================
    # AI 分析區 (只針對選定的合約)
    # ==========================================
    st.subheader(f"💡 雙 AI 控盤推演：{selected_code}")
    if st.button("🚀 啟動莊家獵殺分析", type="primary"):
        c_ai1, c_ai2 = st.columns(2)
        with c_ai1:
            st.markdown(f"**Gemini ({gemini_model_name})**")
            with st.spinner("Gemini 運算中..."):
                st.info(ask_ai("gemini", target_df, final_taiex, selected_code, target_date, inst_data))
        with c_ai2:
            st.markdown(f"**ChatGPT ({openai_model_name})**")
            with st.spinner("ChatGPT 運算中..."):
                st.info(ask_ai("openai", target_df, final_taiex, selected_code, target_date, inst_data))

    st.markdown("---")
    
    # ==========================================
    # 圖表區 (只針對選定的合約)
    # ==========================================
    st.plotly_chart(plot_tornado_chart(
        target_df, 
        f"<b>【主力合約: {selected_code}】 結算日: {target_date}</b>", 
        final_taiex
    ), use_container_width=True)

if __name__ == "__main__":
    main()
