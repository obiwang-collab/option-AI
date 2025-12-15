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

# --- 智慧模型設定 ---
def configure_gemini(api_key):
    if not api_key or "請輸入" in api_key:
        return None, "尚未設定 GEMINI Key"
    genai.configure(api_key=api_key)
    try:
        available_models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
        for target in ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-pro"]:
            for m in available_models:
                if target in m: return genai.GenerativeModel(m), m
        return (genai.GenerativeModel(available_models[0]), available_models[0]) if available_models else (None, "無可用模型")
    except Exception as e:
        return None, f"連線錯誤: {str(e)}"

def configure_openai(api_key):
    if not api_key or "請輸入" in api_key:
        return None, "尚未設定 OPENAI Key"
    try:
        client = OpenAI(api_key=api_key)
        _ = client.models.list()
        return client, "gpt-4o-mini"
    except Exception as e:
        return None, f"連線錯誤: {str(e)}"

gemini_model, gemini_model_name = configure_gemini(GEMINI_API_KEY)
openai_client, openai_model_name = configure_openai(OPENAI_API_KEY)

# --- 輔助函式 ---
MANUAL_SETTLEMENT_FIX = {"202501W1": "2025/01/02"}

def get_settlement_date(contract_code: str) -> str:
    code = str(contract_code).strip().upper()
    for key, fix_date in MANUAL_SETTLEMENT_FIX.items():
        if key in code: return fix_date
    try:
        if len(code) < 6: return "9999/99/99"
        year, month = int(code[:4]), int(code[4:6])
        c = calendar.monthcalendar(year, month)
        wednesdays = [week[calendar.WEDNESDAY] for week in c if week[calendar.WEDNESDAY] != 0]
        if "W" in code:
            match = re.search(r"W(\d)", code)
            day = wednesdays[int(match.group(1)) - 1] if match and len(wednesdays) >= int(match.group(1)) else None
        else:
            day = wednesdays[2] if len(wednesdays) >= 3 else None
        return f"{year}/{month:02d}/{day:02d}" if day else "9999/99/99"
    except: return "9999/99/99"

# --- 現貨即時價 ---
@st.cache_data(ttl=60)
def get_realtime_data():
    taiex = None
    ts = int(time.time())
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII?interval=1d&range=1d&_={ts}"
        res = requests.get(url, headers=headers, timeout=5).json()
        meta = res["chart"]["result"][0]["meta"]
        taiex = float(meta.get("regularMarketPrice") or meta.get("chartPreviousClose"))
    except:
        pass
    return taiex

# --- 新增：期貨法人與大戶籌碼 (替代玩股網爬蟲) ---
# 說明：因為玩股網擋爬蟲，我們直接去源頭(期交所)抓，並增加「日期回溯」確保抓到最近一次有效數據
@st.cache_data(ttl=3600)
def get_future_chips():
    chips_data = {
        "Date": "",
        "Foreign_Net_OI": 0, # 外資淨未平倉
        "Top5_Net_OI": 0,    # 前五大淨未平倉
        "Top10_Net_OI": 0,   # 前十大淨未平倉
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # 1. 抓取外資期貨 (嘗試最近 5 天，因為假日沒資料)
    found_date = None
    for i in range(5):
        q_date = datetime.now(tz=TW_TZ) - timedelta(days=i)
        
        # 如果是今天下午 3 點前，直接跳過今天(因為期交所還沒出報告)
        if i == 0 and datetime.now(tz=TW_TZ).hour < 15:
            continue
            
        q_date_str = q_date.strftime("%Y/%m/%d")
        
        try:
            # A. 三大法人
            url = "https://www.taifex.com.tw/cht/3/futContractsDate"
            payload = {"queryType": "1", "queryDate": q_date_str, "commodityId": "TXF"}
            res = requests.post(url, data=payload, headers=headers, timeout=5)
            
            if "查無資料" not in res.text and len(res.text) > 500:
                dfs = pd.read_html(StringIO(res.text))
                df_inst = dfs[0]
                # 尋找外資列
                row_foreign = df_inst[df_inst.iloc[:, 0].astype(str).str.contains("外資", na=False)]
                if not row_foreign.empty:
                    # 期交所格式變動大，取最後一欄通常是「未平倉損益」或「未平倉淨額」
                    # 但準確來說是：多方OI | 空方OI | 淨OI (通常是倒數欄位)
                    # 我們直接取 iloc[:, -1] 並清理逗號
                    val = str(row_foreign.iloc[0, -1]).replace(",", "").strip()
                    chips_data["Foreign_Net_OI"] = int(val)
                    chips_data["Date"] = q_date_str
                    found_date = q_date_str
                    break # 找到資料就停止
        except:
            continue
    
    # 2. 抓取大戶期貨 (使用上面找到的有效日期)
    if found_date:
        try:
            url_big = "https://www.taifex.com.tw/cht/3/largeTraderFutQry"
            payload_big = {"queryDate": found_date, "contractId": "TX"}
            res_big = requests.post(url_big, data=payload_big, headers=headers, timeout=5)
            if "查無資料" not in res_big.text:
                dfs_big = pd.read_html(StringIO(res_big.text))
                df_big = dfs_big[0]
                # 欄位：買方前五(2), 買方前十(4), 賣方前五(6), 賣方前十(8) (依據HTML結構)
                # 需防呆處理
                def get_val(r, idx):
                    return int(str(r.iloc[idx]).replace(",", "").strip())
                
                row = df_big.iloc[0] # 近月或全月
                buy_5 = get_val(row, 2)
                buy_10 = get_val(row, 4)
                sell_5 = get_val(row, 6)
                sell_10 = get_val(row, 8)
                
                chips_data["Top5_Net_OI"] = buy_5 - sell_5
                chips_data["Top10_Net_OI"] = buy_10 - sell_10
        except:
            pass

    return chips_data

# --- 期交所選擇權資料 ---
@st.cache_data(ttl=300)
def get_option_data():
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    headers = {"User-Agent": "Mozilla/5.0"}
    for i in range(5):
        # 同樣邏輯：如果是下午3點前，今天資料一定沒有，直接從昨天開始找
        if i == 0 and datetime.now(tz=TW_TZ).hour < 15:
            continue

        query_date = (datetime.now(tz=TW_TZ) - timedelta(days=i)).strftime("%Y/%m/%d")
        payload = {"queryType": "2", "marketCode": "0", "commodity_id": "TXO", "queryDate": query_date, "MarketCode": "0", "commodity_idt": "TXO"}
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=5)
            if "查無資料" in res.text or len(res.text) < 500: continue
            
            dfs = pd.read_html(StringIO(res.text))
            df = dfs[0]
            df.columns = [str(c).replace(" ", "").replace("*", "").replace("契約", "").strip() for c in df.columns]
            
            # 動態找欄位
            col_map = {
                "Month": next((c for c in df.columns if "月" in c or "週" in c), None),
                "Strike": next((c for c in df.columns if "履約" in c), None),
                "Type": next((c for c in df.columns if "買賣" in c), None),
                "OI": next((c for c in df.columns if "未沖銷" in c or "OI" in c), None),
                "Price": next((c for c in df.columns if "結算" in c or "收盤" in c or "Price" in c), None)
            }
            if not all(col_map.values()): continue
            
            df = df.rename(columns=col_map)[list(col_map.keys())].dropna(subset=["Type"])
            df["Strike"] = pd.to_numeric(df["Strike"].astype(str).str.replace(",", ""), errors="coerce")
            df["OI"] = pd.to_numeric(df["OI"].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
            df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",", "").replace("-", "0"), errors="coerce").fillna(0)
            df["Amount"] = df["OI"] * df["Price"] * 50
            
            if df["OI"].sum() > 0: return df, query_date
        except: continue
    return None, None

# --- Tornado 圖 (保持原樣) ---
def plot_tornado_chart(df_target, title_text, spot_price):
    is_call = df_target["Type"].str.contains("買|Call", case=False, na=False)
    df_call = df_target[is_call][["Strike", "OI", "Amount"]].rename(columns={"OI": "Call_OI", "Amount": "Call_Amt"})
    df_put = df_target[~is_call][["Strike", "OI", "Amount"]].rename(columns={"OI": "Put_OI", "Amount": "Put_Amt"})
    data = pd.merge(df_call, df_put, on="Strike", how="outer").fillna(0).sort_values("Strike")

    total_put_money = data["Put_Amt"].sum()
    total_call_money = data["Call_Amt"].sum()
    
    # 智慧過濾：只顯示有意義的區間
    data = data[(data["Call_OI"] > 200) | (data["Put_OI"] > 200)]
    if spot_price and spot_price > 0:
        data = data[(data["Strike"] >= spot_price - 800) & (data["Strike"] <= spot_price + 800)]
    
    max_oi = max(data["Put_OI"].max(), data["Call_OI"].max()) if not data.empty else 1000
    x_limit = max_oi * 1.1

    fig = go.Figure()
    fig.add_trace(go.Bar(y=data["Strike"], x=-data["Put_OI"], orientation="h", name="Put (支撐)", marker_color="#2ca02c", opacity=0.85, customdata=data["Put_Amt"]/1e8, hovertemplate="<b>%{y}</b><br>Put OI: %{x}<br>Amt: %{customdata:.2f}億"))
    fig.add_trace(go.Bar(y=data["Strike"], x=data["Call_OI"], orientation="h", name="Call (壓力)", marker_color="#d62728", opacity=0.85, customdata=data["Call_Amt"]/1e8, hovertemplate="<b>%{y}</b><br>Call OI: %{x}<br>Amt: %{customdata:.2f}億"))

    annotations = []
    if spot_price and spot_price > 0:
        fig.add_hline(y=spot_price, line_dash="dash", line_color="#ff7f0e", line_width=2)
        annotations.append(dict(x=1, y=spot_price, xref="paper", yref="y", text=f"現貨 {int(spot_price)}", showarrow=False, bgcolor="#ff7f0e", font=dict(color="white")))

    annotations.append(dict(x=0.05, y=1.05, xref="paper", yref="paper", text=f"Put總額: {total_put_money/1e8:.1f}億", showarrow=False, font=dict(color="#2ca02c")))
    annotations.append(dict(x=0.95, y=1.05, xref="paper", yref="paper", text=f"Call總額: {total_call_money/1e8:.1f}億", showarrow=False, font=dict(color="#d62728")))

    fig.update_layout(title=dict(text=title_text, x=0.5), xaxis=dict(range=[-x_limit, x_limit]), barmode="overlay", height=700, annotations=annotations)
    return fig

# --- AI 分析 Prompt 建構 (整合期貨籌碼) ---
def build_dealer_prompt(contract_code, settlement_date, taiex_price, chips_data, data_str):
    f_oi = chips_data.get('Foreign_Net_OI', 0)
    t10_oi = chips_data.get('Top10_Net_OI', 0)
    
    # 簡單判讀
    f_status = "大舉看多" if f_oi > 5000 else "看空" if f_oi < -5000 else "中性震盪"
    
    prompt = f"""
你現在是【主力莊家】。
【市場關鍵情報】
1. **現貨**：{taiex_price}
2. **期貨籌碼 (莊家底牌)**：
   - 外資期貨淨未平倉：{f_oi} 口 ({f_status})
   - 十大交易人淨未平倉：{t10_oi} 口
   (注意：若期貨是大空單，且選擇權 Put OI 很高，代表莊家可能準備殺盤)

【任務：控盤劇本】
請根據 CSV 選擇權籌碼與上述期貨籌碼分析：
1. **肥羊區**：散戶重倉在哪？
2. **劇本**：結合期貨方向，預測未來怎麼走(殺多/軋空/區間盤整)？
3. **指令**：給出如 "Sell Call @ 23000" 的簡短指令。

數據：
{data_str}
"""
    return prompt

# --- AI 呼叫 ---
def ask_ai(model_type, df, price, code, date, chips):
    try:
        df_ai = df.nlargest(60, "Amount") if "Amount" in df.columns else df
        prompt = build_dealer_prompt(code, date, price, chips, df_ai.to_csv(index=False))
        
        if model_type == "gemini" and gemini_model:
            return gemini_model.generate_content(prompt).text
        elif model_type == "openai" and openai_client:
            return openai_client.chat.completions.create(
                model=openai_model_name,
                messages=[{"role": "system", "content": "你是冷血莊家。"}, {"role": "user", "content": prompt}]
            ).choices[0].message.content
        return "模型未設定"
    except Exception as e: return f"分析失敗: {e}"

# --- 主程式 ---
def main():
    st.title("🤖 台指期籌碼戰情室 (莊家獵殺版)")
    
    # 1. 抓資料
    with st.spinner("連線期交所資料庫 (自動回溯有效交易日)..."):
        chips_data = get_future_chips() # 新增：抓期貨
        df, data_date = get_option_data()
        auto_taiex = get_realtime_data()

    if df is None:
        st.error("無法抓取選擇權資料，請檢查期交所連線。")
        return

    # 2. 側邊欄與價格校正
    st.sidebar.info(f"籌碼日期: {data_date}\n(盤中僅能顯示昨日盤後籌碼)")
    st.sidebar.download_button("下載CSV", df.to_csv(index=False).encode("utf-8-sig"), "opt.csv")
    
    with st.expander("🛠️ 價格校正", expanded=False):
        manual_price = st.number_input("手動輸入現貨價", value=0.0)
    final_price = manual_price if manual_price > 0 else (auto_taiex if auto_taiex else 0)

    # 3. 儀表板 (新增期貨數據)
    st.markdown("### 🧭 莊家期貨籌碼 (Trend Dashboard)")
    k1, k2, k3 = st.columns(3)
    f_oi = chips_data['Foreign_Net_OI']
    t10_oi = chips_data['Top10_Net_OI']
    
    k1.metric("外資期貨淨單", f"{f_oi:,}", "多" if f_oi>0 else "空", delta_color="normal" if f_oi>0 else "inverse")
    k2.metric("十大交易人淨單", f"{t10_oi:,}", "大戶多" if t10_oi>0 else "大戶空", delta_color="normal" if t10_oi>0 else "inverse")
    
    msg = "震盪"
    if f_oi > 3000 and t10_oi > 1000: msg = "🔥 多頭共振"
    elif f_oi < -3000 and t10_oi < -1000: msg = "❄️ 空頭共振"
    k3.metric("AI 風向判讀", msg)
    st.markdown("---")

    # 4. AI 分析區
    st.markdown("### 💡 雙 AI 莊家控盤")
    if st.button("🚀 啟動莊家思維 (含期貨籌碼)"):
        c1, c2 = st.columns(2)
        with c1: st.info(ask_ai("gemini", df, final_price, "近月", data_date, chips_data))
        with c2: st.info(ask_ai("openai", df, final_price, "近月", data_date, chips_data))

    # 5. 圖表區
    st.markdown("### 🌪️ 籌碼龍捲風")
    codes = sorted([c for c in df["Month"].unique() if len(c) < 9]) # 簡單過濾
    if codes:
        target_code = codes[0] # 取第一個合約(通常是近月)
        st.plotly_chart(plot_tornado_chart(df[df["Month"]==target_code], f"{target_code} 合約分佈", final_price), use_container_width=True)

if __name__ == "__main__":
    main()
