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
st.set_page_config(layout="wide", page_title="台指期籌碼戰情室 (法人透視版)")
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
        for target in ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]:
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

# 手動修正結算日
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
            week_num = int(re.search(r"W(\d)", code).group(1))
            day = wednesdays[week_num - 1] if len(wednesdays) >= week_num else None
        else:
            day = wednesdays[2] if len(wednesdays) >= 3 else None
        return f"{year}/{month:02d}/{day:02d}" if day else "9999/99/99"
    except: return "9999/99/99"

# --- 1. 現貨即時價 ---
@st.cache_data(ttl=60)
def get_realtime_data():
    ts = int(time.time())
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII?interval=1d&range=1d&_={ts}"
        res = requests.get(url, headers=headers, timeout=5)
        meta = res.json()["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice") or meta.get("chartPreviousClose")
        if price: return float(price)
    except: pass
    
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

# --- 2. 選擇權籌碼 (全市場 OI) ---
@st.cache_data(ttl=300)
def get_option_data():
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    for i in range(5):
        query_date = (datetime.now(tz=TW_TZ) - timedelta(days=i)).strftime("%Y/%m/%d")
        try:
            payload = {"queryType": "2", "marketCode": "0", "commodity_id": "TXO", "queryDate": query_date, "MarketCode": "0", "commodity_idt": "TXO"}
            res = requests.post(url, data=payload, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            if "查無資料" in res.text or len(res.text) < 500: continue
            
            df = pd.read_html(StringIO(res.text))[0]
            df.columns = [str(c).replace(" ", "").replace("*", "").replace("契約", "").strip() for c in df.columns]
            
            # 欄位對應
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

# --- 3. [新增] 三大法人選擇權籌碼 ---
@st.cache_data(ttl=3600)
def get_institutional_data(ref_date_str):
    # ref_date_str 格式為 YYYY/MM/DD
    url = "https://www.taifex.com.tw/cht/3/callsAndPutsDate"
    
    # 嘗試抓取當天，若無則抓前一天
    try_dates = [ref_date_str]
    dt_obj = datetime.strptime(ref_date_str, "%Y/%m/%d")
    try_dates.append((dt_obj - timedelta(days=1)).strftime("%Y/%m/%d"))
    
    for d in try_dates:
        try:
            payload = {
                "queryType": "1",
                "goDay": "",
                "doQuery": "1",
                "dateaddcnt": "",
                "queryDate": d,
                "commodityId": "TXO"
            }
            res = requests.post(url, data=payload, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            if "查無資料" in res.text: continue

            dfs = pd.read_html(StringIO(res.text))
            df = dfs[0]
            
            # 清理表格 (三大法人)
            # 格式通常是: 序號, 身分, 買權口數(買/賣/淨), 賣權口數(買/賣/淨), 未平倉...
            # 簡化抓取邏輯：找 "外資"、"自營商" 的行
            
            # 定義需要的欄位名稱
            result = {
                "Date": d,
                "Foreign_Call_Net": 0, "Foreign_Put_Net": 0, "Foreign_Call_OI_Net": 0, "Foreign_Put_OI_Net": 0,
                "Dealer_Call_Net": 0, "Dealer_Put_Net": 0, "Dealer_Call_OI_Net": 0, "Dealer_Put_OI_Net": 0,
            }
            
            for idx, row in df.iterrows():
                row_str = str(row.values)
                if "外資" in row_str:
                    # 依據期交所表格結構，通常倒數幾個欄位是 OI 淨額
                    # 這裡用比較粗暴但有效的方式：依賴表格固定結構
                    # 假設 columns 是多層索引，我們只取 values
                    vals = [str(x).replace(",", "").strip() for x in row.values if str(x).replace(",", "").strip().replace("-","").isdigit()]
                    vals = [int(x) for x in vals]
                    # 期交所格式：買權(買/賣/淨) -> 賣權(買/賣/淨) -> 買權OI(買/賣/淨) -> 賣權OI(買/賣/淨)
                    # 這邊簡化取 OI 淨額 (最後兩組的最後一個值)
                    if len(vals) >= 12:
                        result["Foreign_Call_OI_Net"] = vals[8]  # 買權OI淨額
                        result["Foreign_Put_OI_Net"] = vals[11]  # 賣權OI淨額
                        
                elif "自營商" in row_str:
                    vals = [str(x).replace(",", "").strip() for x in row.values if str(x).replace(",", "").strip().replace("-","").isdigit()]
                    vals = [int(x) for x in vals]
                    if len(vals) >= 12:
                        result["Dealer_Call_OI_Net"] = vals[8]
                        result["Dealer_Put_OI_Net"] = vals[11]

            return result
        except:
            continue
    return None

# --- Tornado 圖 (維持原樣，不贅述) ---
def plot_tornado_chart(df_target, title_text, spot_price):
    is_call = df_target["Type"].str.contains("買|Call", case=False, na=False)
    df_call = df_target[is_call][["Strike", "OI", "Amount"]].rename(columns={"OI": "Call_OI", "Amount": "Call_Amt"})
    df_put = df_target[~is_call][["Strike", "OI", "Amount"]].rename(columns={"OI": "Put_OI", "Amount": "Put_Amt"})
    data = pd.merge(df_call, df_put, on="Strike", how="outer").fillna(0).sort_values("Strike")
    
    total_put_money, total_call_money = data["Put_Amt"].sum(), data["Call_Amt"].sum()
    data = data[(data["Call_OI"] > 300) | (data["Put_OI"] > 300)]
    
    FOCUS_RANGE = 800
    center = spot_price if spot_price and spot_price > 0 else (data.loc[data["Put_OI"].idxmax(), "Strike"] if not data.empty else 0)
    if center > 0:
        base = round(center / 50) * 50
        data = data[(data["Strike"] >= base - FOCUS_RANGE) & (data["Strike"] <= base + FOCUS_RANGE)]

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
                      yaxis=dict(dtick=50, tick0=0, tickformat="d"), barmode="overlay", height=750, margin=dict(l=50, r=50, t=100, b=50))
    return fig

# --- AI Prompt (加入法人邏輯) ---
def get_ai_prompt(contract_code, settlement_date, taiex_price, data_str, inst_data):
    inst_text = "目前無法人詳細數據"
    if inst_data:
        inst_text = f"""
【三大法人籌碼結構 (重要)】
資料日期: {inst_data['Date']}
1. 外資 (Foreign):
   - Call OI 淨額: {inst_data['Foreign_Call_OI_Net']} 口 (正為多，負為空)
   - Put OI 淨額: {inst_data['Foreign_Put_OI_Net']} 口
2. 自營商 (Dealer/莊家):
   - Call OI 淨額: {inst_data['Dealer_Call_OI_Net']} 口
   - Put OI 淨額: {inst_data['Dealer_Put_OI_Net']} 口
        """
    
    return f"""
你現在是台指選擇權市場的【頂級莊家】。你的對手是散戶，你的盟友(或競爭者)是外資。
目標：推演結算日如何收割最大利益。

【市場參數】
- 合約：{contract_code} (結算: {settlement_date})
- **控盤基準價：{taiex_price}**
{inst_text}

【散戶 vs 法人 對抗邏輯】
* 如果「大量 OI」是外資/自營商賣出的 -> 這是**鐵壁**，很難突破。
* 如果「大量 OI」是散戶賣出(且法人在對作買入) -> 這是**燃料**，容易發生軋空(Short Squeeze)或殺多。

【任務】
根據 CSV 數據(全市場 OI)與上述法人籌碼：
1. **籌碼透視**：
   - 觀察目前 Call/Put 最大 OI 的位置。
   - 結合法人淨部位判斷：這些牆是「鋼板」(法人賣) 還是 「紙板」(散戶賣)？
2. **劇本推演**：
   - 若外資做多 Call，且上方 Call OI 巨大，是否可能軋空噴出？
   - 若自營商 Put 避險部位大，下方支撐是否強勁？
3. **結算目標**：給出最佳收割點位。
4. **操作指令**：簡短指令。

【語氣】
- 第一人稱(本莊)。自信、冷血。
- **必須明確指出「誰是肥羊」(散戶在做什麼方向)。**

數據(全市場前80大合約):
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
            return openai_client.chat.completions.create(
                model=openai_model_name,
                messages=[{"role": "system", "content": "你是冷血莊家。"}, {"role": "user", "content": prompt}],
                temperature=0.7
            ).choices[0].message.content
    except Exception as e: return f"忙碌中 ({str(e)})"

# --- 主程式 ---
def main():
    st.title("🦅 台指期籌碼戰情室 (法人透視版)")
    if st.sidebar.button("🔄 重新整理"): st.cache_data.clear(); st.rerun()

    with st.spinner("連線期交所中..."):
        df, data_date = get_option_data()
        auto_taiex = get_realtime_data()
        # 抓取法人數據
        inst_data = get_institutional_data(data_date) if data_date else None

    if df is None: st.error("查無選擇權資料"); return

    st.sidebar.download_button("📥 下載數據", df.to_csv(index=False).encode("utf-8-sig"), f"opt_{data_date.replace('/','')}.csv", "text/csv")

    # --- UI: 報價與法人摘要 ---
    with st.container(border=True):
        st.markdown("##### 🛠️ 控盤數據中心")
        c1, c2 = st.columns([1, 2])
        with c1: st.metric("📡 系統報價", f"{auto_taiex if auto_taiex else 'N/A'}")
        with c2:
            manual_input = st.number_input("🎹 手動校正點位", min_value=0.0, value=0.0, step=1.0, format="%.2f")

    final_taiex = manual_input if manual_input > 0 else (auto_taiex if auto_taiex else 0)

    # --- 法人籌碼看板 ---
    if inst_data:
        st.markdown("### 🏦 三大法人籌碼結構 (Smart Money)")
        i1, i2, i3, i4 = st.columns(4)
        
        f_call = inst_data.get('Foreign_Call_OI_Net', 0)
        f_put = inst_data.get('Foreign_Put_OI_Net', 0)
        d_call = inst_data.get('Dealer_Call_OI_Net', 0)
        d_put = inst_data.get('Dealer_Put_OI_Net', 0)
        
        i1.metric("外資 Call 淨OI", f"{f_call:,}", delta="偏多" if f_call>0 else "偏空")
        i2.metric("外資 Put 淨OI", f"{f_put:,}", delta="看空" if f_put>0 else "看多", delta_color="inverse")
        i3.metric("自營 Call 淨OI", f"{d_call:,}", delta="避險/造市" if abs(d_call)>10000 else "中性")
        i4.metric("自營 Put 淨OI", f"{d_put:,}", delta="避險/造市" if abs(d_put)>10000 else "中性", delta_color="inverse")
        
        # 簡單解讀
        f_trend = "外資做多" if f_call > 5000 and f_put < 5000 else ("外資做空" if f_put > 5000 and f_call < 5000 else "外資觀望")
        st.caption(f"📊 法人動態日期：{inst_data['Date']} | 簡易判讀：**{f_trend}** | 自營商通常為賣方(莊家)，外資通常為趨勢發動者。")
        st.markdown("---")

    # --- 顯示 P/C Ratio ---
    total_call = df[df["Type"].str.contains("Call|買")]["Amount"].sum()
    total_put = df[df["Type"].str.contains("Put|賣")]["Amount"].sum()
    ratio = (total_put/total_call*100) if total_call > 0 else 0
    st.metric("全市場 P/C 金額比", f"{ratio:.1f}%", "偏多" if ratio>100 else "偏空")

    # --- 合約邏輯 ---
    unique_codes = df["Month"].unique()
    all_contracts = sorted([{"code": c, "date": get_settlement_date(c)} for c in unique_codes if get_settlement_date(c) > data_date], key=lambda x: x["date"])
    
    nearest_code = all_contracts[0]["code"] if all_contracts else None
    nearest_date = all_contracts[0]["date"] if all_contracts else data_date
    nearest_df = df[df["Month"] == nearest_code] if nearest_code else df

    # --- AI 分析 ---
    st.subheader("💡 雙 AI 莊家控盤 (含法人籌碼分析)")
    if st.button("🚀 啟動推演 (分析誰是肥羊)", type="primary"):
        c_ai1, c_ai2 = st.columns(2)
        with c_ai1:
            st.markdown(f"**Gemini ({gemini_model_name})**")
            with st.spinner("Gemini 分析法人動向..."):
                st.info(ask_ai("gemini", nearest_df, final_taiex, nearest_code, nearest_date, inst_data))
        with c_ai2:
            st.markdown(f"**ChatGPT ({openai_model_name})**")
            with st.spinner("ChatGPT 計算軋空機率..."):
                st.info(ask_ai("openai", nearest_df, final_taiex, nearest_code, nearest_date, inst_data))

    # --- 圖表 ---
    if all_contracts:
        cols = st.columns(min(len(all_contracts), 2))
        for i, target in enumerate(all_contracts[:2]):
            with cols[i]:
                d_t = df[df["Month"] == target["code"]]
                title = f"<b>【{target['code']}】 結算: {target['date']}</b>"
                st.plotly_chart(plot_tornado_chart(d_t, title, final_taiex), use_container_width=True)

if __name__ == "__main__":
    main()
