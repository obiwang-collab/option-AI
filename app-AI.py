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
        # 優先順序
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
        if len(code) < 6: return "9999/99/99"
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
                if len(wednesdays) >= week_num: day = wednesdays[week_num - 1]
        elif "F" in code:
            match = re.search(r"F(\d)", code)
            if match:
                week_num = int(match.group(1))
                if len(fridays) >= week_num: day = fridays[week_num - 1]
        else:
            if len(wednesdays) >= 3: day = wednesdays[2]
        
        if day: return f"{year}/{month:02d}/{day:02d}"
        else: return "9999/99/99"
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
    # 1. Yahoo
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII?interval=1d&range=1d&_={ts}"
        res = requests.get(url, headers=headers, timeout=5)
        data = res.json()
        meta = data["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice")
        if price is None: price = meta.get("chartPreviousClose") 
        if price: taiex = float(price)
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

# --- 新增：期貨法人與大戶籌碼 (替代玩股網爬蟲) ---
@st.cache_data(ttl=3600)
def get_future_chips():
    """
    從期交所直接抓取三大法人與大戶期貨數據
    (比爬玩股網網頁更穩定且合法)
    """
    chips_data = {
        "Date": "",
        "Foreign_Net_OI": 0, # 外資淨未平倉
        "Dealer_Net_OI": 0,  # 自營淨未平倉
        "Top5_Net_OI": 0,    # 前五大淨未平倉
        "Top10_Net_OI": 0,   # 前十大淨未平倉
    }
    
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # 1. 抓取三大法人期貨 (區分日夜盤合併)
    # https://www.taifex.com.tw/cht/3/futContractsDate
    try:
        # 嘗試抓最近 3 天，直到有資料
        for i in range(3):
            q_date = datetime.now(tz=TW_TZ) - timedelta(days=i)
            q_date_str = q_date.strftime("%Y/%m/%d")
            
            url = "https://www.taifex.com.tw/cht/3/futContractsDate"
            payload = {
                "queryType": "1",
                "goDay": "",
                "doDay": "",
                "queryDate": q_date_str,
                "commodityId": "TXF" # 台指期
            }
            res = requests.post(url, data=payload, headers=headers, timeout=5)
            if "查無資料" not in res.text:
                dfs = pd.read_html(StringIO(res.text))
                # 通常是第一個表格
                df_inst = dfs[0]
                
                # 尋找 "外資" 且 "多空淨額"
                # 欄位結構較複雜，通常第 3 欄是身份，最後幾欄是 OI 淨額
                # 簡化處理：轉成 string 搜尋
                
                # 整理欄位 (去除多層 index)
                df_inst.columns = [str(c[-1]).strip() for c in df_inst.columns]
                
                # 抓取外資列
                row_foreign = df_inst[df_inst.iloc[:, 0].astype(str).str.contains("外資", na=False)]
                if not row_foreign.empty:
                    # 假設最後一欄附近的 "未平倉餘額" -> "口數" -> "多空淨額"
                    # 或是直接找 "多空淨額" 欄位
                    # 期交所表格：身分 | ... | 未平倉餘額(多) | 未平倉餘額(空) | 未平倉餘額(淨)
                    net_oi_val = row_foreign.iloc[0, -1] # 最後一欄通常是淨額
                    chips_data["Foreign_Net_OI"] = int(str(net_oi_val).replace(",", ""))
                
                # 抓取自營商
                row_dealer = df_inst[df_inst.iloc[:, 0].astype(str).str.contains("自營商", na=False)]
                if not row_dealer.empty:
                    net_oi_val = row_dealer.iloc[0, -1]
                    chips_data["Dealer_Net_OI"] = int(str(net_oi_val).replace(",", ""))

                chips_data["Date"] = q_date_str
                break
    except Exception as e:
        print(f"法人抓取失敗: {e}")

    # 2. 抓取大戶期貨 (大台)
    # https://www.taifex.com.tw/cht/3/largeTraderFutQry
    try:
        if chips_data["Date"]:
            url_big = "https://www.taifex.com.tw/cht/3/largeTraderFutQry"
            payload_big = {
                "queryDate": chips_data["Date"],
                "contractId": "TX", # 大台
            }
            res_big = requests.post(url_big, data=payload_big, headers=headers, timeout=5)
            if "查無資料" not in res_big.text:
                dfs_big = pd.read_html(StringIO(res_big.text))
                df_big = dfs_big[0]
                # 格式: 契約 | 到期月份 | 前五大(買) | 前五大(賣) | ...
                # 只需要 "全月" 或 "所有的加總" ? 通常看近月或全月。這裡簡化抓第一列(通常是近月或最大量月)
                
                # 處理欄位
                # 需計算: (前五大買方OI - 前五大賣方OI)
                # 欄位通常包含: "買方" -> "前五大交易人" -> "部位數", "賣方"...
                
                # 為了穩健，直接用 string parse 或 iloc 硬抓特定位置
                # 假設第一列是近月合約
                # 買方前五大部位: col 3 (index 2) ?? 需視表格結構
                # 結構通常是: 買方[前五(部位, %), 前十(部位, %)] | 賣方[前五(部位, %), 前十(部位, %)]
                
                # 讓我們用比較笨但穩的方法：找數字
                row = df_big.iloc[0] # 第一列數據
                
                # 欄位很多，透過觀察期交所 HTML
                # 買方-前五大-部位數 (idx 2)
                # 買方-前十大-部位數 (idx 4)
                # 賣方-前五大-部位數 (idx 6)
                # 賣方-前十大-部位數 (idx 8)
                # *注意：期交所網頁改版可能會動，但相對穩定*
                
                buy_5 = int(str(df_big.iloc[0, 2]).replace(",", ""))
                buy_10 = int(str(df_big.iloc[0, 4]).replace(",", ""))
                sell_5 = int(str(df_big.iloc[0, 6]).replace(",", ""))
                sell_10 = int(str(df_big.iloc[0, 8]).replace(",", ""))
                
                chips_data["Top5_Net_OI"] = buy_5 - sell_5
                chips_data["Top10_Net_OI"] = buy_10 - sell_10

    except Exception as e:
        print(f"大戶抓取失敗: {e}")

    return chips_data

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
            if "查無資料" in res.text or len(res.text) < 500: continue
            dfs = pd.read_html(StringIO(res.text))
            df = dfs[0]
            df.columns = [str(c).replace(" ", "").replace("*", "").replace("契約", "").strip() for c in df.columns]
            
            month_col = next((c for c in df.columns if "月" in c or "週" in c), None)
            strike_col = next((c for c in df.columns if "履約" in c), None)
            type_col = next((c for c in df.columns if "買賣" in c), None)
            oi_col = next((c for c in df.columns if "未沖銷" in c or "OI" in c), None)
            price_col = next((c for c in df.columns if "結算" in c or "收盤" in c or "Price" in c), None)

            if not all([month_col, strike_col, type_col, oi_col, price_col]): continue
            
            df = df.rename(columns={month_col: "Month", strike_col: "Strike", type_col: "Type", oi_col: "OI", price_col: "Price"})
            df = df[["Month", "Strike", "Type", "OI", "Price"]].copy().dropna(subset=["Type"])
            
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

# --- Tornado 圖 (保留原格式) ---
def plot_tornado_chart(df_target, title_text, spot_price):
    is_call = df_target["Type"].str.contains("買|Call", case=False, na=False)
    df_call = df_target[is_call][["Strike", "OI", "Amount"]].rename(columns={"OI": "Call_OI", "Amount": "Call_Amt"})
    df_put = df_target[~is_call][["Strike", "OI", "Amount"]].rename(columns={"OI": "Put_OI", "Amount": "Put_Amt"})
    data = pd.merge(df_call, df_put, on="Strike", how="outer").fillna(0).sort_values("Strike")

    total_put_money = data["Put_Amt"].sum()
    total_call_money = data["Call_Amt"].sum()
    data = data[(data["Call_OI"] > 300) | (data["Put_OI"] > 300)]
    
    FOCUS_RANGE = 1200
    if spot_price and spot_price > 0: center_price = spot_price
    elif not data.empty: center_price = data.loc[data["Put_OI"].idxmax(), "Strike"]
    else: center_price = 0

    if center_price > 0:
        min_s = center_price - FOCUS_RANGE
        max_s = center_price + FOCUS_RANGE
        data = data[(data["Strike"] >= min_s) & (data["Strike"] <= max_s)]

    max_oi = max(data["Put_OI"].max(), data["Call_OI"].max()) if not data.empty else 1000
    x_limit = max_oi * 1.1

    fig = go.Figure()
    fig.add_trace(go.Bar(y=data["Strike"], x=-data["Put_OI"], orientation="h", name="Put (支撐)", marker_color="#2ca02c", opacity=0.85, customdata=data["Put_Amt"] / 1e8, hovertemplate="<b>履約價: %{y}</b><br>Put OI: %{x} 口<br>Put 市值: %{customdata:.2f}億<extra></extra>"))
    fig.add_trace(go.Bar(y=data["Strike"], x=data["Call_OI"], orientation="h", name="Call (壓力)", marker_color="#d62728", opacity=0.85, customdata=data["Call_Amt"] / 1e8, hovertemplate="<b>履約價: %{y}</b><br>Call OI: %{x} 口<br>Call 市值: %{customdata:.2f}億<extra></extra>"))

    annotations = []
    if spot_price and spot_price > 0 and not data.empty:
        if data["Strike"].min() <= spot_price <= data["Strike"].max():
            fig.add_hline(y=spot_price, line_dash="dash", line_color="#ff7f0e", line_width=2)
            annotations.append(dict(x=1, y=spot_price, xref="paper", yref="y", text=f" 現貨 {int(spot_price)} ", showarrow=False, xanchor="left", align="center", font=dict(color="white", size=12), bgcolor="#ff7f0e", bordercolor="#ff7f0e", borderpad=4))

    annotations.append(dict(x=0.02, y=1.05, xref="paper", yref="paper", text=f"<b>Put 總金額</b><br>{total_put_money/1e8:.1f} 億", showarrow=False, align="left", font=dict(size=14, color="#2ca02c"), bgcolor="white", bordercolor="#2ca02c", borderwidth=2, borderpad=6))
    annotations.append(dict(x=0.98, y=1.05, xref="paper", yref="paper", text=f"<b>Call 總金額</b><br>{total_call_money/1e8:.1f} 億", showarrow=False, align="right", font=dict(size=14, color="#d62728"), bgcolor="white", bordercolor="#d62728", borderwidth=2, borderpad=6))

    fig.update_layout(title=dict(text=title_text, y=0.95, x=0.5, xanchor="center", yanchor="top", font=dict(size=20, color="black")), xaxis=dict(title="未平倉量 (OI)", range=[-x_limit, x_limit], showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor="black", tickmode="array", tickvals=[-x_limit * 0.75, -x_limit * 0.5, -x_limit * 0.25, 0, x_limit * 0.25, x_limit * 0.5, x_limit * 0.75], ticktext=[f"{int(x_limit*0.75)}", f"{int(x_limit*0.5)}", f"{int(x_limit*0.25)}", "0", f"{int(x_limit*0.25)}", f"{int(x_limit*0.5)}", f"{int(x_limit*0.75)}"]), yaxis=dict(title="履約價", tickmode="linear", dtick=100, tickformat="d"), barmode="overlay", legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"), height=750, margin=dict(l=40, r=80, t=140, b=60), annotations=annotations, paper_bgcolor="white", plot_bgcolor="white")
    return fig

# --- AI 分析共用 Prompt 建構 ---
def build_dealer_prompt(contract_code, settlement_date, taiex_price, chips_data, data_str):
    # 判斷外資與大戶態度
    f_oi = chips_data.get('Foreign_Net_OI', 0)
    t5_oi = chips_data.get('Top5_Net_OI', 0)
    
    f_status = "大幅看多" if f_oi > 10000 else "偏多" if f_oi > 2000 else "中性" if abs(f_oi) <= 2000 else "偏空" if f_oi > -10000 else "大幅看空"
    t5_status = "主力做多" if t5_oi > 2000 else "主力做空" if t5_oi < -2000 else "主力觀望"

    prompt = f"""
你現在是台指選擇權市場的【主力莊家】。你的目標只有一個：**在結算日吃掉最多散戶的權利金，並配合期貨部位獲利，讓自己的利潤最大化**。

【市場關鍵數據】
1. **結算合約**：{contract_code} (結算日: {settlement_date})
2. **現貨指數**：{taiex_price} (控盤基準)
3. **期貨籌碼 (極重要)**：
   - 外資期貨淨未平倉：{f_oi} 口 ({f_status})
   - 前五大交易人淨未平倉：{t5_oi} 口 ({t5_status})
   *提示：若外資期貨大空單，但選擇權Put OI很厚，可能發生「摜壓殺多」；若外資期貨大，選擇權Call OI很厚，可能發生「軋空飛越」。*

【任務：莊家控盤推演】
請根據 CSV 選擇權籌碼（OI/金額）與上述期貨籌碼，進行分析：

1. **肥羊與雷區**：
   - **肥羊區**：散戶在哪個價位重倉（高OI）？這些是你收割的對象。
   - **籌碼矛盾點**：如果外資期貨看空（{f_oi}口），但散戶瘋狂買 Call，你要怎麼殺？反之亦然。

2. **操盤劇本 (Script)**：
   - 結合期貨籌碼，寫出未來幾天的「畫線」劇本。
   - 例如：「外資期貨握有萬口空單，上方 xxxx 壓力沈重，週三結算前我會先拉高誘多，再殺破 xxxx...」

3. **最佳結算點位**：
   - 給出一個具體的「結算點位」或「區間」，讓你的期貨部位賺錢，同時讓選擇權賣方利潤最大化。

4. **莊家指令**：
   - 簡短指令（例如：期貨避險做空，Sell Call @ xxxx）。

【回答要求】
- 使用第一人稱（本莊、我）。
- 語氣：**自信、冷血、貪婪**。
- 分析要詳細，字數約 300-500 字。

選擇權籌碼數據：
{data_str}
"""
    return prompt

# --- AI 分析函式 (Gemini) ---
def ask_gemini_brief(df_recent, taiex_price, contract_code, settlement_date, chips_data):
    if not gemini_model: return f"⚠️ {gemini_model_name}"
    try:
        df_ai = df_recent.nlargest(80, "Amount") if "Amount" in df_recent.columns else df_recent
        data_str = df_ai.to_csv(index=False)
        prompt = build_dealer_prompt(contract_code, settlement_date, taiex_price, chips_data, data_str)
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"分析忙碌中 ({str(e)})"

# --- AI 分析函式 (ChatGPT) ---
def ask_openai_brief(df_recent, taiex_price, contract_code, settlement_date, chips_data):
    if not openai_client: return f"⚠️ {openai_model_name}"
    try:
        df_ai = df_recent.nlargest(80, "Amount") if "Amount" in df_recent.columns else df_recent
        data_str = df_ai.to_csv(index=False)
        user_prompt = build_dealer_prompt(contract_code, settlement_date, taiex_price, chips_data, data_str)
        
        response = openai_client.chat.completions.create(
            model=openai_model_name,
            messages=[
                {"role": "system", "content": "你是一個冷血、唯利是圖的期貨市場莊家。"},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7, max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"分析忙碌中 ({str(e)})"

# --- 主程式 ---
def main():
    st.title("🤖 台指期籌碼戰情室 (莊家獵殺版)")

    if st.sidebar.button("🔄 重新整理", key="refresh_main"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("正在連線期交所 (選擇權 + 期貨法人數據)..."):
        df, data_date = get_option_data()
        chips_data = get_future_chips() # 新增：抓取期貨籌碼
        auto_taiex = get_realtime_data()

    if df is None:
        st.error("查無選擇權資料，請稍後再試。")
        return

    # --- 側邊欄下載 ---
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.sidebar.download_button("📥 下載選擇權數據", csv, f"option_{data_date.replace('/','')}.csv", "text/csv")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**期貨籌碼日期**: {chips_data['Date']}")
    st.sidebar.markdown(f"外資期貨淨OI: **{chips_data['Foreign_Net_OI']:,}**")
    st.sidebar.markdown(f"前五大淨OI: **{chips_data['Top5_Net_OI']:,}**")

    # --- 價格校正 ---
    with st.expander("🛠️ 數據校正設定", expanded=False):
        mc1, mc2 = st.columns([1, 2])
        with mc1: st.info(f"系統自動抓取: {auto_taiex}")
        with mc2:
            manual_price_input = st.number_input("請輸入看盤軟體最新價格 (0為自動):", min_value=0.0, value=0.0, step=1.0, format="%.2f")
    
    final_taiex = manual_price_input if manual_price_input > 0 else (auto_taiex if auto_taiex else 0)
    price_source_msg = "⚠️ 手動校正" if manual_price_input > 0 else "系統自動"

    # --- 頂部指標 ---
    total_call_amt = df[df["Type"].str.contains("買|Call", case=False, na=False)]["Amount"].sum()
    total_put_amt = df[df["Type"].str.contains("賣|Put", case=False, na=False)]["Amount"].sum()
    pc_ratio_amt = ((total_put_amt / total_call_amt) * 100 if total_call_amt > 0 else 0)

    # 顯示基礎資訊
    c1, c2, c3, c4 = st.columns([1.2, 0.8, 1, 1])
    c1.markdown(f"<div style='text-align: left;'><span style='font-size: 14px; color: #555;'>更新時間</span><br><span style='font-size: 18px; font-weight: bold;'>{datetime.now(tz=TW_TZ).strftime('%Y/%m/%d %H:%M:%S')}</span></div>", unsafe_allow_html=True)
    c2.metric(f"大盤/期貨 ({price_source_msg})", f"{int(final_taiex) if final_taiex else 'N/A'}")
    c3.metric("P/C 金額比", f"{pc_ratio_amt:.1f}%", "偏多" if pc_ratio_amt > 100 else "偏空", delta_color="normal" if pc_ratio_amt > 100 else "inverse")
    c4.metric("資料日期", data_date)
    
    st.markdown("---")

    # ==========================================
    # 🆕 籌碼多空儀表板 (期貨數據展示區)
    # ==========================================
    st.markdown("### 🧭 莊家期貨籌碼儀表板 (Trend Dashboard)")
    k1, k2, k3, k4 = st.columns(4)
    
    # 外資期貨
    f_oi = chips_data['Foreign_Net_OI']
    k1.metric("外資期貨淨未平倉", f"{f_oi:,} 口", "偏多" if f_oi > 0 else "偏空", delta_color="normal" if f_oi > 0 else "inverse")
    
    # 十大交易人 (代表大戶)
    t10_oi = chips_data['Top10_Net_OI']
    k2.metric("十大交易人淨未平倉", f"{t10_oi:,} 口", "大戶多" if t10_oi > 0 else "大戶空", delta_color="normal" if t10_oi > 0 else "inverse")
    
    # 自營商 (通常做選擇權避險，參考用)
    d_oi = chips_data['Dealer_Net_OI']
    k3.metric("自營商期貨淨未平倉", f"{d_oi:,} 口", "避險多" if d_oi > 0 else "避險空")
    
    # 綜合解讀 (簡易邏輯)
    signal = "震盪整理"
    if f_oi > 3000 and t10_oi > 1000: signal = "多頭共振 🔥"
    elif f_oi < -3000 and t10_oi < -1000: signal = "空頭共振 ❄️"
    elif f_oi * t10_oi < 0: signal = "土洋對作 ⚔️"
    
    k4.metric("AI 籌碼風向判讀", signal)
    
    st.markdown("---")

    # ==========================================
    # 合約選擇與 AI
    # ==========================================
    nearest_code, nearest_date = None, None
    plot_targets = []
    unique_codes = df["Month"].unique()
    all_contracts = []
    for code in unique_codes:
        s_date_str = get_settlement_date(code)
        if s_date_str == "9999/99/99" or s_date_str <= data_date: continue
        all_contracts.append({"code": code, "date": s_date_str})
    all_contracts.sort(key=lambda x: x["date"])

    if all_contracts:
        nearest = all_contracts[0]
        nearest_code, nearest_date = nearest["code"], nearest["date"]
        nearest_df = df[df["Month"] == nearest_code]
        plot_targets.append({"title": "最近結算", "info": nearest})
        monthly = next((c for c in all_contracts if len(c["code"]) == 6), None)
        if monthly and monthly["code"] != nearest_code:
            plot_targets.append({"title": "當月月選", "info": monthly})

    st.markdown("### 💡 雙 AI 莊家控盤室")
    if nearest_code:
        st.caption(f"目標合約：**{nearest_code}** | 結算日：**{nearest_date}** | 結合期貨籌碼進行綜合分析")
        
        if st.button("🚀 啟動莊家思維推演 (含期貨籌碼)", type="primary"):
            ai_col1, ai_col2 = st.columns(2)
            with ai_col1:
                st.markdown(f"#### 💎 Gemini 莊家")
                with st.spinner("Gemini 正在計算多空力道..."):
                    # 傳入 chips_data
                    msg = ask_gemini_brief(nearest_df, final_taiex, nearest_code, nearest_date, chips_data)
                st.info(msg)
            with ai_col2:
                st.markdown(f"#### 💬 ChatGPT 莊家")
                with st.spinner("ChatGPT 正在擬定獵殺劇本..."):
                    msg = ask_openai_brief(nearest_df, final_taiex, nearest_code, nearest_date, chips_data)
                st.info(msg)

    st.markdown("---")

    # ==========================================
    # 龍捲風圖
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
                
                title_text = f"<b>【{target['title']}】 {m_code}</b><br><span style='font-size: 14px;'>結算: {s_date} | P/C比: {sub_ratio:.1f}%</span>"
                st.plotly_chart(plot_tornado_chart(df_target, title_text, final_taiex), use_container_width=True)
    else:
        st.info("目前無可識別的未來結算合約。")

if __name__ == "__main__":
    main()
