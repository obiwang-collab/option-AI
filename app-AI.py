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
st.set_page_config(layout="wide", page_title="台指期籌碼戰情室 (雙 AI 對決版)")
TW_TZ = timezone(timedelta(hours=8)) 

# ==========================================
# 🔑 金鑰設定區 (自動讀取 Secrets 或本地變數)
# ==========================================
try:
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
except:
    GEMINI_KEY = ""
    OPENAI_KEY = ""

# --- 🧠 1. Gemini 模型設定 (自動找最佳模型) ---
def get_gemini_model(api_key):
    if not api_key: return None, "未設定"
    genai.configure(api_key=api_key)
    try:
        # 取得可用模型列表
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # 優先順序: Flash (快) -> 1.5 Pro (強) -> Pro (舊)
        for target in ['flash', 'gemini-1.5-pro', 'gemini-pro']:
            for m in models:
                if target in m.lower(): return genai.GenerativeModel(m), m
        
        # 兜底：隨便回傳第一個
        return (genai.GenerativeModel(models[0]), models[0]) if models else (None, "無可用模型")
    except Exception as e: return None, str(e)

# --- 🧠 2. ChatGPT 模型設定 ---
def get_openai_client(api_key):
    if not api_key: return None
    return OpenAI(api_key=api_key)

# 初始化模型
gemini_model, gemini_name = get_gemini_model(GEMINI_KEY)
openai_client = get_openai_client(OPENAI_KEY)

# 手動修正結算日
MANUAL_SETTLEMENT_FIX = {
    '202501W1': '2025/01/02', 
}

# --- 核心函式 ---
def get_settlement_date(contract_code):
    code = str(contract_code).strip().upper()
    for key, fix_date in MANUAL_SETTLEMENT_FIX.items():
        if key in code: return fix_date
    try:
        if len(code) < 6: return "9999/99/99"
        year = int(code[:4])
        month = int(code[4:6])
        c = calendar.monthcalendar(year, month)
        wednesdays = [week[calendar.WEDNESDAY] for week in c if week[calendar.WEDNESDAY] != 0]
        fridays = [week[calendar.FRIDAY] for week in c if week[calendar.FRIDAY] != 0]
        day = None
        if 'W' in code:
            match = re.search(r'W(\d)', code)
            if match:
                week_num = int(match.group(1))
                if len(wednesdays) >= week_num: day = wednesdays[week_num - 1]
        elif 'F' in code:
            match = re.search(r'F(\d)', code)
            if match:
                week_num = int(match.group(1))
                if len(fridays) >= week_num: day = fridays[week_num - 1]
        else:
            if len(wednesdays) >= 3: day = wednesdays[2]
        if day: return f"{year}/{month:02d}/{day:02d}"
        else: return "9999/99/99"
    except: return "9999/99/99"

@st.cache_data(ttl=60)
def get_realtime_data():
    taiex = None
    ts = int(time.time())
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_t00.tw&json=1&delay=0&_={ts}000"
        res = requests.get(url, timeout=2)
        data = res.json()
        if 'msgArray' in data and len(data['msgArray']) > 0:
            val = data['msgArray'][0].get('z', '-')
            if val == '-': val = data['msgArray'][0].get('o', '-')
            if val != '-': taiex = float(val)
    except: pass
    if taiex is None:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII?interval=1m&range=1d&_={ts}"
            res = requests.get(url, headers=headers, timeout=3)
            data = res.json()
            price = data['chart']['result'][0]['meta'].get('regularMarketPrice')
            if price: taiex = float(price)
        except: pass
    return taiex

@st.cache_data(ttl=300)
def get_option_data():
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    headers = {'User-Agent': 'Mozilla/5.0'}
    for i in range(5):
        query_date = (datetime.now(tz=TW_TZ) - timedelta(days=i)).strftime('%Y/%m/%d')
        payload = {'queryType': '2', 'marketCode': '0', 'dateaddcnt': '', 'commodity_id': 'TXO', 'commodity_id2': '', 'queryDate': query_date, 'MarketCode': '0', 'commodity_idt': 'TXO'}
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=5)
            if "查無資料" in res.text or len(res.text) < 500: continue 
            dfs = pd.read_html(StringIO(res.text))
            df = dfs[0]
            df.columns = [str(c).replace(' ', '').replace('*', '').replace('契約', '').strip() for c in df.columns]
            month_col = next((c for c in df.columns if '月' in c or '週' in c), None)
            strike_col = next((c for c in df.columns if '履約' in c), None)
            type_col = next((c for c in df.columns if '買賣' in c), None)
            oi_col = next((c for c in df.columns if '未沖銷' in c or 'OI' in c), None)
            price_col = next((c for c in df.columns if '結算' in c or '收盤' in c or 'Price' in c), None)
            vol_col = next((c for c in df.columns if '成交量' in c or 'Volume' in c), None)

            if not all([month_col, strike_col, type_col, oi_col, price_col]): continue
            rename_dict = {month_col:'Month', strike_col:'Strike', type_col:'Type', oi_col:'OI', price_col:'Price'}
            if vol_col: rename_dict[vol_col] = 'Volume'
            df = df.rename(columns=rename_dict)
            
            cols_to_keep = ['Month', 'Strike', 'Type', 'OI', 'Price']
            if 'Volume' in df.columns: cols_to_keep.append('Volume')
            df = df[cols_to_keep].copy()
            
            df = df.dropna(subset=['Type'])
            df['Type'] = df['Type'].astype(str).str.strip()
            df['Strike'] = pd.to_numeric(df['Strike'].astype(str).str.replace(',', ''), errors='coerce')
            df['OI'] = pd.to_numeric(df['OI'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df['Price'] = df['Price'].astype(str).str.replace(',', '').replace('-', '0')
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
            if 'Volume' in df.columns: df['Volume'] = pd.to_numeric(df['Volume'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df['Amount'] = df['OI'] * df['Price'] * 50
            if df['OI'].sum() == 0: continue 
            return df, query_date
        except: continue 
    return None, None

def plot_tornado_chart(df_target, title_text, spot_price):
    is_call = df_target['Type'].str.contains('買|Call', case=False, na=False)
    df_call = df_target[is_call][['Strike', 'OI', 'Amount']].rename(columns={'OI': 'Call_OI', 'Amount': 'Call_Amt'})
    df_put = df_target[~is_call][['Strike', 'OI', 'Amount']].rename(columns={'OI': 'Put_OI', 'Amount': 'Put_Amt'})
    data = pd.merge(df_call, df_put, on='Strike', how='outer').fillna(0).sort_values('Strike')
    
    total_put_money = data['Put_Amt'].sum()
    total_call_money = data['Call_Amt'].sum()
    
    data = data[(data['Call_OI'] > 300) | (data['Put_OI'] > 300)]
    FOCUS_RANGE = 1200 
    center_price = spot_price if (spot_price and spot_price > 0) else (data.loc[data['Put_OI'].idxmax(), 'Strike'] if not data.empty else 0)
    
    if center_price > 0:
        min_s = center_price - FOCUS_RANGE
        max_s = center_price + FOCUS_RANGE
        data = data[(data['Strike'] >= min_s) & (data['Strike'] <= max_s)]
    
    max_oi = max(data['Put_OI'].max(), data['Call_OI'].max()) if not data.empty else 1000
    x_limit = max_oi * 1.1

    fig = go.Figure()
    fig.add_trace(go.Bar(y=data['Strike'], x=-data['Put_OI'], orientation='h', name='Put (支撐)', marker_color='#2ca02c', opacity=0.85, customdata=data['Put_Amt'] / 100000000, hovertemplate='<b>履約價: %{y}</b><br>Put OI: %{x} 口<br>Put 市值: %{customdata:.2f}億<extra></extra>'))
    fig.add_trace(go.Bar(y=data['Strike'], x=data['Call_OI'], orientation='h', name='Call (壓力)', marker_color='#d62728', opacity=0.85, customdata=data['Call_Amt'] / 100000000, hovertemplate='<b>履約價: %{y}</b><br>Call OI: %{x} 口<br>Call 市值: %{customdata:.2f}億<extra></extra>'))

    annotations = []
    if spot_price and spot_price > 0:
        if not data.empty and data['Strike'].min() <= spot_price <= data['Strike'].max():
            fig.add_hline(y=spot_price, line_dash="dash", line_color="#ff7f0e", line_width=2)
            annotations.append(dict(x=1, y=spot_price, xref="paper", yref="y", text=f" 現貨 {int(spot_price)} ", showarrow=False, xanchor="left", align="center", font=dict(color="white", size=12), bgcolor="#ff7f0e", bordercolor="#ff7f0e", borderpad=4))

    annotations.append(dict(x=0.02, y=1.05, xref="paper", yref="paper", text=f"<b>Put 總金額</b><br>{total_put_money/100000000:.1f} 億", showarrow=False, align="left", font=dict(size=14, color="#2ca02c"), bgcolor="white", bordercolor="#2ca02c", borderwidth=2, borderpad=6))
    annotations.append(dict(x=0.98, y=1.05, xref="paper", yref="paper", text=f"<b>Call 總金額</b><br>{total_call_money/100000000:.1f} 億", showarrow=False, align="right", font=dict(size=14, color="#d62728"), bgcolor="white", bordercolor="#d62728", borderwidth=2, borderpad=6))

    # Margin Top 加大，避免標題重疊
    fig.update_layout(title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='top', font=dict(size=20, color="black")), xaxis=dict(title='未平倉量 (OI)', range=[-x_limit, x_limit], showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='black', tickmode='array', tickvals=[-x_limit*0.75, -x_limit*0.5, -x_limit*0.25, 0, x_limit*0.25, x_limit*0.5, x_limit*0.75], ticktext=[f"{int(x_limit*0.75)}", f"{int(x_limit*0.5)}", f"{int(x_limit*0.25)}", "0", f"{int(x_limit*0.25)}", f"{int(x_limit*0.5)}", f"{int(x_limit*0.75)}"]), yaxis=dict(title='履約價', tickmode='linear', dtick=100, tickformat='d'), barmode='overlay', legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"), height=750, margin=dict(l=40, r=80, t=140, b=60), annotations=annotations, paper_bgcolor='white', plot_bgcolor='white')
    return fig

# --- 資料準備函式 (供 AI 使用) ---
def prepare_ai_data(df):
    df_ai = df.copy()
    if 'Amount' in df_ai.columns:
        df_ai = df_ai.nlargest(15, 'Amount') # 瘦身：只取前15大
    keep = ['Strike', 'Type', 'OI', 'Amount']
    df_ai = df_ai[keep]
    return df_ai.to_csv(index=False)

# --- helper：從 df 與 data_date 找出接下來要畫的合約（與你原本邏輯一致） ---
def get_next_contracts(df, data_date):
    unique_codes = df['Month'].unique()
    all_contracts = []
    for code in unique_codes:
        s_date_str = get_settlement_date(code)
        if s_date_str == "9999/99/99" or s_date_str <= data_date: continue
        all_contracts.append({'code': code, 'date': s_date_str})
    all_contracts.sort(key=lambda x: x['date'])
    
    plot_targets = []
    if all_contracts:
        nearest = all_contracts[0]
        plot_targets.append({'title': '最近結算', 'info': nearest})
        monthly = next((c for c in all_contracts if len(c['code']) == 6), None)
        if monthly:
            if monthly['code'] != nearest['code']: plot_targets.append({'title': '當月月選', 'info': monthly})
            else: plot_targets[0]['title'] = '最近結算 (同月選)'
    return plot_targets

# --- 統一 prompt 建構器（Gemini / ChatGPT 共用） ---
def build_ai_prompt(data_str, taiex_price, contract_info, data_date):
    """
    contract_info: {'code':..., 'date':...} or None
    data_date: string like '2025/12/08'
    """
    contract_note = "無法判斷要結算的合約資訊" 
    if contract_info:
        contract_note = f"系統判斷下一個即將結算合約為：{contract_info.get('code')}，結算日：{contract_info.get('date')}。"
    prompt = f"""
你是一位專業的台指期 / 選擇權交易員助理。注意：本 prompt 的資料由系統端【已經判斷並過濾】為「下一個即將結算的合約」資料（包含週選與月選判斷），**請勿重新推斷或更改結算日**。若你發現資料日期與系統標注的結算日不一致，請直接回報「資料日期異常」而非自行假設。

系統指示：
1) 本資料來源日期（期交所頁面日期）: {data_date}
2) 大盤現貨：{taiex_price}
3) {contract_note}
4) 你會收到 CSV（前15大籌碼）：請以該 CSV 做分析，不要重新判斷結算日或挑出別的月份。若 CSV 包含多個月份/週別，請以上面系統標記的合約為第一優先。
5) 輸出規則（一定要遵守）：
   - 只給出結論：**偏多 / 偏空 / 震盪**（一行）
   - 接著 30~80 字的簡短理由（條列式或一句話）
   - 不要輸出過程計算
   - 若你懷疑資料不是「尚未結算」的最新資料，回答要以「⚠️ 資料日期異常」為開頭

下面是 CSV（前15大），格式：Strike,Type,OI,Amount
{data_str}
"""
    return prompt.strip()

# --- AI 分析 (Gemini) ---
def ask_gemini(prompt_text):
    if not gemini_model: return "⚠️ 未設定 Gemini Key"
    try:
        # Gemini: 使用 generate_content，直接拿 text
        res = gemini_model.generate_content(prompt_text)
        # Some Gemini SDKs return object with .text, or .candidates[0].content - handle both
        if hasattr(res, "text"):
            return res.text
        if hasattr(res, "candidates") and len(res.candidates) > 0:
            return getattr(res.candidates[0], "content", str(res.candidates[0]))
        return str(res)
    except Exception as e:
        return f"Gemini 錯誤: {str(e)}"

# --- AI 分析 (ChatGPT - 使用 gpt-4o-mini) ---
def ask_chatgpt(prompt_text):
    if not openai_client: return "⚠️ 未設定 OpenAI Key"
    try:
        # Using the same chat.completions.create pattern you had, with chosen model gpt-4o-mini
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional trader."},
                {"role": "user", "content": prompt_text}
            ],
            # optional: 可以設定 max_tokens, temperature 等
            # max_tokens=200,
            # temperature=0.0
        )
        # 對於不同 SDK 回傳格式，盡量穩健取值
        try:
            return response.choices[0].message.content
        except:
            try:
                return response.choices[0].message['content']
            except:
                return str(response)
    except Exception as e:
        error_msg = str(e)
        # --- 防呆判斷 ---
        if "insufficient_quota" in error_msg:
            return "⚠️ OpenAI 額度不足 (請至官網儲值)"
        elif "429" in error_msg:
            return "⚠️ 請求過於頻繁 (請稍後再試)"
        else:
            return f"ChatGPT 錯誤: {error_msg}"

# --- 主程式 ---
def main():
    st.title("🤖 台指期籌碼戰情室 (雙 AI 對決版)")
    
    col_title, col_btn = st.columns([3, 1])
    
    if st.sidebar.button("🔄 重新整理"): st.cache_data.clear(); st.rerun()

    # 顯示 AI 狀態
    st.sidebar.markdown("---")
    st.sidebar.markdown("**AI 連線狀態:**")
    st.sidebar.caption(f"🔵 Gemini ({gemini_name}): {'✅' if gemini_model else '❌'}")
    st.sidebar.caption(f"🟢 ChatGPT: {'✅' if openai_client else '❌'}")

    with st.spinner('連線期交所中...'):
        df, data_date = get_option_data()
        taiex_now = get_realtime_data()

    if df is None: st.error("查無資料"); return

    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.sidebar.download_button("📥 下載完整數據", csv, f"option_{data_date.replace('/','')}.csv", "text/csv")

    # --- 預先計算接下來要使用的合約（同你原本邏輯） ---
    plot_targets = get_next_contracts(df, data_date)

    # --- 雙 AI 分析區 ---
    st.markdown("### 💡 AI 觀點對決")
    if st.button("✨ 啟動 AI 雙重分析", type="primary"):
        if not gemini_model and not openai_client:
            st.error("請至少設定一個 API Key")
        else:
            # 以你原本的 prepare_ai_data 準備 csv（前15筆）
            data_str = prepare_ai_data(df)
            # 選擇要給 AI 的合約資訊（若有多個 plot_targets，就取第一個）
            contract_info = plot_targets[0]['info'] if plot_targets else None

            prompt_text = build_ai_prompt(data_str, taiex_now, contract_info, data_date)
            
            # 建立左右兩欄
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔵 Google Gemini")
                if gemini_model:
                    with st.spinner("Gemini 分析中..."):
                        res_gemini = ask_gemini(prompt_text)
                        st.info(res_gemini)
                else:
                    st.warning("未設定 Gemini Key")
            
            with col2:
                st.subheader("🟢 OpenAI ChatGPT")
                if openai_client:
                    with st.spinner("ChatGPT 分析中..."):
                        res_chatgpt = ask_chatgpt(prompt_text)
                        # 如果是額度不足警告，顯示黃色；正常則顯示綠色
                        if "⚠️" in res_chatgpt:
                             st.warning(res_chatgpt)
                        else:
                             st.success(res_chatgpt)
                else:
                    st.warning("未設定 OpenAI Key")

    # 數據指標與圖表
    total_call_amt = df[df['Type'].str.contains('買|Call', case=False, na=False)]['Amount'].sum()
    total_put_amt = df[df['Type'].str.contains('賣|Put', case=False, na=False)]['Amount'].sum()
    pc_ratio_amt = (total_put_amt / total_call_amt) * 100 if total_call_amt > 0 else 0

    c1, c2, c3, c4 = st.columns([1.2, 0.8, 1, 1])
    c1.markdown(f"<div style='text-align: left;'><span style='font-size: 14px; color: #555;'>製圖時間</span><br><span style='font-size: 18px; font-weight: bold;'>{datetime.now(tz=TW_TZ).strftime('%Y/%m/%d %H:%M:%S')}</span></div>", unsafe_allow_html=True)
    c2.metric("大盤現貨", f"{int(taiex_now) if taiex_now else 'N/A'}")
    trend = "偏多" if pc_ratio_amt > 100 else "偏空"
    c3.metric("全市場 P/C 金額比", f"{pc_ratio_amt:.1f}%", f"{trend}格局", delta_color="normal" if pc_ratio_amt > 100 else "inverse")
    c4.metric("資料來源日期", data_date)
    st.markdown("---")

    # 若 plot_targets 原本是要顯示的合約，依舊照原本畫圖
    cols = st.columns(len(plot_targets)) if plot_targets else []
    for i, target in enumerate(plot_targets):
        with cols[i]:
            m_code = target['info']['code']
            s_date = target['info']['date']
            df_target = df[df['Month'] == m_code]
            sub_call = df_target[df_target['Type'].str.contains('Call|買', case=False, na=False)]['Amount'].sum()
            sub_put = df_target[df_target['Type'].str.contains('Put|賣', case=False, na=False)]['Amount'].sum()
            sub_ratio = (sub_put / sub_call * 100) if sub_call > 0 else 0
            title_text = f"<b> {m_code}</b><br><span style='font-size: 14px;'>結算: {s_date}</span><br><span style='font-size: 14px;'>P/C金額比: {sub_ratio:.1f}% ({'偏多' if sub_ratio > 100 else '偏空'})</span>"
            st.plotly_chart(plot_tornado_chart(df_target, title_text, taiex_now), use_container_width=True)

if __name__ == "__main__":
    main()
