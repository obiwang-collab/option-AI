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

# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="台指期籌碼戰情室 (AI 決策版)")
TW_TZ = timezone(timedelta(hours=8)) 

# ==========================================
# 🔑 金鑰設定區 (雲端安全版)
# ==========================================
# 在本地執行時，如果沒有設定 secrets，會嘗試讀取這裡的變數
# 但強烈建議在 Streamlit Cloud 後台設定 Secrets
try:
    # 程式碼會優先從 Streamlit Cloud 的 Secrets 讀取金鑰
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    # 如果你是要在本地測試，請暫時將 Key 貼在下方引號中
    # 上傳到 GitHub 前請務必刪除，以免外洩！
    API_KEY = "請輸入你的API_KEY"

# --- 智慧模型設定 ---
def configure_gemini(api_key):
    if not api_key or "請輸入" in api_key:
        return None, "尚未設定 API Key"
    
    genai.configure(api_key=api_key)
    try:
        # 自動偵測可用模型
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 修正的模型優先順序: 2.5 Flash -> 1.5 Flash -> Pro
        # 這樣可以避免舊模型名稱找不到的錯誤 (404 Error)
        for target in ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']:
            for m in available_models:
                if target in m: return genai.GenerativeModel(m), m
        
        # 兜底
        if available_models: return genai.GenerativeModel(available_models[0]), available_models[0]
        return None, "無可用模型"
    except Exception as e:
        # 如果是連線錯誤或金鑰錯誤，會顯示在這裡
        return None, f"連線錯誤: {str(e)}"

# 初始化模型
model, model_name = configure_gemini(API_KEY)

# 手動修正結算日 (確保程式處理正確)
MANUAL_SETTLEMENT_FIX = {
    '202501W1': '2025/01/02', 
}

# --- 核心函式 ---
def get_settlement_date(contract_code):
    """計算台指選結算日，主要針對周選和月選"""
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
        if 'W' in code: # 周選，每週三
            match = re.search(r'W(\d)', code)
            if match:
                week_num = int(match.group(1))
                if len(wednesdays) >= week_num: day = wednesdays[week_num - 1]
        elif 'F' in code: # 某些特殊周五結算，雖然 TXO 不常見
            match = re.search(r'F(\d)', code)
            if match:
                week_num = int(match.group(1))
                if len(fridays) >= week_num: day = fridays[week_num - 1]
        else: # 月選，第三個周三
            if len(wednesdays) >= 3: day = wednesdays[2]
        if day: return f"{year}/{month:02d}/{day:02d}"
        else: return "9999/99/99"
    except: return "9999/99/99"

@st.cache_data(ttl=60)
def get_realtime_data():
    """取得台指現貨即時報價"""
    taiex = None
    ts = int(time.time())
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        # 嘗試從證交所 API 取得
        url = f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_t00.tw&json=1&delay=0&_={ts}000"
        res = requests.get(url, timeout=2)
        data = res.json()
        if 'msgArray' in data and len(data['msgArray']) > 0:
            val = data['msgArray'][0].get('z', '-') # z: 最新成交價
            if val == '-': val = data['msgArray'][0].get('o', '-') # o: 開盤價
            if val != '-': taiex = float(val)
    except: pass
    if taiex is None:
        try:
            # 嘗試從 Yahoo 取得 (備援)
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII?interval=1m&range=1d&_={ts}"
            res = requests.get(url, headers=headers, timeout=3)
            data = res.json()
            price = data['chart']['result'][0]['meta'].get('regularMarketPrice')
            if price: taiex = float(price)
        except: pass
    return taiex

@st.cache_data(ttl=300)
def get_option_data():
    """從期交所取得最近一期選擇權每日交易資訊"""
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    headers = {'User-Agent': 'Mozilla/5.0'}
    # 嘗試抓取最近五天的數據，以防今天還沒資料或剛結算
    for i in range(5):
        query_date = (datetime.now(tz=TW_TZ) - timedelta(days=i)).strftime('%Y/%m/%d')
        payload = {'queryType': '2', 'marketCode': '0', 'dateaddcnt': '', 'commodity_id': 'TXO', 'commodity_id2': '', 'queryDate': query_date, 'MarketCode': '0', 'commodity_idt': 'TXO'}
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=5)
            # 檢查是否查無資料或內容太少
            if "查無資料" in res.text or len(res.text) < 500: continue 
            dfs = pd.read_html(StringIO(res.text))
            df = dfs[0]
            
            # --- 數據清理與標準化 ---
            df.columns = [str(c).replace(' ', '').replace('*', '').replace('契約', '').strip() for c in df.columns]
            
            # 智能匹配欄位名稱
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
            
            # 轉換數據類型
            df = df.dropna(subset=['Type'])
            df['Type'] = df['Type'].astype(str).str.strip()
            df['Strike'] = pd.to_numeric(df['Strike'].astype(str).str.replace(',', ''), errors='coerce')
            df['OI'] = pd.to_numeric(df['OI'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df['Price'] = df['Price'].astype(str).str.replace(',', '').replace('-', '0')
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
            if 'Volume' in df.columns: df['Volume'] = pd.to_numeric(df['Volume'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
            # 計算未平倉合約市值 (金額 = OI * 價格 * 50)
            df['Amount'] = df['OI'] * df['Price'] * 50
            
            if df['OI'].sum() == 0: continue # 如果沒有任何 OI，則跳過
            return df, query_date
        except: continue 
    return None, None

def plot_tornado_chart(df_target, title_text, spot_price):
    """繪製 Call/Put OI 龍捲風圖"""
    is_call = df_target['Type'].str.contains('買|Call', case=False, na=False)
    df_call = df_target[is_call][['Strike', 'OI', 'Amount']].rename(columns={'OI': 'Call_OI', 'Amount': 'Call_Amt'})
    df_put = df_target[~is_call][['Strike', 'OI', 'Amount']].rename(columns={'OI': 'Put_OI', 'Amount': 'Put_Amt'})
    data = pd.merge(df_call, df_put, on='Strike', how='outer').fillna(0).sort_values('Strike')
    
    # 計算總金額
    total_put_money = data['Put_Amt'].sum()
    total_call_money = data['Call_Amt'].sum()
    
    # 篩選掉 OI 過小的履約價，聚焦主要籌碼區
    data = data[(data['Call_OI'] > 300) | (data['Put_OI'] > 300)]
    
    # 聚焦範圍 (以現貨價為中心，上下各 1200 點)
    FOCUS_RANGE = 1200 
    center_price = spot_price if (spot_price and spot_price > 0) else (data.loc[data['Put_OI'].idxmax(), 'Strike'] if not data.empty else 0)
    
    if center_price > 0:
        min_s = center_price - FOCUS_RANGE
        max_s = center_price + FOCUS_RANGE
        data = data[(data['Strike'] >= min_s) & (data['Strike'] <= max_s)]
    
    # 決定 X 軸最大值
    max_oi = max(data['Put_OI'].max(), data['Call_OI'].max()) if not data.empty else 1000
    x_limit = max_oi * 1.1

    fig = go.Figure()
    # 繪製 Put (支撐) - 往左邊延伸 (負值)
    fig.add_trace(go.Bar(y=data['Strike'], x=-data['Put_OI'], orientation='h', name='Put (支撐)', marker_color='#2ca02c', opacity=0.85, customdata=data['Put_Amt'] / 100000000, hovertemplate='<b>履約價: %{y}</b><br>Put OI: %{x} 口<br>Put 市值: %{customdata:.2f}億<extra></extra>'))
    # 繪製 Call (壓力) - 往右邊延伸 (正值)
    fig.add_trace(go.Bar(y=data['Strike'], x=data['Call_OI'], orientation='h', name='Call (壓力)', marker_color='#d62728', opacity=0.85, customdata=data['Call_Amt'] / 100000000, hovertemplate='<b>履約價: %{y}</b><br>Call OI: %{x} 口<br>Call 市值: %{customdata:.2f}億<extra></extra>'))

    annotations = []
    # 標記現貨價格
    if spot_price and spot_price > 0:
        if not data.empty and data['Strike'].min() <= spot_price <= data['Strike'].max():
            fig.add_hline(y=spot_price, line_dash="dash", line_color="#ff7f0e", line_width=2)
            annotations.append(dict(x=1, y=spot_price, xref="paper", yref="y", text=f" 現貨 {int(spot_price)} ", showarrow=False, xanchor="left", align="center", font=dict(color="white", size=12), bgcolor="#ff7f0e", bordercolor="#ff7f0e", borderpad=4))

    # 標記 Put/Call 總金額
    annotations.append(dict(x=0.02, y=1.05, xref="paper", yref="paper", text=f"<b>Put 總金額</b><br>{total_put_money/100000000:.1f} 億", showarrow=False, align="left", font=dict(size=14, color="#2ca02c"), bgcolor="white", bordercolor="#2ca02c", borderwidth=2, borderpad=6))
    annotations.append(dict(x=0.98, y=1.05, xref="paper", yref="paper", text=f"<b>Call 總金額</b><br>{total_call_money/100000000:.1f} 億", showarrow=False, align="right", font=dict(size=14, color="#d62728"), bgcolor="white", bordercolor="#d62728", borderwidth=2, borderpad=6))

    fig.update_layout(
        title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='top', font=dict(size=20, color="black")),
        xaxis=dict(title='未平倉量 (OI)', range=[-x_limit, x_limit], showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='black', 
                   # 調整 X 軸刻度顯示為正值
                   tickmode='array', 
                   tickvals=[-x_limit*0.75, -x_limit*0.5, -x_limit*0.25, 0, x_limit*0.25, x_limit*0.5, x_limit*0.75], 
                   ticktext=[f"{int(x_limit*0.75)}", f"{int(x_limit*0.5)}", f"{int(x_limit*0.25)}", "0", f"{int(x_limit*0.25)}", f"{int(x_limit*0.5)}", f"{int(x_limit*0.75)}"]), 
        yaxis=dict(title='履約價', tickmode='linear', dtick=100, tickformat='d'), 
        barmode='overlay', 
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"), 
        height=750, 
        margin=dict(l=40, r=80, t=140, b=60), 
        annotations=annotations, 
        paper_bgcolor='white', 
        plot_bgcolor='white')
    return fig

# --- AI 分析函式 (極簡版) ---
def ask_gemini_brief(df, taiex_price):
    if not model:
        # 顯示連線錯誤而非單純的未設定 Key
        return f"⚠️ AI 服務連線失敗。原因: {model_name}"
    
    try:
        # 只取 OI 或 Amount 較大的前 40 筆資料，節省運算
        df_ai = df.copy()
        if 'Amount' in df_ai.columns:
            # 優先以 Amount 排序，更能代表大戶的成本和意圖
            df_ai = df_ai.nlargest(40, 'Amount') 
        
        data_str = df_ai.to_csv(index=False)
        
        # --- 關鍵提示詞 (Prompt) ---
        prompt = f"""
        你是一個台指期貨交易助手。
        現在大盤現貨價格：{taiex_price}。
        
        請分析這份選擇權籌碼 (CSV)，並直接給出【短線操作建議】。
        
        規則：
        1. **不要** 解釋你的分析過程 (不要提P/C ratio、不要提支撐壓力位怎麼算)。
        2. **直接告訴我結論**：市場目前是偏多、偏空、還是震盪？
        3. **給出具體建議**：例如「拉回找買點」、「反彈空」、「區間操作」等。
        4. 字數控制在 100 字以內，語氣簡潔有力。

        數據：
        {data_str}
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"分析忙碌中或請求超時 ({str(e)})"

# --- 主程式 ---
def main():
    st.title("🤖 台指期籌碼戰情室 (AI 決策版)")
    
    # 側邊欄重新整理按鈕
    if st.sidebar.button("🔄 重新整理"): st.cache_data.clear(); st.rerun()

    with st.spinner('連線期交所中...'):
        df, data_date = get_option_data()
        taiex_now = get_realtime_data()

    if df is None: st.error("查無資料，請稍後再試。"); return

    # --- 下載按鈕 (側邊欄) ---
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.sidebar.download_button("📥 下載完整數據", csv, f"option_{data_date.replace('/','')}.csv", "text/csv")
    
    # 數據指標計算
    total_call_amt = df[df['Type'].str.contains('買|Call', case=False, na=False)]['Amount'].sum()
    total_put_amt = df[df['Type'].str.contains('賣|Put', case=False, na=False)]['Amount'].sum()
    pc_ratio_amt = (total_put_amt / total_call_amt) * 100 if total_call_amt > 0 else 0

    # --- AI 極簡分析顯示 ---
    st.markdown("### 💡 AI 短線錦囊")
    if model:
        if st.button("✨ 取得操作建議", type="primary"):
            with st.spinner("AI 正在擬定策略..."):
                advice = ask_gemini_brief(df, taiex_now)
            st.info(advice)
    else:
        # 如果模型初始化失敗，顯示錯誤原因
        st.error(f"⚠️ AI 服務無法啟動。原因: {model_name}。請檢查 Streamlit Secrets 中的 API Key 是否設定正確且有效。")


    c1, c2, c3, c4 = st.columns([1.2, 0.8, 1, 1])
    c1.markdown(f"<div style='text-align: left;'><span style='font-size: 14px; color: #555;'>製圖時間</span><br><span style='font-size: 18px; font-weight: bold;'>{datetime.now(tz=TW_TZ).strftime('%Y/%m/%d %H:%M:%S')}</span></div>", unsafe_allow_html=True)
    c2.metric("大盤現貨", f"{int(taiex_now) if taiex_now else 'N/A'}")
    trend = "偏多" if pc_ratio_amt > 100 else "偏空"
    c3.metric("全市場 P/C 金額比", f"{pc_ratio_amt:.1f}%", f"{trend}格局", delta_color="normal" if pc_ratio_amt > 100 else "inverse")
    c4.metric("資料來源日期", data_date)
    st.markdown("---")

    # 繪圖準備
    unique_codes = df['Month'].unique()
    all_contracts = []
    for code in unique_codes:
        s_date_str = get_settlement_date(code)
        # 排除已結算或無法判斷結算日的合約
        if s_date_str == "9999/99/99" or s_date_str <= data_date: continue
        all_contracts.append({'code': code, 'date': s_date_str})
    all_contracts.sort(key=lambda x: x['date']) # 按結算日排序
    
    # 決定要繪製的圖表：最近結算合約，以及最近的月選合約
    plot_targets = []
    if all_contracts:
        nearest = all_contracts[0]
        plot_targets.append({'title': '最近結算', 'info': nearest})
        
        # 尋找最近的月選合約 (代碼長度為 6)
        monthly = next((c for c in all_contracts if len(c['code']) == 6), None)
        if monthly:
            if monthly['code'] != nearest['code']: plot_targets.append({'title': '當月月選', 'info': monthly})
            else: plot_targets[0]['title'] = '最近結算 (同月選)'

    # 繪圖區
    cols = st.columns(len(plot_targets))
    for i, target in enumerate(plot_targets):
        with cols[i]:
            m_code = target['info']['code']
            s_date = target['info']['date']
            df_target = df[df['Month'] == m_code]
            
            # 計算單一合約的 P/C 金額比
            sub_call = df_target[df_target['Type'].str.contains('Call|買', case=False, na=False)]['Amount'].sum()
            sub_put = df_target[df_target['Type'].str.contains('Put|賣', case=False, na=False)]['Amount'].sum()
            sub_ratio = (sub_put / sub_call * 100) if sub_call > 0 else 0
            
            title_text = f"<b>【{target['title']}】 {m_code}</b><br><span style='font-size: 14px;'>結算: {s_date}</span><br><span style='font-size: 14px;'>P/C金額比: {sub_ratio:.1f}% ({'偏多' if sub_ratio > 100 else '偏空'})</span>"
            st.plotly_chart(plot_tornado_chart(df_target, title_text, taiex_now), use_container_width=True)

if __name__ == "__main__":
    main()
