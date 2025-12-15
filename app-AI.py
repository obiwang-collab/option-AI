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
from concurrent.futures import ThreadPoolExecutor
import streamlit.components.v1 as components 

# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="台指期籌碼戰情室 (莊家控盤版)")
TW_TZ = timezone(timedelta(hours=8))

# ==========================================
# 🔑 金鑰設定區
# ==========================================
try:
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
except FileNotFoundError:
    GEMINI_KEY = ""
    OPENAI_KEY = ""

# --- 🧠 1. Gemini 模型設定 ---
def get_gemini_model(api_key):
    if not api_key: return None, "未設定"
    genai.configure(api_key=api_key)
    try:
        target_model_name = 'gemini-1.5-flash'
        return genai.GenerativeModel(target_model_name), target_model_name
    except Exception as e:
        try:
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            for target in ['flash', 'gemini-1.5-pro']:
                for m in models:
                    if target in m.lower(): return genai.GenerativeModel(m), m
            return (genai.GenerativeModel(models[0]), models[0]) if models else (None, "無可用模型")
        except Exception as e2:
            return None, f"模型設定錯誤: {str(e)}"

# --- 🧠 2. ChatGPT 模型設定 ---
def get_openai_client(api_key):
    if not api_key: return None
    return OpenAI(api_key=api_key)

# 初始化模型
gemini_model, gemini_name = get_gemini_model(GEMINI_KEY)
openai_client = get_openai_client(OPENAI_KEY)

MANUAL_SETTLEMENT_FIX = {'202501W1': '2025/01/02'}


# ⭐⭐⭐ AdSense 整合代碼區塊 ⭐⭐⭐
ADSENSE_PUB_ID = 'ca-pub-4585150092118682'

ADSENSE_AUTO_ADS_FULL = f"""
<!DOCTYPE html>
<html>
<head>
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={ADSENSE_PUB_ID}"
         crossorigin="anonymous"></script>
</head>
<body>
    <div style="min-height: 1px;"></div>
</body>
</html>
"""

def get_display_ad_code(ad_slot_id):
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={ADSENSE_PUB_ID}"
             crossorigin="anonymous"></script>
    </head>
    <body>
        <ins class="adsbygoogle"
             style="display:block"
             data-ad-client="{ADSENSE_PUB_ID}"
             data-ad-slot="{ad_slot_id}"
             data-ad-format="auto"
             data-full-width-responsive="true"></ins>
        <script>
             (adsbygoogle = window.adsbygoogle || []).push({{}});
        </script>
    </body>
    </html>
    """

def inject_adsense_head():
    st.markdown(f"""
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={ADSENSE_PUB_ID}"
         crossorigin="anonymous"></script>
    """, unsafe_allow_html=True)
    components.html(ADSENSE_AUTO_ADS_FULL, height=1, scrolling=False)

def show_ad_placeholder():
    st.markdown(f"""
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={ADSENSE_PUB_ID}"
         crossorigin="anonymous"></script>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 40px 20px; border-radius: 8px; text-align: center;
                border: 2px dashed #dee2e6; min-height: 250px;
                display: flex; align-items: center; justify-content: center;'>
        <div style='max-width: 400px;'>
            <svg width="60" height="60" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" 
                 style="margin: 0 auto 15px; opacity: 0.3; display: block;">
                <rect x="3" y="3" width="18" height="18" rx="2" stroke="#6c757d" stroke-width="2"/>
                <path d="M3 9h18M9 3v18" stroke="#6c757d" stroke-width="2"/>
            </svg>
            <p style='color: #6c757d; font-size: 16px; font-weight: 600; margin: 10px 0 5px 0;'>廣告位置</p>
            <p style='color: #adb5bd; font-size: 13px; margin: 0;'>Google AdSense 審核通過後將顯示廣告</p>
            <p style='color: #adb5bd; font-size: 11px; margin-top: 10px;'>Publisher ID: """ + ADSENSE_PUB_ID + """</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------


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

# --- 🆕 三大法人選擇權籌碼數據獲取 ---
@st.cache_data(ttl=300)
def get_institutional_option_data():
    """獲取三大法人選擇權籌碼數據（最近兩天）"""
    url = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
    headers = {'User-Agent': 'Mozilla/5.0'}
    all_inst_data = []
    
    for i in range(10):
        query_date = (datetime.now(tz=TW_TZ) - timedelta(days=i)).strftime('%Y/%m/%d')
        payload = {
            'down_type': '1',
            'queryStartDate': query_date,
            'queryEndDate': query_date,
            'commodity_id': 'TXO'
        }
        
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=5)
            res.encoding = 'utf-8'
            
            if "查無資料" in res.text or len(res.text) < 500:
                continue
            
            dfs = pd.read_html(StringIO(res.text))
            
            if not dfs or len(dfs) == 0:
                continue
            
            df = dfs[0]
            
            # 清理欄位名稱
            df.columns = [str(c).strip().replace(' ', '').replace('\n', '') for c in df.columns]
            
            # 尋找自營商、投信、外資的列
            df_filtered = df[df.iloc[:, 0].astype(str).str.contains('自營商|投信|外資', na=False)]
            
            if df_filtered.empty:
                continue
            
            all_inst_data.append({'date': query_date, 'df': df_filtered})
            
            if len(all_inst_data) >= 2:
                break
                
        except Exception as e:
            continue
    
    if len(all_inst_data) < 2:
        return None, None, None, None
    
    return all_inst_data[0]['df'], all_inst_data[0]['date'], all_inst_data[1]['df'], all_inst_data[1]['date']

# --- 修正後的資料獲取函式：獲取兩天數據 ---
@st.cache_data(ttl=300)
def get_option_data():
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    headers = {'User-Agent': 'Mozilla/5.0'}
    all_data = []

    for i in range(10): 
        query_date = (datetime.now(tz=TW_TZ) - timedelta(days=i)).strftime('%Y/%m/%d')
        payload = {'queryType': '2', 'marketCode': '0', 'dateaddcnt': '', 'commodity_id': 'TXO', 'commodity_id2': '', 'queryDate': query_date, 'MarketCode': '0', 'commodity_idt': 'TXO'}
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=5)
            res.encoding = 'utf-8' 
            if "查無資料" in res.text or len(res.text) < 500: continue 
            dfs = pd.read_html(StringIO(res.text))
            df = dfs[0]
            
            df.columns = [str(c).replace(' ', '').replace('*', '').replace('契約', '').strip() for c in df.columns]
            month_col = next((c for c in df.columns if '月' in c or '週' in c), None)
            strike_col = next((c for c in df.columns if '履約' in c), None)
            type_col = next((c for c in df.columns if '買賣' in c), None)
            oi_col = next((c for c in df.columns if '未沖銷' in c or 'OI' in c), None)
            price_col = next((c for c in df.columns if '結算' in c or '收盤' in c or 'Price' in c), None)
            
            if not all([month_col, strike_col, type_col, oi_col, price_col]): continue
            
            df = df.rename(columns={month_col:'Month', strike_col:'Strike', type_col:'Type', oi_col:'OI', price_col:'Price'})
            df = df[['Month', 'Strike', 'Type', 'OI', 'Price']].copy()
            
            df = df.dropna(subset=['Type'])
            df['Type'] = df['Type'].astype(str).str.strip()
            df['Strike'] = pd.to_numeric(df['Strike'].astype(str).str.replace(',', ''), errors='coerce')
            df['OI'] = pd.to_numeric(df['OI'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace(',', '').replace('-', '0'), errors='coerce').fillna(0)
            df['Amount'] = df['OI'] * df['Price'] * 50
            
            if df['OI'].sum() == 0: continue 

            all_data.append({'date': query_date, 'df': df})
            
            if len(all_data) >= 2: break
        except: continue 
    
    if len(all_data) < 2: 
        return None, None, None, None 

    df_today = all_data[0]['df']
    date_today = all_data[0]['date']
    df_yesterday = all_data[1]['df']
    date_yesterday = all_data[1]['date']
    
    return df_today, date_today, df_yesterday, date_yesterday

# --- 新增差異計算函式 ---
def calculate_dod_change(df_today, df_yesterday):
    """計算未平倉量 (OI) 的日差異"""
    
    df_today = df_today[['Month', 'Strike', 'Type', 'OI', 'Amount']].copy()
    df_yesterday = df_yesterday[['Month', 'Strike', 'Type', 'OI']].copy()

    df_yesterday = df_yesterday.rename(columns={'OI': 'Prev_OI'})
    
    df_merged = pd.merge(df_today, df_yesterday, on=['Month', 'Strike', 'Type'], how='left').fillna(0)
    
    df_merged['OI_Change'] = df_merged['OI'] - df_merged['Prev_OI']
    
    return df_merged

# --- 修正圖表函式：顯示差異口數 ---
def plot_tornado_chart(df_target, title_text, spot_price):
    is_call = df_target['Type'].str.contains('買|Call', case=False, na=False)
    
    df_call = df_target[is_call][['Strike', 'OI', 'Amount', 'OI_Change']].rename(columns={'OI': 'Call_OI', 'Amount': 'Call_Amt', 'OI_Change': 'Call_OI_Change'})
    df_put = df_target[~is_call][['Strike', 'OI', 'Amount', 'OI_Change']].rename(columns={'OI': 'Put_OI', 'Amount': 'Put_Amt', 'OI_Change': 'Put_OI_Change'})
    
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

    data['Put_Text'] = data.apply(lambda row: f"{'+' if row['Put_OI_Change'] > 0 else ''}{int(row['Put_OI_Change'])}" if row['Put_OI'] > 300 else "", axis=1)
    data['Call_Text'] = data.apply(lambda row: f"{'+' if row['Call_OI_Change'] > 0 else ''}{int(row['Call_OI_Change'])}" if row['Call_OI'] > 300 else "", axis=1)

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=data['Strike'], 
        x=-data['Put_OI'], 
        orientation='h', 
        name='Put (支撐)', 
        marker_color='#2ca02c', 
        opacity=0.85, 
        customdata=data['Put_Amt'] / 100000000, 
        hovertemplate='<b>履約價: %{y}</b><br>Put OI: %{x} 口<br>Put 變化: %{text} 口<br>Put 市值: %{customdata:.2f}億<extra></extra>',
        text=data['Put_Text'],       
        textposition='outside',      
        cliponaxis=False             
    ))
    
    fig.add_trace(go.Bar(
        y=data['Strike'], 
        x=data['Call_OI'], 
        orientation='h', 
        name='Call (壓力)', 
        marker_color='#d62728', 
        opacity=0.85, 
        customdata=data['Call_Amt'] / 100000000, 
        hovertemplate='<b>履約價: %{y}</b><br>Call OI: %{x} 口<br>Call 變化: %{text} 口<br>Call 市值: %{customdata:.2f}億<extra></extra>',
        text=data['Call_Text'],      
        textposition='outside',      
        cliponaxis=False
    ))

    annotations = []
    if spot_price and spot_price > 0:
        if not data.empty and data['Strike'].min() <= spot_price <= data['Strike'].max():
            fig.add_hline(y=spot_price, line_dash="dash", line_color="#ff7f0e", line_width=2)
            annotations.append(dict(x=1.05, y=spot_price, xref="paper", yref="y", text=f" 現貨 {int(spot_price)} ", showarrow=False, xanchor="left", align="center", font=dict(color="white", size=12), bgcolor="#ff7f0e", bordercolor="#ff7f0e", borderpad=4))

    annotations.append(dict(x=0.02, y=1.05, xref="paper", yref="paper", text=f"<b>Put 總金額</b><br>{total_put_money/100000000:.1f} 億", showarrow=False, align="left", font=dict(size=14, color="#2ca02c"), bgcolor="white", bordercolor="#2ca02c", borderwidth=2, borderpad=6))
    annotations.append(dict(x=0.98, y=1.05, xref="paper", yref="paper", text=f"<b>Call 總金額</b><br>{total_call_money/100000000:.1f} 億", showarrow=False, align="right", font=dict(size=14, color="#d62728"), bgcolor="white", bordercolor="#d62728", borderwidth=2, borderpad=6))

    fig.update_layout(
        title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='top', font=dict(size=20, color="black")), 
        xaxis=dict(title='未平倉量 (OI)', range=[-x_limit, x_limit], showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='black', tickmode='array', tickvals=[-x_limit*0.75, -x_limit*0.5, -x_limit*0.25, 0, x_limit*0.25, x_limit*0.5, x_limit*0.75], ticktext=[f"{int(x_limit*0.75)}", f"{int(x_limit*0.75)}", f"{int(x_limit*0.25)}", "0", f"{int(x_limit*0.25)}", f"{int(x_limit*0.5)}", f"{int(x_limit*0.75)}"]), 
        yaxis=dict(title='履約價', tickmode='linear', dtick=100, tickformat='d'), 
        barmode='overlay', 
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"), 
        height=750, 
        margin=dict(l=40, r=100, t=140, b=60), 
        annotations=annotations, 
        paper_bgcolor='white', 
        plot_bgcolor='white'
    )
    return fig

# --- 🆕 資料準備函式（整合所有數據給 AI）---
def prepare_ai_data(df, inst_df_today=None, inst_df_yesterday=None):
    """整合選擇權籌碼 + 三大法人數據，確保 AI 能讀取所有資訊"""
    
    # 1. 選擇權籌碼數據（前 25 大）
    df_ai = df.copy()
    if 'Amount' in df_ai.columns:
        df_ai = df_ai.nlargest(25, 'Amount')
    
    keep_cols = [c for c in ['Strike', 'Type', 'OI', 'Amount', 'OI_Change'] if c in df_ai.columns]
    df_ai = df_ai[keep_cols]
    
    option_data_csv = df_ai.to_csv(index=False)
    
    # 2. 三大法人數據
    institutional_summary = ""
    
    if inst_df_today is not None and not inst_df_today.empty:
        institutional_summary += "\n\n【三大法人選擇權籌碼 - 最新】\n"
        institutional_summary += inst_df_today.to_string(index=False)
    
    if inst_df_yesterday is not None and not inst_df_yesterday.empty:
        institutional_summary += "\n\n【三大法人選擇權籌碼 - 前一日】\n"
        institutional_summary += inst_df_yesterday.to_string(index=False)
    
    # 3. 合併成完整的數據字串
    full_data = f"""
=== 選擇權未平倉籌碼分析（資金前 25 大）===
{option_data_csv}

=== 三大法人動向 ===
{institutional_summary if institutional_summary else "（三大法人數據暫無）"}
"""
    
    return full_data.strip()

# --- helper ---
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

# --- 🆕 莊家控盤思維 Prompt（加強版：整合三大法人）---
def build_ai_prompt(data_str, taiex_price, contract_info):
    contract_note = f"結算合約：{contract_info.get('code')}" if contract_info else ""

    prompt = f"""
    你是台指期市場的『理性鐵血莊家』(Ruthless Market Maker)。
    你的目標是：**透過籌碼優勢，讓賣方利潤最大化 (Max Pain)**。
    目前現貨：{taiex_price}。{contract_note}
    
    請根據下方數據進行【莊家控盤劇本】推演：
    1. 「選擇權未平倉籌碼」- 顯示資金最集中的戰場
    2. 「三大法人動向」- 自營商、投信、外資的布局變化
    
    【請依此格式輸出】：
    🎯 **莊家結算目標 (Max Pain)**：
    (請預估一個點位或區間，這是讓 Call 和 Put 賣方通殺的甜蜜點)
    
    🏦 **三大法人解讀**：
    (分析自營商、投信、外資的多空部位變化，誰在主導？誰在對作？)
    
    🩸 **散戶狙擊區 (Kill Zone)**：
    (指出哪個價位的 Call 或 Put 散戶最多？如果拉過去或殺下去，迫使他們停損？)
    
    ☠️ **控盤劇本**：
    (偏多誘空？還是拉高出貨？還是區間盤整吃權利金？請直接給出你的極致控盤策略)

    完整數據：
    {data_str}
    """
    return prompt.strip()

# --- AI 分析 (Gemini) ---
def ask_gemini(prompt_text):
    if not gemini_model: return "⚠️ 未設定 Gemini Key"
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    try:
        res = gemini_model.generate_content(prompt_text, safety_settings=safety_settings)
        return res.text
    except ValueError:
        return "⚠️ Gemini 拒絕回答：Prompt 觸發了安全審查，請嘗試修飾用詞。"
    except Exception as e:
        return f"Gemini 錯誤: {str(e)}"

# --- AI 分析 (ChatGPT) ---
def ask_chatgpt(prompt_text):
    if not openai_client: return "⚠️ 未設定 OpenAI Key"
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a ruthless market maker."},
                {"role": "user", "content": prompt_text}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        if "insufficient_quota" in str(e): return "⚠️ OpenAI 額度不足"
        return f"ChatGPT 錯誤: {str(e)}"

# --- 主程式 ---
def main():
    # 確保 Session State 狀態初始化
    if 'analysis_unlocked' not in st.session_state:
        st.session_state.analysis_unlocked = False
        st.session_state.show_analysis_results = False 

    # ⭐ 注入 AdSense 代碼
    inject_adsense_head()
    
    st.title("🧛‍♂️ 台指期籌碼戰情室 (莊家控盤版)")
    
    col_title, col_btn = st.columns([3, 1])
    if st.sidebar.button("🔄 重新整理"): 
        st.session_state.analysis_unlocked = False 
        st.session_state.show_analysis_results = False 
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(f"🔵 Gemini: {'✅' if gemini_model else '❌'}")
    st.sidebar.caption(f"🟢 ChatGPT: {'✅' if openai_client else '❌'}")

    with st.spinner('連線期交所中...'):
        df_today, date_today, df_yesterday, date_yesterday = get_option_data()
        taiex_now = get_realtime_data()
        
        # 🆕 獲取三大法人數據
        inst_today, inst_date_today, inst_yesterday, inst_date_yesterday = get_institutional_option_data()

    if df_today is None or df_yesterday is None: 
        st.error("查無資料。需至少取得兩天有效數據以計算日變化 (DoD)。")
        return

    df_full = calculate_dod_change(df_today, df_yesterday)
    df = df_full 
    data_date = date_today
    
    # 數據指標與圖表
    total_call_amt = df[df['Type'].str.contains('買|Call', case=False, na=False)]['Amount'].sum()
    total_put_amt = df[df['Type'].str.contains('賣|Put', case=False, na=False)]['Amount'].sum()
    pc_ratio_amt = (total_put_amt / total_call_amt) * 100 if total_call_amt > 0 else 0
    
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.sidebar.download_button("📥 下載完整數據", csv, f"option_{data_date.replace('/', '')}_dod.csv", "text/csv")
    
    # 🆕 三大法人數據顯示
    if inst_today is not None and not inst_today.empty:
        with st.sidebar.expander("📊 三大法人選擇權籌碼", expanded=False):
            st.caption(f"數據日期: {inst_date_today}")
            st.dataframe(inst_today, use_container_width=True)
            
            inst_csv = inst_today.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下載法人數據", inst_csv, f"institutional_{inst_date_today.replace('/', '')}.csv", "text/csv")
    
    c1, c2, c3, c4 = st.columns([1.2, 0.8, 1, 1])
    c1.markdown(f"<div style='text-align: left;'><span style='font-size: 14px; color: #555;'>製圖時間</span><br><span style='font-size: 18px; font-weight: bold;'>{datetime.now(tz=TW_TZ).strftime('%Y/%m/%d %H:%M:%S')}</span></div>", unsafe_allow_html=True)
    c2.metric("大盤現貨", f"{int(taiex_now) if taiex_now else 'N/A'}")
    trend = "偏多" if pc_ratio_amt > 100 else "偏空"
    c3.metric("全市場 P/C 金額比", f"{pc_ratio_amt:.1f}%", f"{trend}格局", delta_color="normal" if pc_ratio_amt > 100 else "inverse")
    c4.metric("資料來源日期", f"{data_date} (與 {date_yesterday} 比較)")

    st.markdown("---")
    
    # --- 廣告與解鎖邏輯 ---
    if st.session_state.analysis_unlocked:
        # 解鎖後：顯示 AI 分析區塊
        st.markdown("### 🎲 莊家控盤劇本 (雙 AI 預測)")
        analyze_button = st.button("🧛‍♂️ 啟動 AI 控盤分析", type="primary", disabled=False)
        
        if analyze_button:
            st.session_state.show_analysis_results = True
            st.rerun()

    else:
        # 未解鎖：顯示廣告和倒數計時
        st.markdown("### 🔓 觀看廣告解鎖 AI 分析")
        st.info("💡 **提示**：此網站使用 Google AdSense 提供免費服務。AdSense 審核通過後，此處將顯示廣告。")
        
        # 顯示廣告佔位符
        show_ad_placeholder()
        
        st.markdown("---")
        
        start_countdown = st.button("⏱️ 點此開始倒數解鎖 AI 分析功能", key="start_timer", type="secondary")
        
        if start_countdown:
            placeholder = st.empty()
            wait_time = 8 
            
            for i in range(wait_time, 0, -1):
                placeholder.warning(f"⏳ 請勿離開頁面，分析功能將在 {i} 秒後自動解鎖...")
                time.sleep(1)
            
            st.session_state.analysis_unlocked = True
            placeholder.success("✅ AI 分析功能已解鎖！請點擊上方的綠色按鈕執行分析。")
            st.rerun()

    # --- AI 執行與結果顯示邏輯 ---
    if st.session_state.show_analysis_results:
        if not st.session_state.analysis_unlocked:
            st.markdown("### 🎲 莊家控盤劇本 (雙 AI 預測)")

        if not gemini_model and not openai_client:
            st.error("請至少設定一個 API Key")
        else:
            # 🆕 整合所有數據（選擇權 + 三大法人）
            data_str = prepare_ai_data(df, inst_today, inst_yesterday)
            plot_targets = get_next_contracts(df, data_date) 
            contract_info = plot_targets[0]['info'] if plot_targets else None
            prompt_text = build_ai_prompt(data_str, taiex_now, contract_info)

            with st.spinner("AI 正在計算最大痛點與獵殺區間..."):
                gemini_result = None
                chatgpt_result = None

                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = {}
                    if gemini_model: futures['gemini'] = executor.submit(ask_gemini, prompt_text)
                    if openai_client: futures['chatgpt'] = executor.submit(ask_chatgpt, prompt_text)

                    for key, future in futures.items():
                        if key == 'gemini': gemini_result = future.result()
                        elif key == 'chatgpt': chatgpt_result = future.result()

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔵 Google Gemini")
                if gemini_model:
                    if gemini_result:
                        st.info(gemini_result)
                    else:
                        st.warning("無回應 (可能觸發安全限制或 API 額度用罄)")
                else:
                    st.warning("未設定 Key")

            with col2:
                st.subheader("🟢 ChatGPT")
                if openai_client:
                    if chatgpt_result and "⚠️" in chatgpt_result:
                        st.warning(chatgpt_result)
                    elif chatgpt_result:
                        st.success(chatgpt_result)
                    else:
                        st.warning("無回應")
                else:
                    st.warning("未設定 Key")
    
    # --- 圖表顯示區 ---
    plot_targets = get_next_contracts(df, data_date)
    cols = st.columns(len(plot_targets)) if plot_targets else []
    for i, target in enumerate(plot_targets):
        with cols[i]:
            m_code = target['info']['code']
            s_date = target['info']['date']
            df_target = df[df['Month'] == m_code]
            sub_call = df_target[df_target['Type'].str.contains('Call|買', case=False, na=False)]['Amount'].sum()
            sub_put = df_target[df_target['Type'].str.contains('Put|賣', case=False, na=False)]['Amount'].sum()
            sub_ratio = (sub_put / sub_call * 100) if sub_call > 0 else 0
            title_text = (f"<b> {m_code}</b><br><span style='font-size: 14px;'>結算: {s_date}</span><br><span style='font-size: 14px;'>P/C金額比: {sub_ratio:.1f}% ({'偏多' if sub_ratio > 100 else '偏空'})</span>")
            st.plotly_chart(plot_tornado_chart(df_target, title_text, taiex_now), use_container_width=True)

if __name__ == "__main__":
    main()
