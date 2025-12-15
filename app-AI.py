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
import streamlit.components.v1 as components
import numpy as np
from scipy.stats import norm

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

# --- 模型設定 ---
def get_gemini_model(api_key):
    if not api_key: return None, "未設定"
    genai.configure(api_key=api_key)
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        target_model_name = None
        priority_targets = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro', 'flash']
        for target in priority_targets:
            for model_id in available_models:
                if target in model_id.lower():
                    target_model_name = model_id
                    break
            if target_model_name: break
        if not target_model_name and available_models: target_model_name = available_models[0]
        return (genai.GenerativeModel(target_model_name), target_model_name) if target_model_name else (None, "無可用模型")
    except Exception as e: return None, f"模型設定錯誤: {str(e)}"

def get_openai_client(api_key):
    if not api_key: return None
    return OpenAI(api_key=api_key)

gemini_model, gemini_name = get_gemini_model(GEMINI_KEY)
openai_client = get_openai_client(OPENAI_KEY)
MANUAL_SETTLEMENT_FIX = {'202501W1': '2025/01/02'}

# ⭐ AdSense
ADSENSE_PUB_ID = 'ca-pub-4585150092118682'
def inject_adsense_head():
    st.markdown(f"""<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={ADSENSE_PUB_ID}" crossorigin="anonymous"></script>""", unsafe_allow_html=True)
    components.html(f"""<!DOCTYPE html><html><body><div style="min-height: 1px;"></div></body></html>""", height=1, scrolling=False)

def show_ad_placeholder():
    st.markdown(f"""<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={ADSENSE_PUB_ID}" crossorigin="anonymous"></script>""", unsafe_allow_html=True)
    st.markdown(f"""<div style='background:#f8f9fa;padding:40px;border:2px dashed #dee2e6;text-align:center;'><p style='color:#6c757d'>廣告位置 (Publisher ID: {ADSENSE_PUB_ID})</p></div>""", unsafe_allow_html=True)

# ----------------------------------------------------------------------

# --- 核心日期函式 ---
def get_settlement_date(contract_code):
    code = str(contract_code).strip().upper()
    for key, fix_date in MANUAL_SETTLEMENT_FIX.items():
        if key in code: return fix_date
    try:
        if len(code) < 6: return "9999/99/99"
        year, month = int(code[:4]), int(code[4:6])
        c = calendar.monthcalendar(year, month)
        wednesdays = [week[calendar.WEDNESDAY] for week in c if week[calendar.WEDNESDAY] != 0]
        fridays = [week[calendar.FRIDAY] for week in c if week[calendar.FRIDAY] != 0]
        day = None
        if 'W' in code:
            match = re.search(r'W(\d)', code)
            if match and len(wednesdays) >= int(match.group(1)): day = wednesdays[int(match.group(1)) - 1]
        elif 'F' in code:
            match = re.search(r'F(\d)', code)
            if match and len(fridays) >= int(match.group(1)): day = fridays[int(match.group(1)) - 1]
        else:
            if len(wednesdays) >= 3: day = wednesdays[2]
        return f"{year}/{month:02d}/{day:02d}" if day else "9999/99/99"
    except: return "9999/99/99"

@st.cache_data(ttl=60)
def get_realtime_data():
    """獲取大盤現貨即時價格 (Yahoo/TWSE)"""
    taiex = None
    ts = int(time.time())
    headers = {'User-Agent': 'Mozilla/5.0'}
    # 1. TWSE MIS
    try:
        url = f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_t00.tw&json=1&delay=0&_={ts}000"
        res = requests.get(url, timeout=2)
        data = res.json()
        if 'msgArray' in data and len(data['msgArray']) > 0:
            val = data['msgArray'][0].get('z', '-')
            if val == '-': val = data['msgArray'][0].get('o', '-') # 若無成交用開盤
            if val == '-': val = data['msgArray'][0].get('y', '-') # 若無開盤用昨收
            if val != '-': taiex = float(val)
    except: pass
    # 2. Yahoo Finance (Backup)
    if taiex is None:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII?interval=1m&range=1d&_={ts}"
            res = requests.get(url, headers=headers, timeout=3)
            data = res.json()
            price = data['chart']['result'][0]['meta'].get('regularMarketPrice')
            if price: taiex = float(price)
        except: pass
    return taiex

# --- 🔥 (修改版) 獲取期貨行情 - 強制回溯 ---
@st.cache_data(ttl=300)
def get_futures_data():
    """獲取台指期貨價格 (自動回溯直到抓到數據)"""
    url = "https://www.taifex.com.tw/cht/3/futContractsDate"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # 嘗試回溯 14 天
    for i in range(14):
        target_date = datetime.now(tz=TW_TZ) - timedelta(days=i)
        
        # 簡單過濾：如果是今天且未過 15:00，期交所日報表還沒出來，跳過
        if i == 0 and datetime.now(tz=TW_TZ).hour < 15:
            continue
            
        query_date = target_date.strftime('%Y/%m/%d')
        payload = {'queryType': '1', 'marketCode': '0', 'commodity_id': 'TX', 'queryDate': query_date}
        
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=5)
            res.encoding = 'utf-8'
            if "查無資料" in res.text: continue
            
            dfs = pd.read_html(StringIO(res.text))
            if not dfs: continue
            df = dfs[0]
            if len(df) > 0:
                futures_price = None
                volume = None
                for col in df.columns:
                    if '收盤價' in str(col) or '成交價' in str(col):
                        try: futures_price = float(str(df.iloc[0][col]).replace(',', ''))
                        except: pass
                    if '成交量' in str(col):
                        try: volume = int(str(df.iloc[0][col]).replace(',', ''))
                        except: pass
                
                # 只要抓到價格就算成功
                if futures_price:
                    return futures_price, volume, query_date
        except: pass
    
    return None, None, "N/A"

# --- 🔥 (修改版) 三大法人期貨 - 強制回溯 ---
@st.cache_data(ttl=300)
def get_institutional_futures_position():
    """獲取法人期貨淨部位 (自動回溯)"""
    url = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for i in range(14):
        target_date = datetime.now(tz=TW_TZ) - timedelta(days=i)
        if i == 0 and datetime.now(tz=TW_TZ).hour < 15: continue 
        
        query_date = target_date.strftime('%Y/%m/%d')
        payload = {'down_type': '1', 'queryStartDate': query_date, 'queryEndDate': query_date, 'commodity_id': 'TX'}
        
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=5)
            res.encoding = 'utf-8'
            if "查無資料" in res.text or len(res.text) < 500: continue
            
            dfs = pd.read_html(StringIO(res.text))
            if not dfs: continue
            df = dfs[0]
            
            inst_data = {}
            for idx, row in df.iterrows():
                row_str = str(row.iloc[0])
                # 尋找關鍵字
                targets = ['外資', '自營商', '投信']
                for t in targets:
                    if t in row_str:
                        for col in df.columns:
                            if '買賣差額' in str(col) or '淨額' in str(col):
                                try: inst_data[t] = int(str(row[col]).replace(',', ''))
                                except: pass
                                break
            
            # 只要有抓到任一法人數據就算成功
            if inst_data:
                inst_data['date'] = query_date
                return inst_data
        except: pass

    return None

# --- 🔥 (修改版) 三大法人選擇權 - 強制回溯 ---
@st.cache_data(ttl=300)
def get_institutional_option_data():
    """獲取法人選擇權數據 (自動回溯，需抓兩天)"""
    url = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
    headers = {'User-Agent': 'Mozilla/5.0'}
    all_inst_data = []
    
    # 嘗試回溯 20 天以確保抓到兩天數據
    for i in range(20):
        target_date = datetime.now(tz=TW_TZ) - timedelta(days=i)
        if i == 0 and datetime.now(tz=TW_TZ).hour < 15: continue
        
        query_date = target_date.strftime('%Y/%m/%d')
        payload = {'down_type': '1', 'queryStartDate': query_date, 'queryEndDate': query_date, 'commodity_id': 'TXO'}
        
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=5)
            res.encoding = 'utf-8'
            if "查無資料" in res.text or len(res.text) < 500: continue
            
            dfs = pd.read_html(StringIO(res.text))
            if not dfs: continue
            df = dfs[0]
            
            df.columns = [str(c).strip().replace(' ', '').replace('\n', '') for c in df.columns]
            df_filtered = df[df.iloc[:, 0].astype(str).str.contains('自營商|投信|外資', na=False)]
            
            if not df_filtered.empty:
                all_inst_data.append({'date': query_date, 'df': df_filtered})
                if len(all_inst_data) >= 2: break # 抓到兩天就停
        except: pass
    
    if len(all_inst_data) < 1: return None, None, None, None
    
    # 處理抓到 1 天或 2 天的情況
    today_df = all_inst_data[0]['df']
    today_date = all_inst_data[0]['date']
    yesterday_df = all_inst_data[1]['df'] if len(all_inst_data) > 1 else None
    yesterday_date = all_inst_data[1]['date'] if len(all_inst_data) > 1 else None
    
    return today_df, today_date, yesterday_df, yesterday_date

# --- 🔥 (修改版) 選擇權全履約價 - 強制回溯 ---
@st.cache_data(ttl=300)
def get_option_data_multi_days(days=3):
    """獲取選擇權數據 (自動回溯)"""
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    headers = {'User-Agent': 'Mozilla/5.0'}
    all_data = []

    for i in range(20):
        target_date = datetime.now(tz=TW_TZ) - timedelta(days=i)
        if i == 0 and datetime.now(tz=TW_TZ).hour < 15: continue

        query_date = target_date.strftime('%Y/%m/%d')
        payload = {
            'queryType': '2', 'marketCode': '0', 'commodity_id': 'TXO', 
            'queryDate': query_date, 'MarketCode': '0', 'commodity_idt': 'TXO'
        }
        
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=5)
            res.encoding = 'utf-8'
            if "查無資料" in res.text or len(res.text) < 500: continue
            
            dfs = pd.read_html(StringIO(res.text))
            df = dfs[0]
            
            # 欄位清洗
            df.columns = [str(c).replace(' ', '').replace('*', '').replace('契約', '').strip() for c in df.columns]
            col_map = {
                'Month': next((c for c in df.columns if '月' in c or '週' in c), None),
                'Strike': next((c for c in df.columns if '履約' in c), None),
                'Type': next((c for c in df.columns if '買賣' in c), None),
                'OI': next((c for c in df.columns if '未沖銷' in c or 'OI' in c), None),
                'Price': next((c for c in df.columns if '結算' in c or '收盤' in c or 'Price' in c), None)
            }
            if not all(col_map.values()): continue
            
            df = df.rename(columns={k:v for k,v in col_map.items() if v})
            df = df[['Month', 'Strike', 'Type', 'OI', 'Price']].dropna(subset=['Type'])
            
            df['Type'] = df['Type'].astype(str).str.strip()
            df['Strike'] = pd.to_numeric(df['Strike'].astype(str).str.replace(',', ''), errors='coerce')
            df['OI'] = pd.to_numeric(df['OI'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace(',', '').replace('-', '0'), errors='coerce').fillna(0)
            df['Amount'] = df['OI'] * df['Price'] * 50
            
            if df['OI'].sum() > 0:
                all_data.append({'date': query_date, 'df': df})
                if len(all_data) >= days: break
        except: continue
    
    return all_data if len(all_data) >= 1 else None # 至少回傳一天

# --- 數學計算 (IV, Greeks, GEX) ---
def calculate_iv(option_price, spot_price, strike, time_to_expiry, option_type='call', risk_free_rate=0.015):
    if option_price <= 0 or spot_price <= 0 or strike <= 0 or time_to_expiry <= 0: return None
    sigma = 0.3
    for i in range(50): # 減少迭代次數加快速度
        d1 = (np.log(spot_price / strike) + (risk_free_rate + 0.5 * sigma ** 2) * time_to_expiry) / (sigma * np.sqrt(time_to_expiry))
        d2 = d1 - sigma * np.sqrt(time_to_expiry)
        if option_type == 'call':
            price = spot_price * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
            vega = spot_price * norm.pdf(d1) * np.sqrt(time_to_expiry)
        else:
            price = strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
            vega = spot_price * norm.pdf(d1) * np.sqrt(time_to_expiry)
        diff = price - option_price
        if abs(diff) < 1e-4: return sigma
        if vega == 0: return None
        sigma = sigma - diff / vega
        if sigma <= 0: return None
    return None

def calculate_greeks(spot_price, strike, time_to_expiry, volatility, option_type='call', risk_free_rate=0.015):
    if volatility is None or volatility <= 0 or time_to_expiry <= 0: return None, None
    try:
        d1 = (np.log(spot_price / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        if option_type == 'call': delta = norm.cdf(d1)
        else: delta = norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (spot_price * volatility * np.sqrt(time_to_expiry))
        return delta, gamma
    except: return None, None

def calculate_dealer_gex(df, spot_price, settlement_date):
    try:
        today = datetime.now(tz=TW_TZ)
        expiry = datetime.strptime(settlement_date, '%Y/%m/%d').replace(tzinfo=TW_TZ)
        time_to_expiry = max((expiry - today).days / 365.0, 0.001)
        gex_data = []
        for idx, row in df.iterrows():
            strike = row['Strike']
            oi = row['OI']
            price = row['Price']
            option_type = 'call' if 'Call' in str(row['Type']) or '買' in str(row['Type']) else 'put'
            if price > 0 and oi > 0:
                iv = calculate_iv(price, spot_price, strike, time_to_expiry, option_type)
                if iv:
                    delta, gamma = calculate_greeks(spot_price, strike, time_to_expiry, iv, option_type)
                    if gamma:
                        gex = -gamma * oi * (spot_price ** 2) * 0.01
                        gex_data.append({'Strike': strike, 'Type': option_type, 'OI': oi, 'Gamma': gamma, 'GEX': gex})
        if gex_data:
            return pd.DataFrame(gex_data).groupby('Strike')['GEX'].sum().reset_index()
    except: pass
    return None

def calculate_risk_reversal(df, spot_price, settlement_date):
    try:
        today = datetime.now(tz=TW_TZ)
        expiry = datetime.strptime(settlement_date, '%Y/%m/%d').replace(tzinfo=TW_TZ)
        time_to_expiry = max((expiry - today).days / 365.0, 0.001)
        atm_strike = min(df['Strike'], key=lambda x: abs(x - spot_price))
        iv_delta_data = []
        for idx, row in df.iterrows():
            strike = row['Strike']
            price = row['Price']
            option_type = 'call' if 'Call' in str(row['Type']) or '買' in str(row['Type']) else 'put'
            if price > 0:
                iv = calculate_iv(price, spot_price, strike, time_to_expiry, option_type)
                if iv:
                    delta, _ = calculate_greeks(spot_price, strike, time_to_expiry, iv, option_type)
                    if delta: iv_delta_data.append({'Strike': strike, 'Type': option_type, 'IV': iv, 'Delta': abs(delta)})
        if not iv_delta_data: return None, None, None
        iv_df = pd.DataFrame(iv_delta_data)
        call_25d = iv_df[(iv_df['Type'] == 'call') & (iv_df['Delta'] > 0.2) & (iv_df['Delta'] < 0.3)]
        put_25d = iv_df[(iv_df['Type'] == 'put') & (iv_df['Delta'] > 0.2) & (iv_df['Delta'] < 0.3)]
        atm_iv = iv_df[iv_df['Strike'] == atm_strike]['IV'].mean()
        if not call_25d.empty and not put_25d.empty:
            rr = call_25d.iloc[0]['IV'] - put_25d.iloc[0]['IV']
            return atm_iv, rr, atm_strike
        return atm_iv, None, atm_strike
    except: return None, None, None

def calculate_multi_day_oi_change(all_data):
    if not all_data or len(all_data) < 1: return None
    df_latest = all_data[0]['df'].copy()
    if len(all_data) > 1:
        for i in range(1, len(all_data)):
            df_prev = all_data[i]['df'].copy()
            df_merged = pd.merge(df_latest[['Month', 'Strike', 'Type', 'OI']], df_prev[['Month', 'Strike', 'Type', 'OI']], on=['Month', 'Strike', 'Type'], how='left', suffixes=('', f'_D{i}')).fillna(0)
            df_latest[f'OI_Change_D{i}'] = df_merged['OI'] - df_merged[f'OI_D{i}']
    return df_latest

# --- 圖表繪製 ---
def plot_tornado_chart(df_target, title_text, spot_price):
    is_call = df_target['Type'].str.contains('買|Call', case=False, na=False)
    df_call = df_target[is_call][['Strike', 'OI', 'Amount']].rename(columns={'OI': 'Call_OI', 'Amount': 'Call_Amt'})
    df_put = df_target[~is_call][['Strike', 'OI', 'Amount']].rename(columns={'OI': 'Put_OI', 'Amount': 'Put_Amt'})
    data = pd.merge(df_call, df_put, on='Strike', how='outer').fillna(0).sort_values('Strike')
    
    FOCUS_RANGE = 1200
    center_price = spot_price if (spot_price and spot_price > 0) else data['Strike'].median()
    if center_price > 0:
        data = data[(data['Strike'] >= center_price - FOCUS_RANGE) & (data['Strike'] <= center_price + FOCUS_RANGE)]
    
    max_oi = max(data['Put_OI'].max(), data['Call_OI'].max()) if not data.empty else 1000
    x_limit = max_oi * 1.1

    # 處理 OI 變化文字
    data['Put_Text'] = ""
    data['Call_Text'] = ""
    if 'OI_Change_D1' in df_target.columns:
        # 簡易合併邏輯
        df_chg = df_target[['Strike', 'Type', 'OI_Change_D1']].copy()
        call_c = df_chg[df_chg['Type'].str.contains('Call|買')].set_index('Strike')['OI_Change_D1']
        put_c = df_chg[~df_chg['Type'].str.contains('Call|買')].set_index('Strike')['OI_Change_D1']
        data['Call_Change'] = data['Strike'].map(call_c).fillna(0)
        data['Put_Change'] = data['Strike'].map(put_c).fillna(0)
        data['Put_Text'] = data.apply(lambda r: f"{'+' if r['Put_Change']>0 else ''}{int(r['Put_Change'])}" if r['Put_OI']>0 else "", axis=1)
        data['Call_Text'] = data.apply(lambda r: f"{'+' if r['Call_Change']>0 else ''}{int(r['Call_Change'])}" if r['Call_OI']>0 else "", axis=1)

    fig = go.Figure()
    fig.add_trace(go.Bar(y=data['Strike'], x=-data['Put_OI'], orientation='h', name='Put (支撐)', marker_color='#2ca02c', opacity=0.85, text=data['Put_Text'], textposition='outside', hovertemplate='Put OI: %{x}<br>Amt: %{customdata:.2f}億', customdata=data['Put_Amt']/1e8))
    fig.add_trace(go.Bar(y=data['Strike'], x=data['Call_OI'], orientation='h', name='Call (壓力)', marker_color='#d62728', opacity=0.85, text=data['Call_Text'], textposition='outside', hovertemplate='Call OI: %{x}<br>Amt: %{customdata:.2f}億', customdata=data['Call_Amt']/1e8))
    
    if spot_price:
        fig.add_hline(y=spot_price, line_dash="dash", line_color="#ff7f0e", line_width=2)
        fig.add_annotation(x=1.05, y=spot_price, text=f"現貨 {int(spot_price)}", showarrow=False, bgcolor="#ff7f0e", font=dict(color="white"))
        
    fig.update_layout(title=dict(text=title_text, x=0.5), xaxis=dict(range=[-x_limit, x_limit]), barmode='overlay', height=750)
    return fig

def plot_gex_chart(gex_df, spot_price):
    if gex_df is None or gex_df.empty: return None
    fig = go.Figure()
    colors = ['green' if x > 0 else 'red' for x in gex_df['GEX']]
    fig.add_trace(go.Bar(x=gex_df['Strike'], y=gex_df['GEX'], marker_color=colors, name='GEX'))
    if spot_price: fig.add_vline(x=spot_price, line_dash="dash", line_color="orange")
    fig.update_layout(title="Dealer Gamma Exposure (GEX)", xaxis_title="履約價", yaxis_title="GEX", height=400, showlegend=False)
    return fig

# --- Prompt & AI ---
def prepare_ai_data(df, inst_opt_today, inst_opt_yesterday, inst_fut, futures_price, spot_price, basis, atm_iv, risk_reversal, gex_summary, data_date):
    df_ai = df.nlargest(30, 'Amount') if 'Amount' in df.columns else df
    cols = [c for c in ['Strike','Type','OI','Amount','OI_Change_D1'] if c in df_ai.columns]
    
    inst_opt_str = inst_opt_today.to_string(index=False) if inst_opt_today is not None else "無"
    inst_fut_str = ""
    if inst_fut:
        for k,v in inst_fut.items(): 
            if k != 'date': inst_fut_str += f"{k}: {v:+,} 口\n"
    
    gex_str = ""
    if gex_summary is not None:
        top_gex = gex_summary.loc[gex_summary['GEX'].abs().idxmax()]
        gex_str = f"最大GEX履約價: {top_gex['Strike']} (GEX: {top_gex['GEX']:.2f})"

    return f"""
    數據日期: {data_date}
    現貨: {spot_price}, 期貨: {futures_price}, 基差: {basis}
    ATM IV: {atm_iv}, Risk Reversal: {risk_reversal}
    Dealer GEX 重點: {gex_str}
    
    【選擇權重倉區】:
    {df_ai[cols].to_csv(index=False)}
    
    【法人選擇權籌碼】:
    {inst_opt_str}
    
    【法人期貨淨單】:
    {inst_fut_str}
    """

def build_ai_prompt(data_str, taiex_price):
    return f"""
    你是台指期莊家分析師。
    目標：分析籌碼結構，預判結算行情 (Max Pain)。
    
    現貨價格：{taiex_price}
    
    請分析：
    1. 莊家與法人佈局解讀 (期貨多空 + 選擇權籌碼)。
    2. 關鍵支撐與壓力位 (Kill Zone)。
    3. 波動率與 Gamma 風險 (是否會加速行情)。
    4. 給出明確的「控盤劇本」與「結算目標區間」。
    
    數據如下：
    {data_str}
    """

def ask_gemini(prompt):
    if not gemini_model: return "未設定 Gemini Key"
    try: return gemini_model.generate_content(prompt).text
    except Exception as e: return str(e)

def ask_chatgpt(prompt):
    if not openai_client: return "未設定 OpenAI Key"
    try:
        res = openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}])
        return res.choices[0].message.content
    except Exception as e: return str(e)

def get_next_contracts(df, data_date):
    unique_codes = sorted(df['Month'].unique())
    targets = []
    for code in unique_codes:
        s_date = get_settlement_date(code)
        if s_date > data_date:
            targets.append({'code': code, 'date': s_date})
            if len(targets) >= 2: break
    return targets

# --- Main ---
def main():
    if 'analysis_unlocked' not in st.session_state: st.session_state.analysis_unlocked = False
    if 'show_analysis_results' not in st.session_state: st.session_state.show_analysis_results = False
    inject_adsense_head()
    
    st.title("🧛‍♂️ 台指期籌碼戰情室 (莊家控盤 - 強制回溯版)")
    
    if st.sidebar.button("🔄 重新整理"):
        st.cache_data.clear()
        st.session_state.show_analysis_results = False
        st.rerun()
    
    st.sidebar.caption(f"Gemini: {'✅' if gemini_model else '❌'} | ChatGPT: {'✅' if openai_client else '❌'}")

    with st.spinner("🔄 正在強制回溯搜尋最新數據..."):
        taiex_now = get_realtime_data()
        
        # 1. 期貨行情 (含日期)
        futures_price, futures_volume, fut_date = get_futures_data()
        
        # 2. 法人期貨 (含日期)
        inst_fut_position = get_institutional_futures_position()
        
        # 3. 法人選擇權
        inst_opt_today, inst_opt_date, inst_opt_prev, _ = get_institutional_option_data()
        
        # 4. 選擇權全市場
        all_option_data = get_option_data_multi_days(days=2)

    if not all_option_data:
        st.error("❌ 無法抓取任何選擇權數據 (已回溯 20 天)")
        return

    # 數據處理
    df_full = calculate_multi_day_oi_change(all_option_data)
    data_date = all_option_data[0]['date']
    basis = (futures_price - taiex_now) if (taiex_now and futures_price) else None
    
    # 下載
    st.sidebar.download_button("📥 下載數據", df_full.to_csv(index=False).encode('utf-8-sig'), "opt_data.csv")

    # === 儀表板 ===
    # 時間與價格
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.caption(f"更新時間: {datetime.now(tz=TW_TZ).strftime('%H:%M:%S')}")
    c2.metric("加權指數 (即時)", f"{int(taiex_now) if taiex_now else 'N/A'}")
    c3.metric(f"台指期 ({fut_date[5:]})", f"{int(futures_price) if futures_price else 'N/A'}")
    c4.metric("基差", f"{basis:.0f}" if basis else "N/A", delta_color="normal" if basis and basis > 0 else "inverse")
    
    # P/C Ratio
    call_amt = df_full[df_full['Type'].str.contains('Call|買')]['Amount'].sum()
    put_amt = df_full[df_full['Type'].str.contains('Put|賣')]['Amount'].sum()
    pc_ratio = (put_amt / call_amt * 100) if call_amt > 0 else 0
    c5.metric(f"P/C 金額比 ({data_date[5:]})", f"{pc_ratio:.1f}%", "偏多" if pc_ratio > 100 else "偏空")
    
    st.markdown("---")
    
    # === 法人籌碼區 (紅綠燈) ===
    st.markdown("### 🏦 三大法人籌碼佈局")
    if inst_fut_position:
        st.caption(f"期貨籌碼日期: {inst_fut_position.get('date', 'N/A')}")
        f1, f2, f3 = st.columns(3)
        for role, col in zip(['外資', '投信', '自營商'], [f1, f2, f3]):
            val = inst_fut_position.get(role, 0)
            col.metric(f"{role}期貨淨單", f"{val:+,} 口", delta_color="inverse" if val > 0 else "normal")
    else:
        st.warning("⚠️ 查無法人期貨數據")

    if inst_opt_today is not None:
        with st.expander(f"📊 法人選擇權淨部位 ({inst_opt_date})"):
            st.dataframe(inst_opt_today, use_container_width=True)

    st.markdown("---")

    # === 進階計算 & 圖表 ===
    targets = get_next_contracts(df_full, data_date)
    if targets:
        target = targets[0]
        df_target = df_full[df_full['Month'] == target['code']]
        
        atm_iv, rr, atm_k = calculate_risk_reversal(df_target, taiex_now or 23000, target['date'])
        gex_df = calculate_dealer_gex(df_target, taiex_now or 23000, target['date'])
        
        st.markdown(f"### 📊 市場指標 ({target['code']} 結算: {target['date']})")
        k1, k2 = st.columns(2)
        k1.metric("ATM IV", f"{atm_iv*100:.2f}%" if atm_iv else "N/A")
        k2.metric("Risk Reversal", f"{rr*100:.2f}%" if rr else "N/A", "看漲" if rr and rr>0 else "看跌")
        
        if gex_df is not None:
            st.plotly_chart(plot_gex_chart(gex_df, taiex_now), use_container_width=True)

        st.plotly_chart(plot_tornado_chart(df_target, f"{target['code']} 籌碼分佈", taiex_now), use_container_width=True)
    
    # === AI 分析 ===
    st.markdown("---")
    if st.session_state.analysis_unlocked:
        if st.button("🧛‍♂️ 啟動 AI 分析"): st.session_state.show_analysis_results = True
    else:
        show_ad_placeholder()
        if st.button("⏱️ 解鎖 AI 分析"):
            with st.empty():
                for i in range(5, 0, -1):
                    st.write(f"⏳ {i}...")
                    time.sleep(1)
            st.session_state.analysis_unlocked = True
            st.rerun()

    if st.session_state.show_analysis_results and targets:
        data_str = prepare_ai_data(df_full, inst_opt_today, inst_opt_prev, inst_fut_position, futures_price, taiex_now, basis, atm_iv, rr, gex_df, data_date)
        prompt = build_ai_prompt(data_str, taiex_now)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🔵 Gemini")
            st.info(ask_gemini(prompt))
        with c2:
            st.subheader("🟢 ChatGPT")
            st.success(ask_chatgpt(prompt))

if __name__ == "__main__":
    main()
