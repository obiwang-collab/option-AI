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
import urllib3

# 忽略 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(layout="wide", page_title="台指期籌碼戰情室 (莊家控盤版)")
TW_TZ = timezone(timedelta(hours=8))

# 金鑰設定
try:
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
except FileNotFoundError:
    GEMINI_KEY = ""
    OPENAI_KEY = ""

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

# AdSense
ADSENSE_PUB_ID = 'ca-pub-4585150092118682'
def inject_adsense_head():
    st.markdown(f"""<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={ADSENSE_PUB_ID}" crossorigin="anonymous"></script>""", unsafe_allow_html=True)
    components.html(f"""<!DOCTYPE html><html><body><div style="min-height: 1px;"></div></body></html>""", height=1, scrolling=False)

def show_ad_placeholder():
    st.markdown(f"""<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={ADSENSE_PUB_ID}" crossorigin="anonymous"></script>""", unsafe_allow_html=True)
    st.markdown(f"""<div style='background:#f8f9fa;padding:40px;border:2px dashed #dee2e6;text-align:center;'><p style='color:#6c757d'>廣告位置 (Publisher ID: {ADSENSE_PUB_ID})</p></div>""", unsafe_allow_html=True)

# 核心日期函式
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
    """獲取大盤現貨即時價格"""
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
            if val == '-': val = data['msgArray'][0].get('y', '-')
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
def get_futures_data():
    """獲取台指期貨價格"""
    url = "https://www.taifex.com.tw/cht/3/futContractsDate"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for i in range(30):
        target_date = datetime.now(tz=TW_TZ) - timedelta(days=i)
        query_date = target_date.strftime('%Y/%m/%d')
        payload = {'queryType': '1', 'marketCode': '0', 'commodity_id': 'TX', 'queryDate': query_date}
        
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=10, verify=False)
            res.encoding = 'utf-8'
            if "查無資料" in res.text: continue
            
            dfs = pd.read_html(StringIO(res.text))
            if not dfs: continue
            df = dfs[0]
            
            futures_price = None
            for col in df.columns:
                if '收盤價' in str(col) or '成交價' in str(col):
                    try: 
                        futures_price = float(str(df.iloc[0][col]).replace(',', ''))
                        if futures_price > 0: return futures_price, None, query_date
                    except: pass
        except: pass
    
    return None, None, "N/A"

@st.cache_data(ttl=300)
def get_institutional_futures_position():
    """獲取法人期貨淨部位 - 使用 queryType=2"""
    url = "https://www.taifex.com.tw/cht/3/futContractsDate"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for i in range(10):  # 回溯10天
        target_date = datetime.now(tz=TW_TZ) - timedelta(days=i)
        query_date = target_date.strftime('%Y/%m/%d')
        
        # 🔥 關鍵修正: 使用 queryType=2
        payload = {
            'queryType': '2',
            'queryDate': query_date,
            'commodity_id': 'TX'
        }
        
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=10, verify=False)
            res.encoding = 'utf-8'
            
            if "查無資料" in res.text or len(res.text) < 5000:
                continue
            
            dfs = pd.read_html(StringIO(res.text))
            if not dfs:
                continue
                
            df = dfs[0]
            
            # 找到「臺股期貨」的三大法人資料
            inst_data = {}
            
            for idx, row in df.iterrows():
                row_str = " ".join([str(x) for x in row.values])
                
                # 必須同時包含「臺股期貨」和法人名稱
                if '臺股期貨' not in row_str:
                    continue
                
                # 提取未平倉淨部位 (第13欄)
                try:
                    net_position = int(str(row.iloc[13]).replace(',', ''))
                except:
                    continue
                
                if '外資' in row_str or '外資及陸資' in row_str:
                    inst_data['外資'] = net_position
                elif '投信' in row_str:
                    inst_data['投信'] = net_position
                elif '自營商' in row_str:
                    inst_data['自營商'] = net_position
            
            if len(inst_data) == 3:  # 確保三個法人都有
                inst_data['date'] = query_date
                return inst_data
                
        except Exception as e:
            continue
    
    return None

@st.cache_data(ttl=300)
def get_institutional_option_data():
    """獲取法人選擇權數據 - 使用 queryType=2"""
    url = "https://www.taifex.com.tw/cht/3/callsAndPutsDate"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for i in range(10):
        target_date = datetime.now(tz=TW_TZ) - timedelta(days=i)
        query_date = target_date.strftime('%Y/%m/%d')
        
        # 🔥 關鍵修正: 使用 queryType=2
        payload = {
            'queryType': '2',
            'queryDate': query_date,
            'commodity_id': 'TXO'
        }
        
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=10, verify=False)
            res.encoding = 'utf-8'
            
            if "查無資料" in res.text or len(res.text) < 5000:
                continue
            
            dfs = pd.read_html(StringIO(res.text))
            if not dfs:
                continue
            
            df = dfs[0]
            
            # 提取台指選擇權的法人資料
            inst_data = {}
            
            for idx, row in df.iterrows():
                row_str = " ".join([str(x) for x in row.values])
                
                # 必須包含「臺指選擇權」
                if '臺指選擇權' not in row_str:
                    continue
                
                # 欄位結構:
                # [0]序號 [1]商品名稱 [2]權別(買權/賣權) [3]身份別
                # [4-9]交易資料 [10-15]未平倉資料
                # [14]未平倉買賣差額口數
                
                try:
                    option_type = str(row.iloc[2])  # 買權/賣權
                    institution = str(row.iloc[3])  # 自營商/投信/外資
                    net_oi = int(str(row.iloc[14]).replace(',', ''))  # 未平倉買賣差額
                    
                    # 建立資料結構
                    if institution not in inst_data:
                        inst_data[institution] = {}
                    
                    if '買權' in option_type:
                        inst_data[institution]['Call'] = net_oi
                    elif '賣權' in option_type:
                        inst_data[institution]['Put'] = net_oi
                        
                except:
                    continue
            
            # 確保至少有一個法人有完整的 Call/Put 資料
            if inst_data and any(len(v) == 2 for v in inst_data.values()):
                inst_data['date'] = query_date
                return inst_data
                
        except Exception as e:
            continue
    
    return None

# 🔥🔥🔥 核心修正:選擇權數據抓取 - 修正欄位對應
@st.cache_data(ttl=300)
def get_option_data_multi_days(days=3):
    """獲取選擇權全市場數據 (修正欄位對應)"""
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    headers = {'User-Agent': 'Mozilla/5.0'}
    all_data = []

    for i in range(30):
        target_date = datetime.now(tz=TW_TZ) - timedelta(days=i)
        query_date = target_date.strftime('%Y/%m/%d')
        payload = {'queryType': '2', 'marketCode': '0', 'commodity_id': 'TXO', 'queryDate': query_date, 'MarketCode': '0', 'commodity_idt': 'TXO'}
        
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=10, verify=False)
            res.encoding = 'utf-8'
            if "查無資料" in res.text or len(res.text) < 500: continue
            
            dfs = pd.read_html(StringIO(res.text))
            if not dfs: continue
            df = dfs[0]
            
            # 🔥 關鍵修正:精確欄位對應
            col_map = {}
            
            for col in df.columns:
                col_str = str(col).strip()
                
                # OI: 必須先檢查 (避免被Month誤判)
                if '未沖銷' in col_str and '契約量' in col_str:
                    col_map['OI'] = col
                
                # Month: 到期月份(週別) 或第一個包含"契約"的欄位
                elif '到期月份' in col_str or '週別' in col_str:
                    col_map['Month'] = col
                elif col_str == '契約' and 'Month' not in col_map:
                    col_map['Month'] = col
                
                # Strike: 履約價
                elif '履約價' in col_str:
                    col_map['Strike'] = col
                
                # Type: 買賣權
                elif '買賣權' in col_str:
                    col_map['Type'] = col
                
                # Price: 結算價優先,其次收盤價
                elif '結算價' in col_str:
                    col_map['Price'] = col
                elif '收盤價' in col_str and 'Price' not in col_map:
                    col_map['Price'] = col
            
            # 驗證是否找到所有必要欄位
            required = ['Month', 'Strike', 'Type', 'OI', 'Price']
            if not all(k in col_map for k in required):
                continue
            
            # 重新命名欄位
            df_renamed = df.rename(columns={v: k for k, v in col_map.items()})
            df_clean = df_renamed[required].dropna(subset=['Type'])
            
            # 資料清理
            df_clean['Type'] = df_clean['Type'].astype(str).str.strip()
            df_clean['Strike'] = pd.to_numeric(df_clean['Strike'].astype(str).str.replace(',', ''), errors='coerce')
            df_clean['OI'] = pd.to_numeric(df_clean['OI'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df_clean['Price'] = pd.to_numeric(df_clean['Price'].astype(str).str.replace(',', '').replace('-', '0'), errors='coerce').fillna(0)
            df_clean['Amount'] = df_clean['OI'] * df_clean['Price'] * 50
            
            if df_clean['OI'].sum() > 0 and len(df_clean) > 10:
                all_data.append({'date': query_date, 'df': df_clean})
                if len(all_data) >= days: break
        except Exception as e:
            continue
            
    return all_data if len(all_data) >= 1 else None

# 數學計算函數
def calculate_iv(option_price, spot_price, strike, time_to_expiry, option_type='call', risk_free_rate=0.015):
    if option_price <= 0 or spot_price <= 0 or strike <= 0 or time_to_expiry <= 0: return None
    sigma = 0.3
    for i in range(50):
        d1 = (np.log(spot_price / strike) + (risk_free_rate + 0.5 * sigma ** 2) * time_to_expiry) / (sigma * np.sqrt(time_to_expiry))
        d2 = d1 - sigma * np.sqrt(time_to_expiry)
        if option_type == 'call': price = spot_price * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
        else: price = strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
        vega = spot_price * norm.pdf(d1) * np.sqrt(time_to_expiry)
        if vega == 0 or abs(price - option_price) < 1e-4: return sigma
        sigma -= (price - option_price) / vega
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
        if gex_data: return pd.DataFrame(gex_data).groupby('Strike')['GEX'].sum().reset_index()
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

# 圖表繪製函數
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

    data['Put_Text'] = ""
    data['Call_Text'] = ""
    if 'OI_Change_D1' in df_target.columns:
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

# AI 相關函數
def prepare_ai_data(df, inst_opt_data, inst_fut, futures_price, spot_price, basis, atm_iv, risk_reversal, gex_summary, data_date):
    df_ai = df.nlargest(30, 'Amount') if 'Amount' in df.columns else df
    cols = [c for c in ['Strike','Type','OI','Amount','OI_Change_D1'] if c in df_ai.columns]
    
    # 選擇權法人資料格式化
    inst_opt_str = ""
    if inst_opt_data and isinstance(inst_opt_data, dict):
        for inst in ['外資', '投信', '自營商']:
            if inst in inst_opt_data and isinstance(inst_opt_data[inst], dict):
                data = inst_opt_data[inst]
                call_net = data.get('Call', 0)
                put_net = data.get('Put', 0)
                inst_opt_str += f"{inst}: Call {call_net:+,} | Put {put_net:+,}\n"
    
    # 期貨法人資料格式化
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

# 主程式
def main():
    if 'analysis_unlocked' not in st.session_state: st.session_state.analysis_unlocked = False
    if 'show_analysis_results' not in st.session_state: st.session_state.show_analysis_results = False
    inject_adsense_head()
    
    st.title("🧛‍♂️ 台指期籌碼戰情室 (莊家控盤版 - 欄位修正)")
    
    if st.sidebar.button("🔄 重新整理"):
        st.cache_data.clear()
        st.session_state.show_analysis_results = False
        st.rerun()
    
    st.sidebar.caption(f"Gemini: {'✅' if gemini_model else '❌'} | ChatGPT: {'✅' if openai_client else '❌'}")
    
    # 🆕 手動輸入現貨點數
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 手動設定現貨")
    manual_spot = st.sidebar.number_input(
        "輸入當前大盤點數 (選填)",
        min_value=0,
        max_value=30000,
        value=0,
        step=10,
        help="若自動抓取有延遲或收盤後,可手動輸入。輸入 0 則使用自動抓取值"
    )

    with st.spinner("🔄 正在搜尋最新數據..."):
        taiex_now = get_realtime_data()
        futures_price, futures_volume, fut_date = get_futures_data()
        inst_fut_position = get_institutional_futures_position()
        inst_opt_data = get_institutional_option_data()
        all_option_data = get_option_data_multi_days(days=2)
    
    # 🆕 如果有手動輸入,使用手動值覆蓋自動抓取值
    if manual_spot > 0:
        taiex_now = manual_spot
        st.sidebar.success(f"✅ 使用手動輸入: {int(manual_spot)} 點")
    elif taiex_now:
        st.sidebar.info(f"ℹ️ 自動抓取: {int(taiex_now)} 點")
    else:
        st.sidebar.warning("⚠️ 無法取得現貨價格,請手動輸入")

    if not all_option_data:
        st.error("❌ 無法抓取選擇權數據")
        return

    # 數據處理
    df_full = calculate_multi_day_oi_change(all_option_data)
    data_date = all_option_data[0]['date']
    basis = (futures_price - taiex_now) if (taiex_now and futures_price) else None
    
    st.sidebar.download_button("📥 下載數據", df_full.to_csv(index=False).encode('utf-8-sig'), "opt_data.csv")

    # === 儀表板 ===
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.caption(f"更新時間: {datetime.now(tz=TW_TZ).strftime('%H:%M:%S')}")
    
    # 🆕 根據數據來源顯示不同標籤
    spot_label = "加權指數 "
    if manual_spot > 0:
        spot_label += "(手動)"
    elif taiex_now:
        spot_label += "(即時)"
    else:
        spot_label += "(無數據)"
    
    c2.metric(spot_label, f"{int(taiex_now) if taiex_now else 'N/A'}")
    c3.metric(f"台指期 ({fut_date[5:]})", f"{int(futures_price) if futures_price else 'N/A'}")
    c4.metric("基差", f"{basis:.0f}" if basis else "N/A", delta_color="normal" if basis and basis > 0 else "inverse")
    
    call_amt = df_full[df_full['Type'].str.contains('Call|買')]['Amount'].sum()
    put_amt = df_full[df_full['Type'].str.contains('Put|賣')]['Amount'].sum()
    pc_ratio = (put_amt / call_amt * 100) if call_amt > 0 else 0
    c5.metric(f"P/C 金額比 ({data_date[5:]})", f"{pc_ratio:.1f}%", "偏多" if pc_ratio > 100 else "偏空")
    
    st.markdown("---")
    
    # === 法人籌碼區 ===
    st.markdown("### 🏦 三大法人籌碼佈局")
    
    # 建立統一的籌碼表格
    institutional_display = []
    
    # 收集期貨數據
    fut_data_date = "N/A"
    if inst_fut_position:
        fut_data_date = inst_fut_position.get('date', 'N/A')
        for inst in ['外資', '投信', '自營商']:
            val = inst_fut_position.get(inst, 0)
            direction = "🟢 偏多" if val > 0 else "🔴 偏空" if val < 0 else "⚪ 中性"
            
            institutional_display.append({
                '法人': inst,
                '期貨淨單': f"{val:+,} 口",
                '期貨傾向': direction,
                'Call淨單': '-',
                'Put淨單': '-',
                '選擇權策略': '-'
            })
    
    # 收集選擇權數據
    opt_data_date = "N/A"
    if inst_opt_data and 'date' in inst_opt_data:
        opt_data_date = inst_opt_data.get('date', 'N/A')
        
        for idx, inst in enumerate(['外資', '投信', '自營商']):
            if inst in inst_opt_data:
                data = inst_opt_data[inst]
                call_net = data.get('Call', 0)
                put_net = data.get('Put', 0)
                
                # 計算策略傾向
                if call_net > 0 and put_net > 0:
                    strategy = "🔵 做多波動 (買雙CALL+PUT)"
                elif call_net < 0 and put_net < 0:
                    strategy = "🟠 做空波動 (賣雙CALL+PUT)"
                elif call_net > 0 > put_net:
                    strategy = "🟢 看多 (買CALL+賣PUT)"
                elif put_net > 0 > call_net:
                    strategy = "🔴 看空 (買PUT+賣CALL)"
                else:
                    strategy = "⚪ 中性"
                
                # 如果已有期貨數據,更新對應列
                if inst_fut_position and idx < len(institutional_display):
                    institutional_display[idx]['Call淨單'] = f"{call_net:+,} 口"
                    institutional_display[idx]['Put淨單'] = f"{put_net:+,} 口"
                    institutional_display[idx]['選擇權策略'] = strategy
                # 否則新增列
                else:
                    institutional_display.append({
                        '法人': inst,
                        '期貨淨單': '-',
                        '期貨傾向': '-',
                        'Call淨單': f"{call_net:+,} 口",
                        'Put淨單': f"{put_net:+,} 口",
                        '選擇權策略': strategy
                    })
    
    # 顯示統一表格
    if institutional_display:
        st.caption(f"📅 期貨籌碼日期: {fut_data_date} | 選擇權籌碼日期: {opt_data_date}")
        st.dataframe(
            pd.DataFrame(institutional_display), 
            use_container_width=True, 
            hide_index=True
        )
    else:
        st.warning("⚠️ 查無法人籌碼數據")
    
    st.markdown("---")
    
    # === 選擇權 OI 龍捲風圖 ===
    st.markdown("### 📊 選擇權未平倉分佈 (Put支撐 vs Call壓力)")
    
    next_contracts = get_next_contracts(df_full, data_date)
    
    if len(next_contracts) >= 2:
        tab1, tab2 = st.tabs([f"近月 {next_contracts[0]['code']} (結算:{next_contracts[0]['date']})", 
                               f"次月 {next_contracts[1]['code']} (結算:{next_contracts[1]['date']})"])
        
        with tab1:
            df_near = df_full[df_full['Month'] == next_contracts[0]['code']]
            if not df_near.empty:
                fig1 = plot_tornado_chart(df_near, f"近月合約 {next_contracts[0]['code']}", taiex_now)
                st.plotly_chart(fig1, use_container_width=True)
                
                # GEX 分析
                gex_near = calculate_dealer_gex(df_near, taiex_now, next_contracts[0]['date'])
                if gex_near is not None:
                    st.markdown("#### Dealer Gamma Exposure (GEX)")
                    fig_gex = plot_gex_chart(gex_near, taiex_now)
                    if fig_gex: st.plotly_chart(fig_gex, use_container_width=True)
        
        with tab2:
            df_far = df_full[df_full['Month'] == next_contracts[1]['code']]
            if not df_far.empty:
                fig2 = plot_tornado_chart(df_far, f"次月合約 {next_contracts[1]['code']}", taiex_now)
                st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # === AI 分析區 ===
    st.markdown("### 🤖 AI 莊家控盤分析")
    
    if not gemini_model and not openai_client:
        st.error("❌ 未設定 AI API Key,無法使用分析功能")
    else:
        col_ai1, col_ai2 = st.columns(2)
        
        with col_ai1:
            if st.button("🔮 Gemini 分析", disabled=not gemini_model, use_container_width=True):
                st.session_state.show_analysis_results = True
                st.session_state.ai_provider = 'gemini'
        
        with col_ai2:
            if st.button("💬 ChatGPT 分析", disabled=not openai_client, use_container_width=True):
                st.session_state.show_analysis_results = True
                st.session_state.ai_provider = 'chatgpt'
        
        if st.session_state.show_analysis_results:
            # 準備分析數據
            df_near = df_full[df_full['Month'] == next_contracts[0]['code']] if next_contracts else df_full
            atm_iv, risk_reversal, atm_strike = calculate_risk_reversal(df_near, taiex_now, next_contracts[0]['date']) if next_contracts else (None, None, None)
            gex_summary = calculate_dealer_gex(df_near, taiex_now, next_contracts[0]['date']) if next_contracts else None
            
            ai_data = prepare_ai_data(
                df_near, inst_opt_data, inst_fut_position, 
                futures_price, taiex_now, basis, 
                atm_iv, risk_reversal, gex_summary, data_date
            )
            
            prompt = build_ai_prompt(ai_data, taiex_now)
            
            with st.spinner(f"🤖 {st.session_state.ai_provider.upper()} 分析中..."):
                if st.session_state.ai_provider == 'gemini':
                    result = ask_gemini(prompt)
                else:
                    result = ask_chatgpt(prompt)
                
                st.markdown("#### 📊 AI 分析結果")
                st.markdown(result)
    
    # 廣告區
    st.markdown("---")
    show_ad_placeholder()

if __name__ == "__main__":
    main()
