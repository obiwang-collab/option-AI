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

# --- 🧠 1. Gemini 模型設定 (修正版：解決 404 錯誤) ---
def get_gemini_model(api_key):
    if not api_key: return None, "未設定"
    genai.configure(api_key=api_key)
    try:
        # 1. 先取得所有支援 generateContent 的模型列表
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        target_model_name = None
        
        # 2. 定義優先搜尋順序 (包含 models/ 前綴以防萬一)
        priority_targets = [
            'gemini-1.5-flash',
            'gemini-1.5-pro',
            'gemini-pro',
            'flash'
        ]
        
        # 3. 匹配模型
        for target in priority_targets:
            for model_id in available_models:
                if target in model_id.lower():
                    target_model_name = model_id
                    break
            if target_model_name:
                break
        
        # 4. 如果都沒匹配到，但有可用模型，取第一個
        if not target_model_name and available_models:
            target_model_name = available_models[0]
            
        if target_model_name:
            return genai.GenerativeModel(target_model_name), target_model_name
        else:
            return None, "無可用模型 (ListModels Empty)"

    except Exception as e:
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

# --- 獲取台指期貨價格與基差 ---
@st.cache_data(ttl=60)
def get_futures_data():
    """獲取台指期貨價格、成交量"""
    url = "https://www.taifex.com.tw/cht/3/futContractsDate"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        query_date = datetime.now(tz=TW_TZ).strftime('%Y/%m/%d')
        payload = {
            'queryType': '1',
            'marketCode': '0',
            'commodity_id': 'TX',
            'queryDate': query_date
        }
        
        res = requests.post(url, data=payload, headers=headers, timeout=5)
        res.encoding = 'utf-8'
        
        if "查無資料" in res.text:
            return None, None
        
        dfs = pd.read_html(StringIO(res.text))
        if not dfs or len(dfs) == 0:
            return None, None
        
        df = dfs[0]
        # 取第一筆（近月合約）
        if len(df) > 0:
            futures_price = None
            volume = None
            
            # 嘗試找到收盤價欄位
            for col in df.columns:
                if '收盤價' in str(col) or '成交價' in str(col):
                    try:
                        futures_price = float(str(df.iloc[0][col]).replace(',', ''))
                    except:
                        pass
                if '成交量' in str(col):
                    try:
                        volume = int(str(df.iloc[0][col]).replace(',', ''))
                    except:
                        pass
            
            return futures_price, volume
        
    except Exception as e:
        pass
    
    return None, None

# --- 🔥(修改版) 三大法人期貨部位 - 自動回溯尋找最近資料 ---
@st.cache_data(ttl=300)
def get_institutional_futures_position():
    """獲取三大法人台指期貨淨部位 (自動回溯最近10天，防止假日或盤中抓不到)"""
    url = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for i in range(10): # 嘗試回溯 10 天
        target_date = datetime.now(tz=TW_TZ) - timedelta(days=i)
        
        # 簡單過濾周末 (週六=5, 週日=6)
        if target_date.weekday() >= 5:
            continue
            
        # 如果是今天(i=0)且時間小於15:00，跳過 (資料還沒出來)
        if i == 0 and datetime.now(tz=TW_TZ).hour < 15:
            continue
            
        query_date = target_date.strftime('%Y/%m/%d')
        
        payload = {
            'down_type': '1',
            'queryStartDate': query_date,
            'queryEndDate': query_date,
            'commodity_id': 'TX'
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
            
            # 尋找外資、自營商、投信的淨部位
            institutional_data = {}
            for idx, row in df.iterrows():
                row_str = str(row.iloc[0])
                if '外資' in row_str or '外資及陸資' in row_str:
                    try:
                        for col in df.columns:
                            if '買賣差額' in str(col) or '淨額' in str(col):
                                net_position = int(str(row[col]).replace(',', ''))
                                institutional_data['外資'] = net_position
                                break
                    except: pass
                elif '自營商' in row_str:
                    try:
                        for col in df.columns:
                            if '買賣差額' in str(col) or '淨額' in str(col):
                                net_position = int(str(row[col]).replace(',', ''))
                                institutional_data['自營商'] = net_position
                                break
                    except: pass
                elif '投信' in row_str:
                    try:
                        for col in df.columns:
                            if '買賣差額' in str(col) or '淨額' in str(col):
                                net_position = int(str(row[col]).replace(',', ''))
                                institutional_data['投信'] = net_position
                                break
                    except: pass
            
            if institutional_data:
                institutional_data['date'] = query_date # 紀錄資料日期
                return institutional_data
            
        except Exception as e:
            continue

    return None

# --- 三大法人選擇權籌碼數據獲取 ---
@st.cache_data(ttl=300)
def get_institutional_option_data():
    """獲取三大法人選擇權籌碼數據（最近兩天）"""
    url = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
    headers = {'User-Agent': 'Mozilla/5.0'}
    all_inst_data = []
    
    for i in range(10):
        query_date = (datetime.now(tz=TW_TZ) - timedelta(days=i)).strftime('%Y/%m/%d')
        
        # 簡單過濾周末
        target_dt = datetime.now(tz=TW_TZ) - timedelta(days=i)
        if target_dt.weekday() >= 5: continue
        if i == 0 and datetime.now(tz=TW_TZ).hour < 15: continue # 盤中不抓今天

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

# --- 獲取近三日完整選擇權數據（全履約價） ---
@st.cache_data(ttl=300)
def get_option_data_multi_days(days=3):
    """獲取近 N 天的完整選擇權數據（全履約價）"""
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    headers = {'User-Agent': 'Mozilla/5.0'}
    all_data = []

    for i in range(15):  # 嘗試更多天以確保獲取足夠數據
        target_dt = datetime.now(tz=TW_TZ) - timedelta(days=i)
        if target_dt.weekday() >= 5: continue # Skip weekend
        if i == 0 and datetime.now(tz=TW_TZ).hour < 15: continue # Skip today if pre-market close

        query_date = target_dt.strftime('%Y/%m/%d')
        payload = {
            'queryType': '2',
            'marketCode': '0',
            'dateaddcnt': '',
            'commodity_id': 'TXO',
            'commodity_id2': '',
            'queryDate': query_date,
            'MarketCode': '0',
            'commodity_idt': 'TXO'
        }
        
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=5)
            res.encoding = 'utf-8'
            
            if "查無資料" in res.text or len(res.text) < 500:
                continue
            
            dfs = pd.read_html(StringIO(res.text))
            df = dfs[0]
            
            df.columns = [str(c).replace(' ', '').replace('*', '').replace('契約', '').strip() for c in df.columns]
            
            month_col = next((c for c in df.columns if '月' in c or '週' in c), None)
            strike_col = next((c for c in df.columns if '履約' in c), None)
            type_col = next((c for c in df.columns if '買賣' in c), None)
            oi_col = next((c for c in df.columns if '未沖銷' in c or 'OI' in c), None)
            price_col = next((c for c in df.columns if '結算' in c or '收盤' in c or 'Price' in c), None)
            
            if not all([month_col, strike_col, type_col, oi_col, price_col]): continue
            
            rename_dict = {
                month_col: 'Month', strike_col: 'Strike', type_col: 'Type', oi_col: 'OI', price_col: 'Price'
            }
            df = df.rename(columns=rename_dict)
            df = df[['Month', 'Strike', 'Type', 'OI', 'Price']].copy().dropna(subset=['Type'])
            
            df['Type'] = df['Type'].astype(str).str.strip()
            df['Strike'] = pd.to_numeric(df['Strike'].astype(str).str.replace(',', ''), errors='coerce')
            df['OI'] = pd.to_numeric(df['OI'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace(',', '').replace('-', '0'), errors='coerce').fillna(0)
            df['Amount'] = df['OI'] * df['Price'] * 50
            
            if df['OI'].sum() == 0: continue
            
            all_data.append({'date': query_date, 'df': df})
            if len(all_data) >= days: break
                
        except Exception as e: continue
    
    return all_data if len(all_data) >= days else None

# --- 計算隱含波動率 (IV) - Black-Scholes 反推 ---
def calculate_iv(option_price, spot_price, strike, time_to_expiry, option_type='call', risk_free_rate=0.015):
    if option_price <= 0 or spot_price <= 0 or strike <= 0 or time_to_expiry <= 0: return None
    sigma = 0.3
    for i in range(100):
        d1 = (np.log(spot_price / strike) + (risk_free_rate + 0.5 * sigma ** 2) * time_to_expiry) / (sigma * np.sqrt(time_to_expiry))
        d2 = d1 - sigma * np.sqrt(time_to_expiry)
        if option_type == 'call':
            price = spot_price * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
            vega = spot_price * norm.pdf(d1) * np.sqrt(time_to_expiry)
        else:
            price = strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
            vega = spot_price * norm.pdf(d1) * np.sqrt(time_to_expiry)
        diff = price - option_price
        if abs(diff) < 1e-5: return sigma
        if vega == 0: return None
        sigma = sigma - diff / vega
        if sigma <= 0: return None
    return None

# --- 計算 Delta 和 Gamma ---
def calculate_greeks(spot_price, strike, time_to_expiry, volatility, option_type='call', risk_free_rate=0.015):
    if volatility is None or volatility <= 0 or time_to_expiry <= 0: return None, None
    try:
        d1 = (np.log(spot_price / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        if option_type == 'call': delta = norm.cdf(d1)
        else: delta = norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (spot_price * volatility * np.sqrt(time_to_expiry))
        return delta, gamma
    except: return None, None

# --- 計算 Dealer Gamma Exposure (GEX) ---
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
            gex_df = pd.DataFrame(gex_data)
            gex_summary = gex_df.groupby('Strike')['GEX'].sum().reset_index()
            return gex_summary
        else: return None
    except Exception as e: return None

# --- 計算 25 Delta Risk Reversal (Skew) ---
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
                    if delta is not None:
                        iv_delta_data.append({'Strike': strike, 'Type': option_type, 'IV': iv, 'Delta': abs(delta)})
        if not iv_delta_data: return None, None, None
        iv_df = pd.DataFrame(iv_delta_data)
        call_25d = iv_df[(iv_df['Type'] == 'call') & (iv_df['Delta'] > 0.2) & (iv_df['Delta'] < 0.3)]
        put_25d = iv_df[(iv_df['Type'] == 'put') & (iv_df['Delta'] > 0.2) & (iv_df['Delta'] < 0.3)]
        atm_options = iv_df[iv_df['Strike'] == atm_strike]
        atm_iv = atm_options['IV'].mean() if not atm_options.empty else None
        if not call_25d.empty and not put_25d.empty:
            rr = call_25d.iloc[0]['IV'] - put_25d.iloc[0]['IV']
            return atm_iv, rr, atm_strike
        else: return atm_iv, None, atm_strike
    except Exception as e: return None, None, None

# --- 計算多日 OI 變化 ---
def calculate_multi_day_oi_change(all_data):
    if not all_data or len(all_data) < 2: return None
    df_latest = all_data[0]['df'].copy()
    for i in range(1, min(3, len(all_data))):
        df_prev = all_data[i]['df'].copy()
        df_merged = pd.merge(df_latest[['Month', 'Strike', 'Type', 'OI']], df_prev[['Month', 'Strike', 'Type', 'OI']], on=['Month', 'Strike', 'Type'], how='left', suffixes=('', f'_D{i}')).fillna(0)
        df_latest[f'OI_Change_D{i}'] = df_merged['OI'] - df_merged[f'OI_D{i}']
    return df_latest

# --- 修正圖表函式 ---
def plot_tornado_chart(df_target, title_text, spot_price):
    is_call = df_target['Type'].str.contains('買|Call', case=False, na=False)
    df_call = df_target[is_call][['Strike', 'OI', 'Amount']].rename(columns={'OI': 'Call_OI', 'Amount': 'Call_Amt'})
    df_put = df_target[~is_call][['Strike', 'OI', 'Amount']].rename(columns={'OI': 'Put_OI', 'Amount': 'Put_Amt'})
    data = pd.merge(df_call, df_put, on='Strike', how='outer').fillna(0).sort_values('Strike')
    total_put_money = data['Put_Amt'].sum()
    total_call_money = data['Call_Amt'].sum()
    FOCUS_RANGE = 1500
    center_price = spot_price if (spot_price and spot_price > 0) else data['Strike'].median()
    if center_price > 0:
        min_s = center_price - FOCUS_RANGE
        max_s = center_price + FOCUS_RANGE
        data = data[(data['Strike'] >= min_s) & (data['Strike'] <= max_s)]
    max_oi = max(data['Put_OI'].max(), data['Call_OI'].max()) if not data.empty else 1000
    x_limit = max_oi * 1.1
    if 'OI_Change_D1' in df_target.columns:
        data_with_change = df_target[['Strike', 'Type', 'OI_Change_D1']].copy()
        call_changes = data_with_change[data_with_change['Type'].str.contains('Call|買', case=False, na=False)].groupby('Strike')['OI_Change_D1'].sum().reset_index().rename(columns={'OI_Change_D1': 'Call_Change'})
        put_changes = data_with_change[~data_with_change['Type'].str.contains('Call|買', case=False, na=False)].groupby('Strike')['OI_Change_D1'].sum().reset_index().rename(columns={'OI_Change_D1': 'Put_Change'})
        data = data.merge(call_changes, on='Strike', how='left').fillna(0).merge(put_changes, on='Strike', how='left').fillna(0)
        data['Put_Text'] = data.apply(lambda row: f"{'+' if row['Put_Change'] > 0 else ''}{int(row['Put_Change'])}" if row['Put_OI'] > 0 else "", axis=1)
        data['Call_Text'] = data.apply(lambda row: f"{'+' if row['Call_Change'] > 0 else ''}{int(row['Call_Change'])}" if row['Call_OI'] > 0 else "", axis=1)
    else:
        data['Put_Text'] = ""; data['Call_Text'] = ""

    fig = go.Figure()
    fig.add_trace(go.Bar(y=data['Strike'], x=-data['Put_OI'], orientation='h', name='Put (支撐)', marker_color='#2ca02c', opacity=0.85, customdata=data['Put_Amt'] / 100000000, hovertemplate='<b>履約價: %{y}</b><br>Put OI: %{x} 口<br>Put 市值: %{customdata:.2f}億<extra></extra>', text=data['Put_Text'], textposition='outside', cliponaxis=False))
    fig.add_trace(go.Bar(y=data['Strike'], x=data['Call_OI'], orientation='h', name='Call (壓力)', marker_color='#d62728', opacity=0.85, customdata=data['Call_Amt'] / 100000000, hovertemplate='<b>履約價: %{y}</b><br>Call OI: %{x} 口<br>Call 市值: %{customdata:.2f}億<extra></extra>', text=data['Call_Text'], textposition='outside', cliponaxis=False))
    annotations = []
    if spot_price and spot_price > 0:
        if not data.empty and data['Strike'].min() <= spot_price <= data['Strike'].max():
            fig.add_hline(y=spot_price, line_dash="dash", line_color="#ff7f0e", line_width=2)
            annotations.append(dict(x=1.05, y=spot_price, xref="paper", yref="y", text=f" 現貨 {int(spot_price)} ", showarrow=False, xanchor="left", align="center", font=dict(color="white", size=12), bgcolor="#ff7f0e", bordercolor="#ff7f0e", borderpad=4))
    annotations.append(dict(x=0.02, y=1.05, xref="paper", yref="paper", text=f"<b>Put 總金額</b><br>{total_put_money/100000000:.1f} 億", showarrow=False, align="left", font=dict(size=14, color="#2ca02c"), bgcolor="white", bordercolor="#2ca02c", borderwidth=2, borderpad=6))
    annotations.append(dict(x=0.98, y=1.05, xref="paper", yref="paper", text=f"<b>Call 總金額</b><br>{total_call_money/100000000:.1f} 億", showarrow=False, align="right", font=dict(size=14, color="#d62728"), bgcolor="white", bordercolor="#d62728", borderwidth=2, borderpad=6))
    fig.update_layout(title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='top', font=dict(size=20, color="black")), xaxis=dict(title='未平倉量 (OI)', range=[-x_limit, x_limit], showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='black'), yaxis=dict(title='履約價', tickmode='linear', dtick=100, tickformat='d'), barmode='overlay', legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"), height=750, margin=dict(l=40, r=100, t=140, b=60), annotations=annotations, paper_bgcolor='white', plot_bgcolor='white')
    return fig

# --- 繪製 GEX 圖表 ---
def plot_gex_chart(gex_df, spot_price):
    if gex_df is None or gex_df.empty: return None
    fig = go.Figure()
    colors = ['green' if x > 0 else 'red' for x in gex_df['GEX']]
    fig.add_trace(go.Bar(x=gex_df['Strike'], y=gex_df['GEX'], marker_color=colors, name='Dealer GEX', hovertemplate='<b>履約價: %{x}</b><br>GEX: %{y:.2f}<extra></extra>'))
    if spot_price: fig.add_vline(x=spot_price, line_dash="dash", line_color="orange", line_width=2)
    fig.update_layout(title="Dealer Gamma Exposure (GEX)", xaxis_title="履約價", yaxis_title="GEX", height=400, showlegend=False)
    return fig

# --- 資料準備函式 ---
def prepare_ai_data(df, inst_opt_today, inst_opt_yesterday, inst_fut, futures_price, spot_price, basis, atm_iv, risk_reversal, gex_summary):
    df_ai = df.copy()
    if 'Amount' in df_ai.columns: df_ai = df_ai.nlargest(30, 'Amount')
    keep_cols = [c for c in ['Strike', 'Type', 'OI', 'Amount'] if c in df_ai.columns]
    if 'OI_Change_D1' in df_ai.columns: keep_cols.append('OI_Change_D1')
    if 'OI_Change_D2' in df_ai.columns: keep_cols.append('OI_Change_D2')
    df_ai = df_ai[keep_cols]
    option_data_csv = df_ai.to_csv(index=False)
    price_info = f"現貨價格: {spot_price if spot_price else 'N/A'}\n期貨價格: {futures_price if futures_price else 'N/A'}\n基差: {basis if basis else 'N/A'}"
    institutional_opt_summary = ""
    if inst_opt_today is not None and not inst_opt_today.empty: institutional_opt_summary += f"\n【三大法人選擇權籌碼 - 最新】\n{inst_opt_today.to_string(index=False)}"
    if inst_opt_yesterday is not None and not inst_opt_yesterday.empty: institutional_opt_summary += f"\n\n【三大法人選擇權籌碼 - 前一日】\n{inst_opt_yesterday.to_string(index=False)}"
    institutional_fut_summary = ""
    if inst_fut:
        institutional_fut_summary = "\n【三大法人期貨淨部位】\n"
        for key, value in inst_fut.items(): institutional_fut_summary += f"{key}: {value:+,} 口\n"
    iv_info = ""
    if atm_iv: iv_info += f"\nATM 隱含波動率: {atm_iv*100:.2f}%"
    if risk_reversal: iv_info += f"\n25Δ Risk Reversal: {risk_reversal*100:.2f}%"
    gex_info = ""
    if gex_summary is not None and not gex_summary.empty:
        max_gex_row = gex_summary.loc[gex_summary['GEX'].abs().idxmax()]
        total_gex = gex_summary['GEX'].sum()
        gex_info = f"\n【Dealer Gamma Exposure】\n總 GEX: {total_gex:.2f}\n最大 GEX 履約價: {max_gex_row['Strike']} (GEX: {max_gex_row['GEX']:.2f})"
    return f"=== 價格資訊 ===\n{price_info}\n\n=== 選擇權未平倉籌碼分析 ===\n{option_data_csv}\n\n=== 三大法人動向（選擇權） ===\n{institutional_opt_summary}\n\n=== 三大法人動向（期貨） ===\n{institutional_fut_summary}\n\n=== 波動率與偏態 ===\n{iv_info}\n\n=== Dealer Gamma Exposure ===\n{gex_info}"

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

# --- Prompt ---
def build_ai_prompt(data_str, taiex_price, contract_info):
    contract_note = f"結算合約：{contract_info.get('code')}" if contract_info else ""
    prompt = f"""
    你是台指期市場的『理性鐵血莊家』(Ruthless Market Maker)。
    你的目標是：**透過籌碼優勢，讓賣方利潤最大化 (Max Pain)**。
    目前現貨：{taiex_price}。{contract_note}
    請根據下方數據進行【莊家控盤劇本】推演，格式：
    🎯 **莊家結算目標**：(點位或區間)
    🏦 **三大法人解讀**：(期貨與選擇權部位分析)
    📊 **波動率與情緒**：(IV, Skew, 恐慌程度)
    ⚡ **Gamma 壓力**：(GEX 分析)
    🩸 **散戶狙擊區**：(Kill Zone)
    ☠️ **控盤劇本**：(具體策略)
    完整數據：
    {data_str}
    """
    return prompt.strip()

# --- AI Call ---
def ask_gemini(prompt_text):
    if not gemini_model: return "⚠️ 未設定 Gemini Key"
    try:
        res = gemini_model.generate_content(prompt_text)
        return res.text
    except Exception as e: return f"Gemini 錯誤: {str(e)}"

def ask_chatgpt(prompt_text):
    if not openai_client: return "⚠️ 未設定 OpenAI Key"
    try:
        response = openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "你是期貨莊家。"}, {"role": "user", "content": prompt_text}], temperature=0.7)
        return response.choices[0].message.content
    except Exception as e: return f"ChatGPT 錯誤: {str(e)}"

# --- 主程式 ---
def main():
    if 'analysis_unlocked' not in st.session_state: st.session_state.analysis_unlocked = False
    if 'show_analysis_results' not in st.session_state: st.session_state.show_analysis_results = False
    inject_adsense_head()
    st.title("🧛‍♂️ 台指期籌碼戰情室 (莊家控盤完整版)")
    
    col_title, col_btn = st.columns([3, 1])
    if st.sidebar.button("🔄 重新整理"): 
        st.session_state.analysis_unlocked = False 
        st.session_state.show_analysis_results = False 
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(f"🔵 Gemini: {'✅' if gemini_model else '❌'}")
    st.sidebar.caption(f"🟢 ChatGPT: {'✅' if openai_client else '❌'}")

    with st.spinner('🔄 連線期交所中...正在獲取完整數據...'):
        all_option_data = get_option_data_multi_days(days=3)
        taiex_now = get_realtime_data()
        futures_price, futures_volume = get_futures_data()
        inst_fut_position = get_institutional_futures_position() # 已修正為自動回溯
        inst_opt_today, inst_opt_date_today, inst_opt_yesterday, inst_opt_date_yesterday = get_institutional_option_data()

    if all_option_data is None or len(all_option_data) < 2:
        st.error("查無資料。需至少取得兩天有效數據。")
        return

    df_full = calculate_multi_day_oi_change(all_option_data)
    df = df_full
    data_date = all_option_data[0]['date']
    basis = (futures_price - taiex_now) if (taiex_now and futures_price) else None
    
    # 數據下載
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.sidebar.download_button("📥 下載完整數據", csv, f"option_{data_date.replace('/', '')}_full.csv", "text/csv")
    
    # 主要指標
    total_call_amt = df[df['Type'].str.contains('買|Call', case=False, na=False)]['Amount'].sum()
    total_put_amt = df[df['Type'].str.contains('賣|Put', case=False, na=False)]['Amount'].sum()
    pc_ratio_amt = (total_put_amt / total_call_amt) * 100 if total_call_amt > 0 else 0
    
    c1, c2, c3, c4, c5 = st.columns([1, 0.8, 1, 0.8, 1])
    c1.markdown(f"<div style='text-align: left;'><span style='font-size: 14px; color: #555;'>製圖時間</span><br><span style='font-size: 18px; font-weight: bold;'>{datetime.now(tz=TW_TZ).strftime('%Y/%m/%d %H:%M:%S')}</span></div>", unsafe_allow_html=True)
    c2.metric("現貨", f"{int(taiex_now) if taiex_now else 'N/A'}")
    c3.metric("期貨", f"{int(futures_price) if futures_price else 'N/A'}")
    c4.metric("基差", f"{basis:+.1f}" if basis else "N/A", delta_color="normal" if basis and basis > 0 else "inverse")
    trend = "偏多" if pc_ratio_amt > 100 else "偏空"
    c5.metric("P/C 金額比", f"{pc_ratio_amt:.1f}%", f"{trend}", delta_color="normal" if pc_ratio_amt > 100 else "inverse")
    st.markdown("---")

    # ==========================================
    # 🔥🔥🔥 新增：三大法人籌碼佈局 (主畫面顯示) 🔥🔥🔥
    # ==========================================
    st.markdown("### 🏦 三大法人籌碼佈局")
    
    # 1. 期貨籌碼 (紅綠燈號顯示)
    if inst_fut_position:
        st.caption(f"期貨籌碼日期: {inst_fut_position.get('date', 'N/A')}")
        f1, f2, f3 = st.columns(3)
        
        # 外資
        f_val = inst_fut_position.get('外資', 0)
        f1.metric("外資期貨淨單", f"{f_val:+,} 口", delta_color="inverse" if f_val > 0 else "normal") # 台股紅多綠空(inverse: 正數紅)
        
        # 投信
        t_val = inst_fut_position.get('投信', 0)
        f2.metric("投信期貨淨單", f"{t_val:+,} 口", delta_color="inverse" if t_val > 0 else "normal")
        
        # 自營商
        d_val = inst_fut_position.get('自營商', 0)
        f3.metric("自營商期貨淨單", f"{d_val:+,} 口", delta_color="inverse" if d_val > 0 else "normal")
    else:
        st.warning("⚠️ 暫無法人期貨數據 (請檢查盤後資料是否已更新)")

    # 2. 選擇權籌碼 (表格顯示)
    if inst_opt_today is not None and not inst_opt_today.empty:
        with st.expander(f"📊 三大法人選擇權淨部位明細 (日期: {inst_opt_date_today})", expanded=True):
            st.dataframe(inst_opt_today, use_container_width=True)
    
    st.markdown("---")
    # ==========================================

    plot_targets = get_next_contracts(df, data_date)
    if plot_targets:
        nearest_contract = plot_targets[0]['info']
        df_nearest = df[df['Month'] == nearest_contract['code']]
        with st.spinner('🧮 計算進階指標...'):
            atm_iv, risk_reversal, atm_strike = calculate_risk_reversal(df_nearest, taiex_now if taiex_now else 23000, nearest_contract['date'])
            gex_summary = calculate_dealer_gex(df_nearest, taiex_now if taiex_now else 23000, nearest_contract['date'])
        
        st.markdown("### 📊 進階市場指標")
        col_iv1, col_iv2 = st.columns(2)
        with col_iv1:
            st.metric("ATM 隱含波動率", f"{atm_iv*100:.2f}%" if atm_iv else "N/A")
            st.caption(f"履約價: {atm_strike}")
        with col_iv2:
            st.metric("25Δ Risk Reversal", f"{risk_reversal*100:.2f}%" if risk_reversal is not None else "N/A")
            st.caption("看漲偏態" if risk_reversal and risk_reversal > 0 else "看跌偏態")
        
        if gex_summary is not None:
            st.markdown("### ⚡ Dealer Gamma Exposure (GEX)")
            gex_fig = plot_gex_chart(gex_summary, taiex_now)
            if gex_fig:
                st.plotly_chart(gex_fig, use_container_width=True)
                st.caption("🔍 正 GEX = 造市商買入支撐 | 負 GEX = 造市商賣出壓力")
    else:
        atm_iv = None; risk_reversal = None; gex_summary = None
    
    st.markdown("---")
    
    if st.session_state.analysis_unlocked:
        st.markdown("### 🎲 莊家控盤劇本 (雙 AI 完整預測)")
        if st.button("🧛‍♂️ 啟動 AI 控盤分析", type="primary"):
            st.session_state.show_analysis_results = True
            st.rerun()
    else:
        st.markdown("### 🔓 觀看廣告解鎖 AI 分析")
        show_ad_placeholder()
        st.markdown("---")
        if st.button("⏱️ 點此開始倒數解鎖 AI 分析功能", key="start_timer", type="secondary"):
            placeholder = st.empty()
            for i in range(8, 0, -1):
                placeholder.warning(f"⏳ 請勿離開頁面，分析功能將在 {i} 秒後自動解鎖...")
                time.sleep(1)
            st.session_state.analysis_unlocked = True
            placeholder.success("✅ AI 分析功能已解鎖！")
            st.rerun()

    if st.session_state.show_analysis_results:
        if not plot_targets: st.error("無法取得合約資訊")
        else:
            data_str = prepare_ai_data(df, inst_opt_today, inst_opt_yesterday, inst_fut_position, futures_price, taiex_now, basis, atm_iv, risk_reversal, gex_summary)
            prompt_text = build_ai_prompt(data_str, taiex_now, plot_targets[0]['info'])
            with st.spinner("🤖 AI 正在計算..."):
                gemini_result = ask_gemini(prompt_text) if gemini_model else "未設定"
                chatgpt_result = ask_chatgpt(prompt_text) if openai_client else "未設定"
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔵 Google Gemini")
                st.info(gemini_result)
            with col2:
                st.subheader("🟢 ChatGPT")
                st.success(chatgpt_result)
    
    st.markdown("---")
    st.markdown("### 📈 選擇權未平倉分布")
    if plot_targets:
        cols = st.columns(len(plot_targets))
        for i, target in enumerate(plot_targets):
            with cols[i]:
                m_code = target['info']['code']
                s_date = target['info']['date']
                df_target = df[df['Month'] == m_code]
                sub_call = df_target[df_target['Type'].str.contains('Call|買', case=False, na=False)]['Amount'].sum()
                sub_put = df_target[df_target['Type'].str.contains('Put|賣', case=False, na=False)]['Amount'].sum()
                sub_ratio = (sub_put / sub_call * 100) if sub_call > 0 else 0
                title_text = (f"<b>{m_code}</b><br>結算: {s_date}<br>P/C金額比: {sub_ratio:.1f}%")
                st.plotly_chart(plot_tornado_chart(df_target, title_text, taiex_now), use_container_width=True)

if __name__ == "__main__":
    main()
