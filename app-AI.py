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

# 忽略 SSL 警告 (必要)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="台指期籌碼戰情室 (除錯版)")
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

# --- 模型設定 (省略) ---
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

# --- 🔥 帶詳細日誌的選擇權數據抓取 ---
@st.cache_data(ttl=300)
def get_option_data_multi_days_debug(days=3):
    """獲取選擇權全市場數據 (帶詳細調試)"""
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    headers = {'User-Agent': 'Mozilla/5.0'}
    all_data = []
    debug_log = []

    for i in range(10):  # 只測試 10 天,加快速度
        target_date = datetime.now(tz=TW_TZ) - timedelta(days=i)
        query_date = target_date.strftime('%Y/%m/%d')
        
        debug_log.append(f"嘗試日期 {i}: {query_date}")
        
        payload = {
            'queryType': '2',
            'marketCode': '0',
            'commodity_id': 'TXO',
            'queryDate': query_date,
            'MarketCode': '0',
            'commodity_idt': 'TXO'
        }
        
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=10, verify=False)
            res.encoding = 'utf-8'
            
            debug_log.append(f"  狀態碼: {res.status_code}")
            debug_log.append(f"  內容長度: {len(res.text)} 字元")
            
            if "查無資料" in res.text:
                debug_log.append(f"  ❌ 查無資料")
                continue
            
            if len(res.text) < 500:
                debug_log.append(f"  ❌ 內容過短")
                continue
            
            dfs = pd.read_html(StringIO(res.text))
            debug_log.append(f"  ✅ 找到 {len(dfs)} 個表格")
            
            if not dfs:
                continue
                
            df = dfs[0]
            debug_log.append(f"  表格大小: {df.shape}")
            debug_log.append(f"  欄位: {list(df.columns)[:3]}")
            
            df.columns = [str(c).replace(' ', '').replace('*', '').replace('契約', '').strip() for c in df.columns]
            
            col_map = {
                'Month': next((c for c in df.columns if '月' in c or '週' in c), None),
                'Strike': next((c for c in df.columns if '履約' in c), None),
                'Type': next((c for c in df.columns if '買賣' in c), None),
                'OI': next((c for c in df.columns if '未沖銷' in c or 'OI' in c), None),
                'Price': next((c for c in df.columns if '結算' in c or '收盤' in c or 'Price' in c), None)
            }
            
            if not all(col_map.values()):
                debug_log.append(f"  ❌ 欄位不完整: {col_map}")
                continue
            
            df = df.rename(columns={k:v for k,v in col_map.items() if v})[['Month', 'Strike', 'Type', 'OI', 'Price']].dropna(subset=['Type'])
            df['Type'] = df['Type'].astype(str).str.strip()
            df['Strike'] = pd.to_numeric(df['Strike'].astype(str).str.replace(',', ''), errors='coerce')
            df['OI'] = pd.to_numeric(df['OI'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace(',', '').replace('-', '0'), errors='coerce').fillna(0)
            df['Amount'] = df['OI'] * df['Price'] * 50
            
            if df['OI'].sum() > 0 and len(df) > 10:
                debug_log.append(f"  ✅✅✅ 成功! OI 總和: {df['OI'].sum()}")
                all_data.append({'date': query_date, 'df': df})
                if len(all_data) >= days:
                    break
            else:
                debug_log.append(f"  ❌ 數據不足 (OI={df['OI'].sum()}, rows={len(df)})")
        except Exception as e:
            debug_log.append(f"  ❌ 錯誤: {str(e)[:100]}")
            continue
    
    # 返回數據和調試日誌
    return all_data if len(all_data) >= 1 else None, debug_log

# --- 簡化的主程式 (只測試數據抓取) ---
def main():
    st.title("🔍 期交所 API 除錯工具")
    st.write("這個版本會顯示詳細的調試信息")
    
    if st.button("🧪 開始測試抓取數據"):
        with st.spinner("測試中..."):
            result, debug_log = get_option_data_multi_days_debug(days=2)
        
        st.markdown("### 📊 調試日誌")
        for log in debug_log:
            st.text(log)
        
        st.markdown("---")
        
        if result:
            st.success(f"✅ 成功抓到 {len(result)} 天的數據!")
            st.write(f"數據日期: {[d['date'] for d in result]}")
            st.write(f"第一天資料筆數: {len(result[0]['df'])}")
            st.dataframe(result[0]['df'].head(10))
        else:
            st.error("❌ 無法抓到任何數據")
            st.write("**可能原因:**")
            st.write("1. 今天期交所尚未更新數據")
            st.write("2. Streamlit Cloud IP 被封鎖")
            st.write("3. SSL 憑證問題")
            st.write("4. 網路超時")
            
            st.markdown("### 💡 解決建議")
            st.write("請手動到期交所網站測試:")
            st.markdown("[https://www.taifex.com.tw/cht/3/optDailyMarketReport](https://www.taifex.com.tw/cht/3/optDailyMarketReport)")

if __name__ == "__main__":
    main()
