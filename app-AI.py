import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from io import StringIO
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(layout="wide", page_title="超詳細調試")
TW_TZ = timezone(timedelta(hours=8))

st.title("🔬 超詳細調試工具")

if st.button("🧪 測試抓取"):
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    query_date = (datetime.now(tz=TW_TZ) - timedelta(days=0)).strftime('%Y/%m/%d')
    
    st.write(f"測試日期: {query_date}")
    
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
        
        st.success(f"✅ HTTP 狀態: {res.status_code}")
        st.info(f"內容長度: {len(res.text)} 字元")
        
        if "查無資料" in res.text:
            st.error("❌ 期交所回應: 查無資料")
        else:
            dfs = pd.read_html(StringIO(res.text))
            st.success(f"✅ 找到 {len(dfs)} 個表格")
            
            if dfs:
                df = dfs[0]
                
                st.markdown("### 📋 原始表格資訊")
                st.write(f"**表格大小:** {df.shape}")
                st.write(f"**資料筆數:** {len(df)}")
                
                st.markdown("### 📝 所有欄位名稱")
                for i, col in enumerate(df.columns):
                    st.text(f"{i}: {col}")
                
                st.markdown("### 🔍 前10筆原始資料")
                st.dataframe(df.head(10))
                
                st.markdown("### 🎯 欄位對應測試")
                
                # 測試欄位對應
                col_map = {}
                for col in df.columns:
                    col_str = str(col)
                    st.text(f"檢查欄位: {col_str}")
                    
                    if '到期月份' in col_str or '週別' in col_str or ('契約' in col_str and '到期' not in col_str and '日期' not in col_str):
                        col_map['Month'] = col
                        st.success(f"  ✅ Month = {col}")
                    elif '履約' in col_str:
                        col_map['Strike'] = col
                        st.success(f"  ✅ Strike = {col}")
                    elif '買賣權' in col_str or '買賣' in col_str:
                        col_map['Type'] = col
                        st.success(f"  ✅ Type = {col}")
                    elif '未沖銷' in col_str or '未平倉' in col_str or 'OI' in col_str:
                        col_map['OI'] = col
                        st.success(f"  ✅ OI = {col}")
                    elif '結算' in col_str or '收盤' in col_str:
                        col_map['Price'] = col
                        st.success(f"  ✅ Price = {col}")
                
                st.markdown("### 📊 對應結果")
                st.json(col_map)
                
                missing = []
                for key in ['Month', 'Strike', 'Type', 'OI', 'Price']:
                    if key not in col_map:
                        missing.append(key)
                
                if missing:
                    st.error(f"❌ 缺少欄位: {missing}")
                    st.warning("請告訴我實際的欄位名稱,我來修正對應邏輯!")
                else:
                    st.success("✅ 所有欄位都找到了!")
                    
                    # 嘗試重新命名
                    try:
                        df_renamed = df.rename(columns={v: k for k, v in col_map.items()})
                        df_clean = df_renamed[['Month', 'Strike', 'Type', 'OI', 'Price']].dropna(subset=['Type'])
                        
                        st.markdown("### ✅ 處理後的資料")
                        st.dataframe(df_clean.head(20))
                        st.success(f"成功處理 {len(df_clean)} 筆資料!")
                    except Exception as e:
                        st.error(f"處理失敗: {str(e)}")
                
    except Exception as e:
        st.error(f"❌ 錯誤: {str(e)}")
