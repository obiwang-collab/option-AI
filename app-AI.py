import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from io import StringIO
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(layout="wide", page_title="法人數據調試")
TW_TZ = timezone(timedelta(hours=8))

st.title("🔬 三大法人數據調試工具")

tab1, tab2 = st.tabs(["📈 法人期貨", "📊 法人選擇權"])

# ==========================================
# 法人期貨數據測試
# ==========================================
with tab1:
    st.markdown("### 📈 三大法人期貨淨部位")
    
    if st.button("🧪 測試法人期貨", key="fut"):
        url = "https://www.taifex.com.tw/cht/3/futContractsDate"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        query_date = (datetime.now(tz=TW_TZ) - timedelta(days=0)).strftime('%Y/%m/%d')
        st.write(f"測試日期: {query_date}")
        
        payload = {
            'queryType': '1',
            'goDay': '',
            'doDay': '',
            'queryDate': query_date,
            'commodityId': 'TXF'
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
                    
                    st.markdown("### 🔍 完整原始資料")
                    st.dataframe(df)
                    
                    st.markdown("### 🎯 尋找法人資料")
                    
                    inst_data = {}
                    for idx, row in df.iterrows():
                        row_str = " ".join([str(x) for x in row.values])
                        st.text(f"Row {idx}: {row_str[:100]}...")
                        
                        if '外資' in row_str or '外資及陸資' in row_str:
                            st.success(f"  ✅ 找到外資 (Row {idx})")
                            st.write(row.values)
                        elif '投信' in row_str:
                            st.success(f"  ✅ 找到投信 (Row {idx})")
                            st.write(row.values)
                        elif '自營商' in row_str:
                            st.success(f"  ✅ 找到自營商 (Row {idx})")
                            st.write(row.values)
                    
        except Exception as e:
            st.error(f"❌ 錯誤: {str(e)}")

# ==========================================
# 法人選擇權數據測試
# ==========================================
with tab2:
    st.markdown("### 📊 三大法人選擇權")
    
    if st.button("🧪 測試法人選擇權", key="opt"):
        url = "https://www.taifex.com.tw/cht/3/callsAndPutsDate"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        query_date = (datetime.now(tz=TW_TZ) - timedelta(days=0)).strftime('%Y/%m/%d')
        st.write(f"測試日期: {query_date}")
        
        payload = {
            'queryType': '1',
            'goDay': '',
            'doDay': '',
            'queryDate': query_date,
            'commodityId': 'TXO'
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
                    
                    st.markdown("### 🔍 完整原始資料")
                    st.dataframe(df)
                    
                    st.markdown("### 🎯 篩選法人資料")
                    
                    df_filtered = df[df.iloc[:, 0].astype(str).str.contains('自營商|投信|外資', na=False)]
                    
                    if not df_filtered.empty:
                        st.success(f"✅ 找到 {len(df_filtered)} 筆法人資料")
                        st.dataframe(df_filtered)
                    else:
                        st.warning("⚠️ 未找到法人資料")
                        st.info("嘗試其他篩選方式...")
                        
                        # 嘗試不同的篩選
                        for idx, row in df.iterrows():
                            row_str = " ".join([str(x) for x in row.values])
                            if '外資' in row_str or '投信' in row_str or '自營商' in row_str:
                                st.write(f"Row {idx}: {row.values}")
                    
        except Exception as e:
            st.error(f"❌ 錯誤: {str(e)}")

st.markdown("---")
st.markdown("### 💡 說明")
st.write("""
**法人期貨數據:**
- URL: https://www.taifex.com.tw/cht/3/futContractsDate
- commodityId: TXF (台指期貨)
- 需要找到: 外資、投信、自營商的淨部位

**法人選擇權數據:**
- URL: https://www.taifex.com.tw/cht/3/callsAndPutsDate
- commodityId: TXO (台指選擇權)
- 需要找到: 外資、投信、自營商的 Call/Put 部位
""")
