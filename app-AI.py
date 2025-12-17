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
        query_date = (datetime.now(tz=TW_TZ) - timedelta(days=0)).strftime('%Y/%m/%d')
        st.write(f"測試日期: {query_date}")
        
        # 測試多個 URL 和參數組合
        test_configs = [
            {
                'name': '測試1: futContractsDate + TXF',
                'url': "https://www.taifex.com.tw/cht/3/futContractsDate",
                'payload': {
                    'queryType': '1',
                    'goDay': '',
                    'doDay': '',
                    'queryDate': query_date,
                    'commodityId': 'TXF'
                }
            },
            {
                'name': '測試2: futContractsDate + TX',
                'url': "https://www.taifex.com.tw/cht/3/futContractsDate",
                'payload': {
                    'queryType': '1',
                    'marketCode': '0',
                    'commodity_id': 'TX',
                    'queryDate': query_date
                }
            },
            {
                'name': '測試3: futDataDown (三大法人)',
                'url': "https://www.taifex.com.tw/cht/3/futDataDown",
                'payload': {
                    'down_type': '1',
                    'queryDate': query_date,
                    'commodity_id': 'TX'
                }
            },
            {
                'name': '測試4: 三大法人交易資訊',
                'url': "https://www.taifex.com.tw/cht/3/futDataDown",
                'payload': {
                    'queryType': '2',
                    'queryDate': query_date
                }
            }
        ]
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        for config in test_configs:
            st.markdown(f"### {config['name']}")
            st.json(config['payload'])
            
            try:
                res = requests.post(config['url'], data=config['payload'], headers=headers, timeout=10, verify=False)
                res.encoding = 'utf-8'
                
                st.info(f"狀態: {res.status_code}, 長度: {len(res.text)}")
                
                if "查無資料" in res.text:
                    st.warning("❌ 查無資料")
                    continue
                
                dfs = pd.read_html(StringIO(res.text))
                if not dfs:
                    st.warning("❌ 無法解析表格")
                    continue
                
                st.success(f"✅ 找到 {len(dfs)} 個表格!")
                
                df = dfs[0]
                st.write(f"表格大小: {df.shape}")
                
                st.markdown("#### 欄位名稱")
                for i, col in enumerate(df.columns):
                    st.text(f"{i}: {col}")
                
                st.markdown("#### 表格內容")
                st.dataframe(df)
                
                st.markdown("#### 搜尋法人資料")
                for idx, row in df.iterrows():
                    row_str = " ".join([str(x) for x in row.values])
                    if '外資' in row_str or '投信' in row_str or '自營商' in row_str:
                        st.success(f"找到法人資料 (Row {idx}): {row_str[:80]}...")
                
                st.success("🎉 這個配置有效!")
                break
                
            except Exception as e:
                st.error(f"❌ 錯誤: {str(e)}")
            
            st.markdown("---")

# ==========================================
# 法人選擇權數據測試
# ==========================================
with tab2:
    st.markdown("### 📊 三大法人選擇權")
    
    if st.button("🧪 測試法人選擇權", key="opt"):
        query_date = (datetime.now(tz=TW_TZ) - timedelta(days=0)).strftime('%Y/%m/%d')
        st.write(f"測試日期: {query_date}")
        
        # 測試多個配置
        test_configs = [
            {
                'name': '測試1: callsAndPutsDate + TXO',
                'url': "https://www.taifex.com.tw/cht/3/callsAndPutsDate",
                'payload': {
                    'queryType': '1',
                    'goDay': '',
                    'doDay': '',
                    'queryDate': query_date,
                    'commodityId': 'TXO'
                }
            },
            {
                'name': '測試2: callsAndPutsDateDown',
                'url': "https://www.taifex.com.tw/cht/3/callsAndPutsDateDown",
                'payload': {
                    'down_type': '1',
                    'queryDate': query_date,
                    'commodity_id': 'TXO'
                }
            },
            {
                'name': '測試3: 選擇權三大法人交易資訊',
                'url': "https://www.taifex.com.tw/cht/3/callsAndPutsDate",
                'payload': {
                    'queryType': '2',
                    'queryDate': query_date
                }
            }
        ]
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        for config in test_configs:
            st.markdown(f"### {config['name']}")
            st.json(config['payload'])
            
            try:
                res = requests.post(config['url'], data=config['payload'], headers=headers, timeout=10, verify=False)
                res.encoding = 'utf-8'
                
                st.info(f"狀態: {res.status_code}, 長度: {len(res.text)}")
                
                if "查無資料" in res.text:
                    st.warning("❌ 查無資料")
                    continue
                
                dfs = pd.read_html(StringIO(res.text))
                if not dfs:
                    st.warning("❌ 無法解析表格")
                    continue
                    
                st.success(f"✅ 找到 {len(dfs)} 個表格!")
                
                df = dfs[0]
                st.write(f"表格大小: {df.shape}")
                
                st.markdown("#### 欄位名稱")
                for i, col in enumerate(df.columns):
                    st.text(f"{i}: {col}")
                
                st.markdown("#### 表格內容")
                st.dataframe(df)
                
                st.markdown("#### 篩選法人資料")
                df_filtered = df[df.iloc[:, 0].astype(str).str.contains('自營商|投信|外資', na=False)]
                
                if not df_filtered.empty:
                    st.success(f"✅ 找到 {len(df_filtered)} 筆法人資料")
                    st.dataframe(df_filtered)
                else:
                    st.warning("⚠️ 篩選失敗,嘗試手動搜尋...")
                    for idx, row in df.iterrows():
                        row_str = " ".join([str(x) for x in row.values])
                        if '外資' in row_str or '投信' in row_str or '自營商' in row_str:
                            st.success(f"找到法人 (Row {idx}): {row_str[:80]}...")
                
                st.success("🎉 這個配置有效!")
                break
                
            except Exception as e:
                st.error(f"❌ 錯誤: {str(e)}")
            
            st.markdown("---")

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
