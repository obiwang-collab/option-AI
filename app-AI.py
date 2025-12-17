import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from io import StringIO
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(layout="wide", page_title="法人數據簡化測試")
TW_TZ = timezone(timedelta(hours=8))

st.title("🔬 法人數據簡化測試 (無需額外套件)")

tab1, tab2 = st.tabs(["📈 法人期貨", "📊 法人選擇權"])

with tab1:
    st.markdown("### 📈 法人期貨 - CSV 格式測試")
    
    if st.button("🧪 測試", key="fut"):
        url = "https://www.taifex.com.tw/cht/3/futDataDown"
        
        # 🔥 修正: 查詢昨天的資料
        query_date = (datetime.now(tz=TW_TZ) - timedelta(days=1)).strftime('%Y/%m/%d')
        
        st.write(f"日期: {query_date} (昨天)")
        st.info("💡 法人資料通常在隔天才更新")
        
        payload = {
            'down_type': '1',
            'queryDate': query_date,
            'commodity_id': 'TX'
        }
        
        try:
            res = requests.post(url, data=payload, timeout=10, verify=False)
            
            # 嘗試不同編碼
            for encoding in ['utf-8', 'big5', 'cp950']:
                try:
                    res.encoding = encoding
                    st.info(f"嘗試編碼: {encoding}")
                    st.info(f"內容長度: {len(res.text)}")
                    
                    # 顯示原始內容
                    with st.expander(f"原始內容 ({encoding})"):
                        st.text(res.text[:1000])
                    
                    # 嘗試解析為 CSV
                    try:
                        df = pd.read_csv(StringIO(res.text))
                        st.success(f"✅ CSV 解析成功! 編碼: {encoding}")
                        st.write(f"表格大小: {df.shape}")
                        st.dataframe(df)
                        
                        # 搜尋法人
                        st.markdown("#### 法人資料搜尋")
                        found = False
                        for idx, row in df.iterrows():
                            row_str = " ".join([str(x) for x in row.values])
                            if '外資' in row_str or '投信' in row_str or '自營商' in row_str:
                                st.success(f"✅ Row {idx}: {row_str}")
                                found = True
                        
                        if found:
                            st.success("🎉 找到法人資料!")
                        break
                        
                    except Exception as e:
                        st.warning(f"CSV 解析失敗 ({encoding}): {str(e)}")
                        
                except Exception as e:
                    st.error(f"編碼 {encoding} 失敗: {str(e)}")
                    
        except Exception as e:
            st.error(f"請求失敗: {str(e)}")

with tab2:
    st.markdown("### 📊 法人選擇權 - CSV 格式測試")
    
    if st.button("🧪 測試", key="opt"):
        url = "https://www.taifex.com.tw/cht/3/callsAndPutsDateDown"
        
        # 🔥 修正: 查詢昨天的資料
        query_date = (datetime.now(tz=TW_TZ) - timedelta(days=1)).strftime('%Y/%m/%d')
        
        st.write(f"日期: {query_date} (昨天)")
        st.info("💡 法人資料通常在隔天才更新")
        
        payload = {
            'down_type': '1',
            'queryDate': query_date,
            'commodity_id': 'TXO'
        }
        
        try:
            res = requests.post(url, data=payload, timeout=10, verify=False)
            
            for encoding in ['utf-8', 'big5', 'cp950']:
                try:
                    res.encoding = encoding
                    st.info(f"嘗試編碼: {encoding}")
                    st.info(f"內容長度: {len(res.text)}")
                    
                    with st.expander(f"原始內容 ({encoding})"):
                        st.text(res.text[:1000])
                    
                    try:
                        df = pd.read_csv(StringIO(res.text))
                        st.success(f"✅ CSV 解析成功! 編碼: {encoding}")
                        st.write(f"表格大小: {df.shape}")
                        st.dataframe(df)
                        
                        st.markdown("#### 法人資料搜尋")
                        df_filtered = df[df.iloc[:, 0].astype(str).str.contains('自營商|投信|外資', na=False)]
                        
                        if not df_filtered.empty:
                            st.success(f"✅ 找到 {len(df_filtered)} 筆法人資料")
                            st.dataframe(df_filtered)
                            st.success("🎉 找到法人資料!")
                        
                        break
                        
                    except Exception as e:
                        st.warning(f"CSV 解析失敗 ({encoding}): {str(e)}")
                        
                except Exception as e:
                    st.error(f"編碼 {encoding} 失敗: {str(e)}")
                    
        except Exception as e:
            st.error(f"請求失敗: {str(e)}")

st.markdown("---")
st.info("""
💡 這個版本只用基本套件,如果能解析成功,就不需要安裝 beautifulsoup4。
如果失敗,請更新 requirements.txt 加入:
- beautifulsoup4
- lxml
""")
