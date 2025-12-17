import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from io import StringIO
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(layout="wide")
TW_TZ = timezone(timedelta(hours=8))

st.title("📊 選擇權法人快速測試")

if st.button("🧪 測試"):
    url = "https://www.taifex.com.tw/cht/3/callsAndPutsDate"
    query_date = (datetime.now(tz=TW_TZ) - timedelta(days=1)).strftime('%Y/%m/%d')
    
    payload = {
        'queryType': '2',
        'queryDate': query_date,
        'commodity_id': 'TXO'
    }
    
    st.json(payload)
    
    try:
        res = requests.post(url, data=payload, timeout=10, verify=False)
        res.encoding = 'utf-8'
        
        st.info(f"狀態: {res.status_code}, 長度: {len(res.text)}")
        
        dfs = pd.read_html(StringIO(res.text))
        st.success(f"✅ 找到 {len(dfs)} 個表格")
        
        df = dfs[0]
        st.write(f"**表格大小:** {df.shape}")
        
        st.markdown("### 📝 欄位")
        for idx, col in enumerate(df.columns):
            st.text(f"{idx}: {col}")
        
        st.markdown("### 📊 完整資料")
        st.dataframe(df)
        
        st.markdown("### 🔍 搜尋臺指選擇權法人")
        
        for idx, row in df.iterrows():
            row_str = " ".join([str(x) for x in row.values])
            
            if '臺指選擇權' in row_str or 'TXO' in row_str:
                if any(kw in row_str for kw in ['外資', '投信', '自營商']):
                    st.success(f"✅ Row {idx}")
                    st.text(row_str[:200])
                    
                    # 顯示數據
                    try:
                        st.write({
                            '身份別': row.iloc[2] if len(row) > 2 else 'N/A',
                            'Call買方': row.iloc[3] if len(row) > 3 else 'N/A',
                            'Call賣方': row.iloc[4] if len(row) > 4 else 'N/A',  
                            'Put買方': row.iloc[5] if len(row) > 5 else 'N/A',
                            'Put賣方': row.iloc[6] if len(row) > 6 else 'N/A',
                        })
                    except:
                        st.write("無法解析欄位")
        
    except Exception as e:
        st.error(f"錯誤: {str(e)}")
