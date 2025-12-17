import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from io import StringIO
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(layout="wide", page_title="法人資料 queryType=2 測試")
TW_TZ = timezone(timedelta(hours=8))

st.title("🎯 法人資料 queryType=2 深度測試")

st.success("""
✅ 發現有效配置:
- 期貨法人: futContractsDate + queryType=2 (475KB, 73行)
- 選擇權法人: callsAndPutsDate + queryType=2 (322KB, 30行)

現在來深入分析這兩個表格!
""")

tab1, tab2 = st.tabs(["📈 期貨法人 (queryType=2)", "📊 選擇權法人 (queryType=2)"])

# ==========================================
# 期貨法人
# ==========================================
with tab1:
    st.markdown("### 📈 期貨法人 - futContractsDate + queryType=2")
    
    if st.button("🔍 深度分析", key="fut"):
        url = "https://www.taifex.com.tw/cht/3/futContractsDate"
        query_date = (datetime.now(tz=TW_TZ) - timedelta(days=1)).strftime('%Y/%m/%d')
        
        payload = {
            'queryType': '2',
            'queryDate': query_date,
            'commodity_id': 'TX'
        }
        
        st.json(payload)
        
        try:
            res = requests.post(url, data=payload, timeout=10, verify=False)
            res.encoding = 'utf-8'
            
            st.info(f"狀態: {res.status_code}, 長度: {len(res.text)}")
            
            dfs = pd.read_html(StringIO(res.text))
            st.success(f"✅ 找到 {len(dfs)} 個表格")
            
            for i, df in enumerate(dfs):
                st.markdown(f"## 表格 {i+1}")
                st.write(f"**大小:** {df.shape}")
                
                st.markdown("### 📝 欄位名稱")
                for idx, col in enumerate(df.columns):
                    st.text(f"{idx}: {col}")
                
                st.markdown("### 📊 完整資料")
                st.dataframe(df)
                
                st.markdown("### 🔍 搜尋法人資料")
                
                # 方法1: 搜尋包含「外資」「投信」「自營商」的行
                st.markdown("#### 方法1: 關鍵字搜尋")
                for idx, row in df.iterrows():
                    row_str = " ".join([str(x) for x in row.values])
                    
                    if any(keyword in row_str for keyword in ['外資', '投信', '自營商', '外資及陸資']):
                        st.success(f"✅ Row {idx}: {row_str[:150]}")
                        st.write("**完整資料:**")
                        st.json(row.to_dict())
                
                # 方法2: 檢查第一欄
                st.markdown("#### 方法2: 第一欄檢查")
                first_col = df.columns[0]
                st.write(f"第一欄名稱: {first_col}")
                st.write(f"第一欄內容樣本: {df[first_col].head(10).tolist()}")
                
                df_filtered = df[df[first_col].astype(str).str.contains('外資|投信|自營商', na=False)]
                if not df_filtered.empty:
                    st.success(f"✅ 在第一欄找到 {len(df_filtered)} 筆法人資料!")
                    st.dataframe(df_filtered)
                
                # 方法3: 檢查所有欄位
                st.markdown("#### 方法3: 搜尋所有欄位")
                for col in df.columns:
                    if df[col].astype(str).str.contains('外資|投信|自營商', na=False).any():
                        st.info(f"欄位 '{col}' 包含法人關鍵字")
                        matching_rows = df[df[col].astype(str).str.contains('外資|投信|自營商', na=False)]
                        st.dataframe(matching_rows)
                
        except Exception as e:
            st.error(f"錯誤: {str(e)}")

# ==========================================
# 選擇權法人
# ==========================================
with tab2:
    st.markdown("### 📊 選擇權法人 - callsAndPutsDate + queryType=2")
    
    if st.button("🔍 深度分析", key="opt"):
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
            
            for i, df in enumerate(dfs):
                st.markdown(f"## 表格 {i+1}")
                st.write(f"**大小:** {df.shape}")
                
                st.markdown("### 📝 欄位名稱")
                for idx, col in enumerate(df.columns):
                    st.text(f"{idx}: {col}")
                
                st.markdown("### 📊 完整資料")
                st.dataframe(df)
                
                st.markdown("### 🔍 搜尋法人資料")
                
                # 關鍵字搜尋
                st.markdown("#### 關鍵字搜尋所有行")
                found_count = 0
                for idx, row in df.iterrows():
                    row_str = " ".join([str(x) for x in row.values])
                    
                    if any(keyword in row_str for keyword in ['外資', '投信', '自營商', '外資及陸資']):
                        st.success(f"✅ Row {idx}: {row_str[:150]}")
                        st.write("**完整資料:**")
                        st.json(row.to_dict())
                        found_count += 1
                
                if found_count > 0:
                    st.success(f"🎉 總共找到 {found_count} 筆法人資料!")
                
                # 檢查第一欄
                st.markdown("#### 第一欄分析")
                first_col = df.columns[0]
                st.write(f"第一欄名稱: {first_col}")
                st.write(f"第一欄內容樣本: {df[first_col].head(10).tolist()}")
                
                df_filtered = df[df[first_col].astype(str).str.contains('外資|投信|自營商', na=False)]
                if not df_filtered.empty:
                    st.success(f"✅ 在第一欄找到 {len(df_filtered)} 筆法人資料!")
                    st.dataframe(df_filtered)
                
        except Exception as e:
            st.error(f"錯誤: {str(e)}")

st.markdown("---")
st.info("""
💡 **這個測試會:**
1. 顯示完整的表格結構
2. 列出所有欄位名稱
3. 用3種方法搜尋法人資料
4. 顯示找到的法人資料的完整內容

如果能找到法人資料,我就能確定正確的欄位名稱和數據結構!
""")
