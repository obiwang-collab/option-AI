import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from io import StringIO
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(layout="wide", page_title="法人數據自動回溯測試")
TW_TZ = timezone(timedelta(hours=8))

st.title("🔬 法人數據自動回溯測試")

tab1, tab2 = st.tabs(["📈 法人期貨", "📊 法人選擇權"])

# ==========================================
# 法人期貨
# ==========================================
with tab1:
    st.markdown("### 📈 法人期貨 - 自動回溯最近5天")
    
    if st.button("🧪 開始測試", key="fut"):
        url = "https://www.taifex.com.tw/cht/3/futDataDown"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        st.info("🔄 自動回溯測試最近5天...")
        
        success = False
        
        for i in range(5):
            query_date = (datetime.now(tz=TW_TZ) - timedelta(days=i)).strftime('%Y/%m/%d')
            
            st.markdown(f"### 測試日期: {query_date} (T-{i})")
            
            payload = {
                'down_type': '1',
                'queryDate': query_date,
                'commodity_id': 'TX'
            }
            
            try:
                res = requests.post(url, data=payload, headers=headers, timeout=10, verify=False)
                
                st.info(f"狀態: {res.status_code}, 長度: {len(res.text)}")
                
                # 檢查是否有錯誤訊息
                if "日期時間錯誤" in res.text or "DateTime error" in res.text:
                    st.warning("❌ 日期時間錯誤 (資料尚未更新)")
                    with st.expander("查看錯誤訊息"):
                        st.text(res.text[:500])
                    continue
                
                # 嘗試多種編碼解析 CSV
                for encoding in ['utf-8', 'big5', 'cp950']:
                    try:
                        res.encoding = encoding
                        df = pd.read_csv(StringIO(res.text))
                        
                        st.success(f"✅ 成功解析! 編碼: {encoding}")
                        st.write(f"**表格大小:** {df.shape}")
                        
                        # 顯示欄位
                        st.markdown("#### 欄位名稱")
                        for idx, col in enumerate(df.columns):
                            st.text(f"{idx}: {col}")
                        
                        # 顯示完整資料
                        st.markdown("#### 完整資料")
                        st.dataframe(df)
                        
                        # 搜尋法人
                        st.markdown("#### 法人資料")
                        found_data = {}
                        for idx, row in df.iterrows():
                            row_str = " ".join([str(x) for x in row.values])
                            
                            if '外資' in row_str or '外資及陸資' in row_str:
                                st.success(f"✅ 外資 (Row {idx})")
                                st.write(row.to_dict())
                                found_data['外資'] = row.to_dict()
                            elif '投信' in row_str:
                                st.success(f"✅ 投信 (Row {idx})")
                                st.write(row.to_dict())
                                found_data['投信'] = row.to_dict()
                            elif '自營商' in row_str:
                                st.success(f"✅ 自營商 (Row {idx})")
                                st.write(row.to_dict())
                                found_data['自營商'] = row.to_dict()
                        
                        if found_data:
                            st.success(f"🎉 成功找到 {len(found_data)} 個法人的資料!")
                            st.json(found_data)
                            success = True
                            break
                        else:
                            st.warning("⚠️ 未找到法人資料,可能欄位格式不符")
                        
                        break
                        
                    except Exception as e:
                        if encoding == 'cp950':  # 最後一個編碼
                            st.error(f"所有編碼都失敗: {str(e)}")
                
                if success:
                    break
                    
            except Exception as e:
                st.error(f"請求失敗: {str(e)}")
            
            st.markdown("---")
        
        if not success:
            st.error("❌ 所有日期都失敗了")
            st.info("""
            可能原因:
            1. 最近幾天都是假日/非交易日
            2. API 參數不正確
            3. 需要使用不同的 URL 端點
            """)

# ==========================================
# 法人選擇權
# ==========================================
with tab2:
    st.markdown("### 📊 法人選擇權 - 自動回溯最近5天")
    
    if st.button("🧪 開始測試", key="opt"):
        url = "https://www.taifex.com.tw/cht/3/callsAndPutsDateDown"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        st.info("🔄 自動回溯測試最近5天...")
        
        success = False
        
        for i in range(5):
            query_date = (datetime.now(tz=TW_TZ) - timedelta(days=i)).strftime('%Y/%m/%d')
            
            st.markdown(f"### 測試日期: {query_date} (T-{i})")
            
            payload = {
                'down_type': '1',
                'queryDate': query_date,
                'commodity_id': 'TXO'
            }
            
            try:
                res = requests.post(url, data=payload, headers=headers, timeout=10, verify=False)
                
                st.info(f"狀態: {res.status_code}, 長度: {len(res.text)}")
                
                if "日期時間錯誤" in res.text or "DateTime error" in res.text:
                    st.warning("❌ 日期時間錯誤")
                    continue
                
                for encoding in ['utf-8', 'big5', 'cp950']:
                    try:
                        res.encoding = encoding
                        df = pd.read_csv(StringIO(res.text))
                        
                        st.success(f"✅ 成功解析! 編碼: {encoding}")
                        st.write(f"**表格大小:** {df.shape}")
                        
                        st.markdown("#### 欄位名稱")
                        for idx, col in enumerate(df.columns):
                            st.text(f"{idx}: {col}")
                        
                        st.markdown("#### 完整資料")
                        st.dataframe(df)
                        
                        st.markdown("#### 法人資料")
                        df_filtered = df[df.iloc[:, 0].astype(str).str.contains('自營商|投信|外資', na=False)]
                        
                        if not df_filtered.empty:
                            st.success(f"✅ 找到 {len(df_filtered)} 筆法人資料")
                            st.dataframe(df_filtered)
                            st.success("🎉 成功!")
                            success = True
                            break
                        else:
                            # 手動搜尋
                            for idx, row in df.iterrows():
                                row_str = " ".join([str(x) for x in row.values])
                                if '外資' in row_str or '投信' in row_str or '自營商' in row_str:
                                    st.success(f"✅ 找到法人 (Row {idx})")
                                    st.write(row.to_dict())
                        
                        break
                        
                    except Exception as e:
                        if encoding == 'cp950':
                            st.error(f"所有編碼都失敗: {str(e)}")
                
                if success:
                    break
                    
            except Exception as e:
                st.error(f"請求失敗: {str(e)}")
            
            st.markdown("---")
        
        if not success:
            st.error("❌ 所有日期都失敗了")

st.markdown("---")
st.markdown("### 💡 說明")
st.info("""
**自動回溯邏輯:**
- 從今天開始往回測試
- 遇到「日期時間錯誤」就跳過
- 找到第一個有資料的日期就停止
- 最多測試5天

**法人資料更新時間:**
- 通常在交易日隔天上午公布
- 假日和週末沒有資料
""")
