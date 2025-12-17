import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from io import StringIO
import urllib3
import subprocess
import sys

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(layout="wide", page_title="法人數據調試 v2")
TW_TZ = timezone(timedelta(hours=8))

st.title("🔬 三大法人數據調試工具 v2")

# 安裝必要套件
if st.button("📦 安裝必要套件 (beautifulsoup4 + lxml)"):
    with st.spinner("安裝中..."):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "beautifulsoup4", "lxml", "--break-system-packages"])
            st.success("✅ 安裝成功!")
            st.rerun()
        except Exception as e:
            st.error(f"安裝失敗: {str(e)}")

st.markdown("---")

tab1, tab2 = st.tabs(["📈 法人期貨", "📊 法人選擇權"])

# ==========================================
# 法人期貨數據測試
# ==========================================
with tab1:
    st.markdown("### 📈 三大法人期貨淨部位")
    
    if st.button("🧪 測試法人期貨", key="fut"):
        query_date = (datetime.now(tz=TW_TZ) - timedelta(days=0)).strftime('%Y/%m/%d')
        st.write(f"測試日期: {query_date}")
        
        # 測試配置 (futDataDown 最有可能)
        test_configs = [
            {
                'name': '測試1: futDataDown (依身份別)',
                'url': "https://www.taifex.com.tw/cht/3/futDataDown",
                'payload': {
                    'down_type': '1',
                    'queryDate': query_date,
                    'commodity_id': 'TX'
                }
            },
            {
                'name': '測試2: futDataDown (全市場)',
                'url': "https://www.taifex.com.tw/cht/3/futDataDown",
                'payload': {
                    'queryType': '2',
                    'queryDate': query_date
                }
            },
            {
                'name': '測試3: 期貨三大法人交易口數',
                'url': "https://www.taifex.com.tw/cht/3/futDataDown",
                'payload': {
                    'down_type': '2',
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
                
                # 顯示部分回應內容
                if len(res.text) < 5000:
                    with st.expander("查看原始回應"):
                        st.code(res.text[:2000])
                
                # 嘗試解析 CSV (可能是下載檔案)
                try:
                    df = pd.read_csv(StringIO(res.text))
                    st.success(f"✅ CSV 格式! 表格大小: {df.shape}")
                    st.dataframe(df)
                    
                    # 搜尋法人資料
                    st.markdown("#### 法人資料")
                    for idx, row in df.iterrows():
                        row_str = " ".join([str(x) for x in row.values])
                        if '外資' in row_str or '投信' in row_str or '自營商' in row_str:
                            st.success(f"✅ Row {idx}: {row_str[:100]}")
                    
                    st.success("🎉 這個配置有效 (CSV)!")
                    break
                except:
                    pass
                
                # 嘗試解析 HTML 表格
                try:
                    dfs = pd.read_html(StringIO(res.text), encoding='utf-8')
                    if dfs:
                        st.success(f"✅ HTML 表格! 找到 {len(dfs)} 個表格")
                        
                        for i, df in enumerate(dfs):
                            st.markdown(f"#### 表格 {i+1} (大小: {df.shape})")
                            st.dataframe(df)
                            
                            # 搜尋法人資料
                            for idx, row in df.iterrows():
                                row_str = " ".join([str(x) for x in row.values])
                                if '外資' in row_str or '投信' in row_str or '自營商' in row_str:
                                    st.success(f"✅ 找到法人 Row {idx}")
                        
                        st.success("🎉 這個配置有效 (HTML)!")
                        break
                except Exception as e:
                    st.warning(f"HTML 解析失敗: {str(e)}")
                
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
        
        test_configs = [
            {
                'name': '測試1: callsAndPutsDateDown (依身份別)',
                'url': "https://www.taifex.com.tw/cht/3/callsAndPutsDateDown",
                'payload': {
                    'down_type': '1',
                    'queryDate': query_date,
                    'commodity_id': 'TXO'
                }
            },
            {
                'name': '測試2: callsAndPutsDate',
                'url': "https://www.taifex.com.tw/cht/3/callsAndPutsDate",
                'payload': {
                    'queryType': '1',
                    'queryDate': query_date,
                    'commodityId': 'TXO'
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
                
                if len(res.text) < 5000:
                    with st.expander("查看原始回應"):
                        st.code(res.text[:2000])
                
                # 嘗試 CSV
                try:
                    df = pd.read_csv(StringIO(res.text))
                    st.success(f"✅ CSV 格式! 表格大小: {df.shape}")
                    st.dataframe(df)
                    st.success("🎉 這個配置有效 (CSV)!")
                    break
                except:
                    pass
                
                # 嘗試 HTML
                try:
                    dfs = pd.read_html(StringIO(res.text), encoding='utf-8')
                    if dfs:
                        st.success(f"✅ HTML 表格! 找到 {len(dfs)} 個表格")
                        
                        for i, df in enumerate(dfs):
                            st.markdown(f"#### 表格 {i+1} (大小: {df.shape})")
                            st.dataframe(df)
                            
                            # 篩選法人
                            df_filtered = df[df.iloc[:, 0].astype(str).str.contains('自營商|投信|外資', na=False)]
                            if not df_filtered.empty:
                                st.success(f"✅ 找到 {len(df_filtered)} 筆法人資料")
                                st.dataframe(df_filtered)
                        
                        st.success("🎉 這個配置有效 (HTML)!")
                        break
                except Exception as e:
                    st.warning(f"HTML 解析失敗: {str(e)}")
                
            except Exception as e:
                st.error(f"❌ 錯誤: {str(e)}")
            
            st.markdown("---")

st.markdown("---")
st.markdown("### 💡 說明")
st.write("""
**如果看到 beautifulsoup4 錯誤:**
1. 點擊上方「安裝必要套件」按鈕
2. 等待安裝完成
3. 重新執行測試

**測試策略:**
- `futDataDown` / `callsAndPutsDateDown` 是專門的下載端點
- 可能回傳 CSV 或 HTML 格式
- down_type=1 通常是依身份別(三大法人)
""")
