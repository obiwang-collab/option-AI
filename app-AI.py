import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from io import StringIO
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(layout="wide", page_title="期交所法人資料 - 全面測試")
TW_TZ = timezone(timedelta(hours=8))

st.title("🔬 期交所法人資料 - 全面測試")

st.info("""
根據期交所網站,法人資料可能在這些地方:
1. 三大法人台指期貨交易口數 (futDataDown)
2. 三大法人選擇權交易口數 (callsAndPutsDateDown)
3. 期貨依日期查詢 + 身份別 (futContractsDate)
4. 三大法人資料獨立頁面
""")

if st.button("🚀 開始全面測試"):
    query_date = (datetime.now(tz=TW_TZ) - timedelta(days=1)).strftime('%Y/%m/%d')
    st.write(f"測試日期: {query_date}")
    
    # 所有可能的配置
    test_configs = [
        # === 期貨法人 ===
        {
            'name': '期貨法人 #1: futDataDown + down_type=1',
            'url': 'https://www.taifex.com.tw/cht/3/futDataDown',
            'payload': {'down_type': '1', 'queryDate': query_date, 'commodity_id': 'TX'}
        },
        {
            'name': '期貨法人 #2: futContractsDateDown',
            'url': 'https://www.taifex.com.tw/cht/3/futContractsDateDown',
            'payload': {'down_type': '1', 'queryDate': query_date, 'commodity_id': 'TX'}
        },
        {
            'name': '期貨法人 #3: futContractsDate + queryType=2',
            'url': 'https://www.taifex.com.tw/cht/3/futContractsDate',
            'payload': {'queryType': '2', 'queryDate': query_date, 'commodity_id': 'TX'}
        },
        {
            'name': '期貨法人 #4: 三大法人期貨 (可能是 CSV 直接下載)',
            'url': 'https://www.taifex.com.tw/file/taifex/Dailydownload/DailydownloadCSV/Daily_' + query_date.replace('/', '') + '.zip',
            'method': 'GET'
        },
        {
            'name': '期貨法人 #5: futDataDown (無 commodity_id)',
            'url': 'https://www.taifex.com.tw/cht/3/futDataDown',
            'payload': {'down_type': '1', 'queryDate': query_date}
        },
        
        # === 選擇權法人 ===
        {
            'name': '選擇權法人 #1: callsAndPutsDateDown + down_type=1',
            'url': 'https://www.taifex.com.tw/cht/3/callsAndPutsDateDown',
            'payload': {'down_type': '1', 'queryDate': query_date, 'commodity_id': 'TXO'}
        },
        {
            'name': '選擇權法人 #2: callsAndPutsDate + queryType=2',
            'url': 'https://www.taifex.com.tw/cht/3/callsAndPutsDate',
            'payload': {'queryType': '2', 'queryDate': query_date, 'commodity_id': 'TXO'}
        },
        {
            'name': '選擇權法人 #3: optDataDown',
            'url': 'https://www.taifex.com.tw/cht/3/optDataDown',
            'payload': {'down_type': '1', 'queryDate': query_date, 'commodity_id': 'TXO'}
        },
        {
            'name': '選擇權法人 #4: callsAndPutsDateDown (無 commodity_id)',
            'url': 'https://www.taifex.com.tw/cht/3/callsAndPutsDateDown',
            'payload': {'down_type': '1', 'queryDate': query_date}
        }
    ]
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    success_configs = []
    
    for idx, config in enumerate(test_configs, 1):
        st.markdown(f"## 測試 {idx}/{len(test_configs)}: {config['name']}")
        
        with st.expander("配置詳情", expanded=False):
            st.json(config)
        
        try:
            # 發送請求
            if config.get('method') == 'GET':
                res = requests.get(config['url'], headers=headers, timeout=10, verify=False)
            else:
                res = requests.post(config['url'], data=config.get('payload', {}), 
                                  headers=headers, timeout=10, verify=False)
            
            st.info(f"📊 狀態碼: {res.status_code} | 長度: {len(res.text)} 字元")
            
            # 檢查錯誤
            if "日期時間錯誤" in res.text or "DateTime error" in res.text:
                st.warning("⚠️ 日期時間錯誤")
                continue
            
            if "查無資料" in res.text:
                st.warning("⚠️ 查無資料")
                continue
            
            if len(res.text) < 100:
                st.warning("⚠️ 回應過短")
                with st.expander("查看內容"):
                    st.text(res.text)
                continue
            
            # 嘗試解析
            parsed = False
            
            # 1. 嘗試 CSV
            try:
                for encoding in ['utf-8', 'big5', 'cp950']:
                    try:
                        res.encoding = encoding
                        df = pd.read_csv(StringIO(res.text))
                        
                        if df.shape[0] > 0:
                            st.success(f"✅ CSV 解析成功! (編碼: {encoding})")
                            st.write(f"**表格大小:** {df.shape}")
                            st.dataframe(df.head(20))
                            
                            # 搜尋法人
                            has_institutional = False
                            for _, row in df.iterrows():
                                row_str = " ".join([str(x) for x in row.values])
                                if '外資' in row_str or '投信' in row_str or '自營商' in row_str:
                                    has_institutional = True
                                    st.success(f"✅ 找到法人資料!")
                                    break
                            
                            if has_institutional:
                                st.success("🎉🎉🎉 這個配置有效!")
                                success_configs.append(config)
                            
                            parsed = True
                            break
                    except:
                        continue
            except:
                pass
            
            # 2. 嘗試 HTML
            if not parsed:
                try:
                    dfs = pd.read_html(StringIO(res.text))
                    if dfs and len(dfs) > 0:
                        st.success(f"✅ HTML 解析成功! 找到 {len(dfs)} 個表格")
                        
                        for i, df in enumerate(dfs[:3]):  # 只顯示前3個表格
                            st.write(f"**表格 {i+1}:** {df.shape}")
                            st.dataframe(df.head(10))
                            
                            # 搜尋法人
                            for _, row in df.iterrows():
                                row_str = " ".join([str(x) for x in row.values])
                                if '外資' in row_str or '投信' in row_str or '自營商' in row_str:
                                    st.success(f"✅ 找到法人資料!")
                                    st.success("🎉🎉🎉 這個配置有效!")
                                    success_configs.append(config)
                                    parsed = True
                                    break
                            
                            if parsed:
                                break
                except:
                    pass
            
            # 3. 如果都失敗,顯示原始內容
            if not parsed:
                st.warning("❌ 無法解析")
                with st.expander("查看原始內容 (前1000字元)"):
                    st.text(res.text[:1000])
            
        except Exception as e:
            st.error(f"❌ 錯誤: {str(e)}")
        
        st.markdown("---")
    
    # 總結
    st.markdown("## 📊 測試總結")
    
    if success_configs:
        st.success(f"✅ 找到 {len(success_configs)} 個有效配置!")
        
        for config in success_configs:
            st.json(config)
    else:
        st.error("❌ 所有配置都失敗了")
        st.info("""
        **可能的原因:**
        1. 法人資料 API 已經改變
        2. 需要特殊的 token 或認證
        3. 需要從期交所首頁先取得 session
        4. 資料格式完全不同
        
        **建議:**
        請直接到期交所網站手動下載法人資料,看看實際的下載 URL 是什麼。
        """)

st.markdown("---")
st.info("""
💡 **如何找到正確的 URL:**
1. 打開期交所網站
2. 找到三大法人資料頁面
3. 按 F12 打開開發者工具
4. 點擊下載或查詢按鈕
5. 在 Network 分頁查看實際的請求 URL 和參數
""")
