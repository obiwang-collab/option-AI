import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import time
from datetime import datetime, timedelta, timezone
from io import StringIO
import calendar
import re
import urllib3

# 忽略 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 頁面設定 (必須在第一行) ---
st.set_page_config(layout="wide", page_title="台指期籌碼戰情室 (極速穩定版)")
TW_TZ = timezone(timedelta(hours=8))

# ==========================================
# 🔑 設定區
# ==========================================
# 若無金鑰則留空，不影響基礎功能
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "") 
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")

# --- 核心請求函式 (優化：加上進度回報) ---
def fetch_taifex_html(url, payload, status_text=None):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://www.taifex.com.tw/',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    try:
        if status_text: 
            date_str = payload.get('queryDate', 'Unknown')
            status_text.text(f"正在連線: {date_str} ...")
        
        session = requests.Session()
        # Timeout 設為 3 秒，網路卡住時快速跳過
        res = session.post(url, data=payload, headers=headers, timeout=3, verify=False)
        
        # 雙重編碼解碼嘗試
        try:
            html_text = res.content.decode('utf-8')
        except UnicodeDecodeError:
            html_text = res.content.decode('big5', errors='ignore')
            
        if "查無資料" in html_text or len(html_text) < 500:
            return None
        return html_text
    except Exception:
        return None

# --- 資料獲取函式 ---

# 1. 獲取期貨行情
def get_futures_data(status_container):
    url = "https://www.taifex.com.tw/cht/3/futDailyMarketReport"
    # 回朔 5 天
    for i in range(5):
        target_date = datetime.now(tz=TW_TZ) - timedelta(days=i)
        if i == 0 and datetime.now(tz=TW_TZ).hour < 15: continue
        
        query_date = target_date.strftime('%Y/%m/%d')
        payload = {'queryType': '2', 'marketCode': '0', 'commodity_id': 'TX', 'queryDate': query_date}
        
        html = fetch_taifex_html(url, payload, status_container)
        if not html: 
            time.sleep(0.5) 
            continue

        try:
            dfs = pd.read_html(StringIO(html))
            df = dfs[0]
            futures_price = None
            for col in df.columns:
                if '收盤價' in str(col) or '成交價' in str(col):
                    val = str(df.iloc[0][col]).replace(',', '').strip()
                    if val and val != '-' and val != '':
                        futures_price = float(val)
                        break
            if futures_price: return futures_price, query_date
        except: pass
    
    return None, "N/A"

# 2. 獲取法人期貨
def get_institutional_futures(status_container):
    url = "https://www.taifex.com.tw/cht/3/futContractsDate"
    for i in range(5):
        target_date = datetime.now(tz=TW_TZ) - timedelta(days=i)
        if i == 0 and datetime.now(tz=TW_TZ).hour < 15: continue 
        
        query_date = target_date.strftime('%Y/%m/%d')
        payload = {'queryType': '1', 'goDay': '', 'doDay': '', 'queryDate': query_date, 'commodityId': 'TXF'}
        
        html = fetch_taifex_html(url, payload, status_container)
        if not html: 
            time.sleep(0.5)
            continue
        
        try:
            dfs = pd.read_html(StringIO(html))
            df = dfs[0]
            inst_data = {}
            for idx, row in df.iterrows():
                row_str = " ".join([str(x) for x in row.values])
                def get_val(r):
                    try: return int(str(r.iloc[-1]).replace(',', '')) 
                    except: return 0
                
                if '外資' in row_str: inst_data['外資'] = get_val(row)
                elif '投信' in row_str: inst_data['投信'] = get_val(row)
                elif '自營商' in row_str: inst_data['自營商'] = get_val(row)
            
            if inst_data: 
                inst_data['date'] = query_date
                return inst_data
        except: pass
    return None

# 3. 獲取法人選擇權
def get_institutional_options(status_container):
    url = "https://www.taifex.com.tw/cht/3/callsAndPutsDate"
    all_data = []
    for i in range(5):
        target_date = datetime.now(tz=TW_TZ) - timedelta(days=i)
        if i == 0 and datetime.now(tz=TW_TZ).hour < 15: continue
        
        query_date = target_date.strftime('%Y/%m/%d')
        payload = {'queryType': '1', 'goDay': '', 'doDay': '', 'queryDate': query_date, 'commodityId': 'TXO'}
        
        html = fetch_taifex_html(url, payload, status_container)
        if not html: continue
        
        try:
            dfs = pd.read_html(StringIO(html))
            df = dfs[0]
            df_filtered = df[df.iloc[:, 0].astype(str).str.contains('自營商|投信|外資', na=False)]
            if not df_filtered.empty:
                all_data.append({'date': query_date, 'df': df_filtered})
                if len(all_data) >= 2: break
        except: pass
        
    if not all_data: return None, None
    return all_data[0]['df'], all_data[0]['date']

# 4. 獲取選擇權全市場 (已修復回傳值 Bug)
def get_option_market(status_container):
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    all_data = []
    
    # 這裡稍微找久一點 (7天)，以免連續假日沒資料
    for i in range(7):
        target_date = datetime.now(tz=TW_TZ) - timedelta(days=i)
        if i == 0 and datetime.now(tz=TW_TZ).hour < 15: continue
        
        query_date = target_date.strftime('%Y/%m/%d')
        payload = {'queryType': '2', 'marketCode': '0', 'commodity_id': 'TXO', 'queryDate': query_date}
        
        html = fetch_taifex_html(url, payload, status_container)
        if not html: continue

        try:
            dfs = pd.read_html(StringIO(html))
            df = dfs[0]
            # 暴力清洗欄位
            df.columns = [str(c).replace(' ', '').replace('*', '').replace('契約', '').strip() for c in df.columns]
            
            # 映射關鍵欄位
            col_map = {
                'Month': next((c for c in df.columns if '月' in c or '週' in c), None),
                'Strike': next((c for c in df.columns if '履約' in c), None),
                'Type': next((c for c in df.columns if '買賣' in c), None),
                'OI': next((c for c in df.columns if '未沖銷' in c or 'OI' in c), None),
                'Price': next((c for c in df.columns if '結算' in c or '收盤' in c or 'Price' in c), None)
            }
            
            if not all(col_map.values()): continue
            
            df = df.rename(columns={k:v for k,v in col_map.items() if v})[['Month', 'Strike', 'Type', 'OI', 'Price']].dropna()
            
            # 轉數值
            for col in ['Strike', 'OI', 'Price']:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').replace('-', '0'), errors='coerce').fillna(0)
            
            df['Amount'] = df['OI'] * df['Price'] * 50
            if df['OI'].sum() > 0:
                all_data.append({'date': query_date, 'df': df})
                if len(all_data) >= 2: break # 抓兩天算差異
        except: continue
        
    # 🔥 重要修復：若無資料，必須回傳兩個 None，否則主程式會報 TypeError
    if not all_data: return None, None
    
    # 計算 OI 變化
    df_curr = all_data[0]['df']
    if len(all_data) > 1:
        df_prev = all_data[1]['df']
        merged = pd.merge(df_curr, df_prev, on=['Month', 'Strike', 'Type'], how='left', suffixes=('', '_prev')).fillna(0)
        df_curr['OI_Change'] = merged['OI'] - merged['OI_prev']
    else:
        df_curr['OI_Change'] = 0
        
    return df_curr, all_data[0]['date']

# --- 繪圖函式 ---
def plot_tornado(df, title, spot):
    df_call = df[df['Type'].str.contains('Call|買')].copy()
    df_put = df[df['Type'].str.contains('Put|賣')].copy()
    
    # 合併並排序
    data = pd.merge(df_call[['Strike', 'OI', 'Amount']], df_put[['Strike', 'OI', 'Amount']], on='Strike', suffixes=('_C', '_P'), how='outer').fillna(0).sort_values('Strike')
    
    # 過濾範圍 (現貨上下 600 點)
    center = spot if spot else data['Strike'].median()
    if center > 0:
        data = data[(data['Strike'] >= center - 600) & (data['Strike'] <= center + 600)]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=data['Strike'], x=-data['OI_P'], orientation='h', name='Put (支撐)', marker_color='green'))
    fig.add_trace(go.Bar(y=data['Strike'], x=data['OI_C'], orientation='h', name='Call (壓力)', marker_color='red'))
    
    if spot:
        fig.add_hline(y=spot, line_dash="dash", line_color="orange", annotation_text=f"現貨 {spot}")
        
    fig.update_layout(title=title, barmode='overlay', xaxis_title="未平倉量 (OI)", height=600)
    return fig

# --- 主程式 ---
def main():
    st.title("🧛‍♂️ 台指期籌碼戰情室 (極速修復版)")
    
    # 側邊欄重新整理
    if st.sidebar.button("🔄 重新抓取"):
        st.cache_data.clear()
        st.rerun()

    # === 數據抓取區 (即時顯示進度) ===
    status_box = st.empty() # 佔位符，用來顯示進度
    
    with st.spinner("🚀 正在啟動數據引擎..."):
        status_box.text("⏳ 正在連線: 期貨行情...")
        fut_price, fut_date = get_futures_data(status_box)
        
        status_box.text("⏳ 正在連線: 法人期貨...")
        inst_fut = get_institutional_futures(status_box)
        
        status_box.text("⏳ 正在連線: 法人選擇權...")
        inst_opt_df, inst_opt_date = get_institutional_options(status_box)
        
        status_box.text("⏳ 正在連線: 全市場選擇權 (請稍候)...")
        
        # 🔥 安全呼叫：防止 None 解包錯誤
        try:
            opt_df, opt_date = get_option_market(status_box)
        except Exception as e:
            opt_df, opt_date = None, None
            # st.error(f"Debug: {e}") # 需要除錯時可打開

        status_box.empty() # 清除進度文字

    # === 檢查數據是否為空 ===
    if opt_df is None:
        st.error("❌ 數據抓取失敗。可能是期交所目前阻擋連線，或非交易時間。")
        st.warning("建議：請過 10 秒後再按一次「重新抓取」。")
        return

    # === 顯示儀表板 ===
    st.success(f"✅ 數據更新完成！資料日期：{opt_date}")
    
    # 1. 核心指標
    k1, k2, k3 = st.columns(3)
    k1.metric("台指期收盤", f"{int(fut_price)}" if fut_price else "N/A")
    
    call_sum = opt_df[opt_df['Type'].str.contains('Call|買')]['Amount'].sum()
    put_sum = opt_df[opt_df['Type'].str.contains('Put|賣')]['Amount'].sum()
    p_c_ratio = put_sum / call_sum * 100 if call_sum > 0 else 0
    k2.metric("P/C 金額比", f"{p_c_ratio:.1f}%", "偏多" if p_c_ratio > 100 else "偏空")
    
    if inst_fut:
        f_net = inst_fut.get('外資', 0)
        k3.metric("外資期貨淨單", f"{f_net:+,}", delta_color="inverse" if f_net > 0 else "normal")

    # 2. 法人籌碼表格
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if inst_fut:
            st.caption("🏦 三大法人期貨佈局:")
            st.json(inst_fut)
    with c2:
        if inst_opt_df is not None:
            st.caption(f"📊 法人選擇權淨部位 ({inst_opt_date}):")
            st.dataframe(inst_opt_df, height=200, use_container_width=True)

    # 3. 龍捲風圖 (找出最近月)
    st.markdown("---")
    months = sorted(opt_df['Month'].unique())
    if months:
        target_month = months[0] # 預設選最近月
        # 如果最近月是週選且已結算，可能要選下一個，這裡簡單先取第一個
        
        st.subheader(f"🌪️ 籌碼分佈圖 ({target_month})")
        df_target = opt_df[opt_df['Month'] == target_month]
        st.plotly_chart(plot_tornado(df_target, f"{target_month} 支撐壓力區", fut_price), use_container_width=True)

    # 4. 下載
    st.download_button("📥 下載 Excel (CSV)", opt_df.to_csv(index=False).encode('utf-8-sig'), "opt_data.csv")

if __name__ == "__main__":
    main()
