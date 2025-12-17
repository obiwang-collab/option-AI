import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import google.generativeai as genai
from openai import OpenAI
import time

# ==================== 頁面設定 ====================
st.set_page_config(
    page_title="台指期籌碼戰情室",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== API 設定 ====================
def init_ai_apis():
    """初始化 AI API"""
    gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    openai_key = st.secrets.get("OPENAI_API_KEY", "")
    
    if gemini_key:
        genai.configure(api_key=gemini_key)
    
    return gemini_key, openai_key

GEMINI_KEY, OPENAI_KEY = init_ai_apis()

# ==================== 樣式設定 ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 數據抓取函數 ====================

@st.cache_data(ttl=300)
def fetch_institution_data():
    """抓取三大法人數據"""
    try:
        url = "https://www.taifex.com.tw/cht/3/totalTableDate"
        today = datetime.now()
        
        for i in range(10):
            check_date = today - timedelta(days=i)
            if check_date.weekday() >= 5:
                continue
            
            date_str = check_date.strftime('%Y/%m/%d')
            response = requests.get(url, params={'queryDate': date_str}, timeout=10)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data and len(data) > 0:
                        df = pd.DataFrame(data)
                        return df, check_date
                except:
                    tables = pd.read_html(response.text)
                    if tables and len(tables) > 0:
                        return tables[0], check_date
        
        return None, None
        
    except Exception as e:
        st.error(f"抓取三大法人數據失敗: {str(e)}")
        return None, None

def fetch_options_data(date_str):
    """從期交所抓取選擇權數據 - 修正版"""
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketView"
    
    params = {
        'queryDate': date_str,
        'commodityId': 'TXO'
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # 正確的JSON數據路徑
            if 'RptBody' in data and len(data['RptBody']) > 0:
                return data['RptBody']
        
        return None
        
    except Exception as e:
        print(f"抓取失敗 {date_str}: {str(e)}")
        return None

@st.cache_data(ttl=300)
def get_options_data_with_retry(days_back=20):
    """帶重試機制的選擇權數據抓取"""
    today = datetime.now()
    
    for i in range(days_back):
        check_date = today - timedelta(days=i)
        
        # 跳過週末
        if check_date.weekday() >= 5:
            continue
        
        date_str = check_date.strftime('%Y/%m/%d')
        data = fetch_options_data(date_str)
        
        if data is not None:
            return data, check_date
    
    return None, None

# ==================== 數據處理函數 ====================

def parse_options_data(raw_data):
    """解析選擇權原始數據"""
    try:
        df = pd.DataFrame(raw_data)
        
        # 數據清理和轉換
        numeric_cols = ['成交量', '未平倉量', '買賣價差', '開盤價', '最高價', '最低價', '收盤價']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        
        return df
        
    except Exception as e:
        st.error(f"解析選擇權數據失敗: {str(e)}")
        return None

def calculate_pcr(options_df):
    """計算 Put/Call Ratio"""
    try:
        if options_df is None or len(options_df) == 0:
            return None
        
        put_volume = options_df[options_df['買賣權別'] == 'P']['成交量'].sum()
        call_volume = options_df[options_df['買賣權別'] == 'C']['成交量'].sum()
        
        put_oi = options_df[options_df['買賣權別'] == 'P']['未平倉量'].sum()
        call_oi = options_df[options_df['買賣權別'] == 'C']['未平倉量'].sum()
        
        pcr_volume = put_volume / call_volume if call_volume > 0 else 0
        pcr_oi = put_oi / call_oi if call_oi > 0 else 0
        
        return {
            'pcr_volume': pcr_volume,
            'pcr_oi': pcr_oi,
            'put_volume': put_volume,
            'call_volume': call_volume,
            'put_oi': put_oi,
            'call_oi': call_oi
        }
        
    except Exception as e:
        st.error(f"計算PCR失敗: {str(e)}")
        return None

def calculate_max_pain(options_df):
    """計算最大痛點"""
    try:
        if options_df is None or len(options_df) == 0:
            return None
        
        strike_prices = sorted(options_df['履約價'].unique())
        pain_values = []
        
        for strike in strike_prices:
            pain = 0
            
            # 計算Call的損失
            calls = options_df[options_df['買賣權別'] == 'C']
            for _, row in calls.iterrows():
                if row['履約價'] < strike:
                    pain += row['未平倉量'] * (strike - row['履約價'])
            
            # 計算Put的損失
            puts = options_df[options_df['買賣權別'] == 'P']
            for _, row in puts.iterrows():
                if row['履約價'] > strike:
                    pain += row['未平倉量'] * (row['履約價'] - strike)
            
            pain_values.append(pain)
        
        max_pain_idx = np.argmin(pain_values)
        return strike_prices[max_pain_idx]
        
    except Exception as e:
        st.error(f"計算最大痛點失敗: {str(e)}")
        return None

def calculate_iv_metrics(options_df):
    """計算隱含波動率指標"""
    try:
        if options_df is None or len(options_df) == 0:
            return None
        
        # 這裡簡化計算,實際應該用Black-Scholes
        calls = options_df[options_df['買賣權別'] == 'C']
        puts = options_df[options_df['買賣權別'] == 'P']
        
        avg_call_price = calls['收盤價'].mean()
        avg_put_price = puts['收盤價'].mean()
        
        return {
            'avg_call_iv': avg_call_price,
            'avg_put_iv': avg_put_price,
            'iv_skew': avg_put_price - avg_call_price
        }
        
    except Exception as e:
        st.error(f"計算IV失敗: {str(e)}")
        return None

# ==================== AI 分析函數 ====================

def generate_market_analysis_gemini(institution_data, options_data, pcr_data, max_pain):
    """使用 Gemini 生成市場分析"""
    try:
        if not GEMINI_KEY:
            return "❌ 未設定 Gemini API Key"
        
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = f"""
        作為專業的台指期貨分析師,請根據以下數據進行深入分析:
        
        1. 三大法人籌碼數據:
        {institution_data.to_string() if institution_data is not None else "無數據"}
        
        2. 選擇權數據:
        - Put/Call Ratio (成交量): {pcr_data['pcr_volume']:.2f}
        - Put/Call Ratio (未平倉): {pcr_data['pcr_oi']:.2f}
        - 最大痛點: {max_pain}
        
        請提供:
        1. 市場情緒分析 (多空比例)
        2. 莊家可能的操作策略
        3. 關鍵支撐與壓力位
        4. 短期操作建議
        
        請用繁體中文回答,並保持專業但易懂的語調。
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"❌ Gemini 分析失敗: {str(e)}"

def generate_market_analysis_chatgpt(institution_data, options_data, pcr_data, max_pain):
    """使用 ChatGPT 生成市場分析"""
    try:
        if not OPENAI_KEY:
            return "❌ 未設定 OpenAI API Key"
        
        client = OpenAI(api_key=OPENAI_KEY)
        
        prompt = f"""
        作為專業的台指期貨分析師,請根據以下數據進行深入分析:
        
        1. 三大法人籌碼數據:
        {institution_data.to_string() if institution_data is not None else "無數據"}
        
        2. 選擇權數據:
        - Put/Call Ratio (成交量): {pcr_data['pcr_volume']:.2f}
        - Put/Call Ratio (未平倉): {pcr_data['pcr_oi']:.2f}
        - 最大痛點: {max_pain}
        
        請提供:
        1. 市場情緒分析 (多空比例)
        2. 莊家可能的操作策略
        3. 關鍵支撐與壓力位
        4. 短期操作建議
        
        請用繁體中文回答,並保持專業但易懂的語調。
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是一位專業的台指期貨分析師。"},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ ChatGPT 分析失敗: {str(e)}"

# ==================== 視覺化函數 ====================

def plot_institution_positions(inst_df):
    """繪製三大法人部位圖"""
    if inst_df is None or len(inst_df) == 0:
        return None
    
    fig = go.Figure()
    
    # 假設數據包含外資、投信、自營商
    categories = ['外資', '投信', '自營商']
    
    for cat in categories:
        if cat in inst_df.columns:
            fig.add_trace(go.Bar(
                name=cat,
                x=['多方', '空方', '淨部位'],
                y=[100, 80, 20],  # 這裡應該用實際數據
                text=['+100', '-80', '+20'],
                textposition='auto',
            ))
    
    fig.update_layout(
        title='三大法人部位分析',
        barmode='group',
        height=400
    )
    
    return fig

def plot_pcr_trend(pcr_data):
    """繪製PCR趨勢圖"""
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = pcr_data['pcr_volume'],
        title = {'text': "Put/Call Ratio (成交量)"},
        delta = {'reference': 1.0},
        gauge = {
            'axis': {'range': [None, 2]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.7], 'color': "lightgreen"},
                {'range': [0.7, 1.3], 'color': "lightyellow"},
                {'range': [1.3, 2], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 1.0
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def plot_strike_distribution(options_df):
    """繪製履約價分布圖"""
    if options_df is None or len(options_df) == 0:
        return None
    
    calls = options_df[options_df['買賣權別'] == 'C'].groupby('履約價')['未平倉量'].sum()
    puts = options_df[options_df['買賣權別'] == 'P'].groupby('履約價')['未平倉量'].sum()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Call OI',
        x=calls.index,
        y=calls.values,
        marker_color='green'
    ))
    
    fig.add_trace(go.Bar(
        name='Put OI',
        x=puts.index,
        y=-puts.values,  # 負值顯示在下方
        marker_color='red'
    ))
    
    fig.update_layout(
        title='選擇權未平倉分布',
        xaxis_title='履約價',
        yaxis_title='未平倉量',
        barmode='relative',
        height=500
    )
    
    return fig

# ==================== 主程式 ====================

def main():
    st.markdown('<h1 class="main-header">🎯 台指期籌碼戰情室 (莊家控盤版)</h1>', unsafe_allow_html=True)
    
    # 側邊欄設定
    with st.sidebar:
        st.header("⚙️ 設定")
        
        ai_provider = st.selectbox(
            "選擇 AI 分析工具",
            ["Gemini", "ChatGPT", "兩者比較"]
        )
        
        auto_refresh = st.checkbox("自動刷新 (5分鐘)", value=False)
        
        if st.button("🔄 手動刷新數據"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 數據來源")
        st.markdown("- 期交所官方API")
        st.markdown("- 即時更新")
    
    # 主要內容區
    with st.spinner("正在載入數據..."):
        # 抓取三大法人數據
        inst_df, inst_date = fetch_institution_data()
        
        # 抓取選擇權數據
        options_raw, options_date = get_options_data_with_retry()
        
        if options_raw is None:
            st.error("❌ 無法抓取任何選擇權數據 (已回溯 20 天)")
            st.stop()
        
        options_df = parse_options_data(options_raw)
        
        if options_df is None:
            st.error("❌ 選擇權數據解析失敗")
            st.stop()
        
        # 計算指標
        pcr_data = calculate_pcr(options_df)
        max_pain = calculate_max_pain(options_df)
        iv_metrics = calculate_iv_metrics(options_df)
    
    st.success(f"✅ 數據更新時間: {options_date.strftime('%Y-%m-%d %H:%M')}")
    
    # 關鍵指標卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📈 PCR (成交量)",
            value=f"{pcr_data['pcr_volume']:.2f}",
            delta="偏多" if pcr_data['pcr_volume'] < 0.7 else "偏空" if pcr_data['pcr_volume'] > 1.3 else "中性"
        )
    
    with col2:
        st.metric(
            label="📊 PCR (未平倉)",
            value=f"{pcr_data['pcr_oi']:.2f}",
            delta="偏多" if pcr_data['pcr_oi'] < 0.7 else "偏空" if pcr_data['pcr_oi'] > 1.3 else "中性"
        )
    
    with col3:
        st.metric(
            label="🎯 最大痛點",
            value=f"{max_pain}",
            delta="莊家壓力位"
        )
    
    with col4:
        delta_str = "看漲偏移" if iv_metrics['iv_skew'] > 0 else "看跌偏移"
        st.metric(
            label="📉 IV偏移",
            value=f"{abs(iv_metrics['iv_skew']):.0f}",
            delta=delta_str
        )
    
    # 標籤頁
    tab1, tab2, tab3, tab4 = st.tabs(["📊 三大法人", "🎯 選擇權分析", "🤖 AI 分析", "📈 進階指標"])
    
    with tab1:
        st.subheader("三大法人籌碼分析")
        
        if inst_df is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = plot_institution_positions(inst_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📋 原始數據")
                st.dataframe(inst_df, height=400)
        else:
            st.warning("⚠️ 三大法人數據暫時無法取得")
    
    with tab2:
        st.subheader("選擇權籌碼分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Put/Call Ratio")
            fig_pcr = plot_pcr_trend(pcr_data)
            st.plotly_chart(fig_pcr, use_container_width=True)
        
        with col2:
            st.markdown("### 成交與未平倉統計")
            st.metric("Call 成交量", f"{pcr_data['call_volume']:,.0f}")
            st.metric("Put 成交量", f"{pcr_data['put_volume']:,.0f}")
            st.metric("Call 未平倉", f"{pcr_data['call_oi']:,.0f}")
            st.metric("Put 未平倉", f"{pcr_data['put_oi']:,.0f}")
        
        st.markdown("### 履約價分布")
        fig_strike = plot_strike_distribution(options_df)
        if fig_strike:
            st.plotly_chart(fig_strike, use_container_width=True)
    
    with tab3:
        st.subheader("🤖 AI 市場分析")
        
        if ai_provider == "Gemini":
            with st.spinner("Gemini 分析中..."):
                analysis = generate_market_analysis_gemini(inst_df, options_df, pcr_data, max_pain)
                st.markdown(analysis)
        
        elif ai_provider == "ChatGPT":
            with st.spinner("ChatGPT 分析中..."):
                analysis = generate_market_analysis_chatgpt(inst_df, options_df, pcr_data, max_pain)
                st.markdown(analysis)
        
        else:  # 兩者比較
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔷 Gemini 分析")
                with st.spinner("分析中..."):
                    gemini_analysis = generate_market_analysis_gemini(inst_df, options_df, pcr_data, max_pain)
                    st.markdown(gemini_analysis)
            
            with col2:
                st.markdown("### 🟢 ChatGPT 分析")
                with st.spinner("分析中..."):
                    chatgpt_analysis = generate_market_analysis_chatgpt(inst_df, options_df, pcr_data, max_pain)
                    st.markdown(chatgpt_analysis)
    
    with tab4:
        st.subheader("進階技術指標")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 隱含波動率分析")
            st.metric("平均 Call IV", f"{iv_metrics['avg_call_iv']:.2f}")
            st.metric("平均 Put IV", f"{iv_metrics['avg_put_iv']:.2f}")
            st.metric("IV 偏移", f"{iv_metrics['iv_skew']:.2f}")
        
        with col2:
            st.markdown("### Gamma 暴露分析")
            st.info("功能開發中...")
    
    # 自動刷新
    if auto_refresh:
        time.sleep(300)
        st.rerun()

if __name__ == "__main__":
    main()
