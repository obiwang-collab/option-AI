import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import time
from datetime import datetime, timedelta, timezone
from io import StringIO
import calendar
import re
import google.generativeai as genai
from openai import OpenAI

# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="台指期籌碼戰情室 (AI 決策版)")
TW_TZ = timezone(timedelta(hours=8)) 

# ==========================================
# 🔑 金鑰初始化
# ==========================================
try:
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_KEY = "請輸入你的API_KEY"

try:
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
except:
    OPENAI_KEY = "請輸入你的OPENAI_KEY"

# === 初始化 OpenAI ===
client = OpenAI(api_key=OPENAI_KEY)

# ==========================================
# ⭐ Gemini 初始化
# ==========================================
def configure_gemini(api_key):
    if not api_key or "請輸入" in api_key:
        return None, "尚未設定 API Key"
    
    genai.configure(api_key=api_key)
    try:
        available_models = [
            m.name for m in genai.list_models()
            if 'generateContent' in m.supported_generation_methods
        ]

        for target in ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']:
            for m in available_models:
                if target in m:
                    return genai.GenerativeModel(m), m
        
        if available_models:
            return genai.GenerativeModel(available_models[0]), available_models[0]
        
        return None, "無可用模型"

    except Exception as e:
        return None, f"連線錯誤: {str(e)}"

model, model_name = configure_gemini(GEMINI_KEY)

# 手動修正結算日
MANUAL_SETTLEMENT_FIX = {
    '202501W1': '2025/01/02',
}

# ==========================================
# 🎯 結算日計算
# ==========================================
def get_settlement_date(contract_code):
    code = str(contract_code).strip().upper()
    for key, fix_date in MANUAL_SETTLEMENT_FIX.items():
        if key in code: 
            return fix_date
    try:
        if len(code) < 6:
            return "9999/99/99"

        year = int(code[:4])
        month = int(code[4:6])
        c = calendar.monthcalendar(year, month)

        weds = [wk[calendar.WEDNESDAY] for wk in c if wk[calendar.WEDNESDAY] != 0]
        fridays = [wk[calendar.FRIDAY] for wk in c if wk[calendar.FRIDAY] != 0]

        day = None
        if "W" in code:
            m = re.search(r"W(\d)", code)
            if m:
                w = int(m.group(1))
                if len(weds) >= w:
                    day = weds[w-1]
        elif "F" in code:
            m = re.search(r"F(\d)", code)
            if m:
                w = int(m.group(1))
                if len(fridays) >= w:
                    day = fridays[w-1]
        else:
            if len(weds) >= 3:
                day = weds[2]

        return f"{year}/{month:02d}/{day:02d}" if day else "9999/99/99"

    except:
        return "9999/99/99"

# ==========================================
# 📈 現貨資料
# ==========================================
@st.cache_data(ttl=60)
def get_realtime_data():
    taiex = None
    ts = int(time.time())
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        url = f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_t00.tw&json=1&delay=0&_={ts}000"
        res = requests.get(url, timeout=2)
        data = res.json()
        if 'msgArray' in data and len(data['msgArray']) > 0:
            val = data['msgArray'][0].get('z', '-')
            if val == '-':
                val = data['msgArray'][0].get('o', '-')
            if val != '-':
                taiex = float(val)
    except:
        pass

    if taiex is None:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII?interval=1m&range=1d&_={ts}"
            res = requests.get(url, headers=headers, timeout=3)
            data = res.json()
            price = data['chart']['result'][0]['meta'].get('regularMarketPrice')
            if price:
                taiex = float(price)
        except:
            pass

    return taiex

# ==========================================
# 🧾 選擇權資料
# ==========================================
@st.cache_data(ttl=300)
def get_option_data():
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    headers = {'User-Agent': 'Mozilla/5.0'}

    for i in range(5):
        query_date = (datetime.now(tz=TW_TZ) - timedelta(days=i)).strftime('%Y/%m/%d')
        payload = {
            'queryType': '2', 'marketCode': '0', 'dateaddcnt': '',
            'commodity_id': 'TXO', 'commodity_id2': '',
            'queryDate': query_date, 'MarketCode': '0', 'commodity_idt': 'TXO'
        }
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=5)
            if "查無資料" in res.text or len(res.text) < 500:
                continue 

            dfs = pd.read_html(StringIO(res.text))
            df = dfs[0]

            df.columns = [
                str(c).replace(' ', '').replace('*', '').replace('契約', '').strip()
                for c in df.columns
            ]

            month_col = next((c for c in df.columns if '月' in c or '週' in c), None)
            strike_col = next((c for c in df.columns if '履約' in c), None)
            type_col = next((c for c in df.columns if '買賣' in c), None)
            oi_col = next((c for c in df.columns if '未沖銷' in c or 'OI' in c), None)
            price_col = next((c for c in df.columns if '結算' in c or '收盤' in c), None)
            vol_col = next((c for c in df.columns if '成交量' in c), None)

            if not all([month_col, strike_col, type_col, oi_col, price_col]):
                continue

            rename_dict = {
                month_col:'Month', strike_col:'Strike',
                type_col:'Type', oi_col:'OI', price_col:'Price'
            }
            if vol_col: 
                rename_dict[vol_col] = 'Volume'

            df = df.rename(columns=rename_dict)
            cols = ['Month','Strike','Type','OI','Price']
            if 'Volume' in df.columns:
                cols.append('Volume')
            df = df[cols].copy()

            df = df.dropna(subset=['Type'])
            df['Type'] = df['Type'].astype(str).strip()
            df['Strike'] = pd.to_numeric(df['Strike'], errors='coerce')
            df['OI'] = pd.to_numeric(df['OI'], errors='coerce').fillna(0)
            df['Price'] = pd.to_numeric(
                df['Price'].astype(str).str.replace(',', '').replace('-', '0'),
                errors='coerce'
            ).fillna(0)

            if 'Volume' in df.columns:
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)

            df['Amount'] = df['OI'] * df['Price'] * 50

            if df['OI'].sum() == 0:
                continue 

            return df, query_date

        except:
            continue

    return None, None

# ==========================================
# 🎨 龍捲風圖
# ==========================================
def plot_tornado_chart(df_target, title_text, spot_price):
    is_call = df_target['Type'].str.contains('買|Call', case=False, na=False)
    df_call = df_target[is_call][['Strike', 'OI', 'Amount']].rename(
        columns={'OI':'Call_OI','Amount':'Call_Amt'}
    )
    df_put = df_target[~is_call][['Strike', 'OI', 'Amount']].rename(
        columns={'OI':'Put_OI','Amount':'Put_Amt'}
    )
    data = pd.merge(df_call, df_put, on='Strike', how='outer').fillna(0)
    data = data.sort_values('Strike')

    total_put_money = data['Put_Amt'].sum()
    total_call_money = data['Call_Amt'].sum()

    data = data[(data['Call_OI'] > 300) | (data['Put_OI'] > 300)]

    FOCUS_RANGE = 1200
    center_price = spot_price if spot_price else data['Strike'].median()
    data = data[(data['Strike'] >= center_price-FOCUS_RANGE) & 
                (data['Strike'] <= center_price+FOCUS_RANGE)]

    max_oi = max(data['Put_OI'].max(), data['Call_OI'].max())
    x_limit = max_oi * 1.1

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=data['Strike'], x=-data['Put_OI'], orientation='h',
        name='Put (支撐)', marker_color='#2ca02c', opacity=0.85,
        customdata=data['Put_Amt'] / 100000000,
        hovertemplate='<b>履約價: %{y}</b><br>Put OI: %{x}<br>Put 市值: %{customdata:.2f}億'
    ))

    fig.add_trace(go.Bar(
        y=data['Strike'], x=data['Call_OI'], orientation='h',
        name='Call (壓力)', marker_color='#d62728', opacity=0.85,
        customdata=data['Call_Amt'] / 100000000,
        hovertemplate='<b>履約價: %{y}</b><br>Call OI: %{x}<br>Call 市值: %{customdata:.2f}億'
    ))

    annotations = []
    if spot_price:
        fig.add_hline(
            y=spot_price, line_dash="dash", line_color="#ff7f0e", line_width=2
        )
        annotations.append(dict(
            x=1, y=spot_price, xref="paper", yref="y",
            text=f" 現貨 {int(spot_price)} ", showarrow=False,
            xanchor="left", font=dict(color="white", size=12),
            bgcolor="#ff7f0e", bordercolor="#ff7f0e"
        ))

    annotations.append(dict(
        x=0.02, y=1.05, xref="paper", yref="paper",
        text=f"<b>Put 總金額</b><br>{total_put_money/100000000:.1f} 億",
        showarrow=False, font=dict(color="#2ca02c", size=14),
        bgcolor="white", bordercolor="#2ca02c", borderwidth=2
    ))
    annotations.append(dict(
        x=0.98, y=1.05, xref="paper", yref="paper",
        text=f"<b>Call 總金額</b><br>{total_call_money/100000000:.1f} 億",
        showarrow=False, font=dict(color="#d62728", size=14),
        bgcolor="white", bordercolor="#d62728", borderwidth=2
    ))

    fig.update_layout(
        title=dict(text=title_text, x=0.5),
        xaxis=dict(
            title='未平倉量 (OI)',
            range=[-x_limit, x_limit],
            tickmode='array',
            tickvals=[-x_limit*0.75, -x_limit*0.5, -x_limit*0.25, 0,
                      x_limit*0.25, x_limit*0.5, x_limit*0.75],
            ticktext=[
                f"{int(x_limit*0.75)}", f"{int(x_limit*0.5)}", f"{int(x_limit*0.25)}",
                "0",
                f"{int(x_limit*0.25)}", f"{int(x_limit*0.5)}", f"{int(x_limit*0.75)}"
            ]
        ),
        yaxis=dict(dtick=100),
        barmode='overlay',
        height=750,
        annotations=annotations
    )

    return fig

# ================================================
# 🤖 Gemini 短線分析
# ================================================
def ask_gemini_brief(df, taiex_price):
    if not model:
        return f"⚠️ Gemini 連線失敗: {model_name}"

    try:
        df_ai = df.copy()
        df_ai = df_ai.nlargest(40, 'Amount')
        data_str = df_ai.to_csv(index=False)

        prompt = f"""
你是一個台指期貨短線交易助手。
大盤現貨：{taiex_price}

請直接提供：
1. 市場偏多 / 偏空 / 震盪
2. 今日短線建議（反彈空 / 拉回多 / 區間）
3. 主力可能控盤方式

不要解釋過程，不要講支撐壓力計算方式。
字數 120 字內。

資料：
{data_str}
"""

        res = model.generate_content(prompt)
        return res.text
    
    except Exception as e:
        return f"Gemini 分析錯誤: {e}"

# ================================================
# 🤖 ChatGPT 短線分析
# ================================================
def ask_chatgpt_brief(df, taiex_price):
    if "請輸入" in OPENAI_KEY:
        return "⚠️ 尚未設定 OpenAI API Key"

    try:
        df_ai = df.copy()
        df_ai = df_ai.nlargest(40, 'Amount')
        data_str = df_ai.to_csv(index=False)

        prompt = f"""
你是一位台指期主力視角操盤手。
大盤：{taiex_price}

請直述結論：
1. 多空（偏多/偏空/震盪）
2. 主力盤中策略（拉高洗、壓盤、誘空等）
3. 短線建議（拉回多 / 反彈空 / 區間）

字數限制 120 字。

資料：
{data_str}
"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message["content"]

    except Exception as e:
        return f"ChatGPT 分析錯誤: {e}"

# ================================================
# ⚔️ AI 兩者比較
# ================================================
def compare_ai(gpt_text, gem_text):
    def detect(text):
        if "偏多" in text:
            return "偏多"
        if "偏空" in text:
            return "偏空"
        if "震盪" in text:
            return "震盪"
        return "無明確判斷"

    gpt = detect(gpt_text)
    gem = detect(gem_text)

    if gpt == gem:
        consensus = f"兩者一致：{gpt}。"
    else:
        consensus = f"觀點不同：ChatGPT={gpt}, Gemini={gem} → 高機率震盪。"

    return f"""
### 🤖 ChatGPT 與 Gemini 短線分析比較

#### ChatGPT：
{gpt_text}

---

#### Gemini：
{gem_text}

---

### 📌 多空結論：
{consensus}
"""

# ================================================
# 🏁 主程式
# ================================================
def main():
    st.title("🤖 台指期籌碼戰情室 (AI 決策版)")

    if st.sidebar.button("🔄 重新整理"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("連線期交所中..."):
        df, data_date = get_option_data()
        taiex_now = get_realtime_data()

    if df is None:
        st.error("查無資料")
        return

    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.sidebar.download_button("📥 下載完整數據", csv, "option.csv")

    total_call_amt = df[df['Type'].str.contains('買|Call', case=False)]['Amount'].sum()
    total_put_amt = df[df['Type'].str.contains('賣|Put', case=False)]['Amount'].sum()
    pc_ratio_amt = total_put_amt * 100 / total_call_amt if total_call_amt > 0 else 0

    st.markdown("### 💡 AI 短線錦囊（Gemini）")
    if st.button("✨ 取得 Gemini 建議"):
        with st.spinner("AI 分析中..."):
            advice = ask_gemini_brief(df, taiex_now)
        st.info(advice)

    # ====================
    # ⚔️ 新增 AI 對決分析
    # ====================
    st.markdown("### 🤖 ChatGPT vs Gemini 短線分析比較")
    if st.button("⚔️ AI 雙模型短線對決分析"):
        with st.spinner("AI 分析中..."):
            gpt = ask_chatgpt_brief(df, taiex_now)
            gem = ask_gemini_brief(df, taiex_now)
            result = compare_ai(gpt, gem)
        st.markdown(result)

    # ==========================================
    # 指標區
    c1, c2, c3, c4 = st.columns([1.2,0.8,1,1])
    c1.markdown(f"製圖時間<br><b>{datetime.now(tz=TW_TZ).strftime('%Y/%m/%d %H:%M:%S')}</b>", unsafe_allow_html=True)
    c2.metric("大盤現貨", f"{int(taiex_now) if taiex_now else 'N/A'}")
    trend = "偏多" if pc_ratio_amt > 100 else "偏空"
    c3.metric("P/C 金額比", f"{pc_ratio_amt:.1f}%", trend)
    c4.metric("資料日期", data_date)
    st.markdown("---")

    # ==========================================
    # 繪圖
    unique_codes = df['Month'].unique()
    all_contracts = []

    for code in unique_codes:
        s_date = get_settlement_date(code)
        if s_date == "9999/99/99" or s_date <= data_date:
            continue
        all_contracts.append({'code': code, 'date': s_date})

    all_contracts.sort(key=lambda x: x['date'])
    plot_targets = []

    if all_contracts:
        nearest = all_contracts[0]
        plot_targets.append({'title':'最近結算','info':nearest})

        monthly = next((c for c in all_contracts if len(c['code']) == 6), None)
        if monthly and monthly['code'] != nearest['code']:
            plot_targets.append({'title':'當月月選','info':monthly})

    cols = st.columns(len(plot_targets))

    for i, target in enumerate(plot_targets):
        with cols[i]:
            code = target['info']['code']
            s_date = target['info']['date']
            df_target = df[df['Month'] == code]

            sub_call_amt = df_target[df_target['Type'].str.contains('Call|買', case=False)]['Amount'].sum()
            sub_put_amt = df_target[df_target['Type'].str.contains('Put|賣', case=False)]['Amount'].sum()
            sub_ratio = sub_put_amt * 100 / sub_call_amt if sub_call_amt > 0 else 0

            title = (
                f"<b>【{target['title']}】 {code}</b><br>"
                f"<span style='font-size:14px;'>結算: {s_date}</span><br>"
                f"<span style='font-size:14px;'>P/C金額比: {sub_ratio:.1f}% "
                f"({'偏多' if sub_ratio>100 else '偏空'})</span>"
            )

            fig = plot_tornado_chart(df_target, title, taiex_now)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
