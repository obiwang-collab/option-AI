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
import numpy as np
from scipy.stats import norm

# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="台指期權碼戰情室 (莊家絕殺版)")
TW_TZ = timezone(timedelta(hours=8))

# ==========================================
# 🔑 金鑰設定區 (雲端安全版)
# ==========================================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = "請輸入你的GEMINI_API_KEY"

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = "請輸入你的OPENAI_API_KEY"

# --- 智慧模型設定:Gemini ---
def configure_gemini(api_key):
    if not api_key or "請輸入" in api_key:
        return None, "尚未設定 GEMINI Key"
    genai.configure(api_key=api_key)
    try:
        available_models = [
            m.name for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        for target in ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]:
            for m in available_models:
                if target in m:
                    return genai.GenerativeModel(m), m
        if available_models:
            return genai.GenerativeModel(available_models[0]), available_models[0]
        return None, "無可用模型"
    except Exception as e:
        return None, f"連線錯誤: {str(e)}"

# --- 智慧模型設定:OpenAI ---
def configure_openai(api_key):
    if not api_key or "請輸入" in api_key:
        return None, "尚未設定 OPENAI Key"
    try:
        client = OpenAI(api_key=api_key)
        _ = client.models.list()
        return client, "gpt-4o-mini"
    except Exception as e:
        return None, f"連線錯誤: {str(e)}"

gemini_model, gemini_model_name = configure_gemini(GEMINI_API_KEY)
openai_client, openai_model_name = configure_openai(OPENAI_API_KEY)

MANUAL_SETTLEMENT_FIX = {"202501W1": "2025/01/02"}

# --- 結算日計算 ---
def get_settlement_date(contract_code: str) -> str:
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
        wednesdays = [week[calendar.WEDNESDAY] for week in c if week[calendar.WEDNESDAY] != 0]
        fridays = [week[calendar.FRIDAY] for week in c if week[calendar.FRIDAY] != 0]
        day = None
        if "W" in code:
            match = re.search(r"W(\d)", code)
            if match:
                week_num = int(match.group(1))
                if len(wednesdays) >= week_num:
                    day = wednesdays[week_num - 1]
        elif "F" in code:
            match = re.search(r"F(\d)", code)
            if match:
                week_num = int(match.group(1))
                if len(fridays) >= week_num:
                    day = fridays[week_num - 1]
        else:
            if len(wednesdays) >= 3:
                day = wednesdays[2]
        if day:
            return f"{year}/{month:02d}/{day:02d}"
        else:
            return "9999/99/99"
    except Exception:
        return "9999/99/99"

# --- 現貨即時價 (強化版) ---
@st.cache_data(ttl=60)
def get_realtime_data():
    taiex = None
    ts = int(time.time())
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII?interval=1d&range=1d&_={ts}"
        res = requests.get(url, headers=headers, timeout=5)
        data = res.json()
        meta = data["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice") or meta.get("chartPreviousClose")
        if price:
            taiex = float(price)
    except Exception:
        pass
    if taiex is None:
        try:
            url = f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_t00.tw&json=1&delay=0&_={ts}000"
            res = requests.get(url, timeout=3)
            data = res.json()
            if "msgArray" in data and len(data["msgArray"]) > 0:
                val = data["msgArray"][0].get("z", "-")
                if val == "-": val = data["msgArray"][0].get("o", "-")
                if val == "-": val = data["msgArray"][0].get("y", "-")
                if val != "-": taiex = float(val)
        except Exception:
            pass
    return taiex

# ==========================================
# 🆕 新功能 1: 期交所選擇權資料 (全履約價 + 近三日)
# ==========================================
@st.cache_data(ttl=300)
def get_option_data_full(days_back=3):
    """抓取近N日的完整選擇權資料（不過濾OI）"""
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    headers = {"User-Agent": "Mozilla/5.0"}
    all_data = []
    
    for i in range(days_back + 5):
        query_date = (datetime.now(tz=TW_TZ) - timedelta(days=i)).strftime("%Y/%m/%d")
        payload = {
            "queryType": "2", "marketCode": "0", "dateaddcnt": "",
            "commodity_id": "TXO", "commodity_id2": "",
            "queryDate": query_date, "MarketCode": "0", "commodity_idt": "TXO",
        }
        try:
            res = requests.post(url, data=payload, headers=headers, timeout=5)
            if "查無資料" in res.text or len(res.text) < 500:
                continue
            dfs = pd.read_html(StringIO(res.text))
            df = dfs[0]
            df.columns = [str(c).replace(" ", "").replace("*", "").replace("契約", "").strip() for c in df.columns]
            
            month_col = next((c for c in df.columns if "月" in c or "週" in c), None)
            strike_col = next((c for c in df.columns if "履約" in c), None)
            type_col = next((c for c in df.columns if "買賣" in c), None)
            oi_col = next((c for c in df.columns if "未沖銷" in c or "OI" in c), None)
            price_col = next((c for c in df.columns if "結算" in c or "收盤" in c or "Price" in c), None)
            
            if not all([month_col, strike_col, type_col, oi_col, price_col]):
                continue
            
            df = df.rename(columns={
                month_col: "Month", strike_col: "Strike",
                type_col: "Type", oi_col: "OI", price_col: "Price",
            })
            
            cols_to_keep = ["Month", "Strike", "Type", "OI", "Price"]
            df = df[cols_to_keep].copy()
            df = df.dropna(subset=["Type"])
            df["Type"] = df["Type"].astype(str).str.strip()
            df["Strike"] = pd.to_numeric(df["Strike"].astype(str).str.replace(",", ""), errors="coerce")
            df["OI"] = pd.to_numeric(df["OI"].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
            df["Price"] = df["Price"].astype(str).str.replace(",", "").replace("-", "0")
            df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)
            df["Amount"] = df["OI"] * df["Price"] * 50
            df["Date"] = query_date
            
            if df["OI"].sum() > 0:
                all_data.append(df)
                if len(all_data) >= days_back:
                    break
        except Exception:
            continue
    
    if not all_data:
        return None, None
    
    return all_data, all_data[0]["Date"].iloc[0] if len(all_data) > 0 else None

# ==========================================
# 🆕 新功能 2: 計算近三日 OI 增減
# ==========================================
def calculate_oi_changes(data_list):
    """計算近三日的OI變化"""
    if len(data_list) < 2:
        return None
    
    df_today = data_list[0].copy()
    df_yesterday = data_list[1].copy() if len(data_list) > 1 else None
    df_2days = data_list[2].copy() if len(data_list) > 2 else None
    
    df_today['OI_Today'] = df_today['OI']
    changes = df_today[['Month', 'Strike', 'Type', 'OI_Today']].copy()
    
    if df_yesterday is not None:
        df_yesterday['OI_Y1'] = df_yesterday['OI']
        changes = changes.merge(
            df_yesterday[['Month', 'Strike', 'Type', 'OI_Y1']], 
            on=['Month', 'Strike', 'Type'], 
            how='left'
        )
        changes['OI_Y1'] = changes['OI_Y1'].fillna(0)
        changes['Change_1D'] = changes['OI_Today'] - changes['OI_Y1']
    
    if df_2days is not None:
        df_2days['OI_Y2'] = df_2days['OI']
        changes = changes.merge(
            df_2days[['Month', 'Strike', 'Type', 'OI_Y2']], 
            on=['Month', 'Strike', 'Type'], 
            how='left'
        )
        changes['OI_Y2'] = changes['OI_Y2'].fillna(0)
        changes['Change_3D'] = changes['OI_Today'] - changes['OI_Y2']
    
    return changes

# ==========================================
# 🆕 新功能 3: IV & Skew 計算 (簡化版)
# ==========================================
def calculate_iv_and_skew(df, spot_price):
    """計算ATM附近的隱含波動率與25Δ Risk Reversal"""
    if spot_price is None or spot_price <= 0:
        return None
    
    df_sorted = df.copy()
    df_sorted['Distance'] = abs(df_sorted['Strike'] - spot_price)
    atm_strike = df_sorted.loc[df_sorted['Distance'].idxmin(), 'Strike']
    
    df_atm = df_sorted[
        (df_sorted['Strike'] >= atm_strike - 200) & 
        (df_sorted['Strike'] <= atm_strike + 200)
    ].copy()
    
    df_atm['IV_Approx'] = df_atm['Price'] / (spot_price * 0.01)
    
    call_25d = df_atm[df_atm['Type'].str.contains('Call|買', case=False)].nlargest(5, 'OI')
    put_25d = df_atm[df_atm['Type'].str.contains('Put|賣', case=False)].nlargest(5, 'OI')
    
    iv_call_25d = call_25d['IV_Approx'].mean() if not call_25d.empty else 0
    iv_put_25d = put_25d['IV_Approx'].mean() if not put_25d.empty else 0
    
    skew = iv_call_25d - iv_put_25d
    
    return {
        'ATM_Strike': atm_strike,
        'ATM_IV': df_atm['IV_Approx'].mean(),
        'Call_25D_IV': iv_call_25d,
        'Put_25D_IV': iv_put_25d,
        'Skew_25D': skew
    }

# ==========================================
# 🆕 新功能 4: 期貨價格 & 基差 & 外資部位
# ==========================================
@st.cache_data(ttl=300)
def get_futures_and_institutional():
    """抓取期貨價格、基差、外資部位"""
    headers = {"User-Agent": "Mozilla/5.0"}
    result = {'futures_price': None, 'basis': None, 'foreign_net': None}
    
    try:
        url = "https://www.taifex.com.tw/cht/3/futDailyMarketReport"
        today = datetime.now(tz=TW_TZ).strftime("%Y/%m/%d")
        payload = {
            "queryType": "2", "marketCode": "0",
            "commodity_id": "TX", "queryDate": today
        }
        res = requests.post(url, data=payload, headers=headers, timeout=5)
        dfs = pd.read_html(StringIO(res.text))
        if dfs:
            df_fut = dfs[0]
            price_col = next((c for c in df_fut.columns if "結算" in str(c) or "收盤" in str(c)), None)
            if price_col:
                price_str = str(df_fut[price_col].iloc[0]).replace(",", "")
                result['futures_price'] = float(price_str)
    except Exception:
        pass
    
    try:
        url = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
        res = requests.post(url, data=payload, headers=headers, timeout=5)
        dfs = pd.read_html(StringIO(res.text))
        if dfs and len(dfs) > 1:
            df_inst = dfs[1]
            for col in df_inst.columns:
                if "外資" in str(col) and "淨額" in str(col):
                    net_str = str(df_inst[col].iloc[0]).replace(",", "")
                    result['foreign_net'] = int(net_str)
                    break
    except Exception:
        pass
    
    return result

# ==========================================
# 🆕 新功能 5: Dealer Gamma Exposure (簡化版)
# ==========================================
def calculate_dealer_gamma(df, spot_price, risk_free_rate=0.015, days_to_expiry=7):
    """計算造市商的Gamma曝險（簡化版）"""
    if spot_price is None or spot_price <= 0:
        return None
    
    df_calc = df.copy()
    df_calc = df_calc[df_calc['OI'] > 0]
    
    S = spot_price
    K = df_calc['Strike'].values
    T = days_to_expiry / 365
    sigma = 0.15
    r = risk_free_rate
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    df_calc['Delta'] = np.where(
        df_calc['Type'].str.contains('Call|買', case=False),
        delta_call,
        delta_put
    )
    df_calc['Gamma'] = gamma
    
    df_calc['Gamma_Exposure'] = df_calc['OI'] * df_calc['Gamma'] * 50
    
    gamma_profile = df_calc.groupby('Strike').agg({
        'Gamma_Exposure': 'sum',
        'Delta': 'mean'
    }).reset_index()
    
    return gamma_profile

# --- Tornado 圖 (移除OI過濾) ---
def plot_tornado_chart(df_target, title_text, spot_price):
    is_call = df_target["Type"].str.contains("買|Call", case=False, na=False)
    df_call = df_target[is_call][["Strike", "OI", "Amount"]].rename(
        columns={"OI": "Call_OI", "Amount": "Call_Amt"}
    )
    df_put = df_target[~is_call][["Strike", "OI", "Amount"]].rename(
        columns={"OI": "Put_OI", "Amount": "Put_Amt"}
    )
    data = (
        pd.merge(df_call, df_put, on="Strike", how="outer")
        .fillna(0)
        .sort_values("Strike")
    )

    total_put_money = data["Put_Amt"].sum()
    total_call_money = data["Call_Amt"].sum()
    
    FOCUS_RANGE = 1200
    if spot_price and spot_price > 0:
        center_price = spot_price
    elif not data.empty:
        center_price = data.loc[data["Put_OI"].idxmax(), "Strike"]
    else:
        center_price = 0

    if center_price > 0:
        min_s = center_price - FOCUS_RANGE
        max_s = center_price + FOCUS_RANGE
        data = data[(data["Strike"] >= min_s) & (data["Strike"] <= max_s)]

    max_oi = max(data["Put_OI"].max(), data["Call_OI"].max()) if not data.empty else 1000
    x_limit = max_oi * 1.1

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=data["Strike"], x=-data["Put_OI"], orientation="h", 
        name="Put (支撐)", marker_color="#2ca02c", opacity=0.85,
        customdata=data["Put_Amt"] / 100000000,
        hovertemplate="<b>履約價: %{y}</b><br>Put OI: %{x} 口<br>Put 市值: %{customdata:.2f}億<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        y=data["Strike"], x=data["Call_OI"], orientation="h",
        name="Call (壓力)", marker_color="#d62728", opacity=0.85,
        customdata=data["Call_Amt"] / 100000000,
        hovertemplate="<b>履約價: %{y}</b><br>Call OI: %{x} 口<br>Call 市值: %{customdata:.2f}億<extra></extra>"
    ))

    annotations = []
    if spot_price and spot_price > 0 and not data.empty:
        if data["Strike"].min() <= spot_price <= data["Strike"].max():
            fig.add_hline(y=spot_price, line_dash="dash", line_color="#ff7f0e", line_width=2)
            annotations.append(dict(
                x=1, y=spot_price, xref="paper", yref="y",
                text=f" 現貨 {int(spot_price)} ", showarrow=False,
                xanchor="left", align="center",
                font=dict(color="white", size=12),
                bgcolor="#ff7f0e", bordercolor="#ff7f0e", borderpad=4
            ))

    annotations.append(dict(
        x=0.02, y=1.05, xref="paper", yref="paper",
        text=f"<b>Put 總金額</b><br>{total_put_money/100000000:.1f} 億",
        showarrow=False, align="left",
        font=dict(size=14, color="#2ca02c"),
        bgcolor="white", bordercolor="#2ca02c", borderwidth=2, borderpad=6
    ))
    annotations.append(dict(
        x=0.98, y=1.05, xref="paper", yref="paper",
        text=f"<b>Call 總金額</b><br>{total_call_money/100000000:.1f} 億",
        showarrow=False, align="right",
        font=dict(size=14, color="#d62728"),
        bgcolor="white", bordercolor="#d62728", borderwidth=2, borderpad=6
    ))

    fig.update_layout(
        title=dict(text=title_text, y=0.95, x=0.5, xanchor="center", yanchor="top", font=dict(size=20, color="black")),
        xaxis=dict(
            title="未平倉量 (OI)", range=[-x_limit, x_limit],
            showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor="black",
            tickmode="array",
            tickvals=[-x_limit*0.75, -x_limit*0.5, -x_limit*0.25, 0, x_limit*0.25, x_limit*0.5, x_limit*0.75],
            ticktext=[f"{int(x_limit*0.75)}", f"{int(x_limit*0.5)}", f"{int(x_limit*0.25)}", 
                     "0", f"{int(x_limit*0.25)}", f"{int(x_limit*0.5)}", f"{int(x_limit*0.75)}"]
        ),
        yaxis=dict(title="履約價", tickmode="linear", dtick=100, tickformat="d"),
        barmode="overlay",
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
        height=750,
        margin=dict(l=40, r=80, t=140, b=60),
        annotations=annotations,
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    return fig

# --- AI 分析函式 (增強版：包含五大數據) ---
def ask_gemini_brief(df_recent, taiex_price, contract_code, settlement_date, 
                     oi_changes=None, iv_metrics=None, futures_data=None, gamma_profile=None):
    if not gemini_model:
        return f"⚠️ {gemini_model_name}"
    try:
        df_ai = df_recent.copy()
        if "Amount" in df_ai.columns:
            df_ai = df_ai.nlargest(80, "Amount")
        data_str = df_ai.to_csv(index=False)
        
        # 🆕 組裝五大數據
        extra_info = "\n【進階數據分析】\n"
        
        # 1. OI 增減變化
        if oi_changes is not None and not oi_changes.empty:
            top_inc = oi_changes.nlargest(5, 'Change_1D')[['Strike', 'Type', 'Change_1D']]
            top_dec = oi_changes.nsmallest(5, 'Change_1D')[['Strike', 'Type', 'Change_1D']]
            extra_info += f"\n📈 近日OI大增前5名:\n{top_inc.to_string(index=False)}\n"
            extra_info += f"\n📉 近日OI大減前5名:\n{top_dec.to_string(index=False)}\n"
        
        # 2. IV & Skew
        if iv_metrics:
            extra_info += f"\n📊 隱含波動率指標:\n"
            extra_info += f"- ATM履約價: {iv_metrics['ATM_Strike']:.0f}\n"
            extra_info += f"- ATM IV: {iv_metrics['ATM_IV']:.2f}\n"
            extra_info += f"- 25Δ Call IV: {iv_metrics['Call_25D_IV']:.2f}\n"
            extra_info += f"- 25Δ Put IV: {iv_metrics['Put_25D_IV']:.2f}\n"
            extra_info += f"- Skew (RR): {iv_metrics['Skew_25D']:.2f} (正=看漲/負=避險)\n"
        
        # 3. 外資部位 & 基差
        if futures_data:
            extra_info += f"\n🏦 三大法人與基差:\n"
            if futures_data.get('foreign_net'):
                extra_info += f"- 外資期貨淨部位: {futures_data['foreign_net']:,} 口\n"
            if futures_data.get('futures_price') and taiex_price:
                basis = futures_data['futures_price'] - taiex_price
                extra_info += f"- 期貨價格: {futures_data['futures_price']:.2f}\n"
                extra_info += f"- 現期基差: {basis:.2f} (正=多頭溢價/負=空頭貼水)\n"
        
        # 4. Gamma 曝險
        if gamma_profile is not None and not gamma_profile.empty:
            max_gamma_strike = gamma_profile.loc[gamma_profile['Gamma_Exposure'].idxmax(), 'Strike']
            max_gamma_value = gamma_profile['Gamma_Exposure'].max()
            extra_info += f"\n⚡ 造市商Gamma曝險:\n"
            extra_info += f"- 最大Gamma點位: {max_gamma_strike:.0f} (造市商避險壓力最大)\n"
            extra_info += f"- Gamma曝險值: {max_gamma_value:.0f}\n"
        
        prompt = f"""
你現在是台指選擇權市場的【主力莊家】。你的目標只有一個:**在結算日吃掉最多散戶的權利金,讓自己的利潤最大化**。

【市場現況】
- 結算合約: {contract_code} (結算日: {settlement_date})
- 現貨指數(即時): {taiex_price}

【任務】
請根據以下**完整數據**進行深度控盤推演:

{extra_info}

【基礎OI籌碼數據】
{data_str}

【分析要求】
1. **肥羊與雷區分析**: 
   - 結合OI增減、IV Skew、Gamma點位,找出散戶重倉區
   - 判斷你的防守底線(不能讓指數突破的價位)
   
2. **操盤劇本 (Script)**: 
   - 利用外資部位、基差、Gamma釘盤效應
   - 寫出未來2-3天的畫線劇本
   
3. **最佳結算目標**: 
   - 綜合所有數據,給出讓Call/Put雙殺的完美點位
   
4. **莊家指令**: 
   - 簡短有力的操作指令(如: Sell Call @ XX, Defend XX支撐)

【回答格式】
- 使用第一人稱(本莊、我)
- 語氣:**自信、冷血、貪婪**
- **不要**風險警語或教育廢話
- 字數: 400-600字,要有具體數字和邏輯推演
"""
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"分析忙碌中 ({str(e)})"

def ask_openai_brief(df_recent, taiex_price, contract_code, settlement_date,
                     oi_changes=None, iv_metrics=None, futures_data=None, gamma_profile=None):
    if not openai_client:
        return f"⚠️ {openai_model_name}"
    try:
        df_ai = df_recent.copy()
        if "Amount" in df_ai.columns:
            df_ai = df_ai.nlargest(80, "Amount")
        data_str = df_ai.to_csv(index=False)
        
        # 🆕 組裝五大數據
        extra_info = "\n【進階數據分析】\n"
        
        if oi_changes is not None and not oi_changes.empty:
            top_inc = oi_changes.nlargest(5, 'Change_1D')[['Strike', 'Type', 'Change_1D']]
            top_dec = oi_changes.nsmallest(5, 'Change_1D')[['Strike', 'Type', 'Change_1D']]
            extra_info += f"\n📈 近日OI大增前5名:\n{top_inc.to_string(index=False)}\n"
            extra_info += f"\n📉 近日OI大減前5名:\n{top_dec.to_string(index=False)}\n"
        
        if iv_metrics:
            extra_info += f"\n📊 隱含波動率指標:\n"
            extra_info += f"- ATM履約價: {iv_metrics['ATM_Strike']:.0f}\n"
            extra_info += f"- ATM IV: {iv_metrics['ATM_IV']:.2f}\n"
            extra_info += f"- 25Δ Call IV: {iv_metrics['Call_25D_IV']:.2f}\n"
            extra_info += f"- 25Δ Put IV: {iv_metrics['Put_25D_IV']:.2f}\n"
            extra_info += f"- Skew (RR): {iv_metrics['Skew_25D']:.2f} (正=看漲/負=避險)\n"
        
        if futures_data:
            extra_info += f"\n🏦 三大法人與基差:\n"
            if futures_data.get('foreign_net'):
                extra_info += f"- 外資期貨淨部位: {futures_data['foreign_net']:,} 口\n"
            if futures_data.get('futures_price') and taiex_price:
                basis = futures_data['futures_price'] - taiex_price
                extra_info += f"- 期貨價格: {futures_data['futures_price']:.2f}\n"
                extra_info += f"- 現期基差: {basis:.2f} (正=多頭溢價/負=空頭貼水)\n"
        
        if gamma_profile is not None and not gamma_profile.empty:
            max_gamma_strike = gamma_profile.loc[gamma_profile['Gamma_Exposure'].idxmax(), 'Strike']
            max_gamma_value = gamma_profile['Gamma_Exposure'].max()
            extra_info += f"\n⚡ 造市商Gamma曝險:\n"
            extra_info += f"- 最大Gamma點位: {max_gamma_strike:.0f} (造市商避險壓力最大)\n"
            extra_info += f"- Gamma曝險值: {max_gamma_value:.0f}\n"
        
        user_prompt = f"""
你現在是台指選擇權市場的【主力莊家】。你的目標只有一個:**在結算日吃掉最多散戶的權利金,讓自己的利潤最大化**。

【市場現況】
- 結算合約: {contract_code} (結算日: {settlement_date})
- 現貨指數(即時): {taiex_price}

【任務】
請根據以下**完整數據**進行深度控盤推演:

{extra_info}

【基礎OI籌碼數據】
{data_str}

【分析要求】
1. **肥羊與雷區分析**: 
   - 結合OI增減、IV Skew、Gamma點位,找出散戶重倉區
   - 判斷你的防守底線(不能讓指數突破的價位)
   
2. **操盤劇本 (Script)**: 
   - 利用外資部位、基差、Gamma釘盤效應
   - 寫出未來2-3天的畫線劇本
   
3. **最佳結算目標**: 
   - 綜合所有數據,給出讓Call/Put雙殺的完美點位
   
4. **莊家指令**: 
   - 簡短有力的操作指令(如: Sell Call @ XX, Defend XX支撐)

【回答格式】
- 使用第一人稱(本莊、我)
- 語氣:**自信、冷血、貪婪**
- **不要**風險警語或教育廢話
- 字數: 400-600字,要有具體數字和邏輯推演
"""
        response = openai_client.chat.completions.create(
            model=openai_model_name,
            messages=[
                {"role": "system", "content": "你是一個冷血、唯利是圖的期貨市場莊家。你不關心散戶死活,只在乎如何讓選擇權歸零。你會深度分析OI增減、IV Skew、外資部位、Gamma曝險等數據,找出最有利的操盤策略。"},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"分析忙碌中 ({str(e)})"

# ==========================================
# 🎯 主程式
# ==========================================
def main():
    st.title("🤖 台指期權碼戰情室 (莊家絕殺版 + 五大增強)")

    if st.sidebar.button("🔄 重新整理", key="refresh_main"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("連線期交所中..."):
        data_list, data_date = get_option_data_full(days_back=3)
        auto_taiex = get_realtime_data()
        futures_data = get_futures_and_institutional()

    if data_list is None:
        st.error("查無資料,請稍後再試。")
        return

    df = data_list[0]

    with st.expander("🛠️ 數據校正設定 (若現貨/期貨價格延遲,請點此展開輸入)", expanded=False):
        mc1, mc2 = st.columns([1, 2])
        with mc1:
            st.info(f"系統自動抓取: {auto_taiex}")
        with mc2:
            manual_price_input = st.number_input(
                "請輸入看盤軟體最新價格 (輸入 0 代表使用系統自動數據):",
                min_value=0.0, value=0.0, step=1.0, format="%.2f"
            )
    
    if manual_price_input > 0:
        final_taiex = manual_price_input
        price_source_msg = "⚠️ 手動校正"
    else:
        final_taiex = auto_taiex if auto_taiex else 0
        price_source_msg = "系統自動"

    total_call_amt = df[df["Type"].str.contains("買|Call", case=False, na=False)]["Amount"].sum()
    total_put_amt = df[df["Type"].str.contains("賣|Put", case=False, na=False)]["Amount"].sum()
    pc_ratio_amt = ((total_put_amt / total_call_amt) * 100 if total_call_amt > 0 else 0)

    c1, c2, c3, c4, c5 = st.columns([1, 0.8, 1, 1, 1])
    c1.markdown(f"<div style='text-align: left;'><span style='font-size: 14px; color: #555;'>製圖時間</span><br><span style='font-size: 18px; font-weight: bold;'>{datetime.now(tz=TW_TZ).strftime('%Y/%m/%d %H:%M:%S')}</span></div>", unsafe_allow_html=True)
    c2.metric(f"大盤/期貨 ({price_source_msg})", f"{int(final_taiex) if final_taiex else 'N/A'}")
    
    trend = "偏多" if pc_ratio_amt > 100 else "偏空"
    c3.metric("全市場 P/C 金額比", f"{pc_ratio_amt:.1f}%", f"{trend}格局")
    c4.metric("資料來源日期", data_date)
    
    if futures_data['futures_price'] and final_taiex:
        basis = futures_data['futures_price'] - final_taiex
        c5.metric("現期基差", f"{basis:.1f}", f"期貨 {futures_data['futures_price']:.0f}")
    else:
        c5.metric("現期基差", "N/A")

    st.markdown("---")

    with st.expander("📈 近三日 OI 增減分析", expanded=False):
        if len(data_list) >= 2:
            oi_changes = calculate_oi_changes(data_list)
            if oi_changes is not None and 'Change_1D' in oi_changes.columns:
                top_increase = oi_changes.nlargest(10, 'Change_1D')[['Month', 'Strike', 'Type', 'Change_1D']]
                top_decrease = oi_changes.nsmallest(10, 'Change_1D')[['Month', 'Strike', 'Type', 'Change_1D']]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🔥 OI 增加 TOP 10")
                    st.dataframe(top_increase, use_container_width=True)
                with col2:
                    st.subheader("❄️ OI 減少 TOP 10")
                    st.dataframe(top_decrease, use_container_width=True)
        else:
            st.warning("歷史資料不足,無法計算OI變化")

    with st.expander("📊 隱含波動率 (IV) & Skew 分析", expanded=False):
        iv_metrics = calculate_iv_and_skew(df, final_taiex)
        if iv_metrics:
            ivc1, ivc2, ivc3, ivc4 = st.columns(4)
            ivc1.metric("ATM 履約價", f"{iv_metrics['ATM_Strike']:.0f}")
            ivc2.metric("ATM IV", f"{iv_metrics['ATM_IV']:.2f}")
            ivc3.metric("25Δ Call IV", f"{iv_metrics['Call_25D_IV']:.2f}")
            ivc4.metric("25Δ RR Skew", f"{iv_metrics['Skew_25D']:.2f}")
            
            st.info("💡 **Skew 解讀**: 正值代表 Call 較貴(看漲情緒),負值代表 Put 較貴(避險需求)")
        else:
            st.warning("無法計算 IV,請確認現貨價格正確")

    with st.expander("🏦 三大法人部位 & 現期基差", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        
        if futures_data['foreign_net'] is not None:
            fc1.metric("外資期貨淨部位", f"{futures_data['foreign_net']:,} 口")
        else:
            fc1.metric("外資期貨淨部位", "N/A")
        
        if futures_data['futures_price']:
            fc2.metric("期貨價格", f"{futures_data['futures_price']:.2f}")
        else:
            fc2.metric("期貨價格", "N/A")
        
        if futures_data['futures_price'] and final_taiex:
            basis = futures_data['futures_price'] - final_taiex
            basis_pct = (basis / final_taiex) * 100
            fc3.metric("基差", f"{basis:.2f}", f"{basis_pct:.2f}%")
        else:
            fc3.metric("基差", "N/A")
        
        st.info("💡 **基差解讀**: 正值代表期貨溢價(多頭),負值代表期貨貼水(空頭)")

    with st.expander("⚡ 造市商 Gamma 曝險分析", expanded=False):
        gamma_profile = calculate_dealer_gamma(df, final_taiex)
        if gamma_profile is not None and not gamma_profile.empty:
            fig_gamma = go.Figure()
            fig_gamma.add_trace(go.Bar(
                x=gamma_profile['Strike'],
                y=gamma_profile['Gamma_Exposure'],
                marker_color='purple',
                name='Gamma Exposure'
            ))
            fig_gamma.update_layout(
                title="造市商 Gamma 曝險分布",
                xaxis_title="履約價",
                yaxis_title="Gamma Exposure",
                height=400
            )
            st.plotly_chart(fig_gamma, use_container_width=True)
            
            max_gamma_strike = gamma_profile.loc[gamma_profile['Gamma_Exposure'].idxmax(), 'Strike']
            st.success(f"🎯 **最大 Gamma 點位**: {max_gamma_strike:.0f} (造市商需大量避險的價位)")
        else:
            st.warning("無法計算 Gamma,請確認現貨價格正確")

    st.markdown("---")

    st.markdown("### 💡 雙 AI 莊家控盤室")

    unique_codes = df["Month"].unique()
    all_contracts = []
    for code in unique_codes:
        s_date_str = get_settlement_date(code)
        if s_date_str == "9999/99/99" or s_date_str <= data_date:
            continue
        all_contracts.append({"code": code, "date": s_date_str})
    
    all_contracts.sort(key=lambda x: x["date"])

    if all_contracts:
        nearest = all_contracts[0]
        nearest_code = nearest["code"]
        nearest_date = nearest["date"]
        nearest_df = df[df["Month"] == nearest_code]
        st.caption(f"本次絕殺目標合約:**{nearest_code}**,結算日 **{nearest_date}**。")
        target_df_for_ai = nearest_df
        target_code = nearest_code
        target_date = nearest_date
    else:
        st.caption("⚠ 找不到合約,使用全市場資料。")
        target_df_for_ai = df
        target_code = "全市場"
        target_date = data_date

    if st.button("🚀 啟動莊家思維推演", type="primary"):
        ai_col1, ai_col2 = st.columns(2)

        # 🆕 準備完整數據給 AI
        oi_changes_data = None
        if len(data_list) >= 2:
            oi_changes_data = calculate_oi_changes(data_list)
        
        iv_metrics_data = calculate_iv_and_skew(df, final_taiex)
        gamma_profile_data = calculate_dealer_gamma(df, final_taiex)

        with ai_col1:
            st.markdown(f"#### 💎 Gemini 莊家 ({gemini_model_name})")
            with st.spinner("Gemini 正在計算最大痛點..."):
                gemini_advice = ask_gemini_brief(
                    target_df_for_ai, final_taiex, target_code, target_date,
                    oi_changes=oi_changes_data,
                    iv_metrics=iv_metrics_data,
                    futures_data=futures_data,
                    gamma_profile=gamma_profile_data
                )
            st.info(gemini_advice)

        with ai_col2:
            st.markdown(f"#### 💬 ChatGPT 莊家 ({openai_model_name})")
            with st.spinner("ChatGPT 正在擬定絕殺劇本..."):
                openai_advice = ask_openai_brief(
                    target_df_for_ai, final_taiex, target_code, target_date,
                    oi_changes=oi_changes_data,
                    iv_metrics=iv_metrics_data,
                    futures_data=futures_data,
                    gamma_profile=gamma_profile_data
                )
            st.info(openai_advice)

    st.markdown("---")

    if all_contracts:
        plot_targets = []
        nearest = all_contracts[0]
        plot_targets.append({"title": "最近結算", "info": nearest})
        
        monthly = next((c for c in all_contracts if len(c["code"]) == 6), None)
        if monthly and monthly["code"] != nearest["code"]:
            plot_targets.append({"title": "當月月選", "info": monthly})
        
        cols = st.columns(len(plot_targets))
        for i, target in enumerate(plot_targets):
            with cols[i]:
                m_code = target["info"]["code"]
                s_date = target["info"]["date"]
                df_target = df[df["Month"] == m_code]

                sub_call = df_target[df_target["Type"].str.contains("Call|買", case=False, na=False)]["Amount"].sum()
                sub_put = df_target[df_target["Type"].str.contains("Put|賣", case=False, na=False)]["Amount"].sum()
                sub_ratio = (sub_put / sub_call * 100) if sub_call > 0 else 0

                title_text = (
                    f"<b>【{target['title']}】 {m_code}</b>"
                    f"<br><span style='font-size: 14px;'>結算: {s_date}</span>"
                    f"<br><span style='font-size: 14px;'>P/C金額比: {sub_ratio:.1f}% ({'偏多' if sub_ratio > 100 else '偏空'})</span>"
                )
                st.plotly_chart(plot_tornado_chart(df_target, title_text, final_taiex), use_container_width=True)
    else:
        st.info("目前無可識別的未來結算合約。")

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.sidebar.download_button(
        "📥 下載完整數據",
        csv,
        f"option_{data_date.replace('/','')}.csv",
        "text/csv",
    )

if __name__ == "__main__":
    main()
