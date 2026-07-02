import streamlit as st

# ==========================================
# 0. 必須為 Streamlit 第一行指令：強制全螢幕寬版配置 (解決版面集中問題)
# ==========================================
st.set_page_config(
    page_title="德基水庫庫容推估系統",
    page_icon="🔌",
    layout="wide"
)

import pandas as pd
import numpy as np
import datetime
import calendar
import io
import plotly.graph_objects as go

# ==========================================
# 0. 內建 德基水庫 高程-庫容 基準對應表 (由高至低)
# ==========================================
ELEV_CAP_DATA = [
    (1408.0, 18155.0),
    (1407.5, 17946.0),
    (1407.0, 17738.0),
    (1406.5, 17531.0),
    (1406.0, 17324.0),
    (1405.5, 17119.0),
    (1405.0, 16916.0),
    (1404.5, 16714.0),
    (1404.0, 16515.0),
    (1403.5, 16317.0),
    (1403.0, 16121.0),
    (1402.5, 15928.0),
    (1402.0, 15740.0),
    (1401.5, 15559.0),
    (1401.0, 15382.0),
    (1400.5, 15207.0),
    (1400.0, 15035.0)
]

def get_capacity_from_elevation(elev: float) -> float:
    """
    非線性雙向查表：給定高程 (EL m) 回推蓄水量 (萬噸)。
    """
    elevs = [x[0] for x in ELEV_CAP_DATA][::-1]
    caps = [x[1] for x in ELEV_CAP_DATA][::-1]
    clipped_elev = max(1400.0, min(1408.0, elev))
    return float(np.interp(clipped_elev, elevs, caps))

def get_elevation_from_capacity(cap: float) -> float:
    """
    非線性雙向查表：給定蓄水量 (萬噸) 回推高程 (EL m)。
    """
    elevs = [x[0] for x in ELEV_CAP_DATA][::-1]
    caps = [x[1] for x in ELEV_CAP_DATA][::-1]
    clipped_cap = max(15035.0, min(18155.0, cap))
    return float(np.interp(clipped_cap, caps, elevs))

# ==========================================
# 0. 內建第一層 德基水庫 標準水文流量資料庫 (Embedded Database)
# ==========================================
RAW_DEFAULT_HYDROLOGY = """工作表\tQ95\tQ90\tQ85\tQ80\tQ75\tQ70\tQ65\tQ60\tQ55\tQ50\tQ45\tQ40\tQ35\tQ30\tQ25\tQ20\tQ15\tQ10\tQ5
1月上旬\t4.39\t5.28\t5.98\t6.94\t7.54\t8.34\t8.69\t9.18\t9.39\t9.5\t9.74\t10.53\t10.63\t10.95\t11.64\t12.84\t15.93\t17.69\t40.69
1月中旬\t4.65\t5.22\t5.49\t6.19\t6.49\t6.73\t7.68\t8.33\t8.99\t9.08\t10.04\t10.11\t10.21\t10.7\t12.7\t14.85\t16.38\t23.52\t36.42
1月下旬\t4.08\t5.14\t5.54\t6.35\t6.77\t6.91\t7.15\t7.53\t7.59\t8.1\t8.52\t9.39\t9.65\t11.03\t13.74\t15.79\t18.24\t22.19\t57.75
2月上旬\t3.62\t4.19\t4.28\t4.69\t4.94\t6.26\t6.77\t6.95\t7.51\t7.93\t9.03\t11.35\t12.49\t15.49\t17.46\t20.92\t26.79\t31.77\t50
2月中旬\t3.6\t4.18\t4.48\t4.85\t5\t6.1\t6.75\t7.01\t7.41\t8.94\t9.73\t11.92\t16.22\t18.12\t22.69\t25.22\t28.62\t31.17\t36.09
2月下旬\t2.99\t3.6\t3.91\t4.23\t4.37\t5.67\t6.27\t7.55\t9.59\t10.87\t11.36\t12.14\t12.56\t14.28\t22.86\t28.99\t41.64\t46.78\t88.92
3月上旬\t3.97\t4.52\t5\t5.33\t6.59\t7.77\t8.69\t9.32\t10.5\t11.36\t15.17\t16.79\t19.8\t22.45\t26.2\t28.04\t34.14\t35.74\t104.13
3月中旬\t3.38\t4.73\t6.1\t6.58\t7.35\t8.23\t8.44\t9.61\t11.19\t11.54\t13.39\t16.3\t19.32\t20.23\t26.45\t26.95\t28.25\t49.32\t92.1
3月下旬\t5.16\t5.31\t5.83\t6.73\t10.07\t11.16\t11.49\t12.24\t13.29\t13.75\t15.35\t16.06\t16.37\t17.33\t18.38\t26.08\t37.58\t50.06\t84.28
4月上旬\t4.21\t7.19\t8.42\t9.83\t11.3\t11.94\t13.73\t15.14\t15.67\t16.82\t19.13\t22.67\t25.49\t28.06\t29.27\t32.6\t37.26\t48.09\t58.33
4月中旬\t3.57\t7.51\t8.93\t10.28\t13.03\t13.61\t14.14\t15.05\t16.18\t16.72\t19.17\t20.54\t24.1\t27.23\t31.36\t36.59\t42.56\t56.52\t66.94
4月下旬\t4.43\t7.19\t8.74\t9.09\t10.07\t11.41\t13.16\t15.83\t17.73\t19.99\t22.92\t24.19\t26.52\t27.3\t36.08\t49.09\t49.64\t49.95\t52.51
5月上旬\t4.04\t7.45\t10.11\t11.06\t11.72\t13.34\t13.83\t17.62\t21.86\t22.87\t24.27\t24.78\t26.32\t29.29\t32.22\t47.52\t51.42\t56.06\t63.96
5月中旬\t3.76\t8.33\t10.82\t11.23\t11.37\t11.56\t12.89\t14.34\t15.68\t21.66\t27.69\t33.2\t49.18\t57.44\t63.24\t76.1\t84.82\t106.39\t145.76
5月下旬\t8.06\t9.26\t10.61\t16.38\t20.66\t21.24\t23.02\t28.67\t33.81\t34.89\t42.4\t43.97\t46.35\t50.45\t62.55\t71.25\t76.8\t86.65\t110.09
6月上旬\t9.5\t17.66\t18.88\t20.35\t23.04\t26.16\t26.45\t27.88\t31.51\t34.49\t35.23\t37.23\t40.78\t42.6\t47.52\t51.86\t73.46\t161.39\t276.82
6月中旬\t13.81\t15.29\t16.9\t18.54\t18.85\t20.29\t24.69\t30.84\t41.08\t44.52\t48.45\t49.63\t51.77\t55.77\t84.07\t120.62\t168.97\t226.69\t279.53
6月下旬\t12.17\t13.39\t14.54\t16.48\t18.04\t18.96\t19.24\t19.47\t21.09\t25.47\t28.95\t32.18\t34.12\t40.44\t44.18\t52.67\t57.88\t73.07\t95.51
7月上旬\t10.01\t11.58\t12.93\t13.87\t15\t15.41\t15.52\t18.86\t21.53\t21.86\t23.98\t26.96\t28.96\t29.83\t31.7\t33.32\t36.78\t42.32\t175.03
7月中旬\t8.09\t10.99\t12.42\t13.21\t14.28\t16.58\t17.07\t17.59\t18.03\t21.18\t22.1\t24.23\t26.64\t32.72\t44.31\t47.16\t70.06\t97.16\t225.26
7月下旬\t7.54\t11.21\t13.21\t13.73\t13.81\t15.27\t16.34\t18.5\t19.63\t23.44\t29.02\t30.4\t43.54\t54.53\t56.83\t82.84\t129.62\t191.02\t229.05
8月上旬\t7.97\t9.55\t11.23\t13.48\t15.9\t20.64\t22.93\t27.31\t34.42\t40.02\t41.44\t43.79\t50.89\t59.22\t62.14\t77.83\t110.17\t210.85\t267.05
8月中旬\t6.66\t9.28\t10.75\t12.81\t15.99\t18.25\t20.14\t23.92\t28.37\t30.29\t31.61\t33.68\t35.09\t47.16\t52.93\t62.52\t105.49\t156.65\t233.24
8月下旬\t6.08\t8.04\t11.1\t14.15\t14.9\t18.72\t20.25\t21\t23\t24.81\t24.9\t27.14\t34.95\t48.7\t58.08\t63.88\t81.63\t102.93\t217.13
9月上旬\t6.45\t13.12\t13.9\t15.01\t15.46\t17.99\t18.99\t20.24\t22\t22.76\t27\t32.05\t36.68\t39.39\t41.55\t56.76\t58.21\t102.2\t254.35
9月中旬\t6.91\t12.16\t14.49\t15.53\t16.7\t18.73\t19.57\t22.22\t24.04\t31.73\t37.12\t42.07\t44.99\t45.99\t47.44\t50.33\t65.82\t124.93\t364.98
9月下旬\t8.12\t11\t12.99\t14.45\t15.08\t16.24\t18.31\t22.8\t24.35\t24.51\t24.76\t27.96\t38.3\t48\t73.86\t106.55\t186.06\t274.34\t289.93
10月上旬\t7.39\t9.55\t11.37\t12.97\t15.49\t16.82\t18.02\t19.57\t20.46\t25.54\t29.39\t29.86\t31.29\t36.6\t44.14\t68.58\t122.13\t170.08\t212.61
10月中旬\t6.58\t11.26\t12.12\t13.47\t14.73\t14.88\t15.15\t16.79\t18.19\t20.4\t22.23\t24.06\t26.09\t30.48\t42.04\t48.43\t62.84\t78.43\t89.3
10月下旬\t6.87\t9.13\t9.89\t10.61\t11.21\t11.68\t13.54\t14.82\t15.58\t18.35\t20.64\t23.03\t23.91\t24.4\t26.56\t26.99\t29.78\t34.66\t43.65
11月上旬\t5.07\t7.93\t8.95\t9.33\t9.89\t10.46\t11.93\t13.23\t13.93\t14.41\t14.68\t14.92\t15.65\t16.97\t17.31\t18.33\t20.13\t23.78\t113.28
11月中旬\t4.78\t7.26\t7.8\t9.04\t9.5\t10.08\t11.25\t12.36\t12.6\t13.93\t13.97\t14.19\t14.86\t16.01\t17.98\t18.67\t22.22\t28.78\t50.98
11月下旬\t4.36\t7.05\t7.24\t7.42\t8.03\t8.56\t10.11\t10.32\t11.68\t12.99\t13.1\t13.69\t15.48\t16.34\t17.32\t19.47\t22.03\t25.23\t28.22
12月上旬\t2.07\t5.06\t5.87\t7.09\t8.12\t8.38\t9.04\t9.83\t10.47\t10.94\t11.74\t11.85\t12.24\t12.51\t14.47\t15.17\t18.15\t38.44\t50.56
12月中旬\t2.84\t5.25\t5.82\t6.12\t6.87\t7.13\t8.31\t9.51\t9.69\t10.08\t10.12\t10.25\t11.04\t12.04\t12.36\t15.55\t17.91\t20.5\t33.99
12月下旬\t2.41\t5.35\t5.9\t7.12\t7.87\t8.44\t8.65\t9.13\t9.34\t9.82\t9.9\t9.97\t10.93\t11.97\t13.23\t14.58\t15.42\t16.34\t23.44"""

# ==========================================
# 0. 內建第一層標準出流需求資料庫（前一年度常態需求）
# ==========================================
RAW_DEFAULT_DEMANDS = """工作表\t葫蘆墩圳需求_cms\t下游五圳需求_cms\t公共給水_萬噸
1月上旬\t4.5\t2.5\t90.0
1月中旬\t4.5\t2.5\t90.0
1月下旬\t4.5\t2.5\t90.0
2月上旬\t5.0\t3.0\t90.0
2月中旬\t6.0\t4.0\t90.0
2月下旬\t8.0\t6.0\t90.0
3月上旬\t8.0\t6.0\t90.0
3月中旬\t7.0\t5.0\t90.0
3月下旬\t6.0\t4.0\t90.0
4月上旬\t6.0\t4.0\t90.0
4月中旬\t6.0\t4.0\t90.0
4月下旬\t6.0\t4.0\t90.0
5月上旬\t5.5\t4.0\t90.0
5月中旬\t5.5\t4.0\t90.0
5月下旬\t7.5\t5.5\t90.0
6月上旬\t8.0\t6.0\t90.0
6月中旬\t6.5\t4.5\t90.0
6月下旬\t6.0\t4.0\t90.0
7月上旬\t6.0\t4.0\t90.0
7月中旬\t7.0\t5.0\t90.0
7月下旬\t9.0\t7.0\t90.0
8月上旬\t9.5\t7.0\t90.0
8月中旬\t8.5\t6.0\t90.0
8月下旬\t7.5\t5.5\t90.0
9月上旬\t6.5\t4.5\t90.0
9月中旬\t6.0\t4.0\t90.0
9月下旬\t6.0\t4.0\t90.0
10月上旬\t6.0\t4.0\t90.0
10月中旬\t7.0\t5.0\t90.0
10月下旬\t8.0\t6.0\t90.0
11月上旬\t6.5\t4.5\t90.0
11月中旬\t5.0\t3.0\t90.0
11月下旬\t5.0\t3.0\t90.0
12月上旬\t4.5\t2.5\t90.0
12月中旬\t4.5\t2.5\t90.0
12月下旬\t4.5\t2.5\t90.0"""

# 定義 36 旬標準順序，用於資料校驗
CANONICAL_PERIODS = [
    "1月上旬", "1月中旬", "1月下旬", "2月上旬", "2月中旬", "2月下旬",
    "3月上旬", "3月中旬", "3月下旬", "4月上旬", "4月中旬", "4月下旬",
    "5月上旬", "5月中旬", "5月下旬", "6月上旬", "6月中旬", "6月下旬",
    "7月上旬", "7月中旬", "7月下旬", "8月上旬", "8月中旬", "8月下旬",
    "9月上旬", "9月中旬", "9月下旬", "10月上旬", "10月中旬", "10月下旬",
    "11月上旬", "11月中旬", "11月下旬", "12月上旬", "12月中旬", "12月下旬"
]

def get_dynamic_techi_inflow(month: int, period: str, scenario: str) -> float:
    """
    動態流量檢索引擎：自 st.session_state.hydrology_df 讀取對應旬別與情境之德基水庫入流量 (單位: cms)。
    """
    scenario_code = scenario.split(" ")[0].strip()
    row_key = f"{month}月{period}"
    
    df = st.session_state.hydrology_df
    match_row = df[df["工作表"].str.strip() == row_key]
    
    if not match_row.empty:
        try:
            return float(match_row.iloc[0][scenario_code])
        except (KeyError, ValueError, TypeError):
            pass
            
    return 10.0

def get_dynamic_demands(month: int, period: str) -> dict:
    """
    動態出流需求檢索引擎：自 st.session_state.demand_df 讀取對應旬別之常態前一年度供水需求。
    """
    row_key = f"{month}月{period}"
    df = st.session_state.demand_df
    match_row = df[df["工作表"].str.strip() == row_key]
    
    if not match_row.empty:
        try:
            return {
                "huludun": float(match_row.iloc[0]["葫蘆墩圳需求_cms"]),
                "down_five": float(match_row.iloc[0]["下游五圳需求_cms"]),
                "public": float(match_row.iloc[0]["公共給水_萬噸"])
            }
        except (KeyError, ValueError, TypeError):
            pass
            
    return {"huludun": 6.0, "down_five": 4.0, "public": 90.0}

def parse_pasted_data(paste_str: str) -> list:
    """
    高容錯解析器：解析從 Excel 複製貼上的橫向或縱向數據。
    """
    if not paste_str.strip():
        return []
    raw_tokens = paste_str.replace(",", "").split()
    parsed_values = []
    for token in raw_tokens:
        try:
            parsed_values.append(float(token))
        except ValueError:
            continue
    return parsed_values

def read_csv_with_fallback(file_obj) -> pd.DataFrame:
    """
    強韌多重解碼回退解碼引擎：解決 950 編碼等台灣 Excel 輸出相容問題。
    """
    bytes_data = file_obj.getvalue()
    for enc in ["utf-8-sig", "cp950", "utf-8", "gb18030"]:
        try:
            decoded_text = bytes_data.decode(enc)
            return pd.read_csv(io.StringIO(decoded_text))
        except Exception:
            continue
    return pd.read_csv(file_obj)

def validate_uploaded_hydrology(df_input: pd.DataFrame) -> tuple:
    """
    強韌防呆校驗器：校驗上傳的德基標準水文資料庫。
    """
    df = df_input.copy()
    df.columns = [str(c).strip() for c in df.columns]
    
    first_col = df.columns[0]
    if first_col not in ["工作表", "旬別"]:
        df.rename(columns={first_col: "工作表"}, inplace=True)
        first_col = "工作表"
        
    required_scenarios = [
        "Q95", "Q90", "Q85", "Q80", "Q75", "Q70", "Q65", "Q60", "Q55", 
        "Q50", "Q45", "Q40", "Q35", "Q30", "Q25", "Q20", "Q15", "Q10", "Q5"
    ]
    missing_cols = [c for c in required_scenarios if c not in df.columns]
    if missing_cols:
        return False, f"上傳檔案缺少必要的情境欄位：{', '.join(missing_cols)}"
        
    if len(df) != 36:
        return False, f"水文年度資料筆數錯誤。預期為 36 旬，實際讀得 {len(df)} 筆。"
        
    df[first_col] = df[first_col].str.strip()
    for idx, expected_name in enumerate(CANONICAL_PERIODS):
        actual_name = df.iloc[idx][first_col]
        if actual_name != expected_name:
            return False, f"第 {idx+1} 列的旬別名稱不符。預期為 '{expected_name}'，實際為 '{actual_name}'。"
            
    try:
        for col in required_scenarios:
            df[col] = pd.to_numeric(df[col]).astype(float)
            if (df[col] < 0).any():
                return False, f"欄位 '{col}' 偵測到負值流量，請確認所有流量皆大於等於 0。"
    except Exception:
        return False, "流量資料中含有非數值的文字或非法空白，請重新檢查檔案。"
        
    final_df = df[[first_col] + required_scenarios].copy()
    if first_col != "工作表":
        final_df.rename(columns={first_col: "工作表"}, inplace=True)
        
    return True, final_df

def validate_uploaded_demands(df_input: pd.DataFrame) -> tuple:
    """
    強韌出流檢驗器：校驗上傳的德基水庫出流常態需求資料庫。
    """
    df = df_input.copy()
    df.columns = [str(c).strip() for c in df.columns]
    
    first_col = df.columns[0]
    if first_col not in ["工作表", "旬別"]:
        df.rename(columns={first_col: "工作表"}, inplace=True)
        first_col = "工作表"
        
    required_cols = ["葫蘆墩圳需求_cms", "下游五圳需求_cms", "公共給水_萬噸"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return False, f"上傳檔案缺少必要的出流需求欄位：{', '.join(missing_cols)}"
        
    if len(df) != 36:
        return False, f"需求年度資料筆數錯誤。預期為 36 旬，實際讀得 {len(df)} 筆。"
        
    df[first_col] = df[first_col].str.strip()
    for idx, expected_name in enumerate(CANONICAL_PERIODS):
        actual_name = df.iloc[idx][first_col]
        if actual_name != expected_name:
            return False, f"第 {idx+1} 列的旬別名稱不符。預期為 '{expected_name}'，實際為 '{actual_name}'。"
            
    try:
        for col in required_cols:
            df[col] = pd.to_numeric(df[col]).astype(float)
            if (df[col] < 0).any():
                return False, f"欄位 '{col}' 偵測到負值，請確認所有需求皆大於等於 0。"
    except Exception:
        return False, "出流需求資料中含有非數值的文字或非法空白，請重新檢查檔案。"
        
    final_df = df[[first_col] + required_cols].copy()
    if first_col != "工作表":
        final_df.rename(columns={first_col: "工作表"}, inplace=True)
        
    return True, final_df

# ==========================================
# 1. 核心物理與曆法引擎 (Calendar Engine)
# ==========================================
def generate_date_profile(start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """
    根據起始日與結束日，生成逐日的時間剖面資料表 [start_date, end_date)。
    採用左閉右開，結束日當天不進行日計算。
    """
    if start_date >= end_date:
        raise ValueError("起始日期不可大於或等於結束日期，請重新檢查您的時間區間。")
        
    dates = []
    curr = start_date
    while curr < end_date:
        dates.append(curr)
        curr += datetime.timedelta(days=1)
        
    data = []
    for d in dates:
        year = d.year
        month = d.month
        day = d.day
        
        if day <= 10:
            period = "上旬"
            period_total_days = 10
        elif day <= 20:
            period = "中旬"
            period_total_days = 10
        else:
            period = "下旬"
            _, total_days_in_month = calendar.monthrange(year, month)
            period_total_days = total_days_in_month - 20
            
        data.append({
            "日期": d,
            "年份": year,
            "月份": month,
            "日": day,
            "旬別": period,
            "該旬實際總天數": period_total_days
        })
        
    return pd.DataFrame(data)

def get_period_date_range(year: int, month: int, period: str) -> tuple:
    """
    計算特定年份、月份與旬別的「實際日曆起迄日期」。
    """
    if period == "上旬":
        return datetime.date(year, month, 1), datetime.date(year, month, 10)
    elif period == "中旬":
        return datetime.date(year, month, 11), datetime.date(year, month, 20)
    else:  # 下旬
        _, last_day = calendar.monthrange(year, month)
        return datetime.date(year, month, 21), datetime.date(year, month, last_day)

def is_overlapping(start1: datetime.date, end1: datetime.date, start2: datetime.date, end2: datetime.date) -> bool:
    """
    判斷兩個日期區間是否有重疊。
    """
    return max(start1, start2) <= min(end1, end2)

def get_historical_milestone_dates_v2(disp_start: datetime.date, proj_start: datetime.date) -> list:
    """
    獲取歷史展示區間的「旬末邊界日」列表。
    """
    milestones = set()
    start_bound = disp_start - datetime.timedelta(days=1)
    end_bound = proj_start - datetime.timedelta(days=1)
    
    milestones.add(start_bound)
    milestones.add(end_bound)
    
    curr = disp_start
    while curr < proj_start:
        is_end_of_period = False
        if curr.day in [10, 20]:
            is_end_of_period = True
        else:
            _, last_day = calendar.monthrange(curr.year, curr.month)
            if curr.day == last_day:
                is_end_of_period = True
                
        if is_end_of_period:
            milestones.add(curr)
        curr += datetime.timedelta(days=1)
        
    return sorted(list(milestones))

def interpolate_historical_capacities_v2(disp_start: datetime.date, proj_start: datetime.date, cap_dict: dict, init_capacity: float) -> dict:
    """
    在 [disp_start - 1, proj_start - 1] 區間內對歷史實際蓄水量進行線性插值。
    """
    start_bound = disp_start - datetime.timedelta(days=1)
    end_bound = proj_start - datetime.timedelta(days=1)
    
    full_caps = {}
    for k, v in cap_dict.items():
        try:
            d = datetime.datetime.strptime(k, "%Y-%m-%d").date()
            full_caps[d] = v
        except ValueError:
            continue
    
    # 強制將「推估起始前一日」鎖定為 init_capacity (重要！)
    full_caps[end_bound] = init_capacity
    
    milestones = sorted(list(full_caps.keys()))
    milestones = [m for m in milestones if start_bound <= m <= end_bound]
    
    daily_caps = {}
    for i in range(len(milestones) - 1):
        d1 = milestones[i]
        d2 = milestones[i+1]
        val1 = full_caps.get(d1, 15000.0)
        val2 = full_caps.get(d2, 15000.0)
        
        days_diff = (d2 - d1).days
        for step in range(days_diff + 1):
            curr_d = d1 + datetime.timedelta(days=step)
            if days_diff == 0:
                daily_caps[curr_d] = val1
            else:
                ratio = step / days_diff
                daily_caps[curr_d] = round(val1 + (val2 - val1) * ratio, 2)
    return daily_caps

# ==========================================
# 4. 輔助函數：水利旬末邊界與目標日期對齊
# ==========================================
def get_sim_target_date(milestone_date: datetime.date) -> datetime.date:
    """
    將旬度標題日期（如 7月1日、7月11日、7月21日、8月1日）
    映射至底層質量守恆模擬日誌對應的「期末結算時間點」：
    - X月1日   -> 實際對應前一月的最後一日
    - X月11日  -> 實際對應本月 10 日 24:00 庫容
    - X月21日  -> 實際對應本月 20 日 24:00 庫容
    """
    if milestone_date.day == 1:
        return milestone_date - datetime.timedelta(days=1)
    elif milestone_date.day == 11:
        return milestone_date.replace(day=10)
    elif milestone_date.day == 21:
        return milestone_date.replace(day=20)
    return milestone_date

def plot_reservoir_capacity_trend(df_sim_results: pd.DataFrame, display_start: datetime.date, start_date: datetime.date, end_date: datetime.date, max_capacity: float) -> go.Figure:
    """
    繪製單一情境之德基水庫蓄水量變化趨勢圖（無背景色塊，黑色實際庫容 + 藍色推估庫容）。
    有