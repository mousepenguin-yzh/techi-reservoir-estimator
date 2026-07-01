import streamlit as st

# ==========================================
# 0. 必須為 Streamlit 第一行指令：強制全螢幕寬版配置 (解決版面集中問題)
# ==========================================
st.set_page_config(
    page_title="德基水庫庫容推估與發電聯調模擬系統",
    page_icon="🔌",
    layout="wide"
)

import pandas as pd
import numpy as np
import datetime
import calendar
import io

# ==========================================
# 0. 內建 德基水庫 高程-庫容 基準對應表 (由高至低)
# ==========================================
# 資料來源：水利署德基水庫基準對應表 (EL 1400.0 ~ 1408.0 公尺)
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
    由於 np.interp 要求自變數必須為遞增，因此需將對應關係進行反轉 (::-1)。
    """
    elevs = [x[0] for x in ELEV_CAP_DATA][::-1]
    caps = [x[1] for x in ELEV_CAP_DATA][::-1]
    # 限制高程在 EL 1400 ~ 1408 之間，防止外推異常
    clipped_elev = max(1400.0, min(1408.0, elev))
    return float(np.interp(clipped_elev, elevs, caps))

def get_elevation_from_capacity(cap: float) -> float:
    """
    非線性雙向查表：給定蓄水量 (萬噸) 回推高程 (EL m)。
    """
    elevs = [x[0] for x in ELEV_CAP_DATA][::-1]
    caps = [x[1] for x in ELEV_CAP_DATA][::-1]
    # 限制蓄水量在對應範圍內，防止外推異常
    clipped_cap = max(15035.0, min(18155.0, cap))
    return float(np.interp(clipped_cap, caps, elevs))

# ==========================================
# 1. 核心物理與曆法引擎 (Calendar Engine)
# ==========================================

def generate_date_profile(start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """
    根據起始日與結束日，生成逐日的時間剖面資料表。
    採用【左閉右開區間 [start_date, end_date)】：
    僅生成到 end_date - 1 天，結束日當天不進行日計算。
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
        
        # 判斷旬別
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
    包含：展示前一日(disp_start - 1)、推估前一日(proj_start - 1)，
    以及展示區間內所有的旬末日（10日、20日、月底日）。
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
    在 [disp_start - 1, proj_start - 1] 區間內進行線性插值。
    cap_dict 包含其餘歷史旬末點的數值。
    init_capacity 為 proj_start - 1 當天 24:00 的數值（鎖定為模擬起點）。
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
# 5. Streamlit 初始化與會話狀態 (狀態持久化以繼承預設值)
# ==========================================

# 德基特定參數初始化
if "control_elevation" not in st.session_state:
    st.session_state.control_elevation = 1408.0

# 側流控制參數初始化 (預設：6-9月係數 1.5，其餘 1.0)
if "monthly_lateral_coeffs" not in st.session_state:
    st.session_state.monthly_lateral_coeffs = {
        1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0,
        6: 1.5, 7: 1.5, 8: 1.5, 9: 1.5,
        10: 1.0, 11: 1.0, 12: 1.0
    }

if "lateral_flow_b" not in st.session_state:
    st.session_state.lateral_flow_b = 0.0  # 馬鞍至石岡側流B，預設 0.0 萬噸/日

# 時間區間與初始值初始化
if "display_start_date" not in st.session_state:
    st.session_state.display_start_date = datetime.date(2026, 5, 1)
if "start_date" not in st.session_state:
    st.session_state.start_date = datetime.date(2026, 6, 21)
if "end_date" not in st.session_state:
    st.session_state.end_date = datetime.date(2026, 9, 1)

# 起始蓄水量初始化
if "init_capacity" not in st.session_state:
    st.session_state.init_capacity = 16500.0  # 對應 EL 1404 左右
if "hist_capacity" not in st.session_state:
    st.session_state.hist_capacity = {}

# ==========================================
# 6. 前端 UI 分頁排版
# ==========================================

st.title("🔌 德基水庫庫容推估與發電聯調模擬系統")
st.markdown("大甲溪複式串聯發電聯調與下游控制模擬（第一階段開發版）")

tab_config, tab_inflow, tab_outflow, tab_simulation, tab_products = st.tabs([
    "⚙️ 第一階段：推估需求基礎資料設定", 
    "🌊 第二階段：入流條件與水文維護（規劃中）",
    "🚰 第三階段：出流需求與抗旱調整（規劃中）",
    "🧮 第四階段：庫容推估演算（規劃中）",
    "📊 第五階段：推估成果產品（規劃中）"
])

# -----------------
# TAB 1: 基礎與曆法 (本次開發重點)
# -----------------
with tab_config:
    st.subheader("⚙️ 水庫基本資料與展示區間")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🏛️ 德基水庫水位與聯調參數設定")
        
        # 1. 控高高程手動調整
        elev_input = st.number_input(
            "德基水庫人為控制水位高程 (EL 公尺)", 
            min_value=1400.0, 
            max_value=1408.0, 
            value=st.session_state.control_elevation, 
            step=0.5,
            help="調整控制水位後，下方會自動查表更新對應的控制庫容上限。真正的滿水位高程為 EL 1408."
        )
        st.session_state.control_elevation = elev_input
        
        # 2. 自動非線性查表轉換
        control_capacity = get_capacity_from_elevation(elev_input)
        st.markdown(
            f"<div style='padding: 8px; background-color: #f0f2f6; border-radius: 5px; margin-bottom: 15px;'>"
            f"📈 查表所得控制庫容上限：<b>{control_capacity:,.0f} 萬噸</b> (對應高程 EL {elev_input:.1f} m)"
            f"</div>", 
            unsafe_allow_html=True
        )
        
        # 3. 馬鞍至石岡側流 B 設定
        st.session_state.lateral_flow_b = st.number_input(
            "馬鞍壩至石岡壩間日側流量 (側流B, 萬噸/日)",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.lateral_flow_b,
            step=0.5,
            help="此側流在模擬中會優先扣減最下游石岡壩之總需求。預設為 0.0 萬噸/日。"
        )

        # 4. 德基至馬鞍側流 A 之 12 個月動態係數折疊選單
        with st.expander("🍂 德基至馬鞍壩間側流係數設定 (側流A, 月份動態)", expanded=False):
            st.caption("側流 A 推估公式：今日側流 A = 當日德基入流量 * 當月側流係數。請於下方微調各月份預設值：")
            m_cols1 = st.columns(6)
            m_cols2 = st.columns(6)
            
            # 前六個月
            for m in range(1, 7):
                with m_cols1[m-1]:
                    st.session_state.monthly_lateral_coeffs[m] = st.number_input(
                        f"{m}月",
                        min_value=0.0,
                        max_value=10.0,
                        value=st.session_state.monthly_lateral_coeffs[m],
                        step=0.1,
                        key=f"lat_m_{m}"
                    )
            # 後六個月
            for m in range(7, 13):
                with m_cols2[m-7]:
                    st.session_state.monthly_lateral_coeffs[m] = st.number_input(
                        f"{m}月",
                        min_value=0.0,
                        max_value=10.0,
                        value=st.session_state.monthly_lateral_coeffs[m],
                        step=0.1,
                        key=f"lat_m_{m}"
                    )
                    
    with col2:
        st.markdown("##### 📅 曆法與時序區間設定")
        st.session_state.display_start_date = st.date_input("展示起始日期(若早於推估起始，需在下方填入實際蓄水量)", value=st.session_state.display_start_date)
        st.session_state.start_date = st.date_input("推估起始日期 (庫容推估守恆起點)", value=st.session_state.start_date)
        st.session_state.end_date = st.date_input("預計推估結束日期 (此結束日當天不計入日計算)", value=st.session_state.end_date)
        
        # 檢驗日期先後關係
        if st.session_state.display_start_date > st.session_state.start_date:
            st.error("⚠️ 錯誤：『展示起始日期』不可晚於『推估起始日期』。")
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("⚠️ 錯誤：『推估起始日期』必須早於『預計推估結束日期』。")
            
        calc_start_day = st.session_state.start_date
        prev_day = calc_start_day - datetime.timedelta(days=1)
        prev_day_label = f"推估起點前一日 ({prev_day.strftime('%m/%d')} 24:00) 實際蓄水量 (萬噸)"
        
        # 起始蓄水量上限鎖定在查表算出的控制庫容
        st.session_state.init_capacity = st.number_input(
            prev_day_label, 
            min_value=0.0, 
            max_value=control_capacity, 
            value=min(st.session_state.init_capacity, control_capacity), 
            step=100.0
        )

    # 處理展示期（歷史觀測期）的逐旬實際蓄水量輸入
    if st.session_state.display_start_date < st.session_state.start_date:
        st.markdown("---")
        st.markdown("##### 📈 歷史觀測展示期 旬末實際蓄水量輸入")
        st.caption(f"請輸入展示期間內，各旬末日前一日 24:00 的實際蓄水量 (單位: 萬噸，最高上限受限於當前控制庫容 {control_capacity:,.0f} 萬噸)：")
        
        milestones = get_historical_milestone_dates_v2(st.session_state.display_start_date, st.session_state.start_date)
        
        # 排除最後一個邊界日（最後一日已由 init_capacity 鎖定）
        end_boundary = st.session_state.start_date - datetime.timedelta(days=1)
        other_milestones = [m for m in milestones if m != end_boundary]
        
        if other_milestones:
            cols_num = min(4, len(other_milestones))
            m_cols = st.columns(cols_num)
            
            for idx, m_date in enumerate(other_milestones):
                col_idx = idx % cols_num
                m_label = f"{m_date.strftime('%m/%d')} 24:00 實際蓄水量"
                
                # 初始預設值
                default_v = st.session_state.hist_capacity.get(m_date.strftime('%Y-%m-%d'), st.session_state.init_capacity)
                # 防止超限
                default_v = min(default_v, control_capacity)
                
                st.session_state.hist_capacity[m_date.strftime('%Y-%m-%d')] = m_cols[col_idx].number_input(
                    m_label, 
                    min_value=0.0, 
                    max_value=control_capacity, 
                    value=default_v, 
                    step=100.0, 
                    key=f"active_hist_{m_date}"
                )

    # 生成總時間剖面與防呆狀態顯示
    if st.session_state.display_start_date < st.session_state.end_date and st.session_state.start_date < st.session_state.end_date:
        df_cal = generate_date_profile(st.session_state.display_start_date, st.session_state.end_date)
        
        # 預先計算後續頁籤會用到的全區間唯一旬度 (與鯉魚潭保持一致)
        unique_periods = df_cal.groupby(["年份", "月份", "旬別"]).size().reset_index().drop(columns=[0])
        period_order = {"上旬": 1, "中旬": 2, "下旬": 3}
        unique_periods["旬別順序碼"] = unique_periods["旬別"].map(period_order)
        unique_periods = unique_periods.sort_values(by=["年份", "月份", "旬別順序碼"]).drop(columns=["旬別順序碼"]).reset_index(drop=True)
        
        # 修正為與鯉魚潭完全一致的單行 Markdown 加粗成功字卡
        st.success(f"📅 曆法配置成功：當前展示+推估計算區間（left-closed right-open，即左閉右開）共計 **{len(df_cal)}** 天。")
    else:
        unique_periods = pd.DataFrame()
        st.error("❌ 日期區間衝突，請先修正上方日期。")

# -----------------
# TAB 2: 第二階段 placeholder
# -----------------
with tab_inflow:
    st.subheader("🌊 第二階段：入流條件與水文維護")
    st.info("💡 本頁籤於第二階段開發：將在此建置 36 旬 19 情境之大甲溪歷史天然入流量資料庫，並支援手動 Excel 複製貼上與混合輸入模式。")

# -----------------
# TAB 3: 第三階段 placeholder
# -----------------
with tab_outflow:
    st.subheader("🚰 第三階段：出流需求與抗旱調整")
    st.info("💡 本頁籤於第三階段開發：將在此建置前一年度下游灌區需求（含葫蘆墩圳、下游五圳）及公共給水自訂（90萬噸/日）之覆寫控制律與抗旱日誌。")

# -----------------
# TAB 4: 第四階段 placeholder
# -----------------
with tab_simulation:
    st.subheader("🧮 第四階段：庫容推估演算")
    st.info("💡 本頁籤於第四階段開發：整合入流量與大甲溪『德基-馬鞍-石岡』聯調控制引擎，逐日執行質量守恆計算與水位內插。")

# -----------------
# TAB 5: 第五階段 placeholder
# -----------------
with tab_products:
    st.subheader("📊 第五階段：推估成果產品")
    st.info("💡 本頁籤於第五階段開發：產出雙時間軸 Plotly 歷線圖、多情境比較、以及轉置成橫向對齊的 Excel 旬推估表下載。")
