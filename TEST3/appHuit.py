import streamlit as st
import pandas as pd
import altair as alt
from datetime import date, timedelta

# ----------------------------------
# 1. IMPORT BACKEND
# ----------------------------------
try:
    from data_service import (
        get_state_times,
        get_machine_alarms,
        get_energy_consumption,
        # get_daily_idle_trend (Removed as requested)
    )
except ImportError:
    st.error("Module 'data_service' missing. Please check your files.")
    st.stop()

# ----------------------------------
# 2. CONFIG & CSS
# ----------------------------------
st.set_page_config(page_title="CNC Dashboard", layout="wide", page_icon="🎛️")

st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] {font-size: 0.9rem; color: #666;}
    div[data-testid="stMetricValue"] {font-size: 1.5rem; font-weight: 600; color: #333;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# 3. GLOBAL CONSTANTS (4 Core States ONLY)
# ----------------------------------
STATE_DOMAIN = ['High Activity', 'Intermediate Activity', 'Low Activity', 'True Idle (Off)']
STATE_RANGE = ['#084594', '#4292c6', '#9ecae1', '#9e3426'] 

EXCLUDED_FROM_GRAPHS = ['PRODUCTION', 'ALARM', 'ALARME'] 
ACTIVE_TAGS = ['RUN', 'ACTIVE', 'AUTO', 'PRODUCTION', 'WORKING', 'HIGH ACTIVITY', 'LOW ACTIVITY', 'INTERMEDIATE ACTIVITY']

# ----------------------------------
# 4. DATA LOADING & CLEANING
# ----------------------------------
def clean_dataframe(df):
    """Adapt SQL columns to App standard."""
    if df.empty: return df
    
    df.columns = [c.lower().strip() for c in df.columns]
    
    rename_map = {
        'etat': 'state', 
        'total_hours': 'total_hours', 
        'total_energy_kwh': 'total_energy_kwh',
        'alarm_code': 'alarm_code',
        'alarm_text': 'description',
        'occurrence_count': 'occurrence_count',
        'last_seen': 'date',
        'timestamp': 'date', 'jour': 'date',
        'idle_hours': 'idle_hours' # Kept for mapping if other functions need it
    }
    
    cols_to_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df.rename(columns=cols_to_rename, inplace=True)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    return df

@st.cache_data(show_spinner=False)
def load_data(start, end):
    s_str = f"{start} 00:00:00"
    e_str = f"{end} 23:59:59"
    try:
        df_s = clean_dataframe(get_state_times(s_str, e_str))
        df_e = clean_dataframe(get_energy_consumption(s_str, e_str))
        df_a = clean_dataframe(get_machine_alarms(s_str, e_str))
        # df_i = clean_dataframe(get_daily_idle_trend(s_str, e_str)) <-- Line removed
        
        # Return only the 3 necessary DataFrames for the app
        return df_s, df_e, df_a
    except Exception as e:
        st.error(f"SQL Error: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# ----------------------------------
# 5. BUSINESS LOGIC (get_kpis, infer_severity)
# ----------------------------------
def get_kpis(df_s, df_e, df_a):
    defaults = {"total_h": 0, "active_h": 0, "idle_h": 0, "energy": 0, "alarms": 0, "avail": 0}
    if df_s.empty and df_e.empty and df_a.empty: return defaults
    
    target_states = ['High Activity', 'Intermediate Activity', 'Low Activity', 'True Idle (Off)']
    df_s_filtered = df_s[df_s['state'].isin(target_states)]
    
    total_4_states_h = df_s_filtered['total_hours'].sum() if not df_s_filtered.empty else 0
    
    active_h = 0
    if not df_s_filtered.empty:
        active_mask = df_s_filtered['state'].astype(str).str.upper().isin(['HIGH ACTIVITY', 'INTERMEDIATE ACTIVITY', 'LOW ACTIVITY'])
        active_h = df_s_filtered[active_mask]['total_hours'].sum()
    
    idle_h = total_4_states_h - active_h 
    
    avail = (active_h / total_4_states_h * 100) if total_4_states_h > 0 else 0
    
    total_energy = df_e['total_energy_kwh'].sum() if not df_e.empty and 'total_energy_kwh' in df_e.columns else 0
    
    nb_alarms = 0
    if not df_a.empty:
        if 'occurrence_count' in df_a.columns:
            nb_alarms = df_a['occurrence_count'].sum()
        else:
            nb_alarms = len(df_a)
    
    return {
        "total_h": total_4_states_h, "active_h": active_h, "idle_h": idle_h,
        "energy": total_energy, "alarms": nb_alarms, "avail": avail
    }

def infer_severity(row):
    """
    Precise classification based on keywords (Spanish/English).
    """
    text = (str(row.get('alarm_code', '')) + " " + str(row.get('description', ''))).upper()
    
    # 1. CRITICAL
    crit_keywords = [
        'FINAL DE CARRERA', 'ERROR', 'ERRÓNEO', 'FALLO', 'FALLA', 'PARADA', 
        'EMERGENCIA', 'COLISIÓN', 'SOBRECARGA', 'DEFECTO', 'STOP', 'FAIL', 
        'FATAL', 'LIMIT', 'EMERGENCY', 'ALARM', 'SYS FAIL', 'AXIS DRIVE'
    ]
    if any(x in text for x in crit_keywords):
        return 'CRITIQUE'
        
    # 2. WARNING
    warn_keywords = [
        'NO SE ENCUENTRA', 'NO ENCONTRADO', 'INCORRECTO', 'RETIRAR', 'ATENCIÓN', 
        'AVISO', 'BAJO', 'ALTO', 'TEMPERATURA', 'MANTENIMIENTO', 'BATERÍA', 
        'DESCONOCIDO', 'IMPOSIBLE', 'DENEGADO', 'WARNING', 'WARN', 'LOW', 
        'HIGH', 'TEMP', 'MAINT', 'MISSING', 'NOT FOUND'
    ]
    if any(x in text for x in warn_keywords):
        return 'WARNING'
        
    # 3. INFO
    return 'INFO'

# ----------------------------------
# 6. PAGES RENDERERS
# ----------------------------------

def render_home(df_s, df_e, df_a):
    st.title("🏠 Overview")
    
    # --- 1. Global KPIs ---
    nb_crit = 0
    if not df_a.empty:
        df_a['severity'] = df_a.apply(infer_severity, axis=1)
        mask_crit = df_a['severity'] == 'CRITIQUE'
        if 'occurrence_count' in df_a.columns:
            nb_crit = df_a[mask_crit]['occurrence_count'].sum()
        else:
            nb_crit = len(df_a[mask_crit])

    kpis = get_kpis(df_s, df_e, df_a)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Time", f"{kpis['total_h']:.1f} h")
    c2.metric("Active Time", f"{kpis['active_h']:.1f} h")
    c3.metric("Total Energy", f"{kpis['energy']:.0f} kWh")
    c4.metric("Critical Alarms", int(nb_crit))
    
    st.markdown("---")
    
    # --- 2. State Chart (Bar Chart - 4 Core States ONLY) ---
    st.subheader("📊 State Distribution ") 
    if not df_s.empty and 'state' in df_s.columns:
        target_states = ['High Activity', 'Intermediate Activity', 'Low Activity', 'True Idle (Off)']
        df_chart = df_s[df_s['state'].isin(target_states)]

        if not df_chart.empty:
            chart = alt.Chart(df_chart).mark_bar().encode(
                x=alt.X('total_hours', title='Hours'),
                y=alt.Y('state', title='State', sort='-x'),
                color=alt.Color('state', scale=alt.Scale(domain=STATE_DOMAIN, range=STATE_RANGE), legend=None),
                tooltip=['state', alt.Tooltip('total_hours', format='.1f')]
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No core state data available.")
    else:
        st.warning("No state data available.")

    # --- 3. Bottom Section: Energy & Alarms (Original V1 Energy Chart) ---
    c_left, c_right = st.columns([2, 1])
    
    with c_left:
        st.subheader("⚡ Energy Trend  - Daily)")
        if not df_e.empty and 'date' in df_e.columns:
            df_e['jour'] = df_e['date'].dt.date
            df_e_day = df_e.groupby('jour')['total_energy_kwh'].sum().reset_index()
            
            line = alt.Chart(df_e_day).mark_line(point=True, color='#FFC107').encode(
                x=alt.X('jour:T', title='Date'),
                y=alt.Y('total_energy_kwh', title='kWh'),
                tooltip=['jour', alt.Tooltip('total_energy_kwh', format='.1f')]
            ).interactive()
            st.altair_chart(line, use_container_width=True)
        else:
            st.info("Insufficient energy data.")
            
    with c_right:
        st.subheader("⏱️ Latest Alarms")
        if not df_a.empty:
            cols_show = ['date', 'alarm_code', 'description']
            final_cols = [c for c in cols_show if c in df_a.columns]
            
            if 'date' in df_a.columns:
                df_sorted = df_a.sort_values('date', ascending=False).head(10)
            else:
                df_sorted = df_a.sort_values('occurrence_count', ascending=False).head(10)

            st.dataframe(
                df_sorted[final_cols], 
                hide_index=True, 
                use_container_width=True,
                column_config={
                    "date": st.column_config.DatetimeColumn("Last Seen", format="MM/DD HH:mm"),
                    "alarm_code": st.column_config.TextColumn("Code"),
                    "description": st.column_config.TextColumn("Message")
                }
            )
        else:
            st.success("No recent alarms.")

def render_ops(df_s, s_date, e_date):
    st.title("⚙️ Operations Analysis")
    
    if df_s.empty: st.warning("No state data."); return

    kpis = get_kpis(df_s, pd.DataFrame(), pd.DataFrame())
    
    # Display KPIs (Availability and Total Idle are based on 4 core states)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Active", f"{kpis['active_h']:.1f} h")
    c2.metric("Total Idle/Off", f"{kpis['idle_h']:.1f} h")
    c3.metric("Availability", f"{kpis['avail']:.1f} %")
    
    st.markdown("---")

    # --- Economic Impact ---
    st.subheader("💰 Economic Impact")
    
    col_input, col_cost, col_gain = st.columns(3)
    with col_input:
        hourly_rate = st.number_input("Machine Hourly Rate ($/h)", value=65, step=5)
    
    lost_money = kpis['idle_h'] * hourly_rate
    produced_value = kpis['active_h'] * hourly_rate
    
    with col_cost:
        st.metric("Opportunity Cost (Idle)", f"{lost_money:,.0f} $", delta="-Loss", delta_color="inverse")
    with col_gain:
        st.metric("Active Time Value", f"{produced_value:,.0f} $", delta="Value Added")

    st.markdown("---")

    # --- Production Quality (Donut showing ALL 4 CORE STATES) ---
    st.subheader("📈 Operation Distribution")
    
    target_states = ['High Activity', 'Intermediate Activity', 'Low Activity', 'True Idle (Off)']
    df_target = df_s[df_s['state'].isin(target_states)].copy()

    if not df_target.empty:
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.write("**Full Time Allocation**")
            
            base = alt.Chart(df_target).encode(theta=alt.Theta("total_hours", stack=True))
            donut = base.mark_arc(outerRadius=120, innerRadius=70).encode(
                color=alt.Color("state", scale=alt.Scale(domain=STATE_DOMAIN, range=STATE_RANGE), legend=alt.Legend(title="State")),
                order=alt.Order("total_hours", sort="descending"),
                tooltip=["state", alt.Tooltip("total_hours", format=".1f")]
            )
            st.altair_chart(donut, use_container_width=True)
        
        with c2:
            st.write("---")
            high_hours = df_target[df_target['state']=='High Activity']['total_hours'].sum()
            total_time = df_target['total_hours'].sum()
            ratio_hard = (high_hours / total_time * 100) if total_time > 0 else 0
            
            st.markdown("### Insights")
            st.info(f"💡 **{ratio_hard:.1f}%** of total time is **High Activity** (Heavy machining).")
            st.write(f"The total time analyzed was **{total_time:.1f} hours**.")
    else:
        st.info("No core state data found for this period.")

def render_energy(df_e):
    st.title("⚡ Energy & Cost Analysis")
    if df_e.empty: st.warning("No energy data."); return
    
    total = df_e['total_energy_kwh'].sum()
    avg_daily = df_e['total_energy_kwh'].mean()
    
    max_val = 0
    max_date_str = "-"
    if not df_e.empty:
        idx_max = df_e['total_energy_kwh'].idxmax()
        row_max = df_e.loc[idx_max]
        max_val = row_max['total_energy_kwh']
        if 'date' in df_e.columns:
            max_date_str = row_max['date'].strftime("%m/%d/%Y")

    with st.container():
        c1, c2 = st.columns([1, 3])
        price = c1.number_input("Electricity Price ($/kWh)", 0.15, step=0.01)
        c2.metric("Total Estimated Cost", f"{(total * price):.2f} $")
    
    st.markdown("---")

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Consumption", f"{total:.0f} kWh")
    k2.metric("Daily Average", f"{avg_daily:.1f} kWh")
    k3.metric("Daily Record", f"{max_val:.1f} kWh", f"On {max_date_str}")
    
    st.markdown("---")
    
    if 'date' in df_e.columns:
        chart = alt.Chart(df_e).mark_area(
            line={'color':'darkgreen'},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='darkgreen', offset=0), alt.GradientStop(color='white', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('total_energy_kwh', title='kWh'),
            tooltip=[alt.Tooltip('date', title='Date', format='%m/%d/%Y'), alt.Tooltip('total_energy_kwh', format='.1f')]
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

def render_alarms(df_a):
    st.title("🚨 Alarms Management")
    
    if df_a.empty: 
        st.success("No alarms recorded.")
        return

    df_a['severity'] = df_a.apply(infer_severity, axis=1)

    if 'occurrence_count' in df_a.columns:
        cnt_crit = df_a[df_a['severity']=='CRITIQUE']['occurrence_count'].sum()
        cnt_warn = df_a[df_a['severity']=='WARNING']['occurrence_count'].sum()
        cnt_info = df_a[df_a['severity']=='INFO']['occurrence_count'].sum()
    else:
        cnt_crit = len(df_a[df_a['severity']=='CRITIQUE'])
        cnt_warn = len(df_a[df_a['severity']=='WARNING'])
        cnt_info = len(df_a[df_a['severity']=='INFO'])

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Critical", int(cnt_crit))
    c2.metric("🟠 Warnings", int(cnt_warn))
    c3.metric("🔵 Info", int(cnt_info))
    
    st.markdown("---")

    st.subheader("📊 Top Recurring Incidents")
    
    stats = df_a.copy()
    val_col = 'occurrence_count' if 'occurrence_count' in stats.columns else None
    
    if val_col:
        top_stats = stats.sort_values(val_col, ascending=False).head(15)
        
        severity_scale = alt.Scale(
            domain=['CRITIQUE', 'WARNING', 'INFO'],
            range=['#d32f2f', '#ffa000', '#2196f3']
        )
        
        chart = alt.Chart(top_stats).mark_bar().encode(
            x=alt.X(val_col, title="Occurrences"),
            y=alt.Y('alarm_code', sort='-x', title="Alarm Code"),
            color=alt.Color('severity', scale=severity_scale, legend=alt.Legend(title="Severity")),
            tooltip=['alarm_code', 'description', 'severity', val_col]
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
    
    st.subheader("📋 Message Details")
    t_crit, t_warn, t_info, t_all = st.tabs(["🔴 Critical", "🟠 Warnings", "🔵 Info", "📑 All"])
    
    cols = ['date', 'alarm_code', 'description', 'occurrence_count']
    final_cols = [c for c in cols if c in df_a.columns]
    
    col_config = {
        "date": st.column_config.DatetimeColumn("Date", format="MM/DD HH:mm"),
        "alarm_code": st.column_config.TextColumn("Code", width="small"),
        "description": st.column_config.TextColumn("Message", width="large"),
        "occurrence_count": st.column_config.NumberColumn("Qty", width="small"),
    }

    def show_table(sev):
        subset = df_a[df_a['severity'] == sev] if sev else df_a
        if subset.empty: st.info("No data.")
        else:
            sort_c = 'date' if 'date' in subset.columns else final_cols[0]
            st.dataframe(subset[final_cols].sort_values(sort_c, ascending=False), hide_index=True, use_container_width=True, column_config=col_config)

    with t_crit: show_table('CRITIQUE')
    with t_warn: show_table('WARNING')
    with t_info: show_table('INFO')
    with t_all: show_table(None)

# ----------------------------------
# 7. MAIN APP
# ----------------------------------
st.sidebar.title("CNC Pro")
DATA_MIN = date(2020, 1, 1)
DATA_MAX = date(2022, 12, 31)

st.sidebar.header("📅 Period")
def_end = date(2022, 2, 23)
def_start = def_end - timedelta(days=7)

dates = st.sidebar.date_input("Select Range", (def_start, def_end), min_value=DATA_MIN, max_value=DATA_MAX)

if isinstance(dates, tuple) and len(dates) == 2:
    s, e = dates
    
    with st.spinner('Loading data...'):
        # Only the 3 necessary DataFrames are returned
        df_s, df_e, df_a = load_data(s, e)
    
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigation", ["Overview", "Operations", "Energy", "Alarms"])
    
    if page == "Overview": 
        render_home(df_s, df_e, df_a)
    elif page == "Operations": 
        # df_i is not passed here as the trend chart was removed
        render_ops(df_s, s, e)
    elif page == "Energy": 
        render_energy(df_e)
    elif page == "Alarms": 
        render_alarms(df_a)

else:
    st.info("Please select a start and end date.")