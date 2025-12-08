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
    )
except ImportError:
    st.error("⛔ Module 'data_service' manquant. Vérifiez vos fichiers.")
    st.stop()

# ----------------------------------
# 2. CONFIGURATION & CSS
# ----------------------------------
st.set_page_config(page_title="CNC Master Dashboard", layout="wide", page_icon="🎛️")

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
# 3. CONSTANTES & COULEURS GLOBALES
# ----------------------------------
STATE_DOMAIN = ['High Activity', 'Intermediate Activity', 'Low Activity', 'True Idle (Off)', 'Idle', 'STANDBY', 'OFF']
STATE_RANGE = ['#084594', '#4292c6', '#9ecae1', '#ef3b2c', '#ef3b2c', '#ef3b2c', '#ef3b2c'] 

EXCLUDED_FROM_GRAPHS = ['PRODUCTION', 'ALARM', 'ALARME']
ACTIVE_TAGS = ['RUN', 'ACTIVE', 'AUTO', 'PRODUCTION', 'WORKING', 'HIGH ACTIVITY', 'LOW ACTIVITY', 'INTERMEDIATE ACTIVITY']

# ----------------------------------
# 4. NETTOYAGE & CHARGEMENT
# ----------------------------------
def clean_dataframe(df):
    """
    Adapte les noms de colonnes du SQL vers le standard de l'application.
    """
    if df.empty: return df
    
    # Standardisation (minuscules)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Mapping précis basé sur votre data_service.py
    rename_map = {
        # États
        'etat': 'state', 
        # Temps & Durée
        'total_hours': 'total_hours', 
        # Énergie
        'total_energy_kwh': 'total_energy_kwh',
        # Alarmes (Sortie SQL -> App)
        'alarm_code': 'alarm_code',
        'alarm_text': 'description',       # SQL: alarm_text -> App: description
        'occurrence_count': 'occurrence_count',
        'last_seen': 'date',               # SQL: last_seen -> App: date (pour le tri)
        # Fallbacks
        'timestamp': 'date', 'jour': 'date'
    }
    
    # Renommage sécurisé
    cols_to_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df.rename(columns=cols_to_rename, inplace=True)
    
    # Conversion date si présente
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    return df

@st.cache_data(show_spinner=False)
def load_data(start, end):
    s_str = f"{start} 00:00:00"
    e_str = f"{end} 23:59:59"
    try:
        # Récupération et nettoyage immédiat
        df_s = clean_dataframe(get_state_times(s_str, e_str))
        df_e = clean_dataframe(get_energy_consumption(s_str, e_str))
        df_a = clean_dataframe(get_machine_alarms(s_str, e_str))
        return df_s, df_e, df_a
    except Exception as e:
        st.error(f"Erreur lors de la requête SQL : {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# ----------------------------------
# 5. LOGIQUE KPI
# ----------------------------------
def get_kpis(df_s, df_e, df_a):
    defaults = {"total_h": 0, "active_h": 0, "idle_h": 0, "energy": 0, "alarms": 0, "avail": 0}
    if df_s.empty and df_e.empty and df_a.empty: return defaults
    
    # Temps Machine
    total_h = df_s['total_hours'].sum() if not df_s.empty and 'total_hours' in df_s.columns else 0
    
    active_h = 0
    if not df_s.empty and 'state' in df_s.columns:
        mask = df_s['state'].astype(str).str.upper().isin(ACTIVE_TAGS)
        active_h = df_s[mask]['total_hours'].sum()
    
    idle_h = total_h - active_h
    avail = (active_h / total_h * 100) if total_h > 0 else 0
    
    # Énergie
    total_energy = df_e['total_energy_kwh'].sum() if not df_e.empty and 'total_energy_kwh' in df_e.columns else 0
    
    # Alarmes : Somme des occurrences (car le SQL renvoie des données agrégées)
    nb_alarms = 0
    if not df_a.empty:
        if 'occurrence_count' in df_a.columns:
            nb_alarms = df_a['occurrence_count'].sum()
        else:
            nb_alarms = len(df_a) # Fallback
    
    return {
        "total_h": total_h, "active_h": active_h, "idle_h": idle_h,
        "energy": total_energy, "alarms": nb_alarms, "avail": avail
    }


def infer_severity(row):
    """
    Classification intelligente basée sur le vocabulaire espagnol/anglais de la machine.
    """
    # On combine Code + Description et on met tout en majuscules
    text = (str(row.get('alarm_code', '')) + " " + str(row.get('description', ''))).upper()
    
    # 1. NIVEAU CRITIQUE (Rouge) : Arrêts, Sécurité, Pannes matérielles
    # "Final de carrera" = Fin de course (très fréquent dans votre fichier)
    crit_keywords = [
        'FINAL DE CARRERA', 'ERROR', 'ERRÓNEO', 'FALLO', 'FALLA', 'PARADA', 
        'EMERGENCIA', 'COLISIÓN', 'SOBRECARGA', 'DEFECTO', 'STOP', 'FAIL', 
        'FATAL', 'LIMIT', 'EMERGENCY', 'ALARM', 'SYS FAIL', 'AXIS DRIVE'
    ]
    if any(x in text for x in crit_keywords):
        return 'CRITIQUE'
        
    # 2. NIVEAU AVERTISSEMENT (Orange) : Actions opérateur, Fichiers, Maintenance
    # "No se encuentra" = Fichier non trouvé
    # "Retirar" = Retirer outil (action requise)
    warn_keywords = [
        'NO SE ENCUENTRA', 'NO ENCONTRADO', 'INCORRECTO', 'RETIRAR', 'ATENCIÓN', 
        'AVISO', 'BAJO', 'ALTO', 'TEMPERATURA', 'MANTENIMIENTO', 'BATERÍA', 
        'DESCONOCIDO', 'IMPOSIBLE', 'DENEGADO', 'WARNING', 'WARN', 'LOW', 
        'HIGH', 'TEMP', 'MAINT', 'MISSING', 'NOT FOUND'
    ]
    if any(x in text for x in warn_keywords):
        return 'WARNING'
        
    # 3. NIVEAU INFO (Bleu) : Le reste (Messages d'état, Presets...)
    return 'INFO'

# ----------------------------------
# 6. PAGES
# ----------------------------------
def render_home(df_s, df_e, df_a):
    st.title("🏠 Dashboard Vue d'Ensemble")
    
    # --- 1. Calcul des KPIs Globaux ---
    nb_crit = 0
    if not df_a.empty:
        df_a['severity'] = df_a.apply(infer_severity, axis=1)
        mask_crit = df_a['severity'] == 'CRITIQUE'
        
        # On utilise occurrence_count fourni par le SQL
        if 'occurrence_count' in df_a.columns:
            nb_crit = df_a[mask_crit]['occurrence_count'].sum()
        else:
            nb_crit = len(df_a[mask_crit])

    kpis = get_kpis(df_s, df_e, df_a)
    
    # Affichage des 4 cartes KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temps Total", f"{kpis['total_h']:.1f} h")
    c2.metric("Temps Actif", f"{kpis['active_h']:.1f} h")
    c3.metric("Énergie Totale", f"{kpis['energy']:.0f} kWh")
    c4.metric("Alertes Critiques", int(nb_crit))
    
    st.markdown("---")
    
    # --- 2. Graphique des États (Bar chart) ---
    st.subheader("📊 Répartition (Hors Prod & Alarmes)")
    if not df_s.empty and 'state' in df_s.columns:
        mask = ~df_s['state'].astype(str).str.upper().isin(EXCLUDED_FROM_GRAPHS)
        df_chart = df_s[mask]
        
        if not df_chart.empty:
            chart = alt.Chart(df_chart).mark_bar().encode(
                x=alt.X('total_hours', title='Heures'),
                y=alt.Y('state', title='État', sort='-x'),
                color=alt.Color('state', scale=alt.Scale(domain=STATE_DOMAIN, range=STATE_RANGE), legend=None),
                tooltip=['state', alt.Tooltip('total_hours', format='.1f')]
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Données filtrées (uniquement Prod ou Alarme).")
    else:
        st.warning("Aucune donnée d'état reçue.")

    # --- 3. Colonnes du bas : Énergie & Dernières Alarmes ---
    c_left, c_right = st.columns([2, 1])
    
    with c_left:
        st.subheader("⚡ Tendance Conso")
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
            st.info("Données Énergie insuffisantes.")
            
    with c_right:
        st.subheader("⏱️ Dernières Alarmes") # Titre mis à jour
        if not df_a.empty:
            # Sélecteur de colonnes : On veut voir la DATE en premier pour le chrono
            cols_show = ['date', 'alarm_code', 'description']
            
            # Vérification que les colonnes existent (sécurité mapping)
            final_cols = [c for c in cols_show if c in df_a.columns]
            
            # --- MODIFICATION ICI : Tri par date (last_seen) descendant ---
            if 'date' in df_a.columns:
                df_sorted = df_a.sort_values('date', ascending=False).head(10)
            else:
                # Fallback si pas de date : tri par fréquence
                df_sorted = df_a.sort_values('occurrence_count', ascending=False).head(10)

            # Affichage du tableau
            st.dataframe(
                df_sorted[final_cols], 
                hide_index=True, 
                use_container_width=True,
                # Optionnel : formatage de la date pour être plus lisible
                column_config={
                    "date": st.column_config.DatetimeColumn(
                        "Dernière fois",
                        format="DD/MM HH:mm"
                    )
                }
            )
        else:
            st.success("RAS - Aucune alarme récente")
            
def render_ops(df_s, s_date, e_date):
    st.title("⚙️ Analyse Opérationnelle")
    
    if df_s.empty: st.warning("Pas de données."); return

    kpis = get_kpis(df_s, pd.DataFrame(), pd.DataFrame())
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Actif", f"{kpis['active_h']:.1f} h")
    c2.metric("Total Idle/Off", f"{kpis['idle_h']:.1f} h")
    c3.metric("Disponibilité", f"{kpis['avail']:.1f} %")

    st.markdown("---")
    
    if 'total_hours' in df_s.columns and 'state' in df_s.columns:
        mask = ~df_s['state'].astype(str).str.upper().isin(EXCLUDED_FROM_GRAPHS)
        df_filtered = df_s[mask].copy()
        
        if df_filtered.empty:
            st.info("Rien à afficher avec les filtres actuels.")
            return
            
        # Pas de groupby nécessaire si le SQL renvoie déjà state/total_hours uniques
        # Mais par sécurité on regroupe
        df_grouped = df_filtered.groupby('state')['total_hours'].sum().reset_index()
        
        # Tri
        rank_map = {name: i for i, name in enumerate(STATE_DOMAIN)}
        df_grouped['rank'] = df_grouped['state'].map(rank_map).fillna(99)
        
        total = df_grouped['total_hours'].sum()
        df_grouped['percent'] = df_grouped['total_hours'] / total if total > 0 else 0

        c_pie, c_bar = st.columns([1, 2])
        
        with c_pie:
            base = alt.Chart(df_grouped).encode(theta=alt.Theta("total_hours", stack=True))
            pie = base.mark_arc(outerRadius=100).encode(
                color=alt.Color("state", scale=alt.Scale(domain=STATE_DOMAIN, range=STATE_RANGE), legend=None), 
                order=alt.Order("rank", sort="ascending"),
                tooltip=["state", alt.Tooltip("total_hours", format=".1f"), alt.Tooltip("percent", format=".1%")]
            )
            st.altair_chart(pie, use_container_width=True)
            
        with c_bar:
            bar = alt.Chart(df_grouped).mark_bar().encode(
                x=alt.X('total_hours', title='Heures Totales'),
                y=alt.Y('state', sort=alt.EncodingSortField(field="rank", order="ascending"), title='État'),
                color=alt.Color('state', scale=alt.Scale(domain=STATE_DOMAIN, range=STATE_RANGE), legend=None),
                tooltip=['state', 'total_hours']
            ).interactive()
            st.altair_chart(bar, use_container_width=True)

def render_energy(df_e):
    st.title("⚡ Analyse Coûts & Énergie")
    if df_e.empty: st.warning("Pas de données."); return
    
    # 1. Calculs
    total = df_e['total_energy_kwh'].sum()
    avg_daily = df_e['total_energy_kwh'].mean()
    
    # Recherche du jour record
    max_val = 0
    max_date_str = "-"
    if not df_e.empty:
        idx_max = df_e['total_energy_kwh'].idxmax()
        row_max = df_e.loc[idx_max]
        max_val = row_max['total_energy_kwh']
        # Formatage date (ex: 12/02/2022)
        if 'date' in df_e.columns:
            max_date_str = row_max['date'].strftime("%d/%m/%Y")

    # 2. Section Coûts (Haut de page)
    with st.container():
        c1, c2 = st.columns([1, 3])
        price = c1.number_input("Prix kWh (€)", 0.15, step=0.01)
        c2.metric("Coût Total Estimé", f"{(total * price):.2f} €")
    
    st.markdown("---")

    # 3. Nouveaux KPIs (Conso / Moyenne / Record)
    k1, k2, k3 = st.columns(3)
    k1.metric("Conso Totale", f"{total:.0f} kWh")
    k2.metric("Moyenne / Jour", f"{avg_daily:.1f} kWh")
    k3.metric("Record Journalier", f"{max_val:.1f} kWh", f"Le {max_date_str}")
    
    st.markdown("---")
    
    # 4. Graphique
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
            tooltip=[alt.Tooltip('date', title='Date', format='%d %b %Y'), alt.Tooltip('total_energy_kwh', format='.1f')]
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

def render_alarms(df_a):
    st.title("🚨 Gestion des Alarmes")
    
    if df_a.empty: 
        st.success("RAS - Aucune alarme sur la période.")
        return

    # 1. Calcul de la Sévérité (Critique / Warning / Info)
    df_a['severity'] = df_a.apply(infer_severity, axis=1)

    # 2. Calcul des KPIs (En utilisant occurrence_count du SQL)
    if 'occurrence_count' in df_a.columns:
        cnt_crit = df_a[df_a['severity']=='CRITIQUE']['occurrence_count'].sum()
        cnt_warn = df_a[df_a['severity']=='WARNING']['occurrence_count'].sum()
        cnt_info = df_a[df_a['severity']=='INFO']['occurrence_count'].sum()
    else:
        cnt_crit = len(df_a[df_a['severity']=='CRITIQUE'])
        cnt_warn = len(df_a[df_a['severity']=='WARNING'])
        cnt_info = len(df_a[df_a['severity']=='INFO'])

    # Affichage des compteurs en haut
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Critiques (Bloquant)", int(cnt_crit))
    c2.metric("🟠 Avertissements", int(cnt_warn))
    c3.metric("🔵 Informations", int(cnt_info))
    
    st.markdown("---")

    # 3. Graphique "Top 15 Fréquence" (Toujours utile pour voir les récurrentes)
    st.subheader("📊 Top Incidents Récurrents")
    stats = df_a.copy()
    val_col = 'occurrence_count' if 'occurrence_count' in stats.columns else None
    
    if val_col:
        top_stats = stats.sort_values(val_col, ascending=False).head(15)
        chart = alt.Chart(top_stats).mark_bar(color='#d32f2f').encode(
            x=alt.X(val_col, title="Nombre d'occurrences"),
            y=alt.Y('alarm_code', sort='-x', title="Code Alarme"),
            tooltip=['alarm_code', 'description', val_col]
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
    
    # 4. Tableaux Détaillés (4 Onglets)
    st.subheader("📋 Détails des Messages")
    
    # Création des 4 onglets demandés
    t_crit, t_warn, t_info, t_all = st.tabs([
        "🔴 Critiques", 
        "🟠 Avertissements", 
        "🔵 Infos", 
        "📑 Tout l'historique"
    ])
    
    # Configuration des colonnes pour un affichage propre
    cols_to_show = ['date', 'alarm_code', 'description', 'occurrence_count']
    # On filtre pour ne garder que les colonnes qui existent vraiment dans le DF
    final_cols = [c for c in cols_to_show if c in df_a.columns]
    
    # Configuration du formatage des colonnes (Date lisible, Titres clairs)
    col_config = {
        "date": st.column_config.DatetimeColumn(
            "Dernière apparition",
            format="DD/MM/YYYY HH:mm",
            width="medium"
        ),
        "alarm_code": st.column_config.TextColumn("Code", width="small"),
        "description": st.column_config.TextColumn("Message", width="large"),
        "occurrence_count": st.column_config.NumberColumn("Qté", width="small"),
    }

    # Fonction locale pour afficher une table triée par date
    def show_filtered_table(severity_filter=None):
        if severity_filter:
            # Filtrage par sévérité
            df_filtered = df_a[df_a['severity'] == severity_filter]
        else:
            # Pas de filtre (Tout afficher)
            df_filtered = df_a
            
        if df_filtered.empty:
            st.info("Aucune donnée dans cette catégorie.")
        else:
            # Tri par date décroissante (le plus récent en haut)
            sort_col = 'date' if 'date' in df_filtered.columns else final_cols[0]
            st.dataframe(
                df_filtered[final_cols].sort_values(sort_col, ascending=False),
                hide_index=True,
                use_container_width=True,
                column_config=col_config
            )

    # Remplissage des onglets
    with t_crit:
        show_filtered_table('CRITIQUE')
        
    with t_warn:
        show_filtered_table('WARNING')
        
    with t_info:
        show_filtered_table('INFO')
        
    with t_all:
        show_filtered_table(None) # Affiche tout
# ----------------------------------
# 7. MAIN APP
# ----------------------------------# ----------------------------------
# 7. MAIN APP (CORRIGÉ)
# ----------------------------------
st.sidebar.title("CNC Pro")
DATA_MIN = date(2020, 1, 1)
DATA_MAX = date(2022, 12, 31)

st.sidebar.header("📅 Période")
# Semaine type avec des données
def_end = date(2022, 2, 23)
def_start = def_end - timedelta(days=7)

dates = st.sidebar.date_input("Sélection", (def_start, def_end), min_value=DATA_MIN, max_value=DATA_MAX)

# On vérifie qu'on a bien une date de début et de fin
if isinstance(dates, tuple) and len(dates) == 2:
    s, e = dates
    
    # 1. On charge les données (Le cache évite de recharger à chaque clic)
    # On ajoute un petit spinner visuel
    with st.spinner('Chargement des données...'):
        df_s, df_e, df_a = load_data(s, e)
    
    # 2. Le menu de navigation est TOUJOURS visible
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigation", ["Home", "Opérations", "Énergie", "Alarmes"])
    
    # 3. Affichage de la page
    if page == "Home": 
        render_home(df_s, df_e, df_a)
    elif page == "Opérations": 
        render_ops(df_s, s, e)
    elif page == "Énergie": 
        render_energy(df_e)
    elif page == "Alarmes": 
        render_alarms(df_a)

else:
    st.info("Veuillez sélectionner une date de début et une date de fin.")