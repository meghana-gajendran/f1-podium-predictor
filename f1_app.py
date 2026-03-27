import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import fastf1
import os
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Podium Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;900&family=Barlow:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8e8;
}

.stApp { background-color: #0a0a0f; }

h1, h2, h3 {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 900;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

.hero {
    background: linear-gradient(135deg, #1a0505 0%, #0a0a0f 50%, #05051a 100%);
    border: 1px solid #e10600;
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #e10600, #ff6b35, #e10600);
}
.hero-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 3rem;
    font-weight: 900;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0;
    line-height: 1;
}
.hero-sub {
    color: #e10600;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

.podium-card {
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
    transition: transform 0.2s;
}
.podium-card:hover { transform: translateY(-4px); }
.podium-p1 { background: linear-gradient(135deg, #1a1400, #2a2000); border-color: #FFD700; }
.podium-p2 { background: linear-gradient(135deg, #0f0f14, #1a1a22); border-color: #C0C0C0; }
.podium-p3 { background: linear-gradient(135deg, #140a00, #1e1200); border-color: #CD7F32; }

.medal { font-size: 2.5rem; }
.driver-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #fff;
    text-transform: uppercase;
    margin: 0.3rem 0;
}
.team-name { color: #aaa; font-size: 0.85rem; letter-spacing: 0.1em; text-transform: uppercase; }
.score-badge {
    display: inline-block;
    background: rgba(225,6,0,0.2);
    border: 1px solid #e10600;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.9rem;
    font-weight: 600;
    color: #e10600;
    margin-top: 0.5rem;
}
.grid-badge {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 0.15rem 0.6rem;
    font-size: 0.8rem;
    color: #aaa;
    margin-top: 0.3rem;
}

.metric-card {
    background: #111118;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #e10600;
}
.metric-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.1em; }

.status-live {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(0,200,100,0.1);
    border: 1px solid #00c864;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.8rem;
    color: #00c864;
    font-weight: 600;
}
.status-cached {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,165,0,0.1);
    border: 1px solid orange;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.8rem;
    color: orange;
    font-weight: 600;
}

.section-header {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #fff;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-left: 3px solid #e10600;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem;
}

.stButton > button {
    background: #e10600 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #ff1a10 !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
BASE_OPENF1 = 'https://api.openf1.org/v1'
BASE_ERGAST  = 'https://api.jolpi.ca/ergast/f1'

TEAM_COLORS = {
    'Red Bull Racing': '#3671C6', 'Ferrari': '#E8002D',
    'Mercedes': '#27F4D2', 'McLaren': '#FF8000',
    'Aston Martin': '#229971', 'Alpine': '#FF87BC',
    'Williams': '#64C4FF', 'RB': '#6692FF',
    'Racing Bulls': '#6692FF', 'Kick Sauber': '#52E252',
    'Haas F1 Team': '#B6BABD', 'Haas': '#B6BABD',
    'Audi': '#FF0000', 'Cadillac': '#FFFFFF',
}

FALLBACK_GRID = [
    ('RUS','George Russell','Mercedes'),
    ('ANT','Kimi Antonelli','Mercedes'),
    ('LEC','Charles Leclerc','Ferrari'),
    ('HAM','Lewis Hamilton','Ferrari'),
    ('NOR','Lando Norris','McLaren'),
    ('PIA','Oscar Piastri','McLaren'),
    ('VER','Max Verstappen','Red Bull Racing'),
    ('HAD','Isack Hadjar','Red Bull Racing'),
    ('ALO','Fernando Alonso','Aston Martin'),
    ('STR','Lance Stroll','Aston Martin'),
    ('SAI','Carlos Sainz','Williams'),
    ('ALB','Alexander Albon','Williams'),
    ('TSU','Yuki Tsunoda','Racing Bulls'),
    ('LIN','Arvid Lindblad','Racing Bulls'),
    ('GAS','Pierre Gasly','Alpine'),
    ('DOO','Jack Doohan','Alpine'),
    ('HUL','Nico Hulkenberg','Audi'),
    ('BOR','Gabriel Bortoleto','Audi'),
    ('HER','Colton Herta','Cadillac'),
    ('ILO','Theo Pourchaire','Cadillac'),
]

FEATURES = ['GridPosition','DriverRecentForm','TeamRecentForm',
            'CircuitWins','DriverPodiumRate','TeamEncoded','RegulationYear']

# ── API helpers ───────────────────────────────────────────────
def openf1_get(endpoint, params=None):
    for attempt in range(3):
        try:
            r = requests.get(f'{BASE_OPENF1}/{endpoint}', params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except:
            time.sleep(2)
    return []

def ergast_get(path):
    for attempt in range(3):
        try:
            r = requests.get(f'{BASE_ERGAST}/{path}', timeout=15)
            r.raise_for_status()
            return r.json()
        except:
            time.sleep(2)
    return {}

# ── Data loading ──────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=86400)
def load_season_results(year):
    os.makedirs('f1_cache', exist_ok=True)
    fastf1.Cache.enable_cache('f1_cache')
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        races = schedule[schedule['EventFormat'] != 'testing']
    except:
        return pd.DataFrame()
    all_results = []
    for _, event in races.iterrows():
        round_num = event['RoundNumber']
        if round_num == 0: continue
        try:
            session = fastf1.get_session(year, round_num, 'R')
            session.load(telemetry=False, weather=False, messages=False)
            results = session.results[['DriverNumber','Abbreviation','FullName',
                                       'TeamName','GridPosition','Position','Points']].copy()
            results['Year'] = year
            results['Round'] = round_num
            results['CircuitName'] = event.get('Location', event['EventName'])
            all_results.append(results)
        except:
            continue
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=1800)
def get_live_session():
    """
    Smart session finder:
    1. If qualifying has happened this weekend → use it (live grid available)
    2. If it's a race weekend but only practice done → show upcoming race name,
       use last qualifying for grid fallback
    3. Otherwise → use last completed qualifying
    """
    sessions = []
    for year in [2026, 2025]:
        sessions = openf1_get('sessions', {'year': year})  # get ALL session types
        if sessions: break
    if not sessions: return None, None

    df = pd.DataFrame(sessions)
    df['date_start'] = pd.to_datetime(df['date_start'], utc=True, errors='coerce')
    now = pd.Timestamp.now(tz='UTC')

    # Find the current or most recent race weekend (any session started)
    completed_any = df[df['date_start'] <= now].sort_values('date_start', ascending=False)
    if completed_any.empty: return None, None

    # What meeting_key is active this weekend?
    latest_meeting_key = completed_any.iloc[0].get('meeting_key')

    # Check if qualifying for THIS weekend is done
    this_weekend = df[df['meeting_key'] == latest_meeting_key]
    quali_done = this_weekend[
        (this_weekend['session_name'] == 'Qualifying') &
        (this_weekend['date_start'] <= now)
    ]

    if not quali_done.empty:
        # ✅ Qualifying happened — use it for live grid
        latest = quali_done.sort_values('date_start', ascending=False).iloc[0]
        name = next((latest[c] for c in ['circuit_short_name','country_name','location']
                     if c in latest.index and pd.notna(latest[c])), 'Unknown')
        return latest, name
    else:
        # 🔄 Practice only so far — show THIS weekend's name, use last quali for grid
        # Get race name from current weekend
        current_name = next(
            (completed_any.iloc[0][c] for c in ['circuit_short_name','country_name','location']
             if c in completed_any.iloc[0].index and pd.notna(completed_any.iloc[0][c])),
            'Unknown'
        )
        # Get last completed qualifying for grid positions
        all_quali = df[
            (df['session_name'] == 'Qualifying') &
            (df['date_start'] <= now)
        ].sort_values('date_start', ascending=False)

        if all_quali.empty: return None, current_name
        last_quali = all_quali.iloc[0]
        # Override the name with current weekend
        last_quali = last_quali.copy()
        return last_quali, current_name

@st.cache_data(show_spinner=False, ttl=1800)
def get_live_grid(session_key):
    drivers = openf1_get('drivers', {'session_key': session_key})
    if not drivers: return pd.DataFrame()
    drv_df = pd.DataFrame(drivers)
    col_map = {}
    for col in drv_df.columns:
        cl = col.lower()
        if cl in ('name_acronym','abbreviation','acronym'): col_map[col] = 'Abbreviation'
        elif 'full' in cl and 'name' in cl: col_map[col] = 'FullName'
        elif 'team' in cl and 'name' in cl: col_map[col] = 'TeamName'
        elif cl == 'driver_number': col_map[col] = 'DriverNumber'
    drv_df = drv_df.rename(columns=col_map)
    for c in ['DriverNumber','Abbreviation','FullName','TeamName']:
        if c not in drv_df.columns: drv_df[c] = 'Unknown'
    drv_df = drv_df[['DriverNumber','Abbreviation','FullName','TeamName']].drop_duplicates('DriverNumber')
    positions = openf1_get('position', {'session_key': session_key})
    if positions:
        pos_df = pd.DataFrame(positions)
        if 'date' in pos_df.columns:
            pos_df['date'] = pd.to_datetime(pos_df['date'], utc=True, errors='coerce')
            final_pos = (pos_df.sort_values('date').groupby('driver_number')['position']
                         .last().reset_index()
                         .rename(columns={'driver_number':'DriverNumber','position':'GridPosition'}))
            drv_df = drv_df.merge(final_pos, on='DriverNumber', how='left')
    if 'GridPosition' not in drv_df.columns:
        drv_df['GridPosition'] = range(1, len(drv_df)+1)
    drv_df['GridPosition'] = pd.to_numeric(drv_df['GridPosition'], errors='coerce').fillna(20).astype(int)
    return drv_df.sort_values('GridPosition').reset_index(drop=True)

@st.cache_data(show_spinner=False, ttl=1800)
def get_forms():
    for year in [2026, 2025]:
        data = ergast_get(f'{year}/results.json?limit=300')
        try:
            races = data['MRData']['RaceTable']['Races']
            rows = []
            for race in races[-5:]:
                for res in race['Results']:
                    code = res['Driver'].get('code', res['Driver']['driverId'][:3].upper())
                    rows.append({'Code': code,
                                 'TeamName': res['Constructor']['name'],
                                 'Round': int(race['round']),
                                 'Points': float(res['points']),
                                 'Pos': int(res['position'])})
            if rows:
                df = pd.DataFrame(rows)
                driver_form = df.groupby('Code')['Points'].mean().to_dict()
                team_pts = df.groupby(['Round','TeamName'])['Points'].sum().reset_index()
                team_form = team_pts.groupby('TeamName')['Points'].mean().to_dict()
                podium_rate = df.groupby('Code').apply(lambda x: (x['Pos'] <= 3).mean()).to_dict()
                return driver_form, team_form, podium_rate
        except: continue
    return {}, {}, {}

@st.cache_data(show_spinner=False)
def train_models():
    all_dfs = []
    weights = {2023: 1, 2024: 1, 2025: 2, 2026: 3}
    for year, w in weights.items():
        df = load_season_results(year)
        if not df.empty:
            for _ in range(w): all_dfs.append(df)
    if not all_dfs: return None, None, None
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all['Position']     = pd.to_numeric(df_all['Position'], errors='coerce')
    df_all['GridPosition'] = pd.to_numeric(df_all['GridPosition'], errors='coerce')
    df_all['Points']       = pd.to_numeric(df_all['Points'], errors='coerce')
    df_all.dropna(subset=['Position'], inplace=True)
    # Feature engineering
    df_all['OnPodium']     = (df_all['Position'] <= 3).astype(int)
    df_all['GridPosition'] = df_all['GridPosition'].fillna(20)
    df_all = df_all.sort_values(['Abbreviation','Year','Round'])
    df_all['DriverRecentForm'] = (df_all.groupby('Abbreviation')['Points']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0))
    team_pts = df_all.groupby(['TeamName','Year','Round'])['Points'].sum().reset_index()
    team_pts['TeamRecentForm'] = (team_pts.groupby('TeamName')['Points']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0))
    df_all = df_all.merge(team_pts[['TeamName','Year','Round','TeamRecentForm']],
                          on=['TeamName','Year','Round'], how='left')
    cw = (df_all[df_all['Position']==1].groupby(['Abbreviation','CircuitName'])
          .size().reset_index(name='CircuitWins'))
    df_all = df_all.merge(cw, on=['Abbreviation','CircuitName'], how='left')
    df_all['CircuitWins'] = df_all['CircuitWins'].fillna(0)
    df_all['DriverPodiumRate'] = (df_all.groupby('Abbreviation')['OnPodium']
        .transform(lambda x: x.shift(1).expanding().mean()).fillna(0))
    df_all['RegulationYear'] = (df_all['Year'] == 2026).astype(int)
    le = LabelEncoder()
    df_all['TeamEncoded'] = le.fit_transform(df_all['TeamName'].fillna('Unknown'))
    X = df_all[FEATURES]; y = df_all['OnPodium']
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, class_weight='balanced')
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    rf.fit(X, y); gb.fit(X, y)
    return rf, gb, le

def stats_score(grid, driver_form, team_form, podium_rate, circuit_wins):
    return round(
        0.40 * max(0, (21-grid)/20) +
        0.25 * min(driver_form/25, 1.0) +
        0.15 * min(team_form/44, 1.0) +
        0.10 * podium_rate +
        0.10 * min(circuit_wins*0.1, 0.3), 4)

# ── Main App ──────────────────────────────────────────────────
def main():
    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-title">🏎️ F1 Podium Predictor</div>
        <div class="hero-sub">Live qualifying data · ML ensemble · AI analysis</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown("### 📊 Model Info")
        st.markdown("""
        **Features used:**
        - Grid position
        - Driver recent form (5-race rolling avg)
        - Team recent form
        - Circuit win history
        - Career podium rate
        - Regulation year flag

        **Ensemble weights:**
        - Random Forest: 40%
        - Gradient Boosting: 35%
        - Stats formula: 25%
        """)
        st.markdown("---")
        st.markdown("### 🔄 Data Sources")
        st.markdown("- FastF1 (2023–2026)")
        st.markdown("- OpenF1 API (live grid)")
        st.markdown("- Ergast/Jolpica (form)")
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")

    # Load data
    col1, col2, col3 = st.columns(3)

    with st.spinner("🏎️ Loading F1 data..."):
        session, race_name = get_live_session()
        driver_form, team_form, podium_rates = get_forms()

    if session is not None:
        race_display = f"{race_name} Grand Prix"
        with col1:
            st.markdown('<div class="metric-card"><div class="metric-value">' +
                        race_name + '</div><div class="metric-label">Current race</div></div>',
                        unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><div class="metric-value">' +
                        str(session.get('year', '2026')) +
                        '</div><div class="metric-label">Season</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><div class="metric-value">' +
                        str(len(driver_form)) +
                        '</div><div class="metric-label">Drivers tracked</div></div>',
                        unsafe_allow_html=True)
    else:
        race_display = "Next F1 Race"
        st.warning("⚠️ Could not fetch live session — using fallback grid")

    st.markdown(f'<div class="section-header">📡 Live Qualifying Grid</div>', unsafe_allow_html=True)

    # Fetch grid
    live_grid = None
    if session is not None:
        with st.spinner("Fetching live qualifying grid..."):
            live_grid = get_live_grid(session['session_key'])

    if live_grid is not None and not live_grid.empty:
        st.markdown('<span class="status-live">● LIVE DATA</span>', unsafe_allow_html=True)
        source_grid = live_grid
    else:
        st.markdown('<span class="status-cached">● FALLBACK GRID</span>', unsafe_allow_html=True)
        source_grid = pd.DataFrame(FALLBACK_GRID, columns=['Abbreviation','FullName','TeamName'])
        source_grid['GridPosition'] = range(1, len(FALLBACK_GRID)+1)

    # Train models
    with st.spinner("🤖 Training ML models on 2023–2026 data..."):
        rf, gb, le = train_models()

    if rf is None:
        st.error("Could not train models — no historical data available")
        return

    # Build pred df
    known_teams = list(le.classes_)
    rows = []
    for i, row in source_grid.iterrows():
        abbr  = str(row.get('Abbreviation', 'UNK'))
        team  = str(row.get('TeamName', 'Unknown'))
        grid  = int(row.get('GridPosition', i+1))
        name  = str(row.get('FullName', abbr))
        if name.isupper() or (len(name.split()) < 2 and not name.isupper()):
            name = abbr
        rows.append({
            'Abbreviation': abbr, 'FullName': name, 'TeamName': team,
            'GridPosition': grid,
            'DriverRecentForm': driver_form.get(abbr, 3.0),
            'TeamRecentForm':   team_form.get(team, 5.0),
            'DriverPodiumRate': podium_rates.get(abbr, 0.05),
            'CircuitWins':      0,
            'RegulationYear':   1,
        })
    pred_df = pd.DataFrame(rows)
    pred_df['TeamEncoded'] = pred_df['TeamName'].apply(
        lambda t: le.transform([t])[0] if t in known_teams else -1)
    pred_df['RF_prob']    = rf.predict_proba(pred_df[FEATURES])[:, 1]
    pred_df['GB_prob']    = gb.predict_proba(pred_df[FEATURES])[:, 1]
    pred_df['StatsScore'] = pred_df.apply(
        lambda r: stats_score(r.GridPosition, r.DriverRecentForm,
                              r.TeamRecentForm, r.DriverPodiumRate, r.CircuitWins), axis=1)
    pred_df['EnsembleScore'] = (0.40*pred_df['RF_prob'] +
                                 0.35*pred_df['GB_prob'] +
                                 0.25*pred_df['StatsScore'])
    ranked = pred_df.sort_values('EnsembleScore', ascending=False).reset_index(drop=True)

    # Podium cards
    st.markdown(f'<div class="section-header">🏆 Predicted Podium — {race_display}</div>',
                unsafe_allow_html=True)
    medals   = ['🥇','🥈','🥉']
    classes  = ['podium-p1','podium-p2','podium-p3']
    cols = st.columns(3)
    for i, col in enumerate(cols):
        row = ranked.iloc[i]
        name  = str(row['FullName'])
        team  = str(row['TeamName'])
        grid  = int(row['GridPosition'])
        score = float(str(row['EnsembleScore']).split('\n')[0])
        with col:
            st.markdown(f"""
            <div class="podium-card {classes[i]}">
                <div class="medal">{medals[i]}</div>
                <div class="driver-name">{name}</div>
                <div class="team-name">{team}</div>
                <div class="score-badge">Score: {score:.3f}</div><br>
                <div class="grid-badge">Grid P{grid}</div>
            </div>
            """, unsafe_allow_html=True)

    # Charts
    st.markdown('<div class="section-header">📊 Prediction Analysis</div>', unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        top10 = ranked.head(10).copy()
        colors = [TEAM_COLORS.get(str(t), '#888888') for t in top10['TeamName']]
        abbrevs = [str(a) for a in top10['Abbreviation']]
        scores  = [float(str(s).split('\n')[0]) for s in top10['EnsembleScore']]
        fig1 = go.Figure(go.Bar(
            x=scores[::-1], y=abbrevs[::-1], orientation='h',
            marker_color=colors[::-1],
            marker_line_color='rgba(255,255,255,0.2)', marker_line_width=0.5,
        ))
        fig1.update_layout(
            title='Podium probability — top 10',
            paper_bgcolor='#111118', plot_bgcolor='#111118',
            font=dict(color='#e8e8e8', family='Barlow'),
            title_font=dict(size=14, color='#fff'),
            height=380, margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(gridcolor='#222', title='Ensemble score'),
            yaxis=dict(gridcolor='#222'),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with chart_col2:
        top5 = ranked.head(5)
        abbrevs5 = [str(a) for a in top5['Abbreviation']]
        rf_vals  = [float(str(v).split('\n')[0]) for v in top5['RF_prob']]
        gb_vals  = [float(str(v).split('\n')[0]) for v in top5['GB_prob']]
        st_vals  = [float(str(v).split('\n')[0]) for v in top5['StatsScore']]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Random Forest', x=abbrevs5, y=rf_vals,
                              marker_color='#3671C6'))
        fig2.add_trace(go.Bar(name='Gradient Boost', x=abbrevs5, y=gb_vals,
                              marker_color='#E8002D'))
        fig2.add_trace(go.Bar(name='Stats formula', x=abbrevs5, y=st_vals,
                              marker_color='#FF8000'))
        fig2.update_layout(
            title='Score breakdown — top 5', barmode='group',
            paper_bgcolor='#111118', plot_bgcolor='#111118',
            font=dict(color='#e8e8e8', family='Barlow'),
            title_font=dict(size=14, color='#fff'),
            height=380, margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(gridcolor='#222'), yaxis=dict(gridcolor='#222'),
            legend=dict(bgcolor='#111118'),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Full table
    st.markdown('<div class="section-header">📋 Full Predicted Order</div>', unsafe_allow_html=True)
    table = ranked[['Abbreviation','FullName','TeamName','GridPosition',
                    'RF_prob','GB_prob','StatsScore','EnsembleScore']].copy()
    table['FullName']  = table['FullName'].apply(lambda x: str(x).split('\n')[0])
    table['TeamName']  = table['TeamName'].apply(lambda x: str(x).split('\n')[0])
    table.index = range(1, len(table)+1)
    table.index.name = 'Pos'
    table.columns = ['Driver','Full Name','Team','Grid','RF','GB','Stats','Ensemble']
    for col in ['RF','GB','Stats','Ensemble']:
        table[col] = table[col].apply(lambda x: round(float(str(x).split('\n')[0]), 3))
    st.dataframe(table.style.background_gradient(subset=['Ensemble'], cmap='RdYlGn')
                 .format({'RF':'{:.3f}','GB':'{:.3f}','Stats':'{:.3f}',
                          'Ensemble':'{:.3f}','Grid':'{:.0f}'}),
                 use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#555;font-size:0.8rem;padding:1rem;">
        Built with FastF1 · OpenF1 API · Ergast/Jolpica · scikit-learn 🏎️
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()