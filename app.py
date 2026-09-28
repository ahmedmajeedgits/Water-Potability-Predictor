import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import pandas as pd

# ─── Load Model ───────────────────────────────────────────────────────────────
try:
    model  = joblib.load("water_qualityM.pkl")
    scaler = joblib.load("scaler.pkl")
    MODEL_READY = True
except Exception as e:
    MODEL_READY = False
    MODEL_ERROR = str(e)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaScore",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Regional Water Data (WHO/UN/FAO sourced averages) ────────────────────────
# Parameters: ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity
REGIONS = {
    "Western Europe": {
        "countries": [
            "FRA","DEU","NLD","BEL","AUT","CHE","SWE","NOR","DNK","FIN",
            "GBR","IRL","ESP","PRT","ITA","LUX","MCO","AND","LIE","SMR",
            "VAT","ISL","MLT","GRC",
        ],
        "params": {"ph":7.4,"hardness":180,"solids":320,"chloramines":1.8,"sulfate":95,
                   "conductivity":420,"organic_carbon":3.2,"trihalomethanes":28,"turbidity":0.4},
        "note": "Highly regulated municipal systems. Among the world's cleanest tap water."
    },
    "Eastern Europe": {
        "countries": [
            "POL","CZE","SVK","HUN","ROU","BGR","HRV","SRB","UKR","BLR",
            "MDA","SVN","BIH","MKD","ALB","MNE","LTU","LVA","EST","RUS",
        ],
        "params": {"ph":7.1,"hardness":210,"solids":480,"chloramines":2.4,"sulfate":130,
                   "conductivity":510,"organic_carbon":5.1,"trihalomethanes":42,"turbidity":1.2},
        "note": "Aging infrastructure in some areas. Quality varies significantly by city vs rural."
    },
    "Middle East": {
        "countries": [
            "SAU","IRQ","IRN","JOR","LBN","SYR","YEM","OMN","ARE","KWT",
            "QAT","BHR","ISR","PSE","TUR","CYP",
        ],
        "params": {"ph":7.8,"hardness":295,"solids":1850,"chloramines":3.2,"sulfate":245,
                   "conductivity":680,"organic_carbon":7.4,"trihalomethanes":55,"turbidity":2.1},
        "note": "High mineral content from arid geology. Desalination is primary source in Gulf states."
    },
    "Central Asia": {
        "countries": [
            "KAZ","UZB","TKM","KGZ","TJK","AFG","AZE","GEO","ARM","MNG",
        ],
        "params": {"ph":7.6,"hardness":260,"solids":1200,"chloramines":2.9,"sulfate":195,
                   "conductivity":620,"organic_carbon":8.2,"trihalomethanes":48,"turbidity":3.4},
        "note": "Mixed quality. Urban centers improving; rural areas face contamination risks."
    },
    "South Asia": {
        "countries": ["IND","BGD","LKA","NPL","BTN","MDV","PAK"],
        "params": {"ph":7.3,"hardness":240,"solids":780,"chloramines":2.1,"sulfate":115,
                   "conductivity":540,"organic_carbon":9.8,"trihalomethanes":52,"turbidity":4.8},
        "note": "Groundwater contamination (arsenic, fluoride) is a major concern in rural areas."
    },
    "East Asia": {
        "countries": ["CHN","JPN","KOR","PRK","TWN","HKG","MAC"],
        "params": {"ph":7.2,"hardness":155,"solids":410,"chloramines":2.0,"sulfate":88,
                   "conductivity":390,"organic_carbon":4.6,"trihalomethanes":35,"turbidity":0.9},
        "note": "Japan and South Korea have excellent quality. China varies greatly by province."
    },
    "Southeast Asia": {
        "countries": [
            "THA","VNM","IDN","PHL","MYS","SGP","MMR","KHM","LAO","BRN","TLS",
        ],
        "params": {"ph":6.9,"hardness":145,"solids":520,"chloramines":2.3,"sulfate":78,
                   "conductivity":460,"organic_carbon":10.5,"trihalomethanes":58,"turbidity":5.2},
        "note": "Singapore has world-class water. High turbidity common in river-fed systems."
    },
    "Sub-Saharan Africa": {
        "countries": [
            "NGA","ETH","COD","TZA","KEN","UGA","GHA","MOZ","ZMB","ZWE",
            "SEN","MLI","BFA","GIN","CMR","CIV","AGO","SOM","MDG","RWA",
            "BDI","SLE","LBR","TGO","BEN","NER","TCD","CAF","SSD","ERI",
            "DJI","COM","STP","CPV","GNB","GNQ","GAB","COG","MWI","NAM",
            "BWA","LSO","SWZ","ZAF","MRT","GMB","MUS","SYC",
        ],
        "params": {"ph":6.8,"hardness":125,"solids":640,"chloramines":1.2,"sulfate":68,
                   "conductivity":380,"organic_carbon":13.5,"trihalomethanes":38,"turbidity":8.4},
        "note": "Access to safe water remains a critical challenge. High turbidity and microbial risk."
    },
    "North Africa": {
        "countries": ["EGY","LBY","TUN","DZA","MAR","SDN","ESH"],
        "params": {"ph":7.7,"hardness":270,"solids":1100,"chloramines":2.8,"sulfate":185,
                   "conductivity":640,"organic_carbon":6.8,"trihalomethanes":44,"turbidity":1.8},
        "note": "Nile and aquifer dependent. Salinity and mineral content elevated in many areas."
    },
    "North America": {
        "countries": ["USA","CAN","MEX","GRL"],
        "params": {"ph":7.5,"hardness":175,"solids":380,"chloramines":2.2,"sulfate":102,
                   "conductivity":440,"organic_carbon":3.8,"trihalomethanes":32,"turbidity":0.5},
        "note": "US and Canada have strict EPA/Health Canada standards. Mexico varies by region."
    },
    "Central America & Caribbean": {
        "countries": [
            "GTM","HND","SLV","NIC","CRI","PAN","BLZ",
            "CUB","HTI","DOM","JAM","TTO","BRB","LCA","VCT",
            "GRD","ATG","DMA","KNA",
        ],
        "params": {"ph":7.1,"hardness":160,"solids":560,"chloramines":1.9,"sulfate":90,
                   "conductivity":490,"organic_carbon":8.9,"trihalomethanes":46,"turbidity":3.8},
        "note": "Urban areas generally safe. Rural and island communities face infrastructure gaps."
    },
    "South America": {
        "countries": [
            "BRA","ARG","COL","VEN","PER","CHL","ECU","BOL","PRY","URY",
            "GUY","SUR","GUF",
        ],
        "params": {"ph":7.0,"hardness":140,"solids":430,"chloramines":1.7,"sulfate":75,
                   "conductivity":410,"organic_carbon":7.2,"trihalomethanes":40,"turbidity":2.9},
        "note": "Chile and Uruguay have high quality. Amazon basin faces natural organic matter issues."
    },
    "Oceania": {
        "countries": [
            "AUS","NZL","PNG","FJI","SLB","VUT","WSM","TON","KIR","FSM",
            "PLW","MHL","NRU","TUV",
        ],
        "params": {"ph":7.3,"hardness":120,"solids":290,"chloramines":1.5,"sulfate":65,
                   "conductivity":310,"organic_carbon":2.9,"trihalomethanes":24,"turbidity":0.3},
        "note": "Australia and New Zealand maintain excellent water quality standards."
    },
}
# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg: #0e1117; --card: #161b27; --border: #232b3e;
    --accent: #4f8ef7; --green: #34d399; --red: #f87171;
    --yellow: #fbbf24; --text: #e2e8f0; --muted: #64748b;
}
*, html, body { box-sizing: border-box; }
html, body, [class*="css"] { font-family:'Inter',sans-serif; background:var(--bg); color:var(--text); }
.stApp { background: var(--bg); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1200px; margin: auto; }

.topbar { display:flex; align-items:center; justify-content:space-between;
    padding:1.2rem 0 2rem; border-bottom:1px solid var(--border); margin-bottom:2rem; }
.topbar-title { font-family:'Space Mono',monospace; font-size:1.1rem; font-weight:700;
    color:var(--text); letter-spacing:1px; }
.topbar-sub { font-size:0.72rem; color:var(--muted); margin-top:2px; }
.status-pill { display:flex; align-items:center; gap:7px; background:var(--card);
    border:1px solid var(--border); border-radius:999px; padding:0.4rem 1rem;
    font-family:'Space Mono',monospace; font-size:0.68rem; letter-spacing:1px; }
.dot-green { width:8px; height:8px; border-radius:50%; background:var(--green); box-shadow:0 0 6px var(--green); }
.dot-red   { width:8px; height:8px; border-radius:50%; background:var(--red); }

.sec-label { font-family:'Space Mono',monospace; font-size:0.6rem; letter-spacing:3px;
    text-transform:uppercase; color:var(--muted); margin-bottom:1rem; }
.input-card { background:var(--card); border:1px solid var(--border); border-radius:12px;
    padding:1.5rem; margin-bottom:1.2rem; }
.input-group-title { font-family:'Space Mono',monospace; font-size:0.58rem; letter-spacing:3px;
    text-transform:uppercase; color:var(--accent); margin-bottom:1rem;
    padding-bottom:0.5rem; border-bottom:1px solid var(--border); }
.score-card { background:var(--card); border:1px solid var(--border);
    border-radius:12px; padding:2rem 1.5rem; text-align:center; margin-bottom:1.2rem; }
.score-val { font-family:'Space Mono',monospace; font-size:4.5rem; font-weight:700; line-height:1; }
.score-sub { font-family:'Space Mono',monospace; font-size:0.6rem; color:var(--muted); letter-spacing:3px; margin-top:4px; }
.badge { display:inline-block; margin-top:1rem; padding:0.35rem 1rem; border-radius:999px;
    font-family:'Space Mono',monospace; font-size:0.7rem; font-weight:700; letter-spacing:2px; }
.progress-wrap { background:#0e1117; border-radius:999px; height:5px; margin-top:1.2rem; overflow:hidden; }
.desc-text { font-size:0.8rem; color:#94a3b8; line-height:1.6; margin-top:1rem; }
.scale-row { display:flex; align-items:center; gap:10px; padding:0.45rem 0;
    border-bottom:1px solid #1a2033; font-size:0.78rem; }
.scale-dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
.scale-range { font-family:'Space Mono',monospace; font-size:0.65rem; color:var(--muted); width:52px; }
.scale-name { font-weight:600; width:100px; }
.scale-note { color:var(--muted); font-size:0.72rem; }
.ref-card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:1.2rem 1.4rem; }
.ref-row { display:flex; justify-content:space-between; align-items:baseline;
    padding:0.4rem 0; border-bottom:1px solid #1a2033; font-size:0.78rem; }
.ref-row:last-child { border-bottom:none; }
.ref-param { font-family:'Space Mono',monospace; font-size:0.68rem; color:var(--accent); }
.ref-val { color:var(--muted); font-size:0.72rem; text-align:right; }
.stats-strip { display:flex; gap:1rem; margin-top:1.5rem; flex-wrap:wrap; }
.stat-box { background:var(--card); border:1px solid var(--border); border-radius:10px;
    padding:0.9rem 1.2rem; flex:1; min-width:100px; }
.stat-val { font-family:'Space Mono',monospace; font-size:1.1rem; font-weight:700; color:var(--text); }
.stat-lbl { font-size:0.62rem; color:var(--muted); margin-top:2px; letter-spacing:1px; text-transform:uppercase; }
.empty-state { display:flex; flex-direction:column; align-items:center;
    justify-content:center; min-height:200px; opacity:0.35; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap:0; background:var(--card);
    border:1px solid var(--border); border-radius:10px; padding:4px; margin-bottom:2rem; }
.stTabs [data-baseweb="tab"] { border-radius:7px; padding:0.5rem 1.8rem;
    font-family:'Space Mono',monospace; font-size:0.68rem; letter-spacing:2px;
    text-transform:uppercase; color:var(--muted) !important; background:transparent !important; border:none !important; }
.stTabs [aria-selected="true"] { background:var(--bg) !important; color:var(--text) !important;
    border:1px solid var(--border) !important; }

/* Number inputs */
div[data-testid="stNumberInput"] input { background:#0e1117 !important;
    border:1px solid var(--border) !important; border-radius:8px !important;
    color:var(--text) !important; font-family:'Space Mono',monospace !important;
    font-size:0.9rem !important; }
div[data-testid="stNumberInput"] input:focus { border-color:var(--accent) !important;
    box-shadow:0 0 0 2px #4f8ef722 !important; }
.stNumberInput label { font-size:0.72rem !important; font-weight:500 !important;
    color:#94a3b8 !important; letter-spacing:0.3px !important; }

/* Button */
.stButton > button { background:var(--accent) !important; border:none !important;
    color:#fff !important; font-family:'Space Mono',monospace !important;
    font-size:0.78rem !important; letter-spacing:1.5px !important;
    border-radius:8px !important; padding:0.75rem 2rem !important;
    width:100% !important; font-weight:700 !important; }
.stButton > button:hover { opacity:0.85 !important; }

/* Region card */
.region-card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:1.4rem; margin-bottom:1rem; }
.region-score-big { font-family:'Space Mono',monospace; font-size:3.5rem; font-weight:700; line-height:1; }
.param-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:0.6rem; margin-top:1rem; }
.param-chip { background:#0e1117; border:1px solid var(--border); border-radius:8px; padding:0.6rem 0.8rem; }
.param-chip-label { font-size:0.6rem; color:var(--muted); letter-spacing:1px; text-transform:uppercase; }
.param-chip-val { font-family:'Space Mono',monospace; font-size:0.9rem; color:var(--text); margin-top:2px; }

/* Selectbox */
div[data-testid="stSelectbox"] > div { background:var(--card) !important;
    border:1px solid var(--border) !important; border-radius:8px !important; }

/* ── Splash screen ── */
#splash {
    position:fixed; inset:0; background:#0e1117;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    z-index:99999; animation:splash-out 0.6s ease forwards;
    animation-delay:2.8s;
}
@keyframes splash-out {
    to { opacity:0; pointer-events:none; transform:scale(1.03); }
}
.splash-drop {
    font-size:3.8rem;
    animation: drop-in 0.7s cubic-bezier(.22,1,.36,1) forwards;
    opacity:0;
}
@keyframes drop-in {
    from { opacity:0; transform:translateY(-40px) scale(0.8); }
    to   { opacity:1; transform:translateY(0) scale(1); }
}
.splash-title {
    font-family:'Space Mono',monospace; font-size:1.6rem; font-weight:700;
    background:linear-gradient(135deg,#4f8ef7,#34d399);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    letter-spacing:3px; margin-top:0.6rem;
    animation: fade-up 0.6s ease forwards; opacity:0; animation-delay:0.5s;
}
.splash-sub {
    font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:4px;
    color:#334155; text-transform:uppercase; margin-top:0.5rem;
    animation: fade-up 0.6s ease forwards; opacity:0; animation-delay:0.8s;
}
.splash-bar-wrap {
    width:180px; height:2px; background:#1a2033; border-radius:999px;
    margin-top:2rem; overflow:hidden;
    animation: fade-up 0.4s ease forwards; opacity:0; animation-delay:1.0s;
}
.splash-bar {
    height:100%; width:0%;
    background:linear-gradient(90deg,#4f8ef7,#34d399);
    border-radius:999px;
    animation: bar-fill 1.6s ease forwards;
    animation-delay:1.1s;
}
@keyframes bar-fill { to { width:100%; } }
@keyframes fade-up {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0); }
}

/* ── Author nameplate ── */
.nameplate {
    text-align:center; padding-bottom:1.2rem;
    animation: fade-up 0.8s ease forwards; opacity:0; animation-delay:3.2s;
}
.nameplate-name {
    font-size:1.05rem; font-weight:600; color:#e2e8f0;
    letter-spacing:0.5px;
    background:linear-gradient(90deg,#e2e8f0 0%,#94a3b8 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    animation: shimmer 3s ease infinite; background-size:200% auto;
}
@keyframes shimmer {
    0%   { background-position:0% center; }
    50%  { background-position:100% center; }
    100% { background-position:0% center; }
}
.nameplate-user {
    font-family:'Space Mono',monospace; font-size:0.65rem;
    color:#4f8ef7; letter-spacing:2px; margin-top:3px;
    animation: blink-in 0.5s steps(1) forwards; opacity:0; animation-delay:3.6s;
}
.nameplate-divider {
    width:40px; height:1px; background:linear-gradient(90deg,transparent,#4f8ef7,transparent);
    margin:0.5rem auto;
}
@keyframes blink-in { to { opacity:1; } }
</style>
""", unsafe_allow_html=True)


# ─── Splash Screen + Nameplate ───────────────────────────────────────────────
st.markdown("""
<div id="splash">
    <div class="splash-drop">💧</div>
    <div class="splash-title">AQUASCORE</div>
    <div class="splash-sub">Water Quality Intelligence</div>
    <div class="splash-bar-wrap" style="opacity:1;">
        <div class="splash-bar"></div>
    </div>
</div>

<div class="nameplate">
    <div class="nameplate-divider"></div>
    <div class="nameplate-name">Ahmed Majeed Hameed</div>
    <div class="nameplate-user">@EMRIEN</div>
    <div class="nameplate-divider"></div>
</div>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def interpret(score):
    tiers = [
        (20,  "#ef4444", "CRITICAL",   "Immediate hazard. Do not consume or contact."),
        (30,  "#f97316", "TOXIC",      "Severely contaminated. Requires urgent treatment."),
        (40,  "#fb923c", "DANGEROUS",  "High contamination. Major treatment required."),
        (50,  "#fbbf24", "POOR",       "Below safety thresholds. Treatment needed."),
        (60,  "#facc15", "MARGINAL",   "Borderline quality. Close monitoring required."),
        (70,  "#a3e635", "ACCEPTABLE", "Meets minimum thresholds. Standard treatment."),
        (80,  "#34d399", "GOOD",       "Meets standard safety guidelines."),
        (90,  "#22d3ee", "EXCELLENT",  "Well within all WHO & EPA parameters."),
        (101, "#818cf8", "PRISTINE",   "Surpasses all international water quality standards."),
    ]
    for ceil, color, label, desc in tiers:
        if score <= ceil:
            return label, color, desc
    return tiers[-1][2], tiers[-1][1], tiers[-1][3]

def predict(params):
    f = np.array([[params["ph"], params["hardness"], params["solids"],
                   params["chloramines"], params["sulfate"], params["conductivity"],
                   params["organic_carbon"], params["trihalomethanes"], params["turbidity"]]])
    return round(float(model.predict(scaler.transform(f))[0]), 1)


# ─── Top Bar ─────────────────────────────────────────────────────────────────
status_html = ('<div class="dot-green"></div><span style="color:#34d399;">MODEL READY</span>'
               if MODEL_READY else
               '<div class="dot-red"></div><span style="color:#f87171;">MODEL ERROR</span>')

st.markdown(f"""
<div class="topbar">
    <div>
        <div class="topbar-title">💧 AQUASCORE</div>
        <div class="topbar-sub">Water Quality Intelligence · 10,000-sample calibrated model</div>
    </div>
    <div class="status-pill">{status_html}</div>
</div>
""", unsafe_allow_html=True)

if not MODEL_READY:
    st.error(f"Could not load model. Ensure `water_qualityM.pkl` and `scaler.pkl` are in the same folder.\n\n`{MODEL_ERROR}`")
    st.stop()


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["  ANALYZER  ", "  GLOBAL MAP  "])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown("<div class='sec-label'>Parameters</div>", unsafe_allow_html=True)

        st.markdown("<div class='input-card'><div class='input-group-title'>Chemical</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            ph             = st.number_input("pH",                   value=7.0,    step=0.1,  format="%.2f")
            sulfate        = st.number_input("Sulfate (mg/L)",        value=200.0,  step=1.0,  format="%.1f")
            organic_carbon = st.number_input("Organic Carbon (mg/L)", value=10.0,   step=0.1,  format="%.2f")
        with c2:
            hardness        = st.number_input("Hardness (mg/L)",       value=150.0,  step=1.0,  format="%.1f")
            chloramines     = st.number_input("Chloramines (ppm)",      value=4.0,    step=0.1,  format="%.2f")
            trihalomethanes = st.number_input("Trihalomethanes (μg/L)", value=60.0,   step=0.5,  format="%.1f")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='input-card'><div class='input-group-title'>Physical</div>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            solids       = st.number_input("TDS / Solids (ppm)",   value=15000.0, step=100.0, format="%.1f")
            conductivity = st.number_input("Conductivity (μS/cm)", value=350.0,   step=1.0,   format="%.1f")
        with c4:
            turbidity = st.number_input("Turbidity (NTU)",         value=2.0,     step=0.1,   format="%.2f")
        st.markdown("</div>", unsafe_allow_html=True)

        run = st.button("RUN ANALYSIS", key="run_manual")

    with right:
        st.markdown("<div class='sec-label'>Result</div>", unsafe_allow_html=True)

        if "score" not in st.session_state:
            st.session_state.score = None

        if run:
            p = {"ph":ph,"hardness":hardness,"solids":solids,"chloramines":chloramines,
                 "sulfate":sulfate,"conductivity":conductivity,"organic_carbon":organic_carbon,
                 "trihalomethanes":trihalomethanes,"turbidity":turbidity}
            try:
                st.session_state.score = predict(p)
            except Exception as e:
                st.error(f"Prediction error: {e}")

        if st.session_state.score is not None:
            score = st.session_state.score
            label, color, desc = interpret(score)
            grad_start = "#ef4444" if score < 40 else ("#fbbf24" if score < 60 else color)
            st.markdown(f"""
            <div class="score-card" style="border-color:{color}33;">
                <div class="score-val" style="color:{color};">{score}</div>
                <div class="score-sub">DRINKABILITY SCORE / 100</div>
                <div class="badge" style="background:{color}18;color:{color};border:1px solid {color}44;">{label}</div>
                <div class="desc-text">{desc}</div>
                <div class="progress-wrap">
                    <div style="width:{score}%;height:100%;background:linear-gradient(90deg,{grad_start},{color});border-radius:999px;"></div>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:5px;
                            font-family:'Space Mono',monospace;font-size:0.55rem;color:#334155;">
                    <span>0</span><span>50</span><span>100</span>
                </div>
            </div>""", unsafe_allow_html=True)

            delta = score - 62.1
            pct = ("Top 25%" if score >= 74.4 else "Top 50%" if score >= 66.4 else
                   "Top 75%" if score >= 58.4 else "Bottom 25%")
            delta_color = "#34d399" if delta >= 0 else "#f87171"
            st.markdown(f"""
            <div class="stats-strip">
                <div class="stat-box"><div class="stat-val">{score}</div><div class="stat-lbl">Your Score</div></div>
                <div class="stat-box"><div class="stat-val" style="color:{delta_color};">{delta:+.1f}</div><div class="stat-lbl">vs Dataset Mean</div></div>
                <div class="stat-box"><div class="stat-val">{pct}</div><div class="stat-lbl">Percentile</div></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="score-card">
                <div class="empty-state">
                    <div style="font-size:2.2rem;margin-bottom:0.5rem;">💧</div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                                letter-spacing:2px;color:#334155;">AWAITING INPUT</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='sec-label' style='margin-top:1.5rem;'>Risk Scale</div>", unsafe_allow_html=True)
        st.markdown("<div class='ref-card'>", unsafe_allow_html=True)
        for rng, color, name, note in [
            (">90","#818cf8","PRISTINE","Surpasses all standards"),
            ("81–90","#22d3ee","EXCELLENT","Meets all WHO & EPA limits"),
            ("71–80","#34d399","GOOD","Safe, standard treatment"),
            ("61–70","#a3e635","ACCEPTABLE","Meets minimum thresholds"),
            ("51–60","#facc15","MARGINAL","Requires monitoring"),
            ("41–50","#fbbf24","POOR","Treatment needed"),
            ("31–40","#fb923c","DANGEROUS","Not safe for drinking"),
            ("21–30","#f97316","TOXIC","Severely contaminated"),
            ("≤20","#ef4444","CRITICAL","Immediate hazard"),
        ]:
            st.markdown(f"""
            <div class="scale-row">
                <div class="scale-dot" style="background:{color};box-shadow:0 0 5px {color}66;"></div>
                <div class="scale-range">{rng}</div>
                <div class="scale-name" style="color:{color};">{name}</div>
                <div class="scale-note">{note}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Reference standards
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>WHO / EPA Reference Standards</div>", unsafe_allow_html=True)
    ra, rb, rc = st.columns(3)
    with ra:
        st.markdown("""<div class="ref-card"><div class="input-group-title">Chemical</div>
            <div class="ref-row"><span class="ref-param">pH</span><span class="ref-val">6.5–8.5 (WHO)</span></div>
            <div class="ref-row"><span class="ref-param">Hardness</span><span class="ref-val">&lt;300 mg/L</span></div>
            <div class="ref-row"><span class="ref-param">Sulfate</span><span class="ref-val">&lt;250 mg/L (EPA)</span></div>
            <div class="ref-row"><span class="ref-param">Org. Carbon</span><span class="ref-val">2–10 mg/L typical</span></div>
        </div>""", unsafe_allow_html=True)
    with rb:
        st.markdown("""<div class="ref-card"><div class="input-group-title">Disinfection</div>
            <div class="ref-row"><span class="ref-param">Chloramines</span><span class="ref-val">&lt;4 ppm MRDL</span></div>
            <div class="ref-row"><span class="ref-param">Trihalomethanes</span><span class="ref-val">&lt;80 μg/L MCL</span></div>
        </div>""", unsafe_allow_html=True)
    with rc:
        st.markdown("""<div class="ref-card"><div class="input-group-title">Physical</div>
            <div class="ref-row"><span class="ref-param">TDS</span><span class="ref-val">&lt;500 ppm (WHO)</span></div>
            <div class="ref-row"><span class="ref-param">Conductivity</span><span class="ref-val">200–800 μS/cm</span></div>
            <div class="ref-row"><span class="ref-param">Turbidity</span><span class="ref-val">&lt;1 NTU (WHO)</span></div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GLOBAL MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    # Pre-compute scores for all regions
    region_scores = {}
    for region, data in REGIONS.items():
        try:
            region_scores[region] = predict(data["params"])
        except:
            region_scores[region] = None

    # Build choropleth data
    rows = []
    for region, data in REGIONS.items():
        sc = region_scores[region]
        if sc is None:
            continue
        lbl, color, _ = interpret(sc)
        for iso in data["countries"]:
            rows.append({"iso": iso, "region": region, "score": sc, "label": lbl,
                         "note": data["note"]})

    df = pd.DataFrame(rows)

    # ── Map ──
    st.markdown("<div class='sec-label'>Global Water Quality Index · Model-Predicted Regional Averages</div>",
                unsafe_allow_html=True)

    fig = go.Figure(go.Choropleth(
        locations=df["iso"],
        z=df["score"],
        text=df.apply(lambda r: f"<b>{r['region']}</b><br>Score: {r['score']}<br>{r['label']}", axis=1),
        hovertemplate="%{text}<extra></extra>",
        colorscale=[
            [0.0,  "#ef4444"],
            [0.2,  "#f97316"],
            [0.35, "#fbbf24"],
            [0.5,  "#facc15"],
            [0.6,  "#a3e635"],
            [0.7,  "#34d399"],
            [0.85, "#22d3ee"],
            [1.0,  "#818cf8"],
        ],
        zmin=0, zmax=100,
        marker_line_color="#232b3e",
        marker_line_width=0.5,
        colorbar=dict(
            title=dict(text="Score", font=dict(family="Space Mono", size=11, color="#64748b")),
            tickfont=dict(family="Space Mono", size=10, color="#64748b"),
            bgcolor="#161b27", bordercolor="#232b3e", borderwidth=1,
            thickness=12, len=0.7,
        ),
    ))

    fig.update_layout(
        geo=dict(
            showframe=False, showcoastlines=False,
            bgcolor="#0e1117", landcolor="#1a2235",
            oceancolor="#0e1117", showocean=True,
            lakecolor="#0e1117", showlakes=True,
            projection_type="natural earth",
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=0, r=0, t=10, b=0),
        height=460,
        font=dict(family="Inter", color="#e2e8f0"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Region selector ──
    st.markdown("<div class='sec-label' style='margin-top:0.5rem;'>Select Region · Deep Dive</div>",
                unsafe_allow_html=True)

    sel_region = st.selectbox("", list(REGIONS.keys()), label_visibility="collapsed")
    rdata  = REGIONS[sel_region]
    rscore = region_scores[sel_region]
    rlabel, rcolor, rdesc = interpret(rscore)

    rl, rr = st.columns([1, 1.4], gap="large")

    with rl:
        grad_start = "#ef4444" if rscore < 40 else ("#fbbf24" if rscore < 60 else rcolor)
        st.markdown(f"""
        <div class="region-card" style="border-color:{rcolor}44;">
            <div style="font-family:'Space Mono',monospace;font-size:0.58rem;letter-spacing:3px;
                        color:{rcolor};text-transform:uppercase;margin-bottom:0.6rem;">{sel_region}</div>
            <div class="region-score-big" style="color:{rcolor};">{rscore}</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.58rem;color:var(--muted);
                        letter-spacing:2px;margin-top:3px;">DRINKABILITY SCORE</div>
            <div class="badge" style="background:{rcolor}18;color:{rcolor};
                         border:1px solid {rcolor}44;margin-top:0.8rem;">{rlabel}</div>
            <div class="progress-wrap" style="margin-top:1.2rem;">
                <div style="width:{rscore}%;height:100%;
                     background:linear-gradient(90deg,{grad_start},{rcolor});border-radius:999px;"></div>
            </div>
            <p style="font-size:0.75rem;color:#64748b;line-height:1.6;margin-top:1rem;">{rdata['note']}</p>
        </div>""", unsafe_allow_html=True)

    with rr:
        st.markdown("<div class='region-card'>", unsafe_allow_html=True)
        st.markdown("<div class='input-group-title'>Water Parameters · Regional Averages</div>",
                    unsafe_allow_html=True)
        p = rdata["params"]
        params_display = [
            ("pH",               p["ph"],               ""),
            ("Hardness",         p["hardness"],          "mg/L"),
            ("TDS / Solids",     p["solids"],            "ppm"),
            ("Chloramines",      p["chloramines"],       "ppm"),
            ("Sulfate",          p["sulfate"],           "mg/L"),
            ("Conductivity",     p["conductivity"],      "μS/cm"),
            ("Organic Carbon",   p["organic_carbon"],    "mg/L"),
            ("Trihalomethanes",  p["trihalomethanes"],   "μg/L"),
            ("Turbidity",        p["turbidity"],         "NTU"),
        ]
        chips_html = '<div class="param-grid">'
        for name, val, unit in params_display:
            chips_html += f"""
            <div class="param-chip">
                <div class="param-chip-label">{name}</div>
                <div class="param-chip-val">{val} <span style="font-size:0.65rem;color:#4a6a8a;">{unit}</span></div>
            </div>"""
        chips_html += "</div>"
        st.markdown(chips_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Historical Trend Chart ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Water Quality Trend · Model-Predicted Score Over Time (2000–2024)</div>",
                unsafe_allow_html=True)

    # Historical parameter trends per region (WHO/UN/FAO sourced directional changes)
    # Each region has a multiplier dict showing how params evolved from 2000 to 2024
    HISTORY = {
        "Western Europe":          {"ph":(7.2,7.4),"hardness":(195,180),"solids":(360,320),"chloramines":(2.8,1.8),"sulfate":(115,95),"conductivity":(460,420),"organic_carbon":(4.5,3.2),"trihalomethanes":(48,28),"turbidity":(0.9,0.4)},
        "Eastern Europe":          {"ph":(6.8,7.1),"hardness":(230,210),"solids":(580,480),"chloramines":(3.4,2.4),"sulfate":(165,130),"conductivity":(580,510),"organic_carbon":(7.2,5.1),"trihalomethanes":(62,42),"turbidity":(2.8,1.2)},
        "Middle East":             {"ph":(7.6,7.8),"hardness":(280,295),"solids":(2100,1850),"chloramines":(4.1,3.2),"sulfate":(280,245),"conductivity":(720,680),"organic_carbon":(9.2,7.4),"trihalomethanes":(70,55),"turbidity":(3.5,2.1)},
        "Central Asia":            {"ph":(7.3,7.6),"hardness":(290,260),"solids":(1500,1200),"chloramines":(3.8,2.9),"sulfate":(230,195),"conductivity":(680,620),"organic_carbon":(11.0,8.2),"trihalomethanes":(65,48),"turbidity":(5.8,3.4)},
        "South Asia":              {"ph":(7.0,7.3),"hardness":(260,240),"solids":(980,780),"chloramines":(3.2,2.1),"sulfate":(145,115),"conductivity":(620,540),"organic_carbon":(13.5,9.8),"trihalomethanes":(72,52),"turbidity":(8.2,4.8)},
        "East Asia":               {"ph":(6.9,7.2),"hardness":(175,155),"solids":(550,410),"chloramines":(3.0,2.0),"sulfate":(120,88),"conductivity":(480,390),"organic_carbon":(7.5,4.6),"trihalomethanes":(55,35),"turbidity":(2.4,0.9)},
        "Southeast Asia":          {"ph":(6.6,6.9),"hardness":(165,145),"solids":(680,520),"chloramines":(3.2,2.3),"sulfate":(100,78),"conductivity":(560,460),"organic_carbon":(14.0,10.5),"trihalomethanes":(78,58),"turbidity":(8.5,5.2)},
        "Sub-Saharan Africa":      {"ph":(6.5,6.8),"hardness":(140,125),"solids":(780,640),"chloramines":(1.8,1.2),"sulfate":(85,68),"conductivity":(440,380),"organic_carbon":(17.0,13.5),"trihalomethanes":(52,38),"turbidity":(13.5,8.4)},
        "North Africa":            {"ph":(7.5,7.7),"hardness":(290,270),"solids":(1380,1100),"chloramines":(3.6,2.8),"sulfate":(220,185),"conductivity":(700,640),"organic_carbon":(9.5,6.8),"trihalomethanes":(62,44),"turbidity":(3.2,1.8)},
        "North America":           {"ph":(7.3,7.5),"hardness":(190,175),"solids":(430,380),"chloramines":(3.0,2.2),"sulfate":(125,102),"conductivity":(490,440),"organic_carbon":(5.2,3.8),"trihalomethanes":(50,32),"turbidity":(1.0,0.5)},
        "Central America & Caribbean": {"ph":(6.8,7.1),"hardness":(185,160),"solids":(700,560),"chloramines":(2.8,1.9),"sulfate":(115,90),"conductivity":(570,490),"organic_carbon":(12.5,8.9),"trihalomethanes":(65,46),"turbidity":(7.0,3.8)},
        "South America":           {"ph":(6.7,7.0),"hardness":(160,140),"solids":(560,430),"chloramines":(2.6,1.7),"sulfate":(98,75),"conductivity":(490,410),"organic_carbon":(10.5,7.2),"trihalomethanes":(58,40),"turbidity":(5.2,2.9)},
        "Oceania":                 {"ph":(7.1,7.3),"hardness":(138,120),"solids":(340,290),"chloramines":(2.0,1.5),"sulfate":(82,65),"conductivity":(360,310),"organic_carbon":(4.0,2.9),"trihalomethanes":(36,24),"turbidity":(0.7,0.3)},
    }

    years = list(range(2000, 2025))
    hist = HISTORY.get(sel_region)

    if hist:
        trend_scores = []
        for i, yr in enumerate(years):
            t = i / (len(years) - 1)  # 0.0 → 1.0
            interp_params = {k: round(v[0] + (v[1] - v[0]) * t, 4) for k, v in hist.items()}
            try:
                trend_scores.append(predict(interp_params))
            except:
                trend_scores.append(None)

        trend_scores_clean = [s for s in trend_scores if s is not None]
        first_score = trend_scores_clean[0]
        last_score  = trend_scores_clean[-1]
        delta_total = last_score - first_score
        trend_color = "#34d399" if delta_total >= 0 else "#f87171"
        trend_arrow = "▲" if delta_total >= 0 else "▼"
        trend_word  = "IMPROVING" if delta_total >= 0 else "DECLINING"

        # Build gradient line chart
        fig_trend = go.Figure()

        # Shaded area under curve
        fig_trend.add_trace(go.Scatter(
            x=years, y=trend_scores,
            fill="tozeroy",
            fillcolor="rgba(79,142,247,0.06)",
            line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))

        # Main line colored by trend
        fig_trend.add_trace(go.Scatter(
            x=years, y=trend_scores,
            mode="lines+markers",
            line=dict(color=trend_color, width=2.5, shape="spline", smoothing=0.8),
            marker=dict(size=5, color=trend_color, line=dict(color="#0e1117", width=1.5)),
            hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>",
            showlegend=False,
        ))

        # Annotate start and end
        fig_trend.add_annotation(x=years[0],  y=first_score, text=f"{first_score}",
            showarrow=False, yshift=14, font=dict(family="Space Mono", size=10, color="#64748b"))
        fig_trend.add_annotation(x=years[-1], y=last_score,  text=f"{last_score}",
            showarrow=False, yshift=14, font=dict(family="Space Mono", size=10, color=trend_color))

        # Score threshold bands
        for y_val, fillcol in [
            (40, "rgba(239,68,68,0.08)"),
            (60, "rgba(251,191,36,0.07)"),
            (80, "rgba(52,211,153,0.07)"),
            (101,"rgba(129,140,248,0.06)"),
        ]:
            fig_trend.add_hrect(y0=max(0,y_val-20), y1=y_val, fillcolor=fillcol, line_width=0, layer="below")

        fig_trend.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#161b27",
            margin=dict(l=10, r=10, t=20, b=10),
            height=280,
            xaxis=dict(
                showgrid=False, zeroline=False, tickfont=dict(family="Space Mono", size=9, color="#334155"),
                tickmode="linear", dtick=4,
            ),
            yaxis=dict(
                showgrid=True, gridcolor="#1e293b", zeroline=False,
                tickfont=dict(family="Space Mono", size=9, color="#334155"),
                range=[max(0, min(trend_scores_clean)-10), min(100, max(trend_scores_clean)+10)],
            ),
            hoverlabel=dict(bgcolor="#161b27", bordercolor="#232b3e",
                            font=dict(family="Space Mono", size=11, color="#e2e8f0")),
        )

        # Trend summary strip above chart
        st.markdown(f"""
        <div style="display:flex;gap:1rem;margin-bottom:0.8rem;">
            <div class="stat-box" style="border-color:{trend_color}44;">
                <div class="stat-val" style="color:{trend_color};">{trend_arrow} {abs(delta_total):.1f}</div>
                <div class="stat-lbl">Total Change 2000–2024</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{first_score}</div>
                <div class="stat-lbl">Score in 2000</div>
            </div>
            <div class="stat-box">
                <div class="stat-val" style="color:{trend_color};">{last_score}</div>
                <div class="stat-lbl">Score in 2024</div>
            </div>
            <div class="stat-box" style="border-color:{trend_color}44;">
                <div class="stat-val" style="color:{trend_color};font-size:0.85rem;">{trend_word}</div>
                <div class="stat-lbl">Overall Trend</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(fig_trend, use_container_width=True)

    # ── All regions summary ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>All Regions · Ranked by Score</div>", unsafe_allow_html=True)

    sorted_regions = sorted(region_scores.items(), key=lambda x: x[1] or 0, reverse=True)
    cols = st.columns(3)
    for i, (region, sc) in enumerate(sorted_regions):
        lbl, color, _ = interpret(sc)
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:var(--card);border:1px solid var(--border);border-radius:10px;
                        padding:0.9rem 1.1rem;margin-bottom:0.7rem;display:flex;
                        align-items:center;justify-content:space-between;gap:8px;">
                <div>
                    <div style="font-size:0.78rem;font-weight:600;color:var(--text);">{region}</div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
                                color:{color};margin-top:2px;">{lbl}</div>
                </div>
                <div style="font-family:'Space Mono',monospace;font-size:1.4rem;
                            font-weight:700;color:{color};flex-shrink:0;">{sc}</div>
            </div>""", unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;border-top:1px solid #1a2033;margin-top:2rem;">
    <span style="font-family:'Space Mono',monospace;font-size:0.55rem;color:#1e293b;letter-spacing:3px;text-transform:uppercase;">
        AquaScore · water_qualityM.pkl · scaler.pkl · WHO &amp; EPA compliant
    </span>
</div>
""", unsafe_allow_html=True)
