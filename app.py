"""Aurum — a premium, multilingual Streamlit chat UI for gold & silver.

Presentation layer over the tested RAG core (`rag.answer`). Adds:
  * a language switcher (English / हिन्दी / Hinglish / मारवाड़ी)
  * a futuristic dark + gold theme with animated shimmer, aura glow, and hover
    light-sweeps on the rate cards and chips
  * a live "today's bhav" rates strip + sidebar editor
"""
# --- Streamlit Cloud compatibility ------------------------------------------
# chromadb needs sqlite >= 3.35; Streamlit Cloud's system sqlite is older, so
# swap in pysqlite3 (installed only on Linux). Must run BEFORE chromadb imports.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
from datetime import date

import streamlit as st

# Bridge Streamlit Cloud secrets into env vars so os.getenv(...) works everywhere.
for _key in ("GROQ_API_KEY", "GOLDAPI_KEY"):
    try:
        if _key not in os.environ and _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
    except Exception:
        pass

import config
import vector_store as vs
from gold_api import GoldAPIError, refresh_rates
from llm import LLMError
from rag import answer
from rates import load_rates, save_rates

st.set_page_config(
    page_title=f"{config.APP_NAME} · {config.APP_TAGLINE}",
    page_icon="🪙",
    layout="centered",
    initial_sidebar_state="expanded",   # show the 🌐 language switcher + bhav editor on load
)

# --------------------------------------------------------------------------- #
# Localised UI strings
# --------------------------------------------------------------------------- #
UI = {
    "English": {
        "tagline": "Gold & Silver Intelligence",
        "chip": "RAG · Groq · grounded + cited",
        "try": "Try asking",
        "placeholder": "Ask Aurum about gold, silver, rates, making charges…",
        "bhav": "Today's bhav",
        "bhav_cap": "Set your shop's rates — Aurum uses these for rate & price answers.",
        "update": "💾 Update rates",
        "lang": "Language",
        "asof": "bhav as set on",
        "daily": "rates change daily",
        "examples": [
            "What's the price of a 10g 22K ring with 12% making charges?",
            "What is today's 22K gold rate?",
            "How much GST applies to gold jewellery?",
            "Is a Sovereign Gold Bond better than buying jewellery?",
            "What is 925 sterling silver?",
            "How do I verify a hallmark's HUID?",
        ],
    },
    "हिन्दी": {
        "tagline": "सोना और चाँदी की समझ",
        "chip": "RAG · Groq · स्रोत के साथ",
        "try": "ये पूछकर देखिए",
        "placeholder": "सोना, चाँदी, भाव, मेकिंग चार्ज के बारे में Aurum से पूछिए…",
        "bhav": "आज का भाव",
        "bhav_cap": "अपनी दुकान का भाव सेट करें — Aurum इसी से भाव और कीमत बताएगा।",
        "update": "💾 भाव अपडेट करें",
        "lang": "भाषा",
        "asof": "भाव सेट किया गया",
        "daily": "भाव रोज़ बदलते हैं",
        "examples": [
            "10 ग्राम 22K अंगूठी की कीमत 12% मेकिंग चार्ज के साथ क्या होगी?",
            "आज 22K सोने का भाव क्या है?",
            "सोने के गहनों पर कितना GST लगता है?",
            "क्या सॉवरेन गोल्ड बॉन्ड गहनों से बेहतर है?",
            "925 स्टर्लिंग सिल्वर क्या है?",
            "हॉलमार्क का HUID कैसे जाँचें?",
        ],
    },
    "Hinglish": {
        "tagline": "Gold & Silver Intelligence",
        "chip": "RAG · Groq · sources ke saath",
        "try": "Try karke dekho",
        "placeholder": "Aurum se gold, silver, bhav, making charges ke baare mein poochho…",
        "bhav": "Aaj ka bhav",
        "bhav_cap": "Apni shop ka bhav set karo — Aurum isi se rate aur price batayega.",
        "update": "💾 Bhav update karo",
        "lang": "Bhasha",
        "asof": "bhav set kiya",
        "daily": "bhav roz badalte hain",
        "examples": [
            "10g 22K ring ki price 12% making charge ke saath kitni hogi?",
            "Aaj 22K gold ka bhav kya hai?",
            "Gold jewellery par kitna GST lagta hai?",
            "Kya Sovereign Gold Bond jewellery se better hai?",
            "925 sterling silver kya hota hai?",
            "Hallmark ka HUID kaise verify karein?",
        ],
    },
    "मारवाड़ी": {
        "tagline": "सोनो अर चाँदी री समझ",
        "chip": "RAG · Groq · स्रोत सूं",
        "try": "ओ पूछ'र देखो",
        "placeholder": "सोनो, चाँदी, भाव, मेकिंग चार्ज बाबत Aurum सूं पूछो…",
        "bhav": "आज रो भाव",
        "bhav_cap": "आपरी दुकान रो भाव सेट करो — Aurum इणी सूं भाव अर कीमत बतावैला।",
        "update": "💾 भाव अपडेट करो",
        "lang": "भाषा",
        "asof": "भाव सेट कर्यो",
        "daily": "भाव रोज़ बदलै",
        "examples": [
            "10 ग्राम 22K री अंगूठी री कीमत 12% मेकिंग चार्ज सूं कितरी होसी?",
            "आज 22K सोना रो भाव कांई है?",
            "सोना रा गहणा माथै कितरो GST लागै?",
            "सॉवरेन गोल्ड बॉन्ड गहणा सूं चोखो है कांई?",
            "925 स्टर्लिंग चाँदी कांई है?",
            "हॉलमार्क रो HUID कियां जाँचां?",
        ],
    },
}

# --------------------------------------------------------------------------- #
# Styling — premium dark + gold, with animation & hover
# --------------------------------------------------------------------------- #
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Outfit:wght@300;400;500;600&display=swap');

:root{
  --gold-1:#FFE39A; --gold-2:#E6B422; --gold-3:#B8860B; --gold-soft:#fff6da;
  --bg-0:#070A10; --bg-1:#0E121C; --glass:rgba(255,255,255,.045);
  --stroke:rgba(230,180,34,.22); --text:#EAECEF; --muted:#9AA3B2;
}

/* Dark base + gold glow baked straight into the app background — one solid,
   reliable layer (no transparency, no separate ::before, no z-index hacks). */
.stApp{
  background:
    radial-gradient(620px 320px at 14% 16%, rgba(230,180,34,.13), transparent 60%),
    radial-gradient(760px 380px at 86% 0%, rgba(255,227,154,.10), transparent 60%),
    radial-gradient(700px 500px at 50% 120%, rgba(184,134,11,.10), transparent 60%),
    linear-gradient(180deg,var(--bg-0),var(--bg-1));
  background-attachment:fixed;
  color:var(--text); font-family:'Outfit',sans-serif;
}
.block-container{ padding-top:2rem; max-width:840px; }
#MainMenu, footer { display:none; }
header[data-testid="stHeader"]{ background:transparent; }
/* The sidebar open button lives in the toolbar — DON'T hide the toolbar.
   Make that button an obvious gold ☰ control in the top-left. */
[data-testid="stExpandSidebarButton"]{ visibility:visible !important; opacity:1 !important; }
[data-testid="stExpandSidebarButton"] button{
  background:linear-gradient(92deg,var(--gold-1),var(--gold-2)) !important;
  color:#1a1205 !important; border:none !important; border-radius:10px !important;
  box-shadow:0 4px 18px rgba(230,180,34,.55) !important;
}

/* Hero */
.aurum-hero{ text-align:center; margin-bottom:.3rem; }
.aurum-logo{
  font-family:'Sora',sans-serif; font-weight:800; font-size:3.4rem; letter-spacing:.04em;
  background:linear-gradient(92deg,var(--gold-1),var(--gold-soft) 28%,var(--gold-2) 52%,var(--gold-3) 72%,var(--gold-1));
  background-size:220% auto; -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
  animation:shine 6s linear infinite, glowPulse 4.5s ease-in-out infinite;
}
@keyframes shine{ to{ background-position:220% center; } }
@keyframes glowPulse{ 0%,100%{filter:drop-shadow(0 0 14px rgba(230,180,34,.28));} 50%{filter:drop-shadow(0 0 28px rgba(230,180,34,.55));} }
.aurum-tag{ color:var(--muted); font-size:1.0rem; letter-spacing:.2em; text-transform:uppercase; margin-top:-.15rem;}
.aurum-chip{
  display:inline-block; margin-top:.7rem; padding:.3rem .85rem; border:1px solid var(--stroke);
  border-radius:999px; color:var(--gold-1); font-size:.72rem; letter-spacing:.1em; background:var(--glass);
  transition:.25s ease;
}
.aurum-chip:hover{ border-color:var(--gold-2); box-shadow:0 0 18px rgba(230,180,34,.25); }
.gold-rule{ height:1px; margin:1.1rem 0 .2rem;
  background:linear-gradient(90deg,transparent,rgba(230,180,34,.55),transparent); }

/* Rates strip */
.rates-row{ display:flex; gap:.6rem; margin:.4rem 0 .3rem; flex-wrap:wrap; }
.rate-card{
  flex:1; min-width:120px; padding:.75rem .95rem; border-radius:16px; position:relative; overflow:hidden;
  border:1px solid var(--stroke); background:linear-gradient(180deg,var(--glass),rgba(255,255,255,.01));
  backdrop-filter:blur(8px); transition:transform .25s ease, box-shadow .25s ease, border-color .25s ease;
}
.rate-card::after{
  content:""; position:absolute; top:0; left:-160%; width:65%; height:100%;
  background:linear-gradient(120deg,transparent,rgba(255,227,154,.28),transparent);
  transform:skewX(-20deg); transition:left .6s ease;
}
.rate-card:hover{ transform:translateY(-5px); border-color:var(--gold-2);
  box-shadow:0 12px 30px rgba(230,180,34,.20), 0 0 0 1px rgba(230,180,34,.28); }
.rate-card:hover::after{ left:160%; }
.rate-card .k{ color:var(--muted); font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; }
.rate-card .v{ font-family:'Sora',sans-serif; font-weight:700; font-size:1.3rem; color:var(--gold-1);
  text-shadow:0 0 16px rgba(230,180,34,.25); }
.rate-card .u{ color:var(--muted); font-size:.7rem; }
.rates-asof{ text-align:right; color:var(--muted); font-size:.72rem; margin-bottom:.3rem; }

/* Chat bubbles */
[data-testid="stChatMessage"]{
  background:var(--glass); border:1px solid rgba(255,255,255,.06); border-radius:18px;
  padding:.45rem .65rem; backdrop-filter:blur(6px); margin-bottom:.25rem; transition:border-color .25s ease;
}
[data-testid="stChatMessage"]:hover{ border-color:rgba(230,180,34,.28); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]){
  border-left:2px solid var(--gold-2); box-shadow:-6px 0 22px -14px rgba(230,180,34,.6);
}

/* Buttons / chips */
.stButton>button{
  background:var(--glass); color:var(--gold-1); border:1px solid var(--stroke); border-radius:999px;
  padding:.5rem .95rem; font-size:.86rem; font-weight:500; width:100%;
  transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease, background .18s ease, color .18s ease;
}
.stButton>button:hover{
  background:linear-gradient(92deg,var(--gold-1),var(--gold-2)); color:#1a1205; border-color:transparent;
  transform:translateY(-2px); box-shadow:0 8px 26px rgba(230,180,34,.32);
}
.stButton>button:active{ transform:translateY(0); }

/* Chat input */
[data-testid="stChatInput"]{
  border:1px solid var(--stroke); border-radius:16px; background:rgba(14,18,28,.85);
  backdrop-filter:blur(10px); transition:box-shadow .25s ease, border-color .25s ease;
}
[data-testid="stChatInput"]:focus-within{ border-color:var(--gold-2); box-shadow:0 0 0 1px var(--gold-2),0 0 26px rgba(230,180,34,.22); }
[data-testid="stChatInput"] textarea{ color:var(--text); }

/* Sidebar */
[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0A0E16,#0B0F18); border-right:1px solid rgba(255,255,255,.05); }
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{ color:var(--gold-1); font-family:'Sora',sans-serif; }

/* Segmented control (language) */
[data-testid="stSegmentedControl"] button:hover{ color:var(--gold-1); }

/* Expander */
[data-testid="stExpander"]{ border:1px solid rgba(255,255,255,.07); border-radius:14px; background:var(--glass); }
.src-pill{ color:var(--gold-1); }

/* Gold scrollbar + selection */
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-thumb{ background:linear-gradient(var(--gold-3),var(--gold-2)); border-radius:8px; }
::-webkit-scrollbar-track{ background:transparent; }
::selection{ background:rgba(230,180,34,.30); color:#fff; }

/* MOBILE: a fixed, transform-animated full-screen layer + backdrop-filter make
   iOS Safari render the whole page blurry. Drop those GPU-heavy effects on small
   screens — the static gold look stays crisp. */
@media (max-width: 820px){
  .stApp::before{ animation:none !important; }
  .aurum-logo{ animation:none !important; filter:none !important; font-size:2.5rem !important; }
  .rate-card, [data-testid="stChatMessage"], [data-testid="stChatInput"],
  .aurum-chip{ backdrop-filter:none !important; -webkit-backdrop-filter:none !important; }
  .rate-card .v{ text-shadow:none !important; }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Warming up the knowledge index…")
def _warm_up() -> int:
    return vs.build_index()


def render_rates_strip(rates: dict, t: dict) -> None:
    pg = rates["per_gram"]
    cards = [("24K Gold", pg["gold_24k"]), ("22K Gold", pg["gold_22k"]),
             ("18K Gold", pg["gold_18k"]), ("Silver", pg["silver"])]
    html = '<div class="rates-row">'
    for label, val in cards:
        html += (f'<div class="rate-card"><div class="k">{label}</div>'
                 f'<div class="v">₹{val:,}</div><div class="u">per gram</div></div>')
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    st.markdown(
        f'<div class="rates-asof">{t["asof"]} <b>{rates.get("updated","—")}</b> '
        f'· {rates.get("city","India")} · {t["daily"]}</div>',
        unsafe_allow_html=True,
    )


def ask(question: str, language: str) -> None:
    st.session_state.history.append({"role": "user", "content": question})
    with st.spinner("…"):
        try:
            r = answer(question, language=language)
            st.session_state.history.append(
                {"role": "assistant", "content": r.text, "sources": r.sources})
        except LLMError as e:
            st.session_state.history.append(
                {"role": "assistant", "content": f"⚠️ {e}", "sources": {}})


# --------------------------------------------------------------------------- #
# Sidebar — language + today's bhav
# --------------------------------------------------------------------------- #
def _auto_sync_rates() -> None:
    """Pull live rates from GoldAPI once per session (if the key is set and the
    cached rates aren't already today's live values)."""
    if not os.getenv("GOLDAPI_KEY") or st.session_state.get("rates_synced"):
        return
    st.session_state.rates_synced = True
    cur = load_rates()
    today = date.today().isoformat()
    if cur.get("source") == "GoldAPI.io" and str(cur.get("updated", "")).startswith(today):
        return
    try:
        refresh_rates()
    except GoldAPIError as e:
        st.session_state.rates_warn = str(e)


_auto_sync_rates()
rates = load_rates()
with st.sidebar:
    st.markdown(f"## {config.APP_NAME}")
    language = st.segmented_control(
        "🌐 Language / भाषा", list(UI.keys()), default="English", key="lang"
    ) or "English"
    t = UI[language]
    st.divider()
    st.subheader(t["bhav"])
    st.caption(t["bhav_cap"])
    if st.button("🔄 Fetch live rates (GoldAPI)", use_container_width=True):
        try:
            refresh_rates()
            st.success("Live rates updated.")
        except GoldAPIError as e:
            st.error(str(e))
        st.rerun()
    if st.session_state.get("rates_warn"):
        st.caption(f"⚠️ live fetch: {st.session_state.rates_warn}")
    pg = rates["per_gram"]
    g24 = st.number_input("24K gold (₹/g)", value=int(pg["gold_24k"]), step=10)
    g22 = st.number_input("22K gold (₹/g)", value=int(pg["gold_22k"]), step=10)
    g18 = st.number_input("18K gold (₹/g)", value=int(pg["gold_18k"]), step=10)
    slv = st.number_input("Silver (₹/g)", value=int(pg["silver"]), step=1)
    city = st.text_input("City", value=rates.get("city", "India"))
    if st.button(t["update"], use_container_width=True):
        save_rates({
            "updated": date.today().isoformat(), "city": city, "currency": "INR",
            "per_gram": {"gold_24k": g24, "gold_22k": g22, "gold_18k": g18, "silver": slv},
        })
        st.success("✓")
        st.rerun()
    st.divider()
    st.caption(f"backend `{config.LLM_BACKEND}` · `{config.GROQ_MODEL}`  \n"
               "⚠️ Knowledge base is an educational sample — verify BIS/GST details.")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
<div class="aurum-hero">
  <div class="aurum-logo">{config.APP_NAME} 🪙</div>
  <div class="aurum-tag">{t["tagline"]}</div>
  <div class="aurum-chip">{t["chip"]}</div>
</div>
<div class="gold-rule"></div>
""",
    unsafe_allow_html=True,
)

_warm_up()
render_rates_strip(rates, t)

if "history" not in st.session_state:
    st.session_state.history = []
if "pending" not in st.session_state:
    st.session_state.pending = None

if not st.session_state.history:
    st.markdown(f"###### {t['try']}")
    cols = st.columns(2)
    for i, ex in enumerate(t["examples"]):
        if cols[i % 2].button(ex, key=f"ex{i}"):
            st.session_state.pending = ex

typed = st.chat_input(t["placeholder"])
if st.session_state.pending:
    ask(st.session_state.pending, language)
    st.session_state.pending = None
elif typed:
    ask(typed, language)

for msg in st.session_state.history:
    avatar = "🪙" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 Sources"):
                for n, src in msg["sources"].items():
                    st.markdown(f'<span class="src-pill">**[{n}]** &nbsp; `{src}`</span>',
                                unsafe_allow_html=True)
