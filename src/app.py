"""WareIQ — friendly warehouse sales assistant.

Run from the project root:
    streamlit run src/app.py
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.agent import run_agent
from src.analysis_tools import (
    get_items_ranked,
    get_sales_by_month,
    get_top_suppliers,
    get_total_sales,
)

st.set_page_config(
    page_title="WareIQ",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# Design system
# ─────────────────────────────────────────────────────────────
CREAM = "#FBFAF3"
WHITE = "#FFFFFF"
INK = "#3E3A37"
MUTED = "#756E68"
TEAL = "#63C8BE"
TEAL_DARK = "#3F9E92"
LIME = "#B7D84B"
LIME_SOFT = "#EDF5CF"
YELLOW = "#F4C95D"
CORAL = "#F06F5E"
PURPLE = "#7657A8"
BORDER = "#E7E1D8"
SHADOW = "0 12px 32px rgba(70, 62, 55, .10)"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: "DM Sans", sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(circle at 90% 5%, rgba(183,216,75,.18), transparent 24rem),
            radial-gradient(circle at 5% 20%, rgba(99,200,190,.16), transparent 25rem),
            {CREAM};
        color: {INK};
    }}

    .block-container {{
        max-width: 1180px;
        padding-top: 1.4rem;
        padding-bottom: 6rem;
    }}

    #MainMenu, footer, header {{
        visibility: hidden;
    }}

    .brand-row {{
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:1rem;
        margin-bottom:1rem;
    }}

    .brand {{
        display:flex;
        align-items:center;
        gap:.7rem;
        font-family:"Manrope",sans-serif;
        font-size:1.25rem;
        font-weight:800;
        color:{INK};
    }}

    .brand-mark {{
        width:42px;
        height:42px;
        display:flex;
        align-items:center;
        justify-content:center;
        border-radius:14px;
        background:linear-gradient(135deg,{TEAL},{LIME});
        box-shadow:0 8px 22px rgba(63,158,146,.24);
        animation:floatMark 3.8s ease-in-out infinite;
    }}

    .status {{
        display:inline-flex;
        align-items:center;
        gap:.45rem;
        padding:.48rem .8rem;
        border-radius:999px;
        background:{WHITE};
        border:1px solid {BORDER};
        color:{MUTED};
        font-size:.78rem;
        font-weight:700;
        box-shadow:0 4px 14px rgba(70,62,55,.05);
    }}

    .status-dot {{
        width:8px;
        height:8px;
        border-radius:50%;
        background:{LIME};
        box-shadow:0 0 0 5px rgba(183,216,75,.17);
    }}

    .hero {{
        background:linear-gradient(135deg, rgba(255,255,255,.96), rgba(237,245,207,.78));
        border:1px solid rgba(183,216,75,.38);
        border-radius:28px;
        padding:2.1rem;
        box-shadow:{SHADOW};
        overflow:hidden;
        position:relative;
        animation:fadeUp .55s ease both;
    }}

    .hero:after {{
        content:"";
        position:absolute;
        right:-60px;
        top:-90px;
        width:240px;
        height:240px;
        border-radius:50%;
        background:rgba(99,200,190,.18);
    }}

    .hero-kicker {{
        color:{TEAL_DARK};
        font-size:.78rem;
        letter-spacing:.09em;
        text-transform:uppercase;
        font-weight:800;
    }}

    .hero-title {{
        font-family:"Manrope",sans-serif;
        font-size:clamp(2.1rem,5vw,4rem);
        line-height:1.03;
        letter-spacing:-.055em;
        max-width:780px;
        margin:.55rem 0 .8rem;
        color:{INK};
    }}

    .hero-copy {{
        max-width:740px;
        color:{MUTED};
        font-size:1.02rem;
        line-height:1.7;
        margin:0;
    }}

    .simple-tabs {{
        margin:1.2rem 0 .8rem;
    }}

    div[role="radiogroup"] {{
        gap:.6rem;
        background:rgba(255,255,255,.74);
        border:1px solid {BORDER};
        border-radius:18px;
        padding:.42rem;
        width:fit-content;
        box-shadow:0 6px 20px rgba(70,62,55,.06);
    }}

    div[role="radiogroup"] label {{
        padding:.55rem .95rem !important;
        border-radius:13px !important;
        transition:all .22s ease;
        font-weight:700;
    }}

    div[role="radiogroup"] label:hover {{
        transform:translateY(-2px);
        background:{LIME_SOFT};
    }}

    .welcome-card {{
        background:{WHITE};
        border:1px solid {BORDER};
        border-radius:24px;
        padding:1.45rem;
        box-shadow:{SHADOW};
        margin-top:1rem;
        animation:fadeUp .6s ease both;
    }}

    .bot-row {{
        display:flex;
        gap:.9rem;
        align-items:flex-start;
    }}

    .bot-avatar {{
        width:44px;
        height:44px;
        border-radius:15px;
        background:linear-gradient(135deg,{TEAL},{LIME});
        display:flex;
        align-items:center;
        justify-content:center;
        flex:0 0 auto;
        font-size:1.2rem;
        box-shadow:0 7px 18px rgba(63,158,146,.22);
    }}

    .bot-copy {{
        color:{INK};
        line-height:1.65;
        font-size:.97rem;
    }}

    .bot-copy strong {{
        font-family:"Manrope",sans-serif;
    }}

    .label {{
        margin:1.35rem 0 .6rem;
        color:{MUTED};
        font-size:.8rem;
        font-weight:800;
        letter-spacing:.05em;
        text-transform:uppercase;
    }}

    div.stButton > button {{
        width:100%;
        border:1px solid rgba(99,200,190,.35);
        border-radius:16px;
        background:{WHITE};
        color:{INK};
        font-weight:700;
        min-height:52px;
        box-shadow:0 6px 18px rgba(70,62,55,.06);
        transition:transform .2s ease, box-shadow .2s ease, border-color .2s ease;
    }}

    div.stButton > button:hover {{
        transform:translateY(-3px) scale(1.01);
        border-color:{TEAL};
        color:{INK};
        box-shadow:0 12px 26px rgba(63,158,146,.18);
    }}

    div.stButton > button:active {{
        transform:translateY(0) scale(.99);
    }}

    .kpi {{
        background:{WHITE};
        border:1px solid {BORDER};
        border-radius:20px;
        padding:1.05rem;
        min-height:136px;
        box-shadow:0 8px 24px rgba(70,62,55,.07);
        transition:transform .22s ease, box-shadow .22s ease;
        animation:fadeUp .5s ease both;
    }}

    .kpi:hover {{
        transform:translateY(-5px);
        box-shadow:0 16px 30px rgba(70,62,55,.12);
    }}

    .kpi-icon {{
        font-size:1.45rem;
    }}

    .kpi-label {{
        color:{MUTED};
        font-size:.78rem;
        font-weight:700;
        margin-top:.45rem;
    }}

    .kpi-value {{
        font-family:"Manrope",sans-serif;
        font-size:1.55rem;
        font-weight:800;
        color:{INK};
        margin-top:.15rem;
    }}

    .kpi-note {{
        color:{MUTED};
        font-size:.73rem;
        margin-top:.22rem;
    }}

    [data-testid="stChatMessage"] {{
        border:0;
        background:transparent;
        padding:.25rem 0;
    }}

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {{
        background:{WHITE};
        border:1px solid {BORDER};
        border-radius:18px;
        padding:.85rem 1rem;
        box-shadow:0 5px 18px rgba(70,62,55,.06);
        line-height:1.65;
    }}

    [data-testid="stChatInput"] {{
        background:{WHITE};
        border:2px solid rgba(99,200,190,.45);
        border-radius:20px;
        box-shadow:0 10px 28px rgba(63,158,146,.14);
    }}

    [data-testid="stChatInput"]:focus-within {{
        border-color:{TEAL_DARK};
    }}

    .story-card {{
        background:{WHITE};
        border:1px solid {BORDER};
        border-radius:22px;
        padding:1.25rem;
        box-shadow:0 8px 24px rgba(70,62,55,.07);
        animation:fadeUp .5s ease both;
    }}

    .story-title {{
        font-family:"Manrope",sans-serif;
        font-size:1.1rem;
        font-weight:800;
        margin-bottom:.25rem;
    }}

    .story-copy {{
        color:{MUTED};
        line-height:1.6;
        font-size:.9rem;
    }}

    .proof-step {{
        background:{WHITE};
        border:1px solid {BORDER};
        border-radius:18px;
        padding:1rem;
        min-height:125px;
        box-shadow:0 6px 18px rgba(70,62,55,.06);
        transition:all .22s ease;
    }}

    .proof-step:hover {{
        transform:translateY(-4px);
        border-color:{LIME};
    }}

    .proof-num {{
        color:{TEAL_DARK};
        font-size:.75rem;
        font-weight:800;
    }}

    .proof-title {{
        font-family:"Manrope",sans-serif;
        font-weight:800;
        margin:.3rem 0;
    }}

    .proof-copy {{
        color:{MUTED};
        font-size:.79rem;
        line-height:1.48;
    }}

    @keyframes fadeUp {{
        from {{ opacity:0; transform:translateY(14px); }}
        to {{ opacity:1; transform:translateY(0); }}
    }}

    @keyframes floatMark {{
        0%,100% {{ transform:translateY(0) rotate(0deg); }}
        50% {{ transform:translateY(-4px) rotate(3deg); }}
    }}

    @media (max-width: 760px) {{
        .hero {{ padding:1.45rem; border-radius:22px; }}
        .brand-row {{ align-items:flex-start; }}
        .status {{ display:none; }}
        .block-container {{ padding-left:1rem; padding-right:1rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────
def read_json(path: str) -> dict[str, Any]:
    target = PROJECT_ROOT / path
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def money(value: float, compact: bool = True) -> str:
    value = float(value or 0)
    if compact and abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if compact and abs(value) >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:,.0f}"


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, Any]:
    total = get_total_sales()
    suppliers = get_top_suppliers(10)
    monthly = get_sales_by_month()
    items = get_items_ranked()
    metrics = read_json("models/metrics.json")
    experiment = read_json("models/ab_results.json")

    warehouse_total = float(total.iloc[0]["total_warehouse_sales"])
    retail_total = float(total.iloc[0]["total_retail_sales"])
    top = suppliers.iloc[0]
    peak = monthly.loc[monthly["total_warehouse_sales"].idxmax()]

    return {
        "warehouse_total": warehouse_total,
        "retail_total": retail_total,
        "suppliers": suppliers,
        "monthly": monthly,
        "items": items,
        "top_supplier": str(top["SUPPLIER"]),
        "top_revenue": float(top["total_warehouse_sales"]),
        "top_share": float(top["total_warehouse_sales"]) / warehouse_total * 100,
        "peak_month": int(peak["MONTH"]),
        "peak_revenue": float(peak["total_warehouse_sales"]),
        "accuracy": float(metrics.get("xgb_accuracy", 0)),
        "summer_opportunity": float(experiment.get("annual_opportunity", 0)),
        "seasonal": experiment.get("seasonal_items", []),
    }


@st.cache_data(show_spinner=False)
def load_forecast() -> pd.DataFrame | None:
    try:
        from src.forecast import forecast_future, load_and_prepare, train_prophet

        history = load_and_prepare()
        model = train_prophet(history)
        return forecast_future(model, periods=6).tail(6).copy()
    except Exception:
        return None


try:
    DATA = load_data()
except Exception as exc:
    st.error("I could not open the warehouse data yet. Run the data pipeline, then refresh this page.")
    with st.expander("Developer details"):
        st.code(str(exc))
    st.stop()


MONTHS = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


# ─────────────────────────────────────────────────────────────
# Conversation behavior
# ─────────────────────────────────────────────────────────────
WELCOME = (
    "Hi! 👋 I’m **WareIQ**, your friendly warehouse assistant.\n\n"
    "Ask me about sales, suppliers, busy months, seasonal demand, or future estimates. "
    "I’ll explain everything in simple language and show the numbers behind the answer."
)


def casual_reply(message: str) -> str | None:
    text = re.sub(r"[^\w\s']", "", message.lower()).strip()

    greetings = {"hi", "hello", "hey", "hii", "hiii", "good morning", "good evening", "good afternoon"}
    if text in greetings:
        return (
            "Hi! 👋 Welcome to WareIQ.\n\n"
            "I’m ready to help you understand warehouse sales in simple language. "
            "What would you like to explore?"
        )

    if any(phrase in text for phrase in ["how are you", "howre you", "how r you"]):
        return (
            "I’m doing great, thanks for asking! 😊\n\n"
            "I’m ready to explore your warehouse data whenever you are."
        )

    if text in {"thanks", "thank you", "thankyou", "thx"}:
        return "You’re welcome! 😊 What would you like to explore next?"

    if any(phrase in text for phrase in ["what can you do", "help me", "how can you help"]):
        return (
            "I can help you understand:\n\n"
            "• which suppliers sold the most\n"
            "• which months were busiest\n"
            "• what changed during summer\n"
            "• what future sales may look like\n"
            "• what the data quality checks found"
        )

    return None


def clean_answer(answer: str) -> str:
    if not answer:
        return "I could not find a clear answer. Please try asking in a different way."

    hidden_phrases = [
        r"(?i)\bI can call\b.*",
        r"(?i)\bI will call\b.*",
        r"(?i)\busing the \w+ tool\b",
        r"(?i)\btool executed\b.*",
        r"(?i)\bfunction returned\b.*",
        r"(?is)\n*Analysis basis:.*$",
        r"(?im)^Metric:\s*None\s*$",
        r"(?im)^Filters:\s*None\s*$",
        r"(?im)^Grouped by:\s*None\s*$",
        r"(?im)^Date range:\s*None\s*$",
        r"(?im)^Records analyzed:\s*None\s*$",
        r"(?im)^Method:\s*None\s*$",
    ]

    cleaned = str(answer)
    for pattern in hidden_phrases:
        cleaned = re.sub(pattern, "", cleaned)

    cleaned = cleaned.replace("generate_business_summary", "business summary")
    cleaned = cleaned.replace("get_top_suppliers", "supplier results")
    cleaned = cleaned.replace("get_items_ranked", "item results")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned or "I found the result, but I could not explain it clearly. Please try again."


def answer_question(question: str) -> str:
    friendly = casual_reply(question)
    if friendly:
        return friendly

    try:
        try:
            raw = run_agent(question, st.session_state.messages[:-1])
        except TypeError:
            raw = run_agent(question)
        return clean_answer(raw)
    except Exception:
        return (
            "I’m having trouble reaching the warehouse analysis right now. "
            "Please check the app connection and try again."
        )


# ─────────────────────────────────────────────────────────────
# Shared UI helpers
# ─────────────────────────────────────────────────────────────
def kpi(icon: str, label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{html.escape(label)}</div>
            <div class="kpi-value">{html.escape(value)}</div>
            <div class="kpi-note">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def base_chart(title: str, height: int = 360) -> dict[str, Any]:
    return {
        "title": {"text": title, "font": {"size": 16, "color": INK}},
        "height": height,
        "paper_bgcolor": WHITE,
        "plot_bgcolor": WHITE,
        "font": {"family": "DM Sans", "color": MUTED},
        "margin": {"l": 10, "r": 10, "t": 55, "b": 25},
        "hoverlabel": {"bgcolor": INK, "font_color": WHITE},
        "showlegend": False,
    }


# ─────────────────────────────────────────────────────────────
# Header + navigation
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="brand-row">
        <div class="brand">
            <div class="brand-mark">📦</div>
            <div>WareIQ</div>
        </div>
        <div class="status"><span class="status-dot"></span> Ready to help</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.radio(
    "Choose a view",
    ["💬 Assistant", "📖 Visual story", "✨ Project proof"],
    horizontal=True,
    label_visibility="collapsed",
    key="main_view",
)

# ─────────────────────────────────────────────────────────────
# Assistant
# ─────────────────────────────────────────────────────────────
if page == "💬 Assistant":
    st.markdown(
        """
        <section class="hero">
            <div class="hero-kicker">Your data, explained simply</div>
            <div class="hero-title">Ask a business question.<br>I’ll show the proof.</div>
            <p class="hero-copy">
                No dashboards to decode. No technical language. Ask about past sales,
                the latest available picture, or future demand.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    with cols[0]:
        kpi("💰", "Money earned", money(DATA["warehouse_total"]), "Warehouse sales from 2017–2020")
    with cols[1]:
        kpi("🏆", "Best-selling supplier", DATA["top_supplier"], f"{DATA['top_share']:.1f}% of warehouse sales")
    with cols[2]:
        kpi("📅", "Busiest month", MONTHS[DATA["peak_month"]], money(DATA["peak_revenue"]) + " historically")
    with cols[3]:
        kpi("🌞", "Extra summer opportunity", money(DATA["summer_opportunity"]), "Estimated annual difference")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": WELCOME}]

    st.markdown('<div class="label">Try asking</div>', unsafe_allow_html=True)
    q1, q2, q3 = st.columns(3)
    suggestions = [
        ("🏆 Which supplier sold the most?", q1),
        ("📅 Which month was the busiest?", q2),
        ("🔮 What may happen in the next 6 months?", q3),
    ]

    selected_question = None
    for index, (question, column) in enumerate(suggestions):
        with column:
            if st.button(question, key=f"suggestion_{index}"):
                selected_question = question.split(" ", 1)[1]

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="📦" if message["role"] == "assistant" else "🙂"):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask WareIQ anything about warehouse sales…")
    prompt = selected_question or prompt

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🙂"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="📦"):
            with st.spinner("Looking through the numbers…"):
                answer = answer_question(prompt)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

    if len(st.session_state.messages) > 1:
        if st.button("Start a new conversation"):
            st.session_state.messages = [{"role": "assistant", "content": WELCOME}]
            st.rerun()


# ─────────────────────────────────────────────────────────────
# Visual story
# ─────────────────────────────────────────────────────────────
elif page == "📖 Visual story":
    st.markdown(
        """
        <section class="hero">
            <div class="hero-kicker">A simple business story</div>
            <div class="hero-title">Past. Latest picture. Future.</div>
            <p class="hero-copy">
                Move through the story and see what the warehouse data proves at each stage.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    period = st.radio(
        "Story period",
        ["Past", "Latest picture", "Future"],
        horizontal=True,
        label_visibility="collapsed",
        key="story_period",
    )

    if period == "Past":
        monthly = DATA["monthly"].copy()
        monthly["month"] = monthly["MONTH"].map(MONTHS)

        st.markdown(
            f"""
            <div class="story-card">
                <div class="story-title">What happened?</div>
                <div class="story-copy">
                    The warehouse generated <strong>{money(DATA['warehouse_total'])}</strong>
                    in warehouse sales between 2017 and 2020.
                    <strong>{MONTHS[DATA['peak_month']]}</strong> was the busiest calendar month.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig = go.Figure(
            go.Scatter(
                x=monthly["month"],
                y=monthly["total_warehouse_sales"],
                mode="lines+markers",
                line={"color": TEAL_DARK, "width": 4, "shape": "spline"},
                marker={"size": 9, "color": LIME, "line": {"color": TEAL_DARK, "width": 2}},
                fill="tozeroy",
                fillcolor="rgba(99,200,190,.14)",
                hovertemplate="%{x}<br>Sales: $%{y:,.0f}<extra></extra>",
            )
        )
        fig.update_layout(**base_chart("Sales pattern across the calendar year"))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor=BORDER, tickprefix="$", separatethousands=True)
        st.plotly_chart(fig, use_container_width=True)

    elif period == "Latest picture":
        st.markdown(
            f"""
            <div class="story-card">
                <div class="story-title">What does the latest available data show?</div>
                <div class="story-copy">
                    The dataset ends in 2020, so this is not live warehouse data.
                    In the latest available picture, <strong>{html.escape(DATA['top_supplier'])}</strong>
                    was the highest-selling supplier and contributed
                    <strong>{DATA['top_share']:.1f}%</strong> of warehouse sales.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        suppliers = DATA["suppliers"].head(8).sort_values("total_warehouse_sales")
        fig = go.Figure(
            go.Bar(
                x=suppliers["total_warehouse_sales"],
                y=suppliers["SUPPLIER"],
                orientation="h",
                marker={"color": TEAL},
                hovertemplate="%{y}<br>Sales: $%{x:,.0f}<extra></extra>",
            )
        )
        fig.update_layout(**base_chart("Highest-selling suppliers"))
        fig.update_xaxes(showgrid=True, gridcolor=BORDER, tickprefix="$", separatethousands=True)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    else:
        future = load_forecast()

        if future is None or future.empty:
            st.warning("The future estimate is not available yet. Run the forecast step and refresh the app.")
        else:
            peak = future.loc[future["yhat"].idxmax()]
            st.markdown(
                f"""
                <div class="story-card">
                    <div class="story-title">What may happen next?</div>
                    <div class="story-copy">
                        Past patterns suggest the strongest upcoming month may be
                        <strong>{peak['ds'].strftime('%B %Y')}</strong>,
                        with about <strong>{money(peak['yhat'])}</strong> in warehouse sales.
                        This is an estimate, not a guaranteed result.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([future["ds"], future["ds"].iloc[::-1]]),
                    y=pd.concat([future["yhat_upper"], future["yhat_lower"].iloc[::-1]]),
                    fill="toself",
                    fillcolor="rgba(183,216,75,.22)",
                    line={"color": "rgba(0,0,0,0)"},
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=future["ds"],
                    y=future["yhat"],
                    mode="lines+markers",
                    line={"color": PURPLE, "width": 4, "shape": "spline"},
                    marker={"size": 9, "color": YELLOW, "line": {"color": PURPLE, "width": 2}},
                    hovertemplate="%{x|%b %Y}<br>Estimated sales: $%{y:,.0f}<extra></extra>",
                )
            )
            fig.update_layout(**base_chart("Six-month sales estimate"))
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor=BORDER, tickprefix="$", separatethousands=True)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("The shaded area shows a reasonable range around the estimate.")


# ─────────────────────────────────────────────────────────────
# Project proof
# ─────────────────────────────────────────────────────────────
else:
    st.markdown(
        """
        <section class="hero">
            <div class="hero-kicker">Behind the friendly experience</div>
            <div class="hero-title">Every answer has a traceable path.</div>
            <p class="hero-copy">
                This page shows how raw warehouse data became reliable business answers.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    lessons = {
        "Bring in the data": {
            "number": "01",
            "summary": "I started by understanding the data before thinking about models.",
            "title": "I thought machine learning would be the hardest part.",
            "body": """
It wasn’t.

At the beginning, I assumed the project would mainly be about choosing the right algorithm and improving accuracy. I expected XGBoost to be the center of the story.

But once I started looking at the 307,000+ transaction records, I realized the real work started much earlier. Before modeling anything, I had to understand what each field meant, how the records were created, and what business story was already present in the data.

That changed my approach completely. The model became the last step, not the first.

**What I learned:** Good analytics starts with curiosity about the data, not excitement about the algorithm.
"""
        },
        "Check the quality": {
            "number": "02",
            "summary": "I learned that data cleaning is part of the analysis, not just preparation.",
            "title": "I learned that clean data creates trustworthy results.",
            "body": """
The biggest surprise was finding an 83% data gap in 2018 and a suspicious USD 7,800 transaction.

Earlier, I treated cleaning like a checklist: remove nulls, fix types, continue. This project taught me that quality issues can completely change the business conclusion.

That is why I separated the data into Bronze, Silver, and Gold layers. The raw data stayed untouched, every correction was documented, and the final business-ready data could always be traced back.

**My mistake:** I originally thought cleaning was something to finish quickly before modeling.

**What I learned:** Validation is not a support task. It is how an analyst earns trust.
"""
        },
        "Prepare the numbers": {
            "number": "03",
            "summary": "I built reusable, traceable data layers instead of repeating calculations.",
            "title": "I stopped treating every question as a separate analysis.",
            "body": """
My early approach was to calculate each result directly inside a notebook. That worked once, but it was difficult to reuse and easy to make inconsistent.

I reorganized the project so the same verified sales, supplier, month, and category definitions could support the dashboard, chatbot, forecast, and reports.

This made the results easier to audit and prevented the chatbot, charts, and tables from showing different numbers for the same question.

**My mistake:** I initially focused on producing an answer, not building a reliable path to reproduce it.

**What I learned:** A strong analytics product needs reusable definitions, not one-time calculations.
"""
        },
        "Find patterns": {
            "number": "04",
            "summary": "I learned to prove patterns before turning them into recommendations.",
            "title": "I stopped making recommendations from assumptions.",
            "body": """
It felt obvious that summer sales were stronger. But obvious is not the same as proven.

I tested the pattern at multiple levels: transaction-level comparison, controlled analysis after removing the largest suppliers, monthly aggregation, category-level analysis, and effect size.

The important result was not only that summer was statistically different. It was that Beer and Liquor showed a seasonal pattern while Wine and Kegs did not.

**My mistake:** I first assumed one seasonal strategy would apply to every product.

**What I learned:** Segmentation is what turns a general pattern into a useful business decision.
"""
        },
        "Estimate the future": {
            "number": "05",
            "summary": "I treated forecasts as ranges for planning, not exact promises.",
            "title": "I learned that a forecast should communicate uncertainty.",
            "body": """
My first instinct was to focus on the predicted number. But one number can create false confidence.

I changed the forecast experience to show the expected value together with a reasonable lower and upper range. I also made the historical limitations visible, especially the large 2018 data gap.

The forecast now helps a user understand direction without pretending it knows the exact future.

**My mistake:** I initially treated the forecast output like a precise answer.

**What I learned:** A useful forecast explains uncertainty and clearly separates estimates from observed facts.
"""
        },
        "Explain it simply": {
            "number": "06",
            "summary": "I redesigned the product around a non-technical user, not around myself.",
            "title": "I realized dashboards do not automatically answer questions.",
            "body": """
My first interface was designed like a technical portfolio: dark colors, model accuracy, statistical terms, and several dashboard pages.

Then I asked who would actually use the product. A general business user does not want to decode Cohen’s d, RAG, SQL, or model internals. They want to know what happened, why it matters, and what they should review next.

That is why I made the chatbot the main experience and moved technical proof into a separate section.

**My mistake:** I originally built the interface for myself and other analysts.

**What I learned:** The best analytics product is not the one with the most charts. It is the one the intended user can understand and trust.
"""
        },
    }

    if "selected_lesson" not in st.session_state:
        st.session_state.selected_lesson = "Bring in the data"

    lesson_names = list(lessons.keys())
    rows = [lesson_names[:3], lesson_names[3:]]

    for row_index, row_names in enumerate(rows):
        columns = st.columns(3)
        for column, lesson_name in zip(columns, row_names):
            lesson = lessons[lesson_name]
            with column:
                button_label = f"{lesson['number']}  {lesson_name}\n\n{lesson['summary']}"
                if st.button(
                    button_label,
                    key=f"lesson_{row_index}_{lesson_name}",
                    use_container_width=True,
                    type="primary" if st.session_state.selected_lesson == lesson_name else "secondary",
                ):
                    st.session_state.selected_lesson = lesson_name
                    st.rerun()

    selected = lessons[st.session_state.selected_lesson]
    st.markdown(
        f"""
        <div class="story-card" style="margin-top:1.1rem;border-left:6px solid {TEAL};">
            <div class="proof-num">{selected['number']} · {html.escape(st.session_state.selected_lesson)}</div>
            <div class="story-title" style="margin-top:.4rem;">{html.escape(selected['title'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(selected["body"])

    st.markdown('<div class="label">Proof found before modeling</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        kpi("⚠️", "Major history gap", "83% missing", "A large 2018 gap was disclosed")
    with p2:
        kpi("🕵️", "Suspicious record", "$7,800", "Removed before trusted analysis")
    with p3:
        kpi("🌞", "Seasonal opportunity", money(DATA["summer_opportunity"]), "Supported by a summer comparison")

    with st.expander("See the technical implementation"):
        st.code(
            """src/
├── ingest.py
├── clean.py
├── features.py
├── train.py
├── forecast.py
├── ab_testing.py
├── analysis_tools.py
├── rag.py
├── agent.py
└── app.py""",
            language="text",
        )