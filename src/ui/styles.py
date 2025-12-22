from typing import Any
import streamlit as st

DARK_BACKGROUND_COLOR = "#0E1117"
SECONDARY_UI_COLOR = "#1E222D"
BORDER_COLOR = "#363A45"
TEXT_SUBDUED = "#808495"
SUCCESS_COLOR = "#00FF41"
DANGER_COLOR = "#FF3131"
WARNING_COLOR = "#FFA500"
HOLD_COLOR = "#00D4FF"
RSI_COLOR = "#BB86FC"
MACD_COLOR = "#FF8C00"

# Fixed Hex Codes for consistency
COLOR_WHITE = "#FFFFFF"
COLOR_GOLD = "#FFD700"

BB_LINE_COLOR = 'rgba(173, 216, 230, 0.2)'
BB_FILL_COLOR = 'rgba(173, 216, 230, 0.05)'

def inject_terminal_stylesheet() -> None:
    st.markdown(f"""
        <style>
        /* Main Container */
        .main {{ background-color: {DARK_BACKGROUND_COLOR}; }}

        /* Metric Styling */
        [data-testid="stMetricLabel"] {{ color: {TEXT_SUBDUED} !important; }}
        [data-testid="stMetricValue"] {{ color: {COLOR_WHITE} !important; }}

        /* Sidebar & Branding */
        .sidebar-header-branding {{ 
            color: {COLOR_WHITE} !important; 
            font-size: 28px !important; 
            font-weight: bold; 
            text-align: center; 
            margin-bottom: 20px; 
            border-bottom: 2px solid {COLOR_GOLD}; 
            padding-bottom: 10px; 
        }}

        /* Intelligence & News Cards */
        .status-card {{ 
            background-color: {SECONDARY_UI_COLOR}; 
            padding: 20px; 
            border-radius: 10px; 
            border: 1px solid {BORDER_COLOR}; 
            margin-bottom: 15px; 
        }}

        .sentiment-item {{ 
            color: {COLOR_WHITE} !important; 
            text-decoration: none !important; 
            display: block; 
            padding: 8px; 
            border-bottom: 1px solid {BORDER_COLOR}; 
            margin-bottom: 5px; 
            font-size: 15px; 
        }}

        .sentiment-item:hover {{ 
            background-color: {SECONDARY_UI_COLOR}; 
            color: {COLOR_GOLD} !important; 
        }}
        </style>
    """, unsafe_allow_html=True)


def render_colored_performance_metric(
        streamlit_column: Any,
        metric_label: str,
        metric_display_value: str,
        numeric_delta: float,
        is_volatility: bool = False
) -> None:


    # Ternary Logic for concise color picking
    if is_volatility:
        display_color = WARNING_COLOR
    else:
        display_color = SUCCESS_COLOR if numeric_delta > 0 else DANGER_COLOR

    streamlit_column.markdown(f"**{metric_label}**")
    streamlit_column.markdown(
        f"<h2 style='color:{display_color}; margin-top:-15px; font-weight:bold;'>"
        f"{metric_display_value}</h2>",
        unsafe_allow_html=True
    )


def render_intelligence_signal(
        signal_title: str,
        primary_value: str,
        status_badge_text: str,
        signal_color: str
) -> None:
    #Renders a status card with a value and a colored badge

    st.markdown(f"""
        <div class="status-card">
            <div style='color:{COLOR_WHITE}; font-size:16px; font-weight:bold; margin-bottom:5px;'>
                {signal_title}
            </div>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span style='color:{COLOR_WHITE}; font-size:26px; font-weight:bold;'>
                    {primary_value}
                </span>
                <span style='background-color:{signal_color}; color:black; padding:2px 10px; 
                             border-radius:5px; font-weight:bold; font-size:12px;'>
                    {status_badge_text}
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)
