"""
UI Styles Module.
Manages global CSS injection and reusable custom Streamlit components
for the Multi-Asset Terminal.
"""
from typing import Any
import streamlit as st

# --- CONSTANTS (C21: No magic variables) ---
DARK_BACKGROUND_COLOR = "#0E1117"
SECONDARY_UI_COLOR = "#1E222D"
BORDER_COLOR = "#363A45"
TEXT_SUBDUED = "#808495"
SUCCESS_COLOR = "#00FF41"
DANGER_COLOR = "#FF3131"
WARNING_COLOR = "#FFA500"


def inject_terminal_stylesheet() -> None:
    """
    Injects global CSS into the Streamlit app to standardize
    the look and feel of the terminal dashboard.
    """
    st.markdown(f"""
        <style>
        .main {{ background-color: {DARK_BACKGROUND_COLOR}; }}
        [data-testid="stMetricLabel"] {{ 
            color: {TEXT_SUBDUED} !important; 
        }}
        [data-testid="stMetricValue"] {{ 
            color: white !important; 
        }}
        .sidebar-header-branding {{ 
            color: white !important; 
            font-size: 28px !important; 
            font-weight: bold; 
            text-align: center; 
            margin-bottom: 20px; 
            border-bottom: 2px solid gold; 
            padding-bottom: 10px; 
        }}
        .status-card {{ 
            background-color: {SECONDARY_UI_COLOR}; 
            padding: 20px; 
            border-radius: 10px; 
            border: 1px solid {BORDER_COLOR}; 
            margin-bottom: 15px; 
        }}
        .sentiment-item {{ 
            color: #FFFFFF !important; 
            text-decoration: none !important; 
            display: block; 
            padding: 8px; 
            border-bottom: 1px solid {BORDER_COLOR}; 
            margin-bottom: 5px; 
            font-size: 15px; 
        }}
        .sentiment-item:hover {{ 
            background-color: {SECONDARY_UI_COLOR}; 
            color: #FFD700 !important; 
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
    """
    Renders a custom-styled metric card with dynamic color-coding
    based on performance direction.
    """
    # 1. Inference Logic (S37: Inference-logic separation)
    if is_volatility:
        display_color = WARNING_COLOR
    elif numeric_delta > 0:
        display_color = SUCCESS_COLOR
    else:
        display_color = DANGER_COLOR
    # 2. UI Rendering (C31: Multi-line logic)
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
    """
    Renders a status box with a value and a colored status badge
    used in the Intelligence Center sidebar.
    """
    st.markdown(f"""
        <div class="status-card">
            <div style='color:white; font-size:16px; font-weight:bold; margin-bottom:5px;'>
                {signal_title}
            </div>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span style='color:white; font-size:26px; font-weight:bold;'>
                    {primary_value}
                </span>
                <span style='background-color:{signal_color}; color:black; padding:2px 10px; 
                             border-radius:5px; font-weight:bold; font-size:12px;'>
                    {status_badge_text}
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)
