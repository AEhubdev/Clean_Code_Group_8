"""
Configuration module for the Multi-Asset Terminal.
Defines technical indicators, asset mappings, and UI constants.
"""

from typing import Dict, Final

# --- ASSET CONFIGURATION ---
# C1: Accurate names instead of abbreviations like GC=F
ASSET_MAPPING: Final[Dict[str, str]] = {
    "Gold (GC=F)": "GC=F",
    "S&P 500 (^GSPC)": "^GSPC",
    "Bitcoin (BTC-USD)": "BTC-USD"
}

DEFAULT_SYMBOL: Final[str] = "GC=F"
DEFAULT_DATA_INTERVAL: Final[str] = "1d"


# --- TECHNICAL INDICATOR PARAMETERS ---
# C1: Full names (Bollinger Bands instead of BB)
# C2: Straightforward groupings
class IndicatorSettings:
    """Standardized periods and thresholds for technical analysis indicators."""

    # Relative Strength Index (RSI)
    RSI_PERIOD: int = 14
    RSI_OVERBOUGHT_LEVEL: int = 70
    RSI_OVERSOLD_LEVEL: int = 30
    RSI_BULLISH_THRESHOLD: int = 55
    RSI_BEARISH_THRESHOLD: int = 45

    # Bollinger Bands
    BOLLINGER_BANDS_PERIOD: int = 20
    BOLLINGER_BANDS_STANDARD_DEVIATIONS: int = 2

    # Moving Averages
    MOVING_AVERAGE_SHORT_TERM: int = 20
    MOVING_AVERAGE_LONG_TERM: int = 50


# --- USER INTERFACE SETTINGS ---
class LayoutSettings:
    """Constants for chart dimensions and display options."""
    MAIN_CHART_HEIGHT: int = 500
    INDICATOR_CHART_HEIGHT: int = 150


# --- TIMEFRAME MAPPINGS ---
# C1: Full names (Timeframes instead of TF)
AVAILABLE_TIMEFRAMES: Final[Dict[str, str]] = {
    "15 Minutes": "15m",
    "1 Hour": "1h",
    "1 Day": "1d",
    "1 Week": "1wk",
    "1 Month": "1mo",
}