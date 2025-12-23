from typing import Dict, Final, List

CLOSE_COLUMN: Final[str] = "Close"
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M"

# Mapping short names with full names
ASSET_MAPPING: Final[Dict[str, str]] = {
    "Gold (GC=F)": "GC=F",
    "S&P 500 (^GSPC)": "^GSPC",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD"
}
#Setting the default Dashboard view
DEFAULT_SYMBOL: Final[str] = "GC=F"
DEFAULT_DATA_INTERVAL: Final[str] = "1d"
INTRADAY_INTERVAL_CODES: Final[List[str]] = ["15m", "1h"]

#Timeframe mapping
AVAILABLE_TIMEFRAMES: Final[Dict[str, str]] = {
    "15 Minutes": "15m",
    "1 Hour": "1h",
    "1 Day": "1d",
    "1 Week": "1wk",
    "1 Month": "1mo",
}

#Technical indicator setup
class IndicatorSettings:

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


#Interface setup
class LayoutSettings:
    MAIN_CHART_HEIGHT: int = 500
    INDICATOR_CHART_HEIGHT: int = 150
    LOGO_WIDTH: int = 160

    # Column Ratios (Streamlit st.columns)
    LOGO_COLUMN_RATIO: List[float] = [0.1, 0.9]
    DASHBOARD_MAIN_RATIO: List[float] = [0.72, 0.28]
    CHART_HEADER_RATIO: List[float] = [0.5, 0.2, 0.3]

    # Scaling, Buffers & Limits
    SIGNAL_CHART_BUFFER_PERCENT: float = 0.02
    MAX_NEWS_ITEMS: int = 5

    # Calculation & Lookback Windows
    TRADING_DAYS_PER_YEAR: int = 252
    RESISTANCE_LOOKBACK_DAYS: int = 20
    CHART_LOOKBACK_PERIODS: int = 150