"""
Engine responsible for retrieving market data, calculating technical indicators,
and processing performance metrics.
"""

from datetime import datetime
from typing import List, Tuple, Dict, Any

import pandas as pd
import yfinance as yf
import streamlit as st

import config



def _get_history_period(interval_code: str) -> str:
    """Determine yfinance lookback period for a given interval code."""
    if interval_code in config.INTRADAY_INTERVAL_CODES:
        return "60d"
    if interval_code == "1wk":
        return "10y"
    return "max"


def _normalize_market_dataframe(market_dataframe: pd.DataFrame) -> pd.DataFrame:
    """ Clean yfinance dataframe and keep required OHLCV columns."""
    if market_dataframe.empty:
        return market_dataframe

    # yfinance sometimes returns MultiIndex columns (e.g., when grouping tickers)
    if isinstance(market_dataframe.columns, pd.MultiIndex):
        market_dataframe = market_dataframe.copy()  # #18: avoid mutating external ref
        market_dataframe.columns = market_dataframe.columns.get_level_values(0)

    market_dataframe = market_dataframe.ffill().dropna()
    return market_dataframe[["Open", "High", "Low", "Close", "Volume"]]


@st.cache_data(ttl=60)
def fetch_market_dashboard_data(
        timeframe_label: str = "1 Day",
        ticker_symbol: str = "GC=F"
) -> Tuple[pd.DataFrame, float, List[Dict[str, Any]], float]:
    """
    Retrieves and prepares all necessary data for the asset dashboard.

    Args:
        timeframe_label: Human-readable timeframe (e.g., "1 Hour").
        ticker_symbol: The market symbol to download.

    Returns:
        A tuple containing the processed DataFrame, current price, news list, and YTD start price.
    """
    interval_code = config.AVAILABLE_TIMEFRAMES.get(timeframe_label, config.DEFAULT_INTERVAL_CODE)  #: avoid magic string


    # Define data lookback period based on interval granularity
    data_history_period = _get_history_period(interval_code)  # Step 3: extracted helper

    market_dataframe = yf.download(
        tickers=ticker_symbol,
        period=data_history_period,
        interval=interval_code,
        auto_adjust=False
    )

    if market_dataframe.empty:
        return pd.DataFrame(), 0.0, [], 0.0

    # Clean multi-index columns from yfinance (C5: Well-formatted)
    market_dataframe = _normalize_market_dataframe(market_dataframe)  # Step 3B
    if market_dataframe.empty:
        return pd.DataFrame(), 0.0, [], 0.0

    market_dataframe = _enrich_with_technical_indicators(market_dataframe)

    current_market_price = float(market_dataframe['Close'].iloc[-1])
    market_news = _fetch_asset_news(ticker_symbol)
    year_to_date_price = _fetch_year_start_price(market_dataframe, ticker_symbol)

    return market_dataframe, current_market_price, market_news, year_to_date_price


def _enrich_with_technical_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculates technical overlays and oscillators."""
    settings = config.IndicatorSettings
    dataframe = dataframe.copy()  # #18 Immutability: avoid mutating caller dataframe

    # 1. Trend Indicators (Moving Averages)
    dataframe['Moving_Average_20'] = dataframe['Close'].rolling(window=settings.BOLLINGER_BANDS_PERIOD).mean()
    dataframe['Moving_Average_50'] = dataframe['Close'].rolling(window=settings.MOVING_AVERAGE_LONG_TERM).mean()

    # 2. Volatility (Bollinger Bands)
    rolling_standard_deviation = dataframe['Close'].rolling(window=settings.BOLLINGER_BANDS_PERIOD).std()
    dataframe['Bollinger_Bands_Upper'] = dataframe['Moving_Average_20'] + (rolling_standard_deviation * settings.BOLLINGER_BANDS_STANDARD_DEVIATIONS)
    dataframe['Bollinger_Bands_Lower'] = dataframe['Moving_Average_20'] - (rolling_standard_deviation * settings.BOLLINGER_BANDS_STANDARD_DEVIATIONS)

    # 3. Momentum (Relative Strength Index)
    price_delta = dataframe['Close'].diff()
    average_gain = (price_delta.where(price_delta > 0, 0)).rolling(window=settings.RSI_PERIOD).mean()
    average_loss = (-price_delta.where(price_delta < 0, 0)).rolling(window=settings.RSI_PERIOD).mean()

    relative_strength = average_gain / (average_loss + 1e-10)
    dataframe['RSI'] = 100 - (100 / (1 + relative_strength))

    # 4. Trend Strength (MACD)
    exponential_ma_12 = dataframe['Close'].ewm(span=12, adjust=False).mean()
    exponential_ma_26 = dataframe['Close'].ewm(span=26, adjust=False).mean()
    macd_line = exponential_ma_12 - exponential_ma_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    dataframe['Moving_Average_Convergence_Divergence_Histogram'] = macd_line - signal_line

    # 5. Strategic Signal Assignment (C2: Straightforward logic)
    dataframe['Buy_Signal'] = (dataframe['RSI'] < settings.RSI_BULLISH_THRESHOLD) & (dataframe['Moving_Average_Convergence_Divergence_Histogram'] > 0)
    dataframe['Sell_Signal'] = (dataframe['RSI'] > settings.RSI_BEARISH_THRESHOLD) & (dataframe['Moving_Average_Convergence_Divergence_Histogram'] < 0)

    return dataframe


def _calculate_period_return(
        current_price: float,
        history_df: pd.DataFrame,
        lookback_days: int
) -> float:
    """Compute percentage return over lookback trading days."""
    if len(history_df) > lookback_days:
        previous_price = history_df["Close"].iloc[-lookback_days]
        return ((current_price - previous_price) / previous_price) * 100
    return 0.0


def calculate_performance_metrics(
        current_price: float,
        history_df: pd.DataFrame,
        ytd_start_price: float
) -> Tuple[float, float, float, float]:
    """
    Computes returns and volatility metrics for the dashboard header.

    Returns:
        Tuple: (Weekly Return, Monthly Return, YTD Return, Annualized Volatility)
    """
    # Risk Metrics: Annualized Volatility (30-day window)
    daily_returns = history_df["Close"].pct_change().tail(30)
    annualized_volatility = daily_returns.std() * (252 ** 0.5) * 100

    # Return Metrics
    weekly_return = _calculate_period_return(current_price, history_df, 5)   # #16
    monthly_return = _calculate_period_return(current_price, history_df, 21) # #16
    year_to_date_return = ((current_price - ytd_start_price) / ytd_start_price) * 100 if ytd_start_price else 0.0

    return weekly_return, monthly_return, year_to_date_return, annualized_volatility


def _fetch_year_start_price(fallback_df: pd.DataFrame, ticker: str) -> float:
    """Attempts to fetch price from Jan 1st of current year."""
    try:
        current_year = datetime.now().year
        ytd_data = yf.download(ticker, start=f"{current_year}-01-01", progress=False)

        if ytd_data.empty:
            raise ValueError(f"No YTD data returned for ticker='{ticker}'")

        if isinstance(ytd_data.columns, pd.MultiIndex):
            ytd_data.columns = ytd_data.columns.get_level_values(0)

        return float(ytd_data["Close"].iloc[0])

    except (IndexError, KeyError, ValueError) as error:
        # #68: clear exception + safe fallback
        # Fallback to earliest available data in provided dataframe
        if fallback_df.empty:
            return 0.0
        return float(fallback_df["Close"].iloc[0])



def _fetch_asset_news(ticker: str) -> List[Dict[str, Any]]:
    """Retrieves recent news items for the specific ticker."""
    try:
        search_engine = yf.Search(ticker, news_count=5)
        return search_engine.news
    except Exception as exc:
        # #68 Clear exceptions: keep behavior; optionally log `exc` for debugging.
        return []

