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
from src.ui import styles
from src.engines import tema_strategy_engine
from src.logic import trading_strategy



def _get_history_period(interval_code: str) -> str:
    """
    Description:
        Determines the appropriate yfinance lookback period (e.g., '60d', '10y',
        'max') based on the provided time interval.

    Args:
        time_interval_code (str): The interval string used for the data request
            (e.g., '1m', '1h', '1wk', '1d').

    Returns:
        str: A string representing the historical lookback period required
            by the yfinance API.

    Example:
        >>> _get_history_period("1wk")
        '10y'
        >>> _get_history_period("1m")
        '60d'
    """
    if interval_code in config.INTRADAY_INTERVAL_CODES:
        return "60d"
    if interval_code == "1wk":
        return "10y"
    return "max"


def _normalize_market_dataframe(market_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Description:
        Cleans a raw yfinance DataFrame by handling MultiIndex columns, filling
        missing values, and filtering for the required OHLCV columns.

    Args:
        market_dataframe (pd.DataFrame): The raw DataFrame retrieved from yfinance,
            which may contain MultiIndex columns or missing price data.

    Returns:
        pd.DataFrame: A cleaned DataFrame containing only 'Open', 'High', 'Low',
            'Close', and 'Volume' columns with no missing values.

    Example:
        >>> raw_data = yf.download("AAPL")
        >>> clean_data = _normalize_market_dataframe(raw_data)
    """
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
    interval_code = config.AVAILABLE_TIMEFRAMES.get(timeframe_label, config.DEFAULT_DATA_INTERVAL)  #: avoid magic string


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
    """
        Description:
            Enhances the input market dataframe by calculating and appending
            technical overlays and oscillators. This includes Moving Averages (20, 50),
            Bollinger Bands, Relative Strength Index (RSI), and MACD Histogram.
            It also generates boolean strategy signals based on pre-defined thresholds.

        Args:
            dataframe (pd.DataFrame): A pandas DataFrame containing at least a
                'Close' price column and chronological index.

        Returns:
            pd.DataFrame: A copy of the input DataFrame enriched with technical
                indicator columns and 'Buy_Signal'/'Sell_Signal' booleans.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'Close': [100, 102, 101, 105]})
            >>> enriched_df = _enrich_with_technical_indicators(df)
        """
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
    """
        Description:
            Calculates the percentage return of an asset by comparing the current price
            against a historical closing price from a specific number of trading days ago.

        Args:
            current_price (float): The most recent live or closing price of the asset.
            history_dataframe (pd.DataFrame): The historical market data containing a
                'Close' column.
            lookback_days (int): The number of trading sessions to look back for the
                base price.

        Returns:
            float: The percentage return (e.g., 5.0 for a 5% gain). Returns 0.0 if the
                dataframe contains insufficient historical data.

        Example:
            >>> import pandas as pd
            >>> hist = pd.DataFrame({'Close': [100.0, 105.0, 110.0]})
            >>> _calculate_period_return(115.0, hist, 2)
            15.0
        """
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
    Description:
        Computes key performance indicators for the asset dashboard, including
        short-term returns (Weekly, Monthly), Year-to-Date (YTD) performance,
        and the annualized volatility based on a 30-day lookback.

    Args:
        current_price (float): The most recent market price of the asset.
        history_df (pd.DataFrame): Historical price data containing the 'Close' column.
        ytd_start_price (float): The opening price of the asset at the start of the year.

    Returns:
        Tuple[float, float, float, float]: A tuple containing:
            - Weekly Return (Percentage)
            - Monthly Return (Percentage)
            - YTD Return (Percentage)
            - Annualized Volatility (Percentage)

    Example:
        >>> metrics = calculate_performance_metrics(2500.0, df, 2300.0)
    """
    # 1. Risk Metrics: Annualized Volatility (30-day window)
    # We use .tail(31) to get 30 returns after the initial pct_change NaN
    daily_returns = history_df["Close"].pct_change().tail(30)

    # Safety Check: Handle cases where price is flat or data is missing
    if daily_returns.empty or daily_returns.std() == 0:
        annualized_volatility = 0.0
    else:
        # Standard deviation of returns scaled by the square root of trading sessions
        annualized_volatility = (
                daily_returns.std() * (config.LayoutSettings.TRADING_DAYS_PER_YEAR ** 0.5) * 100
        )

    # 2. Return Metrics
    weekly_return = _calculate_period_return(current_price, history_df, 5)
    monthly_return = _calculate_period_return(current_price, history_df, 21)

    # YTD Calculation
    if ytd_start_price and ytd_start_price != 0:
        year_to_date_return = ((current_price - ytd_start_price) / ytd_start_price) * 100
    else:
        year_to_date_return = 0.0

    return weekly_return, monthly_return, year_to_date_return, annualized_volatility


def _fetch_year_start_price(fallback_df: pd.DataFrame, ticker: str) -> float:
    """
    Description:
        Attempts to download the market opening price for the first trading day
        of the current calendar year using the yfinance API. If the network
        request fails or returns empty data, it defaults to the earliest
        available price in the provided fallback dataframe.

    Args:
        fallback_dataframe (pd.DataFrame): A pandas DataFrame containing historical
            price data to be used if the YTD fetch fails.
        ticker (str): The financial ticker symbol to query (e.g., 'AAPL', 'BTC-USD').

    Returns:
        float: The closing price from the start of the year or the fallback
            period. Returns 0.0 if both the fetch and the fallback fail.

    Example:
        >>> import pandas as pd
        >>> historical_data = pd.DataFrame({'Close': [150.0, 155.0]})
        >>> _fetch_year_start_price(historical_data, "AAPL")
        182.45
    """
    try:
        current_year = datetime.now().year
        ytd_data = yf.download(ticker, start=f"{current_year}-01-01", progress=False)

        if ytd_data.empty:
            raise ValueError(f"No YTD data returned for ticker='{ticker}'")

        if isinstance(ytd_data.columns, pd.MultiIndex):
            ytd_data.columns = ytd_data.columns.get_level_values(0)

        return float(ytd_data["Close"].iloc[0])

    except (IndexError, KeyError, ValueError) as error:
        # Fallback to earliest available data in provided dataframe
        if fallback_df.empty:
            return 0.0
        return float(fallback_df["Close"].iloc[0])



def _fetch_asset_news(ticker: str) -> List[Dict[str, Any]]:
    """
        Description:
            Queries the yfinance Search API to retrieve the most recent news articles
            associated with a specific financial ticker. The function limits the result
            set to a maximum of five items to ensure dashboard performance.

        Args:
            ticker (str): The financial ticker symbol to research (e.g., 'TSLA', 'MSFT').

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains
                metadata for a news story, such as 'title', 'publisher', and 'link'.
                Returns an empty list if the API request fails.

        Example:
            >>> news_items = _fetch_asset_news("AAPL")
            >>> print(news_items[0]['title'])
            'Apple Reaches New All-Time High'
        """
    try:
        search_engine = yf.Search(ticker, news_count=5)
        return search_engine.news
    except Exception as exc:
        return []

def calculate_market_signals(df: pd.DataFrame, current_price: float) -> Dict:
    """
        Description:
            Transforms raw price action and technical indicators into a dictionary of
            actionable market signals. It evaluates the current market regime (Bullish/Bearish),
            calculates the gap to local resistance levels, generates a TEMA-based price
            target, and retrieves the primary trading strategy signal.

        Args:
            market_dataframe (pd.DataFrame): The technical dataset containing 'High',
                'Moving_Average_20', and 'Moving_Average_50' columns.
            current_price (float): The most recent market price of the asset.

        Returns:
            Dict: A dictionary containing:
                - 'regime' (str): "BULLISH" or "BEARISH" text label.
                - 'regime_color' (str): HEX/Color code for UI rendering.
                - 'resistance_gap' (float): Percentage distance to the recent high.
                - 'target_price' (float): The forecasted TEMA price.
                - 'upside_pct' (float): Potential percentage gain to target.
                - 'strategy_signal' (str): The specific action (Buy/Sell/Hold).
                - 'strategy_color' (str): Visual indicator color for the signal.

        Example:
            >>> import pandas as pd
            >>> data = pd.DataFrame({'High': [105], 'Moving_Average_20': [100], 'Moving_Average_50': [95]})
            >>> calculate_market_signals(data, 102.5)
            {'regime': 'BULLISH', 'resistance_gap': 2.439, ...}
        """
    latest = df.iloc[-1]

    # 1. Regime Logic
    is_above_ma20 = current_price > latest['Moving_Average_20']
    is_ma20_above_ma50 = latest['Moving_Average_20'] > latest['Moving_Average_50']
    is_bullish = is_above_ma20 and is_ma20_above_ma50

    # We define these specifically so the dictionary below can find them
    regime_text = "BULLISH" if is_bullish else "BEARISH"
    regime_color = styles.SUCCESS_COLOR if is_bullish else styles.DANGER_COLOR

    # 2. Resistance Gap Logic
    resistance_lookback = config.LayoutSettings.RESISTANCE_LOOKBACK_DAYS
    peak_price = df['High'].tail(resistance_lookback).max()

    # We define this as 'resistance_gap' to match the dictionary key
    resistance_gap = ((peak_price - current_price) / current_price) * 100

    #TEMA Logic
    prediction = tema_strategy_engine.generate_tema_forecast(df)
    target_price = prediction['Predicted'].iloc[-1] if not prediction.empty else 0
    upside = ((target_price - current_price) / current_price) * 100 if target_price > 0 else 0

    #Strategy Logic
    strategy_signal, strategy_color = trading_strategy.evaluate_market_signal(df)

    return {
        "regime": regime_text,
        "regime_color": regime_color,
        "resistance_gap": resistance_gap,
        "target_price": target_price,
        "upside_pct": upside,
        "strategy_signal": strategy_signal,
        "strategy_color": strategy_color
    }


def prepare_header_metrics(name: str, price: float, df: pd.DataFrame, performance: Tuple) -> Dict:
    """
        Description:
            Standardizes asset metadata and calculates the daily price delta for
            the dashboard header.

        Args:
            name (str): Asset name with ticker suffix.
            price (float): Current asset price.
            market_dataframe (pd.DataFrame): Historical data for delta calculation.
            performance_metrics (Tuple): (Weekly, Monthly, YTD, Volatility) returns.

        Returns:
            Dict[str, Any]: Formatted header data for UI metrics.

        Example:
            >>> prepare_header_metrics("Gold (GC=F)", 2000.0, df, (1.0, 2.0, 5.0, 0.5))
        """
    # Calculation: Clean Name
    clean_name = name.split(' (')[0]

    # Calculation: Daily Delta
    yesterday_close = df['Close'].iloc[-2] if len(df) > 1 else price
    daily_delta = ((price - yesterday_close) / yesterday_close) * 100

    return {
        "display_name": clean_name,
        "daily_delta": daily_delta,
        "weekly": performance[0],
        "monthly": performance[1],
        "ytd": performance[2],
        "volatility": performance[3]
    }

def calculate_fundamental_snapshot(df: pd.DataFrame) -> Dict:
    """
        Description:
            Extracts 52-week price range metrics and calculates the asset's current
            relative position within that range.

        Args:
            market_dataframe (pd.DataFrame): Historical price data with 'High', 'Low', and 'Close'.

        Returns:
            Dict[str, float]: 52-week high, low, percentage position, and normalized progress.

        Example:
            >>> calculate_fundamental_snapshot(market_dataframe)
            {'high_52w': 150.0, 'low_52w': 100.0, 'range_pos_pct': 50.0, 'range_pos_normalized': 0.5}
        """
    # 1. High/Low Calculations
    high_52w = df['High'].tail(config.LayoutSettings.TRADING_DAYS_PER_YEAR).max()
    low_52w = df['Low'].tail(config.LayoutSettings.TRADING_DAYS_PER_YEAR).min()
    current = df['Close'].iloc[-1]

    # 2. Percentage Logic
    price_range = high_52w - low_52w
    range_pos_pct = ((current - low_52w) / price_range) * 100 if price_range != 0 else 0

    # 3. Normalized value for st.progress (0.0 to 1.0)
    normalized_progress = min(max(range_pos_pct / 100, 0.0), 1.0)

    return {
        "high_52w": high_52w,
        "low_52w": low_52w,
        "range_pos_pct": range_pos_pct,
        "range_pos_normalized": normalized_progress
    }