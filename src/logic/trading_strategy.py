"""
Trading Strategy Module.
Evaluates market conditions using confluence between RSI, MACD, and Price Action
to generate actionable trading signals.
"""

from typing import Tuple
import pandas as pd
import config


def evaluate_market_signal(latest_bar: pd.Series) -> Tuple[str, str]:
    """
    Analyzes the most recent market data bar to determine the current strategy status.

    Args:
        latest_bar: A pandas Series containing 'RSI', 'MACD_Hist', 'Close', 'MA20', and 'MA50'.

    Returns:
        A tuple containing the signal string (e.g., "STRONG BUY") and a hex color code.
    """
    settings = config.IndicatorSettings

    # 1. Extract Indicator Values (C1: Articulate Naming)
    relative_strength_index = latest_bar.get('RSI')
    macd_histogram = latest_bar.get('MACD_Hist')
    current_close = latest_bar.get('Close')
    moving_average_20 = latest_bar.get('MA20')

    # 2. Safety Check (P3: Prescriptively-Failing)
    if relative_strength_index is None or macd_histogram is None:
        return "DATA INCOMPLETE", "#808495"

    # 3. Define Signal Conditions (C10: Explanatory Variables / C22: Natural conditions)
    is_rsi_oversold = relative_strength_index < settings.RSI_OVERSOLD_LEVEL
    is_rsi_overbought = relative_strength_index > settings.RSI_OVERBOUGHT_LEVEL
    is_macd_bullish = macd_histogram > 0
    is_price_above_trend = current_close > moving_average_20

    # 4. Decision Logic (C2: Straightforward / S3: Assertive)
    # Strong Buy: RSI is oversold AND MACD shows bullish momentum
    if is_rsi_oversold and is_macd_bullish:
        return "STRONG BUY", "#00FF41"

    # Strong Sell: RSI is overbought AND MACD shows bearish momentum
    if is_rsi_overbought and not is_macd_bullish:
        return "STRONG SELL", "#FF3131"

    # Warning/Caution: Price is drifting below trend despite neutral RSI
    if not is_price_above_trend and relative_strength_index < 50:
        return "CAUTION: WEAK TREND", "#FFD700"

    return "NEUTRAL / HOLD", "#808495"