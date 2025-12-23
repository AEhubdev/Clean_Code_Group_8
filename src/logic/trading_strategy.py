import pandas as pd
import config


def evaluate_market_signal(df: pd.DataFrame) -> tuple[str, str]:
    """
    Description:
        Analyzes a 10-bar window of technical indicators (RSI, MACD, and Moving Average)
        to determine a momentum-based trading signal. It looks for confluence between
        recent extremes and current price action.

    Args:
        df (pd.DataFrame): Market dataframe containing 'RSI', 'Close', 'Moving_Average_20',
            and 'Moving_Average_Convergence_Divergence_Histogram' columns.

    Returns:
        tuple[str, str]: A tuple containing the signal string (e.g., 'STRONG BUY')
            and a corresponding HEX color code for UI rendering.

    Example:
        >>> signal, color = evaluate_market_signal(market_dataframe)
        >>> print(f"Action: {signal}")
    """

    # Guard clause: Ensure we have enough data for a 10-bar analysis
    if len(df) < 10:
        return "WAITING FOR DATA", "#808495"

    settings = config.IndicatorSettings

    # 1. Define the 10-bar window
    window = df.tail(10)
    latest = window.iloc[-1]

    # 2. Extract current values
    rsi_now = latest.get('RSI')
    macd_now = latest.get('Moving_Average_Convergence_Divergence_Histogram')
    close_now = latest.get('Close')
    Moving_Average_20_now = latest.get('Moving_Average_20')

    if rsi_now is None or macd_now is None:
        return "DATA INCOMPLETE", "#808495"

    # 3. Sequence Logic: Check if conditions were met ANYWHERE in the last 10 bars
    # This filters out 'noisy' signals by looking for recent extremes
    was_recently_oversold = (window['RSI'] < settings.RSI_OVERSOLD_LEVEL).any()
    was_recently_overbought = (window['RSI'] > settings.RSI_OVERBOUGHT_LEVEL).any()

    # 4. Confluence: Current momentum must confirm the recent extreme
    is_momentum_confirmed_up = macd_now > 0
    is_momentum_confirmed_down = macd_now < 0
    is_above_trend = close_now > Moving_Average_20_now

    # --- Decision Logic ---

    # STRONG BUY: Was recently 'cheap' (oversold) and momentum just turned positive
    if was_recently_oversold and is_momentum_confirmed_up:
        return "STRONG BUY", "#00FF41"

    # STRONG SELL: Was recently 'expensive' (overbought) and momentum just turned negative
    if was_recently_overbought and is_momentum_confirmed_down:
        return "STRONG SELL", "#FF3131"

    # CAUTION: Price is breaking trend (below Moving_Average_20) while RSI is weak
    if not is_above_trend and rsi_now < 50:
        return "CAUTION: WEAK TREND", "#FFD700"

    return "NEUTRAL / HOLD", "#808495"