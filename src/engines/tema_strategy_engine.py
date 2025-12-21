"""
TEMA Strategy Engine.
Implements the Triple Exponential Moving Average (TEMA) algorithm to
forecast trend direction and generate smoothed price paths.
"""

from typing import Tuple
import pandas as pd
import numpy as np

def generate_tema_forecast(
    market_dataframe: pd.DataFrame,
    forecast_steps: int = 30
) -> pd.DataFrame:
    """
    Generates a forecast based on the Triple Exponential Moving Average trend.

    TEMA = (3 * EMA1) - (3 * EMA2) + EMA3
    This provides a much faster response to price changes than a standard SMA.
    """
    if market_dataframe.empty:
        return pd.DataFrame()

    # 1. Calculate TEMA components
    # We use a 20-period window for the primary trend
    close_prices = market_dataframe['Close']

    ema1 = close_prices.ewm(span=20, adjust=False).mean()
    ema2 = ema1.ewm(span=20, adjust=False).mean()
    ema3 = ema2.ewm(span=20, adjust=False).mean()

    tema = (3 * ema1) - (3 * ema2) + ema3

    # 2. Calculate the "Trend Velocity"
    # We look at the slope of the TEMA over the last 5 periods
    latest_tema = tema.iloc[-1]
    previous_tema = tema.iloc[-5]
    tema_velocity = (latest_tema - previous_tema) / 5

    # 3. Project Future Path
    latest_price = float(close_prices.iloc[-1])

    # The forecast follows the TEMA trajectory
    forecast_path = []
    for step in range(1, forecast_steps + 1):
        # We blend the current price with the projected TEMA velocity
        projected_point = latest_price + (tema_velocity * step)
        forecast_path.append(projected_point)

    # 4. Generate Timestamps
    future_dates = _calculate_future_indices(market_dataframe, forecast_steps)

    # 5. Assemble Result
    forecast_results = pd.concat([
        pd.DataFrame({'Predicted': [latest_price]}, index=[market_dataframe.index[-1]]),
        pd.DataFrame({'Predicted': forecast_path}, index=future_dates)
    ])

    return forecast_results

def _calculate_future_indices(df: pd.DataFrame, steps: int) -> pd.DatetimeIndex:
    """Computes future timestamps based on existing frequency."""
    if len(df) < 2:
        return pd.date_range(start=df.index[-1], periods=steps + 1, freq='D')[1:]

    delta = df.index[-1] - df.index[-2]
    return [df.index[-1] + (i * delta) for i in range(1, steps + 1)]