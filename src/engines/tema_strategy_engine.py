"""
TEMA Strategy Engine.
Implements the Triple Exponential Moving Average (TEMA) algorithm to
forecast trend direction and generate smoothed price paths.
"""

import pandas as pd

# Step 1 (C2/C5/#21/#49): remove magic values + unused imports; make choices explicit
CLOSE_COLUMN = "Close"
PREDICTED_COLUMN = "Predicted"

TEMA_SPAN = 20
VELOCITY_LOOKBACK_PERIODS = 5

DEFAULT_FALLBACK_FREQUENCY = "D"


def generate_tema_forecast(
    market_dataframe: pd.DataFrame,
    forecast_steps: int = 30
) -> pd.DataFrame:
    """
        Description:
            Generates a price forecast by calculating the Triple Exponential Moving
            Average (TEMA) and projecting its current velocity into the future.

        Args:
            market_dataframe (pd.DataFrame): Data containing a 'Close' price column
                and a DatetimeIndex.
            forecast_steps (int): The number of future periods to predict.

        Returns:
            pd.DataFrame: A single-column DataFrame ('Predicted') containing the
                last actual price followed by the projected price path.

        Example:
            >>> forecast = generate_tema_forecast(market_data, forecast_steps=10)
        """
    forecast_results = pd.DataFrame()

    if not market_dataframe.empty:
        if CLOSE_COLUMN not in market_dataframe.columns:
            raise KeyError(f"Missing required column '{CLOSE_COLUMN}' in market_dataframe columns.")

        # 1. Calculate TEMA components
        close_prices = market_dataframe[CLOSE_COLUMN].copy()  # #18: avoid side effects on shared data

        ema1 = close_prices.ewm(span=TEMA_SPAN, adjust=False).mean()
        ema2 = ema1.ewm(span=TEMA_SPAN, adjust=False).mean()
        ema3 = ema2.ewm(span=TEMA_SPAN, adjust=False).mean()

        tema = (3 * ema1) - (3 * ema2) + ema3

        # 2. Calculate the "Trend Velocity"
        latest_tema = tema.iloc[-1]
        effective_lookback = min(VELOCITY_LOOKBACK_PERIODS, len(tema) - 1)  # #68: avoid IndexError on short series
        if effective_lookback < 1:
            tema_velocity = 0.0
        else:
            previous_tema = tema.iloc[-effective_lookback - 1]
            tema_velocity = (latest_tema - previous_tema) / effective_lookback

        # 3. Project Future Path
        latest_price = float(close_prices.iloc[-1])

        steps_index = pd.RangeIndex(1, forecast_steps + 1)
        forecast_path = (latest_price + tema_velocity * steps_index).to_list()

        # 4. Generate Timestamps
        future_dates = _calculate_future_indices(market_dataframe, forecast_steps)

        # 5. Assemble Result
        forecast_results = pd.concat([
            pd.DataFrame({PREDICTED_COLUMN: [latest_price]}, index=[market_dataframe.index[-1]]),
            pd.DataFrame({PREDICTED_COLUMN: forecast_path}, index=future_dates)
        ])

    return forecast_results


def _calculate_future_indices(df: pd.DataFrame, steps: int) -> pd.DatetimeIndex:
    """
        Description:
            Computes future timestamps based on the existing frequency of the
            input dataframe's index.

        Args:
            df (pd.DataFrame): Input dataframe with a DatetimeIndex.
            steps (int): Number of future periods to generate.

        Returns:
            pd.DatetimeIndex: Index containing future timestamps.

        Example:
            >>> future_index = _calculate_future_indices(df, 30)
        """
    if len(df) < 2:
        return pd.date_range(
            start=df.index[-1],
            periods=steps + 1,
            freq=DEFAULT_FALLBACK_FREQUENCY
        )[1:]

    delta = df.index[-1] - df.index[-2]
    return pd.DatetimeIndex([df.index[-1] + (i * delta) for i in range(1, steps + 1)])

