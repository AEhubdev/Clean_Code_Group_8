"""
Risk Engine Logic.
Calculates risk-adjusted performance metrics like Sharpe Ratio and Drawdown.
"""
import pandas as pd


def calculate_risk_metrics(df: pd.DataFrame) -> dict:
    """
    Calculates Sharpe Ratio and Max Drawdown from price data.

    Args:
        df (pd.DataFrame): Dataframe containing a 'Close' column.

    Returns:
        dict: Dictionary containing 'sharpe' and 'maximum_drawdown' values.
    """
    if df.empty or 'Close' not in df.columns:
        return {"sharpe": 0.0, "maximum_drawdown": 0.0}

    returns = df['Close'].pct_change().dropna()

    # Annualized Sharpe Ratio (Simplified)
    if returns.std() != 0:
        sharpe = (returns.mean() / returns.std()) * (252 ** 0.5)
    else:
        sharpe = 0.0

    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    maximum_drawdown = drawdown.min() * 100

    return {
        "sharpe": sharpe,
        "maximum_drawdown": maximum_drawdown
    }