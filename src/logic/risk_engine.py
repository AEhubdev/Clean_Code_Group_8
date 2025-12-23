import pandas as pd
import config


def calculate_risk_metrics(df: pd.DataFrame) -> dict:
    """
    Description:
        Calculates professional-grade risk metrics including the Sharpe Ratio
        (adjusted for the risk-free rate) and the Maximum Drawdown (as an
        absolute magnitude).

    Args:
        df (pd.DataFrame): Dataframe containing historical 'Close'
            prices used for return and volatility calculations.

    Returns:
        dict: A dictionary containing:
            - 'sharpe' (float): The annualized Sharpe Ratio representing excess
              return per unit of risk.
            - 'maximum_drawdown' (float): The absolute percentage value of the
              largest peak-to-valley decline.

    Example:
        >>> risk_results = calculate_risk_metrics(df)
        >>> print(f"Risk-Adjusted Return: {risk_results['sharpe']:.2f}")
    """
    if df.empty or 'Close' not in df.columns:
        return {"sharpe": 0.0, "maximum_drawdown": 0.0}

    # 1. Periodic Return Calculation
    returns = df['Close'].pct_change().dropna()

    # Early exit for constant price series to avoid division by zero
    if returns.std() == 0:
        return {"sharpe": 0.0, "maximum_drawdown": 0.0}

    # 2. Annualized Sharpe Ratio with Risk-Free Rate Adjustment
    # Convert annual risk-free rate from config to daily basis
    daily_rf = config.IndicatorSettings.RISK_FREE_RATE / config.LayoutSettings.TRADING_DAYS_PER_YEAR

    # Sharpe = ((Mean Return - Risk Free Rate) / Standard Deviation) * sqrt(Trading Days)
    excess_return = returns.mean() - daily_rf
    sharpe = (excess_return / returns.std()) * (config.LayoutSettings.TRADING_DAYS_PER_YEAR ** 0.5)

    # 3. Maximum Drawdown (Absolute Magnitude)
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak

    # Return as a positive absolute percentage (standard for UI metrics)
    maximum_drawdown = abs(drawdown.min() * 100)

    return {
        "sharpe": sharpe,
        "maximum_drawdown": maximum_drawdown
    }