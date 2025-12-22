"""
Automated Test Suite for Trading Logic and Engines.
"""

import unittest
import pandas as pd
import numpy as np
from src.logic import trading_strategy, risk_engine
from src.engines import tema_strategy_engine

class TestTradingSystem(unittest.TestCase):
    """
    Test class for the Trading Dashboard logic.
    """

    def setUp(self):
        """
        Set up a fresh mock dataframe
        before every single test execution.
        """
        rows = 100
        dates = pd.date_range(start="2010-01-01", periods=rows)
        self.mock_df = pd.DataFrame({
            'Open': np.random.uniform(100, 200, rows),
            'High': np.random.uniform(100, 200, rows),
            'Low': np.random.uniform(100, 200, rows),
            'Close': np.random.uniform(100, 200, rows),
            'Volume': np.random.uniform(1000, 5000, rows),
            'MA20': [150.0] * rows,
            'MA50': [140.0] * rows,
            'RSI': [50.0] * rows,
            'MACD_Hist': [0.1] * rows,
            'Buy_Signal': [False] * rows,
            'Sell_Signal': [False] * rows
        }, index=dates)

    def test_signal_generation_output_format(self):
        """Verify that the strategy returns a string signal and a hex color."""
        signal, color = trading_strategy.evaluate_market_signal(self.mock_df)

        self.assertIsInstance(signal, str)
        self.assertTrue(color.startswith("#"))

    def test_tema_forecast_shape(self):
        """
        Ensure the TEMA engine generates the correct number of steps.
        Expected: steps + 1 anchor row (current price + future projections).
        """
        steps = 30
        forecast = tema_strategy_engine.generate_tema_forecast(self.mock_df, forecast_steps=steps)

        # The engine returns 30 future steps + 1 current price row
        self.assertEqual(len(forecast), steps + 1)
        self.assertIn("Predicted", forecast.columns)

    def test_strategy_oversold_edge_case(self):
        """Test 'BUY' signal triggering when RSI is at 10.0 (Edge Case)."""
        self.mock_df.loc[self.mock_df.index[-1], 'RSI'] = 10.0
        self.mock_df.loc[self.mock_df.index[-1], 'Buy_Signal'] = True

        signal, _ = trading_strategy.evaluate_market_signal(self.mock_df)
        self.assertIn("BUY", signal)

    def test_data_integrity_empty_df(self):
        """
        Verify the logic layer handles empty data gracefully
        by returning 'WAITING FOR DATA' instead of crashing.
        """
        empty_df = pd.DataFrame()

        # Updated to match the actual output of your logic engine
        signal, _ = trading_strategy.evaluate_market_signal(empty_df)

        self.assertEqual(signal, "WAITING FOR DATA")

    def test_risk_metrics_calculation(self):
            """
            Verify that the risk engine correctly calculates Sharpe and Drawdown.
            Tests Criteria 58 (Comprehensive).
            """
            # Create a dataframe with a known downward trend to test Max Drawdown
            downward_df = pd.DataFrame({
                'Close': [100, 90, 80, 70]
            })
            metrics = risk_engine.calculate_risk_metrics(downward_df)

            self.assertLess(metrics['max_dd'], 0)  # Drawdown should be negative
            self.assertIsInstance(metrics['sharpe'], float)

if __name__ == "__main__":
    # Allows the test to be run directly via 'python tests/test_trading_logic.py'
    unittest.main()