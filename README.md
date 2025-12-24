# Multi-Asset Analytics Dashboard
### Group 8

## Project Overview
This application is a **Mock Algorithmic Trading Dashboard** designed to simulate a professional live-trading environment. It retrieves real-time market data and applies a technical strategy to generate automated Buy/Sell signals and identify future market trends.

**Target Assets:**  Index (e.g., ^GSPC), Crypto (e.g., BTC-USD, ETH-USD), and Commodities Futures(e.g., Gold Future (GC=F)).

---
## Key Features
**Live Data Simulation:** Updates market data and performance metrics every 60 seconds using Streamlit fragments to output a live terminal feed.

### A. Asset Selection & Dynamic Refresh
* **Functionality:** The sidebar allows users to switch between Gold, S&P 500, Bitcoin, and Ethereum.
* **Scenario:** When a user selects "Bitcoin," the `@st.fragment` engine triggers a localized refresh. All metrics, including the **Live Price** and **24h Delta**, update instantly without refreshing the entire browser page.
* **Logic:** This is handled by the `render_live_dashboard` function, which maps the selected label to the correct `yfinance` ticker via `config.ASSET_MAPPING`.

### B. Interactive Charting & View Customization
* **Functionality:** The "Price Action" chart supports both **Candlestick** and **Line** styles. 
* **Scenario:** A user can use the "Style" selectbox within the chart container to switch from a high-detail Candlestick view to a simplified Line view for trend analysis.
* **Timeframe Control:** Each chart (Price, Volume, RSI, MACD) has an independent **Timeframe (TF)** selector. A user can analyze the "1 Day" trend on the main chart while checking "15 Minutes" momentum on the RSI oscillator.
* **Interactivity:** Powered by Plotly, users can hover over any data point to see the exact Open, High, Low, or Close prices.



### C. Technical Overlays & Oscillators
* **Overlays:** The main chart renders **Bollinger Bands** (shaded area) and **Moving Averages** (MA20/MA50) as dynamic layers.
* **Indicators:** * **Volume Surge:** Color-coded bars (Green/Red) indicating buying or selling pressure.
    * **RSI Momentum:** A dedicated oscillator chart with "Overbought" and "Oversold" dashed boundaries.
    * **MACD Trend Strength:** A histogram showing the acceleration or deceleration of the current trend.



### D. Market Intelligence & Clickable News Feed
* **Intelligence Signals:** The right-hand column provides a "Quick Glance" at the **Market Regime** (Bullish/Bearish) and the **Resistance Gap** (percentage distance to the 20-day high).
* **Live News Feed:** The `_render_news_sentiment_feed` retrieves the latest headlines related to the selected asset. 
* **Action:** Each news headline is a live hyperlink. Clicking the title redirects the user to the original source (e.g., Yahoo Finance) in a new browser tab.

### E. Risk Analytics & Fundamental Snapshot
* **Risk Engine:** Users can view the **Sharpe Ratio** and **Max Drawdown** calculated in real-time from the price data.
* **52-Week Snapshot:** A visual progress bar shows where the current price sits relative to its yearly High and Low, providing immediate fundamental context.

### E. Technical Forecasting
* **TEMA Forecasting:** Implements a Triple Exponential Moving Average (TEMA) engine to generate predictive price targets.
* **Primary Strategy:** A multi-factor logic engine evaluating RSI, MACD, and Moving Average crossovers to issue "BUY", "SELL", or "HOLD" signals.
* **Market Regime Detection:** Real-time analysis of trend alignment (MA20 vs MA50) to classify the environment as Bullish or Bearish.

---

### Public API Example
For developers looking to extend this terminal, the core engines can be accessed via our internal API:

```python
# 1. Fetching Data via Data Engine
from src.engines.data_engine import fetch_market_dashboard_data
df, price, news, ytd = fetch_market_dashboard_data("1d", "BTC-USD")

# 2. Generating Forecast via TEMA Engine
from src.engines.tema_strategy_engine import generate_tema_forecast
forecast_df = generate_tema_forecast(df, forecast_steps=30)

# 3. Executing Strategy via Logic Engine
from src.logic.trading_strategy import evaluate_market_signal
signal, ui_color = evaluate_market_signal(df)
```
---

## Tech Stack & Libraries
* **Python 3.10+**
* **Streamlit:** UI orchestration and dashboard layout.
* **Pandas & NumPy:** Backend data processing and statistical indicator logic.
* **Plotly:** Interactive financial data visualization.
* **YFinance:** Retrieval of realistic historical and live-simulated market data.
---
## Project Structure
```text
Clean_Code_Group_8/
├── assets                         # Additional Docs for logo
├── config.py                      # Global constants, indicator settings
├── main.py                        # UI Orchestration & Main Entry Point
├── tests                  
│   └── test_trading_logic         # Test the functionality and data availability
├── streamlit/
│   └── config.toml                # Force Dark Mode & App Theme Configuration
└── src/
    ├── engines/                   # Data acquisition & Prediction Logic
    │   └── data_engine.py     
    │   └── tema_strategy_engine.py           
    ├── logic/                     # Trading strategy evaluation & Signal math
    │    └── risk_engine.py     
    │    └── trading_strategy.py   
    └── ui/                        # Stylesheets, CSS Injection, & Custom Components
        └── styles.py   
```
---
## Data Sources
* **Primary Source:** https://finance.yahoo.com/

* **Data description:** Open, High, Low, Close, Volume data points are utilized 

* **Real-time Simulation:** The data available is historical, however, we use Streamlit functionalities to refresh the view every 60 seconds, simulating a live-streaming data feed.

* **Update Time:** 60 seconds are chosen as an update time for the user-friendly interface purposes

---

## Formulas & Logic
We use a multi-layered financial engineering stack pf functions and formulas:

### A. Predictive Analytics (The Forecasting Engine)
* **Triple Exponential Moving Average (TEMA):**
    * **Formula:** $TEMA = (3 \times EMA_1) - (3 \times EMA_2) + EMA_3$
    * **Logic:** Standard moving averages suffer from Lag effects. TEMA uses three layers of Exponential Moving Averages (EMAs) to mathematically subtract that lag.
    * **Application:** TEMA Trend line reacts almost instantly to price pivots, allowing us to generate 30-step forward-looking price targets.

    
### B. Risk Engineering (The Safety Layer)
* **Annualized Sharpe Ratio:**
    * **Formula:** $S = \frac{\mu}{\sigma} \times \sqrt{252}$
    * **Logic:** We take the average daily return ($\mu$) and divide it by the volatility or standard deviation ($\sigma$). We then multiply by the square root of 252 (the number of trading days in a year) to annualize the result.
    * **Meaning:** This tells the user if the returns are high enough to justify the fluctuations. A value $>1.0$ is considered a good investment.
* **Max Drawdown (MDD):**
    * **Formula:** $MDD = \frac{\text{Trough Value} - \text{Peak Value}}{\text{Peak Value}}$
    * **Meaning:** This quantifies the "Worst Case Scenario." It tracks the largest percentage drop from a peak to a valley, showing the maximum historical risk an investor would have faced in the current period.
* **52-Week Range Positioning:**
    * **Formula:** $\text{Pos} = \frac{\text{Current} - \text{Low}_{52w}}{\text{High}_{52w} - \text{Low}_{52w}} \times 100$
    * **Application:** Displayed as a progress bar, this helps identify "Value" vs. "Overextended" zones based on annual boundaries.

### C. Momentum & Market Signals
* **Bollinger Bands (Volatility Boundaries):**
    * **Formula:** $\text{Upper/Lower} = SMA_{20} \pm (2 \times \sigma)$
    * **Logic:** We plot lines 2 standard deviations away from a 20-day average. Statistically, 95% of price action stays within these bands.
    * **Signal:** When the price touches the Lower Band while RSI is low, our strategy flags a "BUY" as the asset is statistically "stretched" too far.


* **Relative Strength Index (RSI):**
    * **Formula:** $RSI = 100 - \left[ \frac{100}{1 + \frac{\text{Avg Gain}}{\text{Avg Loss}}} \right]$
    * **Application:** We use a 14-period window to identify overbought ($>70$) or oversold ($<30$) conditions.
* **MACD Trend Strength:**
    * **Formula:** $MACD = EMA_{12} - EMA_{26}$
    * **Confluence Logic:** Our terminal requires **Confluence** to issue a "STRONG BUY": the RSI must have been oversold within the last 10 bars **AND** the MACD Histogram must confirm a positive momentum shift.


### D. Market Structural Analysis
* **Resistance Gap:**
    * **Formula:** $\frac{\text{20D High} - \text{Current Price}}{\text{Current Price}}$
    * **Application:** Measures the percentage "headroom" available before the asset hits immediate historical resistance.
* **Regime Detection:**
    * **Logic:** If $\text{Price} > MA_{20} > MA_{50}$, the market is classified as **"BULLISH"**. This creates a trend-filter, preventing the algorithm from "Buying the dip" in a crashing market where the long-term trend is broken.

---

## Setup Instructions & Dependencies

### Prerequisites
* **Python 3.10+** (Required for Streamlit Fragment support and advanced typing)

### Installation
1. **Clone the repo:**
   Clone the repository using the https address
   ```bash
   git clone https://github.com/AEhubdev/Clean_Code_Group_8.git

2. **Install Libraries:**
   Instead of installing libraries one by one, use the provided requirements file to ensure version compatibility:
   ```bash
      pip install -r requirements.txt
 
3. **Run application:**
   Launch the terminal dashboard from the root directory:
   ```bash
    python -m streamlit run main.py

---

## Peculiarities & Evaluation Notes

### Code Evaluation Guidelines
For the "Clean Code" review, we recommend focusing on the following modules which contain our custom mathematical logic:
* `src/logic/trading_strategy.py`: Contains the multi-bar confluence logic.
* `src/engines/tema_strategy_engine.py`: Contains the Triple EMA forecasting algorithm.
* `main.py`: Demonstrates modular UI decomposition and state management using Streamlit fragments.

### Intentional Behaviors (Not Bugs)
* **Signal Lag:** Users may observe "Sell" signals during an upward trend. This is a deliberate feature of our risk-averse strategy; signals are triggered when oscillators reach extreme "Overbought" levels ($RSI > 70$), prioritizing capital preservation over chasing the final peak of a trend.
* **YFinance Latency:** The dashboard relies on the `yfinance` library. Occasionally, API rate-limiting may cause a brief delay in data retrieval. We have implemented error handling in the `data_engine` to alert the user if the data provider is unresponsive. Sometimes Gold data takes longer to load than that of others.
* **Date Alignment:** Because different assets (Crypto vs. Stocks) have different trading calendars, the X-axis on charts is treated as a categorical index to prevent "gaps" in the visual rendering during weekends.
* **Refresh delay time:** Every 60 seconds the dashboard reruns automatically. Currently, we have a broad structure and 1 orchestrator, therefore it might take some seconds for the dashboard to reload.
---

