from typing import List, Dict
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src.engines import data_engine, tema_strategy_engine
from src.ui import styles
from src.logic import trading_strategy, risk_engine

# Initialize
st.set_page_config(page_title="Multi-Asset Analytics Dashboard", layout="wide")
styles.inject_terminal_stylesheet()

@st.fragment(run_every="1m")
def render_live_dashboard(ticker_symbol: str, asset_display_name: str) -> None:
    """
        Orchestrates the data acquisition and UI rendering for the main dashboard.

        This function acts as the primary entry point for the live-updating portion of
        the application, handling data fetching, metric calculation, and layout dispatching.

        Args:
            ticker_symbol (str): The financial ticker symbol (e.g., 'BTC-USD').
            asset_display_name (str): The human-readable name for the sidebar UI.

        Returns:
            None: Renders the UI directly to the Streamlit app.

        Example:
            >>> render_live_dashboard("GC=F", "Gold")
        """
    if not isinstance(ticker_symbol, str) or not ticker_symbol:
        raise ValueError(
            f"The parameter 'ticker_symbol' must be a non-empty string, but '{ticker_symbol}' "
            f"(type: {type(ticker_symbol).__name__}) was provided instead. "
            f"Please check your config.ASSET_MAPPING to ensure the ticker is correctly defined."
        )
    #Logo setup
    col1, _ = st.columns(config.LayoutSettings.LOGO_COLUMN_RATIO)

    with col1:
        st.image("assets/logo.png", width=config.LayoutSettings.LOGO_WIDTH)

#Orchestrates the data acquisition and UI rendering for the terminal
    market_data, current_price, news_list, ytd_price = data_engine.fetch_market_dashboard_data(
        timeframe_label=config.AVAILABLE_TIMEFRAMES["1 Day"],
        ticker_symbol=ticker_symbol
    )

    if market_data.empty:
        st.error(
            f"Market data for {ticker_symbol} is currently unavailable. Please check your internet connection or the ticker symbol.")
        return

    performance_metrics = data_engine.calculate_performance_metrics(current_price, market_data, ytd_price)

    header_stats = data_engine.prepare_header_metrics(
        name=asset_display_name,
        price=current_price,
        df=market_data,
        performance=performance_metrics
    )

    _render_header(asset_display_name, current_price, header_stats)

    st.divider()
    chart_col, intelligence_col = st.columns(config.LayoutSettings.DASHBOARD_MAIN_RATIO)

    with chart_col:
        _render_all_charts(ticker_symbol)

    with intelligence_col:
        _render_market_signals(market_data)

    _render_news_sentiment_feed(news_list, asset_display_name)


def _render_header(name: str, price: float, header_data: Dict) -> None:
    """
        Displays the asset title and top-level metric row.

        Args:
            header_data (Dict): Pre-calculated metrics including price, delta, and historical returns.

        Returns:
            None: Renders Streamlit metric widgets.

        Example:
            >>> _render_header({'display_name': 'Gold', 'daily_delta': 1.2, 'weekly': 2.0...})
    """
    st.title(f"{header_data['display_name']} Analytics Dashboard")

    metric_columns = st.columns(5)

    metric_columns[0].metric(
            label="Live Price",
            value=f"${price:,.2f}",
            delta=f"{header_data['daily_delta']:+.2f}%"
        )

    metric_configs = [
            ("Weekly", header_data['weekly'], False),
            ("Monthly", header_data['monthly'], False),
            ("YTD", header_data['ytd'], False),
            ("Volatility", header_data['volatility'], True)
        ]

    for i, (label, value, is_vol) in enumerate(metric_configs, 1):
            styles.render_colored_performance_metric(
                metric_columns[i],
                label,
                f"{value:+.2f}%" if not is_vol else f"{value:.2f}%",
                value,
                is_volatility=is_vol
            )


def _render_market_signals(market_df: pd.DataFrame) -> None:
    """
    Renders intelligence signals, trading strategy outputs, and risk analytics.

    Args:
        market_dataframe (pd.DataFrame): The historical price and indicator data.

    Returns:
        None: Renders information cards to the sidebar intelligence column.

    Example:
        >>> _render_market_signals(pd.DataFrame({'Close': [100, 101]}))
    """
    current_price = market_df['Close'].iloc[-1]

    strat_label, strat_color = trading_strategy.evaluate_market_signal(market_df)

    engine_signals = data_engine.calculate_market_signals(market_df, current_price)
    risk_metrics = risk_engine.calculate_risk_metrics(market_df)
    snapshot_data = data_engine.calculate_fundamental_snapshot(market_df)

    st.markdown("### Market Signals")

    if engine_signals.get("target_price", 0) > 0:
        styles.render_intelligence_signal(
            "ESTIMATED TARGET",
            f"${engine_signals['target_price']:,.2f}",
            f"{engine_signals['upside_pct']:+.2f}%",
            styles.COLOR_GOLD
        )

    # Strategy Signal
    styles.render_intelligence_signal(
        "PRIMARY STRATEGY",
        strat_label,
        "LIVE",
        strat_color
    )

    # Market Regime (From data_engine)
    styles.render_intelligence_signal(
        "MARKET REGIME",
        engine_signals["regime"],
        "TREND",
        engine_signals["regime_color"]
    )

    # Structural Analysis (From data_engine)
    styles.render_intelligence_signal(
        "RESISTANCE GAP",
        f"{engine_signals['resistance_gap']:.2f}%",
        "TO 20D HIGH",
        styles.HOLD_COLOR
    )
    _render_fundamental_snapshot(snapshot_data)
    _render_risk_analytics(risk_metrics)
    _render_signal_definitions_expander()


def _render_signal_definitions_expander() -> None:
    """
        Description:
            Displays a collapsible glossary (Streamlit expander) containing definitions
            for technical terms like TEMA, Market Regime, and Sharpe Ratio.

        Args:
            None

        Returns:
            None: Renders a markdown-styled glossary directly to the UI.

        Example:
            >>> _render_signal_definitions_expander()
        """
    with st.expander("ℹ Signal Definitions"):
        st.markdown(f"""
            <div style="font-size: 0.85rem; color: {styles.TEXT_SUBDUED};">
                <b>TEMA TARGET:</b> Price prediction based on Triple Exponential Moving Averages.<br><br>
                <b>PRIMARY STRATEGY:</b> Current actionable signal (Buy/Sell/Hold) derived from momentum.<br><br>
                <b>MARKET REGIME:</b> Global trend state. Bullish if price is above key Moving Averages.<br><br>
                <b>SHARPE RATIO:</b> Risk-adjusted return. Above 1.0 is considered good.<br><br>
                <b>MAX DRAWDOWN:</b> Largest peak-to-valley drop in the current period.<br><br>
                <b>RESISTANCE GAP:</b> Percentage distance to the 20-day high.
            </div>
        """, unsafe_allow_html=True)


def _render_fundamental_snapshot(snapshot: Dict) -> None:
    """
        Description:
            Displays price-action based fundamental boundaries, including the 52-week
            high/low and a progress bar showing the current price's relative position.

        Args:
            snapshot (Dict): A dictionary containing 'high_52w' (float), 'low_52w' (float),
                'range_pos_normalized' (float), and 'range_pos_pct' (float).

        Returns:
            None: Renders fundamental metrics and a progress bar to the UI.

        Example:
            >>> data = {'high_52w': 150.0, 'low_52w': 100.0, 'range_pos_normalized': 0.5, 'range_pos_pct': 50.0}
            >>> _render_fundamental_snapshot(data)
        """
    st.markdown("### Fundamental Snapshot")
    with st.container(border=True):
        c1, c2 = st.columns(2)

        # Displaying pre-calculated numbers
        c1.caption("52W High")
        c1.markdown(f"**${snapshot['high_52w']:,.2f}**")

        c2.caption("52W Low")
        c2.markdown(f"**${snapshot['low_52w']:,.2f}**")

        # UI components using pre-calculated logic
        st.progress(snapshot['range_pos_normalized'])
        st.caption(f"Price is at {snapshot['range_pos_pct']:.1f}% of its 52-week range")


def _render_risk_analytics(metrics: Dict) -> None:
    """
        Description:
            Displays risk-adjusted performance metrics, specifically the Sharpe Ratio
            and the Maximum Drawdown for the selected asset.

        Args:
            metrics (Dict): A dictionary containing 'sharpe' (float) and 'maximum_drawdown' (float).

        Returns:
            None: Renders Streamlit metric widgets within a bordered container.

        Example:
            >>> risk_data = {'sharpe': 1.42, 'maximum_drawdown': 12.5}
            >>> _render_risk_analytics(risk_data)
        """
    st.markdown("### Risk Analytics")
    with st.container(border=True):
        r1, r2 = st.columns(2)
        r1.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")

        # Max Drawdown is now a positive magnitude from risk_engine
        r2.metric(
            label="Max Drawdown",
            value=f"{metrics['maximum_drawdown']:.1f}%",
            delta="Risk Depth",
            delta_color="off"
        )

def _render_all_charts(ticker: str) -> None:
    """
        Iterates through defined chart types to render the full technical suite.

        Args:
            ticker (str): The financial ticker symbol for data retrieval.

        Returns:
            None: Renders multiple Plotly charts in containers.

        Example:
            >>> _render_all_charts("AAPL")
        """
    chart_definitions = [
        ("PRICE ACTION & TEMA FORECAST", "price", "p1"),
        ("VOLUME SURGE", "volume", "v1"),
        ("MOMENTUM (RSI)", "rsi", "r1"),
        ("TREND STRENGTH (MACD)", "macd", "m1")
    ]
    for title, chart_type, key in chart_definitions:
        _render_chart_container(title, chart_type, key, ticker)


def _render_chart_container(title: str, chart_type: str, key: str, ticker: str) -> None:
    """
        Creates a UI container with toggles for chart styles and timeframes.

        Args:
            title (str): Title displayed above the chart.
            chart_type (str): The logic selector (price, volume, rsi, macd).
            key (str): Streamlit unique widget identifier.
            ticker (str): The financial asset ticker.

        Returns:
            None: Renders the chart and selection widgets.

        Example:
            >>> _render_chart_container("Price Chart", "price", "c1", "BTC-USD")
        """
    with st.container(border=True):
        header_col, type_col, selector_col = st.columns(config.LayoutSettings.CHART_HEADER_RATIO)
        header_col.markdown(f"**{title}**")

        # Toggle only appears for the main price chart
        price_style = "Candlestick"
        if chart_type == "price":
            price_style = type_col.selectbox(
                "Style", ["Candlestick", "Line"], index=0,
                key=f"style_{key}_{ticker}", label_visibility="collapsed"
            )

        timeframe = selector_col.selectbox(
            "timeframe", list(config.AVAILABLE_TIMEFRAMES.keys()), index=2,
            key=f"{key}_{ticker}", label_visibility="collapsed"
        )

        df, _, _, _ = data_engine.fetch_market_dashboard_data(timeframe, ticker_symbol=ticker)
        if df.empty:
            st.warning(f"No chart data available for {ticker} at {timeframe} timeframe.")
            return

        plot_data = df.tail(config.LayoutSettings.CHART_LOOKBACK_PERIODS).copy()
        fig = go.Figure()

        # Modified to pass the price_style
        _dispatch_chart_type(fig, plot_data, chart_type, price_style)

        fig.update_layout(
            template="plotly_dark", margin=dict(t=30, b=5, l=5, r=5),
            showlegend=(chart_type == "price"), xaxis_rangeslider_visible=False,
            xaxis=dict(type='category', nticks=10)
        )
        st.plotly_chart(fig, use_container_width=True)


def _dispatch_chart_type(fig: go.Figure, data: pd.DataFrame, chart_type: str, price_style: str = "Candlestick") -> None:
    """
        Routes a Plotly figure to specific drawing layers based on the indicator type.

        Args:
            figure (go.Figure): The Plotly figure object to modify.
            data (pd.DataFrame): The technical indicator data.
            chart_type (str): Type of indicator to plot.
            price_style (str): Formatting choice for price (Candlestick/Line).

        Returns:
            None: Modifies the figure object in-place.

        Example:
            >>> _dispatch_chart_type(go.Figure(), df, "rsi", "Line")
        """
    if chart_type == "price":
        _plot_price_layer(fig, data, price_style)
    elif chart_type == "volume":
        _plot_volume_layer(fig, data)
    elif chart_type == "rsi":
        _plot_rsi_layer(fig, data)
    elif chart_type == "macd":
        _plot_macd_layer(fig, data)


def _plot_price_layer(fig: go.Figure, data: pd.DataFrame, price_style: str) -> None:
    """
        Plots the primary price layer including Bollinger Bands and TEMA Forecast.

        Args:
            figure (go.Figure): Figure to update.
            data (pd.DataFrame): Market price data.
            price_style (str): Visual style choice.

        Returns:
            None
        """
    forecast = tema_strategy_engine.generate_tema_forecast(data)
    date_index = data.index.strftime(config.DATE_FORMAT)

    # 1. Structural Bands
    fig.add_scatter(
        x=date_index,
        y=data['Bollinger_Bands_Upper'],
        line=dict(color=styles.BB_LINE_COLOR, width=1),
        name="BB Upper"
    )

    fig.add_scatter(
        x=date_index,
        y=data['Bollinger_Bands_Lower'],
        line=dict(color=styles.BB_LINE_COLOR, width=1),
        fill='tonexty',
        fillcolor=styles.BB_FILL_COLOR,
        name="BB Lower"
    )
    # 2. Price Representation
    if price_style == "Candlestick":
        fig.add_trace(go.Candlestick(x=date_index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
                                     name="Price"))
    else:
        fig.add_trace(
            go.Scatter(x=date_index, y=data['Close'], line=dict(color=styles.COLOR_WHITE, width=2), name="Price (Line)"))

    fig.add_trace(go.Scatter(x=date_index, y=data['Moving_Average_20'], line=dict(color=styles.HOLD_COLOR, width=1.5), name="Moving_Average_20"))
    fig.add_trace(go.Scatter(x=date_index, y=data['Moving_Average_50'], line=dict(color=styles.MACD_COLOR, width=1.5), name="Moving_Average_50"))

    # 3. Forecast & Signals
    if not forecast.empty:
        f_date_index = forecast.index.strftime(config.DATE_FORMAT)
        fig.add_scatter(x=f_date_index, y=forecast['Predicted'], line=dict(color=styles.COLOR_GOLD, width=3, dash='dashdot'),
                        name="TEMA Trend")

    _add_signal_markers(fig, data, date_index)
    fig.update_layout(height=config.LayoutSettings.MAIN_CHART_HEIGHT)


def _add_signal_markers(fig: go.Figure, data: pd.DataFrame, date_index: pd.Index) -> None:
    """
    Description:
        Calculates and overlay technical 'Buy' and 'Sell' signal markers onto a Plotly
        figure. It applies a vertical buffer to ensure markers do not overlap with
        the price candles.

    Args:
        figure (go.Figure): The Plotly figure object where markers will be drawn.
        market_dataframe (pd.DataFrame): The technical dataset containing boolean
            'Buy_Signal' and 'Sell_Signal' columns.
        date_index (pd.Index): The formatted index of dates used for the X-axis.

    Returns:
        None: Modifies the provided figure object in-place by adding scatter traces.

    Example:
        >>> import plotly.graph_objects as go
        >>> fig = go.Figure()
        >>> dates = pd.Index(["2023-01-01", "2023-01-02"])
        >>> _add_signal_markers(fig, market_data, dates)
    """
    buffer = (data['High'].max() - data['Low'].min()) * config.LayoutSettings.SIGNAL_CHART_BUFFER_PERCENT  # 2% buffer to avoid the overlap on the chart

    for sig_type, symbol, color, col in [('Buy', 'triangle-up', styles.SUCCESS_COLOR, 'Low'),
                                         ('Sell', 'triangle-down', styles.DANGER_COLOR, 'High')]:
        mask = data[f'{sig_type}_Signal']
        y_pos = data[col] - buffer if sig_type == 'Buy' else data[col] + buffer
        fig.add_trace(go.Scatter(
            x=date_index[mask], y=y_pos[mask], mode='markers',
            marker=dict(symbol=symbol, size=12, color=color), name=f"{sig_type} Signal"
        ))


def _plot_volume_layer(fig: go.Figure, data: pd.DataFrame) -> None:
    """Draws color-coded volume bars"""
    date_index = data.index.strftime(config.DATE_FORMAT)
    colors = [styles.SUCCESS_COLOR if c >= o else styles.DANGER_COLOR for c, o in zip(data['Close'], data['Open'])]
    fig.add_bar(x=date_index, y=data['Volume'], marker_color=colors)
    fig.update_layout(height= config.LayoutSettings.INDICATOR_CHART_HEIGHT)


def _plot_rsi_layer(fig: go.Figure, data: pd.DataFrame) -> None:
    """Draws RSI line and overbought/oversold thresholds"""
    date_index = data.index.strftime(config.DATE_FORMAT)
    fig.add_scatter(x=date_index, y=data['RSI'], line=dict(color=styles.RSI_COLOR))
    fig.add_hline(y=config.IndicatorSettings.RSI_OVERBOUGHT_LEVEL, line_color="red", line_dash="dash")
    fig.add_hline(y=config.IndicatorSettings.RSI_OVERSOLD_LEVEL, line_color="green", line_dash="dash")
    fig.update_layout(height=config.LayoutSettings.INDICATOR_CHART_HEIGHT, yaxis=dict(range=[0, 100]))


def _plot_macd_layer(fig: go.Figure, data: pd.DataFrame) -> None:
    """Draws the MACD histogram"""
    date_index = data.index.strftime(config.DATE_FORMAT)
    colors = [styles.SUCCESS_COLOR if x >= 0 else styles.DANGER_COLOR for x in data['Moving_Average_Convergence_Divergence_Histogram']]
    fig.add_bar(x=date_index, y=data['Moving_Average_Convergence_Divergence_Histogram'], marker_color=colors)
    fig.update_layout(height=config.LayoutSettings.INDICATOR_CHART_HEIGHT)


def _render_news_sentiment_feed(news: List[Dict], asset_name: str) -> None:
    """
        Description:
            Renders a list of clickable market news headlines within individual bordered
            containers. The function extracts the core asset name (removing ticker
            suffixes) for the subheader and limits the display to a pre-defined maximum
            number of items.

        Args:
            news_list (List[Dict]): A list of dictionaries, where each dictionary
                contains a 'title' (str) and a 'link' (str).
            asset_name (str): The full display name of the financial asset.

        Returns:
            None: Renders Streamlit markdown and container components directly to the app.

        Example:
            >>> feed = [{"title": "Market Surges", "link": "https://news.com/1"}]
            >>> _render_news_sentiment_feed(feed, "Bitcoin (BTC-USD)")
        """
    st.divider()
    st.subheader(f"{asset_name.split(' (')[0]} Market News")
    if not news:
        st.info("No news items found.")
        return
    for item in news[:config.LayoutSettings.MAX_NEWS_ITEMS]:
        with st.container(border=True):
            st.markdown(f"**[{item['title']}]({item['link']})**")


if __name__ == "__main__":
    st.sidebar.markdown('<div class="sidebar-header-branding">TERMINAL</div>', unsafe_allow_html=True)
    asset_label = st.sidebar.selectbox("Select Asset", options=list(config.ASSET_MAPPING.keys()), index=0)
    render_live_dashboard(config.ASSET_MAPPING[asset_label], asset_label)