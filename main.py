from typing import List, Dict, Tuple
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

    _render_header(asset_display_name, current_price, market_data, performance_metrics)

    st.divider()
    chart_col, intelligence_col = st.columns(config.LayoutSettings.DASHBOARD_MAIN_RATIO)

    with chart_col:
        _render_all_charts(ticker_symbol)

    with intelligence_col:
        _render_market_signals(market_data, current_price)

    _render_news_sentiment_feed(news_list, asset_display_name)


def _render_header(name: str, price: float, header_data: Dict) -> None:
        """
            Displays the asset title and top-level metric row.

            Args:
                name (str): Full asset name.
                price (float): Current live price.

            Returns:
                None
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


def _render_market_signals(market_df: pd.DataFrame, current_price: float) -> None:
    """
    Renders intelligence signals and risk analytics.

    Focused Task: Orchestrate the display of signals fetched from engines.
    """
    #Fetch calculated signals from engine
    signals = data_engine.calculate_market_signals(market_df, current_price)

    st.markdown("### Market Signals")

    #Focused Rendering


    styles.render_intelligence_signal(
        "MARKET REGIME",
        signals["regime"],
        "TREND",
        signals["regime_color"]
    )

    styles.render_intelligence_signal(
        "RESISTANCE GAP",
        f"{signals['resistance_gap']:.2f}%",
        "TO 20D HIGH",
        styles.HOLD_COLOR
    )

    #Sub-components
    _render_fundamental_snapshot(market_df)
    _render_risk_analytics(market_df)

    #Definitions
    _render_signal_definitions_expander()


def _render_signal_definitions_expander() -> None:
    """Focused Task: Displays the glossary of terms for the user."""
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

def _render_fundamental_snapshot(df: pd.DataFrame) -> None:
    """Displays key price-action based fundamental boundaries."""
    st.markdown("### Fundamental Snapshot")
    with st.container(border=True):
        high_52w = df['High'].tail(config.LayoutSettings.TRADING_DAYS_PER_YEAR).max()
        low_52w = df['Low'].tail(config.LayoutSettings.TRADING_DAYS_PER_YEAR).min()
        current = df['Close'].iloc[-1]

        # Calculate where we are in the 52-week range
        range_pos = ((current - low_52w) / (high_52w - low_52w)) * 100 if (high_52w - low_52w) != 0 else 0

        c1, c2 = st.columns(2)
        c1.caption("52W High")
        c1.markdown(f"**${high_52w:,.2f}**")
        c2.caption("52W Low")
        c2.markdown(f"**${low_52w:,.2f}**")

        st.progress(min(max(range_pos / 100, 0.0), 1.0))
        st.caption(f"Price is at {range_pos:.1f}% of its 52-week range")


def _render_risk_analytics(df: pd.DataFrame) -> None:
    """Calculates and displays risk-adjusted performance."""
    st.markdown("### Risk Analytics")
    with st.container(border=True):
        # Call the new logic engine instead of doing math here
        metrics = risk_engine.calculate_risk_metrics(df)

        r1, r2 = st.columns(2)
        r1.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
        r2.metric("Max Drawdown", f"{metrics['maximum_drawdown']:.1f}%", delta_color="inverse")


def _render_all_charts(ticker: str) -> None:
    """Wrapper for categorical chart rendering."""
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
        Creates a UI container with toggles for chart types and timeframes.

        Args:
            title (str): Display title for the chart.
            chart_type (str): Type identifier (price, rsi, etc.).
            key (str): Unique Streamlit key for widgets.
            ticker (str): Asset ticker symbol.

        Returns:
            None
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
    """Routes the figure to the specific technical layer."""
    if chart_type == "price":
        _plot_price_layer(fig, data, price_style)
    elif chart_type == "volume":
        _plot_volume_layer(fig, data)
    elif chart_type == "rsi":
        _plot_rsi_layer(fig, data)
    elif chart_type == "macd":
        _plot_macd_layer(fig, data)


def _plot_price_layer(fig: go.Figure, data: pd.DataFrame, price_style: str) -> None:
    """Main price action layer with conditional rendering for Candlestick vs Line."""
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
    """Sets the triangle signals on the chart"""
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
    """Renders a list of clickable market news headlines"""
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