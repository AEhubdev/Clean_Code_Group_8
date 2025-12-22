from typing import List, Dict, Tuple
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src.engines import data_engine, tema_strategy_engine
from src.ui import styles
from src.logic import trading_strategy


#Initialize
st.set_page_config(page_title="Multi-Asset Analytics Dashboard", layout="wide")
styles.inject_terminal_stylesheet()

@st.fragment(run_every="1m")
def render_live_dashboard(ticker_symbol: str, asset_display_name: str) -> None:
    """Orchestrates the data acquisition and UI rendering for the terminal."""
    market_data, current_price, news_list, ytd_price = data_engine.fetch_market_dashboard_data(
        timeframe_label="1 Day",
        ticker_symbol=ticker_symbol
    )

    if market_data.empty:
        st.error("Market data currently unavailable.")
        return

    performance_metrics = data_engine.calculate_performance_metrics(current_price, market_data, ytd_price)

    _render_header(asset_display_name, current_price, market_data, performance_metrics)

    st.divider()
    chart_col, intelligence_col = st.columns([0.72, 0.28])

    with chart_col:
        _render_all_charts(ticker_symbol)

    with intelligence_col:
        _render_market_signals(market_data, current_price)

    _render_news_sentiment_feed(news_list, asset_display_name)


def _render_header(name: str, price: float, df: pd.DataFrame, metrics: Tuple) -> None:
    """Displays the asset title and top-level metric row."""
    clean_name = name.split(' (')[0]
    st.title(f"{clean_name} Analytics Dashboard")

    yesterday_close = df['Close'].iloc[-2] if len(df) > 1 else price
    daily_delta = ((price - yesterday_close) / yesterday_close) * 100

    cols = st.columns(5)
    cols[0].metric(label="Live Price", value=f"${price:,.2f}", delta=f"{daily_delta:+.2f}%")

    # Map metrics to their labels and types
    metric_configs = [
        ("Weekly", metrics[0], False),
        ("Monthly", metrics[1], False),
        ("YTD", metrics[2], False),
        ("Volatility", metrics[3], True)
    ]

    for i, (label, value, is_vol) in enumerate(metric_configs, 1):
        styles.render_colored_performance_metric(
            cols[i], label, f"{value:+.2f}%" if not is_vol else f"{value:.2f}%", value, is_volatility=is_vol
        )


def _render_market_signals(market_df: pd.DataFrame, current_price: float) -> None:
    st.markdown("### Market Signals")
    latest = market_df.iloc[-1]

    # 1. TEMA Prediction
    prediction = tema_strategy_engine.generate_tema_forecast(market_df)
    if not prediction.empty:
        target_price = prediction['Predicted'].iloc[-1]
        upside = ((target_price - current_price) / current_price) * 100
        styles.render_intelligence_signal("TEMA ESTIMATED TARGET", f"${target_price:,.2f}", f"{upside:+.2f}%",
                                          styles.COLOR_GOLD)

    # 2. Strategy & Regime
    signal, color = trading_strategy.evaluate_market_signal(market_df)
    styles.render_intelligence_signal("PRIMARY STRATEGY", signal, "LIVE", color)

    is_bullish = current_price > latest['MA20'] > latest['MA50']
    regime_text, regime_color = ("BULLISH", styles.SUCCESS_COLOR) if is_bullish else ("BEARISH", styles.DANGER_COLOR)
    styles.render_intelligence_signal("MARKET REGIME", regime_text, "TREND", regime_color)

    # 3. Structural Analysis
    res_20d = market_df['High'].tail(20).max()
    res_gap = ((res_20d - current_price) / current_price) * 100
    styles.render_intelligence_signal("RESISTANCE GAP", f"{res_gap:.2f}%", "TO 20D HIGH", styles.HOLD_COLOR)

def _render_all_charts(ticker: str) -> None:
    """Wrapper for categorical chart rendering."""
    chart_definitions = [
        ("PRICE ACTION & TEMA FORECAST", "price", "p1"),
        ("VOLUME SURGE", "volume", "v1"),
        ("MOMENTUM (RSI)", "rsi", "r1"),
        ("TREND STRENGTH (MACD)", "macd", "m1")
    ]
    for title, c_type, key in chart_definitions:
        _render_chart_container(title, c_type, key, ticker)


def _render_chart_container(title: str, chart_type: str, key: str, ticker: str) -> None:
    """Generic container with added toggles for Price chart types."""
    with st.container(border=True):
        header_col, type_col, selector_col = st.columns([0.5, 0.2, 0.3])
        header_col.markdown(f"**{title}**")

        #Toggle only appears for the main price chart
        price_style = "Candlestick"
        if chart_type == "price":
            price_style = type_col.selectbox(
                "Style", ["Candlestick", "Line"], index=0,
                key=f"style_{key}_{ticker}", label_visibility="collapsed"
            )

        timeframe = selector_col.selectbox(
            "TF", list(config.AVAILABLE_TIMEFRAMES.keys()), index=2,
            key=f"{key}_{ticker}", label_visibility="collapsed"
        )

        df, _, _, _ = data_engine.fetch_market_dashboard_data(timeframe, ticker_symbol=ticker)
        if df.empty: return

        plot_data = df.tail(150).copy()
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
    idx = data.index.strftime(config.DATE_FORMAT)

    # 1. Structural Bands
    fig.add_scatter(
        x=idx,
        y=data['BB_U'],
        line=dict(color=styles.BB_LINE_COLOR, width=1),
        name="BB Upper"
    )

    fig.add_scatter(
        x=idx,
        y=data['BB_L'],
        line=dict(color=styles.BB_LINE_COLOR, width=1),
        fill='tonexty',
        fillcolor=styles.BB_FILL_COLOR,
        name="BB Lower"
    )
    # 2. Price Representation
    if price_style == "Candlestick":
        fig.add_trace(go.Candlestick(x=idx, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
                                     name="Price"))
    else:
        fig.add_trace(go.Scatter(x=idx, y=data['Close'], line=dict(color=styles.COLOR_WHITE, width=2), name="Price (Line)"))

    fig.add_trace(go.Scatter(x=idx, y=data['MA20'], line=dict(color=styles.HOLD_COLOR, width=1.5), name="MA20"))
    fig.add_trace(go.Scatter(x=idx, y=data['MA50'], line=dict(color=styles.MACD_COLOR, width=1.5), name="MA50"))

    # 3. Forecast & Signals
    if not forecast.empty:
        f_idx = forecast.index.strftime(config.DATE_FORMAT)
        fig.add_scatter(x=f_idx, y=forecast['Predicted'], line=dict(color=styles.COLOR_GOLD, width=3, dash='dashdot'),
                        name="TEMA Trend")

    _add_signal_markers(fig, data, idx)
    fig.update_layout(height=config.LayoutSettings.MAIN_CHART_HEIGHT)


def _add_signal_markers(fig: go.Figure, data: pd.DataFrame, idx: pd.Index) -> None:
    """Helper to cleanly place Buy/Sell triangles on the chart."""
    buffer = (data['High'].max() - data['Low'].min()) * 0.02 #2% buffer to avoid the overlap on the chart

    for sig_type, symbol, color, col in [('Buy', 'triangle-up', styles.SUCCESS_COLOR, 'Low'),
                                         ('Sell', 'triangle-down', styles.DANGER_COLOR, 'High')]:
        mask = data[f'{sig_type}_Signal']
        y_pos = data[col] - buffer if sig_type == 'Buy' else data[col] + buffer
        fig.add_trace(go.Scatter(
            x=idx[mask], y=y_pos[mask], mode='markers',
            marker=dict(symbol=symbol, size=12, color=color), name=f"{sig_type} Signal"
        ))


def _plot_volume_layer(fig: go.Figure, data: pd.DataFrame) -> None:
    idx = data.index.strftime(config.DATE_FORMAT)
    colors = [styles.SUCCESS_COLOR if c >= o else styles.DANGER_COLOR for c, o in zip(data['Close'], data['Open'])]
    fig.add_bar(x=idx, y=data['Volume'], marker_color=colors)
    fig.update_layout(height=150)


def _plot_rsi_layer(fig: go.Figure, data: pd.DataFrame) -> None:
    idx = data.index.strftime(config.DATE_FORMAT)
    fig.add_scatter(x=idx, y=data['RSI'], line=dict(color=styles.RSI_COLOR))
    fig.add_hline(y=config.IndicatorSettings.RSI_OVERBOUGHT_LEVEL, line_color="red", line_dash="dash")
    fig.add_hline(y=config.IndicatorSettings.RSI_OVERSOLD_LEVEL, line_color="green", line_dash="dash")
    fig.update_layout(height=150, yaxis=dict(range=[0, 100]))


def _plot_macd_layer(fig: go.Figure, data: pd.DataFrame) -> None:
    idx = data.index.strftime(config.DATE_FORMAT)
    colors = [styles.SUCCESS_COLOR if x >= 0 else styles.DANGER_COLOR for x in data['MACD_Hist']]
    fig.add_bar(x=idx, y=data['MACD_Hist'], marker_color=colors)
    fig.update_layout(height=150)


def _render_news_sentiment_feed(news: List[Dict], asset_name: str) -> None:
    st.divider()
    st.subheader(f"{asset_name.split(' (')[0]} Market News")
    if not news:
        st.info("No news items found.")
        return
    for item in news[:5]:
        with st.container(border=True):
            st.markdown(f"**[{item['title']}]({item['link']})**")


if __name__ == "__main__":
    st.sidebar.markdown('<div class="sidebar-header-branding">TERMINAL</div>', unsafe_allow_html=True)
    asset_label = st.sidebar.selectbox("Select Asset", options=list(config.ASSET_MAPPING.keys()), index=0)
    render_live_dashboard(config.ASSET_MAPPING[asset_label], asset_label)