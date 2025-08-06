import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import ta
import json
import os

# Configure page
st.set_page_config(
    page_title="Stock Signals Analyzer",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .signal-buy {
        color: #00ff00;
        font-weight: bold;
    }
    .signal-sell {
        color: #ff0000;
        font-weight: bold;
    }
    .signal-hold {
        color: #ffaa00;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

PORTFOLIO_FILE = "portfolio.json"


def calculate_technical_indicators(df):
    """Calculate various technical indicators"""
    # Moving Averages
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)

    # MACD
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
    df['MACD_hist'] = ta.trend.macd(df['Close'])

    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

    # Bollinger Bands
    df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_middle'] = ta.volatility.bollinger_mavg(df['Close'])
    df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'])

    # Stochastic Oscillator
    df['Stoch_k'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
    df['Stoch_d'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])

    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()

    return df


def generate_signals(df):
    """Generate buy/sell signals based on technical indicators"""
    signals = []
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # Moving Average Crossover Signal
    if latest['SMA_20'] > latest['SMA_50'] and prev['SMA_20'] <= prev['SMA_50']:
        signals.append(("MA Crossover", "BUY", "20-day SMA crossed above 50-day SMA"))
    elif latest['SMA_20'] < latest['SMA_50'] and prev['SMA_20'] >= prev['SMA_50']:
        signals.append(("MA Crossover", "SELL", "20-day SMA crossed below 50-day SMA"))

    # RSI Signals
    if latest['RSI'] < 30:
        signals.append(("RSI", "BUY", f"RSI oversold at {latest['RSI']:.2f}"))
    elif latest['RSI'] > 70:
        signals.append(("RSI", "SELL", f"RSI overbought at {latest['RSI']:.2f}"))

    # MACD Signal
    if latest['MACD'] > latest['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
        signals.append(("MACD", "BUY", "MACD crossed above signal line"))
    elif latest['MACD'] < latest['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
        signals.append(("MACD", "SELL", "MACD crossed below signal line"))

    # Bollinger Bands
    if latest['Close'] < latest['BB_lower']:
        signals.append(("Bollinger Bands", "BUY", "Price below lower Bollinger Band"))
    elif latest['Close'] > latest['BB_upper']:
        signals.append(("Bollinger Bands", "SELL", "Price above upper Bollinger Band"))

    # Stochastic
    if latest['Stoch_k'] < 20 and latest['Stoch_d'] < 20:
        signals.append(("Stochastic", "BUY", "Stochastic in oversold territory"))
    elif latest['Stoch_k'] > 80 and latest['Stoch_d'] > 80:
        signals.append(("Stochastic", "SELL", "Stochastic in overbought territory"))

    return signals


def create_main_chart(df, symbol):
    """Create the main price chart with technical indicators"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Stock Price', 'Volume', 'MACD', 'RSI'),
        row_width=[0.2, 0.1, 0.1, 0.1]
    )

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price"
    ), row=1, col=1)

    # Moving averages
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_20'],
        line=dict(color='orange', width=1),
        name='SMA 20'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_50'],
        line=dict(color='blue', width=1),
        name='SMA 50'
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BB_upper'],
        line=dict(color='gray', width=1),
        name='BB Upper',
        opacity=0.3
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['BB_lower'],
        line=dict(color='gray', width=1),
        name='BB Lower',
        fill='tonexty',
        opacity=0.3
    ), row=1, col=1)

    # Volume
    colors = ['red' if row['Close'] < row['Open'] else 'green' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        name='Volume',
        marker_color=colors
    ), row=2, col=1)

    # MACD
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD'],
        line=dict(color='blue', width=2),
        name='MACD'
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD_signal'],
        line=dict(color='red', width=1),
        name='MACD Signal'
    ), row=3, col=1)

    fig.add_trace(go.Bar(
        x=df.index, y=df['MACD_hist'],
        name='MACD Histogram',
        marker_color='gray'
    ), row=3, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI'],
        line=dict(color='purple', width=2),
        name='RSI'
    ), row=4, col=1)

    # RSI horizontal lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=4, col=1)

    fig.update_layout(
        title=f"{symbol} Technical Analysis Dashboard",
        xaxis_title="Date",
        height=800,
        showlegend=True
    )

    fig.update_xaxes(rangeslider_visible=False)

    return fig


def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_portfolio(portfolio):
    try:
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(portfolio, f)
    except Exception:
        pass


def portfolio_tab():
    st.header("💼 My Portfolio")

    # Initialize portfolio in session state and load from file if not set
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = load_portfolio()

    # Add stock to portfolio
    with st.form("add_stock_form"):
        new_symbol = st.text_input("Add Stock Symbol to Portfolio", value="", key="portfolio_input")
        add_btn = st.form_submit_button("Add Stock")
        if add_btn and new_symbol:
            symbol = new_symbol.strip().upper()
            if symbol not in st.session_state.portfolio:
                st.session_state.portfolio.append(symbol)
                save_portfolio(st.session_state.portfolio)
                st.success(f"Added {symbol} to portfolio.")
            else:
                st.warning(f"{symbol} is already in your portfolio.")

    # Remove stock from portfolio
    if st.session_state.portfolio:
        remove_symbol = st.selectbox("Remove Stock", st.session_state.portfolio)
        if st.button("Remove Selected Stock"):
            st.session_state.portfolio.remove(remove_symbol)
            save_portfolio(st.session_state.portfolio)
            st.success(f"Removed {remove_symbol} from portfolio.")

    # Display portfolio table
    if st.session_state.portfolio:
        st.subheader("📋 Portfolio Overview")
        portfolio_data = []
        for symbol in st.session_state.portfolio:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="1mo", interval="1d")
                if df.empty:
                    continue
                df = calculate_technical_indicators(df)
                signals = generate_signals(df)
                latest = df.iloc[-1]
                info = ticker.info
                company_name = info.get('longName', symbol)
                sector = info.get('sector', 'N/A')
                price = latest['Close']
                rsi = latest['RSI']
                signal_summary = ", ".join([f"{s[1]} ({s[0]})" for s in signals]) if signals else "HOLD"
                portfolio_data.append({
                    "Symbol": symbol,
                    "Company": company_name,
                    "Sector": sector,
                    "Price": f"${price:.2f}",
                    "RSI": f"{rsi:.2f}",
                    "Signals": signal_summary
                })
            except Exception as e:
                portfolio_data.append({
                    "Symbol": symbol,
                    "Company": "Error",
                    "Sector": "-",
                    "Price": "-",
                    "RSI": "-",
                    "Signals": f"Error: {str(e)}"
                })
        st.dataframe(pd.DataFrame(portfolio_data))
    else:
        st.info("Your portfolio is empty. Add stocks to get started.")


def main():
    # Add tabs for Dashboard and Portfolio
    tab1, tab2 = st.tabs(["Dashboard", "Portfolio"])
    with tab1:
        st.title("📈 Stock Signals Analysis Dashboard")
        st.sidebar.title("Settings")

        # Input controls
        symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", help="Enter stock symbol (e.g., AAPL, GOOGL, TSLA)")
        period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
        period = st.sidebar.selectbox("Time Period", period_options, index=3)
        interval_options = ["1d", "5d", "1wk", "1mo"]
        interval = st.sidebar.selectbox("Data Interval", interval_options, index=0)

        # Auto-refresh option (checked by default, 60s)
        auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=True)
        if auto_refresh:
            st.sidebar.info("Dashboard will refresh every 60 seconds")
            st.query_params.update({"_": datetime.now().timestamp()})  # updated API

        if st.sidebar.button("Analyze Stock") or auto_refresh:
            try:
                # Fetch data
                with st.spinner(f"Fetching data for {symbol}..."):
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period=period, interval=interval)

                    if df.empty:
                        st.error(f"No data found for symbol {symbol}")
                        return

                    # Get company info
                    try:
                        info = ticker.info
                        company_name = info.get('longName', symbol)
                        sector = info.get('sector', 'N/A')
                        market_cap = info.get('marketCap', 'N/A')
                    except:
                        company_name = symbol
                        sector = 'N/A'
                        market_cap = 'N/A'

                # Calculate indicators
                df = calculate_technical_indicators(df)

                # Generate signals
                signals = generate_signals(df)

                # Display company info
                st.subheader(f"{company_name} ({symbol})")

                col1, col2, col3, col4 = st.columns(4)

                latest_price = df['Close'].iloc[-1]
                prev_close = df['Close'].iloc[-2]
                price_change = latest_price - prev_close
                price_change_pct = (price_change / prev_close) * 100

                with col1:
                    st.metric("Current Price", f"${latest_price:.2f}",
                              f"{price_change:+.2f} ({price_change_pct:+.2f}%)")

                with col2:
                    st.metric("Volume", f"{df['Volume'].iloc[-1]:,}")

                with col3:
                    st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")

                with col4:
                    st.metric("Sector", sector)

                # Display signals
                st.subheader("🚦 Trading Signals")

                if signals:
                    for indicator, signal, description in signals:
                        if signal == "BUY":
                            st.markdown(f"🟢 **{indicator}**: <span class='signal-buy'>{signal}</span> - {description}",
                                        unsafe_allow_html=True)
                        elif signal == "SELL":
                            st.markdown(f"🔴 **{indicator}**: <span class='signal-sell'>{signal}</span> - {description}",
                                        unsafe_allow_html=True)
                        else:
                            st.markdown(f"🟡 **{indicator}**: <span class='signal-hold'>{signal}</span> - {description}",
                                        unsafe_allow_html=True)
                else:
                    st.info("No strong signals detected. Consider holding current position.")

                # Display main chart
                st.subheader("📊 Technical Analysis Chart")
                fig = create_main_chart(df, symbol)
                st.plotly_chart(fig, use_container_width=True)

                # Technical indicators summary
                st.subheader("📋 Technical Indicators Summary")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Trend Indicators:**")
                    st.write(f"• SMA 20: ${df['SMA_20'].iloc[-1]:.2f}")
                    st.write(f"• SMA 50: ${df['SMA_50'].iloc[-1]:.2f}")
                    st.write(f"• EMA 12: ${df['EMA_12'].iloc[-1]:.2f}")
                    st.write(f"• EMA 26: ${df['EMA_26'].iloc[-1]:.2f}")

                    st.write("**Volatility Indicators:**")
                    st.write(f"• BB Upper: ${df['BB_upper'].iloc[-1]:.2f}")
                    st.write(f"• BB Lower: ${df['BB_lower'].iloc[-1]:.2f}")

                with col2:
                    st.write("**Momentum Indicators:**")
                    st.write(f"• RSI: {df['RSI'].iloc[-1]:.2f}")
                    st.write(f"• MACD: {df['MACD'].iloc[-1]:.4f}")
                    st.write(f"• MACD Signal: {df['MACD_signal'].iloc[-1]:.4f}")
                    st.write(f"• Stochastic %K: {df['Stoch_k'].iloc[-1]:.2f}")
                    st.write(f"• Stochastic %D: {df['Stoch_d'].iloc[-1]:.2f}")

                # Recent data table
                st.subheader("📈 Recent Price Data")
                recent_data = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10)
                st.dataframe(recent_data.style.format({
                    'Open': '${:.2f}',
                    'High': '${:.2f}',
                    'Low': '${:.2f}',
                    'Close': '${:.2f}',
                    'Volume': '{:,}'
                }))

                # Auto-refresh
                if auto_refresh:
                    st.experimental_rerun()

            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                st.info("Please check the stock symbol and try again.")

    with tab2:
        portfolio_tab()

    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.markdown("### How to use:")
    st.sidebar.markdown("1. Enter a stock symbol (e.g., AAPL)")
    st.sidebar.markdown("2. Select time period and interval")
    st.sidebar.markdown("3. Click 'Analyze Stock'")
    st.sidebar.markdown("4. Review signals and charts")

    st.sidebar.markdown("### Signal Types:")
    st.sidebar.markdown("🟢 **BUY**: Bullish signal")
    st.sidebar.markdown("🔴 **SELL**: Bearish signal")
    st.sidebar.markdown("🟡 **HOLD**: Neutral signal")


if __name__ == "__main__":
    main()