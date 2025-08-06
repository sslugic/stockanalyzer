import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import ta

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

    return df


def compute_rsi(prices, length=14):
    if len(prices) < length + 1:
        return None
    gains = []
    losses = []
    for i in range(1, length + 1):
        delta = prices[i] - prices[i - 1]
        if delta >= 0:
            gains.append(delta)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-delta)
    avg_gain = sum(gains) / length
    avg_loss = sum(losses) / length
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def ema(prices, length):
    multiplier = 2 / (length + 1)
    ema_values = [prices[0]]
    for price in prices[1:]:
        ema_values.append((price - ema_values[-1]) * multiplier + ema_values[-1])
    return ema_values

def macd_signal(prices):
    min_required = max(12, 26) + 4
    if len(prices) < min_required:
        return "N", 0, "Flat/Neutral"
    ema_fast = np.array(ema(prices, 12))
    ema_slow = np.array(ema(prices, 26))
    min_len = min(len(ema_fast), len(ema_slow))
    macd_line = ema_fast[-min_len:] - ema_slow[-min_len:]
    signal_line = np.array(ema(macd_line.tolist(), 9))
    hist_len = min(len(macd_line), len(signal_line))
    macd_line = macd_line[-hist_len:]
    signal_line = signal_line[-hist_len:]
    histogram = macd_line - signal_line
    recent_hist = histogram[-3:]
    current_hist = histogram[-1]
    prev_hist = histogram[-2]
    macd_now = macd_line[-1]
    signal_now = signal_line[-1]
    slope = np.diff(histogram)
    slope_now = slope[-1] if len(slope) > 0 else 0
    if current_hist > 0 and slope_now > 0:
        macd_confidence = "Strong Bullish"
    elif current_hist > 0 and slope_now < 0:
        macd_confidence = "Fading Bullish"
    elif current_hist < 0 and slope_now < 0:
        macd_confidence = "Strong Bearish"
    elif current_hist < 0 and slope_now > 0:
        macd_confidence = "Fading Bearish"
    else:
        macd_confidence = "Flat/Neutral"
    if macd_now > signal_now:
        if all(h > 0 for h in recent_hist) and recent_hist[-1] > recent_hist[1]:
            status = "Y"
        elif current_hist > 0:
            status = "W"
        else:
            status = "N"
    elif current_hist > 0 and prev_hist < 0:
        status = "Y"
    else:
        status = "N"
    score = {"Y": 2, "W": 1, "N": 0}[status]
    return status, score, macd_confidence

def get_vix():
    try:
        vix = yf.Ticker("^VIX").history(period="25d")
        return vix["Close"].iloc[-1]
    except Exception:
        return None

def qqq_momentum():
    try:
        hist = yf.Ticker("QQQ").history(period="50d")["Close"]
        return hist.iloc[-1] > hist.iloc[-2]
    except Exception:
        return False

def generate_signals(df, symbol=None):
    """Generate buy/sell/hold signals using improved thresholds."""
    signals = []
    scores = {"BUY": 0, "SELL": 0, "HOLD": 0}
    prices = df['Close'].tolist()
    rsi_val = compute_rsi(prices[-15:]) if len(prices) >= 15 else None
    macd_status, macd_score, macd_confidence = macd_signal(prices[-90:] if len(prices) >= 90 else prices)
    vix_val = get_vix()
    qqq_up = qqq_momentum()
    price_now = prices[-1] if prices else None

    macd_is_bullish = (macd_score == 2 and macd_confidence == "Strong Bullish")

    # --- RSI ---
    if rsi_val is not None:
        if rsi_val < 30:
            signals.append(("RSI", "BUY", f"RSI oversold at {rsi_val:.2f}"))
            scores["BUY"] += 1
        elif rsi_val > 70:
            signals.append(("RSI", "SELL", f"RSI overbought at {rsi_val:.2f}"))
            scores["SELL"] += 1
        elif 30 <= rsi_val <= 70:
            signals.append(("RSI", "HOLD", f"RSI neutral at {rsi_val:.2f}"))
            scores["HOLD"] += 1

    # --- VIX ---
    if vix_val is not None:
        if vix_val < 15:
            signals.append(("VIX", "BUY", f"Very low volatility (VIX={vix_val:.2f})"))
            scores["BUY"] += 1
        elif 15 <= vix_val <= 20:
            signals.append(("VIX", "HOLD", f"Moderate volatility (VIX={vix_val:.2f})"))
            scores["HOLD"] += 1
        elif vix_val > 20:
            signals.append(("VIX", "SELL", f"High volatility (VIX={vix_val:.2f})"))
            scores["SELL"] += 1

    # --- MACD ---
    if macd_score == 2 and macd_confidence == "Strong Bullish":
        signals.append(("MACD", "BUY", f"MACD Strong Bullish ({macd_confidence})"))
        scores["BUY"] += 1
    elif macd_score == 1 and macd_confidence in ["Fading Bullish"]:
        signals.append(("MACD", "HOLD", f"MACD Fading Bullish ({macd_confidence})"))
        scores["HOLD"] += 1
    elif macd_score == 0 and macd_confidence in ["Strong Bearish", "Fading Bearish"]:
        signals.append(("MACD", "SELL", f"MACD Bearish ({macd_confidence})"))
        scores["SELL"] += 1
    else:
        signals.append(("MACD", "HOLD", f"MACD Neutral ({macd_confidence})"))
        scores["HOLD"] += 1

    # --- QQQ ---
    if qqq_up:
        signals.append(("QQQ", "BUY", "QQQ trending up (bullish market)"))
        scores["BUY"] += 1
    else:
        signals.append(("QQQ", "SELL", "QQQ flat/down (bearish market)"))
        scores["SELL"] += 1

    # --- Stochastic Oscillator ---
    stoch_k = df['Stoch_k'].iloc[-1] if 'Stoch_k' in df.columns else None
    stoch_d = df['Stoch_d'].iloc[-1] if 'Stoch_d' in df.columns else None
    if stoch_k is not None and stoch_d is not None:
        if stoch_k < 20 and stoch_d < 20:
            signals.append(("Stochastic", "BUY", f"Stochastic oversold (K={stoch_k:.2f}, D={stoch_d:.2f})"))
            scores["BUY"] += 1
        elif stoch_k > 80 and stoch_d > 80:
            signals.append(("Stochastic", "SELL", f"Stochastic overbought (K={stoch_k:.2f}, D={stoch_d:.2f})"))
            scores["SELL"] += 1
        else:
            signals.append(("Stochastic", "HOLD", f"Stochastic neutral (K={stoch_k:.2f}, D={stoch_d:.2f})"))
            scores["HOLD"] += 1

    return signals, scores

def score_signals(scores, macd_is_bullish):
    buy_count = scores.get("BUY", 0)
    sell_count = scores.get("SELL", 0)
    hold_count = scores.get("HOLD", 0)
    total_signals = buy_count + sell_count + hold_count

    # Only allow BUY if MACD is bullish
    if macd_is_bullish:
        if buy_count == total_signals and total_signals > 0:
            return "STRONG BUY"
        elif buy_count >= 4:
            return "BUY"
        elif sell_count >= 4:
            return "SELL"
        elif hold_count >= 2 and hold_count >= buy_count and hold_count >= sell_count:
            return "HOLD"
        elif buy_count <= 1:
            return "SELL"
        else:
            return "HOLD"
    else:
        if sell_count >= 4:
            return "SELL"
        elif hold_count >= 2 and hold_count >= buy_count and hold_count >= sell_count:
            return "HOLD"
        elif buy_count <= 1:
            return "SELL"
        else:
            return "HOLD"


def create_main_chart(df, symbol):
    """Create the main price chart with technical indicators"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Stock Price', 'MACD', 'RSI'),
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


def get_overall_action(df):
    """Centralized logic for signals and overall action, used by both dashboard and portfolio."""
    signals, scores = generate_signals(df)
    # Find MACD signal type from signals list (not by recalculating)
    macd_signal_type = None
    for s in signals:
        if s[0] == "MACD":
            macd_signal_type = s[1]
            break
    # If MACD signal is "BUY", treat as bullish for scoring
    macd_is_bullish = macd_signal_type == "BUY"
    # Always pass the same scores and macd_is_bullish to score_signals
    overall_action = score_signals(scores, macd_is_bullish)
    return signals, overall_action

def portfolio_tab():
    st.header("💼 My Portfolio")

    # Initialize portfolio in session state
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []

    # Add stock to portfolio
    with st.form("add_stock_form"):
        new_symbol = st.text_input("Add Stock Symbol to Portfolio", value="", key="portfolio_input")
        add_btn = st.form_submit_button("Add Stock")
        if add_btn and new_symbol:
            symbol = new_symbol.strip().upper()
            if symbol not in st.session_state.portfolio:
                st.session_state.portfolio.append(symbol)
                st.success(f"Added {symbol} to portfolio.")
            else:
                st.warning(f"{symbol} is already in your portfolio.")

    # Remove stock from portfolio
    if st.session_state.portfolio:
        remove_symbol = st.selectbox("Remove Stock", st.session_state.portfolio)
        if st.button("Remove Selected Stock"):
            st.session_state.portfolio.remove(remove_symbol)
            st.success(f"Removed {remove_symbol} from portfolio.")

    # Display portfolio table
    if st.session_state.portfolio:
        st.subheader("📋 Portfolio Overview")
        portfolio_rows = []
        for symbol in st.session_state.portfolio:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="1mo", interval="1d")
                if df is None or df.empty or 'Close' not in df.columns or len(df['Close']) == 0:
                    portfolio_rows.append({
                        "Symbol": symbol,
                        "Company": "No Data",
                        "Sector": "-",
                        "Price": "-",
                        "RSI": "-",
                        "Signals": "No data found",
                        "Action": "HOLD"
                    })
                    continue
                df = calculate_technical_indicators(df)
                signals, overall_action = get_overall_action(df)
                info = {}
                try:
                    info = ticker.info
                except Exception:
                    info = {}
                company_name = info.get('longName', symbol)
                sector = info.get('sector', 'N/A')
                price = df['Close'].iloc[-1] if not df['Close'].empty else "-"
                rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns and not df['RSI'].empty else "-"
                # Use the same Trading Signals data as Dashboard (exclude Summary)
                trading_signals = []
                for indicator, signal, description in signals:
                    if indicator == "Summary":
                        continue
                    # Format similar to dashboard
                    if signal == "BUY":
                        trading_signals.append(f"🟢 {indicator}: {signal} - {description}")
                    elif signal == "SELL":
                        trading_signals.append(f"🔴 {indicator}: {signal} - {description}")
                    else:
                        trading_signals.append(f"🟡 {indicator}: {signal} - {description}")
                signals_display = "\n".join(trading_signals) if trading_signals else "HOLD"
                portfolio_rows.append({
                    "Symbol": symbol,
                    "Company": company_name,
                    "Sector": sector,
                    "Price": f"${price:.2f}" if isinstance(price, (float, int, np.float64, np.int64)) else price,
                    "RSI": f"{rsi:.2f}" if isinstance(rsi, (float, int, np.float64, np.int64)) else rsi,
                    "Trading Signals": signals_display,
                    "Action": overall_action
                })
            except Exception as e:
                portfolio_rows.append({
                    "Symbol": symbol,
                    "Company": "Error",
                    "Sector": "-",
                    "Price": "-",
                    "RSI": "-",
                    "Trading Signals": f"Error: {str(e)}",
                    "Action": "HOLD"
                })

        if portfolio_rows:
            df_portfolio = pd.DataFrame(portfolio_rows)
            st.dataframe(df_portfolio)
        else:
            st.info("No valid data for your portfolio stocks.")
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

        # Disable auto-refresh and set box to unchecked by default
        auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=False)
        if auto_refresh:
            st.sidebar.info("Dashboard will refresh every 60 seconds")
            # No auto-refresh logic

        analyze_clicked = st.sidebar.button("Analyze Stock")

        # Use a session state flag to keep context after rerun
        if "last_symbol" not in st.session_state:
            st.session_state["last_symbol"] = symbol
        if analyze_clicked:
            st.session_state["last_symbol"] = symbol

        # Use last_symbol for display and add-to-portfolio
        display_symbol = st.session_state["last_symbol"]

        try:
            # Fetch data
            with st.spinner(f"Fetching data for {display_symbol}..."):
                ticker = yf.Ticker(display_symbol)
                df = ticker.history(period=period, interval=interval)

                if df.empty:
                    st.error(f"No data found for symbol {display_symbol}")
                    return

                # Get company info
                try:
                    info = ticker.info
                    company_name = info.get('longName', display_symbol)
                    sector = info.get('sector', 'N/A')
                    market_cap = info.get('marketCap', 'N/A')
                except:
                    company_name = display_symbol
                    sector = 'N/A'
                    market_cap = 'N/A'

            # Calculate indicators
            df = calculate_technical_indicators(df)
            # Generate signals
            signals, overall_action = get_overall_action(df)
            # Display company info
            st.subheader(f"{company_name} ({display_symbol})")
            # --- Add to Portfolio button between subheader and metrics ---
            add_clicked = st.button(f"Add {display_symbol} to Portfolio", key="add_portfolio_btn")
            if add_clicked:
                if "portfolio" not in st.session_state:
                    st.session_state.portfolio = []
                if display_symbol not in st.session_state.portfolio:
                    st.session_state.portfolio.append(display_symbol)
                    st.success(f"Added {display_symbol} to portfolio.")
                    # Force rerun to reload context and update portfolio
                    st.experimental_rerun()
                else:
                    st.warning(f"{display_symbol} is already in your portfolio.")
            # Metrics
            col1, col2, col3, col4 = st.columns(4)

            latest_price = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2]
            price_change = latest_price - prev_close
            price_change_pct = (price_change / prev_close) * 100

            with col1:
                st.metric("Current Price", f"${latest_price:.2f}",
                          f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
            with col2:
                st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
            with col3:
                st.metric("Sector", sector)
            with col4:
                st.metric("Action", overall_action)

            # Display signals (sync with portfolio)
            st.subheader("🚦 Trading Signals")
            if signals:
                for indicator, signal, description in signals:
                    if indicator == "Summary":
                        continue  # Remove Summary from Dashboard
                    # Use smaller font for individual signals
                    if signal == "BUY":
                        st.markdown(f"<span style='font-size:0.95em;'>🟢 <b>{indicator}</b>: <span class='signal-buy'>{signal}</span> - {description}</span>", unsafe_allow_html=True)
                    elif signal == "SELL":
                        st.markdown(f"<span style='font-size:0.95em;'>🔴 <b>{indicator}</b>: <span class='signal-sell'>{signal}</span> - {description}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span style='font-size:0.95em;'>🟡 <b>{indicator}</b>: <span class='signal-hold'>{signal}</span> - {description}</span>", unsafe_allow_html=True)
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
                st.rerun()

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