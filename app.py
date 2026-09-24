# ============================================================
# REAL-TIME / LATEST-DATA STOCK PRICE PREDICTION
# LSTM + TRANSFORMER + FINANCIAL SENTIMENT
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf

from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LSTM,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
    Concatenate
)

from tensorflow.keras.models import Model

from transformers import pipeline


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Real-Time Stock Prediction",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "📈 Real-Time Stock Price Prediction"
)

st.markdown(
    """
    ### Hybrid Deep Learning Model

    **LSTM + Transformer + FinBERT Sentiment Analysis**

    The system automatically fetches the latest available
    stock-market data and recent financial-news sentiment.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Prediction Settings")

stock_symbol = st.sidebar.text_input(
    "Stock Symbol",
    "AAPL"
).upper()

# Number of historical trading days used for model training
history_period = st.sidebar.selectbox(
    "Historical Training Period",
    [
        "1y",
        "2y",
        "5y"
    ],
    index=1
)

sequence_length = st.sidebar.slider(
    "Sequence Length",
    30,
    100,
    60
)

epochs = st.sidebar.slider(
    "Training Epochs",
    5,
    50,
    10
)

batch_size = st.sidebar.selectbox(
    "Batch Size",
    [16, 32, 64],
    index=1
)


run_prediction = st.sidebar.button(
    "🚀 Fetch Latest Data & Predict"
)


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        rate=0.1,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.ffn = tf.keras.Sequential([
            Dense(
                ff_dim,
                activation="relu"
            ),
            Dense(embed_dim)
        ])

        self.norm1 = LayerNormalization(
            epsilon=1e-6
        )

        self.norm2 = LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)


    def call(
        self,
        inputs,
        training=False
    ):

        attention = self.attention(
            inputs,
            inputs
        )

        attention = self.dropout1(
            attention,
            training=training
        )

        x = self.norm1(
            inputs + attention
        )

        ffn = self.ffn(x)

        ffn = self.dropout2(
            ffn,
            training=training
        )

        return self.norm2(
            x + ffn
        )


    def get_config(self):

        config = super().get_config()

        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })

        return config


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    prices,
    period=14
):

    delta = prices.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(
        period
    ).mean()

    avg_loss = loss.rolling(
        period
    ).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi


# ============================================================
# DOWNLOAD LATEST MARKET DATA
# ============================================================

@st.cache_data(ttl=60)
def get_latest_stock_data(
    symbol,
    period
):

    ticker = yf.Ticker(symbol)

    data = ticker.history(
        period=period,
        interval="1d",
        auto_adjust=False
    )

    if data.empty:

        raise ValueError(
            "No stock data found."
        )

    data = data[
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    ]

    return data.dropna()


# ============================================================
# LATEST PRICE
# ============================================================

def get_latest_price(
    symbol
):

    ticker = yf.Ticker(symbol)

    data = ticker.history(
        period="1d",
        interval="1m"
    )

    if not data.empty:

        return float(
            data["Close"].iloc[-1]
        )

    return None


# ============================================================
# FINBERT
# ============================================================

@st.cache_resource
def load_finbert():

    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert"
    )


# ============================================================
# LATEST NEWS SENTIMENT
# ============================================================

def get_latest_sentiment(
    symbol
):

    ticker = yf.Ticker(symbol)

    try:

        news = ticker.news

    except Exception:

        news = []

    if not news:

        return 0.0, []


    analyzer = load_finbert()

    scores = []

    news_results = []


    for article in news[:10]:

        try:

            content = article.get(
                "content",
                {}
            )

            title = content.get(
                "title",
                ""
            )

            if not title:

                title = article.get(
                    "title",
                    ""
                )

            if not title:

                continue


            result = analyzer(
                title
            )[0]


            label = (
                result["label"]
                .lower()
            )

            confidence = result[
                "score"
            ]


            if label == "positive":

                sentiment_score = (
                    confidence
                )

            elif label == "negative":

                sentiment_score = (
                    -confidence
                )

            else:

                sentiment_score = 0.0


            scores.append(
                sentiment_score
            )


            news_results.append({
                "Title": title,
                "Sentiment": label,
                "Confidence": round(
                    confidence,
                    3
                )
            })


        except Exception:

            continue


    if not scores:

        return 0.0, []


    average_sentiment = float(
        np.mean(scores)
    )


    return (
        average_sentiment,
        news_results
    )


# ============================================================
# BUILD HYBRID MODEL
# ============================================================

def build_hybrid_model(
    price_shape,
    sentiment_shape
):

    # --------------------------------------------------------
    # PRICE INPUT
    # --------------------------------------------------------

    price_input = Input(
        shape=price_shape,
        name="Price_Input"
    )


    # LSTM
    x = LSTM(
        64,
        return_sequences=True
    )(price_input)


    x = Dropout(
        0.2
    )(x)


    # Transformer
    x = TransformerBlock(
        embed_dim=64,
        num_heads=4,
        ff_dim=128
    )(x)


    # Pooling
    x = GlobalAveragePooling1D()(
        x
    )


    price_branch = Dense(
        32,
        activation="relu"
    )(x)


    # --------------------------------------------------------
    # SENTIMENT INPUT
    # --------------------------------------------------------

    sentiment_input = Input(
        shape=sentiment_shape,
        name="Sentiment_Input"
    )


    sentiment_branch = Dense(
        16,
        activation="relu"
    )(
        sentiment_input
    )


    # --------------------------------------------------------
    # HYBRID FUSION
    # --------------------------------------------------------

    merged = Concatenate()([
        price_branch,
        sentiment_branch
    ])


    x = Dense(
        32,
        activation="relu"
    )(merged)


    x = Dropout(
        0.2
    )(x)


    x = Dense(
        16,
        activation="relu"
    )(x)


    output = Dense(
        1,
        name="Predicted_Close"
    )(x)


    model = Model(
        inputs=[
            price_input,
            sentiment_input
        ],
        outputs=output
    )


    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )


    return model


# ============================================================
# MAIN
# ============================================================

if run_prediction:

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    st.subheader(
        "📡 Fetching Latest Market Data"
    )


    with st.spinner(
        f"Downloading latest {stock_symbol} data..."
    ):

        try:

            data = get_latest_stock_data(
                stock_symbol,
                history_period
            )

        except Exception as e:

            st.error(
                f"Unable to fetch data: {e}"
            )

            st.stop()


    # --------------------------------------------------------
    # CURRENT PRICE
    # --------------------------------------------------------

    latest_price = get_latest_price(
        stock_symbol
    )


    if latest_price is None:

        latest_price = float(
            data["Close"].iloc[-1]
        )


    latest_date = data.index[-1]


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Latest Price",
            f"${latest_price:.2f}"
        )


    with col2:

        st.metric(
            "Latest Available Date",
            str(latest_date.date())
        )


    with col3:

        st.metric(
            "Trading Days",
            len(data)
        )


    # --------------------------------------------------------
    # STEP 2
    # TECHNICAL INDICATORS
    # --------------------------------------------------------

    data["MA_20"] = (
        data["Close"]
        .rolling(20)
        .mean()
    )


    data["RSI"] = calculate_rsi(
        data["Close"]
    )


    data.dropna(
        inplace=True
    )


    # --------------------------------------------------------
    # DATA DISPLAY
    # --------------------------------------------------------

    st.subheader(
        "📊 Latest Market Data"
    )


    st.dataframe(
        data.tail(10),
        use_container_width=True
    )


    # --------------------------------------------------------
    # PRICE CHART
    # --------------------------------------------------------

    st.subheader(
        "📈 Stock Price + Moving Average"
    )


    fig, ax = plt.subplots(
        figsize=(13, 5)
    )


    ax.plot(
        data.index,
        data["Close"],
        label="Close Price"
    )


    ax.plot(
        data.index,
        data["MA_20"],
        label="20-Day MA"
    )


    ax.set_title(
        f"{stock_symbol} Latest Market Data"
    )


    ax.set_xlabel(
        "Date"
    )


    ax.set_ylabel(
        "Price (USD)"
    )


    ax.legend()


    ax.grid(
        True,
        linestyle=":"
    )


    st.pyplot(fig)


    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    st.subheader(
        "📉 RSI Indicator"
    )


    fig_rsi, ax_rsi = plt.subplots(
        figsize=(13, 4)
    )


    ax_rsi.plot(
        data.index,
        data["RSI"],
        label="RSI"
    )


    ax_rsi.axhline(
        70,
        linestyle="--",
        label="Overbought"
    )


    ax_rsi.axhline(
        30,
        linestyle="--",
        label="Oversold"
    )


    ax_rsi.set_title(
        f"{stock_symbol} Relative Strength Index"
    )


    ax_rsi.legend()


    ax_rsi.grid(
        True,
        linestyle=":"
    )


    st.pyplot(fig_rsi)


    # --------------------------------------------------------
    # STEP 3
    # SENTIMENT
    # --------------------------------------------------------

    st.subheader(
        "📰 Latest Financial News Sentiment"
    )


    with st.spinner(
        "Analyzing latest financial news..."
    ):

        sentiment_score, news = (
            get_latest_sentiment(
                stock_symbol
            )
        )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Sentiment Score",
            f"{sentiment_score:.4f}"
        )


    with col2:

        if sentiment_score > 0.1:

            sentiment_text = (
                "Positive 🟢"
            )

        elif sentiment_score < -0.1:

            sentiment_text = (
                "Negative 🔴"
            )

        else:

            sentiment_text = (
                "Neutral 🟡"
            )


        st.metric(
            "Overall Sentiment",
            sentiment_text
        )


    if news:

        news_df = pd.DataFrame(
            news
        )

        st.dataframe(
            news_df,
            use_container_width=True
        )

    else:

        st.warning(
            "No recent financial news was available."
        )


    # --------------------------------------------------------
    # STEP 4
    # PREPARE FEATURES
    # --------------------------------------------------------

    features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "MA_20",
        "RSI"
    ]


    price_scaler = MinMaxScaler()


    sentiment_scaler = MinMaxScaler()


    scaled_prices = (
        price_scaler.fit_transform(
            data[features]
        )
    )


    # Historical sentiment:
    # for this demonstration, the latest
    # available news sentiment is aligned
    # with the downloaded training window.

    sentiment_values = np.full(
        (len(data), 1),
        sentiment_score
    )


    scaled_sentiment = (
        sentiment_scaler.fit_transform(
            sentiment_values
        )
    )


    # --------------------------------------------------------
    # STEP 5
    # SEQUENCES
    # --------------------------------------------------------

    X_price = []

    X_sentiment = []

    y = []


    for i in range(
        sequence_length,
        len(data)
    ):

        X_price.append(
            scaled_prices[
                i-sequence_length:i
            ]
        )


        X_sentiment.append(
            scaled_sentiment[i]
        )


        # Close column = index 3
        y.append(
            scaled_prices[i, 3]
        )


    X_price = np.array(
        X_price
    )


    X_sentiment = np.array(
        X_sentiment
    )


    y = np.array(
        y
    )


    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    train_size = int(
        len(X_price) * 0.8
    )


    X_train = X_price[
        :train_size
    ]


    X_test = X_price[
        train_size:
    ]


    S_train = X_sentiment[
        :train_size
    ]


    S_test = X_sentiment[
        train_size:
    ]


    y_train = y[
        :train_size
    ]


    y_test = y[
        train_size:
    ]


    # --------------------------------------------------------
    # STEP 6
    # BUILD MODEL
    # --------------------------------------------------------

    st.subheader(
        "🧠 Training Hybrid LSTM + Transformer"
    )


    model = build_hybrid_model(
        (
            X_train.shape[1],
            X_train.shape[2]
        ),
        (
            S_train.shape[1],
        )
    )


    with st.spinner(
        "Training model with latest market data..."
    ):

        history = model.fit(
            [
                X_train,
                S_train
            ],
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(
                [
                    X_test,
                    S_test
                ],
                y_test
            ),
            verbose=0
        )


    st.success(
        "Model training completed."
    )


    # --------------------------------------------------------
    # STEP 7
    # TEST PREDICTIONS
    # --------------------------------------------------------

    pred_scaled = model.predict(
        [
            X_test,
            S_test
        ],
        verbose=0
    )


    # Convert Close back to dollars

    dummy_pred = np.zeros(
        (
            len(pred_scaled),
            len(features)
        )
    )


    dummy_actual = np.zeros(
        (
            len(y_test),
            len(features)
        )
    )


    dummy_pred[:, 3] = (
        pred_scaled[:, 0]
    )


    dummy_actual[:, 3] = (
        y_test
    )


    predictions = (
        price_scaler.inverse_transform(
            dummy_pred
        )[:, 3]
    )


    actual = (
        price_scaler.inverse_transform(
            dummy_actual
        )[:, 3]
    )


    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    mse = mean_squared_error(
        actual,
        predictions
    )


    rmse = np.sqrt(
        mse
    )


    mae = mean_absolute_error(
        actual,
        predictions
    )


    st.subheader(
        "📊 Model Performance"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "MSE",
            f"{mse:.4f}"
        )


    with c2:

        st.metric(
            "RMSE",
            f"{rmse:.4f}"
        )


    with c3:

        st.metric(
            "MAE",
            f"{mae:.4f}"
        )


    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    st.subheader(
        "📈 Actual vs Predicted"
    )


    fig_pred, ax_pred = plt.subplots(
        figsize=(13, 6)
    )


    ax_pred.plot(
        actual,
        label="Actual"
    )


    ax_pred.plot(
        predictions,
        label="Predicted",
        linestyle="--"
    )


    ax_pred.set_title(
        f"{stock_symbol} Actual vs Predicted"
    )


    ax_pred.set_xlabel(
        "Trading Days"
    )


    ax_pred.set_ylabel(
        "Price (USD)"
    )


    ax_pred.legend()


    ax_pred.grid(
        True,
        linestyle=":"
    )


    st.pyplot(fig_pred)


    # --------------------------------------------------------
    # STEP 8
    # NEXT SESSION PREDICTION
    # --------------------------------------------------------

    st.subheader(
        "🔮 Next Trading Session Prediction"
    )


    last_sequence = scaled_prices[
        -sequence_length:
    ]


    last_sequence = np.expand_dims(
        last_sequence,
        axis=0
    )


    latest_sentiment_scaled = (
        scaled_sentiment[-1:]
    )


    next_scaled = model.predict(
        [
            last_sequence,
            latest_sentiment_scaled
        ],
        verbose=0
    )


    dummy_next = np.zeros(
        (
            1,
            len(features)
        )
    )


    dummy_next[
        0, 3
    ] = next_scaled[0, 0]


    predicted_price = (
        price_scaler.inverse_transform(
            dummy_next
        )[0, 3]
    )


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    difference = (
        predicted_price -
        latest_price
    )


    percentage = (
        difference /
        latest_price
    ) * 100


    if percentage > 1:

        outlook = "📈 Bullish"

    elif percentage < -1:

        outlook = "📉 Bearish"

    else:

        outlook = "➡️ Neutral"


    st.success(
        f"### Predicted Next Price: "
        f"${predicted_price:.2f}"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Latest Price",
            f"${latest_price:.2f}"
        )


    with c2:

        st.metric(
            "Predicted Price",
            f"${predicted_price:.2f}",
            f"{percentage:+.2f}%"
        )


    with c3:

        st.metric(
            "Market Outlook",
            outlook
        )


    st.caption(
        "Prediction is generated from the latest "
        "available market data and recent financial "
        "news sentiment. It is not financial advice."
    )


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        "Enter a stock ticker and click "
        "**Fetch Latest Data & Predict**."
    )


    st.markdown(
        """
        ### 🔬 Model Components

        **Market Data**
        - Open
        - High
        - Low
        - Close
        - Volume

        **Technical Indicators**
        - 20-Day Moving Average
        - RSI

        **Deep Learning**
        - LSTM
        - Transformer
        - Multi-Head Attention

        **NLP**
        - FinBERT financial sentiment

        **Fusion**
        - Stock features + sentiment

        **Output**
        - Next trading-session predicted price
        """
    )