"""
Quantitative analysis functions for Task 2: Stock price analysis using TA-Lib and PyNance
"""
import pandas as pd
import talib
import matplotlib.pyplot as plt
import pynance as pn
from textblob import TextBlob

class QuantAnalysis:
    @staticmethod
    def load_stock_data(filepath):
        """Load stock price data with OHLCV columns."""
        try:
            df = pd.read_csv(filepath)
            required_cols = {"Open", "High", "Low", "Close", "Volume"}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")
            return df
        except Exception as e:
            print(f"Error loading stock data: {e}")
            return None

    @staticmethod
    def calculate_technical_indicators(df):
        """Calculate SMA, RSI, and MACD using TA-Lib."""
        try:
            df = df.copy()
            df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
            df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
            macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df['MACD'] = macd
            df['MACD_signal'] = macdsignal
            df['MACD_hist'] = macdhist
            return df
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            return df

    @staticmethod
    def plot_technical_indicators(df, ticker="Stock"):
        """Visualize Close price, SMA, RSI, and MACD."""
        try:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            # Price and SMA
            axes[0].plot(df['Close'], label='Close')
            axes[0].plot(df['SMA_20'], label='SMA 20')
            axes[0].set_title(f'{ticker} Close Price & SMA')
            axes[0].legend()
            # RSI
            axes[1].plot(df['RSI_14'], label='RSI 14', color='orange')
            axes[1].axhline(70, color='red', linestyle='--', alpha=0.5)
            axes[1].axhline(30, color='green', linestyle='--', alpha=0.5)
            axes[1].set_title('RSI 14')
            axes[1].legend()
            # MACD
            axes[2].plot(df['MACD'], label='MACD')
            axes[2].plot(df['MACD_signal'], label='Signal')
            axes[2].bar(df.index, df['MACD_hist'], label='Hist', color='gray', alpha=0.3)
            axes[2].set_title('MACD')
            axes[2].legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting technical indicators: {e}")

    @staticmethod
    def get_pynance_metrics(ticker, start, end):
        """Fetch financial metrics using PyNance."""
        try:
            data = pn.data.history(ticker, start, end)
            return data
        except Exception as e:
            print(f"Error fetching data from PyNance: {e}")
            return None

def load_news_data(news_path):
    df = pd.read_csv(news_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_stock_data(stock_path):
    stock_df = pd.read_csv(stock_path)
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    return stock_df

def merge_data(news_df, stock_df):
    merged_df = pd.merge(news_df, stock_df, on='date', how='inner')
    return merged_df

def get_sentiment(text):
    if pd.isna(text):
        return 0
    return TextBlob(str(text)).sentiment.polarity

def add_sentiment(merged_df):
    merged_df['sentiment'] = merged_df['headline'].apply(get_sentiment)
    return merged_df

def add_daily_return(merged_df):
    merged_df = merged_df.sort_values('date')
    merged_df['daily_return'] = merged_df['close'].pct_change()
    return merged_df

def aggregate_daily(merged_df):
    daily_sentiment = merged_df.groupby('date')['sentiment'].mean()
    daily_return = merged_df.groupby('date')['daily_return'].mean()
    return daily_sentiment, daily_return

def compute_correlation(daily_sentiment, daily_return):
    return daily_sentiment.corr(daily_return)
