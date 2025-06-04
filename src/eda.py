# eda.py
"""
Modular EDA functions for Financial News Analysis
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from textblob import TextBlob


def load_data(filepath):
    """Load CSV data into a DataFrame."""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None


def add_headline_length(df, headline_col="headline"):
    """Add a column for headline length."""
    try:
        df = df.copy()
        df["headline_length"] = df[headline_col].str.len()
        return df
    except Exception as e:
        print(f"Error adding headline length: {e}")
        return df


def describe_headline_length(df):
    df['headline_length'] = df['headline'].astype(str).apply(len)
    desc = df['headline_length'].describe()
    plt.figure(figsize=(8,4))
    sns.histplot(df['headline_length'], bins=30, kde=True)
    plt.title('Headline Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.show()
    return desc


def count_articles_per_publisher(df, publisher_col="publisher"):
    """Return article counts per publisher."""
    try:
        return df[publisher_col].value_counts()
    except Exception as e:
        print(f"Error counting articles per publisher: {e}")
        return None


def plot_articles_per_day(df, date_col="date"):
    """Plot number of articles per day, handling tz-aware and tz-naive values."""
    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if hasattr(df[date_col], 'dt'):
            df[date_col] = df[date_col].dt.tz_localize(None)
        daily_counts = df[date_col].dt.date.value_counts().sort_index()
        plt.figure(figsize=(10,4))
        daily_counts.plot()
        plt.title("Articles per Day")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting articles per day: {e}")


def extract_top_keywords(df, text_col="headline", n_keywords=20):
    """Extract top keywords from text column."""
    try:
        vectorizer = CountVectorizer(stop_words="english", max_features=n_keywords)
        X = vectorizer.fit_transform(df[text_col].fillna(""))
        return vectorizer.get_feature_names_out()
    except Exception as e:
        print(f"Error extracting top keywords: {e}")
        return []


def plot_weekly_article_count(df, date_col="date"):
    """Plot weekly article count."""
    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        weekly_counts = df.set_index(date_col).resample("W").size()
        plt.figure(figsize=(10,4))
        weekly_counts.plot()
        plt.title("Weekly Article Count")
        plt.xlabel("Week")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting weekly article count: {e}")


def publisher_domain_analysis(df, publisher_col="publisher"):
    """Return value counts of publisher domains."""
    try:
        df = df.copy()
        df["publisher_domain"] = df[publisher_col].str.extract(r'@([\w\.-]+)')
        return df["publisher_domain"].value_counts()
    except Exception as e:
        print(f"Error analyzing publisher domains: {e}")
        return None


def parse_date_column(df, date_col="date"):
    """Parse a date column, handle errors, and normalize tz-aware/naive datetimes."""
    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if hasattr(df[date_col], 'dt'):
            df[date_col] = df[date_col].dt.tz_localize(None)
        return df
    except Exception as e:
        print(f"Error parsing date column: {e}")
        return df


def plot_top_publishers(df, top_n=10):
    publisher_counts = df['publisher'].value_counts()
    plt.figure(figsize=(10,5))
    sns.barplot(x=publisher_counts.head(top_n).index, y=publisher_counts.head(top_n).values)
    plt.title(f'Top {top_n} Publishers by Article Count')
    plt.ylabel('Number of Articles')
    plt.xlabel('Publisher')
    plt.xticks(rotation=45)
    plt.show()
    return publisher_counts.head(top_n)


def merge_news_stock(news_df, stock_df):
    news_df['date'] = pd.to_datetime(news_df['date'])
    stock_df['date'] = pd.to_datetime(stock_df['date'])
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


def correlation_analysis(daily_sentiment, daily_return):
    correlation = daily_sentiment.corr(daily_return)
    print(f'Pearson correlation between daily sentiment and stock returns: {correlation:.4f}')
    return correlation


def plot_sentiment_vs_return(daily_sentiment, daily_return):
    plt.figure(figsize=(12,5))
    plt.plot(daily_sentiment.index, daily_sentiment.values, label='Avg Daily Sentiment')
    plt.plot(daily_return.index, daily_return.values, label='Daily Return')
    plt.legend()
    plt.title('Daily Sentiment vs. Stock Return')
    plt.xlabel('Date')
    plt.show()
