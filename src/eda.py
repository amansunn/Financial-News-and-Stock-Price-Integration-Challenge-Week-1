# eda.py
"""
Modular EDA functions for Financial News Analysis
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


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
    """Return descriptive statistics for headline length."""
    try:
        return df["headline_length"].describe()
    except Exception as e:
        print(f"Error describing headline length: {e}")
        return None


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
