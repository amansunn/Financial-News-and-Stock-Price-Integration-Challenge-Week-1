# Exploratory Data Analysis (EDA)
# This notebook covers descriptive statistics and visualizations for the news dataset.

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from textblob import TextBlob

# Display settings
pd.set_option('display.max_columns', None)
sns.set(style='whitegrid')

## Load Data

# Example: df = pd.read_csv('../data/news.csv')
df = pd.read_csv('../data/news.csv')  # Update path as needed
stock_df = pd.read_csv('../data/yfinance_data/AAPL_historical_data.csv')  # Update path as needed

# The rest of the analysis will be imported from src modules.