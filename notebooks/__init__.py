# Exploratory Data Analysis (EDA)
# This notebook covers descriptive statistics and visualizations for the news dataset.

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Display settings
pd.set_option('display.max_columns', None)
sns.set(style='whitegrid')

## Load Data

# Example: df = pd.read_csv('../data/news.csv')
df = pd.read_csv('../data/news.csv')  # Update path as needed
df.head()

## Descriptive Statistics

### Headline/Text Length

# Assuming 'headline' column exists
df['headline_length'] = df['headline'].astype(str).apply(len)
df['headline_length'].describe()

# Visualize headline length distribution
plt.figure(figsize=(8,4))
sns.histplot(df['headline_length'], bins=30, kde=True)
plt.title('Headline Length Distribution')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.show()

### Articles per Publisher

# Assuming 'publisher' column exists
publisher_counts = df['publisher'].value_counts()
publisher_counts.head(10)

# Visualize top publishers
plt.figure(figsize=(10,5))
sns.barplot(x=publisher_counts.head(10).index, y=publisher_counts.head(10).values)
plt.title('Top 10 Publishers by Article Count')
plt.ylabel('Number of Articles')
plt.xlabel('Publisher')
plt.xticks(rotation=45)
plt.show()

### Publication Dates Trend

# Assuming 'date' column exists
df['date'] = pd.to_datetime(df['date'])
articles_per_day = df.groupby(df['date'].dt.date).size()
articles_per_day.plot(figsize=(12,5))
plt.title('Articles Published Over Time')
plt.ylabel('Number of Articles')
plt.xlabel('Date')
plt.show()