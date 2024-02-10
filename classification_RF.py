import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from upload_overleaf.upload import upload
from time import time
import os

# Set environment variable for Rust backtrace
os.environ["RUST_BACKTRACE"] = "1"
# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

start = time()

#%% ################## Classification ##################
# The following script calculates the likelihood of a stock to outperform the market.
# A Random Forest model is trained to classify the stocks into two categories: outperform and underperform.
# The trading strategy is to buy the stocks that are classified as outperform
# and sell the stocks that are classified as underperform.

# Import data
print("Reading data...")
# The data consists of intraday data with every 120th observation.
df = (pl.scan_parquet('intraday.parquet')
      .with_columns(
        # Calculate excess return as the return of the stock minus the return of the market
        (pl.col('return_1d') - pl.col('mkt_return_1min')).cast(pl.Float32).alias('excess_return_1min'),
        (pl.col('return_1d') - pl.col('mkt_return_5min')).cast(pl.Float32).alias('excess_return_5min'),
        (pl.col('return_1d') - pl.col('mkt_return_10min')).cast(pl.Float32).alias('excess_return_10min'),
        (pl.col('return_1d') - pl.col('mkt_return_30min')).cast(pl.Float32).alias('excess_return_30min'),
        (pl.col('return_1d') - pl.col('mkt_return_1h')).cast(pl.Float32).alias('excess_return_1h'),
        (pl.col('return_1d') - pl.col('mkt_return_2h')).cast(pl.Float32).alias('excess_return_2h'),
        (pl.col('return_1d') - pl.col('mkt_return_4h')).cast(pl.Float32).alias('excess_return_4h'),
        (pl.col('return_1d') - pl.col('mkt_return_1d')).cast(pl.Float32).alias('excess_return_1d')
    ).select([
        'ticker',
        'datetime',
        'log_close',
        'volume',
        'excess_return_1min',
        'excess_return_5min',
        'excess_return_10min',
        'excess_return_30min',
        'excess_return_1h',
        'excess_return_2h',
        'excess_return_4h',
        'excess_return_1d'
    ])
    .gather_every(2400)
    .collect()
    .to_pandas()
    ).dropna()

# Drop infinities
df = df.replace([float('inf'), float('-inf')], float('nan')).dropna()

# Create target variable
print("Creating target variable...")
# The target variable is the excess 30 return of the stock over the market.
# If the excess 30 return is positive, the stock is classified as outperform.
# If the excess 30 return is negative, the stock is classified as underperform.
df['target'] = (df['excess_return_30min'] > 0).astype(int)

# Train Random Forest model
print("Training Random Forest model...")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Split data into training and test set
X = df.drop(['ticker', 'datetime', 'target', 'excess_return_30min'], axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
print("Model evaluation...")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature importance
print("Feature importance...")
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.title("Feature Importance")
plt.show()
