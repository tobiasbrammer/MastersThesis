# Hide tensorflow warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from upload_overleaf.upload import upload
import tensorflow as tf
from datetime import datetime


print(
    f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}"
)

# %% # Load prices and coacs data
lf_intraday = (
    pl.scan_parquet("prices.parquet")
    .join(
        pl.scan_parquet("coacs.parquet").select(["ticker", "date", "OldNoOfStocks"]),
        on=["ticker", "date"],
        how="left",
    )
    .sort(["ticker", "datetime"])
    .with_columns(
        pl.when(pl.col("ticker").shift(1) != pl.col("ticker"))
        .then(1)
        .otherwise(0)
        .alias("first_obs")
    )
    .with_columns(
        pl.when(pl.col("first_obs") == 1)
        .then(1)
        .otherwise(pl.col("OldNoOfStocks"))
        .alias("OldNoOfStocks")
    )
    .with_columns(
        pl.col("OldNoOfStocks").fill_null(strategy="forward").alias("OldNoOfStocks")
    )
    .with_columns(
        (pl.col("StockClose") / pl.col("OldNoOfStocks")).alias("AdjStockClose")
    )
    .with_columns(
        [
            pl.col("datetime")
            .cast(pl.Datetime)
            .dt.replace_time_zone("America/New_York"),
            pl.col("StockClose").log().alias("log_close"),
            pl.col("AdjStockClose").log().alias("adj_log_close"),
            pl.col("StockVol").alias("volume"),
        ]
    )
    .select(
        [
            "ticker",
            "date",
            "datetime",
            "OldNoOfStocks",
            "StockClose",
            "AdjStockClose",
            "log_close",
            "adj_log_close",
            "volume",
        ]
    )
)


# %%
def plotAdjvsNonAdj(ticker: str):
    """
    Plots time series analysis of stock with logarithmic and adjusted logarithmic closing prices.

    Parameters:
    - ticker (str): Ticker symbol of the stock to plot.

    Returns:
    None
    """
    # Get upper for ticker
    ticker = ticker.upper()
    df = lf_intraday.filter(pl.col("ticker") == ticker).collect().to_pandas()
    # Set up the figure and axes
    plt.figure(figsize=(12, 8))
    ax = sns.lineplot(x="date", y="log_close", data=df, color="blue", label="Log Close")
    ax2 = ax.twinx()
    ax2 = sns.lineplot(
        x="date", y="adj_log_close", data=df, color="orange", label="Adj Log Close"
    )
    # Adjusting y-axis limits
    # Get lower and upper bounds
    lower_bound = min(df["log_close"].min(), df["adj_log_close"].min())
    upper_bound = max(df["log_close"].max(), df["adj_log_close"].max())
    ax.set_ylim(lower_bound, upper_bound)
    ax2.set_ylim(lower_bound, upper_bound)
    # Adding grid lines
    ax.grid(True, linestyle="--", alpha=0.7)
    ax2.grid(False)
    # Adding a legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")
    # Adding labels and title
    ax.set_title(
        f"Time Series Analysis of {ticker} Stock: Logarithmic and Adjusted Logarithmic Closing Prices"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Logarithmic Closing Price")
    ax2.set_ylabel("Adjusted Logarithmic Closing Price")
    # Formatting date labels
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
    # Upload to Overleaf
    upload(plt, "Master's Thesis", f"figures/{ticker}_test.png")


#%%
plotAdjvsNonAdj("AAPL")
plotAdjvsNonAdj("TSLA")
plotAdjvsNonAdj("MSFT")
# %%

# Join SP500 data with the intraday data, and calculate the excess market return
lf_intraday = lf_intraday.join(
    (
        lf_intraday.filter(pl.col("ticker") == "US500").select(
            [
                pl.col("datetime").alias("datetime"),
                pl.col("log_close").alias("mkt_log_close"),
                pl.col("volume").alias("mkt_volume"),
            ]
        )
    ),
    on="datetime",
    how="left",
)

# Group by date and ticker and sum volumne to get daily volume
lf_daily = (
    lf_intraday.group_by(["ticker", "date"])
    .agg(
        pl.last("datetime").alias("datetime"),
        pl.last("log_close").cast(pl.Float32),
        pl.last("adj_log_close").cast(pl.Float32),
        pl.last("OldNoOfStocks").cast(pl.Float32),
        pl.sum("volume")
        .cast(pl.Float32)
        .alias("volume"),  # Sum volume to get daily volume
    )
    .group_by(["ticker", "date"])
    .last()  # Select the last row in each group
    .sort(["ticker", "date"])
    .select(
        [
            "ticker",
            "date",
            "datetime",
            "OldNoOfStocks",
            "adj_log_close",
            "log_close",
            "volume",
            (pl.col("adj_log_close") - pl.col("adj_log_close").shift(1))
            .over(pl.col("ticker"))
            .alias("adj_return_1d"),
            (pl.col("log_close") - pl.col("log_close").shift(1))
            .over(pl.col("ticker"))
            .alias("return_1d"),
        ]
    )
)

test = lf_intraday.filter(pl.col("ticker") == "MSFT").collect().to_pandas()


print(df_aapl.head())


# Plot time series of StockClose
plt.figure(figsize=(15, 10))
# Add second y-axis. Left is log price and right is return
ax = sns.lineplot(x="date", y="log_close", data=df_aapl, color="red")
ax2 = ax.twinx()
ax = sns.lineplot(x="date", y="adj_log_close", data=df_aapl, color="green")
# Disable grid
ax.grid(False)
ax2.grid(False)
ax.set_title("AAPL Stock Price and Volume")
ax.set_ylabel("Log Price")
ax2.set_ylabel("Adj Log Price")
ax.set_xlabel("Date")
upload(plt, "Master's Thesis", "figures/aapl_test.png")

print("Done")
