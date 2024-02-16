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
#
# # %% # Load prices and coacs data
# lf_intraday = (
#     pl.scan_parquet("prices.parquet")
#     .join(
#         pl.scan_parquet("coacs.parquet").select(["ticker", "date", "OldNoOfStocks"]),
#         on=["ticker", "date"],
#         how="left",
#     )
#     .sort(["ticker", "datetime"])
#     .with_columns(
#         pl.when(pl.col("ticker").shift(1) != pl.col("ticker"))
#         .then(1)
#         .otherwise(0)
#         .alias("first_obs")
#     )
#     .with_columns(
#         pl.when(pl.col("first_obs") == 1)
#         .then(1)
#         .otherwise(pl.col("OldNoOfStocks"))
#         .alias("OldNoOfStocks")
#     )
#     .with_columns(
#         pl.col("OldNoOfStocks").fill_null(strategy="forward").alias("OldNoOfStocks")
#     )
#     .with_columns(
#         (pl.col("StockClose") / pl.col("OldNoOfStocks")).alias("AdjStockClose")
#     )
#     .with_columns(
#         [
#             pl.col("datetime")
#             .cast(pl.Datetime)
#             .dt.replace_time_zone("America/New_York"),
#             pl.col("StockClose").log().alias("log_close"),
#             pl.col("AdjStockClose").log().alias("adj_log_close"),
#             pl.col("StockVol").alias("volume"),
#         ]
#     )
#     .select(
#         [
#             "ticker",
#             "date",
#             "datetime",
#             "OldNoOfStocks",
#             "StockClose",
#             "AdjStockClose",
#             "log_close",
#             "adj_log_close",
#             "volume",
#         ]
#     )
# )


# %%
lf = pl.scan_parquet("daily.parquet")

def plotAdjvsNonAdj(ticker: str, lf: pl.LazyFrame):
    """
    Plots time series analysis of stock with logarithmic and adjusted logarithmic closing prices.

    Parameters:
    - ticker (str): Ticker symbol of the stock to plot.
    - lf (pl.LazyFrame): Polars LazyFrame containing the stock price data.

    Returns:
    None
    """
    # Get upper for ticker
    ticker = ticker.upper()
    df = lf.filter(pl.col("ticker") == ticker).collect().to_pandas()
    # Set up the figure and axes
    plt.figure(figsize=(12, 8))
    ax = sns.lineplot(x="date", y="log_close", data=df, color="blue", label="Log Close")
    ax2 = ax.twinx()
    ax2 = sns.lineplot(
        x="date", y="adj_log_close", data=df, color="orange", label="Adj Log Close"
    )
    # Adjusting y-axis limits
    # Get lower and upper bounds
    lower_bound = 0
    upper_bound = 10
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

# Get input from user
ticker = input("Enter ticker symbol: ")

# Plot the time series analysis
plotAdjvsNonAdj(ticker, lf)

print(f"Figure for {ticker} has been uploaded to Overleaf.")
# %%

