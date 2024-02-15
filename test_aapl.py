# Hide tensorflow warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from upload_overleaf.upload import upload
import tensorflow as tf

print(
    f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}"
)

lf_intraday = (
    pl.scan_parquet("prices.parquet")
    .join(
        pl.scan_parquet("coacs.parquet").select(["ticker", "date", "OldNoOfStocks"]),
        on=["ticker", "date"],
        how="left",
    )
    .with_columns(
        pl.when(pl.col("date") == pl.col("date").shift(1))
        .then(0)
        .otherwise(1)
        .alias("first_obs")
    )
    # If first_obs is 1 then set OldNoOfStocks = 1
    .with_columns(
        pl.when(pl.col("first_obs") == 1)
        .then(1)
        .otherwise(pl.col("OldNoOfStocks"))
        .alias("OldNoOfStocks")
    )
    # Fill missing values in OldNoOfStocks with strategy 'backward'
    .with_columns(pl.col("OldNoOfStocks").fill_null(strategy="backward"))
    # Multiply StockOpen, StockHigh, StockLow, and StockClose with OldNoOfStocks to get the adjusted price
    .with_columns((pl.col("StockClose") * pl.col("OldNoOfStocks")).alias("AdjStockClose"))
    .with_columns(
        [
            pl.col("ticker"),
            pl.col("date").alias("date"),
            pl.col("datetime").alias("datetime"),
            pl.col("StockClose").log().alias("log_close"),
            pl.col("AdjStockClose").log().alias("adj_log_close"),
            pl.col("StockVol").alias("volume"),
        ]
    )
    .sort(["ticker", "datetime"])
    .select(
        [
            "ticker",
            "date",
            "datetime",
            "OldNoOfStocks",
            "log_close",
            "adj_log_close",
            "volume",
            (pl.col("log_close") - pl.col("log_close").shift(1))
            .over(pl.col("ticker"))
            .alias("return_1min"),
            (pl.col("log_close") - pl.col("log_close").shift(5))
            .over(pl.col("ticker"))
            .alias("return_5min"),
            (pl.col("log_close") - pl.col("log_close").shift(10))
            .over(pl.col("ticker"))
            .alias("return_10min"),
            (pl.col("log_close") - pl.col("log_close").shift(30))
            .over(pl.col("ticker"))
            .alias("return_30min"),
            (pl.col("log_close") - pl.col("log_close").shift(60))
            .over(pl.col("ticker"))
            .alias("return_1h"),
            (pl.col("log_close") - pl.col("log_close").shift(120))
            .over(pl.col("ticker"))
            .alias("return_2h"),
            (pl.col("log_close") - pl.col("log_close").shift(240))
            .over(pl.col("ticker"))
            .alias("return_4h"),
            (pl.col("log_close") - pl.col("log_close").shift(390))
            .over(pl.col("ticker"))
            .alias("return_1d"),
        ]
    )
)

# Join SP500 data with the intraday data, and calculate the excess market return
lf_intraday = lf_intraday.join(
    (
        lf_intraday.filter(pl.col("ticker") == "US500").select(
            [
                pl.col("datetime").alias("datetime"),
                pl.col("log_close").alias("mkt_log_close"),
                pl.col("volume").alias("mkt_volume"),
                pl.col("return_1min").alias("mkt_return_1min"),
                pl.col("return_5min").alias("mkt_return_5min"),
                pl.col("return_10min").alias("mkt_return_10min"),
                pl.col("return_30min").alias("mkt_return_30min"),
                pl.col("return_1h").alias("mkt_return_1h"),
                pl.col("return_2h").alias("mkt_return_2h"),
                pl.col("return_4h").alias("mkt_return_4h"),
                pl.col("return_1d").alias("mkt_return_1d"),
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
        pl.sum("volume").cast(pl.Float64).alias("volume"),  # Sum volume to get daily volume
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

df_aapl = lf_daily.filter(pl.col("ticker") == "AAPL").collect().to_pandas()


print(df_aapl.head())


# Plot time series of StockClose
plt.figure(figsize=(15, 10))
# Add second y-axis. Left is log price and right is return
ax = sns.lineplot(x="date", y="log_close", data=df_aapl, color="red")
ax2 = ax.twinx()
ax = sns.lineplot(x="date", y="OldNoOfStocks", data=df_aapl, color="green")
# Disable grid
ax.grid(False)
ax2.grid(False)
ax.set_title("AAPL Stock Price and Volume")
ax.set_ylabel("Log Price")
ax2.set_ylabel("Adj Log Price")
ax.set_xlabel("Date")
upload(plt, "Master's Thesis", "figures/aapl_test.png")

print("Done")
