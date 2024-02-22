import os
import re
from time import time

import polars as pl

# Set environment variable for Rust backtrace
os.environ["RUST_BACKTRACE"] = "1"
# Suppress tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# TODO: Check stability. Sometimes the script fails to read the parquet files.

start = time()

# %% ################## CSV TO PARQUET ##################
# The following section converts all .csv files in 'data_raw_csv' to .parquet files.

# Get all files in data_raw_csv
files = [f for f in os.listdir("data_raw_csv") if f.endswith(".csv")]

# # Convert each file to parquet
print("Converting CSV files to Parquet...")
for file in files:
    # Construct the file path
    file_path = os.path.join("data_raw_csv", file)
    # Read CSV file and try to parse dates
    df = pl.scan_csv(file_path)
    # Write to Parquet
    df.sink_parquet(file_path.replace("csv", "parquet"))
    del df, file, file_path


# %% ################## PRICES ##################
# The following section extracts data from price files in 'data_raw_parquet', i.e. those that start with 'prices'.
# Tickers are extracted using regular expressions. Numerical columns are cast as Float32, to optimize memory.
# Datetime is extracted from 'StartTime' and TZ is set to Copenhagen.
# Each file is appended onto the previous dataframe.
# Finally, the entire dataframe is saved as 'prices.parquet'.
files = [f for f in os.listdir("data_raw_parquet") if f.endswith(".parquet")]

# Define the pattern using regular expression
pattern = re.compile(r"_(.*?)-")
# For files that begin with prices
prices = sorted([f for f in files if f.startswith("prices")])

# Create an empty list to store DataFrames
df_list = []
numeric_cols = ["StockOpen", "StockClose", "StockHigh", "StockLow", "StockVol"]

print("Reading files...")

# Iterate over each file, add it to df_list
for file in prices:
    # Construct the file path
    file_path = os.path.join("data_raw_parquet", file)
    # Read CSV file and try to parse dates
    df_temp = pl.scan_parquet(file_path).with_columns(
        **{col: pl.col(col).cast(pl.Float32) for col in numeric_cols}
    )
    # Extract the ticker from the filename and add it as a column
    ticker = re.search(pattern, file).group(1)
    print(f"Reading {ticker}...")
    df_temp = (
        df_temp.with_columns(pl.lit(ticker).alias("ticker"))
        .with_columns(
            pl.col("StartTime")
            .str.strptime(
                pl.Datetime(time_unit="us", time_zone="Europe/Copenhagen"),
                format="%Y-%m-%d %H:%M:%S",
            )
            .cast(pl.Datetime(time_unit="us", time_zone="Europe/Copenhagen"))
            .alias("datetime"),
            pl.col("StartTime")
            .str.strptime(
                pl.Datetime(time_unit="us", time_zone="Europe/Copenhagen"),
                format="%Y-%m-%d %H:%M:%S",
            )
            .cast(pl.Date)
            .alias("date"),
            pl.col("StartTime")
            .str.strptime(
                pl.Datetime(time_unit="us", time_zone="Europe/Copenhagen"),
                format="%Y-%m-%d %H:%M:%S",
            )
            .cast(pl.Time)
            .alias("time"),
        )
        .drop("StartTime")
    )
    # Add df_temp to df_list
    df_list.append(df_temp)
    del df_temp, file, file_path, ticker

print("Concatenating dataframes...")

# Apply date/time transformations
pl.concat(df_list).sink_parquet("prices.parquet")

print("")
print("Finished saving prices.parquet")
print("")

# Clean up unused variables
del df_list

# %% ################## COACS ##################
# 'Co-ownership Authorised Contractual Schemes'
# The following section extracts data from CoACS files in 'data_raw_parquet', i.e. those that start with 'coacs'.
# Tickers are extracted using regular expressions. Columns are transformed according to schema, with Float32 numeric.
# Datetime is extracted from 'Date' and TZ is set to Copenhagen.
# Each file is appended onto the previous dataframe.
# Finally, the entire dataframe is saved as 'prices.parquet'.

print("Getting CoACS...")
pattern = re.compile(r"_(.*?)-")
# For files that start with coacs
coacs = [
    f
    for f in [f for f in os.listdir("data_raw_parquet") if f.endswith(".parquet")]
    if f.startswith("coacs")
]

col_transforms = [
    ("UIC", pl.Int64),
    ("Date", pl.String),
    ("BOID", pl.Int64),
    ("StartDate", pl.String),
    ("UTCEndDateTime", pl.String),
    ("UTCStartDateTime", pl.String),
    ("OldNoOfStocks", pl.Float32),
    ("NewNoOfStocks", pl.Float32),
    ("AccSplitFactor", pl.Float32),
    ("NeedSplit", pl.Boolean),
    ("AccSplitReady", pl.Boolean),
    ("timestamp", pl.String),
    ("EventCode", pl.String),
    ("EventName", pl.String),
    ("Comment", pl.String),
    ("EntitlementId", pl.String),
]

# Create empty dataframe
df_list = []

# Iterate over each file, add it to df_list
for file in coacs:
    # Construct the file path
    file_path = os.path.join("data_raw_parquet", file)
    # Read CSV file and try to parse dates
    df_temp = pl.scan_parquet(file_path)
    ticker = re.search(pattern, file).group(1)
    df_temp = df_temp.with_columns(
        pl.col("Date").str.replace("00:00:00", "18:00:00").alias("Date")
    )
    # Ensure columns are of the correct type. #
    for col, dt in col_transforms:
        df_temp = df_temp.with_columns(pl.col(col).cast(dt))
    # Replace the time in datetime and time to be 12:00:00
    df_temp = df_temp.with_columns(
        pl.col("Date")
        .str.strptime(
            pl.Datetime(time_unit="us", time_zone="Europe/Copenhagen"),
            format="%Y-%m-%d %H:%M:%S",
        )
        .cast(pl.Datetime)
        .alias("datetime")
        .dt.replace_time_zone("Europe/Copenhagen"),
        pl.col("Date")
        .str.strptime(
            pl.Datetime(time_unit="us", time_zone="Europe/Copenhagen"),
            format="%Y-%m-%d %H:%M:%S",
        )
        .cast(pl.Date)
        .alias("date"),
        pl.col("Date")
        .str.strptime(
            pl.Datetime(time_unit="us", time_zone="Europe/Copenhagen"),
            format="%Y-%m-%d %H:%M:%S",
        )
        .cast(pl.Time)
        .alias("time"),
    )
    df_temp = df_temp.with_columns(pl.lit(ticker).alias("ticker")).drop(
        "UTCEndDateTime",
        "UTCStartDateTime",
        "StartDate",
        "Comment",
        "EntitlementId",
        "timestamp",
        "Date",
    )
    # Add df_temp to df_list
    df_list.append(df_temp)
    del df_temp, file, file_path, ticker, dt, col

pl.concat(df_list).sink_parquet("coacs.parquet")

del col_transforms

print("")
print("Finished saving coacs.parquet")
print("")

# %%
# The following cell reads the parquet files created by the previous sections.
# We join the prices and coacs tables on the ticker and date columns, to adjust the stock prices.
# Doing so, we create a new column called OldNoOfStocks, which serves as an adjustment factor.
# Each adjustment factor is initialized as 1, i.e. no adjustment, and we forward fill the missing values.
# Forward filling ensures the stock is adjusted according to the most recent known value.
# Furthermore, we apply the log transformation to the adjusted prices, and calculate the log return for each ticker.
# The log return is calculated for the following time intervals: 1, 5, 10, 30, 60, 120, 240, and 390 minutes.

print("Creating intraday returns...")
print("Reading prices.parquet and coacs.parquet...")

lf_intraday = (
    pl.scan_parquet("prices.parquet")
    .join(
        pl.scan_parquet("coacs.parquet").select(["ticker", "date", "OldNoOfStocks"]),
        on=["ticker", "date"],
        how="left",
    )
    .sort(["ticker", "datetime"])
    .with_columns(  # Create column with binary indicator for change in ticker, i.e. first observation of ticker.
        pl.when(pl.col("ticker").shift(1) != pl.col("ticker"))
        .then(1)
        .otherwise(0)
        .alias("first_obs")
    )
    .with_columns(  # Initialize adjustment factor ('OldNoOfStocks') as 1.
        pl.when(pl.col("first_obs") == 1)
        .then(1)
        .otherwise(pl.col("OldNoOfStocks"))
        .alias("OldNoOfStocks")
    )
    .with_columns(  # Fill missing values of adjustment factor forward, i.e. last known value.
        pl.col("OldNoOfStocks").fill_null(strategy="forward").alias("OldNoOfStocks")
    )
    .with_columns(  # Adjust stock prices by dividing with adjustment factor, since using forward fill.
        (pl.col("StockClose") / pl.col("OldNoOfStocks")).alias("AdjStockClose")
    )
    .with_columns(
        [  # Apply logarithmic transformation to stock price and adjusted stock price.
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

# %% ################## SP500 ##################
print("Reading SP500 and calculating market excess return...")
# Join SP500 data with the intraday data, and calculate the excess market return
# Calculate excess market return by subtracting the market return from the return of each stock.
lf_intraday = lf_intraday.join(
    (
        lf_intraday.filter(pl.col("ticker") == "US500").select(
            [
                pl.col("datetime").alias("datetime"),
                pl.col("log_close").alias("mkt_log_close"),
                pl.col("volume").alias("mkt_volume"), # Not really sure if this is valid.
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
).with_columns(
    [
        pl.col("return_1min") - pl.col("mkt_return_1min").alias("excess_return_1min"),
        pl.col("return_5min") - pl.col("mkt_return_5min").alias("excess_return_5min"),
        pl.col("return_10min")
        - pl.col("mkt_return_10min").alias("excess_return_10min"),
        pl.col("return_30min")
        - pl.col("mkt_return_30min").alias("excess_return_30min"),
        pl.col("return_1h") - pl.col("mkt_return_1h").alias("excess_return_1h"),
        pl.col("return_2h") - pl.col("mkt_return_2h").alias("excess_return_2h"),
        pl.col("return_4h") - pl.col("mkt_return_4h").alias("excess_return_4h"),
        pl.col("return_1d") - pl.col("mkt_return_1d").alias("excess_return_1d"),
    ]
)

# %% ################## Save to parquet ##################
print("Saving intraday returns to parquet...")
lf_intraday.collect().write_parquet("intraday.parquet")

# %% ################## Daily ##################
print("Calculating daily returns...")

# Group by date and ticker and sum volume to get daily volume
lf_daily = (
    lf_intraday.group_by(["ticker", "date"])
    .agg(
        pl.last("datetime").alias("datetime"),
        pl.last("log_close").cast(pl.Float32),
        pl.last("adj_log_close").cast(pl.Float32),
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
            "log_close",
            "adj_log_close",
            "volume",
            (pl.col("log_close") - pl.col("log_close").shift(1))
            .over(pl.col("ticker"))
            .alias("return_1d"),
        ]
    )
)

# %% ################## Save to parquet ##################
print("Saving daily returns to parquet...")
lf_daily.collect().write_parquet("daily.parquet")

print("")
print("")
print("")
print("Done!")
print(f"Time elapsed: {time() - start:.2f} seconds")
