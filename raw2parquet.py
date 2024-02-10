import os
import re
from time import time

import polars as pl

# Set environment variable for Rust backtrace
os.environ["RUST_BACKTRACE"] = "1"

start = time()
# %% ################## CSV TO PARQUET ##################

# # Get all files in data_raw_csv
files = [f for f in os.listdir('data_raw_csv') if f.endswith('.csv')]

# # Convert each file to parquet
print('Converting CSV files to Parquet...')
for file in files:
    # Construct the file path
    file_path = os.path.join('data_raw_csv', file)
    # If file path contains Visa, skip it
    #if 'Visa' in file_path:
    #    continue
    #if 'Interpublic_Group' in file_path:
    #    continue
    # Read CSV file and try to parse dates
    df = pl.scan_csv(file_path)
    # Write to Parquet
    print(f'Converted {file_path.replace("csv", "parquet")}')
    df.sink_parquet(file_path.replace('csv', 'parquet'))
    del df, file, file_path



# %% ################## PRICES ##################
files = [f for f in os.listdir("data_raw_parquet") if f.endswith(".parquet")]
pattern = re.compile(r"_(.*?)-")
# For files that begin with prices
prices = sorted([f for f in files if f.startswith("prices")])

# Omit ROL from prices


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
        **{col: pl.col(col).cast(pl.Float64) for col in numeric_cols}
    ).select([
        "StartTime",
        "StockClose",
        "StockVol",
    ])
    # Extract the ticker from the filename and add it as a column
    ticker = re.search(pattern, file).group(1)
    # Strip ".I" from the end of the ticker
    ticker = ticker.rstrip(".I")  # new line
    # If the ticker is ROL, skip it
    #if ticker == "ROL":
    #    print(f"Skipping {ticker}")
    #    continue
    df_temp = df_temp.with_columns(pl.lit(ticker).alias("ticker"))
    # Test if df_temp is corrupted, if so skip it
    if df_temp.fetch().shape[0] == 0:
        print(f"Failed to read {ticker}")
        continue
    # Test if collect fails on df_temp, if so skip it
    try:
        df_temp.collect()
    except:
        print(f"Failed to collect {ticker}")
        continue
    # Add df_temp to df_list
    df_list.append(df_temp)

    del df_temp, file, file_path, ticker

print("Concatenating dataframes...")

# Apply date/time transformations
(
    pl.concat(df_list)
    .with_columns(
        pl.col("StockClose").cast(pl.Float32).alias("StockClose"),
        pl.col("StockVol").cast(pl.Float32).alias("StockVol"),
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
).sink_parquet("prices.parquet")

# Clean up unused variables
del df_list

# %% ################## COACS ##################

print("Getting CoACS...")

pattern = re.compile(r"_(.*?)-")
# For files that start with coacs
coacs = [f for f in files if f.startswith("coacs")]

col_transforms = [
    ("UIC", pl.Int64),
    ("Date", pl.String),
    ("BOID", pl.Int64),
    ("StartDate", pl.String),
    ("UTCEndDateTime", pl.String),
    ("UTCStartDateTime", pl.String),
    ("OldNoOfStocks", pl.Float64),
    ("NewNoOfStocks", pl.Float64),
    ("AccSplitFactor", pl.Float64),
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
    # Ensure columns are of the correct type. #
    for col, dt in col_transforms:
        df_temp = df_temp.with_columns(pl.col(col).cast(dt))
    df_temp = df_temp.with_columns(pl.lit(ticker).alias("ticker")).drop(
        "UTCEndDateTime", "UTCStartDateTime", "timestamp", "Comment", "EntitlementId"
    )
    # Add df_temp to df_list
    df_list.append(df_temp)
    del df_temp, file, file_path, ticker, dt, col

(
    pl.concat(df_list)
    .with_columns(
        pl.col("Date")
        .str.strptime(
            pl.Datetime(time_unit="us", time_zone="Europe/Copenhagen"),
            format="%Y-%m-%d %H:%M:%S",
        )
        .cast(pl.Datetime)
        .alias("datetime"),
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
    .drop("Date")
).sink_parquet("coacs.parquet")

print("")
print("")
print("")
print("Done!")
print(f"Time elapsed: {time() - start:.2f} seconds")
