import os
import polars as pl
import re

# Set environment variable for Rust backtrace
os.environ["RUST_BACKTRACE"] = "1"

# %% ################## CSV TO PARQUET ##################

# # Get all files in data_raw_csv
# files = [f for f in os.listdir('data_raw_csv') if f.endswith('.csv')]

# # Convert each file to parquet
# print('Converting CSV files to Parquet...')
# for file in files:
#     # Construct the file path
#     file_path = os.path.join('data_raw_csv', file)
#     # Read CSV file and try to parse dates
#     df = pl.scan_csv(file_path)
#     # Write to Parquet
#     df.sink_parquet(file_path.replace('.csv', '.parquet'))
#     del df, file, file_path

files = [f for f in os.listdir("data_raw_parquet") if f.endswith(".parquet")]

# Define the pattern using regular expression
pattern = re.compile(r"_(.*?)-")


# %% ################## PRICES ##################

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
        **{col: pl.col(col).cast(pl.Float64) for col in numeric_cols}
    )
    # Extract the ticker from the filename and add it as a column
    ticker = re.search(pattern, file).group(1)
    df_temp = df_temp.with_columns(pl.lit(ticker).alias("ticker"))
    # Add df_temp to df_list
    df_list.append(df_temp)
    del df_temp, file, file_path, ticker

print("Concatenating dataframes...")

# Apply date/time transformations
(
    pl.concat(df_list)
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
).sink_parquet("prices.parquet")

# Clean up unused variables
del df_list

# %% ################## COACS ##################

print("Getting CoACS...")

# For files that start with coacs
coacs = [f for f in files if f.startswith("coacs")]

# Create empty dataframe
df_list = []

# Iterate over each file, add it to df_list
for file in coacs:
    # Construct the file path
    file_path = os.path.join("data_raw_parquet", file)
    # Read CSV file and try to parse dates
    df_temp = pl.scan_parquet(file_path)
    # Extract the ticker from the filename and add it as a column
    ticker = re.search(pattern, file).group(1)
    df_temp = df_temp.with_columns(pl.lit(ticker).alias("ticker"))
    # Add df_temp to df_list
    df_list.append(df_temp)
    del df_temp, file, file_path, ticker


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

# Delete variables from workspace
del df_list, coacs, files, pattern

print("")
print("")
print("")
print("Done!")
