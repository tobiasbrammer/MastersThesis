import os
import polars as pl
import re


# Get all files in data_raw
files = os.listdir('data_raw')
files = [f for f in files if f.endswith('.csv')]

# Define the pattern using regular expression
pattern = re.compile(r'_(.*?)-')

##################### PRICES #####################

# For files that begins with prices
prices = [f for f in files if f.startswith('prices')]

# Create empty dataframe
df_list = []
# For each file, add it to df
for file in prices:
    # Read csv file and try to parse dates
    df_temp = pl.scan_csv('data_raw/' + file, low_memory=True)
    # Force StockOpen  ┆ StockClose ┆ StockHigh  ┆ StockLow   ┆ StockVol to f64
    df_temp = df_temp.with_columns(
        pl.col('StockOpen').cast(pl.Float64),
        pl.col('StockClose').cast(pl.Float64),
        pl.col('StockHigh').cast(pl.Float64),
        pl.col('StockLow').cast(pl.Float64),
        pl.col('StockVol').cast(pl.Float64)
    )
    # If df_temp is empty, continue
    # Extraxt the ticker from the filename and add it as a column
    ticker = re.search(pattern, file).group(1)
    # Add ticker as a column
    df_temp = df_temp.with_columns(pl.lit(ticker).alias("ticker"))
    # Add df_temp to df_list
    df_list.append(df_temp)
    del df_temp
    print(ticker)


print('Concatenating dataframes...')

df = pl.concat(df_list)

(
    df
    .with_columns(
        pl.col('StartTime').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S').cast(pl.Datetime).alias('datetime'),
        pl.col('StartTime').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S').cast(pl.Date).alias('date'),
        pl.col('StartTime').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S').cast(pl.Time).alias('time')
    )
    .drop('StartTime')
)

df.sink_parquet('prices.parquet')


# Delete all from workspace expect globals
del df_list, file, ticker, df_temp, prices

print('Getting daily prices...')

# Scan parquet file, and create a LazyFrame
lf = pl.scan_parquet('prices.parquet', low_memory=True)

# Group by date and ticker and sum StockVol to get daily volume
lf_dvol = lf.group_by(['date', 'ticker']).agg(pl.sum('StockVol').alias('daily_volume'))

# Join lf and lf_dvol
lf = lf.join(lf_dvol, on=['date', 'ticker'])

# Only keep rows with time 20:59:00
lf.filter(pl.col('time').str.contains('20:59:00')).sink_parquet('prices_daily.parquet')

# Clear LazyFrame
del lf, lf_dvol

##################### COACS #####################

print('Getting CoACS...')

# For files that start with coacs
coacs = [f for f in files if f.startswith('coacs')]

# Create empty dataframe
df_list = []

# For each file, add it to df
for file in coacs:
    # Read csv file and try to parse dates
    df_temp = pl.scan_csv('data_raw/' + file)
    # If df_temp is empty, continue
    ticker = re.search(pattern, file).group(1)
    # Add ticker as a column
    df_temp = df_temp.with_columns(pl.lit(ticker).alias("ticker"))
    df_list.append(df_temp)
    print(ticker)

pl.concat(df_list).sink_parquet('coacs.parquet')

# Delete variables from workspace
del df_list, file, ticker, df_temp, coacs, files, pattern
print('')
print('')
print('Done!')
