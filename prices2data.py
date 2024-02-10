import polars as pl
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

#%% ################## Setup ##################
# The following cell reads the parquet files created by raw2parquet.py.

# It joins the prices and coacs tables on the ticker and date columns.
# It also creates a new column called OldNoOfStocks, which is the number of stocks before the stock split/dividend.
# If there is not a match in the coacs table, the OldNoOfStocks is set to 1.
# The StockOpen, StockHigh, StockLow, and StockClose columns are multiplied with OldNoOfStocks to get the adjusted price.

# It applies the log transformation to the adjusted prices, and the calculates the log return for each ticker.
# The log return is calculated for the following time intervals: 1, 5, 10, 30, 60, 120, 240, and 390 minutes.

print("Reading parquet files...")
lf_intraday = (
    pl.scan_parquet('prices.parquet')
    .join(pl.scan_parquet('coacs.parquet').select(['ticker', 'date', 'OldNoOfStocks']), on=['ticker', 'date'], how='left')
    # If there is not a match in the coacs table, the OldNoOfStocks is set to 1.
    .with_columns(pl.when(pl.col('OldNoOfStocks').is_null()).then(1).otherwise(pl.col('OldNoOfStocks')).alias('OldNoOfStocks'))
    # Multiply StockOpen, StockHigh, StockLow, and StockClose with OldNoOfStocks to get the adjusted price
    .with_columns([
        pl.col('StockClose') * pl.col('OldNoOfStocks').alias('StockClose')
    ])
    .with_columns([
        pl.col('ticker'),
        pl.col('date').alias('date'),
        pl.col('datetime').alias('datetime'),
        pl.col('StockClose').log().alias('log_close'),
        pl.col('StockVol').alias('volume')
    ])
    .sort(['ticker', 'datetime'])
    .select([
        'ticker',
        'date',
        'datetime',
        'log_close',
        'volume',
        (pl.col('log_close') - pl.col('log_close').shift(1)).over(pl.col('ticker')).alias('return_1min'),
        (pl.col('log_close') - pl.col('log_close').shift(5)).over(pl.col('ticker')).alias('return_5min'),
        (pl.col('log_close') - pl.col('log_close').shift(10)).over(pl.col('ticker')).alias('return_10min'),
        (pl.col('log_close') - pl.col('log_close').shift(30)).over(pl.col('ticker')).alias('return_30min'),
        (pl.col('log_close') - pl.col('log_close').shift(60)).over(pl.col('ticker')).alias('return_1h'),
        (pl.col('log_close') - pl.col('log_close').shift(120)).over(pl.col('ticker')).alias('return_2h'),
        (pl.col('log_close') - pl.col('log_close').shift(240)).over(pl.col('ticker')).alias('return_4h'),
        (pl.col('log_close') - pl.col('log_close').shift(390)).over(pl.col('ticker')).alias('return_1d')
    ])
)

#%% ################## SP500 ##################
print("Reading SP500 and calculating market excess return...")
# Join SP500 data with the intraday data, and calculate the excess market return
lf_intraday = (
    lf_intraday
    .join((lf_intraday.filter(pl.col('ticker') == 'US500.I')
    .select([
        pl.col('datetime').alias('datetime'),
        pl.col('log_close').alias('mkt_log_close'),
        pl.col('volume').alias('mkt_volume'),
        pl.col('return_1min').alias('mkt_return_1min'),
        pl.col('return_5min').alias('mkt_return_5min'),
        pl.col('return_10min').alias('mkt_return_10min'),
        pl.col('return_30min').alias('mkt_return_30min'),
        pl.col('return_1h').alias('mkt_return_1h'),
        pl.col('return_2h').alias('mkt_return_2h'),
        pl.col('return_4h').alias('mkt_return_4h'),
        pl.col('return_1d').alias('mkt_return_1d')
    ])
    ), on='datetime', how='left')
    .with_columns([
        pl.col('return_1min') - pl.col('mkt_return_1min').alias('excess_return_1min'),
        pl.col('return_5min') - pl.col('mkt_return_5min').alias('excess_return_5min'),
        pl.col('return_10min') - pl.col('mkt_return_10min').alias('excess_return_10min'),
        pl.col('return_30min') - pl.col('mkt_return_30min').alias('excess_return_30min'),
        pl.col('return_1h') - pl.col('mkt_return_1h').alias('excess_return_1h'),
        pl.col('return_2h') - pl.col('mkt_return_2h').alias('excess_return_2h'),
        pl.col('return_4h') - pl.col('mkt_return_4h').alias('excess_return_4h'),
        pl.col('return_1d') - pl.col('mkt_return_1d').alias('excess_return_1d')
    ])
)

#%% ################## Save to parquet ##################
print("Saving intraday returns to parquet...")
lf_intraday.collect().write_parquet('intraday.parquet')

#%% ################## Daily ##################
print("Calculating daily returns...")
# Group by date and ticker and sum volumne to get daily volume
lf_daily = (
    lf_intraday
    .group_by(['ticker', 'date'])
    .agg(
        pl.last('datetime').alias('datetime'),
        pl.last('log_close').cast(pl.Float64),
        pl.sum('volume').cast(pl.Float64).alias('volume') # Sum volume to get daily volume
    )
    .group_by(['ticker', 'date']).last() # Select the last row in each group
    .sort(['ticker', 'date'])
    .select([
        'ticker',
        'date',
        'datetime',
        'log_close',
        'volume',
        (pl.col('log_close') - pl.col('log_close').shift(1)).over(pl.col('ticker')).alias('return_1d')
    ])
)

#%% ################## Save to parquet ##################
print("Saving daily returns to parquet...")
lf_daily.collect().write_parquet('daily.parquet')

print("")
print("")
print("")
print("Done!")
print(f"Time elapsed: {time() - start:.2f} seconds")

