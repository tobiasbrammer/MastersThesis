# Get data from parquet file
import polars as pl
from datetime import datetime

# Scan parquet file
lf = pl.scan_parquet("prices_daily.parquet", low_memory=True)

# Load LazyFrame to memory
df = lf.collect(streaming=True)

# Convert to datetime
df = (
    df
    .with_columns(
        pl.col('StartTime').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S').cast(pl.Datetime).alias('datetime'),
        pl.col('StartTime').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S').cast(pl.Date).alias('date'),
        pl.col('StartTime').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S').cast(pl.Time).alias('time')
    )
    .drop('StartTime')
)

# Create date column



# Get ticker with most data
obs_per_ticker = (
    df
    .group_by("ticker")
    .agg(pl.count("ticker").alias("obs"))
    .sort("obs", descending=True)
)


