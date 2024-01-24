# Get data from parquet file
import polars as pl

# Read parquet file
lf = pl.scan_parquet('prices_daily.parquet', low_memory=True)

# Load LazyFrame to memory
df = lf.collect(streaming=True)

df.describe()

# Get ticker with most data
df.groupby('ticker').count().sort('StartTime').head(1)
