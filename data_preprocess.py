# Get data from parquet file
import polars as pl

# Read parquet file
sc = pl.scan_parquet('prices.parquet', low_memory=True)

# Describe the LazyFrame
df = sc.collect()

sc.head()

