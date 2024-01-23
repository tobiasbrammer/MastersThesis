# Get data from parquet file
import polars as pl

# Read parquet file
df = pl.read_parquet('prices.parquet')

df.describe()
