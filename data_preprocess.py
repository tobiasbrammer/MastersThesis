# Get data from parquet file
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import upload_overleaf.upload as upload


# Scan parquet file
lf_intraday = (
    pl.scan_parquet("prices.parquet")
    .select(
        pl.col("ticker"),
        pl.col("datetime").alias("date"),
        pl.col("StockOpen").log().alias("Open"),
        pl.col("StockHigh").log().alias("High"),
        pl.col("StockLow").log().alias("Low"),
        pl.col("StockClose").log().alias("Close"),
        pl.col("StockVol").alias("Volume"),
        (pl.col("StockClose").log() - pl.col("StockOpen").log()).alias("Return")
    )
)

# Describe data
df_aapl = lf_intraday.filter(pl.col("ticker") == "AAPL").collect().to_pandas()

