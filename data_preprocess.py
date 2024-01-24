# Get data from parquet file
import polars as pl
import matplotlib.pyplot as plt
from upload_overleaf.upload import upload

# Scan parquet file
#lf_daily = pl.scan_parquet("prices_daily.parquet", low_memory=True)
lf_intraday = pl.scan_parquet("prices.parquet", low_memory=True)
#lf_coacs = pl.scan_parquet("coacs.parquet", low_memory=True)

#df_daily = lf_daily.collect(streaming=True)
df_intraday = lf_intraday.fetch(n_rows=1000000)
#df_coacs = lf_coacs.collect(streaming=True)


# Get ticker with most data
# obs_per_ticker = (
#     df_daily
#     .group_by("ticker")
#     .agg(pl.count("ticker").alias("obs"))
#     .sort("obs", descending=True)
# )


# Get data for AAPL

df_intraday_aapl = (lf_intraday.filter(pl.col("ticker") == "AAPL")).collect(streaming=True)

# Plot StockClose and StockVol over datetime with plt
plt.plot(df_intraday_aapl["datetime"], df_intraday_aapl["StockClose"])
# plt.show()

upload(plt, "Master's Thesis", 'figures/aapl_test.png')

plt.savefig("aapl_test.png")