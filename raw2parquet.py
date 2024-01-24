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
prices.sort()


# Create empty dataframe
df_list = []
# For each file, add it to df
for file in prices:
    # Read csv file and try to parse dates
    df_temp = pl.read_csv('data_raw/' + file)
    # If df_temp is empty, continue
    if df_temp.shape[0] == 0:
        continue
    else:
        # Extraxt the ticker from the filename and add it as a column
        ticker = re.search(pattern, file).group(1)
        # Add ticker as a column
        df_temp = df_temp.with_columns(pl.lit(ticker).alias("ticker"))
        # Add df_temp to df_list
        df_list.append(df_temp)
    print(ticker)


print('Concatenating dataframes...')
df = pl.concat(df_list)

# Save df as parquet
df.write_parquet('prices.parquet')

# Delete all from workspace expect globals
del df_list, file, ticker, df_temp, df, prices

print('Getting daily prices...')

# Scan parquet file
lf = pl.scan_parquet('prices.parquet', low_memory=True)

# Only keep rows with time 20:59:00
lf.filter(pl.col('StartTime').str.contains('20:59:00')).collect().write_parquet('prices_daily.parquet')

# Clear LazyFrame
del lf

##################### COACS #####################

print('Getting CoACS...')

# For files that start with coacs
coacs = [f for f in files if f.startswith('coacs')]
coacs.sort()

# Create empty dataframe
df_list = []

# For each file, add it to df
for file in coacs:
    # Read csv file and try to parse dates
    df_temp = pl.read_csv('data_raw/' + file)
    # If df_temp is empty, continue
    if df_temp.shape[0] == 0:
        continue
    else:
        ticker = re.search(pattern, file).group(1)
        # Add ticker as a column
        df_temp = df_temp.with_columns(pl.lit(ticker).alias("ticker"))
        df_list.append(df_temp)
    print(ticker)

df = pl.concat(df_list)

# Save df as parquet
df.write_parquet('coacs.parquet')

# Delete variables from workspace
del df_list, file, ticker, df_temp, df, coacs, files, pattern

print('Done!')

