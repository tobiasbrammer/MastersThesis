import os
import polars as pl
import re

# Get all files in data_raw
files = os.listdir('data_raw')
files = [f for f in files if f.endswith('.csv')]
# For files that begins with prices
prices = [f for f in files if f.startswith('prices')]
prices.sort()

# Define the pattern using regular expression
pattern = re.compile(r'_(.*?)-')

# Create empty dataframe
df_list = []
# For each file, add it to df
for file in prices:
    print(file)
    # Read csv file and try to parse dates
    df_temp = pl.read_csv('data_raw/' + file, try_parse_dates=True)
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

df = pl.concat(df_list)

# Save df as parquet
df.write_parquet('prices.parquet')

# For files that begins with prices
coacs = [f for f in files if f.startswith('coacs')]
coacs.sort()

# Create empty dataframe
df_list = []

# For each file, add it to df
for file in coacs:
    print(file)
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

df = pl.concat(df_list)

# Save df as parquet
df.write_parquet('coacs.parquet')