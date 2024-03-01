import pandas as pd
import wrds
from tqdm import tqdm
import polars as pl

db = wrds.Connection(wrds_username='tobiasbrammer')

# Set your start and end dates
start_date = "2022-12-31"
end_date = "2024-01-01"

# List fundamental variables
# fund_vars = db.describe_table(library="comp", table="funda")

# Define the variables of interest
variables = [
    "tic",  # Ticker
    "cusip",  # CUSIP
    "gvkey",  # Global company key
    "datadate",  # Date
    "at",  # Total assets
    "lt",  # Total liabilities
    "act",  # Current assets
    "che",  # Cash and equivalents
    "lct",  # Current liabilities
    "dlc",  # Current debt
    "txp",  # Income taxes payable
    "sale",  # Sales
    "txditc",  # Deferred income taxes
    "seq",  # Shareholders' equity
    "ceq",  # Common equity
    "pstkl",  # Preferred stock
    "pstk",  # Preferred stock
    "ni",  # Net income
    "dp",  # Dividends paid
    "capx",  # Capital expenditures
    "txdb",  # Debt
]

tickers = pd.read_parquet('daily.parquet')['ticker'].unique()
tickers = tickers[0:10]

# Join tickers in a string like 'AAPL', 'MSFT', ...
tickers = [f"'{t}'" for t in tickers]

# Create the compustat query
compustat_query = f"""
    SELECT {', '.join(variables)}
    FROM comp.funda
    WHERE indfmt = 'INDL' AND datafmt = 'STD' AND consol = 'C' AND datadate BETWEEN '{start_date}' AND '{end_date}'
"""

df = db.raw_sql(compustat_query)

df[~(df['tic'].isna())].pivot(columns='tic').pl.sink_parquet('compustat.parquet')


