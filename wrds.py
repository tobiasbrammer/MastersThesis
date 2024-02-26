import pandas as pd
import os
import wrds

db = wrds.Connection(
    wrds_username=os.environ["WRDS_USERNAME"]
)

# Set your start and end dates
start_date = "2019-01-01"
end_date = "2023-12-31"

# List fundamental variables
fund_vars = db.describe_table(library="comp", table="funda")

# Define the variables of interest
variables = [
    "tic", # Ticker
    "cusip", # CUSIP
    "gvkey", # Global company key
    "datadate", # Date
    "at", # Total assets
    "lt", # Total liabilities
    "prc", # Price
    "act", # Current assets
    "che", # Cash and equivalents
    "lct", # Current liabilities
    "dlc", # Current debt
    "txp", # Income taxes payable
    "sale", # Sales
    "txditc", # Deferred income taxes
    "seq", # Shareholders' equity
    "ceq", # Common equity
    "pstkl", # Preferred stock
    "pstk", # Preferred stock
    "ni", # Net income
    "dp", # Dividends paid
    "wcapch", # Working capital change
    "capx", # Capital expenditures
    "txdb", # Debt
]

# List variables from 'variables' that are not in 'fund_vars'
missing_vars = [var for var in variables if var not in fund_vars["name"].values]

# Remove missing variables from 'variables'
variables = [var for var in variables if var not in missing_vars]



# Download Compustat data
compustat_query = f"""
    SELECT {', '.join(variables)}
    FROM comp.funda
    WHERE indfmt = 'INDL' AND datafmt = 'STD' AND consol = 'C' AND datadate BETWEEN '{start_date}' AND '{end_date}'
"""
compustat_data = db.raw_sql(compustat_query)

# Close the connection
db.close()

# Merge the data
data = pd.merge(
    crsp_data,
    compustat_data,
    how="left",
    left_on=["permno", "date"],
    right_on=["gvkey", "datadate"],
)
