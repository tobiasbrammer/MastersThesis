import pandas as pd
import wrds

tickers = pd.read_parquet('daily.parquet')['ticker'].unique()

db = wrds.Connection()

# Create the compustat query
compustat_query = f"""
    SELECT *
    FROM comp.fundq
    WHERE indfmt = 'INDL' AND datafmt = 'STD' AND consol = 'C' AND datadate BETWEEN '{start_date}' AND '{end_date}'
"""

df = db.raw_sql(compustat_query)

### CRSP ###

def get_crsp_data(db, start_date, end_date):
    query = f"""
        SELECT a.permno, a.permco, b.ncusip, a.date, 
        b.shrcd, b.exchcd, b.siccd,
        a.ret, a.vol, a.shrout, a.prc, a.cfacpr, a.cfacshr
        FROM crsp.msf AS a
        LEFT JOIN crsp.msenames AS b
        ON a.permno=b.permno
        AND b.namedt<=a.date
        AND a.date<=b.nameendt
        WHERE 1=1
        AND a.date BETWEEN '{start_date}' AND '{end_date}'
        AND b.shrcd BETWEEN 10 AND 11
    """
    crsp_m = db.raw_sql(query, date_cols=['date'])
    crsp_m[['permco', 'permno', 'shrcd', 'exchcd']] = crsp_m[['permco', 'permno', 'shrcd', 'exchcd']].astype(int)
    crsp_m['jdate'] = crsp_m['date'] + MonthEnd(0)  # Align dates as end of month
    crsp_m['p'] = crsp_m['prc'].abs() / crsp_m['cfacpr']  # Adjust prices
    crsp_m['tso'] = crsp_m['shrout'] * crsp_m['cfacshr'] * 1e3  # Adjust shares
    crsp_m['me'] = crsp_m['p'] * crsp_m['tso'] / 1e6  # Market cap in millions
    crsp_summe = crsp_m.groupby(['jdate', 'permco'])['me'].sum().reset_index().rename(columns={'me': 'me_comp'})
    crsp_m = pd.merge(crsp_m, crsp_summe, how='inner', on=['jdate', 'permco'])
    return crsp_m

# Define the variables of interest for CRSP data
crsp_variables = [
    "tic",  # PERMNO, the unique CRSP identifier
    "cmth", # Month
    "cyear", # Year
    "prc",  # Price
    "shrout",  # Shares outstanding
]

# Create the CRSP query
crsp_query = f"""
    SELECT {', '.join(crsp_variables)}
    FROM comp.secm
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
"""
# Fetch CRSP data
crsp_data = db.raw_sql(crsp_query)

db.close()

del db, compustat_query, crsp_query

# Filter the data to only include the tickers of interest
df = df[df['tic'].isin(tickers)]
crsp_data = crsp_data[crsp_data['tic'].isin(tickers)]

# Make sure variables are numeric
df[variables[2:]] = df[variables[2:]].apply(pd.to_numeric, errors='coerce')

# Convert datadate to date
df['datadate'] = pd.to_datetime(df['datadate']).dt.date

# Order by ticker and date
df = df.sort_values(['tic', 'datadate'])

df = df.fillna(0)

# Drop rows with atq = 0
df = df[df['atq'] > 0] # No companies with no assets

df.reset_index(drop=True, inplace=True)

### Inspired by https://www.tidy-finance.org/r/wrds-crsp-and-compustat.html
# Calculate the book value of preferred stock and equity
df['be'] = df['seqq'].combine_first(df['ceqq'] + df['pstkq']).combine_first(df['atq'] - df['ltq']) + \
           df['txditcq'].combine_first(df['txdbq'] + df['itccy']).fillna(0) - \
           df['pstkq'].combine_first(df['pstkq']).combine_first(df['pstkq']).fillna(0)

df['be'] = df['be'].apply(
    lambda x: x if x >= 0 else 0)  # Fama French 1992: The cross-section of expected stock returns.
df['lev'] = (df['dlttq'] + df['dlcq']) / df['atq']  # Leverage
df['op'] = (df['saleq'] - df['cogsq'].fillna(0) - df['xsgaq'].fillna(0) - df['xintq'].fillna(0)) / \
           (df['be'] + 1e-8) # Operating profitability
df['profit'] = df['niq'] / df['atq']  # Return on assets
df['oancfy'] = df['oancfy'].fillna(0)  # Operating cash flow
df['fcf'] = df['oancfy'] - df['capxy']  # Free cash flow

# Calculate investment ratio by using one-year lagged total assets
df['inv'] = df['atq'] / df['atq'].shift(4) - 1
df['inv'] = df['inv'].apply(lambda x: x if x >= 0 else 0)


# Merge Compustat data with CRSP data based on PERMNO and date
merged_data = pd.merge(df, crsp_data, how='left', left_on=['tic', 'datadate'], right_on=['permno', 'date'])

# Calculate market capitalization
merged_data['market_cap'] = merged_data['prc'] * merged_data['shrout']

# Calculate ratios using market data
merged_data['at_to_market_cap'] = merged_data['atq'] / merged_data['market_cap']

# Drop unnecessary columns
merged_data.drop(['permno', 'date', 'prc', 'shrout'], axis=1, inplace=True)

# Replace missing values
df = df.fillna(0)  # Shit breaks if we have missing values.