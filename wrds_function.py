# Get WRDS data for Instrumented PCA
def get_wrds(start_date="1970-01-01", end_date="2024-01-01"):
    # ToDo: Figure out why there are so many rows. Seems like some part of the code uses daily data...
    from pandas.tseries.offsets import MonthEnd
    import pandas as pd
    import numpy as np
    from sqlalchemy import create_engine
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

    pd.set_option("future.no_silent_downcasting", True)

    connection_string = (
        "postgresql+psycopg2://"
        f"tobiasbrammer:naqgUf-bantas-1ruwby"
        "@wrds-pgdata.wharton.upenn.edu:9737/wrds"
    )

    wrds = create_engine(connection_string, pool_pre_ping=True)

    def get_daily_crsp_data(wrds, start_date, end_date):
        # ToDo: Not finished.
        from functions import get_risk_free_rate
        from tqdm import tqdm

        crsp_monthly = get_monthly_crsp_data(wrds, start_date, end_date)
        rf = get_risk_free_rate(start_date, end_date)
        rf.rename(columns={"Adj Close": "rf"}, inplace=True)

        permnos = list(crsp_monthly["permno"].unique().astype(str))

        batch_size = 500
        batches = np.ceil(len(permnos) / batch_size).astype(int)

        daily_rets = pd.DataFrame()

        # for j in range(1, batches + 1):
        for j in tqdm(range(1, batches + 1), miniters=25):

            permno_batch = permnos[
                ((j - 1) * batch_size) : (min(j * batch_size, len(permnos)))
            ]

            permno_batch_formatted = ", ".join(f"'{permno}'" for permno in permno_batch)
            permno_string = f"({permno_batch_formatted})"

            crsp_daily_sub_query = (
                "SELECT permno, dlycaldt AS date, dlyret AS ret, dlyticker AS ticker "
                "FROM crsp.dsf_v2 "
                f"WHERE permno IN {permno_string} "
                f"AND dlycaldt BETWEEN '{start_date}' AND '{end_date}'"
            )

            crsp_daily_sub = pd.read_sql_query(
                sql=crsp_daily_sub_query,
                con=wrds,
                dtype={"permno": int},
                parse_dates={"date"},
            ).dropna()

            if not crsp_daily_sub.empty:

                crsp_daily_sub = (
                    crsp_daily_sub.assign(
                        month=lambda x: x["date"].dt.to_period("M").dt.to_timestamp()
                    )
                    .merge(rf[["date", "rf"]], on="date", how="left")
                    .assign(ret_excess=lambda x: ((x["ret"] - x["rf"]).clip(lower=-1)))
                    .get(["permno", "date", "month", "ret_excess"])
                )

                if j == 1:
                    daily_rets = crsp_daily_sub
                else:
                    daily_rets = daily_rets.append(crsp_daily_sub)

    def get_monthly_crsp_data(wrds, start_date, end_date):
        import yfinance as yf
        from statsmodels.formula.api import ols

        # Read tickers from file
        tickers = pd.read_csv("tickers.csv")["ticker"].tolist()

        crsp_monthly_query = f"""
        SELECT msf.ticker, msf.permno, msf.mthcaldt AS date, 
            date_trunc('month', msf.mthcaldt)::date AS month, 
            msf.mthret AS ret, msf.shrout, msf.mthprc AS altprc, 
            msf.primaryexch, msf.siccd, msf.mthvol AS vol
            FROM crsp.msf_v2 AS msf 
            LEFT JOIN crsp.stksecurityinfohist AS ssih 
            ON msf.permno = ssih.permno AND 
            ssih.secinfostartdt <= msf.mthcaldt AND 
            msf.mthcaldt <= ssih.secinfoenddt 
            WHERE msf.mthcaldt BETWEEN '{start_date}' AND '{end_date}' 
            AND msf.ticker IN {tuple(tickers)}
            AND ssih.sharetype = 'NS' 
            AND ssih.securitytype = 'EQTY' 
            AND ssih.securitysubtype = 'COM' 
            AND ssih.usincflg = 'Y' 
            AND ssih.issuertype in ('ACOR', 'CORP')
        """
        print(f"Fetching monthly CRSP data")
        crsp_monthly = pd.read_sql_query(
            sql=crsp_monthly_query,
            con=wrds,
            dtype={"permno": int, "siccd": int},
            parse_dates={"date", "month"},
        ).assign(shrout=lambda x: x["shrout"] * 1000)

        crsp_monthly = crsp_monthly.assign(
            mktcap=lambda x: x["shrout"] * x["altprc"]
        ).assign(mktcap=lambda x: x["mktcap"].replace(0, np.nan))

        crsp_monthly["jdate"] = crsp_monthly["date"] + MonthEnd(0)

        mktcap_lag = crsp_monthly.assign(
            month=lambda x: x["month"] + pd.DateOffset(months=1),
            mktcap_lag=lambda x: x["mktcap"],
        ).get(["permno", "month", "mktcap_lag"])

        crsp_monthly = crsp_monthly.merge(
            mktcap_lag, how="left", on=["permno", "month"]
        )

        def assign_exchange(primaryexch):
            if primaryexch == "N":
                return "NYSE"
            elif primaryexch == "A":
                return "AMEX"
            elif primaryexch == "Q":
                return "NASDAQ"
            else:
                return "Other"

        crsp_monthly["exchange"] = crsp_monthly["primaryexch"].apply(assign_exchange)

        print(f"Fetching risk free rate and calculating excess returns")
        rf = yf.download("^IRX", start=start_date, end=end_date)["Adj Close"]
        rf = pd.DataFrame(
            rf.apply(lambda x: (1 + x) ** (1 / 12) - 1) / 100
        )  # Monthly risk free rate.
        # Set Date as datetime
        rf.index = pd.to_datetime(rf.index)
        # Add a column with the end of month date
        rf["jdate"] = rf.index + MonthEnd(0)
        # Rename Adj Close to rf
        rf.rename(columns={"Adj Close": "rf"}, inplace=True)
        # Keep the last observation of each month
        rf = rf.groupby("jdate").last()

        crsp_monthly = (
            crsp_monthly.merge(rf, how="left", on="jdate")
            .assign(ret_excess=lambda x: x["ret"] - x["rf"])
            .assign(ret_excess=lambda x: x["ret_excess"].clip(lower=-1))
            .drop(columns=["rf"])
        )

        mkt = yf.download("^GSPC", start=start_date, end=end_date)["Adj Close"]
        mkt = pd.DataFrame(mkt.apply(lambda x: np.log(x)))
        mkt.index = pd.to_datetime(mkt.index)
        mkt["jdate"] = mkt.index + MonthEnd(0)
        mkt.rename(columns={"Adj Close": "mkt"}, inplace=True)
        mkt = mkt.groupby("jdate").last()
        mkt["mkt_ret"] = mkt["mkt"] - mkt["mkt"].shift(1)  # ToDo: This is not correct.
        # Calculate market return
        mkt["mkt_ret"] = mkt["mkt_ret"].fillna(0)

        # Sort by date
        mkt = mkt.sort_values("jdate")
        crsp_monthly = crsp_monthly.merge(mkt, how="left", on="jdate")

        print(f"Fetching market returns and calculating betas")
        # Regress ret on mkt_ret to extract betas
        _beta = (
            crsp_monthly.copy()[["jdate", "permno", "ret", "mkt_ret"]]
            .sort_values(["permno", "jdate"])
            .set_index("jdate")
        )
        _beta["ret"] = _beta["ret"].fillna(0)
        _beta["mkt_ret"] = _beta["mkt_ret"].fillna(0)
        _beta = (
            _beta.groupby("permno")
            .apply(
                lambda x: pd.Series(
                    ols(
                        "ret ~ mkt_ret",
                        data=pd.DataFrame({"ret": x["ret"], "mkt_ret": x["mkt_ret"]}),
                    )
                    .fit()
                    .params
                ),
                include_groups=False,
            )
            .reset_index()
        )
        # Drop intercept
        _beta = _beta.drop(columns=["Intercept"])
        # Rename mkt_ret to beta
        _beta.rename(columns={"mkt_ret": "beta"}, inplace=True)
        crsp_monthly = crsp_monthly.merge(_beta, how="left", on="permno")

        return crsp_monthly

    def get_comp_data(wrds, start_date):
        compustat_query = f"""
            SELECT 
            gvkey, datadate, seq, ceq, at, lt, txditc, txdb, itcb, pstkrv, 
            pstkl, capx, oancf, cogs, xint, xsga, che, ivao, dlc, dltt, mib, pstk, 
            dp, act, lct, txp, sale, dvt, wcapch, ppegt, ni, ib, xrd, oiadp, ajex, csho
            FROM comp.funda
            WHERE indfmt = 'INDL' 
            AND datafmt = 'STD' 
            AND consol = 'C' 
            AND datadate BETWEEN '{start_date}' AND '{end_date}'
            """

        print(f"Fetching Compustat data")

        compustat = pd.read_sql_query(
            sql=compustat_query,
            con=wrds,
            dtype={"gvkey": str},
            parse_dates={"datadate"},
        )

        compustat = (
            compustat.assign(
                be=lambda x: (
                    x["seq"]
                    .combine_first(x["ceq"] + x["pstk"])
                    .combine_first(x["at"] - x["lt"])
                    + x["txditc"].combine_first(x["txdb"] + x["itcb"]).fillna(0)
                    - x["pstkrv"]
                    .combine_first(x["pstkl"])
                    .combine_first(x["pstk"])
                    .fillna(0)
                )
            )
            .assign(be=lambda x: x["be"].apply(lambda y: np.nan if y <= 0 else y))
            .assign(
                op=lambda x: (
                    (
                        x["sale"]
                        - x["cogs"].fillna(0)
                        - x["xsga"].fillna(0)
                        - x["xint"].fillna(0)
                    )
                    / x["be"]
                )
            )
        )
        # We keep only the last available information for each firm-year group.
        compustat = (
            compustat.assign(year=lambda x: pd.DatetimeIndex(x["datadate"]).year)
            .sort_values("datadate")
            .groupby(["gvkey", "year"])
            .tail(1)
            .reset_index()
        )
        # We also compute the investment ratio (inv) according to Kenneth French’s variable definitions as the change in total assets from one fiscal year to another.
        compustat_lag = (
            compustat.get(["gvkey", "year", "at"])
            .assign(year=lambda x: x["year"] + 1)
            .rename(columns={"at": "at_lag"})
        )

        compustat = (
            compustat.merge(compustat_lag, how="left", on=["gvkey", "year"])
            .assign(inv=lambda x: x["at"] / x["at_lag"] - 1)
            .assign(inv=lambda x: np.where(x["at_lag"] <= 0, np.nan, x["inv"]))
        )

        return compustat

    def get_ccm_data(wrds):
        ccmxpf_linktable_query = """
            SELECT lpermno AS permno, gvkey, linkdt, 
            COALESCE(linkenddt, CURRENT_DATE) AS linkenddt 
            FROM crsp.ccmxpf_linktable 
            WHERE linktype IN ('LU', 'LC') 
            AND linkprim IN ('P', 'C') 
            AND usedflag = 1
        """

        ccm = pd.read_sql_query(
            sql=ccmxpf_linktable_query,
            con=wrds,
            dtype={"permno": int, "gvkey": str},
            parse_dates={"linkdt", "linkenddt"},
        )
        return ccm

    crsp_monthly = get_monthly_crsp_data(wrds, start_date, end_date)

    compustat = get_comp_data(wrds, start_date)

    ccmxpf_linktable = get_ccm_data(wrds)

    print(f"Merging CRSP and Compustat data")

    ccm_links = (
        crsp_monthly.merge(ccmxpf_linktable, how="inner", on="permno")
        .query("~gvkey.isnull() & (date >= linkdt) & (date <= linkenddt)")
        .get(["permno", "gvkey", "date"])
    )

    crsp_monthly = crsp_monthly.merge(ccm_links, how="left", on=["permno", "date"])

    comp = crsp_monthly.assign(year=lambda x: pd.DatetimeIndex(x["month"]).year).merge(
        compustat, how="left", on=["gvkey", "year"]
    )

    # Use Fama French 1993 timing convention, and use balance-sheet data from the fiscal year ending in year t − 1 for returns from July of year t to June of year t + 1.
    comp["jdate"] = (
        comp["datadate"] + MonthEnd(0) + pd.DateOffset(months=6)
    )  # Fama French 1993 timing convention

    comp["year"] = comp["jdate"].dt.year

    #########################
    # Momentum Factor       #
    #########################

    # Create (12,1) Momentum Factor with at least 6 months of returns

    print(f"Calculating momentum factor")

    _tmp_crsp = (
        comp.copy()[["permno", "date", "ret", "exchange", "mktcap"]]
        .sort_values(["permno", "date"])
        .set_index("date")
    )
    # replace missing return with 0
    _tmp_crsp["ret"] = _tmp_crsp["ret"].fillna(0)
    _tmp_crsp["logret"] = np.log(1 + _tmp_crsp["ret"])
    _tmp_cumret = (
        _tmp_crsp.groupby(["permno"])["logret"].rolling(12, min_periods=7).sum()
    )
    _tmp_cumret = _tmp_cumret.reset_index()
    _tmp_cumret["cumret"] = np.exp(_tmp_cumret["logret"]) - 1

    sizemom = pd.merge(
        _tmp_crsp.reset_index(),
        _tmp_cumret[["permno", "date", "cumret"]],
        how="left",
        on=["permno", "date"],
    )
    del _tmp_crsp, _tmp_cumret
    sizemom["mom"] = sizemom.groupby("permno")["cumret"].shift(1)
    sizemom = (
        sizemom[sizemom["date"].dt.month == 6]
        .drop(["logret", "cumret"], axis=1)
        .rename(columns={"mktcap": "size"})
    )

    #########################
    # CAPM Beta       #
    #########################
    # Calculate CAPM Beta
    # Product of correlations between the excess return of stock i and the market excess return and
    # the ratio of volatilities. We calculate volatilities from the standard deviations of daily log excess returns
    # over a one-year horizon requiring at least 120 observations.
    # We estimate correlations using overlapping three-day log excess returns over a five-year period requiring at
    # least 750 non-missing observations

    comp["logret"] = np.log(1 + comp["ret"])

    # Calculate Volatility
    comp["volatility"] = (
        comp.groupby("permno")["logret"]
        .rolling(252, min_periods=120)
        .std()
        .reset_index()["logret"]
    )

    # Calculate Correlation
    comp["laglogret"] = comp.groupby("permno")["logret"].shift(1)
    comp["laglogret2"] = comp.groupby("permno")["logret"].shift(2)
    comp["laglogret3"] = comp.groupby("permno")["logret"].shift(3)

    comp["laglogret"] = comp["laglogret"].fillna(0)
    comp["laglogret2"] = comp["laglogret2"].fillna(0)
    comp["laglogret3"] = comp["laglogret3"].fillna(0)

    #########################
    # NYSE Size Breakpoint  #
    #########################

    print(f"Calculating NYSE size breakpoints")

    # Get Size Breakpoints for NYSE firms
    sizemom = sizemom.sort_values(["date", "permno"]).drop_duplicates()
    nyse = sizemom[sizemom["exchange"] == "NYSE"]
    nyse_break = (
        nyse.groupby(["date"])["size"]
        .describe(percentiles=[0.2, 0.4, 0.6, 0.8])
        .reset_index()
    )
    nyse_break = nyse_break[["date", "20%", "40%", "60%", "80%"]].rename(
        columns={"20%": "dec20", "40%": "dec40", "60%": "dec60", "80%": "dec80"}
    )

    sizemom = pd.merge(sizemom, nyse_break, how="left", on="date")
    del nyse, nyse_break

    # Add NYSE Size Breakpoints to the Data
    def size_group(row):
        if 0 <= row["size"] < row["dec20"]:
            value = 1
        elif row["size"] < row["dec40"]:
            value = 2
        elif row["size"] < row["dec60"]:
            value = 3
        elif row["size"] < row["dec80"]:
            value = 4
        elif row["size"] >= row["dec80"]:
            value = 5
        else:
            value = np.nan
        return value

    sizemom["group"] = sizemom.apply(size_group, axis=1)
    sizemom["year"] = sizemom["date"].dt.year - 1
    sizemom = sizemom[["permno", "year", "mom", "group", "ret"]]
    comp = pd.merge(comp, sizemom, how="left", on=["permno", "year"])

    del sizemom

    # Close connection to wrds odbc database
    wrds.dispose()

    # Save in ipca_test_data/comp.parquet
    # comp.to_parquet('ipca_test_data/comp.parquet')

    return comp


def process_compustat(save=False):

    import pandas as pd
    import numpy as np
    import os

    # Set .streamlit/config.toml to [global]
    # dataFrameSerialization = "legacy"

    df = get_wrds()

    print(f"Constructing final characteristics")

    df = df.fillna(0)

    dimTicker = pd.DataFrame()
    dimTicker["permno"] = df["permno"]
    dimTicker["ticker"] = df["ticker"].astype("str")

    if not os.path.exists("factor_data"):
        os.makedirs("factor_data")
    dimTicker.to_parquet("factor_data/TickersPermnos.parquet")

    fundamentals = pd.DataFrame()
    fundamentals["permno"] = df["permno"]
    fundamentals["ticker"] = df["ticker"].astype("str")
    fundamentals["date"] = df["date"]
    fundamentals["year"] = df["date"].dt.year
    fundamentals["noa"] = (df["at"] - df["che"] - df["ivao"]) - (
        df["at"] - df["dlc"] - df["dltt"] - df["mib"] - df["pstk"] - df["ceq"]
    )
    fundamentals["prc"] = df["altprc"]
    fundamentals["shrout"] = df["shrout"]
    fundamentals["cap"] = fundamentals["prc"] * fundamentals["shrout"]
    fundamentals["me"] = fundamentals["cap"]
    # fundamentals['lme'] = df['lme'] # Not available yet...
    fundamentals["a2me"] = df["at"] / (fundamentals["shrout"] * fundamentals["prc"])
    fundamentals["ac"] = (df["act"] - df["che"] - df["lct"] - df["txp"]) / (
        df["be"] / (fundamentals["prc"] * fundamentals["prc"])
    )
    fundamentals["at"] = df["at"]
    fundamentals["ato"] = df["sale"] / fundamentals["noa"]
    fundamentals["beme"] = df["be"] / (fundamentals["prc"] * fundamentals["shrout"])
    # fundamentals['beta'] = df['beta'] # Not available yet...
    fundamentals["c"] = df["che"] / df["at"]
    fundamentals["cf"] = (df["ni"] + df["dp"]) / df["at"]
    fundamentals["cf2p"] = (df["ib"] + df["dp"] + df["txdb"]) / (
        fundamentals["prc"] * fundamentals["shrout"]
    )
    fundamentals["cto"] = df["sale"] / df["at"]
    fundamentals["d2a"] = df["dp"] / df["at"]

    fundamentals["dpi2a"] = (df["ppegt"] + df["inv"]) / df["at"]
    fundamentals["e2p"] = df["ni"] / fundamentals["prc"]
    fundamentals["fc2y"] = (df["xsga"] + df["xrd"]) / df["sale"]
    # fundamentals['IdioVol']  Standard deviation of the residuals from aregression of excess returns on the Fama and French three-factor model. Not available yet...
    fundamentals["lev"] = (df["dltt"] + df["dlc"]) / (
        df["dltt"] + df["dlc"] + df["seq"]
    )
    # fundamentals['lt_rev'] "Cumulative return from 60 months before the return prediction to 13 months before."
    # fundamentals['lturnover'] Coefficient of the market excess return from the regression on excess returns in the past 60 months (24 months minimum).
    fundamentals["pcm"] = (df["sale"] - df["cogs"]) / df["sale"]
    fundamentals["pm"] = df["oiadp"] / df["sale"]
    fundamentals["prof"] = (df["sale"] - df["cogs"]) / df["be"]
    fundamentals["q"] = (df["at"] + fundamentals["cap"] - df["ceq"] - df["txdb"]) / df[
        "at"
    ]
    # fundamentals['r2_1'] = df['laglogret']
    fundamentals["r12_2"] = df["mom"]
    fundamentals["rna"] = df["oiadp"] / fundamentals["noa"]
    fundamentals["roa"] = df["ib"] / df["at"]
    fundamentals["roe"] = df["ib"] / df["be"]
    fundamentals["s2p"] = df["sale"] / (fundamentals["prc"] * fundamentals["shrout"])
    fundamentals["sga2s"] = df["xsga"] / df["sale"]
    fundamentals["csho"] = df["csho"]
    fundamentals["ajex"] = df["ajex"]
    fundamentals_lags = (
        (
            fundamentals[["ticker", "year", "at", "cap", "csho", "ajex"]]
            .assign(year=lambda x: x["year"] + 1)
            .rename(
                columns={
                    "at": "at_lag",
                    "cap": "lme",
                    "csho": "csho_lag",
                    "ajex": "ajex_lag",
                }
            )
        )
        .groupby(["ticker", "year"])
        .last()
    )
    fundamentals = fundamentals.merge(
        fundamentals_lags, how="left", on=["ticker", "year"]
    )
    fundamentals = fundamentals.replace([np.inf, -np.inf], np.nan)
    fundamentals.fillna(0, inplace=True)
    fundamentals["ni"] = (
        np.log(1 + fundamentals["csho"] * fundamentals["ajex"])
        - np.log(1 + fundamentals["csho_lag"] * fundamentals["ajex_lag"])
    ).fillna(0)
    fundamentals["investment"] = (df["at_lag"] - df["at"]) / df["at_lag"]
    fundamentals["beta"] = df["beta"]
    fundamentals["oa"] = (
        df["act"] - df["che"] - df["lct"] - df["txp"] - df["dp"]
    ) / df["at_lag"]
    fundamentals["ol"] = (df["lt"] - df["dlc"] - df["dltt"]) / df["at_lag"]
    fundamentals["op"] = (
        df["sale"] - df["cogs"] - df["xsga"] - df["xint"] - df["txditc"]
    ) / df["at_lag"]
    # fundamentals['d2p'] = df['divamty'] / df['lme']

    fundamentals.fillna(0, inplace=True)

    # Drop ticker column
    fundamentals = fundamentals.drop(columns=["ticker"])

    if save:
        print(f"Saving characteristics")
        fundamentals.to_parquet("factor_data/MonthlyData.parquet")
        return

    return fundamentals
