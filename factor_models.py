import os
import numpy as np
import pandas as pd
import datetime
import time

from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import warnings
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

"""
Prerequisites 
"""


# Need this to run code from super-computer
def get_daily_data():
    """
    Returns daily excess returns, adjusted close, and volume
    """

    # Get tickers
    # tickers = pd.read_parquet('daily.parquet')['ticker'].unique()
    tickers = pd.read_csv("tickers.csv")["ticker"].unique()
    risk_free = get_risk_free_rate()

    # Get data
    data = []
    for tick in tqdm(tickers, desc="Downloading data", miniters=10):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            df = yf.download(tick, start="1998-12-31", end="2024-01-01", progress=False)
        df["ticker"] = tick
        df["return"] = (
            np.log(df["Adj Close"]) - np.log(df["Adj Close"].shift(1)) - risk_free
        )
        data.append(df[["ticker", "Adj Close", "Volume", "return"]])

    data = pd.concat(data)
    data = data[~(data["ticker"].isna())]
    data = data.pivot(columns="ticker")

    # Save data
    data.to_parquet("daily_data.parquet")

    return data


def de_annualize(annual_rate, periods=365):
    return (1 + annual_rate) ** (1 / periods) - 1


def get_risk_free_rate():
    # download 3-month us treasury bills rates
    annualized = yf.download("^IRX", start="1998-12-31", end="2024-01-01")["Adj Close"]

    # de-annualize
    daily = annualized.apply(de_annualize)

    # create dataframe
    return daily


"""
Run factor models
"""


def run_factor_models():
    # Initialize parameters for PCA
    factor_list = [1, 3, 5, 7]
    sizeCovarianceWindow = 252
    sizeWindow = 60
    intitialOOSYear = 2000
    df = pd.read_parquet("daily_data.parquet")

    # Fix NaN values
    df = df["return"]
    nan_percent = df.isna().mean() * 100
    drop_tickers = list(nan_percent[nan_percent > 0.16].index)
    df.drop(drop_tickers, axis=1, inplace=True)
    df.replace(np.nan, 0, inplace=True)
    df = df[1:]

    # Run PCA
    pca(factor_list, sizeCovarianceWindow, sizeWindow, intitialOOSYear, df)

    return


"""
PCA
"""


def pca(factor_list: list, sizeCovarianceWindow, sizeWindow, intitialOOSYear, df):
    # Get returns from data
    Rdaily = np.array(df.copy().reset_index(drop=True))
    T, N = Rdaily.shape
    firstOOSDailyIdx = np.argmax(df.index.year >= intitialOOSYear)
    factor_list = factor_list

    start_time = time.time()

    # Making sure all assets
    assets_to_consider = (
        np.count_nonzero(~np.isnan(Rdaily[firstOOSDailyIdx:, :]), axis=0) >= 30
    )
    Ntilde = np.sum(assets_to_consider)
    print(f"Number of assets to consider: {Ntilde} out of {N}")

    # ToDo: Here they run a filter on market-cap (we could potentially put a filter of volume in here)

    for factor in factor_list:
        residualsOOS = np.zeros((T - firstOOSDailyIdx, N), dtype=float)
        residualMatricesOOS = np.zeros(
            (T - firstOOSDailyIdx, Ntilde, Ntilde), dtype=np.float32
        )

        for t in tqdm(range(T - firstOOSDailyIdx), miniters=25):

            # ToDo: They drop the asset if it has a zero return in the period, but nearly all our assets have a zero
            # ToDo: return so I'll ignore it for now (change == 5670 to == 0 to change back)
            idxsSelected = ~np.any(
                Rdaily[
                    (t + firstOOSDailyIdx - sizeCovarianceWindow + 1) : (
                        t + firstOOSDailyIdx + 1
                    ),
                    :,
                ]
                == 5670,
                axis=0,
            ).ravel()

            if factor == 0:
                residualsOOS[t : (t + 1), idxsSelected] = Rdaily[
                    (t + firstOOSDailyIdx) : (t + firstOOSDailyIdx + 1), idxsSelected
                ]
            else:
                res_cov_window = Rdaily[
                    (t + firstOOSDailyIdx - sizeCovarianceWindow) : (
                        t + firstOOSDailyIdx
                    ),
                    idxsSelected,
                ]
                res_mean = np.mean(res_cov_window, axis=0, keepdims=True)
                res_vol = np.sqrt(
                    np.mean((res_cov_window - res_mean) ** 2, axis=0, keepdims=True)
                )
                res_normalized = (res_cov_window - res_mean) / res_vol
                corr = np.dot(
                    res_normalized.T, res_normalized
                )  # (x_1 - x_1_mean) * (x_2 - X_2_mean) / std_1 * std_2
                eigenvalues, eigenvectors = np.linalg.eig(corr)
                temp = np.argpartition(-eigenvalues, factor)
                idxs = temp[:factor]
                loadings = eigenvectors[
                    :, idxs
                ].real  # Takes eigenvector corresponding to factor largest eigenvalues
                factors = np.dot(res_cov_window[-sizeWindow:, :] / res_vol, loadings)
                DayFactors = np.dot(
                    Rdaily[t + firstOOSDailyIdx, idxsSelected] / res_vol, loadings
                )
                old_loadings = loadings
                regr = LinearRegression(fit_intercept=False, n_jobs=-1).fit(
                    factors, res_cov_window[-sizeWindow:, :]
                )
                loadings = regr.coef_
                residuals = Rdaily[t + firstOOSDailyIdx, idxsSelected] - DayFactors.dot(
                    loadings.T
                )
                residualsOOS[t : (t + 1), idxsSelected] = residuals

                Nprime = len(res_cov_window[-1:, :].ravel())
                MatrixFull = np.zeros((N, N))
                # MatrixReduced = I - 1 / res_vol * weights * beta.T  (equation 1 in DLSA)
                MatrixReduced = (
                    np.eye(Nprime)
                    - np.diag(1 / res_vol.squeeze()) @ old_loadings @ loadings.T
                )
                idxsSelected2 = idxsSelected.reshape((N, 1)) @ idxsSelected.reshape(
                    (1, N)
                )
                MatrixFull[idxsSelected2] = MatrixReduced.ravel()
                residuals2 = res_cov_window[-1:, :] @ MatrixReduced

                residualMatricesOOS[t : (t + 1)] = MatrixFull[assets_to_consider][
                    :, assets_to_consider
                ].T

        np.nan_to_num(residualsOOS, copy=False)
        np.nan_to_num(residualMatricesOOS, copy=False)

        print(f"Finished for factor {factor}")

        np.save(
            os.path.join(
                os.getcwd() + "/factor_outputs",
                f"OOSResidualsmatrix_PCA_factor{factor}_rollingwindow_{sizeWindow}.npy",
            ),
            residualMatricesOOS,
        )
        np.save(
            os.path.join(
                os.getcwd() + "/factor_outputs",
                f"OOSResiduals_PCA_factor{factor}_rollingwindow_{sizeWindow}.npy",
            ),
            residualsOOS,
        )

        print(f"Took {(time.time() - start_time) / 60} minutes to run PCA")

    return


"""
IPCA
"""


def run_ipca():
    import wrds_function

    print("Loading characteristics data")
    if not os.path.exists("factor_data/MonthlyData.parquet"):
        wrds_function.process_compustat(save=True)
    MonthlyData = pd.read_parquet("factor_data/MonthlyData.parquet")
    print("Loading daily returns")
    print("Preprocessing monthly characteristics data")
    preprocessMonthlyData(MonthlyData, normalizeCharacteristics=True)
    preprocessMonthlyData(MonthlyData, normalizeCharacteristics=False)
    print("Preprocessing daily returns")
    preprocessDailyReturns()
    print("Initializing IPCA factor model")
    ipca = IPCA(logdir=os.path.join("factor_data/residuals", "ipca_normalized"))
    for capProportion in [0.01]:  # , 0.001]:
        for sizeWindow in [4 * 12]:  # 15*12]:
            print(
                f"Running IPCA for window size {sizeWindow}, cap proportion {capProportion}"
            )
            ipca.DailyOOSRollingWindow(
                listFactors=[0, 1, 3, 5, 8, 10, 15],  # [0, 1, 3, 5, 8, 10, 15]
                initialMonths=36,  # 210
                sizeWindow=sizeWindow,
                CapProportion=capProportion,
                maxIter=1000,
                weighted=False,
                save=True,
                save_beta=False,
                save_gamma=False,
                save_rmonth=False,
                save_mask=False,
                save_sparse_weights_month=False,
                skip_oos=False,
                reestimationFreq=12,
            )
    return


def get_sharpe_tangencyPortfolio(df):  # returns has shape TxN
    if df.shape[1] == 1:
        return np.abs(np.mean(df)) / np.std(df)
    else:
        mean_ret = np.mean(df, axis=0, keepdims=True)
        cov_ret = np.cov(df.T)
        return float(np.sqrt(mean_ret @ np.linalg.solve(cov_ret, mean_ret.T)))


def preprocessMonthlyData(
    df,
    normalizeCharacteristics=True,
    logdir=os.getcwd(),
    name="factor_data/MonthlyDataNormalized.npz",
):
    org_df = df.copy()
    if normalizeCharacteristics:
        # Drop index
        df = df.copy().reset_index(drop=True)
        # Set index from date and ticker
        df.index = pd.MultiIndex.from_frame(df[["date", "ticker"]])
        # Omit ticker and date new df
        _df = df.drop(columns=["date", "ticker"])
        # Group _df by date and 'normalize'
        _df = _df.groupby("date").apply(
            lambda x: x.rank(method="first") / x.count() - 0.5
        )  # DLSA does it differently
        df = _df.copy()
    else:
        name = name.replace("Normalized.npz", "Unnormalized.npz")
        df.reset_index(inplace=True, drop=True)
        df.index = pd.MultiIndex.from_frame(org_df[["date", "ticker"]])
        df = df.drop(columns=["date", "ticker"])

    savepath = os.path.join(logdir, name)
    if os.path.exists(savepath):
        print(f"Monthly characteristics data already processed; skipping")
        return

    df.sort_index(inplace=True)
    shape = df.index.levshape + tuple([len(df.columns)])
    data = np.full(shape, np.nan)
    # ToDo: Need to come up with something else since we use ticker instead of permno.
    data[tuple(df.index.ticker)] = df.values

    date = df.index.levels[0].to_numpy()
    ticker = df.index.levels[1].to_numpy()
    variable = df.columns.to_numpy()

    np.save(os.path.join(logdir, "factor_data/MonthlyDataTickers.npy"), ticker)
    np.savez(savepath, data=data, date=date, ticker=ticker, variable=variable)
    return


def preprocessDailyReturns(
    logdir=os.getcwd(),
    name="daily_data.npz",
):
    from functions import get_daily_data

    savepath = os.path.join(logdir, name)
    if os.path.exists(savepath):
        print("Daily returns already processed; skipping")
        return
    df = get_daily_data(pivot=False)
    df["date"] = df.index
    df = df.reset_index(drop=True)[["date", "ticker", "return"]]
    df.index = pd.MultiIndex.from_frame(df[["date", "ticker"]])
    df = df.drop(columns=df.columns[range(0, 2)])
    df.sort_index(inplace=True)
    shape = df.index.levshape + tuple([len(df.columns)])
    _data = np.full(shape, np.nan)
    _data[tuple(df.index.codes)] = df.values
    data = _data[:, :, 0]

    date = df.index.levels[0].to_numpy()
    ticker = df.index.levels[1].to_numpy()

    # restrict returns data to only cover ticker that we have characteristics data for
    tmask = np.load(
        os.path.join(logdir, "factor_data/MonthlyDataTickers.npy"), allow_pickle=True
    )
    data = data[:, np.isin(ticker, tmask)]
    ticker = tmask

    np.savez(savepath, data=data, date=date, ticker=ticker)
    return


class IPCA:
    def __init__(
        self,
        individual_feature_dim=33,
        logdir=os.getcwd(),
        debug=True,
        pathMonthlyData="factor_data/MonthlyDataNormalized.npz",
        pathDailyData="daily_data.npz",
        pathMonthlyDataUnnormalized="factor_data/MonthlyDataUnnormalized.npz",
    ):
        self._individual_feature_dim = (
            individual_feature_dim  # this is the number of characteristics, L
        )
        self._logdir = logdir
        self._UNK = np.nan

        self._debug = debug

        monthlyData = np.load(pathMonthlyData, allow_pickle=True)
        dailyData = np.load(pathDailyData, allow_pickle=True)
        self.monthlyData = np.nan_to_num(monthlyData["data"])
        self.dailyData = dailyData
        self.monthlyDataUnnormalized = np.load(
            pathMonthlyDataUnnormalized, allow_pickle=True
        )["data"]
        # Extract the market cap
        self.monthlyCaps = self.monthlyData[
            :, :, 6
        ]  # ToDo: Ensure correct column index.

        self.dailyDates = pd.to_datetime(dailyData["date"])
        self.monthlyDates = pd.to_datetime(monthlyData["date"])
        self.weight_matrices = []
        self.mask = np.zeros(0)
        self.weighted = False

    def _step_factor(
        self, R_list, I_list, Gamma, calculate_residual=False, startIndex=0
    ):  # I are the characteristics Z in the paper
        f_list = []
        if calculate_residual:
            residual_list = []
        for t, riTuple in enumerate(zip(R_list, I_list)):
            R_t, I_t = riTuple
            beta_t = I_t.dot(Gamma)
            try:
                if self.weighted:
                    W_t = self.weight_matrices[t + startIndex]
                    A = beta_t.T @ W_t @ beta_t
                    f_t = np.linalg.solve(A, beta_t.T @ W_t @ R_t)
                else:
                    A = beta_t.T.dot(beta_t)
                    f_t = np.linalg.solve(A, beta_t.T.dot(R_t))

            except np.linalg.LinAlgError as err:
                # print(str(err))
                if self.weighted:
                    W_t = self.weight_matrices[t + startIndex]
                    f_t = np.linalg.pinv(beta_t.T @ W_t @ beta_t).dot(
                        beta_t.T @ W_t @ R_t
                    )
                else:
                    f_t = np.linalg.pinv(beta_t.T.dot(beta_t)).dot(beta_t.T.dot(R_t))
            f_list.append(f_t)
            if calculate_residual:
                residual_list.append(R_t - beta_t.dot(f_t))
        if calculate_residual:
            return f_list, residual_list
        else:
            return f_list, None

    def _step_gamma(self, R_list, I_list, f_list, nFactors, startIndex=0):
        A = np.zeros(
            (
                self._individual_feature_dim * nFactors,
                self._individual_feature_dim * nFactors,
            )
        )
        b = np.zeros((self._individual_feature_dim * nFactors, 1))
        for t, rifTuple in enumerate(zip(R_list, I_list, f_list)):
            R_t, I_t, f_t = rifTuple
            tmp_t = np.kron(I_t, f_t.T)
            if self.weighted:
                W_t = self.weight_matrices[t + startIndex]
                A += tmp_t.T @ W_t @ tmp_t
                b += tmp_t.T @ W_t @ R_t
            else:
                A += tmp_t.T.dot(tmp_t)
                b += tmp_t.T.dot(R_t)
        try:
            Gamma = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as err:
            # print(str(err))
            Gamma = np.linalg.pinv(A).dot(b)
        return Gamma.reshape((self._individual_feature_dim, nFactors))

    def _initial_factors(self, R_list, I_list, nFactors, startIndex=0):
        T = len(R_list)
        X = np.zeros((T, self._individual_feature_dim))
        for t in range(T):
            if self.weighted:
                W_t = self.weight_matrices[t + startIndex]
                X[t, :] = np.squeeze(I_list[t].T @ W_t @ R_list[t]) / len(R_list[t])
            else:
                X[t, :] = np.squeeze(I_list[t].T.dot(R_list[t])) / len(R_list[t])
        pca = PCA(n_components=nFactors)
        pca.fit(X.T)  # pca.components_ matrix is of shape nFactors x T
        f_list = pca.components_
        return np.split(f_list, f_list.shape[1], axis=1)

    def matrix_debug(self, matrix, name, extra=False):
        if not self._debug:
            return
        nanct = np.sum(np.isnan(matrix))
        zeroct = matrix.size - np.count_nonzero(matrix)
        nanpct = nanct / matrix.size * 100
        zeropct = zeroct / matrix.size * 100
        print(
            f"'{name}' {matrix.shape}: has {matrix.size} entries and {nanct} NaNs ({nanpct:0.4f}%) and {zeroct} zeros ({zeropct:0.4f}%)"
        )
        if extra:
            # also print which rows/columns are mostly/all NaNs
            nanrowcts = np.sum(np.isnan(matrix), axis=1)  # nans per row
            nancolcts = np.sum(np.isnan(matrix), axis=0)  # nans per column
            nanrowct = np.sum(
                nanrowcts > 0
            )  # number of rows in which there is 1+ NaN entries
            nancolct = np.sum(
                nancolcts > 0
            )  # number of cols in which there is 1+ NaN entries
            nanrowpct = nanrowct / matrix.shape[0] * 100
            nancolpct = nancolct / matrix.shape[1] * 100
            print(
                f"----> '{name}' {matrix.shape}: has {nanrowct} rows with NaNs ({nanrowpct:0.4f}%) and {nancolct} cols with NaNs ({nancolpct:0.4f}%)"
            )
            nanidxs = np.argwhere(np.isnan(matrix))  # nan indices
            nancols = np.unique(nanidxs[:, 1])  # column indices with nans
            colnancts = {col: np.sum(nanidxs[:, 1] == col) for col in nancols}
            print(
                f"----> '{name}' {matrix.shape}: NaNs counts in column indices (idx : count) {colnancts}"
            )

    def compute_sparse_residuals(
        self,
        full_residuals_month,
        beta_month,
        R_month_clean,
        sparsity_quantile_threshold=0.01,
    ):
        """
        Computes a set of sparse residuals closest to those in full_residuals_month by
        solving a sparse approximation problem for returns in R_month_clean using
        a candidate set of permnos given by information in beta_month.

        E.g. if
            S = full_residuals_month in R(T x P)
            R = R_month_clean        in R(T x P)
            P = proj(beta_month)     in R(P x P)
        and
            S = R(1-P)
        then we want to estimate a sparse matrix
            C in R(P x P)
        such that C is close to P in some norm and
            S ~= RC

        Currently, we solve that problem by
            1. Hard thresholding I-P by only keeping values greater in magnitude than the
               99th quantile
            2. Extracting a set of tickers from the thresholded I-P, the index mask 'h'
            3. Using this candidate set of entries in each row of P as selected variables
               to solve the regression for C[h,i], column i of matrix C restricted to rows in h:
                   S[:,i] = R[:,h] * C[h,i]
               I.e. all indices of C[:,i] are zero except for those in 'h', and we only
               select columns of R which are in 'h'.
            4. Running this regression for each col of S for which card(R[:,h]) > 1
            5. Letting the sparse residuals for the month be defined as C * R

        Future ideas include:
            - better sparse approximation/recovery than hard thresholding + linear regression
                (simple lasso regression, variations on OMP, (F)ISTA, etc.)
            - denoising S_i prior to approximation (e.g. solving the above regression
                minimizing the L1 norm of error instead of by L2 norm (e.g. by least squares))
            - using information in factor weighting matrix more carefully, e.g. by computing sparse eigenvectors
                via sparse PCA or sparse robust PCA and using those to get reconstructed, sparse
                (I-)P instead of thresholding (I-)P; (can sparse I-P be seen as negative graph lap?)
            - thresholding returns as well? how would we do this consistently through time? perhaps
                returns would need to be normalized (by rolling stats) first, but why would r_t+1 be
                near zero if r_t was? this may not be a realistic direction to explore.
            - making portfolios more consistent through time:
                - add penalty s.t.
                    |C(t-1) - C(t)|_1 <= f(|R(t-1)'R(t-1) - R(t)'R(t)|_2, |Z(t-1)'Z(t-1) - Z(t)'Z(t)|_2)
                    where f is some sum, product, or ratio of the changes in volatility amongst R and Z
                    in adjacent time periods
                - initialize with an initial hard thresholding or sparse robust PCA reconstruction
                    or lasso regression of (I-)P
            - learn sparsification over time given trading policy
                - make proportional to drawdown?
                - select thresholding parameter/sparsity penalty based on SCEMPC branch n bound?
                - include interaction with testing for mean reversion
            - learn way to filter out outliers in distributions of characteristics, or use heuristic:
                - market cap filter
                - volume
                - learn which permnos will be around next month from function of characteristics in prior months
        """
        # (1) do hard thresholding
        eye = np.identity(beta_month.shape[0])
        proj = (
            eye - beta_month @ np.linalg.pinv(beta_month.T @ beta_month) @ beta_month.T
        )
        # thresholding should only be done wrt observed (nonzero) returns' indices' weights,
        # so we set columns of proj to zero if more than half of the respective return column is missing
        # get idxs of columns where more than half of R_month_clean is zero
        mostly_zero_returns_col_idxs = np.argwhere(
            np.count_nonzero(R_month_clean, axis=0) <= R_month_clean.shape[0] // 2
        ).ravel()
        # set those columns in proj to zero
        proj[:, mostly_zero_returns_col_idxs] = 0
        # ToDo: Maybe hack
        if proj.size > 0:
            threshold = np.quantile(np.abs(proj), 1 - sparsity_quantile_threshold)
        else:
            threshold = 0  # or any other default value
        # threshold = np.quantile(np.abs(proj), 1 - sparsity_quantile_threshold) # ToDo: IndexError: index -1 is out of bounds for axis 0 with size 0
        np.putmask(proj, np.abs(proj) < threshold, 0)
        # (2) print stats and such
        if np.random.rand() > 0.9:
            nonzero_portfolios = R_month_clean @ (eye - proj)
            num_nonzero_portfolios = np.sum(
                np.count_nonzero(nonzero_portfolios, axis=0)
                >= nonzero_portfolios.shape[0] * 0.9
            )

            nonzero_portfolios = R_month_clean @ proj
            num_nonzero_portfolios = np.sum(
                np.count_nonzero(nonzero_portfolios, axis=0)
                >= nonzero_portfolios.shape[0] * 0.9
            )

            ps = np.sum(np.abs(proj) > 0, axis=1)  # portfolio sizes
            ps = ps[ps > 0]
            # the rows which include the ticker itself (that is, the diagonal entry is nonzero)
            include_self_idxs = np.argwhere(np.diag(np.abs(proj)) > 1e-12).ravel()
            num_include_self = min(len(include_self_idxs), num_nonzero_portfolios)
            num_just_self = np.sum(
                np.count_nonzero(proj[include_self_idxs, :], axis=1) == 1
            )
        # (3) now perform regression and return sparse residuals
        C = np.zeros_like(proj)
        for i in range(proj.shape[0]):
            # get indices of nonzero weights
            h = np.argwhere(proj[:, i] != 0).ravel()
            if len(h) == 0:
                continue
            elif len(h) == 1:
                if h[0] == i:
                    C[i, i] = 1
                    continue
                else:
                    # print("Hard thresholding got single weight which is not permno itself!")
                    continue
            Rh = R_month_clean[:, h]
            Si = full_residuals_month[:, i]
            if np.linalg.norm(Rh) <= 1e-8:
                pass
            C[h, i] = np.linalg.pinv(Rh.T.dot(Rh)).dot(Rh.T.dot(Si))
        return C

    def compute_weight_matrices(self, mask):
        """
        Computes a T-long sequence of NtxNt weight matrices Wt for the daily returns and mask provided,
        where Nt is the number of permnos in a given month and T is the number of months.
        Currently, only implements equal observation volatility weighting.
        """
        # Force mask to match length of dailyDate
        mask2 = mask[:, 0 : self.dailyData["data"].shape[1]]  # ToDo: This is a hack.
        weight_matrices = []
        for month in range(len(self.monthlyDates)):
            if month == 0:
                idxs_days_month = self.dailyDates <= self.monthlyDates[0]
            else:
                idxs_days_month = (self.dailyDates > self.monthlyDates[month - 1]) & (
                    self.dailyDates <= self.monthlyDates[month]
                )
            rmonth = self.dailyData["data"][:, mask2[month, :]][
                idxs_days_month, :
            ]  # TtxNt matrix
            nans_per_col = np.count_nonzero(np.isnan(rmonth), axis=0)
            insufficient_data_mask = nans_per_col >= np.round(0.9 * rmonth.shape[0])
            vols = np.nanvar(rmonth, axis=0)  # * rmonth.shape[0]  # Nt long vector
            vols[insufficient_data_mask] = np.nan
            weight_mtx = np.diag(1 / vols)
            weight_mtx[~np.isfinite(weight_mtx)] = 0
            weight_mtx[weight_mtx >= 10**6] = 0
            weight_matrices.append(weight_mtx)

        return weight_matrices

    # CapProportion = 0.001, 0.01  #The betas are going to be constant each month, so essentially constant at the daily level (and the factors will change daily)
    # SizeWindow must divide  = months of Data - training months

    def DailyOOSRollingWindow(
        self,
        save=True,
        weighted=False,
        listFactors=list(range(1, 21)),
        maxIter=1024,
        printOnConsole=True,
        printFreq=100,
        tol=1e-02,
        initialMonths=35 * 12,
        sizeWindow=1 * 12,  # ToDo: Original value was 15*12
        CapProportion=0.001,
        save_beta=False,
        save_gamma=False,
        save_rmonth=False,
        save_mask=False,
        save_sparse_weights_month=False,
        skip_oos=False,
        reestimationFreq=12,
    ):
        # ToDo: monthlyData loads 447 tickers, but dailyData loads 444 tickers. This is a problem.
        R = self.monthlyData[:, :, 0]  #
        # Replace NaNs with zeros
        R[np.isnan(R)] = 0  # ToDo: Test
        I = self.monthlyData[:, :, 1:]  #
        I[np.isnan(I)] = 0  # ToDo: Test
        cap_chosen_idxs = (
            self.monthlyCaps / np.nansum(self.monthlyCaps, axis=1, keepdims=True)
            >= CapProportion * 0.01
        )
        mask = (~np.isnan(R)) * cap_chosen_idxs
        self.mask = mask
        if save_mask:
            mask_path = os.path.join(
                self._logdir,
                f"mask_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy",
            )
            np.save(mask_path, mask)
        R_reshape = np.expand_dims(R[mask], axis=1)
        I_reshape = I[mask]
        splits = np.sum(mask, axis=1).cumsum()[
            :-1
        ]  # np.sum(mask, axis=1) how many stocks we have per year; the other cumukatively except for the last one
        R_list = np.split(R_reshape, splits)
        I_list = np.split(I_reshape, splits)
        self.R_list = R_list
        self.I_list = I_list
        nWindows = int((R.shape[0] - initialMonths) / reestimationFreq)
        print(f"nWindows {nWindows}")

        if weighted:
            self.weighted = True
            self.weight_matrices = self.compute_weight_matrices(mask)

        firstOOSDailyIdx = np.argmax(
            self.dailyDates
            >= (
                pd.to_datetime(
                    datetime.datetime(
                        self.dailyDates.year[0], self.dailyDates.month[0], 1
                    )
                )
                + pd.DateOffset(months=initialMonths)
            )
        )
        print(f"firstidx {firstOOSDailyIdx}")
        print(f"self.dailyData.shape[0] {self.dailyData['data'].shape[0]}")
        Rdaily = self.dailyData["data"][firstOOSDailyIdx:, :]
        sharpesFactors = np.zeros(len(listFactors))
        counter = 0

        DataTrain = Rdaily

        # DataTrain = np.load( # ToDo: What is this file? Not generated until later.
        #      os.path.join(
        #          "factor_data/residuals",
        #          "ipca_normalized",
        #          f"IPCA_DailyOOSresiduals_1_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy",
        #      )
        #  )
        assetsToConsider = (
            np.count_nonzero(DataTrain, axis=0) >= 30
        )  # chooses stocks which have at least #lookback non-missing observations in all the training time
        print(np.where(assetsToConsider))
        print(f"sum a2c {np.sum(assetsToConsider)}")
        Ntilde = np.sum(
            assetsToConsider
        )  # the residuals we are actually going to trade
        T, N = Rdaily.shape
        print(f"N {N} Ntilde {Ntilde}")
        superMask = np.count_nonzero(mask[initialMonths - 1 :], axis=0) >= 1
        superMask = superMask[0 : Rdaily.shape[1]]  # ToDo: This is a hack.
        Nsupertilde = np.sum(
            superMask
        )  # the maximum assets that are going to be involved in the interesting residuals
        print(f"superMask {superMask.shape} {Nsupertilde} {len(superMask)}")
        np.save("factor_data/residuals/super_mask.npy", superMask)

        if not os.path.isdir(
            self._logdir + "_stuff"
        ):  # ToDo: No idea why this is necessary.
            try:
                os.mkdir(self._logdir + "_stuff")
            except Exception as e:
                print(f"Could not create folder '{self._logdir + '_stuff'}'!")
                raise e

        if printOnConsole:
            print("Beginning daily residual computations")
        for nFactors in listFactors:
            residualsOOS = np.zeros_like(Rdaily, dtype=float)
            factorsOOS = np.zeros_like(Rdaily[:, :nFactors], dtype=float)
            sparse_oos_residuals = np.zeros_like(Rdaily, dtype=float)
            T, N = residualsOOS.shape
            residualsMatricesOOS = np.zeros((T, Ntilde, Nsupertilde), dtype=np.float32)

            # WeightsFactors = np.zeros((T,N,N))
            # WeightsSparseFactors = np.zeros((T,N,N))
            # Force mask to match length of dailyDate
            mask2 = mask[:, 0 : Rdaily.shape[1]]  # ToDo: This is a hack.
            if nFactors == 0:
                for month in range((initialMonths), R.shape[0]):
                    idxs_days_month = (
                        self.dailyDates[firstOOSDailyIdx:]
                        > self.monthlyDates[month - 1]
                    ) & (self.dailyDates[firstOOSDailyIdx:] <= self.monthlyDates[month])
                    R_month = Rdaily[:, mask2[month - 1, :]][
                        idxs_days_month, :
                    ]  # TxN # ToDo: IndexError: boolean index did not match indexed array along dimension 1; dimension is 444 but corresponding boolean dimension is 447

                    # change missing values to zeros to exclude them from calculation
                    R_month_clean = R_month.copy()
                    R_month_clean[np.isnan(R_month_clean)] = 0
                    residuals_month = R_month
                    # set residuals equal to zero wherever there are NaNs from missing returns, as (NaN - prediction) is still NaN
                    residuals_month[np.isnan(residuals_month)] = 0
                    sparse_residuals_month = residuals_month
                    temp = residualsOOS[:, mask2[month - 1, :]].copy()
                    temp[idxs_days_month, :] = residuals_month
                    residualsOOS[:, mask2[month - 1, :]] = temp
                    sparse_temp = sparse_oos_residuals[:, mask2[month - 1, :]].copy()
                    sparse_temp[idxs_days_month, :] = sparse_residuals_month
                    sparse_oos_residuals[:, mask2[month - 1, :]] = sparse_temp
            else:
                for nWindow in range(nWindows):
                    # Load or estimate Gamma; use save_gamma=True to force estimation
                    # Gamma estimation
                    # print("Estimating gamma")
                    if nWindow == 0:
                        gamma_path = os.path.join(
                            self._logdir + "_stuff",
                            f"gamma_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{reestimationFreq}_reestimationFreq_{CapProportion}_cap_{nWindow}.npy",
                        )
                        if os.path.isfile(gamma_path) and not save_gamma:
                            Gamma = np.load(gamma_path)
                            self._Gamma = Gamma
                        else:
                            f_list = self._initial_factors(
                                R_list[
                                    (
                                        initialMonths
                                        + nWindow * reestimationFreq
                                        - sizeWindow
                                    ) : (initialMonths + nWindow * reestimationFreq)
                                ],
                                I_list[
                                    (
                                        initialMonths
                                        + nWindow * reestimationFreq
                                        - sizeWindow
                                    ) : (initialMonths + nWindow * reestimationFreq)
                                ],
                                nFactors,
                                startIndex=initialMonths
                                + nWindow * reestimationFreq
                                - sizeWindow,
                            )
                            self._Gamma = np.zeros(
                                (self._individual_feature_dim, nFactors)
                            )
                            nIter = 0
                            while nIter < maxIter:
                                Gamma = self._step_gamma(
                                    R_list[
                                        (
                                            initialMonths
                                            + nWindow * reestimationFreq
                                            - sizeWindow
                                        ) : (initialMonths + nWindow * reestimationFreq)
                                    ],
                                    I_list[
                                        (
                                            initialMonths
                                            + nWindow * reestimationFreq
                                            - sizeWindow
                                        ) : (initialMonths + nWindow * reestimationFreq)
                                    ],
                                    f_list,
                                    nFactors,
                                    startIndex=initialMonths
                                    + nWindow * reestimationFreq
                                    - sizeWindow,
                                )
                                f_list, _ = self._step_factor(
                                    R_list[
                                        (
                                            initialMonths
                                            + nWindow * reestimationFreq
                                            - sizeWindow
                                        ) : (initialMonths + nWindow * reestimationFreq)
                                    ],
                                    I_list[
                                        (
                                            initialMonths
                                            + nWindow * reestimationFreq
                                            - sizeWindow
                                        ) : (initialMonths + nWindow * reestimationFreq)
                                    ],
                                    Gamma,
                                    startIndex=initialMonths
                                    + nWindow * reestimationFreq
                                    - sizeWindow,
                                )
                                nIter += 1
                                dGamma = np.max(np.abs(self._Gamma - Gamma))
                                self._Gamma = Gamma
                                if printOnConsole and nIter % printFreq == 0:
                                    print(
                                        "nFactor: %d,\t nWindow: %d/%d,\t nIter: %d,\t dGamma: %0.2e"
                                        % (nFactors, nWindow, nWindows, nIter, dGamma)
                                    )
                                if nIter > 1 and dGamma < tol:
                                    break
                            if save_gamma:
                                np.save(gamma_path, Gamma)
                    else:
                        gamma_path = os.path.join(
                            self._logdir + "_stuff",
                            f"gamma_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{reestimationFreq}_reestimationFreq_{CapProportion}_cap_{nWindow}.npy",
                        )
                        if os.path.isfile(gamma_path) and not save_gamma:
                            Gamma = np.load(gamma_path)
                            self._Gamma = Gamma
                        else:
                            nIter = 0
                            while nIter < maxIter / 2:
                                f_list, _ = self._step_factor(
                                    R_list[
                                        (
                                            initialMonths
                                            + nWindow * reestimationFreq
                                            - sizeWindow
                                        ) : (initialMonths + nWindow * reestimationFreq)
                                    ],
                                    I_list[
                                        (
                                            initialMonths
                                            + nWindow * reestimationFreq
                                            - sizeWindow
                                        ) : (initialMonths + nWindow * reestimationFreq)
                                    ],
                                    self._Gamma,
                                    startIndex=initialMonths
                                    + nWindow * reestimationFreq
                                    - sizeWindow,
                                )
                                Gamma = self._step_gamma(
                                    R_list[
                                        (
                                            initialMonths
                                            + nWindow * reestimationFreq
                                            - sizeWindow
                                        ) : (initialMonths + nWindow * reestimationFreq)
                                    ],
                                    I_list[
                                        (
                                            initialMonths
                                            + nWindow * reestimationFreq
                                            - sizeWindow
                                        ) : (initialMonths + nWindow * reestimationFreq)
                                    ],
                                    f_list,
                                    nFactors,
                                    startIndex=initialMonths
                                    + nWindow * reestimationFreq
                                    - sizeWindow,
                                )
                                nIter += 1
                                dGamma = np.max(np.abs(self._Gamma - Gamma))
                                self._Gamma = Gamma
                                if printOnConsole and nIter % printFreq == 0:
                                    print(
                                        "nFactor: %d,\t nWindow: %d/%d,\t nIter: %d,\t dGamma: %0.2e"
                                        % (nFactors, nWindow, nWindows, nIter, dGamma)
                                    )
                                if nIter > 1 and dGamma < tol:
                                    break
                            if save_gamma:
                                np.save(gamma_path, Gamma)

                    if not skip_oos:
                        # Computation of out-of-sample residuals
                        for month in range(
                            (initialMonths + nWindow * reestimationFreq),
                            (initialMonths + (nWindow + 1) * reestimationFreq),
                        ):
                            if self._debug:
                                print(
                                    f"--- Month: {month}/{(initialMonths+(nWindow+1)*sizeWindow)} ----"
                                )
                            beta_month = I[month - 1, mask[month - 1, :]].dot(
                                self._Gamma
                            )  # N x nfactors
                            # self.matrix_debug(beta_month, "beta_month")
                            idxs_days_month = (
                                self.dailyDates[firstOOSDailyIdx:]
                                > self.monthlyDates[month - 1]
                            ) & (
                                self.dailyDates[firstOOSDailyIdx:]
                                <= self.monthlyDates[month]
                            )
                            R_month = Rdaily[
                                :, mask2[month - 1, :]
                            ][  # ToDo: Hack the size of mask.
                                idxs_days_month, :
                            ]  # TxN
                            if save_rmonth:
                                r_path = os.path.join(
                                    self._logdir + "_stuff",
                                    f"rmonth_{month}_month_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{reestimationFreq}_reestimationFreq_{CapProportion}_cap.npy",
                                )
                                np.save(r_path, R_month)
                            # change missing values to zeros to exclude them from calculation

                            R_month_clean = R_month.copy()
                            beta_month[np.isnan(beta_month)] = 0
                            R_month_clean[np.isnan(R_month_clean)] = 0

                            # Check the shapes of the matrices
                            # print(beta_month.shape[0])  # e.g., (a, b)
                            # print(R_month_clean.shape[1])  # e.g., (c, d)

                            # Make sure a = d. In case a > d, we need to remove some columns from beta_month.
                            # In case a < d, we need to remove some columns from R_month_clean.
                            # In case a = d, we don't need to do anything.
                            if beta_month.shape[0] == R_month_clean.shape[1]:
                                pass
                            elif beta_month.shape[0] > R_month_clean.shape[1]:
                                beta_month = beta_month[: R_month_clean.shape[1], :]
                            elif beta_month.shape[0] < R_month_clean.shape[1]:
                                R_month_clean = R_month_clean[:, : beta_month.shape[0]]

                            if weighted:
                                # factors_month = np.linalg.pinv( # ToDo: Figure out how to extract W_month
                                #     beta_month.T @ W_month @ beta_month
                                # ).dot(
                                #     beta_month.T @ W_month @ R_month_clean.T
                                # )  # nfactors x T
                                factors_month = np.linalg.pinv(
                                    beta_month.T @ beta_month
                                ).dot(
                                    beta_month.T @ R_month_clean.T
                                )  # nfactors x T
                            else:
                                factors_month = np.linalg.pinv(
                                    beta_month.T @ beta_month
                                ).dot(
                                    beta_month.T @ R_month_clean.T
                                )  # nfactors x T
                            residuals_month = R_month - factors_month.T.dot(
                                beta_month.T
                            )
                            # set residuals equal to zero wherever there are NaNs from missing returns, as (NaN - prediction) is still NaN
                            residuals_month[np.isnan(residuals_month)] = 0

                            sparse_weights_month = self.compute_sparse_residuals(
                                residuals_month, beta_month, R_month_clean
                            )
                            sparse_residuals_month = (
                                R_month_clean @ sparse_weights_month
                            )
                            if save_sparse_weights_month:
                                sw_path = os.path.join(
                                    self._logdir + "_stuff",
                                    f"sparseweights_{month}_month_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{reestimationFreq}_reestimationFreq_{CapProportion}_cap.npy",
                                )
                                np.save(sw_path, sparse_weights_month)
                            if save_beta:
                                beta_path = os.path.join(
                                    self._logdir + "_stuff",
                                    f"beta_{month}_month_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{reestimationFreq}_reestimationFreq_{CapProportion}_cap.npy",
                                )
                                np.save(beta_path, beta_month)

                            temp = residualsOOS[:, mask2[month - 1, :]].copy()
                            temp[idxs_days_month, :] = residuals_month
                            residualsOOS[:, mask2[month - 1, :]] = temp
                            sparse_temp = sparse_oos_residuals[
                                :, mask2[month - 1, :]
                            ].copy()
                            sparse_temp[idxs_days_month, :] = sparse_residuals_month
                            sparse_oos_residuals[:, mask2[month - 1, :]] = sparse_temp
                            factorsOOS[idxs_days_month, :] = factors_month.T

                            Tprime, Nprime = R_month_clean.shape
                            MatrixFull = np.zeros((Tprime, N, N))
                            MatrixReduced = np.nan_to_num(
                                np.eye(Nprime)
                                - beta_month
                                @ np.linalg.pinv(beta_month.T @ beta_month)
                                @ beta_month.T
                            ).T  # Nprime x Nprime
                            mask_month = np.broadcast_to(
                                ~np.isnan(R_month).reshape((Tprime, Nprime, 1)),
                                (Tprime, Nprime, Nprime),
                            )
                            MatrixFull[
                                np.broadcast_to(
                                    mask2[month - 1, :].reshape((N, 1))
                                    @ mask2[month - 1, :].reshape((1, N)),
                                    (Tprime, N, N),
                                )
                            ] = (
                                mask_month
                                * np.broadcast_to(
                                    MatrixReduced, (Tprime, Nprime, Nprime)
                                )
                            ).ravel()
                            residualsMatricesOOS[idxs_days_month] = MatrixFull[
                                :, assetsToConsider
                            ][:, :, superMask]
                            portfolio1 = residualsOOS[idxs_days_month][
                                :, assetsToConsider
                            ]
                            portfolio2 = np.matmul(
                                residualsMatricesOOS[idxs_days_month],
                                np.nan_to_num(
                                    Rdaily[idxs_days_month][:, superMask]
                                ).reshape(Tprime, Nsupertilde, 1),
                            ).squeeze()
                            transition_error = (
                                np.linalg.norm(portfolio1 - portfolio2) / Tprime
                            )
                            # print(transition_error)

                if not skip_oos:
                    factorsOOS = np.nan_to_num(factorsOOS)
                    sharpesFactors[counter] = get_sharpe_tangencyPortfolio(factorsOOS)
                    counter += 1

            if printOnConsole:
                print("Finished (nFactors = %d)" % nFactors)
            if save and not skip_oos:
                rsavepath = os.path.join(
                    self._logdir,
                    f"IPCA_DailyOOSresiduals_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy",
                )
                msavepath = os.path.join(
                    self._logdir,
                    f"IPCA_DailyMatrixOOSresiduals_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy",
                )
                print(f"Saving {rsavepath}")
                # Ensure folders exist
                os.makedirs(os.path.dirname(rsavepath), exist_ok=True)
                np.save(rsavepath, residualsOOS)
                print(f"Saving {msavepath}")
                os.makedirs(os.path.dirname(msavepath), exist_ok=True)
                np.save(msavepath, residualsMatricesOOS)
        if not skip_oos:
            pass

        return

    def DailyOOSExpandingWindow(
        self,
        save=True,
        weighted=True,
        listFactors=list(range(1, 21)),
        maxIter=1024,
        printOnConsole=True,
        printFreq=8,
        tol=1e-03,
        initialMonths=30 * 12,
        sizeWindow=24 * 12,
        CapProportion=0.001,
        save_beta=False,
        save_gamma=False,
        save_rmonth=False,
        save_mask=False,
        save_sparse_weights_month=False,
        skip_oos=False,
    ):
        matrix_debug = self.matrix_debug
        R = self.monthlyData[:, :, 0]
        I = self.monthlyData[:, :, 1:]
        cap_chosen_idxs = (
            self.monthlyCaps / np.nansum(self.monthlyCaps, axis=1, keepdims=True)
            >= CapProportion * 0.01
        )
        mask = (~np.isnan(R)) * cap_chosen_idxs
        self.mask = mask
        if save_mask:
            mask_path = os.path.join(
                self._logdir,
                f"mask_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy",
            )
            np.save(mask_path, mask)
        with np.printoptions(threshold=np.inf):
            print(np.count_nonzero(mask, axis=1))
        R_reshape = np.expand_dims(R[mask], axis=1)
        I_reshape = I[mask]
        splits = np.sum(mask, axis=1).cumsum()[
            :-1
        ]  # np.sum(mask, axis=1) how many stocks we have per year; the other cumukatively except for the last one
        R_list = np.split(R_reshape, splits)
        I_list = np.split(I_reshape, splits)
        self.R_list = R_list
        self.I_list = I_list
        nWindows = int((R.shape[0] - initialMonths) / sizeWindow)
        print(f"nWindows {nWindows}")

        if weighted:
            self.weighted = True
            self.weight_matrices = self.compute_weight_matrices(mask)

        firstOOSDailyIdx = np.argmax(
            self.dailyDates
            >= (
                pd.datetime(self.dailyDates.year[0], self.dailyDates.month[0], 1)
                + pd.DateOffset(months=initialMonths)
            )
        )
        print(f"firstidx {firstOOSDailyIdx}")
        print(f"self.dailyData.shape[0] {self.dailyData.shape[0]}")
        Rdaily = self.dailyData[firstOOSDailyIdx:, :]
        sharpesFactors = np.zeros(len(listFactors))
        counter = 0

        if not os.path.isdir(self._logdir + "_stuff"):
            try:
                os.mkdir(self._logdir + "_stuff")
            except Exception as e:
                print(f"Could not create folder '{self._logdir + '_stuff'}'!")
                raise e

        if printOnConsole:
            print("Beginning daily residual computations")
        for nFactors in listFactors:
            residualsOOS = np.zeros_like(Rdaily, dtype=float)
            factorsOOS = np.zeros_like(Rdaily[:, :nFactors], dtype=float)
            sparse_oos_residuals = np.zeros_like(Rdaily, dtype=float)
            T, N = residualsOOS.shape
            # WeightsFactors = np.zeros((T,N,N))
            # WeightsSparseFactors = np.zeros((T,N,N))
            for nWindow in range(nWindows):
                if nFactors == 0:
                    for month in range(
                        (initialMonths + nWindow * sizeWindow),
                        (initialMonths + (nWindow + 1) * sizeWindow),
                    ):
                        idxs_days_month = (
                            self.dailyDates[firstOOSDailyIdx:]
                            > self.monthlyDates[month - 1]
                        ) & (
                            self.dailyDates[firstOOSDailyIdx:]
                            <= self.monthlyDates[month]
                        )
                        R_month = Rdaily[:, mask[month - 1, :]][
                            idxs_days_month, :
                        ]  # TxN
                        # change missing values to zeros to exclude them from calculation
                        R_month_clean = R_month.copy()
                        R_month_clean[np.isnan(R_month_clean)] = 0
                        residuals_month = R_month
                        # set residuals equal to zero wherever there are NaNs from missing returns, as (NaN - prediction) is still NaN
                        residuals_month[np.isnan(residuals_month)] = 0
                        sparse_residuals_month = residuals_month
                        temp = residualsOOS[:, mask[month - 1, :]].copy()
                        temp[idxs_days_month, :] = residuals_month
                        residualsOOS[:, mask[month - 1, :]] = temp
                        sparse_temp = sparse_oos_residuals[:, mask[month - 1, :]].copy()
                        sparse_temp[idxs_days_month, :] = sparse_residuals_month
                        sparse_oos_residuals[:, mask[month - 1, :]] = sparse_temp

                else:
                    # Load or estimate Gamma; use save_gamma=True to force estimation
                    gamma_path = os.path.join(
                        self._logdir + "_stuff",
                        f"gamma_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy",
                    )
                    if os.path.isfile(gamma_path) and not save_gamma:
                        Gamma = np.load(gamma_path)
                        self._Gamma = Gamma
                    # Gamma estimation
                    else:
                        # print("Estimating gamma")
                        if nWindow == 0:
                            f_list = self._initial_factors(
                                R_list[:initialMonths], I_list[:initialMonths], nFactors
                            )
                            self._Gamma = np.zeros(
                                (self._individual_feature_dim, nFactors)
                            )
                            nIter = 0
                            while nIter < maxIter:
                                Gamma = self._step_gamma(
                                    R_list[:initialMonths],
                                    I_list[:initialMonths],
                                    f_list,
                                    nFactors,
                                )
                                f_list, _ = self._step_factor(
                                    R_list[:initialMonths],
                                    I_list[:initialMonths],
                                    Gamma,
                                )
                                nIter += 1
                                dGamma = np.max(np.abs(self._Gamma - Gamma))
                                self._Gamma = Gamma
                                if printOnConsole and nIter % printFreq == 0:
                                    print(
                                        "nFactor: %d,\t nWindow: %d/%d,\t nIter: %d,\t dGamma: %0.2e"
                                        % (nFactors, nWindow, nWindows, nIter, dGamma)
                                    )
                                if nIter > 1 and dGamma < tol:
                                    break
                        else:
                            nIter = 0
                            while nIter < maxIter:
                                f_list, _ = self._step_factor(
                                    R_list[: (initialMonths + nWindow * sizeWindow)],
                                    I_list[: (initialMonths + nWindow * sizeWindow)],
                                    self._Gamma,
                                )
                                Gamma = self._step_gamma(
                                    R_list[: (initialMonths + nWindow * sizeWindow)],
                                    I_list[: (initialMonths + nWindow * sizeWindow)],
                                    f_list,
                                    nFactors,
                                )
                                nIter += 1
                                dGamma = np.max(np.abs(self._Gamma - Gamma))
                                self._Gamma = Gamma
                                if printOnConsole and nIter % printFreq == 0:
                                    print(
                                        "nFactor: %d,\t nWindow: %d/%d,\t nIter: %d,\t dGamma: %0.2e"
                                        % (nFactors, nWindow, nWindows, nIter, dGamma)
                                    )
                                if nIter > 1 and dGamma < tol:
                                    break
                        if save_gamma:
                            gamma_path = os.path.join(
                                self._logdir + "_stuff",
                                f"gamma_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy",
                            )
                            np.save(gamma_path, Gamma)

                    if not skip_oos:
                        # Computation of out-of-sample residuals
                        for month in range(
                            (initialMonths + nWindow * sizeWindow),
                            (initialMonths + (nWindow + 1) * sizeWindow),
                        ):
                            if self._debug:
                                print(
                                    f"--- Month: {month}/{(initialMonths+(nWindow+1)*sizeWindow)} ----"
                                )
                            beta_month = I[month - 1, mask[month - 1, :]].dot(
                                self._Gamma
                            )  # N x nfactors
                            idxs_days_month = (
                                self.dailyDates[firstOOSDailyIdx:]
                                > self.monthlyDates[month - 1]
                            ) & (
                                self.dailyDates[firstOOSDailyIdx:]
                                <= self.monthlyDates[month]
                            )
                            R_month = Rdaily[:, mask[month - 1, :]][
                                idxs_days_month, :
                            ]  # TxN
                            if save_rmonth:
                                r_path = os.path.join(
                                    self._logdir + "_stuff",
                                    f"rmonth_{month}_month_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy",
                                )
                                np.save(r_path, R_month)
                            # change missing values to zeros to exclude them from calculation
                            R_month_clean = R_month.copy()
                            R_month_clean[np.isnan(R_month_clean)] = 0
                            try:
                                if weighted:
                                    W_month = self.weight_matrices[month - 1]
                                    factors_month = np.linalg.solve(
                                        beta_month.T @ W_month @ beta_month,
                                        beta_month.T @ W_month @ R_month_clean.T,
                                    )  # nfactors x T
                                else:
                                    factors_month = np.linalg.solve(
                                        beta_month.T.dot(beta_month),
                                        beta_month.T.dot(R_month_clean.T),
                                    )  # nfactors x T
                            except np.linalg.LinAlgError as err:
                                print(f"----> Linear algebra error: {str(err)}")
                                if weighted:
                                    factors_month = np.linalg.pinv(
                                        beta_month.T @ W_month @ beta_month
                                    ).dot(
                                        beta_month.T @ W_month @ R_month.T
                                    )  # nfactors x T
                                else:
                                    factors_month = np.linalg.pinv(
                                        beta_month.T @ beta_month
                                    ).dot(
                                        beta_month.T @ R_month.T
                                    )  # nfactors x T
                            residuals_month = R_month - factors_month.T.dot(
                                beta_month.T
                            )
                            # set residuals equal to zero wherever there are NaNs from missing returns, as (NaN - prediction) is still NaN
                            residuals_month[np.isnan(residuals_month)] = 0
                            sparse_weights_month = self.compute_sparse_residuals(
                                residuals_month, beta_month, R_month_clean
                            )
                            sparse_residuals_month = (
                                R_month_clean @ sparse_weights_month
                            )
                            if save_sparse_weights_month:
                                sw_path = os.path.join(
                                    self._logdir + "_stuff",
                                    f"sparseweights_{month}_month_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy",
                                )
                                np.save(sw_path, sparse_weights_month)
                            if save_beta:
                                beta_path = os.path.join(
                                    self._logdir + "_stuff",
                                    f"beta_{month}_month_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy",
                                )
                                np.save(beta_path, beta_month)
                            temp = residualsOOS[:, mask[month - 1, :]].copy()
                            temp[idxs_days_month, :] = residuals_month
                            residualsOOS[:, mask[month - 1, :]] = temp
                            sparse_temp = sparse_oos_residuals[
                                :, mask[month - 1, :]
                            ].copy()
                            sparse_temp[idxs_days_month, :] = sparse_residuals_month
                            sparse_oos_residuals[:, mask[month - 1, :]] = sparse_temp
                            factorsOOS[idxs_days_month, :] = factors_month.T

                            if printOnConsole:
                                self.matrix_debug(residualsOOS, "residualsOOS")
                                self.matrix_debug(
                                    sparse_oos_residuals, "sparse_oos_residuals"
                                )

                if not skip_oos:
                    factorsOOS = np.nan_to_num(factorsOOS)
                    sharpesFactors[counter] = get_sharpe_tangencyPortfolio(factorsOOS)
                    counter += 1

            if printOnConsole:
                print("Finished! (nFactors = %d)" % nFactors)
            if save and not skip_oos:
                rsavepath = os.path.join(
                    self._logdir,
                    f"IPCA_DailyOOSresiduals_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy",
                )
                msavepath = os.path.join(
                    self._logdir,
                    f"IPCA_DailyMatrixOOSresiduals_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy",
                )
                print(f"Saving {rsavepath}")
                np.save(rsavepath, residualsOOS)

        if not skip_oos:
            pass

        return


"""
Fama French 
"""


class FamaFrench:
    def __init__(self, logdir=os.getcwd()):
        pathDailyData = "daily_data.npz"
        pathMonthlyDataUnnormalized = "factor_data/MonthlyDataUnnormalized.npz"
        pathMonthlyData = "factor_data/MonthlyDataNormalized.npz"
        self.monthlyDataUnnormalized = np.load(
            pathMonthlyDataUnnormalized, allow_pickle=True
        )["data"]
        self.monthlyCaps = np.nan_to_num(
            self.monthlyDataUnnormalized[:, :, 19]
        )  # ToDo: Check column index

        dailyData = np.load(pathDailyData, allow_pickle=True)
        monthlyData = np.load(pathMonthlyData, allow_pickle=True)
        self.monthlyData = monthlyData["data"]
        self.dailyData = dailyData["data"]
        self.dailyDates = pd.to_datetime(dailyData["date"])
        self.monthlyDates = pd.to_datetime(monthlyData["date"])

        self._logdir = logdir
        self.FamaFrenchFiveFactorsDaily = (
            pd.read_csv(
                "factor_data/F-F_Research_Data_5_Factors_2x3_daily.CSV", index_col=0
            )
            / 100
        )
        print(self.FamaFrenchFiveFactorsDaily.head())
        # breakpoint()

    def OOSRollingWindowPermnos(
        self,
        save=True,
        printOnConsole=True,
        initialOOSYear=1998,
        sizeWindow=60,
        cap=0.01,
        listFactors=list(range(8)),
    ):
        Rdaily = self.dailyData.copy()  # np.nan_to_num(self.dailyData)
        T, N = Rdaily.shape
        firstOOSDailyIdx = np.argmax(self.dailyDates.year >= initialOOSYear)
        firstOOSMonthlyIdx = np.argmax(self.monthlyDates.year >= initialOOSYear)
        firstOOSFFDailyIdx = np.argmax(
            self.FamaFrenchFiveFactorsDaily.index >= initialOOSYear * 10000
        )
        FamaFrenchDaily = self.FamaFrenchFiveFactorsDaily.to_numpy()
        OOSDailyDates = self.dailyDates[firstOOSDailyIdx:]
        cap_chosen_idxs = (
            self.monthlyCaps / np.nansum(self.monthlyCaps, axis=1, keepdims=True)
            >= cap * 0.01
        )
        mask = (~np.isnan(self.monthlyData[:, :, 0])) * cap_chosen_idxs

        filename = f"DailyFamaFrench_OOSresiduals_{3}_factors_{initialOOSYear}_initialOOSYear_{sizeWindow}_rollingWindow_{cap}_Cap.npy"
        DataTrain = np.load(os.path.join(self._logdir, filename))
        # chooses stocks which have at least #lookback non-missing observations in all the training time
        assetsToConsider = np.count_nonzero(DataTrain, axis=0) >= 30
        Ntilde = np.sum(assetsToConsider)
        print("N", N, "Ntilde", Ntilde)

        if printOnConsole:
            print("Computing residuals")

        for factor in listFactors:
            residualsOOS = np.zeros((T - firstOOSDailyIdx, N), dtype=float)
            notmissingOOS = np.zeros((T - firstOOSDailyIdx), dtype=float)
            monthlyIdx = firstOOSMonthlyIdx - 2
            residualsMatricesOOS = np.zeros(
                (T - firstOOSDailyIdx, Ntilde, Ntilde + factor), dtype=np.float32
            )

            for t in range(T - firstOOSDailyIdx):
                if (
                    self.dailyDates[t + firstOOSDailyIdx - 1].month
                    != self.dailyDates[t + firstOOSDailyIdx].month
                ):
                    monthlyIdx += 1
                idxsNotMissingValues = ~np.any(
                    np.isnan(
                        Rdaily[
                            (t + firstOOSDailyIdx - sizeWindow) : (
                                t + firstOOSDailyIdx
                            ),
                            :,
                        ]
                    ),
                    axis=0,
                ).ravel()
                print(idxsNotMissingValues.shape, mask[monthlyIdx, :].shape)
                print(self.monthlyDates[monthlyIdx], OOSDailyDates[t])
                idxsSelected = idxsNotMissingValues * mask[monthlyIdx, :]
                notmissingOOS[t] = np.sum(idxsNotMissingValues)

                if t % 100 == 0 and printOnConsole:
                    print(
                        f"At date {OOSDailyDates[t]}, Not-missing permnos: {notmissingOOS[t]}, "
                        f"Permnos with cap {np.sum(mask[monthlyIdx,:])}, Selected: {sum(idxsSelected)}"
                    )
                    print(
                        np.sum(idxsSelected) - np.sum(assetsToConsider * idxsSelected)
                    )
                if factor == 0:
                    residualsOOS[t : (t + 1), idxsSelected] = Rdaily[
                        (t + firstOOSDailyIdx) : (t + firstOOSDailyIdx + 1),
                        idxsSelected,
                    ]
                    residualsMatricesOOS[t : (t + 1), :, :Ntilde] = np.diag(
                        idxsSelected[assetsToConsider]
                    )
                else:
                    Y = Rdaily[
                        (t + firstOOSDailyIdx - sizeWindow) : (t + firstOOSDailyIdx),
                        idxsSelected,
                    ]
                    X = FamaFrenchDaily[
                        (t + firstOOSFFDailyIdx - sizeWindow) : (
                            t + firstOOSFFDailyIdx
                        ),
                        :factor,
                    ]
                    regr = LinearRegression(fit_intercept=False, n_jobs=48).fit(X, Y)
                    loadings = regr.coef_.T  # 5 x N
                    OOSreturns = Rdaily[
                        (t + firstOOSDailyIdx) : (t + firstOOSDailyIdx + 1),
                        idxsSelected,
                    ]
                    factors = FamaFrenchDaily[
                        (t + firstOOSFFDailyIdx) : (t + firstOOSFFDailyIdx + 1), :factor
                    ]  # TxnFactors
                    residuals = OOSreturns - factors.dot(loadings)
                    residualsOOS[t : (t + 1), idxsSelected] = np.nan_to_num(
                        residuals, copy=False
                    )

                    Loadings = np.zeros((N, factor))
                    Loadings[idxsSelected] = -loadings.T
                    residualsMatricesOOS[t, :, :Ntilde] = np.diag(
                        idxsSelected[assetsToConsider]
                    )  # np.eye(Ntilde)
                    residualsMatricesOOS[t, :, Ntilde:] = np.nan_to_num(
                        Loadings[assetsToConsider], copy=False
                    )
                    if t % 50 == 0 and printOnConsole:
                        concatenate = np.concatenate(
                            (
                                np.nan_to_num(
                                    Rdaily[(t + firstOOSDailyIdx), assetsToConsider],
                                    copy=False,
                                ),
                                FamaFrenchDaily[(t + firstOOSFFDailyIdx), :factor],
                            ),
                            axis=0,
                        )
                        print(
                            np.linalg.norm(
                                residualsOOS[t, assetsToConsider]
                                - residualsMatricesOOS[t] @ concatenate
                            )
                        )

            print("Transforming NaNs to nums")
            np.nan_to_num(residualsOOS, copy=False)
            np.nan_to_num(residualsMatricesOOS, copy=False)
            if printOnConsole:
                print(f"Finished! Cap: {cap}, factor: {factor}")
            if save:
                print(f"Saving")
                residuals_mtx_filename = (
                    f"DailyFamaFrench_OOSMatrixresiduals"
                    + f"_{factor}_factors_{initialOOSYear}_initialOOSYear_{sizeWindow}_rollingWindow_{cap}_Cap.npy"
                )
                np.save(
                    os.path.join(self._logdir, residuals_mtx_filename),
                    residualsMatricesOOS,
                )
                residuals_filename = (
                    f"DailyFamaFrench_OOSresiduals"
                    + f"_{factor}_factors_{initialOOSYear}_initialOOSYear_{sizeWindow}_rollingWindow_{cap}_Cap.npy"
                )
                np.save(os.path.join(self._logdir, residuals_filename), residualsOOS)
                print(f"Saved")


if __name__ == "__main__":
    run_factor_models()
