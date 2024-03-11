import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import warnings
import yfinance as yf


"""
Run factor models
"""


def run_factor_models():
    # Initialize parameters for PCA
    factor_list = [5]
    sizeCovarianceWindow = 252
    sizeWindow = 60
    intitialOOSYear = 2000
    # df = pd.read_parquet('daily_data.parquet')
    df = get_daily_data()

    # Fix NaN values
    df = df['return']
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
    assets_to_consider = np.count_nonzero(~np.isnan(Rdaily[firstOOSDailyIdx:, :]), axis=0) >= 30
    Ntilde = np.sum(assets_to_consider)
    print(f"Number of assets to consider: {Ntilde} out of {N}")

    # ToDo: Here they run a filter on market-cap (we could potentially put a filter of volume in here)

    for factor in factor_list:
        residualsOOS = np.zeros((T - firstOOSDailyIdx, N), dtype=float)
        residualMatricesOOS = np.zeros((T - firstOOSDailyIdx, Ntilde, Ntilde), dtype=np.float32)

        for t in tqdm(range(T - firstOOSDailyIdx), miniters=25):

            # ToDo: They drop the asset if it has a zero return in the period, but nearly all our assets have a zero
            # ToDo: return so I'll ignore it for now (change == 5670 to == 0 to change back)
            idxsSelected = ~np.any(
                Rdaily[(t + firstOOSDailyIdx - sizeCovarianceWindow + 1):(t + firstOOSDailyIdx + 1), :] == 5670,
                axis=0).ravel()

            if factor == 0:
                residualsOOS[t:(t + 1), idxsSelected] = Rdaily[(t + firstOOSDailyIdx):(t + firstOOSDailyIdx + 1),
                                                        idxsSelected]
            else:
                res_cov_window = Rdaily[(t + firstOOSDailyIdx - sizeCovarianceWindow):(t + firstOOSDailyIdx),
                                 idxsSelected]
                res_mean = np.mean(res_cov_window, axis=0, keepdims=True)
                res_vol = np.sqrt(np.mean((res_cov_window - res_mean) ** 2, axis=0, keepdims=True))
                res_normalized = (res_cov_window - res_mean) / res_vol
                corr = np.dot(res_normalized.T, res_normalized)  # (x_1 - x_1_mean) * (x_2 - X_2_mean) / std_1 * std_2
                eigenvalues, eigenvectors = np.linalg.eig(corr)
                temp = np.argpartition(-eigenvalues, factor)
                idxs = temp[:factor]
                loadings = eigenvectors[:, idxs].real  # Takes eigenvector corresponding to factor largest eigenvalues
                factors = np.dot(res_cov_window[-sizeWindow:, :] / res_vol, loadings)
                DayFactors = np.dot(Rdaily[t + firstOOSDailyIdx, idxsSelected] / res_vol, loadings)
                old_loadings = loadings
                regr = LinearRegression(fit_intercept=False, n_jobs=-1).fit(factors, res_cov_window[-sizeWindow:, :])
                loadings = regr.coef_
                residuals = Rdaily[t + firstOOSDailyIdx, idxsSelected] - DayFactors.dot(loadings.T)
                residualsOOS[t:(t + 1), idxsSelected] = residuals

                Nprime = len(res_cov_window[-1:, :].ravel())
                MatrixFull = np.zeros((N, N))
                # MatrixReduced = I - 1 / res_vol * weights * beta.T  (equation 1 in DLSA)
                MatrixReduced = (np.eye(Nprime) - np.diag(1 / res_vol.squeeze()) @ old_loadings @ loadings.T)
                idxsSelected2 = idxsSelected.reshape((N, 1)) @ idxsSelected.reshape((1, N))
                MatrixFull[idxsSelected2] = MatrixReduced.ravel()
                residuals2 = res_cov_window[-1:, :] @ MatrixReduced

                residualMatricesOOS[t:(t + 1)] = MatrixFull[assets_to_consider][:, assets_to_consider].T

        np.nan_to_num(residualsOOS, copy=False)
        np.nan_to_num(residualMatricesOOS, copy=False)

        print(f"Finished for factor {factor}")

        np.save(os.path.join(os.getcwd() + "/factor_outputs",
                             f'OOSResidualsmatrix_PCA_factor{factor}_rollingwindow_{sizeWindow}.npy'),
                residualMatricesOOS)
        np.save(os.path.join(os.getcwd() + "/factor_outputs",
                             f'OOSResiduals_PCA_factor{factor}_rollingwindow_{sizeWindow}.npy'),
                residualsOOS)

        print(f"Took {(time.time() - start_time) / 60} minutes to run PCA")

    return


# Need this to run code from super-computer
def get_daily_data():
    """
    Returns daily excess returns, adjusted close, and volume
    """

    # Get tickers
    # tickers = pd.read_parquet('daily.parquet')['ticker'].unique()
    tickers = pd.read_csv('tickers.csv')['ticker'].unique()
    risk_free = get_risk_free_rate()

    # Get data
    data = []
    for tick in tqdm(tickers, desc='Downloading data', miniters=10):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            df = yf.download(tick, start="1998-12-31", end="2024-01-01", progress=False)
        df['ticker'] = tick
        df['return'] = np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(1)) - risk_free
        data.append(df[['ticker', 'Adj Close', 'Volume', 'return']])

    data = pd.concat(data)
    data = data[~(data['ticker'].isna())]
    data = data.pivot(columns='ticker')

    # Save data
    data.to_parquet('daily_data.parquet')

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


if __name__ == "__main__":
    run_factor_models()
