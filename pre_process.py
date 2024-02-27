import torch
import numpy as np

"""
Functions to preprocess residual time series into time series for use with models.
"""


def preprocess_fourier(df, lookback):

    signal_length = lookback
    T, N = df.shape
    cumsums = np.cumsum(df, axis=0)
    windows = np.zeros((T - lookback, N, signal_length), dtype=np.float32)
    idxsSelected = np.zeros((T - lookback, N), dtype=bool)

    for t in range(lookback, T):

        idxsSelected[t - lookback, :] = ~np.any(
            df[(t - lookback) : t, :] == 0, axis=0
        ).ravel()

        idxs = idxsSelected[t - lookback, :]

        if t == lookback:
            windows[t - lookback, idxs, :] = cumsums[t - lookback : t, idxs].T

        else:
            windows[t - lookback, idxs, :] = cumsums[
                t - lookback : t, idxs
            ].T - cumsums[t - lookback - 1, idxs].reshape(int(sum(idxs)), 1)

    idxSelected = torch.as_tensor(idxsSelected)
    Fouriers = np.fft.rfft(windows, axis=-1)
    windows[:, :, : (lookback // 2 + 1)] = np.real(Fouriers)
    windows[:, :, (lookback // 2 + 1) :] = np.imag(Fouriers[:, :, 1:-1])

    del Fouriers
    return windows, idxSelected


def preprocess_ou(df, lookback):

    signal_length = lookback  # ToDo: Why does DLSA use 4?
    T, N = df.shape
    cumsums = np.cumsum(df, axis=0)
    windows = np.zeros((T - lookback, N, signal_length), dtype=np.float32)
    idxs_selected = np.zeros((T - lookback, N), dtype=bool)

    for t in range(lookback, T):
        # ToDo: Add jackknife to remove bias from kappas.
        # Update the idxs_selected array at the t - lookback row to mark features in the data array that
        # do not have any zero value in the last 'lookback' time steps.
        idxs_selected[t - lookback, :] = ~np.any(
            df[(t - lookback) : t, :] == 0, axis=0
        ).ravel()

        # Select the features that have no zero value in the last 'lookback' time steps.
        idxs = idxs_selected[t - lookback, :]

        if t == lookback:
            # Initialize the windows array with the selected features.
            x = cumsums[t - lookback : t, idxs].T
        else:
            # Calculate the relative change in the cumulative sum of the selected features.
            x = cumsums[t - lookback : t, idxs].T - cumsums[
                t - lookback - 1, idxs
            ].reshape(int(sum(idxs)), 1)
        # See DLSA appendix B for the OU model.
        Nx, Tx = (
            x.shape
        )  # Nx is assigned the number of rows in x, and Tx is assigned the number of columns.
        Ys = x[
            :, 1:
        ]  # Separate target variables in a dataset from the feature variables.
        Xs = x[
            :, :-1
        ]  # Separate feature variables in a dataset from the target variables.
        meansX = np.mean(Xs, axis=1)  # Calculate the mean of the feature variables.
        meansY = np.mean(Ys, axis=1)  # Calculate the mean of the target variables.
        VarsX = np.var(Xs, axis=1)  # Calculate the variance of the feature variables.
        VarsY = np.var(Ys, axis=1)  # Calculate the variance of the target variables.
        Covs = np.mean(  # Calculate the covariance of the feature and target variables.
            (Xs - meansX.reshape((Nx, 1))) * (Ys - meansY.reshape((Nx, 1))), axis=1
        )
        R2 = Covs**2 / (VarsX * VarsY)  # Calculate the R-squared value.
        bs = Covs / VarsX  # Calculate the coefficient of the feature variables.
        cs = meansY - bs * meansX  # Calculate the residual of the target variables.
        mus = cs / (1 - bs + 0.000001)  # \mu = \frac{a}{1 - b} in the paper.
        mask = (bs > 0) * (bs < 1)  # Analytical solution is only valid for 0 < b < 1.
        kappas = -np.log(bs) # \delta = -\log(b)/\Delta t, however \Delta t = 1 in the paper.
        residuals = Ys - bs.reshape((Nx, 1)) * Xs - cs.reshape((Nx, 1))  # (N,T-1)
        sigmas = np.sqrt(np.var(residuals, axis=1)*kappas / np.abs(1 - bs**2 + 0.000001))
        # Initialize the signal array with zeros.
        signal = np.zeros((Nx))
        # Update the signal array with the transformed residuals.
        signal[mask] = (mus[mask] - Ys[:, -1][mask]) / sigmas[mask]
        windows[t - lookback, idxs, 0] = Ys[
            :, -1
        ]  # Update the windows array with the target variables.
        windows[t - lookback, idxs, 1] = (
            mus  # Update the windows array with the transformed residuals.
        )
        windows[t - lookback, idxs, 2] = (
            sigmas  # Update the windows array with the standard deviation of the residuals.
        )
        windows[t - lookback, idxs, 3] = (
            R2  # Update the windows array with the R-squared value.
        )
        idxs_selected[t - lookback, idxs] = idxs_selected[t - lookback, idxs] & mask

    idxs_selected = torch.as_tensor(idxs_selected)
    return windows, idxs_selected
