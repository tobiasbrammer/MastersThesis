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
            df[(t - lookback): t, :] == 0, axis=0
        ).ravel()

        idxs = idxsSelected[t - lookback, :]

        if t == lookback:
            windows[t - lookback, idxs, :] = cumsums[t - lookback:t, idxs].T

        else:
            windows[t - lookback, idxs, :] = cumsums[t - lookback:t, idxs].T - \
                                             cumsums[t - lookback - 1, idxs].reshape(int(sum(idxs)), 1)

    idxSelected = torch.as_tensor(idxsSelected)
    Fouriers = np.fft.rfft(windows, axis=-1)
    windows[:, :, :(lookback // 2 + 1)] = np.real(Fouriers)
    windows[:, :, (lookback // 2 + 1):] = np.imag(Fouriers[:, :, 1:-1])

    del Fouriers
    return windows, idxSelected
