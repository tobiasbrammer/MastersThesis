########################################################################################################################
# The objective of this class is to run the Fast Fourier Transform (FFT), followed by a Feed-Forward Neural Network
# (FFN) to predict the stock price of a given company.
########################################################################################################################

# Import packages
import numpy as np
import pandas as pd
import torch


class FFT_FFN():
    def __init__(self, data):
        self.df = data
        self.lookback = 30

    def preprocess_fourier(self):

        signal_length = self.lookback
        T, N = self.df.shape
        cumsums = np.cumsum(self.df, axis=0)
        windows = np.zeros((T - self.lookback, N, signal_length), dtype=np.float32)
        idxsSelected = np.zeros((T - self.lookback, N), dtype=bool)

        # Make sure no there is no missing data
        self.df = self.df.ffill().bfill()

        for t in range(self.lookback, T):

            idxsSelected[t - self.lookback, :] = ~np.any(
                self.df[(t - self.lookback): t, :] == 0, axis=0
            ).ravel()

            idxs = idxsSelected[t - self.lookback, :]

            if t == self.lookback:
                windows[t - self.lookback, idxs, :] = cumsums[t - self.lookback:t, idxs].T

            else:
                windows[t - self.lookback, idxs, :] = cumsums[t - self.lookback:t, idxs].T - \
                                                      cumsums[t - self.lookback - 1, idxs].reshape(int(sum(idxs)), 1)

        idxSelected = torch.as_tensor(idxsSelected)
        Fouriers = np.fft.rfft(windows, axis=-1)
        windows[:, :, :(self.lookback // 2 + 1)] = np.real(Fouriers)
        windows[:, :, (self.lookback // 2 + 1):] = np.imag(Fouriers[:, :, 1:-1])

        del Fouriers
        return windows, idxSelected







# Testing zone

# Import packages
import yfinance as yf
import math
import pandas as pd
from keras import models, layers, optimizers, regularizers
import numpy as np
import random
from sklearn import model_selection, preprocessing
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

# Get data
df_googl = yf.download('GOOGL', start='2010-01-01', end='2024-01-01')
df_appl = yf.download('AAPL', start='2010-01-01', end='2024-01-01')

df = df_googl.join(df_appl, lsuffix='_googl', rsuffix='_appl')
df = df[['Close_googl', 'Close_appl']]


def calc_ffn(df):

    # Split data
    df_train = df.iloc[:round(len(df) * 0.66)]
    df_test = df.iloc[round(len(df) * 0.66):]

    train_x = df_train[:-1]
    train_y = df_train[1:]
    test_x = df_test[:-1]
    test_y = df_test[1:]

    # Normalize the data (when multiple features)
    train_x = preprocessing.normalize(train_x)
    test_x = preprocessing.normalize(test_x)

    # Initializing model
    hidden_units = [30, 16, 8, 4]
    activation = 'relu'
    l2 = 0.01
    learning_rate = 0.01
    epochs = 5
    batch_size = 16

    # Create model
    model = models.Sequential()

    # Add the hidden layers
    # ToDo: We need multiple layers, going from 30 - 16 - 8 - 4 (remember the input has to be the previous layer)
    model.add(layers.Dense(input_dim=len(df.columns), units=hidden_units[0], activation=activation))
    model.add(layers.Dense(input_dim=len(df.columns), units=hidden_units[1], activation=activation))
    model.add(layers.Dense(input_dim=len(df.columns), units=hidden_units[2], activation=activation))
    model.add(layers.Dense(input_dim=len(df.columns), units=hidden_units[3], activation=activation))
    # Jeg tror måske de skal sættes i en liste og køres?

    # Add the output layer
    model.add(layers.Dense(input_dim=hidden_units, units=1, activation='relu'))

    # Define our loss function and optimizer
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

    # Train and evaluate model
    train_acc, test_acc = train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs, batch_size, 20)


def train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs, batch_size, iter):
    train_accs = []
    test_accs = []

    for _ in tqdm(range(iter)):
        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=False)
        train_accs.append(model.evaluate(train_x, train_y, batch_size=batch_size, verbose=False)[1])
        test_accs.append(model.evaluate(test_x, test_y, batch_size=batch_size, verbose=False)[1])

    print('Avgerage Training Accuracy: %s' % np.average(train_accs))
    print('Avgerage Testing Accuracy: %s' % np.average(test_accs))

    return train_accs, test_accs
