# Import packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pre_process import *
from FFT_FFN import *
import time

""" 
Functions needed to run the models
"""


# Intialize everything here - will clean up later
df = pd.read_parquet('daily.parquet')
df = df[(df['ticker'] == 'AAPL') | (df['ticker'] == 'GOOGL')][['ticker', 'datetime', 'return_1d']].\
    set_index('datetime').pivot(columns='ticker').dropna()
df.columns = ['AAPL', 'GOOGL']
# Make sure no there is no missing data
df = df.replace([np.inf, -np.inf], np.nan)
df = df.ffill().bfill()
df = np.array(df)

########################################################################################################################
# preprocess_fourier function (done)
########################################################################################################################

windows, idxs_selected = preprocess_fourier(df, lookback=30)

########################################################################################################################
# General training function
########################################################################################################################
from torch.optim import Adam

# input to function is training data
model = FFT_FFN(lookback=30, random_seed=69, hidden_units=[30, 16, 8, 4], dropout=0.25)  # ToDo: Will this work?
df_train = df[:round(len(df) * 0.66)]
parallelize = False  # We probably want "True"
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]  # must use device='cuda' to parallelize  (I don't know what this is - yet!)
lookback = 30
optimizer_opts = {"lr": 0.001}
num_epochs = 100
batchsize = 200


# Preprocess data
# assets_to_trade chooses assets which have at least `lookback` non-missing observations in the training period
# this does not induce lookahead bias because idxs_selected is backward-looking and
# will only select assets with at least `lookback` non-missing obs
assets_to_trade = np.count_nonzero(df_train, axis=0) >= lookback
df_train = df_train[:, assets_to_trade]

# ToDo: What data is used for residual_weights is still unclear at this moment - I'll hopefully find out
residual_weights_train = None
if residual_weights_train is not None:
    residual_weights_train = residual_weights_train[:, assets_to_trade]

T, N = df_train.shape

# ToDo: Uncomment this, when everything is in class
# windows, idxs_selected = self.preprocess_fourier(df_train, lookback)

# Start to train
if parallelize:
    model = nn.DataParallel(model, device_ids=device_ids)
model.to(None)  # ToDo: Change this if we start using GPU
model.train()  # Sets the mode of the model to training
optimizer = Adam(model.paramteters(), **optimizer_opts)

# Initialize variables
min_dev_loss = np.inf
patience = 0
trial = 0
already_trained = False

# ToDo: I'll drop the "check if already trained, checkpoint" (line 83-100 in train_test)

start_time = time.time
for epoch in range(num_epochs):
    rets_full = np.zeros(T - lookback)
    short_proportion = np.zeros(T - lookback)
    turnover = np.zeros(T - lookback)

    # Break input data into batches of size 'batchsize' and train over them, for computational efficiency
    
