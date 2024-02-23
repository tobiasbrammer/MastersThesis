########################################################################################################################
# The objective of this class is to run the Fast Fourier Transform (FFT), followed by a Feed-Forward Neural Network
# (FFN) to predict the stock price of a given company.
########################################################################################################################

# Import packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pre_process import *


class FFT_FFN(nn.Module):

    def __init__(self, lookback=30, random_seed=69, hidden_units=[30, 16, 8, 4], dropout=0.25, device="cpu"):

        super(FFT_FFN, self).__init__()
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        self.device = torch.device(device)
        self.hidden_units = hidden_units

        self.hiddenLayers = nn.ModuleList()
        for i in range(len(hidden_units) - 1):
            self.hiddenLayers.append(
                nn.Sequential(
                    nn.Linear(hidden_units[i], hidden_units[i + 1]),
                    nn.ReLU(True),
                    nn.Dropout(dropout),
                )
            )
        self.finalLayer = nn.Linear(hidden_units[-1], 1)
        self.is_trainable = True

    def forward(self, x):
        for i in range(len(self.hidden_units) - 1):
            x = self.hiddenLayers[i](x)
        return self.finalLayer(x).squeeze()






