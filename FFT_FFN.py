########################################################################################################################
# The objective of this class is to run the Fast Fourier Transform (FFT), followed by a Feed-Forward Neural Network
# (FFN) to predict the stock price of a given company.
########################################################################################################################

# Import packages
import numpy as np
import pandas as pd


class FFT_FFN():
    def __init__(self, data):
        self.df = data



