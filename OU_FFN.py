########################################################################################################################
# The objective of this class is to run the Ornstein–Uhlenbeck, followed by a Feed-Forward Neural Network
# (FFN) to predict the stock price of a given company.
########################################################################################################################


import torch
import torch.nn as nn

# Defining the custom model class
class OU_FFN(nn.Module):
    def __init__(
        self,
        logdir,
        lookback=30,
        random_seed=0,
        device="cpu",
        hidden_units=[4, 4, 4, 4],
        dropout=0.25,
    ):
        # Calling the constructor of the parent nn.Module class
        super(OU_FFN, self).__init__()

        # Setting class instance properties
        self.logdir = logdir
        self.random_seed = random_seed

        # Setting the seed for generating random numbers
        torch.manual_seed(self.random_seed)

        # Setting the device for tensor computations
        self.device = torch.device(device)

        # Setting the architecture of the hidden layers
        self.hidden_units = hidden_units
        self.is_trainable = True

        # Creating the hidden layers
        self.hiddenLayers = nn.ModuleList()
        for i in range(len(hidden_units) - 1):
            # Each hidden layer consists of a linear transformation, a Sigmoid activation,
            # and a dropout regularization
            layer = nn.Sequential(
                nn.Linear(hidden_units[i], hidden_units[i + 1]),
                nn.Sigmoid(),
                nn.Dropout(dropout),
            )
            self.hiddenLayers.append(layer)

        # Creating the final layer
        self.finalLayer = nn.Linear(hidden_units[-1], 1)

    def forward(self, x):
        # Defining how an input tensor is processed through the network layers
        for layer in self.hiddenLayers:
            x = layer(x)
        return self.finalLayer(x).squeeze()
