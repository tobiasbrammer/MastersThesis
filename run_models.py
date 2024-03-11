# Import packages and functions
from functions import *
import yaml
from pre_process import *

########################################################################################################################
# ToDo's:
# 1. Get correct data
#   a) Get residual data for IPCA and Fama-French
#       - Done for PCA

# 3. We still need to make parallelization work - But it is not needed for now

# 4. How should we handle missing data? E.g. AirBnB (ABNB) only has data from 2021
#    (vi har 496 aktier, hvoraf 345 har data fra 1999 til 2024)
#    Possible solution: Set all NaN to 0, as we always run "assets_to_consider (count_nonzero >= 30)"
########################################################################################################################

########################################################################################################################
########################################################################################################################
# Run model for FFT
########################################################################################################################
########################################################################################################################


########################################################################################################################
# Initialize parameters
########################################################################################################################
# Load data
df = np.load('daily_data_run.npy')
daily_dates = np.load('daily_dates.npy', allow_pickle=True)

# Set output path and cwd
outdir = os.path.join(os.getcwd(), "Outputs")
cwd = os.getcwd()

# Initialize torch
torch.set_default_dtype(torch.float)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = False

# Choose model
# with open('configs/FFT.yaml', 'r') as file:
#     config = yaml.safe_load(file)
# model_name = FFT_FFN
# preprocess = preprocess_fourier

# with open('configs/OU.yaml', 'r') as file:
#     config = yaml.safe_load(file)
# model_name = OU_FFN
# preprocess = preprocess_ou

with open('configs/CNNTransformer.yaml', 'r') as file:
    config = yaml.safe_load(file)
model_name = CNNTransformer_FFN
preprocess = preprocess_cnn

# Load factors - ToDo: Files for our residual data should be in the list:
factors = [config["res_pca_path"]]  # ['PCA', 'IPCA', 'FamaFrench']


run_model(factors, model_name, preprocess, config, cwd, daily_dates)

