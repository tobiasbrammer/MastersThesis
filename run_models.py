# Import packages and functions
from functions import *
import yaml
from pre_process import *

########################################################################################################################
# ToDo's:
# 1. Get correct data
#   a) Get residual data for IPCA and Fama-French
#       - Done for PCA

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
df = np.load("factor_data/daily_data.npz", allow_pickle=True)['data']
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


run_model(factors, model_name, preprocess, config, cwd, pd.DataFrame(daily_dates))


# Checking results
with open('results/CNNTransformer_results.pkl', 'rb') as f:
    results_CNN = pickle.load(f)

with open('results/FFT_results.pkl', 'rb') as f:
    results_FFT = pickle.load(f)

with open('results/OU_results.pkl', 'rb') as f:
    results_OU = pickle.load(f)


print(f'CNNTransformer: \n Sharpe: {results_CNN['CNNTransformer']['sharpe_test'] * np.sqrt(252) :.3f} \n '
      f'Return: {results_CNN['CNNTransformer']['ret_test'] * 252 * 100 :.3f}% \n '
      f'Std: {results_CNN['CNNTransformer']['std_test'] * np.sqrt(252) :.3f} \n '
      f'FFT: \n Sharpe: {results_FFT['FFT']['sharpe_test'] * np.sqrt(252) :.3f} \n '
      f'Return: {results_FFT['FFT']['ret_test'] * 252 * 100 :.3f}% \n '
      f'Std: {results_FFT['FFT']['std_test'] * np.sqrt(252) :.3f} \n '
      f'OU: \n Sharpe: {results_OU['OU']['sharpe_test'] * np.sqrt(252) :.3f} \n '
      f'Return: {results_OU['OU']['ret_test'] * 252 * 100 :.3f}% \n '
      f'Std: {results_OU['OU']['std_test'] * np.sqrt(252) :.3f} \n ')


plt.plot(daily_dates[(6289-5037):], (1 + results_CNN['CNNTransformer']['returns_test']).cumprod())

