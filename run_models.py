# Import packages and functions
from functions import *
import pickle

########################################################################################################################
# ToDo's:
# 1. Get correct data
#   a) Get residual data for IPCA and Fama-French
#       - Done for PCA

#   b) Get data for residual_weights (still not sure what that is - I think its the residual composition matrix
#       (see top of page 22 in Deep Learning Statistical Arbitrage))

#   c) We also need to get the risk-free rate
#       - Done (using yf's 3 months treasure bill)

# 2. Set everything up for the other models
#   a) Set up for OU model (Everything is ready to be fit into the train/test functions I believe)

#   b) Set up for CNN model

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
# Get data - (Temporary - maybe keep for the daily_dates?)
# Test if daily_data.parquet exists, else get it
if not os.path.exists('daily_data.parquet'):
    print("Downloading daily data")
    get_daily_data()

df = pd.read_parquet('daily_data.parquet')[['return']][1:]
df.replace(np.nan, 0, inplace=True)
daily_dates = df.index.date
df = np.array(df)
df = np.nan_to_num(df, nan=0, posinf=0, neginf=0)

res_pca_path = "factor_outputs/OOSResiduals_PCA_factor5_rollingwindow_60.npy"
residual_weights = None  # The residual composition matrix
use_residual_weights = False  # Should be True

# FFT model_tag
model_tag = "FFT"  # ToDo: What do we want to include? lookback, objective, trans_cost, hold_cost, num_epochs, lr ??

# Output path
outdir = os.path.join(os.getcwd(), "Outputs")

# Initialize torch
torch.set_default_dtype(torch.float)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = False

# Model parameters
model_name = FFT_FFN
model_name_ = "FFT_FFN"  # For saving in correct directory
factor_models = {"IPCA": [5], "PCA": [5], "FamaFrench": [5]}  # We are only running 5-factor models (for now)
objective = 'sharpe' # Can be 'sharpe', 'meanvar' or 'sqrtMeanSharpe'
cwd = os.getcwd()
results_dict = {}

# Set up data
# Load IPCA, PCA, and Fama-French factors - ToDo: Files for our residual data should be in the list:
factors = [res_pca_path]   # ['PCA', 'IPCA', 'FamaFrench']



for i in range(len(factors)):

    print(f"Testing factor model: {factors[i]}")
    start_time = time.time()

    residuals = np.load(factors[i])
    # Residuals er TxNxN fordi at den indeholder vores portfølje for HVER asset

    if use_residual_weights:
        residual_weights = ""  # ToDo: Load residual weights belonging to the factor model we are running
    else:
        residual_weights = None

    # Define model
    model = model_name()
    preprocess = preprocess_fourier
    model_tag = model_tag

    print("Starting: " + model_tag)

    outdir = os.path.join(cwd, "Outputs", model_name_)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Estimate (train) model
    # ToDo: Costs are wrong - They have length_training=1000 and test_size=125 and batch_size=125
    rets_train, sharpe_train, ret_train, std_train, turnover_train, short_proportion_train = estimate(
        Data=residuals, model=FFT_FFN(), preprocess=preprocess_fourier,
        residual_weights=residual_weights, save_params=True, force_retrain=True, parallelize=False,
        log_dev_progress_freq=10,
        device=None, device_ids=[0, 1, 2, 3, 4, 5, 6, 7], output_path=outdir, num_epochs=100,
        lr=0.00000001, early_stopping=False, model_tag=model_tag, batchsize=50, length_training=200, test_size=50,
        lookback=30, trans_cost=0, hold_cost=0, objective=objective
    )

    # Test model
    rets_test, sharpe_test, ret_test, std_test, turnover_test, short_proportion_test = test(
        Data=residuals, daily_dates=daily_dates, model=FFT_FFN(),
        preprocess=preprocess_fourier,
        residual_weights=residual_weights, save_params=True,
        force_retrain=True, parallelize=False,
        log_dev_progress_freq=10, log_plot_freq=149, device=None,
        device_ids=[0, 1, 2, 3, 4, 5, 6, 7], output_path=outdir,
        num_epochs=10, lr=0.001, early_stopping=False,
        model_tag=model_tag, batchsize=50, retrain_freq=125,
        rolling_retrain=True, length_training=200, lookback=30,
        trans_cost=0, hold_cost=0, objective=objective
    )

    results_dict[model_tag] = {
        "returns_train": rets_train, "returns_test": rets_test,
        "sharpe_train": sharpe_train, "sharpe_test": sharpe_test,
        "ret_train": ret_train, "ret_test": ret_test,
        "std_train": std_train, "std_test": std_test,
        "turnover_train": turnover_train, "turnover_test": turnover_test,
        "short_proportion_train": short_proportion_train, "short_proportion_test": short_proportion_test
    }

    # Save results
    with open(f"{cwd}/results/{model_tag}_results.pkl", "wb") as f:
        pickle.dump(results_dict, f)

    print(f"Time for {model_name_} factor model {factors[i]}: {time.time() - start_time}")
