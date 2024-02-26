# Import packages and functions
from functions import *
import pickle

########################################################################################################################
# ToDo's:
# 1. Get correct data
#   a) Get residual data for PCA, IPCA, and Fama-French
#   b) Get data for residual_weights (still not sure what that is - I think its the residual composition matrix
#       (see top of page 22 in Deep Learning Statistical Arbitrage))
#   c) We also need to get the risk-free rate

# 2. Set everything up for the other models
#   a) Set up for OU model (Everything is ready to be fit into the train/test functions I believe)
#   b) Set up for CNN model
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
df = pd.read_parquet('daily.parquet')[['ticker', 'datetime', 'return_1d']].set_index('datetime').pivot(
    columns='ticker').replace([np.inf, -np.inf], np.nan).ffill().bfill()
daily_dates = df.index.date
df = np.array(df)

residuals = df  # temporary residual matrix
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
objective = 'sharpe'
cwd = os.getcwd()
results_dict = {}

# Set up data
# Load IPCA, PCA, and Fama-French factors - ToDo: Files for our residual data should be in the list:
factors = ['PCA', 'IPCA', 'FamaFrench']

# Testing:
factors = ['TEST']
residuals = df

for i in range(len(factors)):

    print(f"Testing factor model: {factors[i]}")
    start_time = time.time()

    # residuals = pd.read_parquet(f"{factors[i]}.parquet")  # ToDo: Read the data in whatever format it is saved

    if use_residual_weights:
        residual_weights = ""  # ToDo: Load residual weigths beloning to the factor model we are running
    else:
        residual_weights = None

    # Define model
    model = model_name()
    preprocess = preprocess_fourier
    model_tag = model_tag + f"_{factors[i]}"

    print("Starting: " + model_tag)

    outdir = os.path.join(cwd, "Outputs", model_name_)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Estimate (train) model
    # ToDo: Costs are wrong - They have length_training=1000 and test_size=125 and batchsize=125
    rets_train, sharpe_train, ret_train, std_train, turnover_train, short_proportion_train = estimate(
        Data=residuals, model=FFT_FFN(), preprocess=preprocess_fourier,
        residual_weights=residual_weights, save_params=True, force_retrain=True, parallelize=False,
        log_dev_progress_freq=10,
        device=None, device_ids=[0, 1, 2, 3, 4, 5, 6, 7], output_path=outdir, num_epochs=100,
        lr=0.001, early_stopping=False, model_tag=model_tag, batchsize=50, length_training=200, test_size=50,
        lookback=30, trans_cost=0, hold_cost=0, objective="sharpe"
    )

    # Test model
    rets_test, sharpe_test, ret_test, std_test, turnover_test, short_proportion_test = test(
        Data=residuals, daily_dates=daily_dates, model=FFT_FFN(),
        preprocess=preprocess_fourier,
        residual_weights=residual_weights, save_params=True,
        force_retrain=True, parallelize=False,
        log_dev_progress_freq=10, log_plot_freq=149, device=None,
        device_ids=[0, 1, 2, 3, 4, 5, 6, 7], output_path=outdir,
        num_epochs=100, lr=0.001, early_stopping=False,
        model_tag=model_tag, batchsize=125, retrain_freq=125,
        rolling_retrain=True, length_training=200, lookback=30,
        trans_cost=0, hold_cost=0, objective="sharpe"
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

