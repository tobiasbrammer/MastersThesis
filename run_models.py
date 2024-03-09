# Import packages and functions
from functions import *
import pickle
import yaml

########################################################################################################################
# ToDo's:
# 1. Get correct data
#   a) Get residual data for IPCA and Fama-French
#       - Done for PCA

#   b) Get data for residual_weights
#       - Done

#   c) We also need to get the risk-free rate
#       - Done (using yf's 3 months treasure bill)

# 2. Set everything up for the other models
#   bonus) Set up config for all models (Start with FFT)
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
with open('configs/FFT.yaml', 'r') as file:
    config = yaml.safe_load(file)
model_name = FFT_FFN
preprocess = preprocess_fourier

# Load factors - ToDo: Files for our residual data should be in the list:
factors = [config["res_pca_path"]]  # ['PCA', 'IPCA', 'FamaFrench']

results_dict = {}
for i in range(len(factors)):

    print(f"Testing factor model: {factors[i]}")
    start_time = time.time()

    residuals = np.load(factors[i])

    if config["use_residual_weights"]:
        # ToDo: Skal laves noget så den kan hente dem ordenligt
        residual_weights = np.load(config["residual_weights"])
    else:
        residual_weights = None

    # Define model
    model = model_name()
    model_tag = config["model_tag"]

    print("Starting: " + model_tag)

    outdir = os.path.join(cwd, "Outputs", config["model_name_"])
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Estimate (train) model
    # ToDo: So should we use data from 1945-1998 for this training func. and then rest for test?
    rets_train, sharpe_train, ret_train, std_train, turnover_train, short_proportion_train = estimate(
        Data=residuals, model=model, preprocess=preprocess_fourier, residual_weights=residual_weights, save_params=True,
        force_retrain=config["force_retrain"], parallelize=config["parallelize"], log_dev_progress_freq=10, device=None,
        device_ids=[0, 1, 2, 3, 4, 5, 6, 7], output_path=outdir, num_epochs=config["num_epochs"], lr=config["lr"],
        early_stopping=config["early_stopping"], model_tag=model_tag, batchsize=config["batchsize"],
        length_training=config["length_training"], test_size=config["retrain_freq"], lookback=config["lookback"],
        trans_cost=config["trans_cost"], hold_cost=config["hold_cost"], objective=config["objective"]
    )

    # Test model
    rets_test, sharpe_test, ret_test, std_test, turnover_test, short_proportion_test = test(
        Data=residuals, daily_dates=daily_dates, model=model, preprocess=preprocess_fourier,
        residual_weights=residual_weights, force_retrain=config["force_retrain"], parallelize=config["parallelize"],
        log_dev_progress_freq=10, log_plot_freq=149, device=None, device_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        output_path=outdir, num_epochs=config["num_epochs"], lr=config["lr"], early_stopping=config["early_stopping"],
        model_tag=model_tag, batchsize=config["batchsize"], retrain_freq=config["retrain_freq"],
        rolling_retrain=config["rolling_retrain"], length_training=config["length_training"],
        lookback=config["lookback"], trans_cost=config["trans_cost"], hold_cost=config["hold_cost"],
        objective=config["objective"]
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

    print(f"Time for {str(model_name)} factor model {factors[i]}: {(time.time() - start_time) / 60} minutes")
