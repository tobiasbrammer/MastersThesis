# Import packages and functions
from functions import *
import yaml
from pre_process import *
import os

from tqdm import tqdm



def PlotSeries(dict, dictTitle, sFactorModel: str, iFactors: int):
    """
    Plot a time series of our results, and upload to overleaf.

    dict: dict e.g. results_CNN
    dictTitle: str e.g. CNNTransformer
    sFactorModel: str e.g. PCA
    iFactors: int e.g. 5

    Returns:
    None
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    from upload_overleaf.upload import upload

    # Read colors from vColor.txt
    with open("vColor.txt", "r") as f:
        colors = f.read().splitlines()

    # Format dictTitle
    if dictTitle == "CNNTransformer":
        formattedTitle = "CNN + Transformer"

        signal = "CNN"
    elif dictTitle == "FFT":
        formattedTitle = "FFT + FFN"
        signal = "FFT"
    elif dictTitle == "OU":
        formattedTitle = "Ornstein-Uhlenbeck + FFN"
        signal = "OU"


    rets = dict[dictTitle]["returns_test"]
    turnover = dict[dictTitle]["turnover_test"]
    short = dict[dictTitle]["short_proportion_test"]
    dates = daily_dates[-len(rets) :]

    """
    Cumulative Returns
    """

    plt.figure(figsize=(12, 8))

    rets_ax = sns.lineplot(x=dates, y=(1 + rets).cumprod(), color=colors[0])

    rets_low = (1 + rets).cumprod().min()
    rets_high = (1 + rets).cumprod().max()

    rets_ax.set_ylim(rets_low, rets_high)

    rets_ax.grid(True, linestyle="--", alpha=0.7)

    rets_ax.set_title(
        f"Cumulative Returns for {sFactorModel} {iFactors} with {formattedTitle}"
    )
    rets_ax.set_xlabel("Date")
    rets_ax.set_ylabel("Cumulative Returns")
    rets_ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))

    upload(
        plt,
        "Master's Thesis",
        f"figures/{signal}/{sFactorModel.upper()}/{dictTitle}_{sFactorModel}_{iFactors}_returns.png",

    )

    """
    Turnover
    """

    plt.figure(figsize=(12, 8))

    turnover_ax = sns.lineplot(x=dates, y=turnover, color=colors[1])

    turnover_low = turnover.min()
    turnover_high = turnover.max()

    turnover_ax.set_ylim(turnover_low, turnover_high)

    turnover_ax.grid(True, linestyle="--", alpha=0.7)

    turnover_ax.set_title(
        f"Turnover for {sFactorModel} {iFactors} with {formattedTitle}"
    )
    turnover_ax.set_xlabel("Date")
    turnover_ax.set_ylabel("Turnover")
    turnover_ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))

    upload(
        plt,
        "Master's Thesis",
        f"figures/{signal}/{sFactorModel.upper()}/{dictTitle}_{sFactorModel}_{iFactors}_turnover.png",
    )

    """
    Short Proportion
    """

    plt.figure(figsize=(12, 8))

    short_ax = sns.lineplot(x=dates, y=short, color=colors[2])

    short_low = short.min()
    short_high = short.max()

    short_ax.set_ylim(short_low, short_high)

    short_ax.grid(True, linestyle="--", alpha=0.7)

    short_ax.set_title(
        f"Short Proportion for {sFactorModel} {iFactors} with {formattedTitle}"
    )

    short_ax.set_xlabel("Date")
    short_ax.set_ylabel("Short Proportion")
    short_ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))

    upload(
        plt,
        "Master's Thesis",
        f"figures/{signal}/{sFactorModel.upper()}/{dictTitle}_{sFactorModel}_{iFactors}_short.png",
    )

    return


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
if os.path.exists("factor_data/daily_data_processed.npz"):
    print("Daily returns already processed; skipping")
else:
    import factor_models as fm

    print("Preprocessing daily returns")
    fm.preprocessDailyReturns()


pathDailyData = "factor_data/daily_data_processed.npz"
dailyData = np.load(pathDailyData, allow_pickle=True)
daily_dates = pd.to_datetime(dailyData["date"], format="%Y%m%d")


# Set output path and cwd
outdir = os.path.join(os.getcwd(), "Outputs")
cwd = os.getcwd()

# Initialize torch
torch.set_default_dtype(torch.float)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = False

# Check if cuda is available
device = "cuda" if torch.cuda.is_available() else "cpu"

ou = True
fft = True
cnn = False

lFactorModels = ["pca", "ipca"]
# lFactorModels = ["ff", "pca", "ipca"]
iFactors = [0, 1, 3, 5, 8, 10, 15]
# iFactors = [0, 1, 3, 5, 8, 10, 15]

pbar1 = tqdm(lFactorModels)

for model in pbar1:

    pbar1.set_description(desc=f"Processing {model}")

    if model == "ff":
        iFactors = range(len(iFactors))
    else:
        iFactors = [0, 1, 3, 5, 8, 10, 15]

    pbar2 = tqdm(iFactors, desc=f"{model}", position=0, leave=False)

    for i in pbar2:
        print("\n")
        print(f"Running models for {model} with {i} factors")
        pbar2.set_description_str(f"Factor: {model} {i}")

        """
        OU
        """
        if ou:

            with open("configs/OU.yaml", "r") as file:
                ou_config = yaml.safe_load(file)

            ou_config["device"] = device

            ou_model_name = OU_FFN
            ou_preprocess = preprocess_ou

            ou_factors = [ou_config[f"{model}_{i}_res_path"]]
            ou_weights = ou_config[f"{model}_{i}_residual_weights"]

            run_model(
                ou_factors,
                ou_model_name,
                ou_preprocess,
                ou_config,
                cwd,
                daily_dates,
                ou_weights,
                i,
                model,
            )

            with open(f"results/OU_{model}_{i}_results.pkl", "rb") as f:
                ou_results = pickle.load(f)

            PlotSeries(ou_results, "OU", model.upper(), i)

        """
        FFT
        """

        if fft:
            with open("configs/FFT.yaml", "r") as file:
                fft_config = yaml.safe_load(file)

            fft_config["device"] = device

            fft_model_name = FFT_FFN
            fft_preprocess = preprocess_fourier

            fft_factors = [fft_config[f"{model}_{i}_res_path"]]
            fft_weights = fft_config[f"{model}_{i}_residual_weights"]

            run_model(
                fft_factors,
                fft_model_name,
                fft_preprocess,
                fft_config,
                cwd,
                daily_dates,
                fft_weights,
                i,
                model,
            )

            with open(f"results/FFT_{model}_{i}_results.pkl", "rb") as f:
                fft_results = pickle.load(f)

            PlotSeries(fft_results, "FFT", model.upper(), i)

        """
        CNNTransformer
        """
        if cnn:
            with open("configs/CNNTransformer.yaml", "r") as file:
                cnn_config = yaml.safe_load(file)
            cnn_model_name = CNNTransformer_FFN
            cnn_preprocess = preprocess_cnn

            cnn_config["device"] = device

            # Load factors - ToDo: Files for our residual data should be in the list:
            cnn_factors = [cnn_config[f"{model}_{i}_res_path"]]
            cnn_weights = cnn_config[f"{model}_{i}_residual_weights"]

            # model_name, preprocess, config, cwd, daily_dates, weights, iFactors
            run_model(
                cnn_factors,
                cnn_model_name,
                cnn_preprocess,
                cnn_config,
                cwd,
                daily_dates,
                cnn_weights,
                i,
                model,
            )

            with open(f"results/CNNTransformer_{model}_{i}_results.pkl", "rb") as f:
                cnn_results = pickle.load(f)

            PlotSeries(cnn_results, "CNNTransformer", model.upper(), i)

# print(f'CNNTransformer: \n Sharpe: {results_CNN['CNNTransformer']['sharpe_test'] * np.sqrt(252) :.3f} \n '
#       f'Return: {results_CNN['CNNTransformer']['ret_test'] * 252 * 100 :.3f}% \n '
#       f'Std: {results_CNN['CNNTransformer']['std_test'] * np.sqrt(252) :.3f} \n '
#       f'FFT: \n Sharpe: {results_FFT['FFT']['sharpe_test'] * np.sqrt(252) :.3f} \n '
#       f'Return: {results_FFT['FFT']['ret_test'] * 252 * 100 :.3f}% \n '
#       f'Std: {results_FFT['FFT']['std_test'] * np.sqrt(252) :.3f} \n '
#       f'OU: \n Sharpe: {results_OU['OU']['sharpe_test'] * np.sqrt(252) :.3f} \n '
#       f'Return: {results_OU['OU']['ret_test'] * 252 * 100 :.3f}% \n '
#       f'Std: {results_OU['OU']['std_test'] * np.sqrt(252) :.3f} \n ')

