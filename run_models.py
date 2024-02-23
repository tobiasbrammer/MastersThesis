# Import packages and functions
from functions import *


########################################################################################################################
# Run pre_processer
########################################################################################################################

windows, idxs_selected = preprocess_fourier(df, lookback=30)


########################################################################################################################
# Train model
########################################################################################################################

rets_full, turnover, short_proportion, weights, assets_to_trade = train(
    model=FFT_FFN(),
    df_train=df[:round(len(df) * 0.66)], parallelize=False, device_ids=[0, 1, 2, 3, 4, 5, 6, 7], lookback=30,
    optimizer_opts={"lr": 0.001}, num_epochs=100, batchsize=200, trans_cost=0.0005, hold_cost=0.0001,
    objective='Sharpe', log_dev_progress=True, log_dev_progress_freq=50, df_dev=pd.DataFrame(),
    preprocess=preprocess_fourier, residual_weights_dev=None, early_stopping=False, early_stopping_max_trials=5,
    lr_decay=0.5, output_path=None, model_tag="FFT", save_params=True, force_retrain=True, device=None
)
