# Import packages
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings
import time
from torch.optim import Adam
from FFT_FFN import *
from pre_process import *
from matplotlib import pyplot as plt
import yfinance as yf


"""
Estimate function
"""


def estimate(Data, model, preprocess, residual_weights=None, log_dev_progress_freq=50,
             num_epochs=100, lr=0.001, batchsize=150, early_stopping=False, save_params=True,
             device="cpu", output_path=os.path.join(os.getcwd(), "results", "Unknown"), model_tag="Unknown",
             lookback=30, length_training=1000, test_size=125, parallelize=True, device_ids=[0, 1, 2, 3, 4, 5, 6, 7],
             trans_cost=0, hold_cost=0, force_retrain=True, objective="sharpe", estimate_start_idx=0):

    # Assets to trade
    assets_to_trade = np.count_nonzero(Data, axis=0) >= lookback
    Data = Data[:, assets_to_trade]
    T, N = Data.shape
    returns = np.zeros(length_training)
    turnovers = np.zeros(length_training)
    short_proportions = np.zeros(length_training)
    all_weights = np.zeros((length_training, len(assets_to_trade)))

    # Load weights
    if residual_weights is not None and "FamaFrenchNew" in model_tag:
        assets_to_trade = np.load('')  # Insert residuals from fama_french universe
        Data = Data[:, assets_to_trade]
        all_weights = np.zeros((T - length_training, len(assets_to_trade)))

    if residual_weights is not None and "Fama" in model_tag and "New" not in model_tag:
        Ndifference = residual_weights.shape[2] - np.sum(assets_to_trade)
        if Ndifference > 0:
            all_weights = np.zeroes((length_training, len(assets_to_trade) + Ndifference))
            assets_to_trade = np.append(assets_to_trade, np.ones(Ndifference, dtype=np.bool))

    if residual_weights is not None and ("IPCA" in model_tag or "Deep" in model_tag):
        assets_to_trade = np.load('')  # Load residuals from IPCA or "Deep" universe?
        all_weights = np.zeros((length_training, len(assets_to_trade)))

    # "Logging"
    print(f"Estimating: {estimate_start_idx}:{min(estimate_start_idx + length_training, T)} for {model_tag}")
    print(f"Testing: {estimate_start_idx + length_training - lookback}:"
          f"{min(estimate_start_idx + length_training + test_size, T)}")

    # Creating the training and dev data
    df_train = Data[estimate_start_idx:min(estimate_start_idx + length_training, T)]
    df_dev = Data[
             estimate_start_idx + length_training - lookback:min(estimate_start_idx + length_training + test_size, T)]
    residual_weights_train = (None if residual_weights is None
                              else residual_weights[estimate_start_idx:min(estimate_start_idx + length_training, T)])
    residual_weights_dev = (None if residual_weights is None
                            else residual_weights[(estimate_start_idx + length_training - lookback):
                                                  min(estimate_start_idx + length_training + test_size, T)])

    # Cleaning up
    del residual_weights
    del Data

    model_tag = (model_tag + f"__estimation{estimate_start_idx}-{length_training}-{test_size}")

    # Train the model
    model1 = model
    returns, turnovers, short_proportions, all_weights, a2t = train(model1, preprocess=preprocess, df_train=df_train,
                                                                    df_dev=df_dev,
                                                                    residual_weights_train=residual_weights_train,
                                                                    residual_weights_dev=residual_weights_dev,
                                                                    log_dev_progress_freq=log_dev_progress_freq,
                                                                    num_epochs=num_epochs, lr=lr,
                                                                    force_retrain=force_retrain,
                                                                    early_stopping=early_stopping,
                                                                    save_params=save_params,
                                                                    output_path=output_path, model_tag=model_tag,
                                                                    device=device, lookback=lookback,
                                                                    parallelize=parallelize, device_ids=device_ids,
                                                                    batchsize=batchsize, trans_cost=trans_cost,
                                                                    hold_cost=hold_cost, objective=objective)

    if device is None:
        device = model.device.type
    if "cpu" not in device:
        with torch.cuda.device(device):  # For Parallelization
            torch.cuda.empty_cache()  # Clear memory

    print("Estimation complete!")

    np.save(os.path.join(output_path, "WeightsComplete_" + model_tag + ".npy"), all_weights)

    full_ret = np.mean(returns)
    full_std = np.std(returns)
    full_sharpe = full_ret / full_std
    print(f"Sharpe: {full_sharpe * np.sqrt(252):0.2f}, ret: {full_ret * 252:0.4f}, std: {full_std * np.sqrt(252):0.4f}"
          f"turnover: {np.mean(turnovers):0.3f}, short proportion: {np.mean(short_proportions):0.4f}")

    return returns, full_sharpe, full_ret, full_std, turnovers, short_proportions


"""
Train function - Used in Estimate function
"""


def train(model, preprocess, df_train, df_dev=None, log_dev_progress=True, log_dev_progress_freq=50, num_epochs=100,
          lr=0.001, batchsize=200, optimizer_opts={"lr": 0.001}, early_stopping=False, early_stopping_max_trials=5,
          lr_decay=0.5, residual_weights_train=None, residual_weights_dev=None, save_params=True, output_path=None,
          model_tag="", lookback=30, trans_cost=0, hold_cost=0, parallelize=True, device=None,
          device_ids=[0, 1, 2, 3, 4, 5, 6, 7], force_retrain=True, objective='sharpe'):
    # Preprocess data
    # assets_to_trade chooses assets which have at least `lookback` non-missing observations in the training period
    # this does not induce lookahead bias because idxs_selected is backward-looking and
    # will only select assets with at least `lookback` non-missing obs
    assets_to_trade = np.count_nonzero(df_train, axis=0) >= lookback
    df_train = df_train[:, assets_to_trade]

    # Get output path
    if output_path is None:
        output_path = os.path.join(os.getcwd(), "Outputs")
    if device is None:
        device = model.device

    residual_weights_train = None
    if residual_weights_train is not None:
        residual_weights_train = residual_weights_train[:, assets_to_trade]

    T, N = df_train.shape
    windows, idxs_selected = preprocess(df_train, lookback)

    # Start to train
    if parallelize:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
    model.train()  # Sets the mode of the model to training
    optimizer = Adam(model.parameters(), **optimizer_opts)

    # Initialize variables
    min_dev_loss = np.inf
    patience = 0
    trial = 0

    already_trained = False

    # Check if we already trained the model
    checkpoint_fname = (
            f"Checkpoint - {model.module.random_seed if parallelize else model.random_seed}_seed_" + model_tag + ".tar")

    if os.path.isfile(os.path.join(output_path, checkpoint_fname)) and not force_retrain:
        already_trained = True
        checkpoint = torch.load(os.path.join(output_path, checkpoint_fname))
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.train()
        print("Model already trained!")

    start_time = time.time()
    for epoch in range(num_epochs):
        rets_full = np.zeros(T - lookback)
        short_proportion = np.zeros(T - lookback)
        turnover = np.zeros(T - lookback)

        # Break input data into batches of size 'batchsize' and train over them, for computational efficiency
        for i in range(int((T - lookback) / batchsize) + 1):
            weights = torch.zeros((min(batchsize * (i + 1), T - lookback) - batchsize * i, N))  # "device" dropped

            # "Logging"
            print(f"Epoch {epoch} batch {i} weights shape: {weights.shape}")

            idxs_batch_i = idxs_selected[(batchsize * i):min(batchsize * (i + 1), T - lookback), :]
            input_data_batch_i = windows[(batchsize * i):min(batchsize * (i + 1), T - lookback)][idxs_batch_i]

            weights[idxs_batch_i] = model(torch.tensor(input_data_batch_i))  # "device" dropped

            if residual_weights_train is None:
                abs_sum = torch.sum(torch.abs(weights), dim=1, keepdim=True)
            else:  # residual_weights_train is TxN1xN2 (multiplied by returns on the right gives residuals)
                assert (weights.shape ==
                        residual_weights_train[(lookback + batchsize * i):(min(lookback + batchsize * (i + 1), T)), :,
                        0].shape)

                T1, N1 = weights.shape
                weights2 = torch.bmm(weights.reshape(T1, 1, N1),
                                     torch.tensor(residual_weights_train[
                                                  (lookback + batchsize * i):min(lookback + batchsize * (i + 1),
                                                                                 T)])).squeeze()  # will be T1xN2

                # "Logging"
                print(f"Epoch {epoch} batch {i} weights2 shape: {weights2.shape}")

                # noinspection PyArgumentList
                abs_sum = torch.sum(torch.abs(weights2), axis=1, keepdim=True)
                # Normalize weights by the sum of their absolute values
                try:
                    weights2 = weights2 / abs_sum
                except:
                    weights2 = weights2 / (abs_sum + 1e-8)

            try:
                weights = weights / abs_sum
            except:
                weights = weights / (abs_sum + 1e-8)

            # noinspection PyArgumentList
            rets_train = torch.sum(
                weights * torch.tensor(df_train[(lookback + batchsize * i):min(lookback + batchsize * (i + 1), T), :]),
                axis=1)

            if residual_weights_train is None:
                # noinspection PyArgumentList
                rets_train = (rets_train - trans_cost * torch.cat(
                    (torch.zeros(1), torch.sum(torch.abs(weights[1:] - weights[:-1]), axis=1))) - hold_cost * torch.sum(
                    torch.abs(torch.min(weights, torch.zeros(1))), axis=1))
            else:
                # noinspection PyArgumentList ,PyUnboundLocalVariable
                rets_train = (rets_train - trans_cost * torch.cat(
                    (torch.zeros(1),
                     torch.sum(torch.abs(weights2[1:] - weights2[:-1]), axis=1))) - hold_cost * torch.sum(
                    torch.abs(torch.min(weights2, torch.zeros(1))), axis=1))

            mean_ret = torch.mean(rets_train)
            std = torch.std(rets_train)
            if objective == "sharpe":
                if std.abs() < 1e-8:
                    loss = torch.zeros_like(mean_ret)
                else:
                    loss = -mean_ret / std
            elif objective == 'meanvar':
                loss = -mean_ret * 252 + std * 15.9  # Hvor kommer 15.9 fra? 252 er antal handelsdage på et år
            elif objective == 'sqrtMeanSharpe':
                loss = -torch.sign(mean_ret) * np.sqrt(np.abs(mean_ret)) / std
            else:
                raise Exception(
                    f'Invalid objective loss {objective}. Needs to be either sharpe, meanvar or sqrtMeanSharpe')

            if not already_trained and (
                    (parallelize and model.module.is_trainable) or (not parallelize and model.is_trainable)):
                optimizer.zero_grad()  # Reset gradient
                loss.backward()  # Calculate new gradients
                optimizer.step()  # Update model param given new gradients

            if residual_weights_train is None:
                weights = weights.detach().numpy()
            else:
                weights = weights2.detach().numpy()

            rets_full[(batchsize * i):min(batchsize * (i + 1), T - lookback)] = (rets_train.detach().numpy())
            turnover[(batchsize * i):(min(batchsize * (i + 1), T - lookback)) - 1] = np.sum(
                np.abs(weights[1:] - weights[:-1]), axis=1)
            # We simply things with this next line, but I'm not sure if why
            turnover[min(batchsize * (i + 1), T - lookback) - 1] = turnover[min(batchsize * (i + 1), T - lookback) - 2]
            short_proportion[(batchsize * i):min(batchsize * (i + 1), T - lookback)] = np.sum(
                np.abs(np.minimum(weights, 0)), axis=1)

            if log_dev_progress and epoch % log_dev_progress_freq == 0:
                dev_loss_description = ""
                if df_dev is not None:
                    (rets_dev, dev_loss, dev_sharpe, dev_turnovers, dev_short_proportions, weights_dev,
                     a2t) = get_returns(
                        model, preprocess=preprocess, objective=objective, df_test=df_dev, lookback=lookback,
                        trans_cost=trans_cost, hold_cost=hold_cost, residual_weights=residual_weights_dev)
                    model.train()
                    dev_mean_ret = np.mean(rets_dev)
                    dev_std = np.std(rets_dev)
                    dev_turnover = np.mean(dev_turnovers)
                    dev_short_proportion = np.mean(dev_short_proportions)
                    dev_loss_description = (
                        f", dev loss {-dev_loss:0.2f}, dev Sharpe: {-dev_sharpe * np.sqrt(252):0.2f}, "
                        f" ret: {dev_mean_ret * 252:0.4f}, std: {dev_std * np.sqrt(252) :0.4f}, "
                        f"turnover: {dev_turnover:0.3f}, short proportion: {dev_short_proportion:0.3f}\n"
                    )

                full_ret = np.mean(rets_full)
                full_std = np.std(rets_full)
                full_sharpe = full_ret / full_std
                full_turnover = np.mean(turnover)
                full_short_proportion = np.mean(short_proportion)

                print(
                    f"Epoch: {epoch}/{num_epochs}, "
                    f"train Sharpe: {full_sharpe * np.sqrt(252):0.2f}, "
                    f"ret: {full_ret * 252:0.4f}, "
                    f"std: {full_std * np.sqrt(252):0.4f}, "
                    f"turnover: {full_turnover:0.3f}, "
                    f"short proportion: {full_short_proportion:0.3f} \n"
                    "       "
                    f" time per epoch: {(time.time() - start_time) / (epoch + 1):0.2f}s"
                )

                if early_stopping:
                    model.random_seed = 69
                    if dev_loss < min_dev_loss:  # It will always start here, as we have min_dev_loss = np.inf
                        patience = 0
                        min_dev_loss = dev_loss
                        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),
                                      "optimizer_state_dict": optimizer.state_dict(), "loss": loss}
                        torch.save(checkpoint,
                                   os.path.join(output_path, f"Checkpoint - {model.random_seed}_seed_{model_tag}.tar"))

                    else:
                        patience += 1
                        if trial == early_stopping_max_trials:
                            print("early stopping max trials reached")
                            break
                        else:
                            trial += 1
                            print(f"Trial {trial} of {early_stopping_max_trials} - Reducing learning rate")
                            lr = optimizer.param_groups[0]["lr"] * lr_decay
                            checkpoint = torch.load(
                                os.path.join(output_path, f"Checkpoint - {model.random_seed}_seed_{model_tag}.tar"))
                            model.load_state_dict(checkpoint["model_state_dict"])
                            model = model.to(device)
                            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                            model.train()

                            for param_group in optimizer.param_groups:
                                param_group["lr"] = lr
                            patience = 0

        if already_trained:
            print("Model has already been trained")
            break

    if save_params and not already_trained:
        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(), "loss": loss}
        # Save the model, such that we know it has been trained
        checkpoint_fname = (
                f"Checkpoint - {model.module.random_seed if parallelize else model.random_seed}_seed_" + model_tag + ".tar")
        torch.save(checkpoint, os.path.join(output_path, checkpoint_fname))

    print(
        f"Training time: {(time.time() - start_time) / 60} minutes, model: {model_tag}, "
        f"seed: {model.module.random_seed if parallelize else model.random_seed}"
    )

    if df_dev is not None:
        return rets_dev, dev_turnovers, dev_short_proportions, weights_dev, a2t
    else:
        return rets_full, turnover, short_proportion, weights, assets_to_trade


"""
Returns function - Used in Train function
"""


def get_returns(model,
                preprocess,
                objective,
                df_test,
                lookback,
                trans_cost,
                hold_cost,
                residual_weights=None,
                load_params=False,
                paths_checkpoints=[None],
                parallelize=False,
                device_ids=[0, 1, 2, 3, 4, 5, 6, 7]):
    # Get device
    device = model.device
    if parallelize:
        model = nn.DataParallel(model, device_ids).to(device)

    # Restrict to assets which have at least 'lookback' non-missing observations in the training period
    assets_to_trade = np.count_nonzero(df_test, axis=0) >= lookback
    df_test = df_test[:, assets_to_trade]
    T, N = df_test.shape
    windows, idxs_selected = preprocess(df_test, lookback)

    rets_test = torch.zeros(T - lookback)
    model.eval()

    with torch.no_grad():
        weights = torch.zeros((T - lookback, N), device=device)

        for i in range(len(paths_checkpoints)):  # This ensembles if many checkpoints are given (whatever that means)
            if load_params:
                checkpoint = torch.load(paths_checkpoints[i], map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(device)

            weights[idxs_selected] += model(torch.tensor(windows[idxs_selected], device=device))
        weights /= len(paths_checkpoints)

        if residual_weights is None:
            # noinspection PyArgumentList
            abs_sum = torch.sum(torch.abs(weights), axis=1, keepdim=True)
        else:
            residual_weights = residual_weights[:, assets_to_trade]
            assert weights.shape == residual_weights[lookback:T, :, 0].shape
            T1, N1 = weights.shape
            weights2 = torch.bmm(weights.reshape(T1, 1, N1),
                                 torch.tensor(residual_weights[lookback:T], device=device)).squeeze()
            # noinspection PyArgumentList
            abs_sum = torch.sum(torch.abs(weights2), axis=1, keepdim=True)

            try:
                weights2 = weights2 / abs_sum
            except:
                weights2 = weights2 / (abs_sum + 1e-8)

        try:
            weights = weights / abs_sum
        except:
            weights = weights / (abs_sum + 1e-8)

        # noinspection PyArgumentList
        rets_test = torch.sum(weights * torch.tensor(df_test[lookback:T, :], device=device), axis=1)

        if residual_weights is not None:
            weights = weights2

        # noinspection PyArgumentList
        turnover = torch.cat((torch.zeros(1, device=device), torch.sum(torch.abs(weights[1:] - weights[:-1]), axis=1)))
        # noinspection PyArgumentList
        short_proportion = torch.sum(torch.abs(torch.min(weights, torch.zeros(1, device=device))), axis=1)
        rets_test = rets_test - trans_cost * turnover - hold_cost * short_proportion

        turnover[0] = torch.mean(turnover[1:])
        mean = torch.mean(rets_test)
        std = torch.std(rets_test)
        sharpe = -mean / (std + 1e-8)
        loss = None

        if objective == "sharpe":
            loss = sharpe
        elif objective == "meanvar":
            loss = -mean * 252 + std * 15.9
        elif objective == "sqrtMeanSharpe":
            loss = -torch.sign(mean) * torch.sqrt(torch.abs(mean)) / std
        else:
            raise Exception(f"Invalid objective loss {objective}")

    return (
        rets_test.numpy(),
        loss,
        sharpe,
        turnover.numpy(),
        short_proportion.numpy(),
        weights.numpy(),
        assets_to_trade,
    )  # If there is problems with the return, I might need to do .cpu() on the tensors


"""
Test function
"""


def test(Data, daily_dates, model, preprocess, residual_weights=None, log_dev_progress_freq=50, log_plot_freq=199,
         num_epochs=100, lr=0.001, batchsize=150, early_stopping=False, save_params=True, device='cpu',
         output_path=os.path.join(os.getcwd(), "results", "Unknown"), model_tag="Unknown", lookback=30,
         retrain_freq=250, length_training=1000, rolling_retrain=True, parallelize=True,
         device_ids=[0, 1, 2, 3, 4, 5, 6, 7], trans_cost=0, hold_cost=0, force_retrain=False, objective="sharpe"):
    # Choose assets to trade
    assets_to_trade = np.count_nonzero(Data, axis=0) >= lookback

    # "Logging"
    print(f"test(): assets_to_trade.shape {assets_to_trade.shape}")

    # Initialize
    Data = Data[:, assets_to_trade]
    T, N = Data.shape
    returns = np.zeros(T - length_training)
    turnovers = np.zeros(T - length_training)
    short_proportions = np.zeros(T - length_training)
    all_weights = np.zeros((T - length_training, len(assets_to_trade)))

    # Load assets_to_trade for the weights
    if residual_weights is not None and "FamaFrenchNew" in model_tag:
        assets_to_trade = np.load('')  # Insert residuals from fama_french universe
        Data = Data[:, assets_to_trade]
        all_weights = np.zeros((T - length_training, len(assets_to_trade)))

    if (residual_weights is not None and "FamaFrench" in model_tag and "New" not in model_tag):
        Ndifference = residual_weights.shape[2] - np.sum(assets_to_trade)
        if Ndifference > 0:
            all_weights = np.zeros((T - length_training, len(assets_to_trade) + Ndifference))
            assets_to_trade = np.append(assets_to_trade, np.ones(Ndifference, dtype=np.bool))
    if residual_weights is not None and ("IPCA" or "Deep" in model_tag):
        assets_to_trade = np.load("")  # Load residuals from IPCA or "Deep" universe?
        all_weights = np.zeros((T - length_training, len(assets_to_trade)))

        # Run train/test
    for t in range(int((T - length_training) / retrain_freq) + 1):
        print(f"At subperiod: {t} / {int((T - length_training) / retrain_freq) + 1}")
        data_train_t = Data[(t * retrain_freq):(length_training + t * retrain_freq)]
        data_test_t = Data[
                      (length_training + t * retrain_freq - lookback):min(length_training + (t + 1) * retrain_freq, T)]
        if residual_weights is not None:
            residual_weights_train_t = residual_weights[(t * retrain_freq):(length_training + t * retrain_freq)]
            residual_weights_test_t = residual_weights[(length_training + t * retrain_freq - lookback):min(
                length_training + (t + 1) * retrain_freq, T)]
        else:
            residual_weights_test_t = None
            residual_weights_train_t = None

        model_tag_t = model_tag + f"__subperiod{t}"

        if rolling_retrain or t == 0:
            model_t = model
            rets_t, turns_t, shorts_t, weights_t, a2t = train(model=model_t, preprocess=preprocess,
                                                               df_train=data_train_t,
                                                               df_dev=data_test_t,  # No validation is done
                                                               residual_weights_train=residual_weights_train_t,
                                                               residual_weights_dev=residual_weights_test_t,
                                                               log_dev_progress_freq=log_dev_progress_freq,
                                                               num_epochs=num_epochs, force_retrain=force_retrain,
                                                               optimizer_opts={"lr": 0.001},
                                                               early_stopping=early_stopping,
                                                               save_params=save_params, output_path=output_path,
                                                               model_tag=model_tag_t, device=device, lookback=lookback,
                                                               parallelize=parallelize, device_ids=device_ids,
                                                               batchsize=batchsize, trans_cost=trans_cost,
                                                               hold_cost=hold_cost, objective=objective)

            print("Train completed")

        else:
            rets_t, _, _, turns_t, shorts_t, weights_t, a2t = get_returns(model=model_t, preprocess=preprocess,
                                                                          objective=objective, df_test=data_test_t,
                                                                          residual_weights=residual_weights_test_t,
                                                                          device=device, lookback=lookback,
                                                                          trans_cost=trans_cost, hold_cost=hold_cost)

            print("get_returns() completed")

        returns[(t * retrain_freq):min((t + 1) * retrain_freq, T - length_training)] = rets_t
        turnovers[(t * retrain_freq):min((t + 1) * retrain_freq, T - length_training)] = turns_t
        short_proportions[(t * retrain_freq):min((t + 1) * retrain_freq, T - length_training)] = shorts_t

        if residual_weights is None:
            w = np.zeros((min((t + 1) * retrain_freq, T - length_training) - t * retrain_freq, len(a2t)))
            w[:, a2t] = weights_t

            print(f"Returned weights.shape {weights_t.shape}")
        else:
            w = weights_t

        # "Logging"
        print(f"Weights selected shape "
              f"{all_weights[(t * retrain_freq):min((t + 1) * retrain_freq, T - length_training), assets_to_trade].shape}")
        print(f"sum(assets_to_trade) {np.sum(assets_to_trade)}")

        all_weights[(t * retrain_freq):min((t + 1) * retrain_freq, T - length_training), assets_to_trade] = w

        if device is None:
            device = model.device.type
        if "cpu" not in device:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    print("Train/Test completed")

    cumRets = np.cumprod(1 + returns)

    # Plot cumulative returns
    plt.figure()
    plt.plot_date(daily_dates[-len(cumRets):], cumRets, marker="None", linestyle="-")
    plt.savefig(os.path.join(output_path, model_tag + "_cumulative-returns.png"))

    # Plot turnover
    plt.figure()
    plt.plot_date(daily_dates[-len(cumRets):], turnovers, marker="None", linestyle="-")
    plt.savefig(os.path.join(output_path, model_tag + "_turnover.png"))

    # Plot short positions
    plt.figure()
    plt.plot_date(daily_dates[-len(cumRets):], short_proportions, marker="None", linestyle="-")
    plt.savefig(os.path.join(output_path, model_tag + "_short-proportions.png"))

    np.save(os.path.join(output_path, "WeightsComplete_" + model_tag + ".npy"), all_weights)

    full_ret = np.mean(returns)
    full_std = np.std(returns)
    full_sharpe = full_ret / full_std
    print(f"Sharpe: {full_sharpe * np.sqrt(252):0.2f}, ret: {full_ret * 252:0.4f}, std: {full_std * np.sqrt(252):0.4f}"
          f"turnover: {np.mean(turnovers):0.4f}, short proportion: {np.mean(short_proportions):0.4f}")

    return returns, full_sharpe, full_ret, full_std, turnovers, short_proportions


"""
Get daily-data
"""


def get_daily_data():
    """
    Returns daily excess returns, adjusted close, and volume
    """

    # Get tickers
    tickers = pd.read_parquet('daily.parquet')['ticker'].unique()
    risk_free = get_risk_free_rate()

    # Get data
    data = []
    for tick in tqdm(tickers, desc='Downloading data', miniters=10):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            df = yf.download(tick, start="1998-12-31", end="2024-01-01", progress=False)
        df['ticker'] = tick
        df['return'] = np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(1)) - risk_free
        data.append(df[['ticker', 'Adj Close', 'Volume', 'return']])

    data = pd.concat(data)
    data = data[~(data['ticker'].isna())]
    data = data.pivot(columns='ticker')

    # Save data
    data.to_parquet('daily_data.parquet')

    return data


"""
Get risk-free rate
"""


def de_annualize(annual_rate, periods=365):
    return (1 + annual_rate) ** (1 / periods) - 1


def get_risk_free_rate():
    # download 3-month us treasury bills rates
    annualized = yf.download("^IRX", start="1998-12-31", end="2024-01-01")["Adj Close"]

    # de-annualize
    daily = annualized.apply(de_annualize)

    # create dataframe
    return daily