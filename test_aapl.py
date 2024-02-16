# Hide tensorflow warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from upload_overleaf.upload import upload

# Load the data
lf = pl.scan_parquet("daily.parquet")


def plotAdjvsNonAdj(ticker: str, lf: pl.LazyFrame):
    """
    Plots time series analysis of stock with logarithmic and adjusted logarithmic closing prices.

    Parameters:
    - ticker (str): Ticker symbol of the stock to plot.
    - lf (pl.LazyFrame): Polars LazyFrame containing the stock price data.

    Returns:
    None
    """
    # Get upper for ticker
    ticker = ticker.upper()
    df = lf.filter(pl.col("ticker") == ticker).collect().to_pandas()
    # Set up the figure and axes
    plt.figure(figsize=(12, 8))

    # Plotting the first line
    ax = sns.lineplot(x="date", y="log_close", data=df, color="blue", label="Log Close")

    # Creating a twin axis for the second line
    ax2 = ax.twinx()

    # Plotting the second line on the twin axis
    ax2 = sns.lineplot(
        x="date", y="adj_log_close", data=df, color="orange", label="Adj Log Close"
    )

    # Get lowest value of log_close
    low = min([df["log_close"].min(), df["adj_log_close"].min()])
    # If low is inf, -inf or nan, set to second lowest value
    if low in [float("inf"), float("-inf"), float("nan")]:
        low = df["log_close"].sort_values().unique()[1]
        # If low is inf, -inf or nan, set to 2
        if low in [float("inf"), float("-inf"), float("nan")]:
            low = 4

    # Get highest value of log_close
    high = max([df["log_close"].max(), df["adj_log_close"].max()])
    # If high is inf, -inf or nan, set to 10
    if high in [float("inf"), float("-inf"), float("nan")]:
        high = 10

    # Adjusting y-axis limits
    ax.set_ylim(low, high)
    ax2.set_ylim(low, high)

    # Adding grid lines
    ax.grid(True, linestyle="--", alpha=0.7)
    ax2.grid(False)

    # Adding a single legend for both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Combine labels from both axes
    combined_labels = labels + labels2
    ax.legend(lines + lines2, combined_labels, loc="upper left")
    ax2.get_legend().remove()

    # Adding labels and title
    ax.set_title(f"Logarithmic and Adjusted Logarithmic Closing Prices for {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Logarithmic Closing Price")
    ax2.set_ylabel("Adjusted Logarithmic Closing Price")

    # Formatting date labels
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
    # Upload to Overleaf
    upload(plt, "Master's Thesis", f"figures/{ticker.lower()}_close.png")


# Get input from user
ticker = input("Enter ticker symbol: ").upper()

# Plot the time series analysis
plotAdjvsNonAdj(ticker, lf)

print(f"Figure for {ticker} has been uploaded to Overleaf.")
