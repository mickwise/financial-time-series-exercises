import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots as tp
from numpy.typing import ArrayLike

def plot_time_acf_pacf(date: pd.DatetimeIndex, series: ArrayLike):
    _, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4), constrained_layout=True)

    # Plot time
    axes[0].set_ylabel('Unemployment rate')
    axes[0].plot(date, series)

    # Calculate and plot sample ACF
    axes[1].set_xlabel('Lag')
    tp.plot_acf(series, ax=axes[1], lags=36)

    # Calculate and plot sample PACF
    axes[2].set_xlabel('Lag')
    tp.plot_pacf(series, ax=axes[2], lags=36)
