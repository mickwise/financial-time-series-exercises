import pandas as pd
import numpy as np
from typing import Iterable
from scipy import stats

# Add percent columns to data frame
def add_percent(data: pd.DataFrame, cols: Iterable[str]):
    for col in cols:
        data[col + '_percent'] = (data[col]*100).round(3)

# Calculate the log returns and add them as new columns 
def add_log_returns(data: pd.DataFrame, returns:Iterable[str]):
    for col in returns:
        data['log_' + col] = np.log(data[col] + 1)

# Perform a t-test for the mean of the requested columns and print the p values
def t_test_for_mean(data: pd.DataFrame, cols:Iterable[str]):
    for col in cols:
        p_val = stats.ttest_1samp(data[col], popmean=0.0)[1]
        print(f'For {col}, the p_value is: {p_val}')