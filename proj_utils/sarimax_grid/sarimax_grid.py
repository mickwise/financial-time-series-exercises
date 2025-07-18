"""
model_grid.py

This module contains the `ModelGrid` class, which is designed to perform a grid search operation on a given time series. 
The class attempts to fit multiple models with varying parameters to the provided time series data and records their 
statistical metrics for comparison.

The `ModelGrid` class allows users to:
- Specify a range of parameters for ARIMA-like models (e.g., `p`, `q`, and `d` values).
- Perform model fitting across the parameter grid.
- Store and compare the results of each model.
- Identify the best-fitting model based on the recorded statistics.

Key Features:
- Input validation to ensure proper configuration of the grid search.
- Support for seasonal adjustments via the `seasonal_period` parameter.
- Parallel processing using the `n_jobs` parameter to speed up the grid search.
- Option to store fitted models for further analysis.

Typical usage example:
    model_grid = ModelGrid(series=my_time_series, p_grid=[0, 1, 2], q_grid=[0, 1], d=1)
    model_grid.fit()
    best_model = model_grid.best_result_

"""
from typing import Any, Tuple, Self, Iterable, Optional
from itertools import product
from dataclasses import dataclass
from arima_spec import ArimaSpec
from proj_utils.validation import param_validation as pv
import pandas as pd
from joblib import Parallel, delayed

@dataclass
class FitRecord:
    """
    A data class to hold the results of fitting an ARIMA model.
    Attributes:
        p(int): The AR order.
        q(int): The MA order.
        d(int): The differencing order.
        loglike(float): The log-likelihood of the fitted model.
        aic(float): The Akaike Information Criterion of the fitted model.
        bic(float): The Bayesian Information Criterion of the fitted model.
        sigma2(float): The estimated variance of the residuals.
        lb_p_vals(dict[int, float]): The p-values from the Ljung-Box test for serial correlation.
        converged(bool): Whether the model fitting converged successfully.
        model_key(tuple): A tuple representing the model order (p, d, q).
    """
    p: int
    d: int
    q: int
    loglike: float
    aic: float
    bic: float
    sigma2: float
    lb_p_vals: dict[int, float]
    converged: bool
    model_key: tuple

class SarimaxGrid:

    def __init__(self,
                 series: pd.Series,
                 diff: int = 0,
                 p_grid: Optional[Iterable[int]] = None,
                 q_grid: Optional[Iterable[int]] = None,
                 seasonal_p_grid: Optional[Iterable[int]] = None,
                 seasonal_q_grid: Optional[Iterable[int]] = None,
                 seasonal_period: int = 0,
                 seasonal_diff: int = 0,
                 lb_lags: Optional[Iterable[int]] = None,
                 alpha: float = 0.05,
                 n_jobs: int = 1,
                 store_model: bool = True,
                 arima_spec: ArimaSpec = None
                 ):

        # ensure we have a spec object before validating
        self.arima_spec = arima_spec or ArimaSpec()
        self.series = series
        self.d = diff
        self.p_grid = p_grid or (0,)
        self.q_grid = q_grid or (0,)
        self.lb_lags = lb_lags or (6, 12, 24)
        self.alpha = alpha
        self.seasonal_p_grid = seasonal_p_grid or (0,)
        self.seasonal_q_grid = seasonal_q_grid or (0,)
        self.seasonal_period = seasonal_period
        self.seasonal_diff = seasonal_diff
        self.n_jobs = n_jobs
        self.store_model = store_model
        self.results_ = None
        self.records_ = []
        self.models_ = {}
        self.best_result_ = None
        self.best_order_ = None
        self.differenced_ = None
        self.fitted_ = False

        # Call the helper method to validate inputs
        self._validate_inputs()

    def fit(self, refit: bool = False) -> Self:
        pass

    def summary(self, sort: str = 'BIC') -> pd.DataFrame:
        pass

    def get_model(self, order: Tuple[int], seasonal_order: int = 0) -> Any:
        pass

    def residuals(self, order: Tuple[int]) -> pd.Series:
        pass

    def augmented_with_seasonal(self,
                                order: Tuple[int],
                                seasonal_type: str = 'MA',
                                seasonal_lag: int = 1):
        pass

    def save_path(self, path: str):
        pass

    def load_path(self, path: str):
        pass

    # Helper methods
    def _validate_inputs(self) -> None:
        missing = self.arima_spec.missing

        # Validate series
        self.series = pv.validate_series(self.series, missing=missing)

        # Validate d
        pv.validate_int(self.d)

        # Validate p_grid
        pv.validate_grid(self.p_grid)

        # Validate q_grid
        pv.validate_grid(self.q_grid)

        # Validate lb_lags
        pv.validate_grid(self.lb_lags, positive=True)

        # Validate alpha
        pv.validate_alpha(self.alpha)

        # Validate seasonal_p_grid
        pv.validate_grid(self.seasonal_p_grid)

        # Validate seasonal_q_grid
        pv.validate_grid(self.seasonal_q_grid)

        # Validate seasonal_period
        pv.validate_int(self.seasonal_period)

        # Validate seasonal_diff
        pv.validate_int(self.seasonal_diff)

        # Validate n_jobs
        pv.validate_int(self.n_jobs, positive=True)

        # Validate store_model
        pv.validate_bool(self.store_model)

    def _fit_model(self, order: Tuple[int, int]) -> FitRecord:
        pass
