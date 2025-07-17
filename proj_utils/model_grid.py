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
from typing import Sequence, Any, Tuple, Self
from dataclasses import dataclass
import pandas as pd
from itertools import product
from joblib import Parallel, delayed

@dataclass
class FitRecord:
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

class ModelGrid:

    def __init__(self,
                 series: pd.Series,
                 d: int = 1,
                 p_grid: Sequence[int] = None,
                 q_grid: Sequence[int] = None,
                 lb_lags: Sequence[int] = None,
                 alpha: float = 0.05,
                 seasonal_period: int = None,
                 n_jobs: int = 1,
                 store_model: bool = True
                 ):

        # Call the helper method to validate inputs
        self._validate_inputs(series,
                              d,
                              p_grid,
                              q_grid,
                              lb_lags,
                              alpha,
                              seasonal_period,
                              n_jobs, store_model)

        # Assign attributes
        self.series = series
        self.d = d
        self.p_grid = p_grid if p_grid is not None else (0,)
        self.q_grid = q_grid if q_grid is not None else (0,)
        self.lb_lags = lb_lags if lb_lags is not None else (6, 12, 24)
        self.alpha = alpha
        self.seasonal_period = seasonal_period
        self.n_jobs = n_jobs
        self.store_model = store_model
        self.results_ = None
        self.records_ = []
        self.models_ = {}
        self.best_result_ = None
        self.best_order_ = None
        self.differenced_ = None
        self.fitted_ = False

    def _validate_inputs(self,
                         series: pd.Series,
                         d: int,
                         p_grid: Sequence[int],
                         q_grid: Sequence[int],
                         lb_lags: Sequence[int],
                         alpha: float,
                         seasonal_period: int,
                         n_jobs: int,
                         store_models: bool):

        # Validate series
        if not isinstance(series, pd.Series):
            raise TypeError("`series` must be a pandas Series.")
        if series.empty:
            raise ValueError("`series` cannot be empty.")

        # Validate d
        if not isinstance(d, int) or d < 0:
            raise ValueError("`d` must be a non-negative integer.")

        # Validate p_grid
        if p_grid is not None:
            if not all(isinstance(p, int) and p >= 0 for p in p_grid):
                raise ValueError("`p_grid` must be a sequence of non-negative integers.")

        # Validate q_grid
        if q_grid is not None:
            if not all(isinstance(q, int) and q >= 0 for q in q_grid):
                raise ValueError("`q_grid` must be a sequence of non-negative integers.")

        # Validate lb_lags
        if lb_lags is not None:
            if not all(isinstance(lag, int) and lag > 0 for lag in lb_lags):
                raise ValueError("`lb_lags` must be a sequence of positive integers.")

        # Validate alpha
        if not isinstance(alpha, float) or not (0 < alpha < 1):
            raise ValueError("`alpha` must be a float between 0 and 1.")

        # Validate seasonal_period
        if seasonal_period is not None:
            if not isinstance(seasonal_period, int) or seasonal_period <= 0:
                raise ValueError("`seasonal_period` must be a positive integer.")

        # Validate n_jobs
        if not isinstance(n_jobs, int) or n_jobs < 1:
            raise ValueError("`n_jobs` must be an integer greater than or equal to 1.")

        # Validate store_models
        if not isinstance(store_models, bool):
            raise TypeError("`store_models` must be a boolean.")

    # Perform the following tasks:
    # 0. if refit then refit otherwise, check for fitted_  if true, return self.
    # 1. Difference the series if d > 0.
    # 2. Generate a grid of (p, d, q) combinations.
    # 3. Fit models for each combination in parallel if n_jobs > 1.
    # 4. within each job, fill up a FitRecord object with the results
    # 4. and store them in self.records_, all records including convergence.
    # 5. After fitting, take the best model based on BIC (tie break AIC) and save its order
    # 5. in self.best_order_.
    # 6. if store_model is true, store the best model in self.best_result_.
    # 7. create a pd.DataFrame from self.records_ and assign it to self.results_.
    # 8. subtract the minimal bic/aic values from the respective cols and add them 
    # 8. as delta_bic and delta_aic to the results_ DataFrame.
    # 9. Return self.
    def _grid_search(self, refit: bool = False) -> Self:
        if refit or not self.fitted_:
            # Step 1: Difference the series if d > 0
            if self.d > 0:
                self.differenced_ = self.series.diff(self.d).dropna()

            # Step 2: Generate a grid of (p, q) combinations
            param_grid = list(product(self.p_grid, self.q_grid))

            # Step 3: Fit models for each combination in parallel
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_model)(order) for order in param_grid
            )

            # Step 4: Store results in records_
            self.records_ = [FitRecord(*result) for result in results]

            # Step 5: Identify the best model based on BIC
            df_results = pd.DataFrame([record.__dict__ for record in self.records_])
            df_results['delta_bic'] = df_results['bic'] - df_results['bic'].min()
            df_results['delta_aic'] = df_results['aic'] - df_results['aic'].min()

            self.results_ = df_results
            self.best_result_ = df_results.loc[df_results['bic'].idxmin()]
            self.best_order_ = (self.best_result_.p, self.best_result_.d, self.best_result_.q)

            if self.store_model:
                self.models_[self.best_order_] = self.get_model(
                    order=self.best_order_,
                    seasonal_order=self.seasonal_period
                )

            self.fitted_ = True

        return self

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
    def _fit_model(self, order: Tuple[int, int]) -> Tuple[int, int, int, float, float, float, float, dict[int, float], bool, Tuple[int, int, int]]:
        """
        Fits a single model for the given (p, q) order and returns the results.

        Args:
            order (Tuple[int, int]): A tuple representing the (p, q) order.

        Returns:
            Tuple: A tuple containing the following:
                - p (int): The AR order.
                - d (int): The differencing order.
                - q (int): The MA order.
                - loglike (float): The log-likelihood of the fitted model.
                - aic (float): The Akaike Information Criterion.
                - bic (float): The Bayesian Information Criterion.
                - sigma2 (float): The variance of the residuals.
                - lb_p_vals (dict[int, float]): Ljung-Box test p-values for residuals.
                - converged (bool): Whether the model fitting converged.
                - model_key (Tuple[int, int, int]): The (p, d, q) order of the model.
        """
        p, q = order
        d = self.d

        try:
            # Fit the model (e.g., ARIMA or SARIMA)
            model = self.get_model(order=(p, d, q), seasonal_order=self.seasonal_period)
            fitted_model = model.fit()

            # Extract statistics
            loglike = fitted_model.llf
            aic = fitted_model.aic
            bic = fitted_model.bic
            sigma2 = fitted_model.sigma2
            lb_p_vals = {lag: pval for lag, pval in enumerate(fitted_model.test_serial_correlation(method='ljungbox'))}

            converged = True  # Assume convergence if no exception is raised
        except Exception as e:
            # Handle fitting errors
            loglike = float('-inf')
            aic = float('inf')
            bic = float('inf')
            sigma2 = float('inf')
            lb_p_vals = {}
            converged = False

        # Return the results as a tuple
        return p, d, q, loglike, aic, bic, sigma2, lb_p_vals, converged, (p, d, q)
