"""
arima_spec.py

Defines ArimaSpec, a dataclass for bundling up all of the
optional parameters to pass into statsmodels ARIMA/SARIMAX
constructor in one neat object.
"""

from typing import Optional, Union, Iterable, Literal
from dataclasses import dataclass
import pandas as pd
from proj_utils.validation import param_validation as pv


@dataclass
class ArimaSpec:
    """
    Class to hold ARIMA/SARIMAX model specifications.

    Attributes:
    ----------
        exog(Optional[pd.Series]): Exogenous regression variables.
        trend(Optional[Union[Literal[str], Iterable]]): Trend polynomial terms.
        enforce_stationarity(bool): Whether to enforce stationarity.
        enforce_invertibility(bool): Whether to enforce invertibility.
        concentrate_scale(bool): If True, concentrate the scale parameter out of the likelihood.
        trend_offset(int): Offset for the trend polynomial.
        date(Optional[pd.Series]): Date index for the time series.
        missing(Literal['drop', 'raise']): How to handle missing values in the series.
    Methods:
    -------
        _to_kwargs():
            Convert non None fields into ARIMA(**kwargs).
    """

    exog: Optional[pd.Series] = None
    trend: Optional[Union[Literal['n', 'c', 't', 'ct'], Iterable]] = None
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True
    concentrate_scale: bool = False
    trend_offset: int = 1
    date: Optional[pd.Series] = None
    missing: Literal['drop', 'raise'] = 'drop'

    def __post_init__(self):

        # Validate missing
        pv.validate_literal(self.missing, ['drop', 'raise'])

        # Validate exog
        pv.validate_series(self.exog, missing=self.missing)

        # Validate trend
        if self.trend is not None:
            if isinstance(self.trend, str):
                if self.trend not in ['n', 'c', 't', 'ct']:
                    raise ValueError("`trend` must be one of ['n', 'c', 't', 'ct'].")
            elif isinstance(self.trend, Iterable):
                if not all(item in (0, 1) for item in self.trend):
                    raise ValueError("If `trend` is an iterable, all items must be 0 or 1.")

        # Validate enforce_stationarity
        pv.validate_bool(self.enforce_stationarity)

        # Validate enforce_invertibility
        pv.validate_bool(self.enforce_invertibility)

        # Validate trend offset
        pv.validate_int(self.trend_offset, positive=True)

        # Validate date
        pv.validate_series(self.date, missing=self.missing, data_type='datetime64[ns]')

    def _to_kwargs(self) -> dict:
        raw = dict(self.__dict__)
        return {k: v for k, v in raw.items() if v is not None}
