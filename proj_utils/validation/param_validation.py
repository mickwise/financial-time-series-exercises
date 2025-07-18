"""
This module contains utility functions for validating parameters.
"""
from typing import Optional, Literal, Iterable, Union, Any
import pandas as pd
from numpy import dtype

def validate_int(val: int, positive: bool = False) -> None:
    """
    Validate that `val` is an integer and positive or non-negative.

    Parameters:
    ----------
    val : int
        The value to validate.
    positive : bool, optional
        If True, `val` must be a positive integer. If False, `val` must be a non-negative integer.
        Default is False.
    Raises:
    ------
    ValueError
        If `val` is not an integer or does not meet the positive/non-negative condition.
    """
    if positive:
        if not isinstance(val, int) or val <= 0:
            raise ValueError("Value must be a positive integer.")
    else:
        if not isinstance(val, int) or val < 0:
            raise ValueError("Value must be a non-negative integer.")

def validate_series(series: pd.Series,
                    missing: Optional[Literal['drop', 'raise']] = 'drop',
                    data_type: Optional[Union[dtype, str]] = None) -> pd.Series:
    """
    Checks if the Series is a valid pandas Series, not empty, and handles 
    missing values based on the `missing` parameter. If data_type is provided,
    it checks if the Series has the specified data type.

    Parameters:
    ----------
    series : pd.Series
        The Series to validate.
    missing : Literal['drop', 'raise'], optional
        How to handle missing values in the Series. If 'drop', NaN values are dropped.
        If 'raise', an error is raised if NaN values are present. Default is 'drop'.
    data_type : Optional[dtype], optional
        The expected data type of the Series. If provided, the Series must match this type.
    Returns:
    -------
    pd.Series
        The validated Series.
    Raises:
    ------
    TypeError
        If `series` is not a pandas Series or if the 
        series is not of the requested dtype.
    ValueError
        If `series` is empty or contains NaN values and `missing` is set to 'raise'.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("`series` must be a pandas Series.")
    if series.empty:
        raise ValueError("`series` cannot be empty.")
    if data_type is not None:
        if series.dtype != data_type:
            raise TypeError(f"`series` must be of type {data_type}, but got {series.dtype}.")
    if series.isnull().any():
        match missing:
            case 'drop':
                series.dropna(inplace=True)
            case 'raise':
                raise ValueError("`series` contains NaN values. Set `missing='drop'` " \
                    "to drop them.")
    return series

def validate_grid(grid: Optional[Iterable[int]], positive: bool = False) -> None:
    """
    Validate that `grid` is a iterable of integers, either positive or non-negative.

    Parameters:
    ----------
    grid : Optional[Iterable[int]]
        The grid to validate. If None, no validation is performed.
    positive : bool, optional
        If True, `grid` must contain only positive integers. If False, it must contain
        non-negative integers. Default is False.
    Raises:
    ------
    ValueError
        If `grid` is not None and does not meet the integer or positive/non-negative condition.
    """
    if grid is not None:
        if positive:
            if not all(isinstance(val, int) and val > 0 for val in grid):
                raise ValueError("'grid' must contain only positive integers.")
        else:
            if not all(isinstance(val, int) and val >= 0 for val in grid):
                raise ValueError("'grid' must contain only non-negative integers.")

def validate_alpha(alpha: float) -> None:
    """
    Validate that `alpha` is a float between 0 and 1.

    Parameters:
    ----------
    alpha : float
        The value to validate.
    Raises:
    ------
    ValueError
        If `alpha` is not a float or is not in the range (0, 1).
    """
    if not isinstance(alpha, float) or not 0 < alpha < 1:
        raise ValueError("`alpha` must be a float between 0 and 1.")

def validate_bool(exp: bool) -> None:
    """
    Validate that `exp` is a boolean value.

    Parameters:
    ----------
    exp : bool
        The value to validate.
    Raises:
    ------
    TypeError
        If `exp` is not a boolean value.
    """
    if not isinstance(exp, bool):
        raise TypeError(f"`{exp}` must be a boolean value.")

def validate_literal(lit: Any, valid_values: Iterable[Any]) -> None:
    """
    Validate that `lit` is one of the valid values.

    Parameters:
    ----------
    lit : Literal
        The value to validate.
    valid_values : Iterable[Literal]
        An iterable of valid values.
    Raises:
    ------
    ValueError
        If `lit` is not one of the valid values.
    """
    if lit not in valid_values:
        raise ValueError(f"`{lit}` must be one of {valid_values}.")