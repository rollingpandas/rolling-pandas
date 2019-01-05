"""
Time series funtions which apply a function over a rolling window
over time (i.e., not cross sectionally)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

def sum(obj, window, min_periods=None):
    return obj.rolling(window=window, min_periods=min_periods).sum()

    
def mean(obj, window, min_periods=None):
    return obj.rolling(window=window, min_periods=min_periods).mean()


def prod(obj, window, min_periods=None):
    return obj.rolling(window=window, min_periods=min_periods).apply(np.prod, raw=True)


def max(obj, window, min_periods=None):
    return obj.rolling(window=window, min_periods=min_periods).max()


def min(obj, window, min_periods=None):
    return obj.rolling(window=window, min_periods=min_periods).min()


def median(obj, window, min_periods=None, interpolation='linear'):
    return quantile(obj, 0.50, window, min_periods=min_periods, interpolation=interpolation)


def std(obj, window, min_periods=None, ddof=1):
    return obj.rolling(window=window, min_periods=min_periods).std(ddof=ddof)


def var(obj, window, min_periods=None, ddof=1):
    return obj.rolling(window=window, min_periods=min_periods).var(ddof=ddof)


def skew(obj, window, min_periods=None):
    return obj.rolling(window=window, min_periods=min_periods).skew()


def kurt(obj, window, min_periods=None):
    return obj.rolling(window=window, min_periods=min_periods).kurt()


def quantile(obj, quantile, window, min_periods=None, interpolation='linear'):
    """
    Find the value which corresponds to a given quantile, based on a rolling window
    """
    return obj.rolling(window=window, min_periods=min_periods).quantile(quantile, interpolation=interpolation)


def quantile_inv(obj, window, min_periods=None):
    """
    Find the quantile which corresponds to a given value, based on a rolling window
    """
    return obj.rolling(window=window, min_periods=min_periods).apply(lambda x: stats.percentileofscore(x, x[-1]))


def cummin(obj, skipna=True):
    return obj.cummin(skipna=skipna)


def cummax(obj, skipna=True):
    return obj.cummax(skipna=skipna)


def cumprod(obj, skipna=True):
    return obj.cumprod(skipna=skipna)


def cumsum(obj, skipna=True):
    return obj.cumsum(skipna=skipna)


"""
Rolling functions which take exactly 2 time series
"""

def corr(obj1, obj2, window, min_periods=None):
    assert isinstance(obj1, pd.core.series.Series) or (isinstance(obj1, pd.core.frame.DataFrame) and obj1.shape[1]==1), 'Expected a single column in obj1'
    assert isinstance(obj2, pd.core.series.Series) or (isinstance(obj2, pd.core.frame.DataFrame) and obj2.shape[1]==1), 'Expected a single column in obj2'

    return obj1.rolling(window=window, min_periods=min_periods).corr(obj2, pairwise=True)


def cov(obj1, obj2, window, min_periods=None):
    assert isinstance(obj1, pd.core.series.Series) or (isinstance(obj1, pd.core.frame.DataFrame) and obj1.shape[1]==1), 'Expected a single column in obj1'
    assert isinstance(obj2, pd.core.series.Series) or (isinstance(obj2, pd.core.frame.DataFrame) and obj2.shape[1]==1), 'Expected a single column in obj2'

    return obj1.rolling(window=window, min_periods=min_periods).cov(obj2, pairwise=True)

"""
Rolling functions which take >=2 time series
"""
