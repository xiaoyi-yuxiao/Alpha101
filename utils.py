"""
@version:0.1
@description:basis function for alpha 101 from world quant, suitable for yfinance lib
https://arxiv.org/ftp/arxiv/papers/1601/1601.00991.pdf
@auther:xiaoyi huang
@time: 2020-10-22
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata


# define basis function
def get_vwap(df):
    """
    Wrapper function to compute vwap data from daily basis data from yfinance
    :param data: multi indexed df, raw data from yfinance download
    :return: a multi indexed df, with all companies vwap time series data
    """
    volume = df['Volume']
    h = df['High']
    l = df['Low']
    ac = df['Adj Close']

    vwap = ((h+l+ac)/3*volume).cumsum()/volume.cumsum()
    return vwap


def get_return(adjClose):
    """
    Wrapper function to compute returns for Adjclose
    :param AdjClose:multi indexed df contains all time series data
    :return:replaces null to 0.001
    """
    returns = adjClose.pct_change().fillna(0.001)
    return returns


def stddev(df, window=10):
    """
    Wrapper function to estimate rolling standard deviation.
    滑动窗口求标准差
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series std over the past 'window' days.
    """
    return df.rolling(window).std().fillna(method='backfill')


def ts_max(df, window=10):
    """
    Wrapper function to estimate rolling max.
    :param df: multi indexed df, raw data from yfinance download.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
            filled with backfill
    """
    return df.rolling(window).max().fillna(method='backfill')


def ts_argmax(df, window=10):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: multi indexed df, raw data from yfinance download.
    :param window: the rolling window.
    :return: a pandas DataFrame with which day ts_max(df, window) occurred on,
            filled with backfill
    """
    return (df.rolling(window).apply(np.argmax) + 1).fillna(method='backfill')


def rank(df):
    """
    Cross sectional rank
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with rank along columns.
    """
    return df.rank(axis=1, pct=True)


def delta(df, period=1):
    """
    Wrapper function to estimate today’s value of x minus the value of x d days ago.
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return df.diff(period).fillna(method='backfill')


def correlation(x1, x2, window=10):
    """
    Compute column wise rolling time series corr of 2 dfs
    :param: x1 dataframe x2 dataframe
    :return time series df of corr with x1 x2
    """
    return (x1.rolling(window).corr(x2)).fillna(method='backfill')


def covariance(x1, x2, window=10):
    """
    :param x1: dataframe
    :param x2: dataframe
    :param window: int window length
    :return: dataframe rolling cov
    """
    return x1.rolling(window).cov(x2).fillna(method='backfill')


def ts_rank(df, window=10):
    """
    Wrapper function to estimate rolling rank.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """
    def _rolling_rank(na):
        """
        Auxiliary function to be used in pd.rolling_apply
        :param na: numpy array.
        :return: The rank of the last value in the array.
        """
        return rankdata(na)[-1]
    return (df.rolling(window).apply(_rolling_rank)).fillna(method='backfill')


def ts_sum(df, window=10):
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """

    return df.rolling(window).sum().fillna(method='backfill')


def ts_ma(df, window=10):
    """
    Wrapper function to estimate SMA.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series moving average over the past 'window' days.
    """
    return df.rolling(window).mean().fillna(method='backfill')


def delay(df, period=1):
    """
    Wrapper function to compute lag
    :param df: dataframe
    :param period: int, lagged grade
    :return: dataframe with lagged time series
    """
    return df.shift(period)


def ts_min(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).min().fillna(method='backfill')


def ts_max(df, window=10):
    """
    Wrapper function to estimate rolling max.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.rolling(window).max().fillna(method='backfill')


def scale(df, k=1):
    """
    Scaling time serie.
    :param df: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """
    return df.mul(k).div(np.abs(df).sum())


def decay_linear_pn(df, period=10):
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    # Clean data
    print(np.shape(df))
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :]
    na_series = df.values

    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    # Estimate the actual lwma with the actual close.
    # The backtest engine should assure to be snooping bias free.
    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1: row + 1, :]
        na_lwma[row, :] = (np.dot(x.T, y))
    return pd.DataFrame(na_lwma, index=df.index, columns=df.columns)
