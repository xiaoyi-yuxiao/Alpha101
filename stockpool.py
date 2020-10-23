"""
@version:0.1
@description: 股票池， 包含历史数据
@auther:xiaoyi huang
@time: 2020-10-22
"""
import yfinance as yf
import yahoo_fin.stock_info as si
import time as _time


class StockPool(object):
    def __init__(self, pool):
        """:param pool nasdaq, sp500, dow"""
        self.pool = pool

    def download(self, start_time, end_time, interval='1d', n='ALL'):
        """
        Wrapper function to get prices
        :param n: number of companies to get price
        :param start_time: rglr start
        :param end_time:rglr end
        :param interval:
        :return: dataframe, same as yfinance download
        """
        # get tickers for the pool
        try:
            f = getattr(si, 'tickers_'+self.pool)
        except AttributeError:
            print('Pool not found')

        # get list of tickers and preprocessing
        tickers = f()
        tickers = [ticker.replace('.','-') for ticker in tickers]

        # download all the ticker prices
        if not isinstance(n,str):
            data = yf.download(tickers=tickers[0:n], start=start_time, end=end_time, interval=interval)
        else:
            data = yf.download(tickers=tickers, start=start_time, end=end_time, interval=interval)

        return data




