"""
@version:0.1
@description:并行计算加速alpha计算
@auther:xiaoyi huang
@time: 2020-10-22
"""
from src.alphas.alpha101Base import Alpha101Base
import multitasking as _multitasking
import time as _time
import pandas as _pd


class Alpha101(object):
    def __init__(self, df):
        self.apbase = Alpha101Base(df)
        self.result = {}

    def calculate(self, alphas='ALL', threaded=True, groupby='alpha'):
        """
        Calculate alphas
        :param alphas: str or list, default download all
        :param threaded: use multithreded feature, default be True
        :param groupby: str if == stock, returned df be group by stock
        :return: dict, saved in result
        """
        # check param validity
        if isinstance(alphas, str):
            # find num of attributes in the instance
            num_of_alpha = sum(['alpha_' in i for i in dir(self.apbase)])
            print(num_of_alpha)
            # return list of alpha number
            alphas = list(range(1, num_of_alpha + 1))
        elif not isinstance(alphas, list):
            raise TypeError("alphas input should be 'ALL' or list of int")

        # reset result
        self.result = {}

        # calculate using threads
        if threaded:
            # set the threads
            threads = min(len(alphas), _multitasking.cpu_count() * 2)
            # set maximum threads
            _multitasking.set_max_threads(threads=threads)
            for n in alphas:
                self._calculate_one_threaded(n)
            while len(self.result) < len(alphas):
                _time.sleep(0.01)
        else:
            for n in alphas:
                data = self._calculate_one(n)
                self.result['alpha_' + str(n)] = data

        # concatenating results
        data = _pd.concat(self.result.values(), axis=1, keys=self.result.keys())

        if groupby == 'stock':
            data.columns = data.columns.swaplevel(0,1)
            # data.sort_index(level=0, axis=1, inplace=True)

        return data

    def _calculate_one(self, n):
        """
        Warpper function to calculate one factor
        :param n: int which factor to be calculated
        :return: dataframe, evaluated function at apbase
        """
        method = 'alpha_' + str(n)
        f = getattr(self.apbase, method)
        return f()

    @_multitasking.task
    def _calculate_one_threaded(self, n):
        data = self._calculate_one(n)
        self.result['alpha_' + str(n)] = data
