"""
@version:0.1
@description:main class for alpha 101 features from world quant, suitable for yfinance lib
https://arxiv.org/ftp/arxiv/papers/1601/1601.00991.pdf
some formula are from regression or industrialized not implemented: 36, 48
@auther:xiaoyi huang
@time: 2020-10-22
"""
import src.alphas.utils as _utils
import numpy as np
import pandas as pd


class Alpha101Base:
    def __init__(self, df):
        """:param data a raw df directly download from yfinance"""

        # close represents adj close
        self.close = df['Adj Close']
        self.volume = df['Volume']
        self.open = df['Open']
        self.low = df['Low']
        self.high = df['High']
        self.returns = _utils.get_return(self.close)
        self.vwap = _utils.get_vwap(df)

    def alpha_1(self):
        """
        Wrapper formula for alpha1, (rank(Ts_ArgMax(SignedPower(((returns < 0)
                ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
        :return: time series alpha feature
        """
        inner = self.close
        # replace all the date where return<0 to std
        # 这里会出warning,不过没什么大事
        inner[self.returns < 0] = _utils.stddev(self.returns, window=20)
        # calculate which day max the time series
        ts_argmax = _utils.ts_argmax(inner ** 2, 5)
        return _utils.rank(ts_argmax) - 0.5

    def alpha_2(self):
        """
        formula for alpha2: (-1*correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
        :return: time series df of all stocks
        """
        x1 = _utils.rank(
            _utils.delta(
                np.log(self.volume), period=2)
        )
        x2 = _utils.rank(
            (self.close - self.open) / self.open
        )
        return -1 * _utils.correlation(x1, x2, window=6)

    def alpha_3(self):
        """
        data size should increse to see the difference
        formula for alpha3: (-1*correlation(rank(open), rank(volume), 10))
        :return:time series df of all stocks
        """
        df = -1 * _utils.correlation(
            _utils.rank(self.open),
            _utils.rank(self.volume),
            window=10
        )
        return df.replace([-np.inf, np.inf], 0)

    def alpha_4(self):
        """
        formula for alpha4: (-1*Ts_Rank(rank(low), 9))
        :return time series rank of a window rank
        """
        df = -1 * _utils.ts_rank(
            _utils.rank(self.low), 9
        )
        return df.replace([-np.inf, np.inf], 0)

    def alpha_5(self):
        """
        formula for alpha5: (rank((open - (sum(vwap, 10) / 10)))*(-1*abs(rank((close - vwap)))))
        :return time series rank of a window rank
        """
        return (_utils.rank(
            self.open - (_utils.ts_sum(self.vwap, window=10) / 10)
        ) * -1 * abs(_utils.rank(self.close - self.vwap))
                )

    def alpha_6(self):
        """
        formula for alpha6: (-1*correlation(open, volume, 10))
        """
        df = -1 * _utils.correlation(self.open, self.volume, window=10)
        return df.replace([-np.inf, np.inf], 0)

    def alpha_7(self):
        """
        formula for alpha7: ((adv20 < volume) ? ((-1*ts_rank(abs(delta(close, 7)), 60))
        *sign(delta(close, 7))) : (-1*1))
        careful, the self.close should be len>60, so that nan won't generated
        """
        adv20 = _utils.ts_ma(self.volume, 20)
        alpha = -1 * _utils.ts_rank(abs(_utils.delta(self.close, 7)), 60) * np.sign(_utils.delta(self.close, 7))
        alpha[adv20 >= self.volume] = -1
        return alpha

    def alpha_8(self):
        """
        formula: (-1*rank(((sum(open, 5)*sum(returns, 5)) - delay((sum(open, 5)*sum(returns, 5)), 10))))
        """
        df = -1 * _utils.rank(
            _utils.ts_sum(self.open, window=5) * _utils.ts_sum(self.returns, window=5)
            -
            _utils.delay(
                _utils.ts_sum(self.open, window=5) * _utils.ts_sum(self.returns, window=5),
                period=10
            )
        )
        return df.fillna(method='backfill')

    def alpha_9(self):
        """
        formula: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0)
         ? delta(close, 1) : (-1*delta(close, 1))))
        """
        delta_close = _utils.delta(self.close, period=1)
        condition_1 = _utils.ts_min(delta_close, window=5) > 0
        condition_2 = _utils.ts_max(delta_close, window=5) < 0
        alpha = -1 * delta_close
        alpha[condition_1 | condition_2] = delta_close
        return alpha

    def alpha_10(self):
        """
        formula: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ?
         delta(close, 1) : (-1*delta(close, 1)))))
        """
        delta_close = _utils.delta(self.close, period=1)
        condition_1 = _utils.ts_min(delta_close, window=4) > 0
        condition_2 = _utils.ts_max(delta_close, window=4) < 0
        alpha = -1 * delta_close
        alpha[condition_1 | condition_2] = delta_close
        return _utils.rank(alpha)

    def alpha_11(self):
        """
        formula: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3)))*rank(delta(volume, 3)))
        :return:df backfilled
        """
        df = (
                     _utils.rank(_utils.ts_max(self.vwap - self.close, window=3))
                     +
                     _utils.rank(_utils.ts_min(self.vwap - self.close, window=3))
             ) * _utils.delta(self.volume, period=3)
        return df.fillna(method='backfill')

    def alpha_12(self):
        """
        formula: (sign(delta(volume, 1))*(-1 * delta(close, 1)))
        """
        return np.sign(_utils.delta(self.volume, period=1) * -1 * _utils.delta(self.close, period=1))

    def alpha_13(self):
        """
        formula: (-1 * rank(covariance(rank(close), rank(volume), 5)))
        :return: dataframe backfilled
        """
        df = -1 * _utils.rank(
            _utils.covariance(_utils.rank(self.close), _utils.rank(self.volume), window=5)
        )
        return df.fillna(method='backfill')

    def alpha_14(self):
        """
        formula:((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
        :return: dataframe need a lot stock to prevent inf
        """
        df = _utils.correlation(self.open, self.volume, window=10)
        df = df.replace([-np.inf, np.inf], 0).fillna(method='backfill')
        return -1 * _utils.rank(_utils.delta(self.returns, period=3)) * df

    def alpha_15(self):
        """
        formula: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
        """
        df = _utils.correlation(_utils.rank(self.high), _utils.rank(self.volume), 3)
        df = df.replace([-np.inf, np.inf], 0).fillna(method='backfill')
        return (-1 * _utils.ts_sum(_utils.rank(df), 3)).fillna(method='ffill')

    def alpha_16(self):
        """
        formula:(-1 * rank(covariance(rank(high), rank(volume), 5)))
        :return: dataframe rank of cov rank
        """
        return -1 * _utils.rank(
            _utils.covariance(_utils.rank(self.high), _utils.rank(self.volume), 5)
        ).fillna(method='backfill')

    def alpha_17(self):
        """
        formula: (((-1 * rank(ts_rank(close, 10))) *
        rank(delta(delta(close, 1), 1)))
        *rank(ts_rank((volume / adv20), 5)))
        :return:
        """
        adv20 = _utils.ts_ma(self.volume, 20)
        return -1 * (_utils.rank(_utils.ts_rank(self.close, 10)) *
                     _utils.rank(_utils.delta(_utils.delta(self.close, period=1), period=1)) *
                     _utils.rank(_utils.ts_rank(self.volume / adv20, window=5)))

    def alpha_18(self):
        """
        formula:  (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open,10))))
        :return
        """
        df = _utils.correlation(self.close, self.open, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * _utils.rank(
            _utils.stddev(abs(self.close - self.open), window=5) +
            self.close - self.open
        ) + df

    def alpha_19(self):
        """
        formula: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns,250)))))
        be careful with the time range, less than 250 will be nan
        """
        return ((-1 * np.sign((self.close - _utils.delay(self.close, 7)) + _utils.delta(self.close, 7))) *
                (1 + _utils.rank(1 + _utils.ts_sum(self.returns, 250))))

    def alpha_20(self):
        """
        formula: (((-1 * rank((open - delay(high, 1)))) *
        rank((open - delay(close, 1)))) *
        rank((open -delay(low, 1))))
        """
        return -1 * (_utils.rank(self.open - _utils.delay(self.high, 1)) *
                     _utils.rank(self.open - _utils.delay(self.close, 1)) *
                     _utils.rank(self.open - _utils.delay(self.low, 1))).fillna(method='backfill')

    def alpha_21(self):
        """
        formula: ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2))
        ? (-1* 1) : (((sum(close, 2) / 2) <
        ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) or ((volume / adv20) == 1)) ?
         1 : (-1* 1))))
        need to reconsider
        """
        condition_1 = _utils.ts_ma(self.close, 8) + _utils.stddev(self.close, 8) < _utils.ts_ma(self.close, 2)
        condition_2 = _utils.ts_ma(self.volume, 20) / self.volume < 1
        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index, columns=self.close.columns)
        alpha[condition_1 | condition_2] = -1
        return alpha

    def alpha_22(self):
        """
        formula: (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
        """
        df = _utils.correlation(self.high, self.volume, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * _utils.delta(df, 5) * _utils.rank(_utils.stddev(self.close, 20))

    def alpha_23(self):
        """
        formula: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
        :return zeros dataframe, hat function
        """
        condition = _utils.ts_ma(self.high, 20) < self.high
        alpha = pd.DataFrame(np.zeros_like(self.close), index=self.close.index, columns=self.close.columns)
        alpha[condition] = -1 * _utils.delta(self.high, 2).fillna(value=0)
        return alpha

    def alpha_24(self):
        """
        formula: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) or
        ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ?
        (-1* (close - ts_min(close, 100))) : (-1* delta(close, 3)))
        """
        condition = _utils.delta(_utils.ts_ma(self.close, 100), 100) / _utils.delay(self.close, 100) <= 0.05
        alpha = -1 * _utils.delta(self.close, 3)
        alpha[condition] = -1 * (self.close - _utils.ts_min(self.close, 100))
        return alpha

    def alpha_25(self):
        """
        formula: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
        """
        return -1 * _utils.ts_max(_utils.correlation(
            _utils.ts_rank(self.volume, window=5), _utils.ts_rank(self.high, window=5), window=5
        ), window=3).fillna(method='backfill')

    def alpha_26(self):
        """
        formula: (-1* ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
        """
        df = _utils.correlation(
            _utils.ts_rank(self.volume, window=5), _utils.ts_rank(self.high, window=5), window=3
        )
        df = df.replace([-np.inf, np.inf], 0)
        df = -1 * _utils.ts_max(df.fillna(method='backfill'), window=3)
        return df.fillna(0.5)

    def alpha_27(self):
        """
        formula: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1* 1) : 1)
        :return replaced nan to -1
        """
        alpha = _utils.rank((_utils.ts_ma(_utils.correlation(
            _utils.rank(self.volume), _utils.rank(self.vwap), 6), 2)))
        alpha[alpha > 0.5] = -1
        alpha[alpha <= 0.5] = 1
        return alpha.fillna(-1)

    def alpha_28(self):
        """
        formula: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
        :return: scaled df
        """
        adv20 = _utils.ts_ma(self.volume, 20)
        df = _utils.correlation(adv20, self.low, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return _utils.scale(((df + ((self.high + self.low) / 2)) - self.close))

    def alpha_29(self):
        """
        formula: (min(product(rank(rank(scale(log(sum(ts_min(rank(
        rank((-1*rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) +
         ts_rank(delay((-1* returns), 6), 5))
        """
        df = _utils.ts_min(_utils.rank(_utils.rank(_utils.scale(np.log(
            _utils.ts_sum(_utils.rank(_utils.rank(-1 * _utils.rank(_utils.delta(
                self.close - 1, 5)))), 2))))), 5) + _utils.ts_rank(_utils.delay(-1 * self.returns, 6), 5)
        return df.fillna(method='backfill')

    def alpha_30(self):
        """
        formula: (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) +
         sign((delay(close, 2) - delay(close, 3))))))* sum(volume, 5)) / sum(volume, 20))
        """
        delta_close = _utils.delta(self.close, 1)  # (close - delay(close, 1)
        inner = np.sign(delta_close) + np.sign(_utils.delay(delta_close, 1)) + np.sign(_utils.delay(delta_close, 2))
        df = (1.0 - _utils.rank(inner)) * _utils.ts_sum(self.volume, 5) / _utils.ts_sum(self.volume, 20)
        return df.fillna(method='backfill')

    def alpha_31(self):
        """
        formula: ((rank(rank(rank(decay_linear((-1* rank(rank(delta(close, 10)))), 10)))) +
         rank((-1* delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
        :return:
        """
        adv20 = _utils.ts_ma(self.volume, 20)
        df = _utils.correlation(adv20, self.low, 12).replace([-np.inf, np.inf], 0).fillna(value=0)
        p1 = _utils.rank(_utils.rank(_utils.rank(
            _utils.decay_linear_pn((-1 * _utils.rank(_utils.rank(_utils.delta(self.close, 10)))), 10))))
        p2 = _utils.rank((-1 * _utils.delta(self.close, 3)))
        p3 = np.sign(_utils.scale(df))
        return p1 + p2 + p3

    def alpha_32(self):
        """
        formula: (scale(((sum(close, 7) / 7) - close)) + (20* scale(correlation(vwap, delay(close, 5), 230))))
        long time series
        :return:
        """
        return _utils.scale(((_utils.ts_ma(self.close, 7) / 7) - self.close)) + (
                20 * _utils.scale(_utils.correlation(self.vwap, _utils.delay(self.close, 5), 230)))

    def alpha_33(self):
        """
        formula: rank((-1* ((1 - (open / close))^1)))
        :return:
        """
        return _utils.rank(-1 + (self.open / self.close))

    def alpha_34(self):
        """
        formula: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
        :return:
        """
        inner = _utils.stddev(self.returns, 2) / _utils.stddev(self.returns, 5)
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return _utils.rank(2 - _utils.rank(inner) - _utils.rank(_utils.delta(self.close, 1)))

    def alpha_35(self):
        """
        formula: ((Ts_Rank(volume, 32)* (1 - Ts_Rank(((close + high) - low), 16)))* (1 - Ts_Rank(returns, 32)))
        :return: df, need long range to modify nan
        """
        return ((_utils.ts_rank(self.volume, 32) *
                 (1 - _utils.ts_rank(self.close + self.high - self.low, 16))) *
                (1 - _utils.ts_rank(self.returns, 32))).fillna(method='backfill')

    def alpha_36(self):
        """
        formula: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
        :return:dataframe, need long range
        """
        return (_utils.rank(_utils.correlation(
            _utils.delay(self.open - self.close, 1), self.close, 200)
        ) + _utils.rank(self.open - self.close)).fillna(method='backfill')

    def alpha_37(self):
        """
        formula:  ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
        :return:
        """
        inner = self.close / self.open
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return -1 * _utils.rank(_utils.ts_rank(self.open, 10)) * _utils.rank(inner)

    def alpha_38(self):
        """
        formula:  ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear_pn((volume / adv20), 9)))))) *
        (1 +rank(sum(returns, 250))))
        :return:
        """
        adv20 = _utils.ts_ma(self.volume, 20)
        return ((-1 * _utils.rank(
            _utils.delta(self.close, 7) * (1 - _utils.rank(
                _utils.decay_linear_pn((self.volume / adv20), 9))))) *
                (1 + _utils.rank(_utils.ts_ma(self.returns, 250))))

    def alpha_39(self):
        """
        formula: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
        :return:
        """
        return -1 * _utils.rank(_utils.stddev(self.high, 10)) * _utils.correlation(self.high, self.volume, 10)

    def alpha_40(self):
        """
        formula: (((high * low)^0.5) - vwap)
        :return:
        """
        return pow((self.high * self.low), 0.5) - self.vwap

    def alpha_41(self):
        """
        formula: (rank((vwap - close)) / rank((vwap + close)))
        :return:
        """
        return _utils.rank((self.vwap - self.close)) / _utils.rank((self.vwap + self.close))

    def alpha_42(self):
        """
        formula: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
        :return:
        """
        adv20 = _utils.ts_ma(self.volume, 20)
        return _utils.ts_rank(self.volume / adv20, 20) * _utils.ts_rank((-1 * _utils.delta(self.close, 7)), 8)

    def alpha_43(self):
        """
        formula: (-1 * correlation(high, rank(volume), 5))
        :return:
        """
        df = _utils.correlation(self.high, _utils.rank(self.volume), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * df

    def alpha_44(self):
        """
        formula: (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2))
        *rank(correlation(sum(close, 5), sum(close, 20), 2))))
        :return: dataframe, need more stock and long range
        """
        df = _utils.correlation(self.close, self.volume, 2)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        df = -1 * (_utils.rank(_utils.ts_ma(_utils.delay(self.close, 5), 20)) * df *
                     _utils.rank(_utils.correlation(_utils.ts_sum(self.close, 5), _utils.ts_sum(self.close, 20), 2)))
        return df.fillna(method='backfill')

    def alpha_45(self):
        """
        formula: (0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?
        (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ?
        1 :((-1 * 1) * (close - delay(close, 1)))))
        :return dataframe， inner has lot of nan
        """
        inner = ((_utils.delay(self.close, 20) - _utils.delay(self.close, 10)) / 10) - \
                ((_utils.delay(self.close, 10) - self.close) / 10)
        alpha = (-1 * _utils.delta(self.close))
        alpha[inner < 0] = 1
        alpha[inner > 0.25] = -1
        return alpha

    def alpha_46(self):
        """
        formula: ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close)))
        / (sum(high, 5) /5))) - rank((vwap - delay(vwap, 5))))
        :return:
        """
        adv20 = _utils.ts_ma(self.volume, 20)
        df = ((((_utils.rank((1 / self.close)) * self.volume) / adv20) * (
                (self.high * _utils.rank((self.high - self.close))) / (_utils.ts_ma(self.high, 5) / 5))) - _utils.rank(
            (self.vwap - _utils.delay(self.vwap, 5))))
        return df.fillna(method='backfill')

    def alpha_47(self):
        """
        formula: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) <
         (-1 *0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        :return:
        """
        inner = (((_utils.delay(self.close, 20) - _utils.delay(self.close, 10)) / 10) -
                 ((_utils.delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * _utils.delta(self.close))
        alpha[inner < -0.1] = 1
        return alpha

    def alpha_48(self):
        """
        formula: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
        :return:
        """
        df = (-1 * _utils.ts_max(_utils.rank(
            _utils.correlation(_utils.rank(self.volume), _utils.rank(self.vwap), 5)), 5)
                )
        return df.fillna(method='bfill')

    def alpha_49(self):
        """
        formula: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.05)) ?
         1 : ((-1 * 1) * (close - delay(close, 1))))
        :return:
        """
        inner = (((_utils.delay(self.close, 20) - _utils.delay(self.close, 10)) / 10) -
                 ((_utils.delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * _utils.delta(self.close))
        alpha[inner < -0.05] = 1
        return alpha

    def alpha_50(self):
        """
        formula: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) -sum(returns, 20))
        / 220))) * ts_rank(volume, 5))
        :return: long range data needed
        """
        return (((-1 * _utils.delta(_utils.ts_min(self.low, 5), 5)) *
                 _utils.rank(
                     ((_utils.ts_sum(self.returns, 240) -
                       _utils.ts_sum(self.returns, 20)) / 220))) * _utils.ts_rank(self.volume, 5))