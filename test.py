import yfinance as yf
import src.alphas.alpha101Base as _alpha101Base
from src.alphas.alpha101 import Alpha101
import pandas as pd
from src.stocks.stockpool import StockPool


# df = yf.download(["MMM","AMD","ABT","ADBE",'A','AAPL','AEE','AEP','AXP','AMGN','APH','CCL'],"2020-05-05","2020-07-07")
# df = yf.download(["MMM","AAPL"],"2020-05-05","2020-06-07")
# AP = Alpha101(df)
# data = AP.calculate('ALL',threaded=False, groupby='stock')
# data.to_csv('result.csv')

pool = StockPool('dow')
data = pool.download('2019-10-12', '2020-01-01', n=5)
print(data)