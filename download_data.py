import pandas as pd
import importlib as implib
import source.helpers as hp
import numpy as np
import yfinance as yf
from datetime import datetime
import source.database as mysql
import time, sys

df_shares = pd.read_csv("./data/setorial_b3_15_01_2021.csv")
df_shares.columns = [hp.snake_case(txt) for txt in df_shares.columns]
df_shares['codigo_ord'] = df_shares.codigo + '3.SA'
df_shares['codigo_pref'] = df_shares.codigo + '4.SA'

tickers = np.concatenate((df_shares.codigo_ord, df_shares.codigo_pref), axis=0)

conn = mysql.create_connection("./data/db.sqlite")
mysql.create_fatch_log_table(conn)

for i in range(len(tickers)):
    try:
        time.sleep(2)
        print("Trying to get info of {}".format(tickers[i]))
        yf_ticker = yf.Ticker(tickers[i])

        hist_data = yf.download(tickers[i], period='max')

        if (len(hist_data) == 0):
            raise Exception("Information about {} not found".format(tickers[i]))
        hist_data.columns = [hp.snake_case(txt) for txt in hist_data.columns]
        hist_data.to_sql("hist_{}".format(tickers[i]), conn, if_exists='replace')

        info = yf_ticker.info
        data_info = {
            "ticker": tickers[i],
            "min_date": hist_data.index.min(),
            "max_date": hist_data.index.max(),
            "long_name": info['longName'],
            "sector": info['sector'],
            "summary": info['longBusinessSummary']
        }
        mysql.save_log(conn, data_info)
        print("Info of {} saved".format(tickers[i]))
    except Exception as e:
        print("{}".format(e))
    except:
        print("Error trying get {} info".format(tickers[i]))

conn.close()
