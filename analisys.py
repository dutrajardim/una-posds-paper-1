import sqlite3
import pandas as pd
import re
import source.analisys as anl
import yfinance as yf
from datetime import datetime, timedelta
import source.helpers as hp

df_bvsp = yf.download("^BVSP", period="max", progress=False)
df_bvsp.columns = [hp.snake_case(col) for col in df_bvsp.columns]

def get_tickers(conn):
    sql = "select name from sqlite_master where type = 'table'"
    cur = conn.cursor()
    cur.execute(sql)

    tables = cur.fetchall()
    ptrn_tticker = re.compile("^hist_.+$")
    tickers = [ticker[0][5:] for ticker in tables if ptrn_tticker.match(ticker[0])]
    return tickers

def histories(conn):
    tickers = get_tickers(conn)
    for ticker in tickers:
        sql = "SELECT * FROM 'hist_%s'"% ticker
        df = pd.read_sql_query(sql, index_col='Date', parse_dates=['Date'], con=conn)
        yield (ticker, df)

def filter_period(df, start, end):
    if start and end:
        return df[start:end]
    else:
        return df

def check_period(df, start, end):
    min_date = datetime.strptime(start, '%Y-%m-%d')
    max_date = datetime.strptime(end, '%Y-%m-%d')
    t_delta = max_date - min_date - timedelta(days=10)
    df_delta = df.index.max() - df.index.min()
    return df_delta >= t_delta


def analisys(hist_iterator, start, end):
    
    bvsp_returns = filter_period(df_bvsp.adj_close.pct_change(), start, end)

    for ticker, df in hist_iterator:
        returns = filter_period(df.close.pct_change(), start, end)
        
        if check_period(returns, start, end):
            
            yield {
                'ticker': ticker,
                'beta': beta(returns, bvsp_returns),
                'cumulative_return': anl.cum_return(returns),
                'annual_rate_return': anl.annual_rate_return(returns),
                'annualyzed_volatility': anl.annualyzed_volatility(returns)
            }

def beta(returns, market):
    data = pd.merge(returns, market, left_index=True, right_index=True)
    cov_matrix = data.cov() * 252
    cov = cov_matrix.iloc[0,1]
    m_var = market.var() * 252
    return cov/m_var

conn = sqlite3.connect("./data/db.sqlite")

hist_iterator = histories(conn)
anl_iterator = analisys(hist_iterator, start='2019-12-20', end='2021-01-08')
results = [result for result in anl_iterator]
df_results = pd.DataFrame(results)
df_results.to_csv("./data/results_2020.csv", index=False)

conn.close()

