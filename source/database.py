import sqlite3
from sqlite3 import Error

create_fatch_log_sql ="""
CREATE TABLE IF NOT EXISTS fatch_log (
    ticker TEXT NOT NULL,
    fetched_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    min_date TEXT,
    max_date TEXT,
    long_name TEXT,
    sector TEXT,
    summary TEXT
)
"""

def save_log (conn, log):
    questions = ['?' for _ in range(len(log))]
    sql = "INSERT INTO fatch_log ({}) VALUES({})" \
        .format(','.join(log.keys()), ','.join(questions))
    cur = conn.cursor()
    cur.execute(sql, [str(col) for col in log.values()])
    conn.commit()
    return cur.lastrowid

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

def create_fatch_log_table(conn):
    cur = conn.cursor()
    cur.execute(create_fatch_log_sql)
    conn.commit()
    return conn