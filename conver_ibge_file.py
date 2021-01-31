import numpy as np
import pandas as pd
from datetime import datetime

MONTHS = {'janeiro': 1,  'fevereiro': 2, u'março': 3,    'abril': 4,
               'maio': 5,     'junho': 6,     'julho': 7,     'agosto': 8,
               'setembro': 9, 'outubro': 10,  'novembro': 11, 'dezembro': 12}


result = None
files = [
    ['./data/vol_servico.csv', 'volume_servico'],
    ['./data/vol_vendas.csv', 'volume_vendas'],
    ['./data/ind_indus_carne.csv', 'industria_carne']
]

for path, name in files:
    df = pd.read_csv(path)
    df = df.transpose()
    df.columns = [name]
    indexes = df.index.str.split(' ')
    df['date'] = [datetime(int(year), MONTHS[month], 1) for month, year in indexes]
    
    if result is None:
        result = df
    else:
        result = pd.merge(result, df, on='date')

result.to_csv('./data/indices.csv', index=False)

