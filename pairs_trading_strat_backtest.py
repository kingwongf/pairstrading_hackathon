
import pandas as pd
import numpy as np
import pickle
from functools import reduce
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)  # or 1000

li_pair_stop_trade_dfs = pickle.load(open('data/backtest/li_pair_stop_trade_dfs.pkl', 'rb'))
apparels_closes = pickle.load(open('data/backtest/apparels_closes.pkl', 'rb'))
apparels_closes.index = pd.to_datetime(apparels_closes.index)

# print(li_pair_stop_trade_dfs[0].dropna())

for ind,pair in enumerate(li_pair_stop_trade_dfs):
    pair_name = pair.columns.tolist()[0]
    df_pair  = pd.merge_asof(apparels_closes[[col for col in apparels_closes.columns.tolist() if col in pair.columns[0]]]
                             , pair, left_index=True, right_index=True,
                                   direction='forward', tolerance=pd.Timedelta('1d'))
    alpha=0.5
    span=20
    df_pair['mean'] = df_pair[pair_name].ewm(span=span).mean()
    df_pair['std'] = df_pair[pair_name].ewm(span=span).std()
    z = 2
    df_pair['UB'], df_pair['LB'] = df_pair['mean'] + df_pair['std']*z , df_pair['mean'] - df_pair['std']*z
    #print(df_pair.columns)
    plt.plot(df_pair[['UB', 'LB', pair_name]])
    #plt.show()
    break
