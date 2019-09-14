
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

def get_UB_touch(ts, UB):
    crit1 = ts.shift(1) < UB.shift(1)
    crit2 = ts > UB
    return ts[(crit1) & (crit2)].rename('UB_touch')

def get_LB_touch(ts, LB):
    crit1 = ts.shift(1) > LB.shift(1)
    crit2 = ts < LB
    return ts[(crit1) & (crit2)].rename('LB_touch')


port_stats = pd.DataFrame([], columns=['port_name', 'PnL', 'return'])
for ind,pair in enumerate(li_pair_stop_trade_dfs):
    pair_name = pair.columns.tolist()[0]



    pair['stop_trading'] = pair['estimates_actuals_Date Actual'].notna()


    df_pair  = pd.merge(apparels_closes[[col for col in apparels_closes.columns.tolist() if col in pair.columns[0]]]
                             , pair, left_index=True, right_index=True, how='left').sort_index()


    # print(df_pair)
    alpha=0.5
    span=50
    df_pair['mean'] = df_pair[pair_name].ewm(span=span).mean()
    df_pair['std'] = df_pair[pair_name].ewm(span=span).std()
    df_pair = df_pair['2016-04-03'::]
    z = 1.645
    df_pair['UB'], df_pair['LB'] = df_pair['mean'] + df_pair['std']*z , df_pair['mean'] - df_pair['std']*z
    #print(df_pair.columns)
    # plt.plot(df_pair[['UB', 'LB', pair_name]])
    # plt.show()

    UB_touch_df, LB_touch_df = get_UB_touch(df_pair[pair_name], df_pair['UB']), get_LB_touch(df_pair[pair_name], df_pair['LB'])

    # stop_trade_df['stop_trading'] = False
    # stop_trade_df['']

    # print(UB_touch_df)
    list_trading_df = [df_pair, UB_touch_df, LB_touch_df]

    trading_df = reduce(lambda X, x: pd.merge_asof(X.sort_index(), x.sort_index(), left_index=True, right_index=True,
                                                  direction='forward', tolerance=pd.Timedelta('1d')), list_trading_df)


    # print(trading_df)
    # print(trading_df.columns)
    # print(trading_df['UB_touch'].notna())

    # print(trading_df['stop_trading'])
    # print(trading_df[['stop_trading', 'UB_touch']].groupby(['stop_trading', 'UB_touch']).agg('count'))
    # print(trading_df[(trading_df['UB_touch'] > 0 )])

    # trading_df['net_long'] = trading_df[['UB_touch']
    # trading_df['net_short'] = trading_df['LB_touch'].apply(lambda x: True if x > 0 else False)

    # print(trading_df['net_long'].dropna(), print(trading_df['stop_trading'].dropna()))

    trading_df['net_long'] = (trading_df['UB_touch'] > 0) & (trading_df['stop_trading'] == False)
    trading_df['net_short'] = (trading_df['LB_touch'] > 0) & (trading_df['stop_trading'] == False)

    # print(trading_df.net_short[trading_df['net_short'] == True])

    trading_df['net_trade'] = trading_df.net_long.astype(int) + trading_df.net_short.astype(int)*-1

    # print(trading_df.net_trade[trading_df['net_trade'] == 1])

    # print(trading_df.columns)

    exit_days = 10

    trading_df.loc[trading_df.net_trade == 1, 'enter_net_book_UB'] = trading_df['net_trade']*-1*trading_df[trading_df.columns.tolist()[0]] + \
                                trading_df['net_trade']*trading_df[trading_df.columns.tolist()[1]]

    trading_df.loc[trading_df.net_trade == 1, 'exit_net_book_UB'] = trading_df['net_trade']*trading_df[trading_df.columns.tolist()[0]].shift(exit_days) + \
                                trading_df['net_trade']*-1*trading_df[trading_df.columns.tolist()[1]].shift(exit_days)

    trading_df['net_book_UB'] = trading_df['enter_net_book_UB'] + trading_df['exit_net_book_UB']

    # print(trading_df['net_book_UB'].fillna(0).cumsum())


    trading_df.loc[trading_df.net_trade == -1, 'enter_net_book_LB'] = trading_df['net_trade']*trading_df[trading_df.columns.tolist()[0]] + \
                                trading_df['net_trade']*-1*trading_df[trading_df.columns.tolist()[1]]

    trading_df.loc[trading_df.net_trade == -1, 'exit_net_book_LB'] = trading_df['net_trade']*-1*trading_df[trading_df.columns.tolist()[0]].shift(exit_days) + \
                                trading_df['net_trade']*trading_df[trading_df.columns.tolist()[1]].shift(exit_days)

    trading_df['net_book_LB'] = trading_df['enter_net_book_LB'] + trading_df['exit_net_book_LB']
    # print(trading_df['net_book_LB'].fillna(0).cumsum())

    trading_df['net_port_position'] = trading_df['net_book_UB'].fillna(0).cumsum() + trading_df['net_book_LB'].fillna(0).cumsum()

    invested_amount = abs(trading_df['enter_net_book_UB'].fillna(0)).cumsum() + abs(trading_df['enter_net_book_LB'].fillna(0)).cumsum()

    # print(pair[pair.columns.tolist()[0]].name)
    port_tup = pd.Series([pair[pair.columns.tolist()[0]].name, trading_df['net_port_position'][-1],
                          trading_df['net_port_position'][-1] / invested_amount[-1]],
                         ['port_name', 'PnL', 'return'])

#    print(port_tup)


    port_stats = port_stats.append([port_tup], ignore_index=True)

port_stats['return'] = np.power(port_stats['return'] + 1, 1/3) - 1
print(port_stats)
