import pandas as pd
import numpy as np
import pickle
pd.set_option('display.max_columns', None)  # or 1000

traffic_signals = pickle.load(open('data/backtest/traffic_signals.pkl', 'rb'))
close_pairs = pickle.load(open('data/backtest/close_pairs.pkl', 'rb'))
apparels_closes = pickle.load(open('data/backtest/apparels_closes.pkl', 'rb'))
apparels_closes.index = pd.to_datetime(apparels_closes.index)



## we create a df for each signal to grab long/ short prices
## we long/ short 1 day before the actuals release, then revert position after 3 days
## we get the long/short prices and PnL by the end of the 3 days

## traffic signals => tuple(ticker, signal_name, signal_df)

for signal in traffic_signals:
    ticker, signal_name, signal_df = signal
    close_to_trade = apparels_closes[[col for col in apparels_closes.columns if ticker in col]]
    signal_df.index = signal_df.index.shift(-1, freq='D')
    exit_pos_df = signal_df.copy().drop(['actual_minus_est'], axis=1)
    exit_pos_df.index  = exit_pos_df.index.shift(5, freq='D')
    exit_pos_df['reverse_pos'] = exit_pos_df['traffic_crosses']*-1
    trading_pos = pd.merge(signal_df, exit_pos_df.drop('traffic_crosses', axis=1), left_index=True, right_index=True, how='outer')
    close_to_trade.columns = ['close']
    trading_pos['net_trade'] = trading_pos['traffic_crosses'].fillna(0) + trading_pos['reverse_pos'].fillna(0)
    trading_df = pd.merge_asof(trading_pos, close_to_trade, left_index=True, right_index=True,direction='forward')
    trading_df['net_pos'] = trading_df['net_trade'] * trading_df['close']
    print(trading_df)








# print(close_pairs.head(5))
# print(apparels_closes.tail(5).sort_index())

