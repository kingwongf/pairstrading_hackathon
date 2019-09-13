
import pandas as pd
import numpy as np
import pickle
from functools import reduce
import itertools
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)  # or 1000

traffic_signals = pickle.load(open('data/backtest/traffic_signals.pkl', 'rb'))
close_pairs = pickle.load(open('data/backtest/close_pairs.pkl', 'rb'))
apparels_closes = pickle.load(open('data/backtest/apparels_closes.pkl', 'rb'))
apparels_closes.index = pd.to_datetime(apparels_closes.index)
signal_close_corr_df = pd.read_csv("data/backtest/corr_signal_close.csv")


def backtest_singals(all_signals_df, signal_close_corr_df, n_most_corr_signals, day_to_revert_pos):
    invested_amounts = []
    ordered_signal_li = signal_close_corr_df.close_ticker.drop_duplicates().tolist()
    test_corr_df_li = [signal_close_corr_df[signal_close_corr_df.close_ticker == close_ticker][0:n_most_corr_signals]
                       for close_ticker in ordered_signal_li]

    test_corr_df_li = [
        nth_signals['traffic_signal'].rename(signal_close_corr_df.close_ticker.drop_duplicates().tolist()[0]).to_frame()
        for nth_signals in test_corr_df_li]

    traffic_signals2 = reduce(lambda X, x: pd.merge(X.reset_index(drop=True), x.reset_index(drop=True), left_index=True,
                                                    right_index=True, how='inner'), test_corr_df_li)

    traffic_signals3 = []

    traffic_signals2_li = traffic_signals2.values.tolist()

    flat_lit = list(itertools.chain.from_iterable(traffic_signals2_li))

    [traffic_signals3.append(tup) for tup in all_signals_df for signal_name in flat_lit if tup[1] == signal_name]

    # traffic_signals3 = list(dict.fromkeys(traffic_signals3))

    ## instead of all signals, we will use only the nth best once/ most correlated with close prices.

    trading_dollar = []
    # for signal in traffic_signals:
    for signal in traffic_signals3:
        ticker, signal_name, signal_df = signal
        close_to_trade = apparels_closes[[col for col in apparels_closes.columns if ticker in col]]
        close_to_trade.columns = ['close']
        signal_df.index = signal_df.index.shift(-1, freq='D')

        ## to calculate return, we need to cumsum the abs amount of the invested amount
        ret_cal = pd.merge_asof(-1 * signal_df['traffic_crosses'].sort_index(), close_to_trade.sort_index(),
                                left_index=True, right_index=True, direction='forward')
        ret_cal['invested'] = ret_cal['traffic_crosses'].abs() * ret_cal['close']
        invested_per_signal = ret_cal.apply(np.sum, axis=0)['invested']
        invested_amounts.append(invested_per_signal)

        exit_pos_df = signal_df.copy().drop(['actual_minus_est'], axis=1)
        exit_pos_df.index = exit_pos_df.index.shift(day_to_revert_pos + 1, freq='D')
        exit_pos_df['reverse_pos'] = exit_pos_df['traffic_crosses']

        trading_pos = pd.merge(signal_df, exit_pos_df.drop('traffic_crosses', axis=1), left_index=True,
                               right_index=True, how='outer')
        trading_pos['net_trade'] = trading_pos['traffic_crosses'].fillna(0) * -1 + trading_pos['reverse_pos'].fillna(0)
        trading_df = pd.merge_asof(trading_pos, close_to_trade, left_index=True, right_index=True, direction='forward')
        trading_df['net_dollar_trade_' + signal_name] = trading_df['net_trade'] * trading_df['close']
        trading_dollar.append(trading_df['net_dollar_trade_' + signal_name])

    dollar_trade_df = reduce(lambda X, x: pd.merge(X.sort_index(), x.sort_index(), left_index=True, right_index=True,
                                                   how='outer'), trading_dollar)

    net_dollar_trade = pd.DataFrame()
    net_dollar_trade['net_trade'] = dollar_trade_df.apply(np.sum, axis=1)
    net_dollar_trade['net_pos'] = net_dollar_trade['net_trade'].cumsum()

    total_invested_amount = reduce(lambda X,x : X+x , invested_amounts)
    return total_invested_amount, net_dollar_trade['net_pos'].rename('n_' + str(n_most_corr_signals) + '_d_' + str(day_to_revert_pos) + '_net_pos')



# print(backtest_singals(traffic_signals, signal_close_corr_df, n_most_corr_signals, day_to_revert_pos))

# there are 22 traffic signals
seek_best_comb = []
port_invested = []
port_stats = pd.DataFrame([], columns=['port_name', 'PnL', 'return'])
for n in range(1,23):
    for d in range(1,10):
        nd_total_invested_amount, nd_port = backtest_singals(traffic_signals, signal_close_corr_df, n, d)
        port_tup = pd.Series([nd_port.name, nd_port[-1], nd_port[-1]/nd_total_invested_amount], ['port_name', 'PnL', 'return'])
        # print(port_tup)
        port_stats = port_stats.append([port_tup], ignore_index=True)
        port_invested.append(nd_total_invested_amount)
        seek_best_comb.append(nd_port)
cum_dollar_port = reduce(lambda X, x: pd.merge(X.sort_index(), x.sort_index(), left_index=True, right_index=True,
                                                   how='outer'), seek_best_comb)

cum_dollar_port = cum_dollar_port.fillna(method='ffill')
cum_dollar_port = cum_dollar_port.fillna(0)
cum_dollar_port.to_csv("data/backtest/cum_dollar_port.csv")

#cum_dollar_port.reset_index().drop("actual_dates", axis=1).plot()
#fig = plt.gcf()
#fig.set_size_inches(18.5, 10.5)
#fig.savefig('test2png.png', dpi=100)
#plt.show()

port_stats.to_csv("data/backtest/port_stats.csv")

print(port_stats.sort_values(by=cum_dollar_port['return'],axis=1, ascending=False))
print(cum_dollar_port.sort_values(by=cum_dollar_port.index[-1],axis=1, ascending=False))
# print(cum_dollar_port.sort_values(cum_dollar_port.tail(1), axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last'))

'''


signal_name | PnL | 3y return
print(cumret)

'''