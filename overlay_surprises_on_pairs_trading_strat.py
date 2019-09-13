import numpy as np
import pandas as pd
from functools import reduce
import re
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

import matplotlib.pyplot as plt

from collections import OrderedDict
import warnings



import pickle
from sklearn import preprocessing as prep

from sklearn.metrics import mean_squared_error
np.set_printoptions(threshold=np.inf)

from datetime import datetime
# pd.set_option('display.max_columns', None)  # or 1000
#pd.set_option('display.max_rows', 1000)  # or 1000
#pd.set_option('display.max_colwidth', 199)  # or 199

def prices_clean(files_path):
    prices_mega_li = []
    for file in files_path:
        prices = pd.read_csv(file).dropna()
        prices.index = pd.to_datetime(prices['Date'])
        close = re.split(r'(;|.csv|/)', file)[-3] + '_Close'
        volume = re.split(r'(;|.csv|/)', file)[-3] + '_Volume'
        prices[close], prices[volume] = prices['Close'], prices['Volume']
        prices_mega_li.append(prices[[close, volume]].sort_index())

    prices_mega = reduce(lambda X, x: pd.merge_asof(X.sort_index(), x.sort_index(), left_index=True, right_index=True,
                                                  direction='forward', tolerance=pd.Timedelta('1d')), prices_mega_li)

    return prices_mega_li, prices_mega

ecommerce_apparels = ['data/OpenData/Prices-Volume/ASOMY.csv',
                       'data/OpenData/Prices-Volume/BHHOF.csv',
                       'data/OpenData/Prices-Volume/EBAY.csv',
                       'data/OpenData/Prices-Volume/ETSY.csv',
                       'data/OpenData/Prices-Volume/GPS.csv',
                      'data/OpenData/Prices-Volume/HNNMY.csv',
                      'data/OpenData/Prices-Volume/JWN.csv',
                      'data/OpenData/Prices-Volume/TJX.csv',
                      'data/OpenData/Prices-Volume/URBN.csv']
ecommerce_apparels_price_li, ecommerce_apparels_prices = prices_clean(ecommerce_apparels)
ecommerce_apparels_prices = ecommerce_apparels_prices[ecommerce_apparels_prices.columns.drop(list(ecommerce_apparels_prices.filter(regex='Volume')))]
ecommerce_apparels_prices = ecommerce_apparels_prices['2009-06-01':]
#print(ecommerce_apparels_prices)

## TODO to pickle/ comment out if debug
ecommerce_apparels_prices.to_pickle("data/ecommerce_apparels_prices.pkl")

## TODO read price pickle
ecommerce_apparels_prices = pd.read_pickle("data/ecommerce_apparels_prices.pkl")


# Compute the correlation matrix
corr = ecommerce_apparels_prices.dropna().corr()



# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 20))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(250, 0, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.8, cbar_kws={"shrink": .5})
yticks = ecommerce_apparels_prices.index
xticks = ecommerce_apparels_prices.index
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.xticks(rotation=90)
plt.gcf().subplots_adjust(bottom=0.15)
plt.title("Empirical Correlation Matrix of E commerces and Apparels")
plt.savefig("data/emp_corr_e_com_apparels.png" , dpi=f.dpi)
#plt.show()
plt.close()

absCorr = corr.abs()

# extract upper triangle without diagonal with k=1
sol = (absCorr.where(np.triu(np.ones(absCorr.shape), k=1).astype(np.bool))
                 .stack()
                 .sort_values(ascending=False)).to_frame()
sol['pairs'] = sol.index
sol = sol.set_index(np.arange(len(sol.index)))



adfStats = []


for i in range(len(sol)):
    # close_1, close_2 = sol['pairs'][i][0], sol['pairs'][i][1]
    pair_dropna = ecommerce_apparels_prices[list(sol['pairs'][i])].dropna()
    model = sm.regression.linear_model.OLS(pair_dropna[pair_dropna.columns[0]], pair_dropna[pair_dropna.columns[1]])
    results = model.fit()
    pairAdfStats = sm.tsa.stattools.adfuller(results.resid)
    adfStats.append(pairAdfStats)

sol['adfStats'] = adfStats
coIntegrate = [(abs(x[0]) > abs(x[4]['10%'])) for x in adfStats]
sol['cointegration'] = coIntegrate


cointegratedPairs = sol[coIntegrate]
cointegratedPairs = cointegratedPairs.reset_index()



## TODO readin fundamentals for pairs > 0.600
'''
0   0.719303    (ASOMY_Close, GPS_Close)
1   0.717439   (ASOMY_Close, ETSY_Close)
2   0.707096   (HNNMY_Close, URBN_Close)
3   0.705522   (ETSY_Close, HNNMY_Close)
4   0.682776      (JWN_Close, TJX_Close)
5   0.665345    (ASOMY_Close, TJX_Close) ## cointegrated 
6   0.662210     (JWN_Close, URBN_Close)
7                 (GPS_Close, JWN_Close) ## cointegrated
8                (EBAY_Close, GPS_Close) ## cointegrated
9             (ASOMY_Close, HNNMY_Close) ## cointegrated
10              (EBAY_Close, URBN_Close) ## cointegrated    
'''
##

## TODO table of refinitiv estimatesactuals

estimatesActuals = pd.read_csv("data/Refinitiv/ESTIMATESACTUALS.csv")
estimatesActuals = estimatesActuals[(estimatesActuals.Instrument == 'ASOS.L') | (estimatesActuals.Instrument == 'GPS')|
                                    (estimatesActuals.Instrument == 'HMb.ST')| (estimatesActuals.Instrument == 'URBN.O')|
                                    (estimatesActuals.Instrument == 'JWN')|(estimatesActuals.Instrument == 'TJX')|(estimatesActuals.Instrument == 'EBAY.O')]

# commented out for | (estimatesActuals.Instrument == 'ETSY.O')| as it is not cointegreated

estimatesActuals.columns = estimatesActuals.columns.map(lambda x: "estimates_actuals_" + x)
estimatesActuals.index = pd.to_datetime(estimatesActuals.estimates_actuals_Date)
ecommerce_apparels_estimates_pivot_by_tickers = estimatesActuals.pivot_table(index=estimatesActuals.index, values=['estimates_actuals_Mean', 'estimates_actuals_High',
                                                                                                       'estimates_actuals_Low'], columns=["estimates_actuals_Instrument","estimates_actuals_Estimate"])
ecommerce_apparels_estimates_pivot_by_tickers = ecommerce_apparels_estimates_pivot_by_tickers.sort_index()
ecommerce_apparels_estimates_pivot_by_tickers.columns = ecommerce_apparels_estimates_pivot_by_tickers.columns.to_series().str.join('_')



actuals = estimatesActuals[['estimates_actuals_Date Actual','estimates_actuals_Actual Value','estimates_actuals_Estimate','estimates_actuals_Instrument']]
actuals.index = pd.to_datetime(actuals['estimates_actuals_Date Actual'])
actuals = actuals.drop_duplicates()
ecommerce_apparels_actuals_pivot_by_tickers = actuals.pivot_table(index=actuals.index, values=['estimates_actuals_Actual Value']
                                                                  , columns=["estimates_actuals_Instrument","estimates_actuals_Estimate"])
ecommerce_apparels_actuals_pivot_by_tickers.columns = ecommerce_apparels_actuals_pivot_by_tickers.columns.to_series().str.join('_')
#print(ecommerce_apparels_actuals_pivot_by_tickers.index)
#print(ecommerce_apparels_estimates_pivot_by_tickers.columns)
#print(ecommerce_apparels_actuals_pivot_by_tickers["estimates_actuals_High_ASOS.L_REV"])

#print(ecommerce_apparels_actuals_pivot_by_tickers)
## TODO plot actual releases affect on price and ratio and export ratios data

pair_names = ['ASOMY_Close to TJX_Close', 'GPS_Close to JWN_Close',
              'EBAY_Close to GPS_Close', 'ASOMY_Close, to HNNMY_Close',
              'EBAY_Close to URBN_Close']

close_pairs = pd.DataFrame([ecommerce_apparels_prices['ASOMY_Close'] / ecommerce_apparels_prices['TJX_Close'],
                            ecommerce_apparels_prices['GPS_Close'] / ecommerce_apparels_prices['JWN_Close'],
                            ecommerce_apparels_prices['EBAY_Close'] / ecommerce_apparels_prices['GPS_Close'],
                            ecommerce_apparels_prices['ASOMY_Close']  / ecommerce_apparels_prices['HNNMY_Close'],
                            ecommerce_apparels_prices['EBAY_Close']   / ecommerce_apparels_prices['URBN_Close']])

close_pairs = close_pairs.swapaxes("index", "columns")
close_pairs.columns = pair_names






## TODO close pairs to pickle

close_pairs.to_pickle("data/backtest/close_pairs.pkl")

close_names = ['ASOMY_Close', 'GPS_Close', 'HNNMY_Close', 'URBN_Close', 'EBAY_Close', 'JWN_Close', 'TJX_Close']
apparels_closes = ecommerce_apparels_prices[close_names]

apparels_closes.to_pickle("data/backtest/apparels_closes.pkl")

def find_actual_dates(col_name1, col_name2):
    actual_releases_index1 = ecommerce_apparels_actuals_pivot_by_tickers[col_name1].dropna().index
    actual_releases_index2 = ecommerce_apparels_actuals_pivot_by_tickers[col_name2].dropna().index
    actual_releases_index = actual_releases_index1.append(actual_releases_index2)
    return actual_releases_index
actual_dates =[]

actual_dates.append(find_actual_dates('estimates_actuals_Actual Value_ASOS.L_EPS', 'estimates_actuals_Actual Value_TJX_EPS'))
actual_dates.append(find_actual_dates('estimates_actuals_Actual Value_GPS_EPS', 'estimates_actuals_Actual Value_JWN_EPS'))
actual_dates.append(find_actual_dates('estimates_actuals_Actual Value_EBAY.O_EPS','estimates_actuals_Actual Value_GPS_EPS'))
actual_dates.append(find_actual_dates('estimates_actuals_Actual Value_ASOS.L_EPS','estimates_actuals_Actual Value_HMb.ST_EPS'))
actual_dates.append(find_actual_dates('estimates_actuals_Actual Value_EBAY.O_EPS','estimates_actuals_Actual Value_URBN.O_EPS'))


# print("ASOS actuals releases date" + str(ecommerce_apparels_actuals_pivot_by_tickers['estimates_actuals_Actual Value_ASOS.L_EPS'].dropna().index.tolist()))
li_pair_stop_trade_dfs = []
for i, pair in enumerate(close_pairs.columns):
    plt.plot(close_pairs[pair])
    plt.title(pair_names[i])
    [plt.axvline(x, color='r', lw=0.5) for x in actual_dates[i]]
    plt.savefig('data/ratios/' + pair_names[i] + 'ratio.png', dpi=f.dpi*2)
    plt.close()
    stop_dates = actual_dates[i].copy().to_frame()
    # print(stop_dates.to_frame())
    stop_dates.index = stop_dates.index.shift(-1, freq='D')
    pair_stop_trade = pd.merge_asof(close_pairs[pair].sort_index(), stop_dates.sort_index(),  left_index=True, right_index=True,
                                                  direction='forward', tolerance=pd.Timedelta('1d'))
    li_pair_stop_trade_dfs.append(pair_stop_trade)

pickle.dump(li_pair_stop_trade_dfs, open("data/backtest/li_pair_stop_trade_dfs.pkl", 'wb'))
    # plt.show()
    # print(actual_releases_index)
#ecommerce_apparels_estimatesActuals = estimatesActuals_pivot_by_tickers[list(set(estimatesActuals_pivot_by_tickers.columns).difference(set(["Ticker"])))]
#estimatesActuals_pivot_by_tickers.to_pickle("data/estimatesActuals_pivot_by_tickers.pkl")



## TODO readin similar web data to predict better estimates => find diff between our predict and analyst estimate => long if outbeat analyst estimates while short if lower than analyst estimates => reverse positions after actual releases and wait for mean reversion

## TODO we want to cumulate similar web data daily until it reaches actual release?
apparels_websites_pivot_by_sites = pd.read_pickle("data/apparels_websites_pivot_by_sites.pkl")
apparels_apps_pivot_by_apps = pd.read_pickle("data/apparels_apps_pivot_by_apps.pkl")


#print(ecommerce_apparels_actuals_pivot_by_tickers.index)
#print(apparels_websites_pivot_by_sites.index)
#print(apparels_apps_pivot_by_apps.index)

## websites traffic join with apps traffic

online_traffic = pd.merge_asof(apparels_apps_pivot_by_apps.sort_index(),apparels_websites_pivot_by_sites.sort_index(),  left_index=True, right_index=True,
                                                  direction='forward', tolerance=pd.Timedelta('1d'))


actuals_ticker_list = [['ASOS.L_REV', 'TJX_REV'],
                       ['GPS_REV', 'JWN_REV'],
                       ['EBAY.O_REV', 'GPS_REV'],
                       ['ASOS.L_REV', 'HMb.ST_REV'],
                       ['EBAY.O_REV'   , 'URBN.O_REV']]

traffic_ticker_list = [['ASC', 'TJX US'],
                       ['GPS', 'JWN US'],
                       ['EBAY US', 'GPS'],
                       ['ASC', 'missing'],
                       ['EBAY US', 'URBN US']]

close_pairs_tickers = [['ASOMY_Close','TJX_Close'],
                           ['GPS_Close','JWN_Close'],
                           ['EBAY_Close','GPS_Close'],
                           ['ASOMY_Close','HNNMY_Close'],
                           ['EBAY_Close','URBN_Close']]



traffic_close_price_mapping = {'ASC': 'ASOMY_Close',
                               'GPS': 'GPS_Close',
                               'EBAY US' : 'EBAY_Close',
                               'URBN US': 'URBN_Close',
                               'JWN US': 'JWN_Close',
                               'TJX US': 'TJX_Close'}

# print(ecommerce_apparels_estimates_pivot_by_tickers.columns)
# print(ecommerce_apparels_estimates_pivot_by_tickers.index)
revenues = []
successful_signals = []
comparables = []
traffic_signals = []
signal_close_corr_dict = {}
signal_close_corr_df = pd.DataFrame([],columns=['close_ticker','traffic_signal','coeff'])

def get_best_corr_func(traffic_ticker_from_li):
    close_ticker = traffic_close_price_mapping[traffic_ticker_from_li]
    traffic_ticker = pair_specific[ten_ma]
    close = apparels_closes[close_ticker]
    signal_fast = pair_specific[ten_ma]
    signal_slow = pair_specific[two_hundred_li[d]]
    corr_check = pd.merge(close, signal_fast, left_index=True, right_index=True, how='inner')
    corr = corr_check[close.name].corr(corr_check[signal_fast.name])
    signal_close_corr_dict[signal_fast.name] = corr



def get_best_corr_func2(traffic_ticker_from_li):
    close_ticker = traffic_close_price_mapping[traffic_ticker_from_li]
    traffic_ticker = pair_specific[ten_ma]
    close = apparels_closes[close_ticker]
    signal_fast = pair_specific[ten_ma]
    signal_slow = pair_specific[two_hundred_li[d]]
    corr_check = pd.merge(close, signal_fast, left_index=True, right_index=True, how='inner')
    corr = corr_check[close.name].corr(corr_check[signal_fast.name])
    corr_temp = pd.DataFrame([[close_ticker, ten_ma, corr]], columns=['close_ticker','traffic_signal','coeff'])
    return corr_temp
    # signal_close_corr_df = signal_close_corr_df.append(corr_temp, ignore_index = True)

for i, idx_list in enumerate(actual_dates):
    # x = traffic[[col for col in traffic.columns.tolist() if 'ASC' in col or 'GPS' in col]].dropna()
    ## TODO find actuals
    actual_dates_df = ecommerce_apparels_actuals_pivot_by_tickers.loc[idx_list]
    actual_dates_df = actual_dates_df[[col for col in actual_dates_df.columns.tolist() if actuals_ticker_list[i][0] in col or actuals_ticker_list[i][1] in col]]
    actual_dates_df['actuals_release_date'] = actual_dates_df.index
    actual_pair_specific = actual_dates_df.sort_index()
    actual_dates_only_dates = pd.DataFrame(actual_dates_df['actuals_release_date'], index=actual_dates_df.index)
    # print(actual_pair_specific)
    #print(actual_dates_df.sort_index())

    ## TODO find mean analyst's estimates
    estimates_temp = pd.merge_asof(ecommerce_apparels_estimates_pivot_by_tickers[[col for col in ecommerce_apparels_estimates_pivot_by_tickers.columns.tolist() if "Mean_" +
                                                                                  actuals_ticker_list[i][0] in col or "Mean_" + actuals_ticker_list[i][1] in col]].sort_index(),
                                   actual_pair_specific.sort_index(), left_index=True, right_index=True,
                                   direction='forward', tolerance=pd.Timedelta('1d'))
    ## need to find a way to forward fill estimates
    # estimates_temp['actuals_release_date'] = estimates_temp['actuals_release_date'].fillna(method='bfill')
    estimates_actuals_pair_specific = estimates_temp
    # print(estimates_pair_specific)


    ## TODO traffic data pair specific
    pair_specific_online_traffic = online_traffic[[col for col in online_traffic.columns.tolist() if traffic_ticker_list[i][0] in col or traffic_ticker_list[i][1] in col]].dropna()

    pair_specific_online_traffic_200_MA = pair_specific_online_traffic.ewm(span=200).mean()
    pair_specific_online_traffic_200_MA.columns = "200_MA_"+ pair_specific_online_traffic_200_MA.columns


    ## TODO change fast MA span 10/ 100
    pair_specific_online_traffic_10_MA = pair_specific_online_traffic.ewm(span=100).mean()      ## TODO change 10/100 here
    pair_specific_online_traffic_10_MA.columns = "100_MA_"+ pair_specific_online_traffic_10_MA.columns ## TODO change 10/100 here

    pair_specific_online_traffic_MAs = pd.merge_asof(pair_specific_online_traffic_10_MA.sort_index(), pair_specific_online_traffic_200_MA.sort_index(), left_index=True, right_index=True,
                                   direction='forward', tolerance=pd.Timedelta('1d'))

    pair_specific = pd.merge_asof(pair_specific_online_traffic_MAs.sort_index(), estimates_actuals_pair_specific, left_index=True, right_index=True,
                                   direction='forward', tolerance=pd.Timedelta('1d'))


    def get_up_cross(col_fast, col_slow):
        crit1 = col_fast.shift(1) < col_slow.shift(1)
        crit2 = col_fast > col_slow
        return col_fast[(crit1) & (crit2)]


    def get_down_cross(col_fast, col_slow):
        crit1 = col_fast.shift(1) > col_slow.shift(1)
        crit2 = col_fast < col_slow
        return col_fast[(crit1) & (crit2)]


    ## TODO join everything together

    rev = pair_specific
    #rev = reduce(lambda X, x: pd.merge_asof(X.sort_index(), x.sort_index(), left_index=True, right_index=True,
    #                                               direction='forward', tolerance=pd.Timedelta('1d')), [actual_pair_specific, estimates_pair_specific, traffic_pair_specific])

    two_hundred_li = pair_specific_online_traffic_200_MA.columns.tolist()



    for d, ten_ma in enumerate(pair_specific_online_traffic_10_MA):
        fig, axs = plt.subplots(3,1)
        axs[0].plot(pair_specific[ten_ma], color='r')
        axs[0].plot(pair_specific[two_hundred_li[d]], color='b')

        ##  1st stock in the pair
        stock_1 = [col for col in actual_dates_df.columns.tolist() if actuals_ticker_list[i][0] in col]
        rev1 = pair_specific[stock_1]
        axs[1].plot(rev1.fillna(method='bfill'),label="Revenue", linewidth=2.0, color='b')
        est1 = pair_specific[[col for col in ecommerce_apparels_estimates_pivot_by_tickers.columns.tolist() if "Mean_" +actuals_ticker_list[i][0] in col]]
        axs[1].plot(est1.fillna(method='ffill'),label="Estimates", color='r')

        ## 2nd stock in the pair
        stock_2 = [col for col in actual_dates_df.columns.tolist() if actuals_ticker_list[i][1] in col]
        rev2 = pair_specific[stock_2]
        axs[2].plot(rev2.fillna(method='bfill'),label="Revenue", linewidth=2.0, color='b')
        est2 = pair_specific[[col for col in ecommerce_apparels_estimates_pivot_by_tickers.columns.tolist() if "Mean_" +actuals_ticker_list[i][1] in col]]
        axs[2].plot(est2.fillna(method='ffill'),label="Estimates", color='r')
        ## 1st stock actual minus estimate
        temp1 = pd.merge_asof(rev1, est1.fillna(method='ffill').shift(1), left_index=True, right_index=True,
                                   direction='forward', tolerance=pd.Timedelta('1d'))

        temp1['actual_minus_est'] = temp1[rev1.columns.tolist()[0]] - temp1[est1.columns.tolist()[0]]

        ae_sides_1 = pd.DataFrame(np.sign(temp1['actual_minus_est'].dropna()), index=np.sign(temp1['actual_minus_est'].dropna()).index)
        ae_sides_1['actual_dates'] = ae_sides_1.index

        # print(np.sign(temp1['actual_minus_est'].dropna()))


        ## 2nd stock actual minus estimate
        temp2 = pd.merge_asof(rev2, est2.fillna(method='ffill').shift(1), left_index=True, right_index=True,
                                   direction='forward', tolerance=pd.Timedelta('1d'))


        temp2['actual_minus_est'] = temp2[rev2.columns.tolist()[0]] - temp2[est2.columns.tolist()[0]]

        ae_sides_2 = pd.DataFrame(np.sign(temp2['actual_minus_est'].dropna()), index=np.sign(temp2['actual_minus_est'].dropna()).index)
        ae_sides_2['actual_dates'] = ae_sides_2.index


        figure = plt.gcf()
        figure.set_size_inches(12, 8)
        # plt.show()

        plt.savefig("data/revenues/move_avgs/" + "200_" + ten_ma + ".png",  dpi = 100)
        plt.close()

        ## TODO to find the best MA pairs, we find the ones which are most correlated to close prices



        if traffic_ticker_list[i][0] in ten_ma:
            # get_best_corr_func(traffic_ticker_list[i][0])
            signal_close_corr_df = signal_close_corr_df.append(get_best_corr_func2(traffic_ticker_list[i][0]))

        elif traffic_ticker_list[i][1] in ten_ma:
            # get_best_corr_func(traffic_ticker_list[i][1])
            signal_close_corr_df = signal_close_corr_df.append(get_best_corr_func2(traffic_ticker_list[i][1]))



        ## TODO get upcross and downcross of MAs and join back to actuals and estimates df

        up = get_up_cross(pair_specific[ten_ma], pair_specific[two_hundred_li[d]])
        down = get_down_cross(pair_specific[ten_ma], pair_specific[two_hundred_li[d]])

        side_up = pd.Series(1, index=up.index)
        side_down = pd.Series(-1, index=down.index)
        side = pd.concat([side_up, side_down]).sort_index()
        side = pd.DataFrame(side, columns=['traffic_crosses'])

        orig_sides_1 = pd.merge(side, ae_sides_1, left_index=True, right_index=True, how='outer')
        sides1 = orig_sides_1.copy()
        sides1['traffic_crosses'] = sides1['traffic_crosses'].fillna(0)
        sides1.index = sides1["actual_dates"].fillna(method='bfill')
        comparable1 = sides1.drop("actual_dates", axis=1).groupby("actual_dates").agg("sum")

        # sides1['actual_dates'] = np.where(sides1.actual_minus_est.notnull(), sides1.index, np.nan)
        # sides1['actual_dates'] = sides1.assign(color=sides1.apply(cond, axis=1))

        # print(sides1.groupby(np.where(sides1.actual_minus_est.notnull())).agg('sum'))
        orig_sides_2 = pd.merge(side, ae_sides_2, left_index=True, right_index=True, how='outer')
        sides2 = orig_sides_2.copy()
        sides2['traffic_crosses'] = sides2['traffic_crosses'].fillna(0)
        sides2.index = sides2["actual_dates"].fillna(method='bfill')
        comparable2 = sides2.drop("actual_dates", axis=1).groupby("actual_dates").agg("sum")

        # comparables_tickers.append()
        if traffic_ticker_list[i][0] in ten_ma:
            traffic_signals.append((close_pairs_tickers[i][0], ten_ma, comparable1))
            # print("it's the first loop", ten_ma)
            if comparable1['traffic_crosses'].equals(comparable1['actual_minus_est']):
                successful_signals.append((stock_1, ten_ma))
                # print(str(ten_ma) + " works !!")
        elif traffic_ticker_list[i][1] in ten_ma:
            traffic_signals.append((close_pairs_tickers[i][1], ten_ma, comparable2))
            if comparable2['traffic_crosses'].equals(comparable2['actual_minus_est']):
                successful_signals.append((stock_2, ten_ma))
                # print(str(ten_ma) + " works !!")

    rev.to_csv("data/revenues/" +str(actuals_ticker_list[i]) + ".csv")
    # rev_scalar = prep.MinMaxScaler().fit(rev.drop('actuals_release_date', axis=1))
    # rev = rev.drop('actuals_release_date', axis=1)
    # rev[rev.columns] = prep.MinMaxScaler().fit_transform(rev)
    # pd.DataFrame(rev).plot()
    # plt.show()
    # revenues.append(rev)


signal_close_corr_df = signal_close_corr_df.sort_values(by=['close_ticker','coeff'], ascending=False)

signal_close_corr_df.drop_duplicates().to_csv("data/backtest/corr_signal_close.csv")

# top_corr_signal = signal_close_corr_df.groupby("close_ticker").agg("max")


## TODO to pickle/ comment out if debug
pickle.dump(traffic_signals, open("data/backtest/traffic_signals.pkl", 'wb'))

    # print(actual_dates_df)

    # traffic_gpby_actual_dates = traffic_gpby_actual_dates.gr


## TODO to pickle/ comment out if debug
pickle.dump(successful_signals, open("data/backtest/successful_signals.pkl", 'wb'))

## TODO for 100MA/ 200MA crosses with sum
## [(['estimates_actuals_Actual Value_URBN.O_REV'], '100_MA_mobile_visit_duration_URBN US'),
## (['estimates_actuals_Actual Value_JWN_REV'], '100_MA_average_sessions_per_user_JWN US'),
##  (['estimates_actuals_Actual Value_JWN_REV'], '100_MA_usage_penetration_JWN US'),
## (['estimates_actuals_Actual Value_URBN.O_REV'], '100_MA_mobile_visit_duration_URBN US')]

## TODO 10MA/ 200MA crosses with sum

##[(['estimates_actuals_Actual Value_JWN_REV'], '10_MA_usage_penetration_JWN US'),
## (['estimates_actuals_Actual Value_JWN_REV'], '10_MA_mobile_visits_JWN US')]

## TODO for 100MA/ 200MA crosses with last
## []

## TODO for 10MA/ 200MA crosses with last
## []


## online traffic join with actuals

#print(ecommerce_apparels_actuals_pivot_by_tickers.index)
ecommerce_apparels_actuals_pivot_by_tickers['actuals_date'] = ecommerce_apparels_actuals_pivot_by_tickers.index

ecommerce_apparels_actuals_pivot_by_tickers.to_csv("data/ecommerce_apparels_actuals_pivot_by_tickers.csv")

XY = pd.merge_asof(online_traffic.sort_index(),ecommerce_apparels_actuals_pivot_by_tickers['actuals_date'].sort_index(),  left_index=True, right_index=True,
                                                  direction='forward', tolerance=pd.Timedelta('1d'))

# print(XY.columns)
XY['actuals_date'] = XY['actuals_date'].fillna(method='bfill')

#print(XY.columns)
traffic = XY.groupby('actuals_date').agg('sum')
traffic_actuals = pd.merge_asof(traffic.sort_index(),ecommerce_apparels_actuals_pivot_by_tickers.sort_index(),  left_index=True, right_index=True,
                                                  direction='forward', tolerance=pd.Timedelta('1d'))

#print(traffic_actuals)

traffic_actuals.to_csv("data/traffic_actuals.csv")
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score

## for asos and gap

## TODO with actutals


## results
## with asos actuals as y1 0.43938772433954054
## The rmse of prediction is: 0.7487404594787564
## with gap actuals as y2 0.18853415633755444
## The rmse of prediction is: 0.9008139894908636

enet = ElasticNet(alpha=0.1, l1_ratio=0.7)
x = traffic[[ col for col in traffic.columns.tolist() if 'ASC' in col or 'GPS' in col]].dropna()
y1 = ecommerce_apparels_actuals_pivot_by_tickers['estimates_actuals_Actual Value_ASOS.L_REV']
y2 = ecommerce_apparels_actuals_pivot_by_tickers['estimates_actuals_Actual Value_GPS_REV']

# print("actuals dates", y1.sort_index().index)

asos_xy1 = pd.merge_asof(x,y1,  left_index=True, right_index=True,
                                                  direction='forward', tolerance=pd.Timedelta('1d'))
asos_xy1['estimates_actuals_Actual Value_ASOS.L_REV'] = asos_xy1['estimates_actuals_Actual Value_ASOS.L_REV'].fillna(method='ffill')

gap_xy2 = pd.merge_asof(y2,x,  left_index=True, right_index=True,
                                                  direction='forward', tolerance=pd.Timedelta('1d'))
gap_xy2['estimates_actuals_Actual Value_GPS_REV'] = gap_xy2['estimates_actuals_Actual Value_GPS_REV'].fillna(method='ffill')

from sklearn import preprocessing as prep

asos_x1, asos_y1 = asos_xy1.dropna()[x.columns.tolist()], asos_xy1.dropna()['estimates_actuals_Actual Value_ASOS.L_REV']
scaler_asos_x1, scaler_asos_y1 = prep.MinMaxScaler().fit(asos_x1.values.reshape(1, -1)),prep.MinMaxScaler().fit(asos_y1.values.reshape(1, -1))
scaled_asos_x1, scaled_asos_y1 = scaler_asos_x1.transform(asos_x1.values.reshape(1, -1)), scaler_asos_y1.transform(asos_y1.values.reshape(1, -1))

gap_x2, gap_y2 = gap_xy2.dropna()[x.columns.tolist()], gap_xy2.dropna()['estimates_actuals_Actual Value_GPS_REV']
scaler_gap_x2, scaler_gap_y2 = prep.MinMaxScaler().fit(gap_x2.values.reshape(1, -1)),prep.MinMaxScaler().fit(gap_y2.values.reshape(1, -1))
scaled_gap_x2, scaled_gap_y2 = scaler_gap_x2.transform(gap_x2.values.reshape(1, -1)), scaler_gap_y2.transform(gap_y2.values.reshape(1, -1))

y_pred_enet1  = enet.fit(scaled_asos_x1, scaled_asos_y1).predict(scaled_asos_x1)

r2_score_str_1 = r2_score(scaled_asos_y1, y_pred_enet1)

# print("with asos actuals as y1 " + str(r2_score_str_1))
# print('The rmse of prediction is:', mean_squared_error(scaled_asos_y1, y_pred_enet1) ** 0.5)


y_pred_enet2 = enet.fit(scaled_gap_x2, scaled_gap_y2).predict(scaled_gap_x2)
r2_score_str_2 = r2_score(scaled_gap_y2, y_pred_enet2)

# print("with gap actuals as y2 " + str(r2_score_str_2))
# print('The rmse of prediction is:', mean_squared_error(scaled_gap_y2, y_pred_enet2) ** 0.5)


## pred rev vs actual rev vs analyst mean

#print(asos_y1)
pred_asos_rev = scaler_asos_y1.inverse_transform(y_pred_enet1)

#print(pred_asos_rev)





## TODO with fundamentals

## results
## R2 with asos fundamentals as y1 0.9193350330952228
## The rmse of prediction is: 0.28401578636543645
## R2 with gap fundamentals as y2 0.9524030045781166
## The rmse of prediction is: 0.2181673564534426
## with lightgbm, The rmse of prediction is: 1.0

enet = ElasticNet(alpha=0.1, l1_ratio=0.7)
x = online_traffic[[ col for col in online_traffic.columns.tolist() if 'ASC' in col or 'GPS' in col]].dropna()
asos_fundamentals = pd.read_csv("data/OpenData/Fundamentals/fundamentals_ASOS PLC.csv")
asos_fundamentals.index = pd.to_datetime(asos_fundamentals.date)
# print("fundamentals dates ", asos_fundamentals.sort_index().index)
gap_fundamentals = pd.read_csv("data/OpenData/Fundamentals/fundamentals_Gap Inc US.csv")
gap_fundamentals.index = pd.to_datetime(gap_fundamentals.date)




'''

y1 = asos_fundamentals['totalRevenue']
y2 = gap_fundamentals['totalRevenue']

asos_xy1 = pd.merge_asof(x,y1.sort_index(),  left_index=True, right_index=True,
                                                  direction='forward', tolerance=pd.Timedelta('1d'))
asos_xy1['totalRevenue'] = asos_xy1['totalRevenue'].fillna(method='ffill')

gap_xy2 = pd.merge_asof(y2.sort_index(),x, left_index=True, right_index=True,
                                                  direction='forward', tolerance=pd.Timedelta('1d'))
gap_xy2['totalRevenue'] = gap_xy2['totalRevenue'].fillna(method='ffill')

from sklearn import preprocessing as prep
import lightgbm as lgb


scaled_asos_xy1 = pd.DataFrame(prep.StandardScaler().fit_transform(asos_xy1.dropna()), columns=asos_xy1.dropna().columns, index=asos_xy1.dropna().index)
scaled_gap_xy2 = pd.DataFrame(prep.StandardScaler().fit_transform(gap_xy2.dropna()), columns=gap_xy2.dropna().columns, index=gap_xy2.dropna().index)

scaled_asos_x1 = scaled_asos_xy1[x.columns.tolist()]
scaled_gap_x2 = scaled_gap_xy2[x.columns.tolist()]

scaled_asos_y1 = scaled_asos_xy1['totalRevenue']
y_pred_enet1  = enet.fit(scaled_asos_x1, scaled_asos_y1).predict(scaled_asos_x1)
r2_score_str_1 = r2_score(scaled_asos_y1, y_pred_enet1)

print("R2 with asos fundamentals as y1 " + str(r2_score_str_1))
print('The rmse of prediction is:', mean_squared_error(scaled_asos_y1, y_pred_enet1) ** 0.5)


scaled_gap_y2 = scaled_gap_xy2['totalRevenue']

y_pred_enet2 = enet.fit(scaled_gap_x2, scaled_gap_y2).predict(scaled_gap_x2)
r2_score_str_2 = r2_score(scaled_gap_y2, y_pred_enet2)

print("R2 with gap fundamentals as y2 " + str(r2_score_str_2))
print('The rmse of prediction is:', mean_squared_error(scaled_gap_y2, y_pred_enet2) ** 0.5)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
lgb_train = lgb.Dataset(scaled_asos_x1, )
lgb_eval = lgb.Dataset(scaled_asos_x1, scaled_asos_y1, reference=lgb_train)


y_pred_lgb1 = lgb.train(params, lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5).predict(scaled_asos_x1)

print('The rmse of prediction is:', mean_squared_error(scaled_asos_y1, y_pred_lgb1) ** 0.5)


#print(y_pred_enet1, scaled_asos_y1)
'''