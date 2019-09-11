import pandas as pd
import os
import numpy as np
from functools import reduce
import operator
import re
#pd.set_option('display.max_columns', None)  # or 1000
#pd.set_option('display.max_rows', None)  # or 1000
#pd.set_option('display.max_colwidth', 199)  # or 199


consensus_files = []
consensus_path = "data/OpenData/Consensus/"
for r, d, f in os.walk(consensus_path):
    for file in f:
        if '.csv' in file:
            consensus_files.append(os.path.join(r, file))


def consensus_clean(files_path):
    consensus_mega_li = []
    for file in files_path:
        consensus = pd.read_csv(file)
        consensus.index = pd.to_datetime(consensus['date'])
        consensus[re.split(r'(;|.csv|/)', file)[-3] + ' analysis_mean'] = consensus['analysis_mean']
        consensus_mega_li.append(consensus[re.split(r'(;|.csv|/)', file)[-3] + ' analysis_mean'].sort_index())
    consensus_mega = reduce(lambda X, x: pd.merge_asof(X.sort_index(), x.sort_index(), left_index=True, right_index=True,
                                                    direction='forward', tolerance=pd.Timedelta('1d')), consensus_mega_li)
    return consensus_mega_li, consensus_mega
## TODO list of consensus
consensus_files.remove('data/OpenData/Consensus/consensus_CHARACTER GROUP.csv')
consensus_files.insert(0, 'data/OpenData/Consensus/consensus_CHARACTER GROUP.csv')
consensus_list, consensus_table = consensus_clean(consensus_files)
consensus_table.to_pickle("data/consensus.pkl")

fundamentals_files = []
fundamentals_path = "data/OpenData/Fundamentals/"
for r, d, f in os.walk(fundamentals_path):
    for file in f:
        if '.csv' in file:
            fundamentals_files.append(os.path.join(r, file))

def fundamentals_clean(files_path):
    fundamentals_mega_li = []
    for file in files_path:
        fundamentals = pd.read_csv(file)
        fundamentals.index = pd.to_datetime(fundamentals['date'])
        costOfRevStr = re.split(r'(;|.csv|/)', file)[-3] + '_costOfRevenue'
        totalRevStr = re.split(r'(;|.csv|/)', file)[-3] + '_totalRevenue'
        fundamentals[costOfRevStr] = fundamentals['costOfRevenue']
        fundamentals[totalRevStr] = fundamentals['totalRevenue']
        fundamentals_mega_li.append(fundamentals[[costOfRevStr, totalRevStr]].sort_index())
    #fundamentals_mega = reduce(lambda X, x: pd.merge_asof(X.sort_index(), x.sort_index(), left_index=True, right_index=True,
    #                                          direction='forward', tolerance=pd.Timedelta('1d')), fundamentals_mega_li)
    return fundamentals_mega_li #fundamentals_mega

## TODO list of fundamentals, index matching rate very low, need to use list instead of merge asof
fundamentals_files.remove("data/OpenData/Fundamentals/fundamentals_Nordstrom Inc.csv")
fundamentals_files.insert(0, "data/OpenData/Fundamentals/fundamentals_Nordstrom Inc.csv")
fundamentals_list = fundamentals_clean(fundamentals_files)
#fundamentals_table.to_pickle(("data/fundamentals.pkl")


prices_files = []
prices_path = "data/OpenData/Prices-Volume/"
for r, d, f in os.walk(prices_path):
    for file in f:
        if '.csv' in file:
            prices_files.append(os.path.join(r, file))



def prices_clean(files_path):
    prices_mega_li = []
    for file in files_path:
        prices = pd.read_csv(file).dropna()
        prices.index = pd.to_datetime(prices['Date'])
        close = re.split(r'(;|.csv|/)', file)[-3] + '_Close'
        volume = re.split(r'(;|.csv|/)', file)[-3] + '_Volume'
        prices[close], prices[volume] = prices['Close'], prices['Volume']
        prices_mega_li.append(prices[[close, volume]].sort_index())

        ## TODO starts merging from JWN
        prices_mega = reduce(lambda X, x: pd.merge_asof(X.sort_index(), x.sort_index(), left_index=True, right_index=True,
                                                  direction='forward', tolerance=pd.Timedelta('1d')), prices_mega_li)

    return prices_mega_li, prices_mega

## TODO list of prices
#print(prices_clean(prices_files))
prices_files.reverse()
prices_files.insert(0, 'data/OpenData/Prices-Volume/JWN.csv')
prices_files.pop(3)
price_list, prices_table = prices_clean(prices_files)

prices_table.to_pickle("data/prices.pkl")

## TODO table of refinitiv estimatesactuals

estimatesActuals = pd.read_csv("data/Refinitiv/ESTIMATESACTUALS.csv")

estimatesActuals.columns = estimatesActuals.columns.map(lambda x: "estimates_actuals_" + x)
estimatesActuals.index = pd.to_datetime(estimatesActuals.estimates_actuals_Date)
estimatesActuals_pivot_by_tickers = estimatesActuals.pivot_table(index=estimatesActuals.index, values=['estimates_actuals_Mean', 'estimates_actuals_High',
                                                                                                       'estimates_actuals_Low',
                                                                                                       'estimates_actuals_Actual Value'], columns=["estimates_actuals_Instrument","estimates_actuals_Estimate"])
estimatesActuals_pivot_by_tickers = estimatesActuals_pivot_by_tickers.sort_index()
estimatesActuals_pivot_by_tickers.columns = estimatesActuals_pivot_by_tickers.columns.to_series().str.join('_')
estimatesActuals_pivot_by_tickers.to_pickle("data/estimatesActuals_pivot_by_tickers.pkl")



## TODO table of SimilarWeb site visits


websites = pd.read_csv("data/SimilarWeb/websites_19_07.csv")
websites.to_pickle("data/SimilarWeb/websites_19_07.pkl")
websites = pd.read_pickle("data/SimilarWeb/websites_19_07.pkl")
xls = pd.ExcelFile("data/SimilarWeb/SimilarWeb Mapped Tickers (Jul-2019).xlsx")


tickers_mapping = pd.read_excel(xls, "Domains")
tickers_mapping['site'] = tickers_mapping['Domain']
websites_ticker = pd.merge(websites,
                 tickers_mapping[['site', 'Ticker']],
                 on='site',
                 how='left')

similarWeb_tickers_list = ["ASC LN", "BOO LN","BKNG US","EBAY US","ETSY US","GPS US","HD US","JWN US","TJX US",
                                "URBN US","W US","ZAL GR","ATVI US","EA US"]

apparels_similarWeb_tickers_list = ["ASC LN", "BOO LN","EBAY US","ETSY US","GPS US","JWN US","TJX US",
                                "URBN US","ZAL GR"]


websites_ticker = websites_ticker[websites_ticker['Ticker'].isin(apparels_similarWeb_tickers_list)]
websites_ticker.date = "20" + websites_ticker.year.map(str) + "-" + websites_ticker.month.map(str) + "-" + websites_ticker.day.map(str)
websites_ticker.index = pd.to_datetime(websites_ticker.date)
websites_ticker = websites_ticker.drop(["site", "country","year","month","day"], axis=1)
websites_pivot_by_sites = websites_ticker.pivot_table(index=websites_ticker.index, values=list(set(websites_ticker.columns).difference(set(["Ticker"]))), columns=["Ticker"])
websites_pivot_by_sites.columns = websites_pivot_by_sites.columns.to_series().str.join('_')
websites_pivot_by_sites.to_pickle("data/apparels_websites_pivot_by_sites.pkl")
websites_pivot_by_sites.to_csv("data/apparels_websites_pivot_by_sites.csv")


## TODO table of SimilarWeb app usages
apps = pd.read_csv("data/SimilarWeb/apps_19_7.csv")
apps.to_pickle("data/SimilarWeb/apps_19_7.pkl")
apps = pd.read_pickle("data/SimilarWeb/apps_19_7.pkl")
app_tickers_mapping = pd.read_excel(xls, "Apps")
app_tickers_mapping['app'] = app_tickers_mapping['App ID']

apps_ticker = pd.merge(apps,
                 app_tickers_mapping[['app', 'Ticker']],
                 on='app',
                 how='left')

similarWeb_tickers_list = ["ASC LN", "BOO LN","BKNG US","EBAY US","ETSY US","GPS US","HD US","JWN US","TJX US",
                                "URBN US","W US","ZAL GR","ATVI US","EA US"]

apps_ticker = apps_ticker[apps_ticker['Ticker'].isin(apparels_similarWeb_tickers_list)]

apps_ticker.date = "20" + apps_ticker.year.map(str) + "-" + apps_ticker.month.map(str) + "-" + apps_ticker.day.map(str)
apps_ticker.index = pd.to_datetime(apps_ticker.date)
apps_ticker = apps_ticker.drop(["app", "app_name", "country_name","year","month","day"], axis=1)
apps_pivot_by_apps = apps_ticker.pivot_table(index=apps_ticker.index, values=list(set(apps_ticker.columns).difference(set(["Ticker"]))), columns=["Ticker"])
apps_pivot_by_apps.columns = apps_pivot_by_apps.columns.to_series().str.join('_')
apps_pivot_by_apps.to_pickle("data/apparels_apps_pivot_by_apps.pkl")
apps_pivot_by_apps.to_csv("data/apparels_apps_pivot_by_apps.csv")

'''
for table in fundamentals_clean(fundamentals_files):
    table.sort_index()
    print(table)
'''


#[print(x.head) for x in consensus_clean(consensus_files)]
# REIT_price abd visits are daily data while distance and demographics are monthly data(both requires pivoting to join on YYYY-MM)

## let's first consider all tables in the model
#print(visits.columns, visits.head)
## TODO pivot by area before resample and change date to sta
'''

list(set(item_list).difference(set(list_to_remove)))


visits.date = visits["date"].map(lambda x: x.strftime('%Y-%m-%d'))
visits.index = pd.to_datetime(visits.date)
visits.columns = visits.columns.map(lambda x: "visits_" + x)
visits_pivot_by_area = visits.pivot_table(index=visits.index, values=['visits_avg_dwell_mins','visits_visits'], columns=["visits_area"], aggfunc=np.sum)
visits_pivot_by_area.columns = visits_pivot_by_area.columns.to_series().str.join('_')
print(visits_pivot_by_area)
visits_pivot_by_area_monthly = visits_pivot_by_area.resample('1M').sum()

print(visits_pivot_by_area_monthly)


REIT_price.date = REIT_price["Date"].map(lambda x: x.strftime('%Y-%m-%d'))
REIT_price.index = pd.to_datetime(REIT_price.date)
REIT_price.columns = REIT_price.columns.map(lambda x: "price_" + x)
REIT_price_monthly = REIT_price.resample('1M').first()
REIT_price_monthly.index = pd.to_datetime(REIT_price_monthly.price_Date)
## since we are missing 2018-10-01 to 2018-12-01 data for distance and demographics. We'll drop the correspondings in price and visits
REIT_price_monthly_dropped = REIT_price_monthly.loc[(REIT_price_monthly.index < "2018-09-30") |(REIT_price_monthly.index > "2018-12-30")]
#print(REIT_price_monthly_dropped)
REIT_price_monthly_dropped['monthly_lookahead_ret'] = np.log(REIT_price_monthly_dropped['price_Adj Close']).diff(1).shift(-1)

bin = np.sign(REIT_price_monthly_dropped['monthly_lookahead_ret'])
#print(bin)



distance.month = distance["month"].map(lambda x: x[0:4] + "-" + x[5:])
distance.month = pd.to_datetime(distance["month"])
distance.index = distance.month
distance.columns = distance.columns.map(lambda x: "distance_" + x)
distance_travelled_pivot_by_area = distance.pivot_table(index=['distance_month'], values=['distance_avg_distance_travelled'], columns=["distance_area"], aggfunc=np.sum)
distance_percent_visits_pivot_by_area = distance.pivot_table(index=['distance_month'], values=['distance_%visitors'], columns=["distance_area"], aggfunc=np.average)
distance_travelled_pivot_by_borough = distance.pivot_table(index=['distance_month'], values=['distance_avg_distance_travelled'], columns=["distance_borough"], aggfunc=np.sum)
distance_percent_visits_pivot_by_borough = distance.pivot_table(index=['distance_month'], values=['distance_%visitors'], columns=["distance_borough"], aggfunc=np.average)

#print(distance_travelled_pivot_by_area, distance_percent_visits_pivot_by_area, distance_travelled_pivot_by_borough, distance_percent_visits_pivot_by_borough)

demographics.month = demographics["month"].map(lambda x: x[0:4] + "-" + x[5:])
demographics.month = pd.to_datetime(demographics["month"])
demographics.index = demographics.month
demographics.columns = demographics.columns.map(lambda x: "demographics_" + x)
demographics_percent_visitors_pivot_by_area = demographics.pivot_table(index=['demographics_month'], values=['demographics_%visitors'], columns=["demographics_area", "demographics_demo_group"], aggfunc=np.average)
demographics_percent_visitors_pivot_by_area.columns = demographics_percent_visitors_pivot_by_area.columns.to_flat_index()
demographics_percent_visitors_pivot_by_area.columns = [reduce(operator.add, tup) for tup in demographics_percent_visitors_pivot_by_area.columns]

#print(demographics_percent_visitors_pivot_by_area)

'''