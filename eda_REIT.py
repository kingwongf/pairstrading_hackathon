import pandas as pd
import numpy as np
from functools import reduce
import operator
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', 199)  # or 199

xls = pd.ExcelFile("/Users/kingf.wong/Development/hackathon/data/Location Sciences - REIT insights.xlsx")

visits = pd.read_excel(xls, "visits")
distance = pd.read_excel(xls, "distance")
demographics = pd.read_excel(xls,"demo")
REIT_price = pd.read_excel(xls,"SHB.L")


# REIT_price abd visits are daily data while distance and demographics are monthly data(both requires pivoting to join on YYYY-MM)

## let's first consider all tables in the model
#print(visits.columns, visits.head)
## TODO pivot by area before resample and change date to sta

visits.date = visits["date"].map(lambda x: x.strftime('%Y-%m-%d'))
visits.index = pd.to_datetime(visits.date)
visits.columns = visits.columns.map(lambda x: "visits_" + x)
visits_pivot_by_area = visits.pivot_table(index=visits.index, values=['visits_avg_dwell_mins','visits_visits'], columns=["visits_area"], aggfunc=np.sum)
visits_pivot_by_area.columns = visits_pivot_by_area.columns.to_series().str.join('_')
#print(visits_pivot_by_area)
visits_pivot_by_area_monthly = visits_pivot_by_area.resample('1M').sum()

print(visits_pivot_by_area_monthly)

'''
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