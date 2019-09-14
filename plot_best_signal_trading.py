import pandas as pd
import numpy as np
import pickle
from functools import reduce
import itertools
import matplotlib.pyplot as plt

cum_dollar_port = pickle.load(open("data/backtest/cum_dollar_port.pkl", 'rb'))
# cum_dollar_port.index = pd.to_datetime(cum_dollar_port.index)
port_stats = pd.read_csv("data/backtest/port_stats.csv")

top_port_names = port_stats[0:5]['port_name'].tolist()

top_ports_cum_dollar = cum_dollar_port[top_port_names]

# print(top_ports_cum_dollar.index)

top_ports_cum_dollar.plot(linewidth=0.5)
plt.legend(loc='lower right')
plt.title("Net Position of Top 10 Portfolios")
plt.show()