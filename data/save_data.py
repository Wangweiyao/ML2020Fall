import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math

from datetime import datetime, timedelta, date
import pandas as pd
from fmz import *

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)*2+1):
        seconds_delta = 3600 * 12 * n 
        yield start_date + timedelta(seconds=seconds_delta)

def generate_interval(symbol, platform, start_date, end_date, time_granualarity):
    from datetime import datetime, timedelta, date
    start_ = start_date.split("-")
    end_ = end_date.split("-")
    start_time = datetime(int(start_[0]), int(start_[1]), int(start_[2]), 0, 0, 0)
    end_time = datetime(int(end_[0]), int(end_[1]), int(end_[2]), 0, 0, 0)
    
    time_ptr_1 = start_time.strftime("%Y-%m-%d %H:%M:%S")
    time_ptr_2 = None
    result = []
    for single_date in daterange(start_time, end_time):
        try:
            time_ptr_2 = single_date.strftime("%Y-%m-%d %H:%M:%S")
            this_period_bars = get_bars('{}_{}'.format(symbol, platform), time_granualarity, start=time_ptr_1, end=time_ptr_2)
            result.append(this_period_bars)
            print("From "+time_ptr_1+" to "+time_ptr_2)
            print(this_period_bars)
            time_ptr_1 = time_ptr_2
        except:
            print("fail")
            pass
    return pd.concat(result)
        
def save_historical_data(symbol, platform, time_periods, time_granularity):
    for start_date, end_date in time_periods:
        bars = generate_interval(symbol, platform, start_date=start_date, end_date=end_date, time_granualarity=time_granualarity)

    save_path = "./data/{}_{}_latest.pkl".format(symbol, platform)
    bars.to_pickle(save_path)
    

platform = 'BITFINEX'#'HUOBI'
today = date.today().strftime("%Y-%m-%d")
time_periods = [('2019-1-1',today)]
#time_periods = [('2019-1-1','2019-1-2')]
time_granualarity ='1m'

# Parallelizing using Pool.apply()

import multiprocessing as mp

# Step 1: Init multiprocessing.Pool()
pool = mp.Pool(int(mp.cpu_count()/2))

# Step 2: `pool.apply` the `howmany_within_range()`
symbols = ['BTC_USD','ETH_USD','LTC_USD','BTH_USD','EOS_USD','XRP_USD']
results = [pool.apply(save_historical_data, args=(symbol, platform, time_periods, time_granualarity)) for symbol in symbols]

# Step 3: Don't forget to close
pool.close()    
