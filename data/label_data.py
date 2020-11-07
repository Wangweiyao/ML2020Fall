import pandas as pd
import time
import numpy as np


data_path = './data'
platform = 'BITFINEX'
symbols = ['BTC_USD','ETH_USD','LTC_USD','BTH_USD','EOS_USD','XRP_USD']

for symbol in symbols:

    pkl_file_path = data_path+'/'+symbol+'_'+platform+'_latest.pkl'
    data = pd.read_pickle(pkl_file_path)
    data = data.dropna()

    def label_data(threshold):
        count = 0
        start = time.time()

        label_name = "label_{}".format(str((100*threshold)))
        data[label_name] = np.nan
        not_labled_idx = [] # maintain a list that we have not checked off

        # we iterate through source data by rows
        for index, row in data.iterrows():
        
            # extract rows
            not_labled_df = data.loc[not_labled_idx, ['close']]

            # the price goes up more than threshold
            up_idx = not_labled_df[row['high']  > (1+threshold) * not_labled_df['close']].index.tolist()
            # the price goes down more than threshold
            down_idx = not_labled_df[row['low'] < (1-threshold) * not_labled_df['close']].index.tolist()
            
            for i in up_idx:
                data[label_name][i] = 1 # 1 for going up 
            for i in down_idx:
                data[label_name][i] = 0 # 0 for going up 
            #data.at[down_idx,'label'] = 0 # 1 for going down
            
            not_labled_idx = [left_idx for left_idx in not_labled_idx if (left_idx not in up_idx) and (left_idx not in down_idx)]
                
            not_labled_idx.append(index) # add current one

            count += 1
            if count % 1440 == 0:
                print("finish {} data {} with {} sec".format(symbol, count/1440, time.time()-start))
                
        return data


    # Parallelizing using Pool

    from multiprocessing import Pool

    thresholds = [0.005, 0.01, 0.02, 0.05]
    with Pool(4) as p:
        results = p.map(label_data, thresholds)
        
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        label_name = "label_{}".format(str((100*threshold)))
        data[label_name] = results[i][label_name]

    save_path = "./data/{}_{}_labled.pkl".format(symbol, platform)
    data.to_pickle(save_path)

