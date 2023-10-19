'''
1. 给定一个时间段，将该时间段内每天的特定leadtime区段的mean_dict和variance_dict计算出来读取进来
2. 将每天的同一个lead time下的mean_dict和variance_dict进行平均，得到一个平均的mean_dict和variance_dict
'''
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import torch
from var_dict import PRESSURE_VARS, SURFACE_VARS, atmos_level, atmos_level2index_in_atmos_level_all, atmos_level_herbie
from matplotlib import pyplot as plt
import pickle
from datetime import datetime, timedelta  

leadtime = 5
leadtime_list = [1, 2, 3, 4, 5, 6]
empty_list = []


for leadtime in leadtime_list:
    start_day_all = "20200701"
    end_day_all = "20210101"
    start_day_time = datetime.strptime(start_day_all, "%Y%m%d")
    end_day_all_time = datetime.strptime(end_day_all, "%Y%m%d")
    n = 0
    while start_day_time < end_day_all_time:
        start_day = start_day_time.strftime("%Y%m%d")
        end_day_time = start_day_time + timedelta(days=1) 
        end_day = end_day_time.strftime("%Y%m%d")
        day_name = f"{start_day}_to_{end_day}_lt{leadtime}"
        try:
            # 从文件中加载字典  
            with open(f"./Nwp_loss_file/{day_name}/mean_dict_lt{leadtime}_24.pkl", "rb") as f:  
                mean_dict = pickle.load(f)  
            with open(f"./Nwp_loss_file/{day_name}/variance_dict_lt{leadtime}_24.pkl", "rb") as f:
                variance_dict = pickle.load(f)
            if start_day == start_day_all:
                mean_dict_all = mean_dict
                variance_dict_all = variance_dict
                n = 1
            else:
                for var_str in mean_dict.keys():
                    mean_dict_all[var_str] = (mean_dict_all[var_str]*n + mean_dict[var_str])/(n+1)
                    variance_dict_all[var_str] = (variance_dict_all[var_str]*n + variance_dict[var_str])/(n+1)
                n += 1
            print(day_name, n)
        except Exception as e:  
            empty_list.append(day_name)
            print(day_name, "no file!!!!!!!")
        start_day_time = end_day_time
    with open(f"./Nwp_loss_file/202007to12/mean_dict_lt{leadtime}.pkl", "wb") as f:  
        pickle.dump(mean_dict_all, f)  
    with open(f"./Nwp_loss_file/202007to12/variance_dict_lt{leadtime}.pkl", "wb") as f:  
        pickle.dump(variance_dict_all, f)     


# empty_list = np.load("./Nwp_loss_file/empty_list1019-1122.npy")
for empty_name in empty_list:
    print(empty_name)

np.save("./Nwp_loss_file/empty_list1019-1122.npy", empty_list)

    
