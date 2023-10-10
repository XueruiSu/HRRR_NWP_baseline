import xarray as xr
import os
import time
from var_dict import PRESSURE_VARS, SURFACE_VARS
import wandb  
import re  
from datetime import datetime, timedelta  
import numpy as np
import pickle  
import torch
from tqdm import tqdm

# 取hrrr.t00z.wrfprsf01.grib2中特定变量的值（预测值）
# 与hrrr.t01z.wrfprsf00.grib2中对应变量的值做loss。
# 1.取两个数据集的时间
    # 如何处理一天内最后一个时间点的数据，下一个时刻的数据在下一天的文件夹中。 -> ok
    # 遍历input的日期和小时，取对应的文件名。 -> ok
# 2.取npy和grib2数据中的温度变量
    # 变量名字要对齐: PRESSURE_VARS, SURFACE_VARS -> ok
    # 变量level要对齐: 暂时找不到level的对应关系，所以全做，后续再优化 -> ok
# 3.将grib2中的温度数据赋给npy中的温度变量
    # 保存npy的数据到新位置。

def generate_date_list(start_date, end_date):  
    start_date_obj = datetime.strptime(start_date, "%Y%m%d")  
    end_date_obj = datetime.strptime(end_date, "%Y%m%d")  
    date_list = []  
    current_date_obj = start_date_obj  
  
    while current_date_obj <= end_date_obj:  
        date_str = current_date_obj.strftime("%Y%m%d")  
        date_list.append(date_str)  
        current_date_obj += timedelta(days=1)  
  
    return date_list 
  
start_date = "20190101"
end_date = "20201231"
day_list = generate_date_list(start_date, end_date)  
hour_list = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09",
             "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
             "20", "21", "22", "23"]

var = "TMP_P0_L103_GLC0"   
print(day_list, hour_list, var) 

for day in day_list:
    for hour in hour_list:
        t1 = time.time()
        
        npy_file = np.load(f"/blob/kmsw0eastau/data/hrrr/hourly2/{day}{hour}.npy")
        
        file_name_input = f"{day}/hrrr.t{hour}z.wrfprsf00.grib2"
        ds_input = xr.open_dataset(os.path.join(f"/blob/kmsw0eastau/data/hrrr/grib2/hrrr", 
                                        file_name_input), engine="pynio")
        L103_var_grib2_value = ds_input[var].to_numpy()
        print(L103_var_grib2_value.shape, npy_file.shape) # (1059, 1799) (1059, 1799)
        
        npy_file[0, 1, :, :] = L103_var_grib2_value
        np.save(f"/blob/kmsw0eastau/data/hrrr/hourly2_fixed_TMP_L103/{day}{hour}.npy", npy_file)
        
        ds_input.close()
        
        t2 = time.time()
        print(f"day:{day}{hour} time:", t2-t1) # 计算一条数据上所有的loss所需的时间。






