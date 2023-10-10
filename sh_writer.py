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
    # 保存npy的数据到原位置。

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
end_date = "20190221"
day_list = generate_date_list(start_date, end_date) 
hour_list = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09",
             "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
             "20", "21", "22", "23"]

# 定义要写入.sh文件的字符  
content = "#!/bin/bash\n"  # 这是脚本文件的起始行  
content += "echo 'Hello, World!'"  # 这是一个简单的echo命令  

# 打开文件进行写入，这里我们创建一个名为"example.sh"的文件  
with open("cp_20190221.sh", "w") as file:  
    for day in day_list:
        for hour in hour_list:
            file.write(f"cp /nfs/weather/hrrr/prs/hourly2/{day}{hour}.npy /blob/kmsw0eastau/data/hrrr/hourly2/\n")  

# 现在example.sh文件已经包含了我们定义的字符  
# 打开文件进行写入，这里我们创建一个名为"example.sh"的文件  
with open("rm_20190221.sh", "w") as file:  
    for day in day_list:
        for hour in hour_list:
            file.write(f"rm /blob/kmsw0eastau/data/hrrr/hourly2/{day}{hour}.npy \n") 