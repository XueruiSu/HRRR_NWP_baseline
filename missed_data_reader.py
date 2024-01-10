import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import pyplot as plt
import pickle
from datetime import datetime, timedelta  
import torch
import os

def generate_date_list(start_date, end_date):  
    start_date_obj = datetime.strptime(start_date, "%Y%m%d")  
    end_date_obj = datetime.strptime(end_date, "%Y%m%d")  
    date_list = []  
    current_date_obj = start_date_obj  
    while current_date_obj < end_date_obj:  
        date_str = current_date_obj.strftime("%Y%m%d")  
        date_list.append(date_str)  
        current_date_obj += timedelta(days=1)  
    return date_list 

directory = "/weather-blob/kms1/hrrr/hourly_new_pre_var"
# 获取目录下的所有文件和文件夹名  
filenames = os.listdir(directory)  
file_names = [name for name in filenames if os.path.isfile(os.path.join(directory, name)) and ".npy" in name] 
sorted_files = sorted(file_names, key=lambda x: x[:-4])  
# 打印结果  
print(file_names, len(file_names))  
torch.save(file_names, "hourly_new_pre_var-2.torch")
assert 1==444

file_names = torch.load("hourly_new_pre_var-2.torch")
print("hourly_new_pre_var file names", len(file_names)) 
# assert 1==2

start_date = "20190101"
end_date = "20210101"
day_list = generate_date_list(start_date, end_date)  
hour_list = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09",
             "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
             "20", "21", "22", "23"]
print("real num of day_list", len(day_list))
data_missed_list = []
all_num = 0
for day in day_list:
    for hour in hour_list:
        all_num += 1
        if f"{day}{hour}.npy" not in file_names:
            print(f"{day}{hour}.npy")
            data_missed_list.append(f"{day}{hour}.npy")
            
for data_missed in data_missed_list:
    print(data_missed)
    
print("data_missed_list", len(data_missed_list))
print("all num:", all_num)
torch.save(data_missed_list, "data_missed_list_fixed_t2m.torch")
np.savetxt("data_missed_list_fixed_t2m.txt", data_missed_list, fmt="%s")