import time
from var_dict import PRESSURE_VARS, SURFACE_VARS, var_mapping_hrrrlong_herbie, atmos_level_herbie
from datetime import datetime, timedelta  
import numpy as np
import torch
from herbie import Herbie
import random
import shutil  
import os  
import sys
import argparse  

def get_args(argv=None):  
    parser = argparse.ArgumentParser(description='Put your hyperparameters')  
    parser.add_argument('-start_date', '--start_date', default=5, type=int, help='start_date')  
    parser.add_argument('-end_date', '--end_date', default=70, type=int, help='end_date')  
    return parser.parse_args(argv)  
# args parser  
args = get_args(sys.argv[1:])  
print(args)  

rand_str1 = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=20))
os.makedirs(f"/home/v-xueruisu/{rand_str1}", exist_ok=True)

file_dir = "/blob/weathers2_FNO/xuerui/Dual-Weather/project/weather_metrics_test/hourly2_data_completion"
file_names_null_dict = []
file_names_null_num = 0
file_names = torch.load("data_missed_list_fixed_t2m.torch")
file_names = file_names[args.start_date:args.end_date]
for index, file_name in enumerate(file_names):
    t1 = time.time()
    day = file_name[0:8]
    hour = file_name[8:10]
    try:  
        H_input = Herbie(f"{day} {hour}:00", model="hrrr", product="prs", 
                            fxx=0, save_dir=f"/home/v-xueruisu/{rand_str1}", overwrite=True)
        print(H_input.grib)
        all_commands_successful = True  
    except Exception as e:  
        print(f"An error occurred: {e}")  
        all_commands_successful = False  
    if all_commands_successful:
        data_all_var = []
        for var in SURFACE_VARS+PRESSURE_VARS:
            if var in SURFACE_VARS:
                try:
                    input_array = H_input.xarray(f"{var_mapping_hrrrlong_herbie[var]}").to_array().values[0] # (1059, 1799)
                except Exception as e:
                    # 删除文件夹及其所有内容  
                    shutil.rmtree(f"/home/v-xueruisu/{rand_str1}/hrrr/") 
                    input_array = H_input.xarray(f"{var_mapping_hrrrlong_herbie[var]}").to_array().values[0]         
                data_all_var.append(input_array)
                print(var)                    
            else:
                for level_ in atmos_level_herbie:
                    try:
                        input_array = H_input.xarray(f"{var_mapping_hrrrlong_herbie[var]}:{level_} mb").to_array().values[0] # (1059, 1799)
                    except Exception as e:  
                        # 删除文件夹及其所有内容  
                        shutil.rmtree(f"/home/v-xueruisu/{rand_str1}/hrrr/") 
                        input_array = H_input.xarray(f"{var_mapping_hrrrlong_herbie[var]}:{level_} mb").to_array().values[0] # (1059, 1799)                    
                    data_all_var.append(input_array)
                    print(var, level_)
        data_all_var = np.array(data_all_var, dtype=np.float32)[np.newaxis, :]
        np.save(f"{file_dir}/{file_name}", data_all_var)
        t2 = time.time()
        print("index:", index, "day:", file_name, "time:", t2-t1, data_all_var.shape)              
    else:
        print("index:", index, f"day:{day}{hour} not in file_names")
        file_names_null_dict.append([index, f"{day}{hour}", file_name])
        file_names_null_num += 1
        if file_names_null_num % 5 == 0:
            torch.save(file_names_null_dict, f"{file_dir}/herbie_null_dict_num{file_names_null_num}.torch")
        continue

if file_names_null_num != 0:
    torch.save(file_names_null_dict, f"{file_dir}/herbie_null_dict_num.torch")
    print("empty data exist, done!")
else:
    print("no empty data, done!")

