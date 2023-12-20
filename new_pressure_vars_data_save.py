import os
import time
from var_dict import PRESSURE_VARS, SURFACE_VARS, var_mapping_hrrrlong_herbie, atmos_level_herbie_new
from datetime import datetime, timedelta  
import numpy as np
import pickle  
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
import sys
import argparse  
from herbie import Herbie
import random
import shutil  
import os  
  
# 带参数表示true，不带参数表示False.
def get_args(argv=None):  
    parser = argparse.ArgumentParser(description='Put your hyperparameters')  
    parser.add_argument('-leadtime', '--leadtime', default=0, type=int, help='leadtime')  
    parser.add_argument('-start_date', '--start_date', default="20190701", type=str, help='start_date')  
    parser.add_argument('-end_date', '--end_date', default="20190801", type=str, help='end_date')  
    parser.add_argument('-file_dir', '--file_dir', default="/weather-blob/kms1/hrrr", type=str, help='file_dir need to add a new folder')  
    return parser.parse_args(argv)  
  
# args parser  
args = get_args(sys.argv[1:])  
print(args)  

# 1.取一个数据集的时间
    # 遍历日期和小时，取对应的文件名。 -> ok
# 2.取这个数据集的特定变量
    # 变量名字要对齐: PRESSURE_VARS, SURFACE_VARS -> ok
    # 加入到dict里
# 3.保存数据
    
def get_file_name(file_name_input, leadtime=1):
    # file_name_input: ["20200701", "00"]
    # 解析日期字符串  
    date_obj = datetime.strptime(file_name_input[0]+file_name_input[1], "%Y%m%d%H")  
    date_obj_true = date_obj + timedelta(hours=leadtime) 
    file_name_true = [date_obj_true.strftime("%Y%m%d"), date_obj_true.strftime("%H")]
    file_name_pre = [date_obj.strftime("%Y%m%d"), date_obj.strftime("%H")]
    return file_name_true, file_name_pre
        
def generate_date_list(start_date, end_date):  
    start_date_obj = datetime.strptime(start_date, "%Y%m%d")  
    end_date_obj = datetime.strptime(end_date, "%Y%m%d")  
    date_list = []  
    current_date_obj = start_date_obj  
  
    while current_date_obj < end_date_obj:  # 给定每个月的第一天就可以
        date_str = current_date_obj.strftime("%Y%m%d")  
        date_list.append(date_str)  
        current_date_obj += timedelta(days=1)  
  
    return date_list 

start_date = args.start_date # "20200701"
end_date = args.end_date # "20200702"
file_dir = args.file_dir
lead_time = args.leadtime
day_list = generate_date_list(start_date, end_date)  
print(day_list)  
hour_list = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09",
             "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
             "20", "21", "22", "23"]

file_names_null_dict = []
file_names_null_num = 0
rand_str1 = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=20))

for day in day_list:
    for hour in hour_list:
        file_name = f"{day}{hour}"
        try:  
            H_input = Herbie(f"{day} {hour}:00", model="hrrr", product="prs", 
                                fxx=lead_time, save_dir=f"./{rand_str1}", overwrite=True)
            print(H_input.grib)
            all_commands_successful = True  
        except Exception as e:  
            print(f"An error occurred: {e}")  
            all_commands_successful = False  
        if all_commands_successful:
            t1 = time.time()
            data_all_var = []
            for var in SURFACE_VARS+PRESSURE_VARS:
                if var in SURFACE_VARS:
                    try:
                        input_array = H_input.xarray(f"{var_mapping_hrrrlong_herbie[var]}").to_array().values[0] # (1059, 1799)
                    except Exception as e:
                        # 删除文件夹及其所有内容  
                        shutil.rmtree(f"./{rand_str1}/hrrr/") 
                        input_array = H_input.xarray(f"{var_mapping_hrrrlong_herbie[var]}").to_array().values[0]         
                    data_all_var.append(input_array)
                    print(var)                    
                else:
                    for level_ in atmos_level_herbie_new:
                        try:
                            input_array = H_input.xarray(f"{var_mapping_hrrrlong_herbie[var]}:{level_} mb").to_array().values[0] # (1059, 1799)
                        except Exception as e:  
                            # 删除文件夹及其所有内容  
                            shutil.rmtree(f"./{rand_str1}/hrrr/") 
                            input_array = H_input.xarray(f"{var_mapping_hrrrlong_herbie[var]}:{level_} mb").to_array().values[0] # (1059, 1799)                    
                        data_all_var.append(input_array)
                        print(var, level_)
            data_all_var = np.array(data_all_var, dtype=np.float32)[np.newaxis, :] # (1, 69, 1059, 1799)
            np.save(f"{file_dir}/{file_name}.npy", data_all_var)
            t2 = time.time()
            print(f"day:{day}{hour}", "time:", t2-t1, data_all_var.shape)              
        else:
            print(f"day:{day}{hour} not in file_names")
            file_names_null_dict.append([f"{day}{hour}", file_name])
            file_names_null_num += 1
            if file_names_null_num % 5 == 0:
                torch.save(file_names_null_dict, f"{file_dir}/herbie_null_dict_num{file_names_null_num}.torch")
            continue

if file_names_null_num != 0:
    torch.save(file_names_null_dict, f"{file_dir}/herbie_null_dict_num.torch")
    print("empty data exist, done!")
else:
    print("no empty data, done!")




