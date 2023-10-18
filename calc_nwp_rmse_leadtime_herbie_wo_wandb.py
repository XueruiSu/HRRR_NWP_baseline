import os
import time
from var_dict import PRESSURE_VARS, SURFACE_VARS, var_mapping_hrrrlong_herbie, atmos_level_herbie
from datetime import datetime, timedelta  
import numpy as np
import pickle  
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
import sys
import argparse  
from herbie import Herbie
import random

# 带参数表示true，不带参数表示False.
def get_args(argv=None):  
    parser = argparse.ArgumentParser(description='Put your hyperparameters')  
    parser.add_argument('-leadtime', '--leadtime', default=1, type=int, help='leadtime')  
    parser.add_argument('-start_date', '--start_date', default="20190701", type=str, help='start_date')  
    parser.add_argument('-end_date', '--end_date', default="20190801", type=str, help='end_date')  
    return parser.parse_args(argv)  
  
# args parser  
args = get_args(sys.argv[1:])  
print(args)  

# 取hrrr.t00z.wrfprsf01.grib2中特定变量的值（预测值）
# 与hrrr.t01z.wrfprsf00.grib2中对应变量的值做loss。
# 1.取两个数据集的时间
    # 如何处理一天内最后一个时间点的数据，下一个时刻的数据在下一天的文件夹中。 -> ok
    # 遍历input的日期和小时，取对应的文件名。 -> ok
# 2.取两个数据集的变量
    # 变量名字要对齐: PRESSURE_VARS, SURFACE_VARS -> ok
# 3.计算两个数据集特定变量的loss
    # 从kms repo中取计算loss的函数 -> ok
    # atmos变量的loss要在每个level上计算。 -> ok
# 4.记录每个变量的loss，每个变量维护均值和方差
    # 迭代计算loss的方法，每次计算完loss，更新均值和方差。 -> ok
    
class mean_M2():
    def __init__(self, loss_dict: dict) -> None:
        self.mean_dict = {}
        self.M2_dict = {}
        self.variance = {}
        self.n = 0
        for key in loss_dict.keys():
            self.mean_dict[key] = 0
            self.M2_dict[key] = 0
    
    def calc_mean_M2(self, loss_dict: dict):
        print("calc_mean_M2:", self.n)
        self.n += 1  
        for key in loss_dict.keys():
            delta = loss_dict[key] - self.mean_dict[key]  
            self.mean_dict[key] += delta / self.n  
            delta2 = loss_dict[key] - self.mean_dict[key]  
            self.M2_dict[key] += delta * delta2
    
    def output_mean_M2(self):
        if self.n < 2:  
            print("exp number is less than 2")
            return float('nan'), float('nan')  
        else:
            for key in self.mean_dict.keys():
                self.variance[key] = self.M2_dict[key] / (self.n - 1)
            return self.mean_dict, self.variance  

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

def RMSE(target, pre):
    return np.sqrt(((pre - target)**2).mean())

def RMSE_cut(target, pre, cut_len=60):
    SE = (pre - target)**2
    return np.sqrt((SE[cut_len:-cut_len, cut_len:-cut_len]).mean())

def SSIM(target, pre):
    target = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)
    pre = torch.from_numpy(pre).unsqueeze(0).unsqueeze(0)
    ssim = StructuralSimilarityIndexMeasure(data_range=None)
    ssim_score = ssim(pre, target)
    return ssim_score.item()

start_date = args.start_date # "20200701"
end_date = args.end_date # "20200801"
day_list = generate_date_list(start_date, end_date)  
print(day_list)  
hour_list = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09",
             "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
             "20", "21", "22", "23"]

loss_func = RMSE_cut

file_dir = "/blob/weathers2_FNO/xuerui/Dual-Weather/project/weather_metrics_test"
file_name = f"{start_date}_to_{end_date}_lt{args.leadtime}"
directory = f"{file_dir}/Nwp_loss_file/{file_name}"
os.makedirs(directory, exist_ok=True)

first_time = True
file_names_null_dict = []
file_names_null_num = 0
rand_str1 = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=20))
rand_str2 = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=20))
rand_str3 = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=20))

for day in day_list:
    for hour in hour_list:
        t1 = time.time()
        file_name_input = [day, hour]
        file_name_true, file_name_pre = get_file_name(file_name_input, leadtime=args.leadtime)
        try:  
            H_input = Herbie(f"{file_name_input[0]} {file_name_input[1]}:00", model="hrrr", product="prs", 
                             fxx=0, save_dir=f"~/{rand_str1}", overwrite=True)
            H_true = Herbie(f"{file_name_true[0]} {file_name_true[1]}:00", model="hrrr", product="prs", 
                            fxx=0, save_dir=f"~/{rand_str2}", overwrite=True)
            H_pre = Herbie(f"{file_name_pre[0]} {file_name_pre[1]}:00", model="hrrr", product="prs", 
                           fxx=args.leadtime, save_dir=f"~/{rand_str3}", overwrite=True)
            all_commands_successful = True  
        except Exception as e:  
            print(f"An error occurred: {e}")  
            all_commands_successful = False  
        if all_commands_successful:
            loss_dict = {}
            for var in PRESSURE_VARS+SURFACE_VARS:
                if var in PRESSURE_VARS:
                    for level_ in atmos_level_herbie:
                        try
                        input_array = H_input.xarray(f"{var_mapping_hrrrlong_herbie[var]}:{level_} mb").to_array().values[0] # (1059, 1799)
                        true_array = H_true.xarray(f"{var_mapping_hrrrlong_herbie[var]}:{level_} mb").to_array().values[0] # (1059, 1799)
                        pre_array = H_pre.xarray(f"{var_mapping_hrrrlong_herbie[var]}:{level_} mb").to_array().values[0] # (1059, 1799)
                        loss_dict[f"{var}_{level_}_hrrr_forecast_mse"] = loss_func(true_array, pre_array)
                        loss_dict[f"{var}_{level_}_hrrr_base_mse"] = loss_func(true_array, input_array)
                        print(var, level_)
                else:
                    input_array = H_input.xarray(f"{var_mapping_hrrrlong_herbie[var]}").to_array().values[0] # (1059, 1799)
                    true_array = H_true.xarray(f"{var_mapping_hrrrlong_herbie[var]}").to_array().values[0] # (1059, 1799)
                    pre_array = H_pre.xarray(f"{var_mapping_hrrrlong_herbie[var]}").to_array().values[0] # (1059, 1799)                    
                    loss_dict[f"{var}_hrrr_forecast_mse"] = loss_func(true_array, pre_array)
                    loss_dict[f"{var}_hrrr_base_mse"] = loss_func(true_array, input_array)
                    print(var)                    
            t2 = time.time()
            print("day:", file_name_input, file_name_true, file_name_pre, "time:", t2-t1) # 计算一条数据上所有的loss所需的时间。          
            # calc mean and variance
            if first_time:
                mean_M2_dict = mean_M2(loss_dict)
                mean_M2_dict.calc_mean_M2(loss_dict)
                first_time = False
            else:
                mean_M2_dict.calc_mean_M2(loss_dict)
            # record mean and variance
            if mean_M2_dict.n >= 2:
                mean_dict, variance_dict = mean_M2_dict.output_mean_M2()
                print("loss dict", mean_M2_dict.n, len(loss_dict), len(mean_dict), len(variance_dict))
            if mean_M2_dict.n % 150 == 0:            
                # 保存字典到文件  
                torch.save(mean_M2_dict, f"{directory}/mean_dict_lt{args.leadtime}_{str(mean_M2_dict.n)}.torch")
                with open(f"{directory}/mean_dict_lt{args.leadtime}_{str(mean_M2_dict.n)}.pkl", "wb") as f:  
                    pickle.dump(mean_dict, f)  
                with open(f"{directory}/variance_dict_lt{args.leadtime}_{str(mean_M2_dict.n)}.pkl", "wb") as f:  
                    pickle.dump(variance_dict, f)              
        else:
            print(f"day:{day}{hour} not in file_names")
            file_names_null_dict.append([f"{day}{hour}", file_name_input, file_name_true, file_name_pre])
            file_names_null_num += 1
            if file_names_null_num % 20 == 0:
                torch.save(file_names_null_dict, f"{directory}/grib2_file_names_null_dict_num{file_names_null_num}.torch")
            continue

# 保存字典到文件
torch.save(mean_M2_dict, f"{directory}/mean_dict_lt{args.leadtime}_{str(mean_M2_dict.n)}.torch")
with open(f"{directory}/mean_dict_lt{args.leadtime}_{str(mean_M2_dict.n)}.pkl", "wb") as f:  
    pickle.dump(mean_dict, f)  
with open(f"{directory}/variance_dict_lt{args.leadtime}_{str(mean_M2_dict.n)}.pkl", "wb") as f:  
    pickle.dump(variance_dict, f)     



