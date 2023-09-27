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

# 取hrrr.t00z.wrfprsf01.grib2中特定变量的值（预测值）
# 与hrrr.t01z.wrfprsf00.grib2中对应变量的值做loss。
# 1.取两个数据集的时间
    # 如何处理一天内最后一个时间点的数据，下一个时刻的数据在下一天的文件夹中。 -> ok
    # 遍历input的日期和小时，取对应的文件名。 -> ok
# 2.取两个数据集的变量
    # 变量名字要对齐: PRESSURE_VARS, SURFACE_VARS -> ok
    # 变量level要对齐: 暂时找不到level的对应关系，所以全做，后续再优化 -> ok
# 3.计算两个数据集特定变量的loss
    # 从kms repo中取计算loss的函数 -> ok
    # atmos变量的loss要在每个level上计算。 -> ok
# 4.记录每个变量的loss，每个变量维护均值和方差
    # 迭代计算loss的方法，每次计算完loss，更新均值和方差。 -> ok
    # 使用wandb记录当前的loss、均值和方差（每个都是两种）。 -> ok

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
        for key in loss_dict.keys():
            self.n += 1  
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

def get_file_name(file_name_input):
    # 定义正则表达式模式  
    pattern = r'(\d{8})/hrrr.t(\d{2})z.wrfprsf(\d{2}).grib2'  
    # 使用 re.search() 查找匹配项  
    match = re.search(pattern, file_name_input)  
    
    if match:  
        date_str = match.group(1)  
        t_value = int(match.group(2))  
        prsf_value = int(match.group(3))
        # 解析日期字符串  
        date_obj = datetime.strptime(date_str, "%Y%m%d")  
    
        # 如果 t 值等于 23，则将日期后延一天  
        if t_value == 23:  
            date_obj += timedelta(days=1)  
            # 将日期对象转换回字符串  
            new_date_str = date_obj.strftime("%Y%m%d") 
            file_name_true = f"{new_date_str}/hrrr.t00z.wrfprsf{prsf_value:02d}.grib2"
            file_name_pre = f"{new_date_str}/hrrr.t{t_value:02d}z.wrfprsf{prsf_value+1:02d}.grib2"
        else:
            file_name_true = f"{date_str}/hrrr.t{t_value+1:02d}z.wrfprsf{prsf_value:02d}.grib2"
            file_name_pre = f"{date_str}/hrrr.t{t_value:02d}z.wrfprsf{prsf_value+1:02d}.grib2"
    else:  
        print("File name wrong. No pre found!")  

    return file_name_true, file_name_pre

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
  
start_date = "20200630"
end_date = "20201231"
day_list = generate_date_list(start_date, end_date)  
print(day_list)  
hour_list = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09",
             "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
             "20", "21", "22", "23"]
LV_SELECTION = {"lv_HTGL1": 10.0}

wandb.init(project="physical_baseline", name="20200630_to_20201231")  
first_time = True
for day in day_list:
    for hour in hour_list:
        t1 = time.time()
        file_name_input = f"{day}/hrrr.t{hour}z.wrfprsf00.grib2"
        file_name_true, file_name_pre = get_file_name(file_name_input)
        ds_true = xr.open_dataset(os.path.join(f"/blob/kmsw0eastau/data/hrrr/grib2/hrrr", 
                                        file_name_true), engine="pynio")
        ds_pre = xr.open_dataset(os.path.join(f"/blob/kmsw0eastau/data/hrrr/grib2/hrrr", 
                                        file_name_pre), engine="pynio")
        ds_input = xr.open_dataset(os.path.join(f"/blob/kmsw0eastau/data/hrrr/grib2/hrrr", 
                                        file_name_input), engine="pynio")
        loss_dict = {}
        for var in PRESSURE_VARS+SURFACE_VARS:
            if var in PRESSURE_VARS:
                atmos_var_true_value = ds_true[var].to_numpy()
                atmos_var_pre_value = ds_pre[var].to_numpy()
                atmos_var_input_value = ds_input[var].to_numpy()
                print(var, atmos_var_true_value.shape, atmos_var_pre_value.shape, atmos_var_input_value.shape)
                for level_index in range(atmos_var_pre_value.shape[0]):
                    loss_dict[f"{var}_{level_index}_hrrr_forecast_mse"] = ((atmos_var_true_value[level_index] - atmos_var_pre_value[level_index])**2).mean()
                    loss_dict[f"{var}_{level_index}_hrrr_base_mse"] = ((atmos_var_true_value[level_index] - atmos_var_input_value[level_index])**2).mean()
            elif var == "UGRD_P0_L103_GLC0" or var == "VGRD_P0_L103_GLC0":
                atmos_var_true_value = ds_true[var].sel(LV_SELECTION).to_numpy()
                atmos_var_pre_value = ds_pre[var].sel(LV_SELECTION).to_numpy()
                atmos_var_input_value = ds_input[var].sel(LV_SELECTION).to_numpy()
                print(var, atmos_var_true_value.shape, atmos_var_pre_value.shape, atmos_var_input_value.shape)
                loss_dict[f"{var}_hrrr_forecast_mse"] = ((atmos_var_true_value - atmos_var_pre_value)**2).mean()
                loss_dict[f"{var}_hrrr_base_mse"] = ((atmos_var_true_value - atmos_var_input_value)**2).mean()
            else:
                atmos_var_true_value = ds_true[var].to_numpy()
                atmos_var_pre_value = ds_pre[var].to_numpy()
                atmos_var_input_value = ds_input[var].to_numpy()
                print(var, atmos_var_true_value.shape, atmos_var_pre_value.shape, atmos_var_input_value.shape)
                loss_dict[f"{var}_hrrr_forecast_mse"] = ((atmos_var_true_value - atmos_var_pre_value)**2).mean()
                loss_dict[f"{var}_hrrr_base_mse"] = ((atmos_var_true_value - atmos_var_input_value)**2).mean()
        t2 = time.time()
        print("day:", file_name_input, "time:", t2-t1) # 计算一条数据上所有的loss所需的时间。          
        # calc mean and variance
        if first_time:
            mean_M2_dict = mean_M2(loss_dict)
            mean_M2_dict.calc_mean_M2(loss_dict)
            first_time = False
        else:
            mean_M2_dict.calc_mean_M2(loss_dict)
        # record mean and variance
        wandb_log_dict = {}
        for key in loss_dict.keys():
            if mean_M2_dict.n >= 2:
                mean_dict, variance_dict = mean_M2_dict.output_mean_M2()
                wandb_log_dict[f"{key}"] = loss_dict[key]
                wandb_log_dict[f"{key}_mean"] = mean_dict[key]
                wandb_log_dict[f"{key}_var"] = variance_dict[key]
        wandb.log(wandb_log_dict)
        print("loss dict", len(loss_dict), len(mean_dict), len(variance_dict))
        if mean_M2_dict.n % 400 == 0:            
            # 保存字典到文件  
            torch.save(mean_M2_dict, f"./Loss_file/mean_dict_{str(mean_M2_dict.n)}.pt")
            with open(f"./Loss_file/mean_dict_{str(mean_M2_dict.n)}.pkl", "wb") as f:  
                pickle.dump(mean_dict, f)  
            with open(f"./Loss_file/variance_dict_{str(mean_M2_dict.n)}.pkl", "wb") as f:  
                pickle.dump(variance_dict, f)              
        ds_true.close()
        ds_pre.close()
        ds_input.close()



