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
import sys
import argparse  

# 带参数表示true，不带参数表示False.
def get_args(argv=None):  
    parser = argparse.ArgumentParser(description='Put your hyperparameters')  
    parser.add_argument('-leadtime', '--leadtime', default=1, type=int, help='leadtime')  
    parser.add_argument('-start_date', '--start_date', default="20190701", type=str, help='start_date')  
    parser.add_argument('-reload_bool', '--reload_bool', action='store_true', help='reload_bool or not')  
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
            file_name_true = f"{new_date_str}/hrrr.t{(leadtime-1):02d}z.wrfprsf{prsf_value:02d}.grib2"
            file_name_pre = f"{new_date_str}/hrrr.t{t_value:02d}z.wrfprsf{prsf_value+leadtime:02d}.grib2"
        else:
            file_name_true = f"{date_str}/hrrr.t{t_value+leadtime:02d}z.wrfprsf{prsf_value:02d}.grib2"
            file_name_pre = f"{date_str}/hrrr.t{t_value:02d}z.wrfprsf{prsf_value+leadtime:02d}.grib2"
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
  
file_dir = "/blob/kmsw0eastau/data/hrrr"
start_date = args.start_date
end_date = "20201231"
day_list = generate_date_list(start_date, end_date)  
print(day_list)  
hour_list = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09",
             "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
             "20", "21", "22", "23"]
LV_SELECTION = {"lv_HTGL1": 10.0}

def RMSE(target, pre):
    return np.sqrt(((pre - target)**2).mean())

def RMSE_cut(target, pre, cut_len=60):
    SE = (pre - target)**2
    return np.sqrt((SE[cut_len:-cut_len, cut_len:-cut_len]).mean())

loss_func = RMSE_cut
os.environ["WANDB_API_KEY"] = "e7b8eb712ec5e4421e767376055ddfafb01432ca"
wandb.init(project=f"physical_baseline_lt{args.leadtime}", name=f"{start_date}_to_{end_date}_lt{args.leadtime}")  
first_time = True
reload_bool = args.reload_bool
file_names_null_dict = []
file_names_null_num = 0
for day in day_list:
    for hour in hour_list:
        t1 = time.time()
        file_name_input = f"{day}/hrrr.t{hour}z.wrfprsf00.grib2"
        file_name_true, file_name_pre = get_file_name(file_name_input, leadtime=args.leadtime)
        true_path = os.path.join(f"/blob/kmsw0eastau/data/hrrr/grib2/hrrr", file_name_true)
        pre_path = os.path.join(f"/blob/kmsw0eastau/data/hrrr/grib2/hrrr", file_name_pre)
        input_path = os.path.join(f"/blob/kmsw0eastau/data/hrrr/grib2/hrrr", file_name_input)
        if os.path.exists(true_path) and os.path.exists(pre_path) and os.path.exists(input_path):
            ds_true = xr.open_dataset(true_path, engine="pynio")
            ds_pre = xr.open_dataset(pre_path, engine="pynio")
            ds_input = xr.open_dataset(input_path, engine="pynio")
            loss_dict = {}
            wandb_log_dict = {}
            for var in PRESSURE_VARS+SURFACE_VARS:
                if var in PRESSURE_VARS:
                    atmos_var_true_value = ds_true[var].to_numpy()
                    atmos_var_pre_value = ds_pre[var].to_numpy()
                    atmos_var_input_value = ds_input[var].to_numpy()
                    print(var, atmos_var_true_value.shape, atmos_var_pre_value.shape, atmos_var_input_value.shape)
                    for level_index in range(atmos_var_pre_value.shape[0]):
                        loss_dict[f"{var}_{level_index}_hrrr_forecast_mse"] = loss_func(atmos_var_true_value[level_index], atmos_var_pre_value[level_index])
                        loss_dict[f"{var}_{level_index}_hrrr_base_mse"] = loss_func(atmos_var_true_value[level_index], atmos_var_input_value[level_index])
                        # 记录图像集合  
                        images = [wandb.Image(atmos_var_input_value[level_index]), 
                                wandb.Image(atmos_var_true_value[level_index]), 
                                wandb.Image(atmos_var_pre_value[level_index])]  
                        wandb_log_dict[f"{var}_{level_index}_input_true_pre"] = images
                elif var == "UGRD_P0_L103_GLC0" or var == "VGRD_P0_L103_GLC0":
                    atmos_var_true_value = ds_true[var].sel(LV_SELECTION).to_numpy()
                    atmos_var_pre_value = ds_pre[var].sel(LV_SELECTION).to_numpy()
                    atmos_var_input_value = ds_input[var].sel(LV_SELECTION).to_numpy()
                    print(var, atmos_var_true_value.shape, atmos_var_pre_value.shape, atmos_var_input_value.shape)
                    loss_dict[f"{var}_hrrr_forecast_mse"] = loss_func(atmos_var_true_value, atmos_var_pre_value)
                    loss_dict[f"{var}_hrrr_base_mse"] = loss_func(atmos_var_true_value, atmos_var_input_value) 
                    # 记录图像集合  
                    images = [wandb.Image(atmos_var_input_value), 
                                wandb.Image(atmos_var_true_value), 
                                wandb.Image(atmos_var_pre_value)]  
                    wandb_log_dict[f"{var}_input_true_pre"] = images 
                else:
                    atmos_var_true_value = ds_true[var].to_numpy()
                    atmos_var_pre_value = ds_pre[var].to_numpy()
                    atmos_var_input_value = ds_input[var].to_numpy()
                    print(var, atmos_var_true_value.shape, atmos_var_pre_value.shape, atmos_var_input_value.shape)
                    loss_dict[f"{var}_hrrr_forecast_mse"] = loss_func(atmos_var_true_value, atmos_var_pre_value)
                    loss_dict[f"{var}_hrrr_base_mse"] = loss_func(atmos_var_true_value, atmos_var_input_value) 
                    # 记录图像集合  
                    images = [wandb.Image(atmos_var_input_value), 
                                wandb.Image(atmos_var_true_value), 
                                wandb.Image(atmos_var_pre_value)]  
                    wandb_log_dict[f"{var}_input_true_pre"] = images
            t2 = time.time()
            print("day:", file_name_input, file_name_true, file_name_pre, "time:", t2-t1) # 计算一条数据上所有的loss所需的时间。          
            # calc mean and variance
            if first_time:
                mean_M2_dict = mean_M2(loss_dict)
                if reload_bool:
                    mean_M2_dict = torch.load(f"{file_dir}/Nwp_loss_file/mean_dict_lt1_3300.torch")
                mean_M2_dict.calc_mean_M2(loss_dict)
                first_time = False
            else:
                mean_M2_dict.calc_mean_M2(loss_dict)
            # record mean and variance
            for key in loss_dict.keys():
                if mean_M2_dict.n >= 2:
                    mean_dict, variance_dict = mean_M2_dict.output_mean_M2()
                    wandb_log_dict[f"{key}"] = loss_dict[key]
                    wandb_log_dict[f"{key}_mean"] = mean_dict[key]
                    wandb_log_dict[f"{key}_var"] = variance_dict[key]
            if mean_M2_dict.n >= 2:
                wandb.log(wandb_log_dict)
                print("loss dict", mean_M2_dict.n, len(loss_dict), len(mean_dict), len(variance_dict))
            if mean_M2_dict.n % 150 == 0:            
                # 保存字典到文件  
                torch.save(mean_M2_dict, f"{file_dir}/Nwp_loss_file/mean_dict_lt{args.leadtime}_{str(mean_M2_dict.n)}.torch")
                with open(f"{file_dir}/Nwp_loss_file/mean_dict_lt{args.leadtime}_{str(mean_M2_dict.n)}.pkl", "wb") as f:  
                    pickle.dump(mean_dict, f)  
                with open(f"{file_dir}/Nwp_loss_file/variance_dict_lt{args.leadtime}_{str(mean_M2_dict.n)}.pkl", "wb") as f:  
                    pickle.dump(variance_dict, f)              
            ds_true.close()
            ds_pre.close()
            ds_input.close()
        else:
            print(f"day:{day}{hour} not in file_names")
            file_names_null_dict.append([f"{day}{hour}", file_name_input, file_name_true, file_name_pre])
            file_names_null_num += 1
            if file_names_null_num % 20 == 0:
                torch.save(file_names_null_dict, f"{file_dir}/Nwp_loss_file/grib2_file_names_null_dict_num{file_names_null_num}.torch")
            continue

# 保存字典到文件
torch.save(mean_M2_dict, f"{file_dir}/Nwp_loss_file/mean_dict_lt{args.leadtime}_{str(mean_M2_dict.n)}.torch")
with open(f"{file_dir}/Nwp_loss_file/mean_dict_lt{args.leadtime}_{str(mean_M2_dict.n)}.pkl", "wb") as f:  
    pickle.dump(mean_dict, f)  
with open(f"{file_dir}/Nwp_loss_file/variance_dict_lt{args.leadtime}_{str(mean_M2_dict.n)}.pkl", "wb") as f:  
    pickle.dump(variance_dict, f)     



