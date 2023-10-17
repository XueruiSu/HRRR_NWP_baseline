import xarray as xr
import os
import time
from var_dict import PRESSURE_VARS, SURFACE_VARS
import re  
from datetime import datetime, timedelta  
import numpy as np
import pickle  
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable  
from torchmetrics.image import StructuralSimilarityIndexMeasure

# 1.取两个数据集的时间
    # 如何处理一天内最后一个时间点的数据，下一个时刻的数据在下一天的文件夹中。 -> ok
    # 遍历input的日期和小时，取对应的文件名。 -> ok
# 2.取两个数据集的变量
    # 变量名字要对齐: PRESSURE_VARS, SURFACE_VARS -> ok
    # 变量level要对齐: 暂时找不到level的对应关系，所以全做，后续再优化 -> ok
# 3.绘制两个数据集特定变量的input-true-pre的图片
    

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

def ploter(atmos_var_input_value, atmos_var_true_value, atmos_var_pre_value, 
           suptitle, file_name, 
           file_dir="/blob/weathers2_FNO/xuerui/Dual-Weather/project/weather_metrics_test/figure"):
    # 创建子图  
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))  
    # 绘制图像  
    im1 = ax1.imshow(atmos_var_input_value)  
    ax1.set_title(f"input_min{atmos_var_input_value.min():.2f}_max{atmos_var_input_value.max():.2f}")  
    im2 = ax2.imshow(atmos_var_true_value)  
    ax2.set_title(f"true_min{atmos_var_true_value.min():.2f}_max{atmos_var_true_value.max():.2f}")  
    im3 = ax3.imshow(atmos_var_pre_value)  
    ax3.set_title(f"pre_min{atmos_var_pre_value.min():.2f}_max{atmos_var_pre_value.max():.2f}")  
    # # 调整子图布局以适应colorbar  
    # divider = make_axes_locatable(ax3)  
    # cax = divider.append_axes("right", size="5%", pad=0.1)      
    # # 为三个子图添加共享的colorbar  
    # fig.colorbar(im3, cax=cax)      
    # 设置总标题和保存图像  
    plt.suptitle(suptitle)  
    plt.savefig(f"{file_dir}/{file_name}")  
    plt.close()  

def RMSE(target, pre):
    return np.sqrt(((pre - target)**2).mean())

def SSIM(target, pre):
    target = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)
    pre = torch.from_numpy(pre).unsqueeze(0).unsqueeze(0)
    ssim = StructuralSimilarityIndexMeasure(data_range=None)
    ssim_score = ssim(pre, target)
    return ssim_score.item()

start_date = "20200701"
end_date = "20200701"
day_list = generate_date_list(start_date, end_date)  
print(day_list)  
hour_list = ["00"]
# hour_list = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09",
#              "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
#              "20", "21", "22", "23"]
LV_SELECTION = {"lv_HTGL1": 10.0}
# PRESSURE_VARS = []
# SURFACE_VARS = ["TMP_P0_L103_GLC0", "TMP_P0_L1_GLC0", ]

loss_func = SSIM
loss_str = "SSIM"

for day in day_list:
    for hour in hour_list:
        t1 = time.time()
        loss_dict = {}
        file_name_input = f"{day}/hrrr.t{hour}z.wrfprsf00.grib2"
        file_name_true, file_name_pre = get_file_name(file_name_input)
        ds_true = xr.open_dataset(os.path.join(f"/blob/kmsw0eastau/data/hrrr/grib2/hrrr", 
                                        file_name_true), engine="pynio")
        ds_pre = xr.open_dataset(os.path.join(f"/blob/kmsw0eastau/data/hrrr/grib2/hrrr", 
                                        file_name_pre), engine="pynio")
        ds_input = xr.open_dataset(os.path.join(f"/blob/kmsw0eastau/data/hrrr/grib2/hrrr", 
                                        file_name_input), engine="pynio")
        for var in PRESSURE_VARS+SURFACE_VARS:
            if var in PRESSURE_VARS:
                atmos_var_true_value = ds_true[var].to_numpy()
                atmos_var_pre_value = ds_pre[var].to_numpy()
                atmos_var_input_value = ds_input[var].to_numpy()
                print(var, atmos_var_true_value.shape, atmos_var_pre_value.shape, atmos_var_input_value.shape)
                for level_index in range(atmos_var_pre_value.shape[0]):
                    nwp_loss = loss_func(atmos_var_true_value[level_index], atmos_var_pre_value[level_index])
                    persistent_loss = loss_func(atmos_var_true_value[level_index], atmos_var_input_value[level_index])
                    loss_dict[f"{var}_{level_index}_hrrr_forecast_mse"] = nwp_loss
                    loss_dict[f"{var}_{level_index}_hrrr_base_mse"] = persistent_loss
                    # 绘制1排3列的图像  
                    ploter(atmos_var_input_value[level_index], 
                           atmos_var_true_value[level_index], 
                           atmos_var_pre_value[level_index],
                           f"{day}_{hour}_{var}_{level_index}_{loss_str} nwp_loss:{nwp_loss:.4f} persistent_loss:{persistent_loss:.4f}",
                           f"{day}_{hour}_{var}_{level_index}_{loss_str}.png")
            else:
                if var == "UGRD_P0_L103_GLC0" or var == "VGRD_P0_L103_GLC0":
                    atmos_var_true_value = ds_true[var].sel(LV_SELECTION).to_numpy()
                    atmos_var_pre_value = ds_pre[var].sel(LV_SELECTION).to_numpy()
                    atmos_var_input_value = ds_input[var].sel(LV_SELECTION).to_numpy()
                else:
                    atmos_var_true_value = ds_true[var].to_numpy()
                    atmos_var_pre_value = ds_pre[var].to_numpy()
                    atmos_var_input_value = ds_input[var].to_numpy()
                print(var, atmos_var_true_value.shape, atmos_var_pre_value.shape, atmos_var_input_value.shape)
                nwp_loss = loss_func(atmos_var_true_value, atmos_var_pre_value)
                persistent_loss = loss_func(atmos_var_true_value, atmos_var_input_value)
                loss_dict[f"{var}_hrrr_forecast_mse"] = nwp_loss
                loss_dict[f"{var}_hrrr_base_mse"] = persistent_loss
                # 绘制1排3列的图像  
                ploter(atmos_var_input_value, atmos_var_true_value, atmos_var_pre_value,
                       f"{day}_{hour}_{var}_{loss_str} nwp_loss:{nwp_loss:.2f} persistent_loss:{persistent_loss:.2f}",
                       f"{day}_{hour}_{var}_{loss_str}.png")
        t2 = time.time()
        print("day:", file_name_input, "time:", t2-t1) # 计算一条数据上所有的loss所需的时间。                     
        ds_true.close()
        ds_pre.close()
        ds_input.close()

with open(f"./Loss_file/loss_dict.pkl", "wb") as f:  
    pickle.dump(loss_dict, f)  
            


