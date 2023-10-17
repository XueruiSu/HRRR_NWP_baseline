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

def RMSE(target, pre):
    return np.sqrt(((pre - target)**2).mean())

def SSIM(target, pre):
    target = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)
    pre = torch.from_numpy(pre).unsqueeze(0).unsqueeze(0)
    ssim = StructuralSimilarityIndexMeasure(data_range=None)
    ssim_score = ssim(pre, target)
    return ssim_score.item()

def split_and_reconstruct(image):  
    # 傅立叶变换  
    fft_image = torch.fft.fft2(image)  
  
    # 按照要求切分矩阵  
    tiles = [  
        [fft_image[:353, :599], fft_image[:353, 599:1200], fft_image[:353, 1200:]],  
        [fft_image[353:706, :599], fft_image[353:706, 599:1200], fft_image[353:706, 1200:]],  
        [fft_image[706:, :599], fft_image[706:, 599:1200], fft_image[706:, 1200:]]  
    ]  
  
    # 储存重建的图像  
    reconstructed_images = []  
  
    # 填充缺失的位置并进行傅立叶逆变换  
    for i, row in enumerate(tiles):  
        for j, tile in enumerate(row):  
            # 创建一个全零的复数矩阵，与原始傅立叶变换后的矩阵具有相同的尺寸  
            padded_tile = torch.zeros_like(fft_image)  
  
            # 计算水平方向上的填充位置  
            if j == 0:  
                col_start, col_end = 0, 599  
            elif j == 1:  
                col_start, col_end = 599, 1200  
            else:  
                col_start, col_end = 1200, 1799  
  
            # 将切分后的矩阵放入全零矩阵的相应位置  
            padded_tile[i*tile.shape[0]:(i+1)*tile.shape[0], col_start:col_end] = tile  
  
            # 傅立叶逆变换  
            reconstructed_image = torch.fft.ifft2(padded_tile)  
  
            # 仅保留实部  
            reconstructed_image = torch.real(reconstructed_image)  
  
            # 将重建的图像添加到列表中  
            reconstructed_images.append(reconstructed_image)  
  
    return reconstructed_images  

def plot_33fd(image, file_name):    
    image = torch.from_numpy(image).float()
    # 分割并重建图像  
    reconstructed_images = split_and_reconstruct(image)  
    # 输出重建图像的形状  
    RMSE_dict = []
    for i, img in enumerate(reconstructed_images):  
        RMSE_loss = torch.sqrt(torch.mean((image - img)**2)).item()
        print(f"重建图像 {i + 1} 的形状：{img.shape}, loss:", RMSE_loss)  
        RMSE_dict.append(RMSE_loss)
    # 创建 2x5 的子图  
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))  
    # 添加原始图像到子图  
    axes[0, 0].imshow(image)  
    axes[0, 0].set_title("origin")  
    axes[0, 0].axis('off')  
    # 将重建的图像添加到子图中  
    for i, img in enumerate(reconstructed_images):  
        row, col = (i + 1) // 5, (i + 1) % 5  
        axes[row, col].imshow(img)  
        axes[row, col].set_title(f"num:{i}, loss:{RMSE_dict[i]:.2f}")  
        axes[row, col].axis('off')  
    plt.savefig(f"./figure/{file_name}")


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
        ds_input = xr.open_dataset(os.path.join(f"/blob/kmsw0eastau/data/hrrr/grib2/hrrr", 
                                        file_name_input), engine="pynio")
        for var in PRESSURE_VARS+SURFACE_VARS:
            if var in PRESSURE_VARS:
                atmos_var_input_value = ds_input[var].to_numpy()
                print(var, atmos_var_input_value.shape)
                for level_index in range(atmos_var_input_value.shape[0]):
                    plot_33fd(atmos_var_input_value[level_index], f"33fd_{var}_{level_index}.png")
            else:
                if var == "UGRD_P0_L103_GLC0" or var == "VGRD_P0_L103_GLC0":
                    atmos_var_input_value = ds_input[var].sel(LV_SELECTION).to_numpy()
                else:
                    atmos_var_input_value = ds_input[var].to_numpy()
                print(var, atmos_var_input_value.shape)
                # 绘制1排3列的图像  
                plot_33fd(atmos_var_input_value, f"33fd_{var}.png")
        t2 = time.time()
        print("day:", file_name_input, "time:", t2-t1) # 计算一条数据上所有的loss所需的时间。                     
        ds_input.close()

            


