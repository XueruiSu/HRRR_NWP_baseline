import gif
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import os
from matplotlib import cm
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

@gif.frame
def helper_plot_U(smoke_values_extracted, quiver=0, UV=1):
    try:
        if quiver == 0:
            # (1059, 1799, 2)
            plt.figure(figsize=(4.7, 8))
            # plotting the pressure field as a contour
            X_128, Y_128 = np.meshgrid(np.linspace(0, smoke_values_extracted.shape[0], smoke_values_extracted.shape[0]), 
                                    np.linspace(0, smoke_values_extracted.shape[1], smoke_values_extracted.shape[1]))
            print(X_128.shape, Y_128.shape, smoke_values_extracted.shape)
            cf = plt.contourf(X_128, Y_128, smoke_values_extracted[..., UV].T, alpha=0.5, cmap=cm.viridis) 
            cbar = plt.colorbar(cf)  
            # 设置colorbar的上下限  
            cbar.set_clim(0, 30)  
            plt.xlabel('X')
            plt.ylabel('Y')
            if UV == 0:
                plt.title("10m_u_component_of_wind(m/s)")
            else:
                plt.title("10m_v_component_of_wind(m/s)")
        else: # quiver=1
            # (1059, 1799, 2)
            plt.figure(figsize=(4.7, 8))
            # plotting the pressure field as a contour
            resolution = 50
            X_64, Y_64 = np.meshgrid(np.linspace(0, smoke_values_extracted.shape[0], smoke_values_extracted[::resolution,:].shape[0]), 
                                    np.linspace(0, smoke_values_extracted.shape[1], smoke_values_extracted[:,::resolution].shape[1]))
            print(X_64.shape, Y_64.shape, smoke_values_extracted[::resolution,::resolution].shape)
            # plt.contourf(X_128, Y_128, smoke_values_extracted, alpha=0.5, cmap=cm.viridis)  
            # plt.colorbar()
            # plotting the pressure field outlines
            # plt.contour(X_64, Y_64, smoke_values_extracted, cmap=cm.viridis)  
            # plotting velocity field
            plt.quiver(X_64, Y_64, 
                    smoke_values_extracted[::resolution,::resolution, 0], 
                    smoke_values_extracted[::resolution,::resolution, 1]) 
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title("quiver of 10m u(v) component of wind(m/s)")
    except:
        print("error")    
      

day_list = ["20200701", "20200702", "20200703"]
hour_list = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", 
             "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", 
             "20", "21", "22", "23"]
file_name_list = []
LV_SELECTION = {"lv_HTGL1": 10.0}

for day in day_list:
    for hour in hour_list:
        file_name_list.append(f"{day}/hrrr.t{hour}z.wrfprsf00.grib2")

def filter(fft_result, cut_frequency=0.5):
    # cut_frequency: 表示砍掉多少比例的频率
    fft_result_filtered = fft_result.clone()
    cut_frequency_x = int(fft_result.shape[0] * cut_frequency)
    cut_frequency_y = int(fft_result.shape[1] * cut_frequency)
    # 去中心
    # fft_result_filtered[cut_frequency_x:-cut_frequency_x, cut_frequency_y:-cut_frequency_y] = 0
    # 去两边
    # fft_result_filtered[:cut_frequency_x, :cut_frequency_y] = 0
    # fft_result_filtered[-cut_frequency_x:, -cut_frequency_y:] = 0
    # 去右边, 截掉多少比例的频率
    # fft_result_filtered[-cut_frequency_x:, -cut_frequency_y:] = 0
    # 留偶数项：
    fft_result_filtered[::2, ::2] = 0
    print(cut_frequency_x, cut_frequency_y, fft_result_filtered[cut_frequency_x:-cut_frequency_x, cut_frequency_y:-cut_frequency_y].shape)
    return fft_result_filtered, fft_result - fft_result_filtered

def SSIM(target, pre):
    target = target.unsqueeze(0).unsqueeze(0)
    pre = pre.unsqueeze(0).unsqueeze(0)
    ssim = StructuralSimilarityIndexMeasure(data_range=None)
    ssim_score = ssim(pre, target)
    return ssim_score.item()

load = False
if load:
    UV_data = []
    for file_name in file_name_list:
        print(file_name)
        ds_true = xr.open_dataset(os.path.join(f"/blob/kmsw0eastau/data/hrrr/grib2/hrrr", 
                                            file_name), engine="pynio")
        U = ds_true["UGRD_P0_L103_GLC0"].sel(LV_SELECTION).to_numpy() # (1059 ,1799)
        V = ds_true["VGRD_P0_L103_GLC0"].sel(LV_SELECTION).to_numpy() # (1059 ,1799)
        UV_data.append(np.stack([U, V], axis=-1))
        print(np.stack([U, V], axis=-1).shape)
    print(np.stack(UV_data, axis=0).shape)
    np.save("UV_data.npy", np.stack(UV_data, axis=0))
else:
    UV_data = np.load("UV_data.npy") # (3*24, 1059, 1799, 2)
    # print(UV_data[...,0] - UV_data[...,1])
# 想要先将(1059, 1799)的image做二维傅立叶变换，观察做变换后的数据结构
# 做滤波处理：
# 1. 观察重建RMSE随截断频率的变化
# 2. 只用偶数频率和只用奇数频率做重建的现象
    UV_data = torch.from_numpy(UV_data).float()
    fft_result = torch.fft.fft2(UV_data[0, ..., 0])  
    print("fft_result", fft_result.shape)  
    # 进行逆傅立叶变换  
    fft_result, res_fft_result = filter(fft_result, cut_frequency=0.5)
    ifft_result = torch.real(torch.fft.ifft2(fft_result))
    ifft_result_res = torch.real(torch.fft.ifft2(res_fft_result))
    print("ifft_result", ifft_result.shape)
    RMSE_loss = torch.sqrt(torch.mean((UV_data[0, ..., 0] - ifft_result)**2)).item()
    SSIM_loss = SSIM(UV_data[0, ..., 0], ifft_result)
    RMSE_loss_res = torch.sqrt(torch.mean((UV_data[0, ..., 0] - ifft_result_res)**2)).item()
    SSIM_loss_res = SSIM(UV_data[0, ..., 0], ifft_result_res)
    print("SSIM loss", SSIM_loss, "RMSE:", RMSE_loss, 
          "\nSSIM res:", SSIM_loss_res, "RMSE:", RMSE_loss_res)
    # 画图
    vmin = -20  
    vmax = 25  
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(UV_data[0, ..., 0].detach().cpu().numpy(), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(ifft_result.detach().cpu().numpy(), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(ifft_result_res.detach().cpu().numpy(), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.suptitle(f"SSIM:{SSIM_loss}, RMSE:{RMSE_loss}; SSIM_res:{SSIM_loss_res}, RMSE_res:{RMSE_loss_res}")
    plt.savefig("./figure/a_filter_plot_even_filter.png")
    
    # # RMSE-cut_fre plot:
    # calc_loss = False
    # cut_frequency_list = np.linspace(0.01, 0.50, 60)
    # if calc_loss:
    #     loss_dict = []
    #     for cut_frequency in cut_frequency_list:
    #         fft_result = torch.fft.fft2(UV_data[0, ..., 0])  
    #         print("fft_result", fft_result.shape)  
    #         # 进行逆傅立叶变换  
    #         fft_result, res_fft_result = filter(fft_result, cut_frequency=cut_frequency)
    #         ifft_result = torch.real(torch.fft.ifft2(fft_result))
    #         ifft_result_res = torch.real(torch.fft.ifft2(res_fft_result))
    #         print("ifft_result", ifft_result.shape)
    #         RMSE_loss = torch.sqrt(torch.mean((UV_data[0, ..., 0] - ifft_result)**2)).item()
    #         SSIM_loss = SSIM(UV_data[0, ..., 0], ifft_result)
    #         RMSE_loss_res = torch.sqrt(torch.mean((UV_data[0, ..., 0] - ifft_result_res)**2)).item()
    #         SSIM_loss_res = SSIM(UV_data[0, ..., 0], ifft_result_res)
    #         print("SSIM loss", SSIM_loss, "RMSE:", RMSE_loss, 
    #             "\nSSIM res:", SSIM_loss_res, "RMSE:", RMSE_loss_res)
    #         loss_dict.append(np.array([RMSE_loss, SSIM_loss, RMSE_loss_res, SSIM_loss_res]))
    #     print(np.array(loss_dict).shape)
    #     np.save("loss_dict_taichi_filter_center.npy", np.array(loss_dict))
    # else:    
    #     # loss_dict = np.load("loss_dict_taichi.npy") # (50, 4) 去右边
    #     loss_dict = np.load("loss_dict_taichi_filter_center.npy") # (50, 4) 去中间
    #     plt.figure(figsize=(8, 4))
    #     plt.subplot(1, 2, 1)
    #     plt.plot(cut_frequency_list, loss_dict[:, 0], label="RMSE")
    #     plt.plot(cut_frequency_list, loss_dict[:, 2], label="RMSE_res")
    #     plt.xlabel("cut_frequency")
    #     plt.ylabel("RMSE loss")
    #     plt.legend()
    #     plt.yscale("log")
    #     plt.subplot(1, 2, 2)
    #     plt.plot(cut_frequency_list, loss_dict[:, 1], label="SSIM")
    #     plt.plot(cut_frequency_list, loss_dict[:, 3], label="SSIM_res")
    #     plt.xlabel("cut_frequency")
    #     plt.ylabel("SSIM loss")
    #     plt.legend()
    #     plt.suptitle("Loss-cut_frequency plot")
    #     plt.savefig("./figure/loss_cut_frequency_plot_filter_center.png")
        
        
        
        
        
        
        