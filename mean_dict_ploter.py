'''
1. 给定一个时间段，将该时间段内每天的特定leadtime区段的mean_dict和variance_dict计算出来读取进来
2. 将每天的同一个lead time下的mean_dict和variance_dict进行平均，得到一个平均的mean_dict和variance_dict
3. 以lead time为横坐标，特定变量的mean为纵坐标，variance为上下限，画出图像
4. 将persistent loss和nwp loss的mean和variance按照同样的方式画在同一张图上
5. 遍历所有变量绘制69张图像
'''
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import torch
from var_dict import PRESSURE_VARS, SURFACE_VARS, atmos_level2index_in_atmos_level_all, atmos_level_herbie
from matplotlib import pyplot as plt
import pickle
from datetime import datetime, timedelta  


def plot_acc_figure(x, y, lower_err, upper_err, x_str, path_save, width_figure=6):
    # 设置 Nature 风格的字体、颜色和线宽  
    plt.rcParams.update({  
        'font.size': 14,  
        'axes.labelsize': 16,  
        'xtick.labelsize': 14,  
        'ytick.labelsize': 14,  
        'axes.linewidth': 1,  
        'xtick.major.width': 1,  
        'ytick.major.width': 1,  
        'axes.grid': True,  
        'grid.alpha': 0.53  
    })
    # 创建主图  
    fig, main_ax = plt.subplots(figsize=(width_figure*2, 3))  
    # 绘制散点图和误差条  
    print("num of var:", x.shape, y.shape)
    main_ax.errorbar(x, y, yerr=(lower_err, upper_err), fmt='o', capsize=5, color='orange', 
                     ecolor='blue', zorder=3) 
    # 在每个数据点上标出数值的大小  
    for i, value in enumerate(y):  
        main_ax.annotate(f"{value:.3g}", (x[i], y[i]), textcoords="offset points", xytext=(0, -15), ha='center')  
  
    # 设置 x 轴标签  
    main_ax.set_xticks(x)  
    main_ax.set_xticklabels(x_str)  
    plt.yscale('log')
    plt.savefig(path_save)
    plt.show()  


def plot_acc_figure_double(x, y_base, y_nwp, x_str, path_save, width_figure=6):
    # 设置 Nature 风格的字体、颜色和线宽  
    plt.rcParams.update({  
        'font.size': 14,  
        'axes.labelsize': 16,  
        'xtick.labelsize': 14,  
        'ytick.labelsize': 14,  
        'axes.linewidth': 1,  
        'xtick.major.width': 1,  
        'ytick.major.width': 1,  
        'axes.grid': True,  
        'grid.alpha': 0.53  
    })
    # 创建主图  
    fig, main_ax = plt.subplots(figsize=(width_figure*2, 3))  
    # 绘制散点图和误差条  
    print("num of var:", x.shape, y_base.shape, y_nwp.shape)
    nwp_better = y_nwp - y_base < 0
    better_str = ["B", "G"]
    num_better = np.sum(nwp_better)
    main_ax.scatter(x, y_base, marker='o', color='orange', zorder=3, label="Persistent") 
    main_ax.scatter(x, y_nwp, marker='d', color='red', zorder=3, label=f"NWP") 
    # 在每个数据点上标出数值的大小  
    for i, value in enumerate(y_base):  
        main_ax.annotate(f"{value:.3g}", (x[i], y_base[i]), textcoords="offset points", 
                         xytext=(0, -15), ha='center', color='orange')  
    for i, value in enumerate(y_nwp):  
        main_ax.annotate(f"{value:.3g}_{better_str[nwp_better[i]]}", (x[i], y_nwp[i]), 
                         textcoords="offset points", 
                         xytext=(0, -30), ha='center', color='red')  
    # 设置 x 轴标签  
    main_ax.set_xticks(x)  
    main_ax.set_xticklabels(x_str)  
    plt.yscale('log')
    plt.legend()
    plt.title(f"RMSE_of_NWP Better_Num:{num_better}")
    plt.savefig(path_save)
    plt.show()  



def plot_based_acc_2_1_2(var_mapping, mean_dict: dict, variance_dict: dict, 
                         file_name: str): 
    '''
    loss_dict: {var_name: loss_value}
    var_name: f"{HRRR_var_name}_{level_index}_hrrr_{base/forecast}_mse_{mean/var}"
    '''
    # print(mean_dict)
    mean_dict_nwp, mean_dict_base = {}, {}
    std_dict_nwp, std_dict_base = {}, {}
    for var_name in PRESSURE_VARS:
        for level_ in atmos_level_herbie:
            for var_str in mean_dict.keys():
                if var_name in var_mapping.keys():
                    var_mapping_key = f"{var_mapping[var_name]}_{level_}"
                    if var_str.startswith(f"{var_name}_{level_}_hrrr_base_mse"):
                        mean_dict_base[var_mapping_key] = mean_dict[var_str]
                        std_dict_base[var_mapping_key] = (variance_dict[var_str] ** 0.5)
                    elif var_str.startswith(f"{var_name}_{level_}_hrrr_forecast_mse"):
                        mean_dict_nwp[var_mapping_key] = mean_dict[var_str]
                        std_dict_nwp[var_mapping_key] = (variance_dict[var_str] ** 0.5)
    for var_name in SURFACE_VARS:
        for var_str in mean_dict.keys():
            if var_name in var_mapping.keys():
                var_mapping_key = f"{var_mapping[var_name]}"
                if var_str.startswith(f"{var_name}_hrrr_base_mse"):
                    mean_dict_base[var_mapping_key] = mean_dict[var_str]
                    std_dict_base[var_mapping_key] = (variance_dict[var_str] ** 0.5)
                elif var_str.startswith(f"{var_name}_hrrr_forecast_mse"):
                    mean_dict_nwp[var_mapping_key] = mean_dict[var_str]
                    std_dict_nwp[var_mapping_key] = (variance_dict[var_str] ** 0.5)
    print("len of mean_dict_nwp:", len(mean_dict_nwp))
    print("len of mean_dict_base:", len(mean_dict_base))
      
    level_mean, level_lower_err, level_upper_err = [], [], []
    level_mean_base, level_lower_err_base, level_upper_err_base = [], [], []
    for mean_dict_key in mean_dict_nwp.keys():
        level_mean.append(mean_dict_nwp[mean_dict_key])
        level_lower_err.append(std_dict_nwp[mean_dict_key])
        level_upper_err.append(std_dict_nwp[mean_dict_key])
        level_mean_base.append(mean_dict_base[mean_dict_key])
        level_lower_err_base.append(std_dict_base[mean_dict_key])
        level_upper_err_base.append(std_dict_base[mean_dict_key])

    path_save_nwp = f"./figure/figure_nwp_{file_name}.png"
    path_save_base = f"./figure/figure_base_{file_name}.png"
    path_save_double = f"./figure/figure_double_{file_name}.png"
    plot_acc_figure(np.linspace(1, len(level_mean), len(level_mean)), 
                    np.array(level_mean), 
                    np.array(level_lower_err), np.array(level_upper_err), 
                    mean_dict_nwp.keys(), path_save_nwp, len(level_mean))
    plot_acc_figure(np.linspace(1, len(level_mean), len(level_mean)), 
                    np.array(level_mean_base), 
                    np.array(level_lower_err_base), np.array(level_upper_err_base), 
                    mean_dict_base.keys(), path_save_base, len(level_mean))
    plot_acc_figure_double(np.linspace(1, len(level_mean), len(level_mean)), 
                           np.array(level_mean_base), np.array(level_mean), 
                           mean_dict_base.keys(), path_save_double, len(level_mean))


def plot_different_lt(leadtime_list, mean_leadtime_plot_nwp, mean_leadtime_plot_persistent, 
                      std_leadtime_plot_nwp, std_leadtime_plot_persistent,
                      var_mapping_key, path_save_nwp):
    plt.figure(figsize=(5, 4))
    plt.plot(leadtime_list, mean_leadtime_plot_nwp, label="NWP Loss")
    plt.fill_between(leadtime_list, mean_leadtime_plot_nwp-std_leadtime_plot_nwp, 
                     mean_leadtime_plot_nwp+std_leadtime_plot_nwp, alpha=0.5)  
    plt.plot(leadtime_list, mean_leadtime_plot_persistent, label="Persistent Loss")
    plt.fill_between(leadtime_list, mean_leadtime_plot_persistent-std_leadtime_plot_persistent, 
                     mean_leadtime_plot_persistent+std_leadtime_plot_persistent, alpha=0.5)  
    plt.legend()
    plt.xlabel("Leadtime")
    plt.ylabel("RMSE")
    plt.title(f"{var_mapping_key}")
    plt.savefig(path_save_nwp)
    plt.close()


var_mapping = {
    "SPFH_P0_L100_GLC0": "q",
    "TMP_P0_L100_GLC0": "t",
    "UGRD_P0_L100_GLC0": "u",
    "VGRD_P0_L100_GLC0": "v",
    "HGT_P0_L100_GLC0": "hgtn",
    "UGRD_P0_L103_GLC0": "10u",
    "VGRD_P0_L103_GLC0": "10v",
    "TMP_P0_L103_GLC0": "2t",
    "MSLMA_P0_L101_GLC0": "msl",
}
print("choosen var name:", SURFACE_VARS + PRESSURE_VARS)

leadtime_list = [1, 2, 3, 4, 5, 6]
# 读取数据
mean_dict_all, variance_dict_all = [], []
for leadtime in leadtime_list:
    # 从文件中加载字典  
    with open(f"./Nwp_loss_file/202007to12/mean_dict_lt{leadtime}.pkl", "rb") as f:  
        mean_dict = pickle.load(f)  
    with open(f"./Nwp_loss_file/202007to12/variance_dict_lt{leadtime}.pkl", "rb") as f:
        variance_dict = pickle.load(f)
    mean_dict_all.append(mean_dict)
    variance_dict_all.append(variance_dict)    
    print(len(mean_dict), len(variance_dict))

# compare nwp loss with persistent loss based on one leadtime
# assert 1==2
# for index, leadtime in enumerate(leadtime_list):
#     mean_dict, variance_dict = mean_dict_all[index], variance_dict_all[index]
#     print("all var:", mean_dict.keys())
#     plot_based_acc_2_1_2(var_mapping, mean_dict, variance_dict, "all_var_LT"+str(leadtime))
#     for var_name_HRRR in var_mapping.keys():
#         var_mapping_single = {var_name_HRRR: var_mapping[var_name_HRRR]}
#         print("var_name_HRRR:", var_name_HRRR)
#         plot_based_acc_2_1_2(var_mapping_single, mean_dict, variance_dict, var_mapping[var_name_HRRR]+"_LT"+str(leadtime))

# compare nwp loss with persistent loss based on different leadtime
nwp_csv, persistent_csv = [], []
print("all var:", mean_dict.keys())
for var_name in SURFACE_VARS:
    var_mapping_key = f"{var_mapping[var_name]}"
    mean_leadtime_plot_persistent, std_leadtime_plot_persistent = [], []
    mean_leadtime_plot_nwp, std_leadtime_plot_nwp = [], []
    for mean_index, mean_dict_leadtime in enumerate(mean_dict_all):
        for var_str in mean_dict.keys():
            if var_str.startswith(f"{var_name}_hrrr_base_mse"):
                mean_leadtime_plot_persistent.append(mean_dict_leadtime[var_str])
                std_leadtime_plot_persistent.append((variance_dict_all[mean_index][var_str] ** 0.5))
            elif var_str.startswith(f"{var_name}_hrrr_forecast_mse"):
                mean_leadtime_plot_nwp.append(mean_dict_leadtime[var_str])
                std_leadtime_plot_nwp.append((variance_dict_all[mean_index][var_str] ** 0.5))
    path_save_nwp = f"./figure_diffLT/figure_nwp_{var_mapping_key}.png"
    print(path_save_nwp)
    plot_different_lt(leadtime_list, 
                      np.array(mean_leadtime_plot_nwp), np.array(mean_leadtime_plot_persistent), 
                      np.array(std_leadtime_plot_nwp), np.array(std_leadtime_plot_persistent), 
                      var_mapping_key, path_save_nwp)
    nwp_csv.append(np.array(mean_leadtime_plot_nwp))
    persistent_csv.append(np.array(mean_leadtime_plot_persistent))

for var_name in PRESSURE_VARS:
    for level_ in atmos_level_herbie:
        var_mapping_key = f"{var_mapping[var_name]}_{level_}"
        mean_leadtime_plot_persistent, std_leadtime_plot_persistent = [], []
        mean_leadtime_plot_nwp, std_leadtime_plot_nwp = [], []
        for mean_index, mean_dict_leadtime in enumerate(mean_dict_all):
            for var_str in mean_dict.keys():
                if var_str.startswith(f"{var_name}_{level_}_hrrr_base_mse"):
                    mean_leadtime_plot_persistent.append(mean_dict_leadtime[var_str])
                    std_leadtime_plot_persistent.append((variance_dict_all[mean_index][var_str] ** 0.5))
                elif var_str.startswith(f"{var_name}_{level_}_hrrr_forecast_mse"):
                    mean_leadtime_plot_nwp.append(mean_dict_leadtime[var_str])
                    std_leadtime_plot_nwp.append((variance_dict_all[mean_index][var_str] ** 0.5))
        path_save_nwp = f"./figure_diffLT/figure_nwp_{var_mapping_key}.png"
        print(path_save_nwp)
        plot_different_lt(leadtime_list, 
                    np.array(mean_leadtime_plot_nwp), np.array(mean_leadtime_plot_persistent), 
                    np.array(std_leadtime_plot_nwp), np.array(std_leadtime_plot_persistent), 
                    var_mapping_key, path_save_nwp)
        nwp_csv.append(np.array(mean_leadtime_plot_nwp))
        persistent_csv.append(np.array(mean_leadtime_plot_persistent))

nwp_csv_path = "./Nwp_loss_file/202007to12/nwp_loss.csv"
persistent_csv_path = "./Nwp_loss_file/202007to12/persistent_loss.csv"
np.savetxt(nwp_csv_path, np.array(nwp_csv), delimiter=',')
np.savetxt(persistent_csv_path, np.array(persistent_csv), delimiter=',')

