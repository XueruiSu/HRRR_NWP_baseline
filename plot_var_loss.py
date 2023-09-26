import torch
import numpy as np
from var_dict import PRESSURE_VARS, SURFACE_VARS, index_in_atmos_level_all, var_mapping, index_level_mapping
from matplotlib import pyplot as plt
import pickle

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
    fig, main_ax = plt.subplots(figsize=(width_figure, 3))  
    # 绘制散点图和误差条  
    main_ax.errorbar(x, y, yerr=(lower_err, upper_err), 
                     fmt='o', capsize=5, color='orange', 
                     ecolor='blue', zorder=3) 
    # 设置 x 轴标签  
    main_ax.set_xticks(x)  
    main_ax.set_xticklabels(x_str)  

    # # 为每个点绘制嵌入的柱状图  
    # for i, data in enumerate(distribution_data):  
    #     # 选择上限数据点（例如：第一个数据点）  
    #     point_data_coord_up = (x[i], y[i] + upper_err[i])  
    #     # 将数据坐标转换为显示坐标  
    #     point_display_coord_up = main_ax.transData.transform(point_data_coord_up)  
    #     # 将显示坐标转换为归一化的轴坐标  
    #     point_axes_coord_up = fig.transFigure.inverted().transform(point_display_coord_up)  
    #     # 选择下限数据点（例如：第一个数据点）  
    #     point_data_coord_down = (x[i], y[i] - lower_err[i])  
    #     # 将数据坐标转换为显示坐标  
    #     point_display_coord_down = main_ax.transData.transform(point_data_coord_down)  
    #     # 将显示坐标转换为归一化的轴坐标  
    #     point_axes_coord_down = fig.transFigure.inverted().transform(point_display_coord_down)  
    #     # 计算嵌入图的垂直位置和高度  
    #     width = point_axes_coord_up[1] - point_axes_coord_down[1] 
    #     height = 0.06
    #     # 创建嵌入的柱状图  
    #     inset_ax = fig.add_axes([point_axes_coord_down[0], point_axes_coord_down[1], height, width])  
    
    #     # 绘制朝右的柱状图  
    #     inset_ax.hist(data, bins=3, color='green', alpha=0.7, 
    #                   orientation='horizontal', zorder=2)  
    
    #     # 设置嵌入图边界和刻度  
    #     inset_ax.set_ylim(-3, 3)  
    #     inset_ax.set_xlim(0, 40)  
    #     inset_ax.set_xticks([])  
    #     inset_ax.set_yticks([])  
    #     inset_ax.axis('off')  # 移除坐标轴  
    # # 绘制散点图和误差条  
    # main_ax.errorbar(x, y, yerr=(lower_err, upper_err), 
    #                  fmt='o', capsize=5, color='blue', 
    #                  ecolor='purple', zorder=1) 
    plt.savefig(path_save)
    plt.show()  



def plot_based_acc_2_1_2(mean_dict: dict, variance_dict: dict): 
    '''
    loss_dict: {var_name: loss_value}
    var_name: f"{HRRR_var_name}_{level_index}_hrrr_{base/forecast}_mse_{mean/var}"
    '''
    # print(mean_dict)

    mean_dict_nwp, mean_dict_base = {}, {}
    std_dict_nwp, std_dict_base = {}, {}
    for var_name in PRESSURE_VARS:
        for level_index in index_in_atmos_level_all:        
            for var_str in mean_dict.keys():
                var_mapping_key = f"{var_mapping[var_name]}_{index_level_mapping[level_index]}"
                if var_str.startswith(f"{var_name}_{level_index}_hrrr_base_mse"):
                    mean_dict_base[var_mapping_key] = mean_dict[var_str]
                    std_dict_base[var_mapping_key] = (variance_dict[var_str] ** 0.5)
                elif var_str.startswith(f"{var_name}_{level_index}_hrrr_forecast_mse"):
                    mean_dict_nwp[var_mapping_key] = mean_dict[var_str]
                    std_dict_nwp[var_mapping_key] = (variance_dict[var_str] ** 0.5)
    for var_name in SURFACE_VARS:
        for var_str in mean_dict.keys():
            var_mapping_key = f"{var_mapping[var_name]}"
            if var_str.startswith(f"{var_name}_hrrr_base_mse"):
                mean_dict_base[var_mapping_key] = mean_dict[var_str]
                std_dict_base[var_mapping_key] = (variance_dict[var_str] ** 0.5)
            elif var_str.startswith(f"{var_name}_hrrr_forecast_mse"):
                mean_dict_nwp[var_mapping_key] = mean_dict[var_str]
                std_dict_nwp[var_mapping_key] = (variance_dict[var_str] ** 0.5)
                    
    level_mean, level_lower_err, level_upper_err = [], [], []
    level_mean_base, level_lower_err_base, level_upper_err_base = [], [], []
    for mean_dict_key in mean_dict.keys():
        level_mean.append(mean_dict_nwp[mean_dict_key])
        level_lower_err.append(std_dict_nwp[mean_dict_key])
        level_upper_err.append(std_dict_nwp[mean_dict_key])
        level_mean_base.append(mean_dict_base[mean_dict_key])
        level_lower_err_base.append(std_dict_base[mean_dict_key])
        level_upper_err_base.append(std_dict_base[mean_dict_key])

    path_save_nwp = "./figure_nwp.png"
    path_save_base = "./figure_base.png"
    plot_acc_figure(np.arange(1, 7), np.array(level_mean), 
                    np.array(level_lower_err), np.array(level_upper_err), 
                    mean_dict.keys(), path_save_nwp)
    plot_acc_figure(np.arange(1, 7), np.array(level_mean_base),
                    np.array(level_lower_err_base), np.array(level_upper_err_base),
                    mean_dict_base.keys(), path_save_base)

# 从文件中加载字典  
with open("mean_dict_400.pkl", "rb") as f:  
    mean_dict = pickle.load(f)  
with open("variance_dict_400.pkl", "rb") as f:
    variance_dict = pickle.load(f)
    
print(mean_dict.keys())
print(variance_dict.keys())
plot_based_acc_2_1_2(mean_dict, variance_dict)
