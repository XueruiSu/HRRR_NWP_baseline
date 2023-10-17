import torch
import numpy as np
from var_dict import PRESSURE_VARS, SURFACE_VARS, atmos_level, atmos_level2index_in_atmos_level_all
from matplotlib import pyplot as plt
import pickle

# 从文件中加载字典  
with open("/blob/kmsw0eastau/data/hrrr/Nwp_loss_file/mean_dict_lt1_4569.pkl", "rb") as f:  
    mean_dict = pickle.load(f)  
    
print("all var:", mean_dict.keys())
print("choosen var name:", SURFACE_VARS + PRESSURE_VARS)
print("choosen lavel:", atmos_level)

var_mapping = {
    "MSLMA_P0_L101_GLC0": "msl",
    "TMP_P0_L103_GLC0": "2t",
    "UGRD_P0_L103_GLC0": "10u",
    "VGRD_P0_L103_GLC0": "10v",
    "HGT_P0_L100_GLC0": "hgtn",
    "UGRD_P0_L100_GLC0": "u",
    "VGRD_P0_L100_GLC0": "v",
    "TMP_P0_L100_GLC0": "t",
    "SPFH_P0_L100_GLC0": "q",
}
# level to be choosen:
atmos_level = [
    5000, 10000, 15000, 20000, 25000, 30000, 40000,
    50000, 60000, 70000, 85000, 92500, 100000
]
index_in_atmos_level_all, index_level_mapping = atmos_level2index_in_atmos_level_all(atmos_level)
base_dict = []
nwp_dict = []
for var_name in SURFACE_VARS:
    for var_str in mean_dict.keys():
        if var_name in var_mapping.keys():
            if var_str.startswith(f"{var_name}_hrrr_base_mse"):
                print(var_str)
                base_dict.append(mean_dict[var_str])
            elif var_str.startswith(f"{var_name}_hrrr_forecast_mse"):
                print(var_str)
                nwp_dict.append(mean_dict[var_str])

for var_name in PRESSURE_VARS:
    for level_index in index_in_atmos_level_all: # chosse level
        for var_str in mean_dict.keys():
            if var_name in var_mapping.keys():
                if var_str.startswith(f"{var_name}_{level_index}_hrrr_base_mse"):
                    print(var_str)
                    base_dict.append(mean_dict[var_str])
                elif var_str.startswith(f"{var_name}_{level_index}_hrrr_forecast_mse"):
                    print(var_str)
                    nwp_dict.append(mean_dict[var_str])
            
base_dict = np.array(base_dict)
nwp_dict = np.array(nwp_dict)
print(base_dict.shape, nwp_dict.shape) # (69,) (69,)
loss_all_dict = np.concatenate([base_dict.reshape(base_dict.shape[0], 1), 
                                nwp_dict.reshape(nwp_dict.shape[0], 1)], axis=1)
print(loss_all_dict.shape) # (69, 2)
np.save("./Loss_file/loss_all_dict_6mouth.npy", loss_all_dict)

np.savetxt("./Loss_file/persistent_and_nwp_loss_6mouth.csv", loss_all_dict, delimiter=',')
