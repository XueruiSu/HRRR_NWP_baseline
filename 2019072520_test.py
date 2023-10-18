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
from tqdm import tqdm

var = "UGRD_P0_L100_GLC0"   

ds_input = xr.open_dataset(os.path.join(f"/blob/weathers2_FNO/xuerui/Dual-Weather/project/",
                            f"weather_metrics_test/hrrr.t04z.wrfprsf00.grib2"), engine="pynio")
print(ds_input.keys())
for key in ds_input.keys():
    if "UGRD" in key:
        print(key)
L103_var_grib2_value = ds_input[var].to_numpy()
print(L103_var_grib2_value.shape)

print(L103_var_grib2_value[7].mean())
