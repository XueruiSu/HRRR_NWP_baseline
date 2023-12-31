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

var = "TMP_P0_L103_GLC0"   

ds_input = xr.open_dataset(os.path.join(f"/blob/kmsw0eastau/data/hrrr/grib2/hrrr",
                            f"20190725/hrrr.t20z.wrfprsf00.grib2"), engine="pynio")
print(ds_input.keys())
for key in ds_input.keys():
    if "TMP" in key:
        print(key)
L103_var_grib2_value = ds_input[var].to_numpy()