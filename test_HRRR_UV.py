from var_dict import PRESSURE_VARS, SURFACE_VARS
import xarray as xr
import os
import numpy as np

day = "20200801"
hour = "00"
file_name_input = f"{day}/hrrr.t{hour}z.wrfprsf00.grib2"
ds_input = xr.open_dataset(os.path.join(f"/blob/kmsw0eastau/data/hrrr/grib2/hrrr", 
                                        file_name_input), engine="pynio")

var = "UGRD_P0_L103_GLC0"   
atmos_var_input_value = ds_input[var].to_numpy()

LV_SELECTION = {"lv_HTGL1": 10.0}
print(ds_input[var].sel(LV_SELECTION).to_numpy().shape) # (1059, 1799)

print(atmos_var_input_value.shape) # (2, 1059, 1799)
print(atmos_var_input_value[0, 0, 0], atmos_var_input_value[1, 0, 0]) # -3.166336 -3.40065
print((atmos_var_input_value[0] - atmos_var_input_value[1]).mean()) # -0.19569597
print((atmos_var_input_value[0] - atmos_var_input_value[1]).max()) # 15.546814
print((atmos_var_input_value[0] - atmos_var_input_value[1]).min()) # -10.515686
print(np.abs((atmos_var_input_value[0] - atmos_var_input_value[1])).min()) # 0.015686035
