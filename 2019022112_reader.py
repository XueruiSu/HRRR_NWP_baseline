from var_dict import PRESSURE_VARS, SURFACE_VARS
import xarray as xr
import os
import numpy as np

day = "20190221"
hour = "13"

print(f"{day}{hour}")
npy_file = np.load(f"/blob/kmsw0eastau/data/hrrr/hourly2/{day}{hour}.npy")
print(f"{day}{hour}", npy_file.shape) # (1, 69, 1059, 1799)


