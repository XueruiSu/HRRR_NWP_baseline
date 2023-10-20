'''
This code is used to test the herbie package and compare the results with grib2 files we already have.
conclusion: the results are the same.
'''
from herbie import Herbie
import numpy as np
# generate a random string
import random
rand_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10))
print(rand_str)
H = Herbie(
    "20200701 04",
    model="hrrr",
    product="prs",
    fxx=0,
    save_dir=f"~/{rand_str}/"
)
print(H.PRODUCTS)
print(H.grib)
print(H.idx)

# print(H.xarray())
print(H.inventory(searchString=":500 mb:"))
# ds = H.xarray("TMP:2 m above")
ds = H.xarray("UGRD:500 mb")
# print(ds.to_array())
print(ds)
numpy_array = ds.to_array().values[0]
np.save(f"./2020070104UGRD-500mb.npy", numpy_array)
        
print(numpy_array.shape)

print(numpy_array.max(), numpy_array.min(), numpy_array.mean(), numpy_array.std())

# from var_dict import PRESSURE_VARS, SURFACE_VARS
# import xarray as xr
# import os
# import numpy as np

# day = "20200701"
# hour = "12"

# var = "TMP_P0_L103_GLC0"   

# ds_input = xr.open_dataset(os.path.join(f"/blob/kmsw0eastau/data/hrrr/grib2/hrrr",
#                             f"{day}/hrrr.t{hour}z.wrfprsf00.grib2"), engine="pynio")
# L103_var_grib2_value = ds_input[var].to_numpy()

# print(L103_var_grib2_value.shape)
# print(L103_var_grib2_value-numpy_array)
# print((L103_var_grib2_value-numpy_array).sum()) # 0.0

