import gif
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import os
from matplotlib import cm

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
    UV_data = np.load("UV_data.npy") # (10, 1059, 1799, 2)
    print(UV_data[...,0] - UV_data[...,1])
    frames = []
    for i in range(UV_data.shape[0]):
        frames.append(helper_plot_U(UV_data[i], 0, 0))
    gif.save(frames, "U_component.gif", duration=100)
    frames = []
    for i in range(UV_data.shape[0]):
        frames.append(helper_plot_U(UV_data[i], 0, 1))
    gif.save(frames, "V_component.gif", duration=100)
    frames = []
    for i in range(UV_data.shape[0]):
        frames.append(helper_plot_U(UV_data[i], 1))
    gif.save(frames, "UV_quiver.gif", duration=100)
    
    
    
    
    
    

    
    