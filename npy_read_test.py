import numpy as np

file_name = "/nfs/weather/hrrr/prs/hourly2_10days"
npy_file = np.load(f"{file_name}/2019010100.npy")
print(npy_file.shape) # (1, 69, 1059, 1799)


