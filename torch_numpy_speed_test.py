import torch
import numpy as np
from time import time


a_numpy = np.random.rand(1059, 1799)
b_numpy = np.random.rand(1059, 1799)

t1 = time()
a = torch.from_numpy(a_numpy)
b = torch.from_numpy(b_numpy)
rmse = torch.sqrt(torch.mean((a - b) ** 2))
t2 = time()

t3 = time()
rmse = np.sqrt(np.mean((a_numpy - b_numpy) ** 2))
t4 = time()

print("Time taken by PyTorch: ", t2 - t1)
print("Time taken by NumPy: ", t4 - t3)