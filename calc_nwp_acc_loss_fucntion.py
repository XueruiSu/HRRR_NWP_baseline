import torch
import numpy as np

def anomaly_correlation_coefficient(pred, gt, mean, std, all_vars):
    dtype = pred.dtype
    device = pred.device
 
    a = pred.clone()
    b = gt.clone()
 
    # unnormliaze pred and gt
    a = a * std['hrrr']['all_data'][None,None,:,None,None].to(device)
    # + mean['hrrr']['all_data'][None,None,:,None,None].to(device)
    b = b * std['hrrr']['all_data'][None,None,:,None,None].to(device)
    # + mean['hrrr']['all_data'][None,None,:,None,None].to(device)
    
    # a = (a - mean['hrrr']['all_data'][None,None,:,None,None].to(device)).to(dtype)
    # b = (b - mean['hrrr']['all_data'][None,None,:,None,None].to(device)).to(dtype)
    
    # 这里应该直接mean吗？
    a_prime = a - a.mean(dim=(0, 1, 3, 4))[None,None,:,None,None]
    b_prime = b - b.mean(dim=(0, 1, 3, 4))[None,None,:,None,None]
 
    all_acc = {}
    for idx, var in enumerate(all_vars['hrrr']):
        acc = (
            torch.sum(a_prime[:,:,idx:idx+1] * b_prime[:,:,idx:idx+1]) /
            torch.sqrt(
                torch.sum(a_prime[:,:,idx:idx+1] ** 2) * torch.sum(b_prime[:,:,idx:idx+1] ** 2)
            )
        )
        all_acc[var] = acc
    return acc
# pred: (B, T, C, H, W)

def anomaly_correlation_coefficient_oneVar(pred, gt, mean):
    # pred: (H, W), gt: (H, W)
    a = pred
    b = gt

    # unnormliaze pred and gt
    a = a - mean
    b = b - mean
    
    # 这里应该直接mean吗？
    a_prime = a - a.mean()
    b_prime = b - b.mean()
    
    acc = (
        np.sum(a_prime * b_prime) /
        np.sqrt(
            np.sum(a_prime ** 2) * np.sum(b_prime ** 2)
        )
    )
    return acc

if __name__ == '__main__':
    a = np.random.randn(1059, 1799)
    b = np.random.randn(1059, 1799)
    mean = np.random.randn(1059, 1799)

    acc = anomaly_correlation_coefficient_oneVar(a, b, mean)
    print(acc)


