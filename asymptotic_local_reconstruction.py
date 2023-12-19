
import torch  
import matplotlib.pyplot as plt  
import numpy as np
import gif
from matplotlib import cm

@gif.frame
def helper_plot_U(smoke_values_extracted, quiver=0, UV=1):
    try:
        # (1059, 1799, 2)
        plt.figure(figsize=(8, 4.7))
        plt.imshow(smoke_values_extracted[..., UV], alpha=1, cmap=cm.viridis, vmin=-30, vmax=30, origin="lower")
        plt.colorbar()
        plt.clim(-30, 30)
        # 设置colorbar的上下限  
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title("10m_v_component_of_wind(m/s)")
    except:
        print("error")    
    

def reconstruct_from_center(image, range_x=175, range_y=300):  
    # 傅立叶变换  
    fft_image = torch.fft.fft2(image)
      
    # 将零频分量移动到频谱中心  
    shifted_fft_x = torch.fft.fftshift(fft_image)  
    
    # 只留中心[355, 601]部分
    # if =0, then only need to cut 1 pixel: No.530 -> index: 529
    # No.899 -> index: 898
    center = shifted_fft_x[529-range_x:530+range_x, 898-range_y:899+range_y]
    padded_tile = torch.zeros_like(fft_image)  
    padded_tile[529-range_x:530+range_x, 898-range_y:898+range_y] = center  

    # 将零频分量移回原点  
    unshifted_fft_x = torch.fft.ifftshift(padded_tile)  
    
    # 傅立叶逆变换  
    reconstructed_image = torch.fft.ifft2(unshifted_fft_x)  
  
    # 仅保留实部  
    reconstructed_image = torch.real(reconstructed_image)  

    return reconstructed_image, shifted_fft_x

# 示例数据  
UV_data = np.load("UV_data.npy") # (3*24, 1059, 1799, 2)
UV_data = torch.from_numpy(UV_data).float()

image = UV_data[0, ..., 0]

# 使用四个角的矩阵进行重建  
cut_num_x, cut_num_y = 0, 0
reconstructed_image, shifted_fft_x = reconstruct_from_center(image, cut_num_x, cut_num_y)  

RMSE_loss = torch.sqrt(torch.mean((image - reconstructed_image)**2)).item()
print(f"重建图像的形状：{reconstructed_image.shape}, loss:", RMSE_loss)  

# 创建 1x2 的子图  
fig, axes = plt.subplots(1, 3, figsize=(12, 6))  

# 添加原始图像到子图  
axes[0].imshow(image)  
axes[0].set_title("origin")  
axes[0].axis('off')  

# 添加重建的图像到子图  
axes[1].imshow(reconstructed_image)  
axes[1].set_title(f"center reconstruction loss:{RMSE_loss:.2f}")  
axes[1].axis('off')  

axes[2].imshow(torch.log(torch.abs(shifted_fft_x) + 1))
axes[2].set_title("center log spectrum")
axes[2].axis('off')

plt.savefig(f"./figure/center_reconstruction_x{cut_num_x+353}_y{cut_num_y+601}.png")

frames = []
for i in range(UV_data.shape[0]):
    print("quiver", i)
    frames.append(helper_plot_U(UV_data[i], 1))
gif.save(frames, "./figure_UV/UV_quiver.gif", duration=100)
    



