import torch  
import numpy as np
import matplotlib.pyplot as plt

def split_and_reconstruct(image):  
    # 傅立叶变换  
    fft_image = torch.fft.fft2(image)  
  
    # 按照要求切分矩阵  
    tiles = [  
        [fft_image[:353, :599], fft_image[:353, 599:1200], fft_image[:353, 1200:]],  
        [fft_image[353:706, :599], fft_image[353:706, 599:1200], fft_image[353:706, 1200:]],  
        [fft_image[706:, :599], fft_image[706:, 599:1200], fft_image[706:, 1200:]]  
    ]  
  
    # 储存重建的图像  
    reconstructed_images = []  
  
    # 填充缺失的位置并进行傅立叶逆变换  
    for i, row in enumerate(tiles):  
        for j, tile in enumerate(row):  
            # 创建一个全零的复数矩阵，与原始傅立叶变换后的矩阵具有相同的尺寸  
            padded_tile = torch.zeros_like(fft_image)  
  
            # 计算水平方向上的填充位置  
            if j == 0:  
                col_start, col_end = 0, 599  
            elif j == 1:  
                col_start, col_end = 599, 1200  
            else:  
                col_start, col_end = 1200, 1799  
  
            # 将切分后的矩阵放入全零矩阵的相应位置  
            padded_tile[i*tile.shape[0]:(i+1)*tile.shape[0], col_start:col_end] = tile  
  
            # 傅立叶逆变换  
            reconstructed_image = torch.fft.ifft2(padded_tile)  
  
            # 仅保留实部  
            reconstructed_image = torch.real(reconstructed_image)  
  
            # 将重建的图像添加到列表中  
            reconstructed_images.append(reconstructed_image)  
  
    return reconstructed_images  



# 示例数据  
UV_data = np.load("UV_data.npy") # (3*24, 1059, 1799, 2)
UV_data = torch.from_numpy(UV_data).float()
    
image = UV_data[0, ..., 0]

# 分割并重建图像  
reconstructed_images = split_and_reconstruct(image)  
  
# 输出重建图像的形状  
RMSE_dict = []
for i, img in enumerate(reconstructed_images):  
    RMSE_loss = torch.sqrt(torch.mean((UV_data[0, ..., 0] - img)**2)).item()
    print(f"重建图像 {i + 1} 的形状：{img.shape}, loss:", RMSE_loss)  
    RMSE_dict.append(RMSE_loss)

# 创建 2x5 的子图  
fig, axes = plt.subplots(2, 5, figsize=(15, 6))  
  
# 添加原始图像到子图  
axes[0, 0].imshow(image)  
axes[0, 0].set_title("origin")  
axes[0, 0].axis('off')  

# 将重建的图像添加到子图中  
for i, img in enumerate(reconstructed_images):  
    row, col = (i + 1) // 5, (i + 1) % 5  
    axes[row, col].imshow(img)  
    axes[row, col].set_title(f"reconstruction {i}, loss:{RMSE_dict[i]:.2f}")  
    axes[row, col].axis('off')  
plt.savefig("./figure/3X3_fourier_decomposition_test.png")



