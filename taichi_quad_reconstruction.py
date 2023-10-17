import torch  
import matplotlib.pyplot as plt  
import numpy as np

def reconstruct_from_four_corners(image):  
    # 傅立叶变换  
    fft_image = torch.fft.fft2(image)  
  
    # 按照要求切分矩阵，选择四个角的矩阵  
    tiles = [  
        fft_image[:353, :599],  # 左上角  
        fft_image[706:, :599],  # 左下角  
        fft_image[:353, 1200:], # 右上角  
        fft_image[706:, 1200:]  # 右下角  
    ]  
    row = [[0, 353], [706, 1059], [0, 353], [706, 1059]]
    col = [[0, 599], [0, 599], [1200, 1799], [1200, 1799]]
    # 创建一个全零的复数矩阵，与原始傅立叶变换后的矩阵具有相同的尺寸  
    padded_tile = torch.zeros_like(fft_image)  
  
    # 将四个角的矩阵放入全零矩阵的相应位置  
    for i, tile in enumerate(tiles):  
        # 计算填充位置  
        row_start, row_end = row[i][0], row[i][1]
        col_start, col_end = col[i][0], col[i][1]
        print(row_start, row_end, col_start, col_end)
        # 将切分后的矩阵放入全零矩阵的相应位置  
        padded_tile[row_start:row_end, col_start:col_end] = tile  
  
    # 傅立叶逆变换  
    reconstructed_image = torch.fft.ifft2(padded_tile)  
  
    # 仅保留实部  
    reconstructed_image = torch.real(reconstructed_image)  
  
    return reconstructed_image  
  
# 示例数据  
UV_data = np.load("UV_data.npy") # (3*24, 1059, 1799, 2)
UV_data = torch.from_numpy(UV_data).float()
    
image = UV_data[0, ..., 0]

# 使用四个角的矩阵进行重建  
reconstructed_image = reconstruct_from_four_corners(image)  

RMSE_loss = torch.sqrt(torch.mean((image - reconstructed_image)**2)).item()
print(f"重建图像的形状：{reconstructed_image.shape}, loss:", RMSE_loss)  
    
# 创建 1x2 的子图  
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  
  
# 添加原始图像到子图  
axes[0].imshow(image)  
axes[0].set_title("origin")  
axes[0].axis('off')  
  
# 添加重建的图像到子图  
axes[1].imshow(reconstructed_image)  
axes[1].set_title(f"reconstruction loss:{RMSE_loss:.2f}")  
axes[1].axis('off')  
  
plt.savefig("./figure/quad_reconstruction.png")
