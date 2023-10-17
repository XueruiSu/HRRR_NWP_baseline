import torch  
  
# 创建一个二维张量（图像）  
image = torch.tensor([  
    [1., 2., 3.],  
    [4., 5., 6.],  
    [7., 8., 9.]  
])  
print("原始图像：")
print(image, image.shape)

# 进行傅立叶变换  
fft_result = torch.fft.fft2(image)  
# 输出傅立叶变换结果  
print("傅立叶变换结果：")  
print(fft_result, fft_result.shape)  
  
# 进行逆傅立叶变换  
ifft_result = torch.real(torch.fft.ifft2(fft_result))
# 输出逆傅立叶变换结果  
print("逆傅立叶变换结果：")  
print(ifft_result, ifft_result.shape)  
