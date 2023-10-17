import torch  
  
def bi_interpolation(input_space, velocity_x, velocity_y):  
    m, n = input_space.shape  
    predicted_space = torch.zeros_like(input_space)  
  
    # 反向追踪  
    for i in range(m):  
        for j in range(n):  
            x = i - velocity_x[i, j]  
            y = j - velocity_y[i, j]  
  
            x0, y0 = int(x), int(y)  
            x1, y1 = x0 + 1, y0 + 1  
  
            x0, y0 = max(0, x0), max(0, y0)  
            x1, y1 = min(m-1, x1), min(n-1, y1)  
  
            wx, wy = x - x0, y - y0  
  
            predicted_space[i, j] = (1-wx)*(1-wy)*input_space[x0, y0] + (1-wx)*wy*input_space[x0, y1] + wx*(1-wy)*input_space[x1, y0] + wx*wy*input_space[x1, y1]  
  
    # 正向追踪  
    contribution = torch.zeros_like(input_space)  
    for i in range(m):  
        for j in range(n):  
            x = i + velocity_x[i, j]  
            y = j + velocity_y[i, j]  
  
            x0, y0 = int(x), int(y)  
            x1, y1 = x0 + 1, y0 + 1  
  
            x0, y0 = max(0, x0), max(0, y0)  
            x1, y1 = min(m-1, x1), min(n-1, y1)  
  
            wx, wy = x - x0, y - y0  
  
            contribution[x0, y0] += (1-wx)*(1-wy)  
            contribution[x0, y1] += (1-wx)*wy  
            contribution[x1, y0] += wx*(1-wy)  
            contribution[x1, y1] += wx*wy  
  
    # 归一化贡献  
    contribution = contribution / torch.sum(contribution)  
  
    # 更新预测空间的插值贡献  
    predicted_space = torch.mul(predicted_space, contribution)  
  
    return predicted_space  
  
# 示例数据  
input_space = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)  
velocity_x = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float)  
velocity_y = torch.tensor([[1, -1], [1, -1]], dtype=torch.float)  

predicted_space = bi_interpolation(input_space, velocity_x, velocity_y)  
print(predicted_space)  



