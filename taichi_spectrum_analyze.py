import torch  
import matplotlib.pyplot as plt  
  
# 创建一个简单的一维信号  
t = torch.linspace(0, 1, 1000)  
x = torch.sin(2 * torch.pi * 50 * t) + 0.5 * torch.sin(2 * torch.pi * 120 * t)  
  
# 对输入信号执行傅立叶变换  
fft_x = torch.fft.fft(x)  
  
# 计算频谱的振幅  
amplitude_spectrum = torch.abs(fft_x)  
  
# 计算对数谱  
log_spectrum = torch.log10(amplitude_spectrum)  
  
# 绘制对数谱  
plt.plot(log_spectrum[:len(log_spectrum) // 2])  
plt.title("Log Spectrum")  
plt.xlabel("Frequency (Hz)")  
plt.ylabel("Amplitude (dB)")  
plt.savefig("./figure/log_spectrum_analyze.png")  
