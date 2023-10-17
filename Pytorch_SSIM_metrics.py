import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

def SSIM(target, pre):
    target = torch.from_numpy(target)
    pre = torch.from_numpy(pre)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim_score = ssim(pre, target)
    return ssim_score.item()

preds = torch.rand([3, 3, 256, 256])
target = preds * 0.75
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
ssim_score = ssim(preds, target)
print(ssim_score)

print(SSIM(target.numpy(), preds.numpy()))

loss_func = SSIM
print(f"{loss_func}") # <function SSIM at 0x7f8b1c0b7d30>
