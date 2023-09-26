import torch

def nowcastnet_forecast_mse(hr_pred, hr_tar, all_vars, all_std):
    loss_dict = {}
    device = hr_pred.device

    assert hr_pred.shape == hr_tar.shape, f"hr_pred shape is [f{hr_pred.shape}], hr_tar shape is [{hr_tar.shape}]"
    batch_size, pred_time, channel, height, width = hr_tar.shape 

    hr_forecast_loss = (hr_tar - hr_pred) ** 2
    loss_dict['loss'] = hr_forecast_loss.mean()

    with torch.no_grad():
        for idx, var in enumerate(all_vars['hrrr']):
            if idx >= channel:
                break
            hr_forecast_error = hr_forecast_loss.mean(dim=(0, 1, 3, 4))
            loss_dict[f"{var}_hrrr_forecast_mse"] = hr_forecast_error[idx].sqrt() * all_std['hrrr'][var].to(device)
    return loss_dict