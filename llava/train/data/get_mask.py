
import torch
import torch.nn.functional as F

def median_pool2d(input, kernel_size, stride=None, padding=0):
    if stride is None:
        stride = kernel_size

    B, C, H, W = input.shape
    input_unf = F.unfold(input, kernel_size=kernel_size, stride=stride, padding=padding)
    kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    output = input_unf.transpose(1, 2).reshape(B, -1, C, kH * kW)
    median = output.median(dim=-1)[0]
    out_H = (H + 2 * padding - kH) // stride + 1
    out_W = (W + 2 * padding - kW) // stride + 1
    return median.permute(0, 2, 1).reshape(B, C, out_H, out_W)


def avg_pool2d(input, kernel_size, stride=None, padding=0):
    if stride is None:
        stride = kernel_size

    B, C, H, W = input.shape
    input_unf = F.unfold(input, kernel_size=kernel_size, stride=stride, padding=padding)
    kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    output = input_unf.transpose(1, 2).reshape(B, -1, C, kH * kW)
    median = output.mean(dim=-1)#[0]
    out_H = (H + 2 * padding - kH) // stride + 1
    out_W = (W + 2 * padding - kW) // stride + 1
    return median.permute(0, 2, 1).reshape(B, C, out_H, out_W)



def median_blur_2d(img_tensor, kernel_size):
    # img_tensor: (1, 1, H, W) for grayscale
    padding = kernel_size // 2
    unfolded = F.unfold(img_tensor, kernel_size, padding=padding)
    median = unfolded.median(dim=1)[0]
    output = median.view(img_tensor.size())
    return output

def max_blur_2d(img_tensor, kernel_size):
    # img_tensor: (1, 1, H, W) for grayscale
    padding = kernel_size // 2
    unfolded = F.unfold(img_tensor, kernel_size, padding=padding)
    median = unfolded.max(dim=1)[0]
    output = median.view(img_tensor.size())
    return output


def diff_mask(input_arr,target_arr):
    mask = ((input_arr - target_arr).abs().max(1,keepdims=True)[0] > 20 / 255)
    mask = avg_pool2d(mask.float(),16)
    mask = mask > 0.5
    mask = median_blur_2d(mask.float(),3)
    mask = max_blur_2d(mask.float(),3)
    return mask


def comput_pixel_diff(input_arr,target_arr,q=0.1):
    # min max > 0
    non_padding = input_arr > 2/255
    pixel_diff = (input_arr - target_arr).abs()[non_padding]
    return pixel_diff.quantile(q)* 100