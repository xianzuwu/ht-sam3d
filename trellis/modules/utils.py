import torch
import torch.nn as nn
from ..modules import sparse as sp

FP16_MODULES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
    sp.SparseConv3d,
    sp.SparseInverseConv3d,
    sp.SparseLinear,
)

def convert_module_to_f16(l):
    """
    [H100-FIX] 自动检测硬件：如果是 H100 则转为 bfloat16，否则使用 float16。
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        target_dtype = torch.bfloat16 if "H100" in gpu_name or "H200" in gpu_name else torch.float16
    else:
        target_dtype = torch.float16

    if isinstance(l, FP16_MODULES):
        for p in l.parameters():
            p.data = p.data.to(target_dtype)


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32.
    """
    if isinstance(l, FP16_MODULES):
        for p in l.parameters():
            p.data = p.data.float()


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)