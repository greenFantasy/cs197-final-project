import torch

from health_multimodal.image import get_biovil_resnet

def _biovision(device="cuda" if torch.cuda.is_available() else "cpu"):
    return get_biovil_resnet().to(device).eval()