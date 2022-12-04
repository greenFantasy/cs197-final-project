from .model import get_image_model
from .util import get_pretrained_dir, modify_moco_dict

import os

import torch

DEFAULT_MODEL_CHECKPOINT_FNAME = "moco-cxr_resnet50.pth.tar"
DEFAULT_MODEL_CHECKPOINT_PATH = os.path.join(get_pretrained_dir(), 
                                             DEFAULT_MODEL_CHECKPOINT_FNAME)

def _mococxr(checkpoint_path=DEFAULT_MODEL_CHECKPOINT_PATH, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = get_image_model()
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path)['state_dict']
        modify_moco_dict(state_dict)
        model.encoder.encoder.load_state_dict(state_dict)
    else:
        raise ValueError("checkpoint_path cannot be empty")
    
    return model.to(device).eval()
        