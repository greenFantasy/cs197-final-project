"""
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import hashlib
import os
import urllib
import warnings
from typing import Union, List

import torch
from tqdm import tqdm

from .clip import build_model
from .model import get_image_model

from health_multimodal.image.model.modules import MLP
from health_multimodal.image.model.model import ImageEncoder, ImageModel, ImageModelOutput

__all__ = ["available_models", "_clip_vision"]

_MODELS = {
    "resnet50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    # "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    # "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    # "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
}


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

class CLIPImageEncoder(ImageEncoder):
    def __init__(self, clip_resnet50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = clip_resnet50
        
    def forward(self, x, return_patch_embeddings=False):
        out = self.encoder(x).float()
        if return_patch_embeddings:
            # Returning same value for patch and global image embedding
            return out, out
        return out

class CLIPImageModel(ImageModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        joint_feature_size = 128
        clip_vision = load()
        self.encoder = CLIPImageEncoder(clip_resnet50=clip_vision, img_model_type="resnet50")
        self.feature_size = 1024
        self.use_MLP = False
        if self.use_MLP:
            self.projector = MLP(input_dim=self.feature_size, output_dim=joint_feature_size,
                                    hidden_dim=joint_feature_size, use_1x1_convs=True)
        else:
            print("Using the new linear projector for CLIP instead")
            self.projector = torch.nn.Linear(self.feature_size, joint_feature_size, bias=False)
            torch.nn.init.normal_(self.projector.weight, std=self.feature_size ** -0.5)
        self.projector = self.projector.to(self.encoder.encoder.conv1.weight.dtype)
    
    def get_dtype(self):
        return list(self.projector.parameters())[0].dtype
    
    def forward(self, x):
        with torch.set_grad_enabled(not self.freeze_encoder):
            patch_x, pooled_x = self.encoder(x, return_patch_embeddings=True)
            if self.use_MLP:
                patch_x = patch_x.unsqueeze(-1).unsqueeze(-1)
                projected_patch_embeddings = self.projector(patch_x.to(self.get_dtype()))
                projected_global_embedding = torch.mean(projected_patch_embeddings, dim=(2, 3))
            else:
                projected_patch_embeddings = self.projector(patch_x.to(self.get_dtype()))
                projected_global_embedding = projected_patch_embeddings

        logits = self.classifier(pooled_x) if self.classifier else None
        return ImageModelOutput(img_embedding=pooled_x,
                                patch_embedding=patch_x,
                                class_logits=logits,
                                projected_patch_embeddings=projected_patch_embeddings,
                                projected_global_embedding=projected_global_embedding)

def load(name: str = "resnet50", device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model (default) or more hackable non-JIT model.

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name])
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")
    
    model = torch.jit.load(model_path, map_location=device).eval()
    
    model = build_model(model.state_dict()).to(device)
    
    if str(device) == "cpu":
        print("Converting to float model")
        model.float()
    
    return model.visual

def _clip_vision(device="cuda" if torch.cuda.is_available() else "cpu"):
    joint_feature_size = 128
    model = CLIPImageModel("resnet50", joint_feature_size=joint_feature_size)
    print(model.projector)
    return model.to(device).eval()