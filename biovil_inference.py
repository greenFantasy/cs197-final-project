import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image

import torch
import torchvision.datasets as datasets
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode, ToTensor

from health_multimodal.image import get_biovil_resnet
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
from health_multimodal.image.data.io import load_image

from data_process import preprocess
 
# resnet_model = get_biovil_resnet()

def get_single_image_path(data_path):
    for root, _, fnames in os.walk(data_path):
        for fname in fnames:
            if fname.split(".")[-1] == "jpg":
                return os.path.join(root, fname)
    raise Exception("No image found")

def get_biovil_transform():
    TRANSFORM_RESIZE = 512
    TRANSFORM_CENTER_CROP_SIZE = 480
    
    biovil_transform = create_chest_xray_transform_for_inference(
        resize=TRANSFORM_RESIZE,
        center_crop_size=TRANSFORM_CENTER_CROP_SIZE,
    )
    
    return biovil_transform

def biovil_transform_from_path(image_path: Path):
    biovil_transform = get_biovil_transform()
    img = load_image(image_path)
    return biovil_transform(img)

def get_chexzero_transform(input_resolution=224):
    return Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
        ])

def chexzero_transform_from_path(path, input_resolution=224):
    chexzero_transform = get_chexzero_transform(input_resolution=input_resolution)

    # in img_to_hdf5
    img = cv2.imread(str(path))
    # convert to PIL Image object
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    # preprocess in data_process
    img = np.array(preprocess(img_pil))
    
    # Done from CXRDataset
    img = np.expand_dims(img, axis=0)
    img = np.repeat(img, 3, axis=0)
    
    # Our own addition that isn't necessary in the dataset loading
    img = torch.tensor(img).float()
        
    return chexzero_transform(img)

# Goal: Recreate biovil transform using a read from an h5

if __name__ == "__main__":
    data_path = "../CheXzero/cxr_data/files/"
    image_path = Path(get_single_image_path(data_path))
    biovil = biovil_transform_from_path(image_path)
    chexzero = chexzero_transform_from_path(image_path)
    
    # print(biovil.shape, chexzero.shape) # torch.abs(biovil - chexzero).sum())

