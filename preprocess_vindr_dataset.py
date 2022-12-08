import subprocess
import numpy as np
import os
import sys
import pandas as pd
from PIL import Image
import h5py
import matplotlib.pyplot as plt
from typing import List, Tuple
from pathlib import Path
import pydicom as dicom

import torch
from torch.utils import data
from tqdm import tqdm
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode, CenterCrop
from transformers import AutoTokenizer

from health_multimodal.image.data.io import load_image, remap_to_uint8
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference

from torchxrayvision.datasets import VinBrain_Dataset

import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_auc_score, roc_curve, classification_report
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import average_precision_score

import clip
from chexmera_model import CLIP, HUGGING_FACE_BERT_URLS
from eval import evaluate, plot_roc, accuracy, sigmoid, bootstrap, compute_cis

import torch
from torch.utils import data
from tqdm import tqdm

from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode, CenterCrop

from zero_shot import VinDrTestDataset
from zero_shot import get_biovil_transform


if __name__ == "__main__":
    use_biovision = True
    pretrained = True

    if not use_biovision:
        # load data
        transformations = [
            # means computed from sample in `cxr_stats` notebook
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        ]
        # if using CLIP pretrained model
        if pretrained: 
            # resize to input resolution of pretrained clip model
            input_resolution = 224
            transformations.append(Resize(input_resolution, interpolation=InterpolationMode.BICUBIC))
            transformations.append(CenterCrop(input_resolution))
        transform = Compose(transformations)
    else:
        transform = get_biovil_transform()
    print(f"Using transforms {transform}")

    torch_dset = VinDrTestDataset(
                imgpath='vindr-cxr/1.0.0/test/',
                csvpath='vindr-cxr/1.0.0/annotations/image_labels_test.csv',
                transform=transform,
                use_biovision=use_biovision
    )

    loader = torch.utils.data.DataLoader(torch_dset, shuffle=False, num_workers=8)

    dataset_length = 3000
    all_data = torch.zeros((dataset_length, 3, 224, 224))

    for i, data in enumerate(tqdm(loader)):
        all_data[i] = data['img']

    # print(processed_images)
    save_path = f'data/{"biovision_" if use_biovision else ""}vindr_processed.pt'
    torch.save(all_data, save_path)
    print(f"Saving to {save_path}")