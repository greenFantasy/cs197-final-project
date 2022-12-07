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
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
from transformers import AutoTokenizer

from health_multimodal.image.data.io import load_image, remap_to_uint8
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference

import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_auc_score, roc_curve, classification_report
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import average_precision_score

import clip
from model import CLIP, HUGGING_FACE_BERT_URLS
from eval import evaluate, plot_roc, accuracy, sigmoid, bootstrap, compute_cis

CXR_FILEPATH = '../../project-files/data/test_cxr.h5'
FINAL_LABEL_PATH = '../../project-files/data/final_paths.csv'

class CXRTestDataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        img_path: Path to hdf5 file containing images.
        label_path: Path to file containing labels 
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(
        self, 
        img_path: str, 
        transform = None, 
        use_biovision = False,
        use_vindr = False,
    ):
        super().__init__()
        self.use_biovision = use_biovision
        self.use_vindr = use_vindr
        if not self.use_biovision:
            self.img_dset = h5py.File(img_path, 'r')['cxr']
        elif self.use_vindr:
            # self.img_dset = h5py.File(img_path, 'r')['cxr']
            self.img_dset = pd.read_csv(img_path)['Path'].tolist()
            assert len(self.img_dset) == 3000
        else:
            self.img_dset = [os.path.join(os.path.dirname(img_path), p) for p in pd.read_csv(img_path)['Path'].tolist() if "view1" in p]
            assert len(self.img_dset) == 500
        self.transform = transform
            
    def __len__(self):
        return len(self.img_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = self.img_dset[idx] # np array, (320, 320)
        if not self.use_biovision:
            img = np.expand_dims(img, axis=0)
            img = np.repeat(img, 3, axis=0)
            img = torch.from_numpy(img) # torch, (320, 320)
        elif self.use_vindr:
            img_path = img
            img = dicom.dcmread(img_path).pixel_array
            img = remap_to_uint8(img)
            img = Image.fromarray(img).convert("L")
        else:
            img_path = img
            if isinstance(img_path, str):
                img_path = Path(img_path)
            img = load_image(img_path)
        
        if self.transform:
            # if self.use_biovision and self.use_vindr:
            #     t = create_chest_xray_transform_for_inference(224, 224)
            #     img = t(img)
            # else:
            #     img = self.transform(img)
            img = self.transform(img)
            
        sample = {'img': img}
    
        return sample

def load_clip(image_tower_type, model_path, pretrained=False, context_length=77, use_huggingface_bert=False, huggingface_bert_key='cxr'): 
    """
    FUNCTION: load_clip
    ---------------------------------
    """
    device = torch.device("cpu")
    if pretrained is False: 
        # use new model params
        raise ValueError("pretrained cannot be False when loading CLIP for zero_shot")
        params = {
            'embed_dim':768,
            'image_resolution': 320,
            'vision_layers': 12,
            'vision_width': 768,
            'vision_patch_size': 16,
            'context_length': context_length, 
            'vocab_size': 49408,
            'transformer_width': 512,
            'transformer_heads': 8,
            'transformer_layers': 12,
            'use_huggingface_bert': use_huggingface_bert,
            'use_biovision': use_biovision,
            'huggingface_bert_key': huggingface_bert_key
        }

        model = CLIP(**params)
    else: 
        model, _ = clip.load("RN50", image_tower_type, device=device, jit=False, use_huggingface_bert=use_huggingface_bert, huggingface_bert_key=huggingface_bert_key) 
    try: 
        model.load_state_dict(torch.load(model_path, map_location=device))
    except: 
        print("Argument error. Set pretrained = True.", sys.exc_info()[0])
        raise
    return model

def zeroshot_classifier(classnames, templates, model, context_length=77, use_huggingface_bert=False, 
                        huggingface_bert_key='cxr'):
    """
    FUNCTION: zeroshot_classifier
    -------------------------------------
    This function outputs the weights for each of the classes based on the 
    output of the trained clip model text transformer. 
    
    args: 
    * classnames - Python list of classes for a specific zero-shot task. (i.e. ['Atelectasis',...]).
    * templates - Python list of phrases that will be indpendently tested as input to the clip model.
    * model - Pytorch model, full trained clip model.
    * context_length (optional) - int, max number of tokens of text inputted into the model.
    
    Returns PyTorch Tensor, output of the text encoder given templates. 
    """
    if use_huggingface_bert:
        url = HUGGING_FACE_BERT_URLS[huggingface_bert_key]
        print(f"Loading Tokenizer from the following Hugging Face Index: {url}")
        tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)

    with torch.no_grad():
        zeroshot_weights = []
        # compute embedding through model for each class
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] # format with class
            
            if use_huggingface_bert:
                texts = tokenizer(text=texts, add_special_tokens=True, padding='longest', return_tensors='pt')
            else:
                texts = clip.tokenize(texts, context_length=context_length) # tokenize
            class_embeddings = model.encode_text(texts) # embed with text encoder
            
            # normalize class_embeddings
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # average over templates 
            class_embedding = class_embeddings.mean(dim=0) 
            # norm over new averaged templates
            class_embedding /= class_embedding.norm() 
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights

def predict(loader, model, zeroshot_weights, softmax_eval=True, verbose=0): 
    """
    FUNCTION: predict
    ---------------------------------
    This function runs the cxr images through the model 
    and computes the cosine similarities between the images
    and the text embeddings. 
    
    args: 
        * loader -  PyTorch data loader, loads in cxr images
        * model - PyTorch model, trained clip model 
        * zeroshot_weights - PyTorch Tensor, outputs of text encoder for labels
        * softmax_eval (optional) - Use +/- softmax method for evaluation 
        * verbose (optional) - bool, If True, will print out intermediate tensor values for debugging.
        
    Returns numpy array, predictions on all test data samples. 
    """
    y_pred = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    zeroshot_weights = zeroshot_weights.to(device)
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            images = data['img'].to(device)

            # predict
            image_features = model.encode_image(images) 
            image_features /= image_features.norm(dim=-1, keepdim=True) # (1, 768)

            # obtain logits
            logits = image_features @ zeroshot_weights # (1, num_classes)
            logits = np.squeeze(logits.to("cpu").numpy(), axis=0) # (num_classes,)
        
            if softmax_eval is False: 
                norm_logits = (logits - logits.mean()) / (logits.std())
                logits = sigmoid(norm_logits) 
            
            y_pred.append(logits)
            
            if verbose: 
                plt.imshow(images[0][0])
                plt.show()
                print('images: ', images)
                print('images size: ', images.size())
                
                print('image_features size: ', image_features.size())
                print('logits: ', logits)
                print('logits size: ', logits.size())
    
    model.to("cpu")
    y_pred = np.array(y_pred)
    return np.array(y_pred)

def run_single_prediction(cxr_labels, template, model, loader, softmax_eval=True, context_length=77, use_huggingface_bert=False, 
                          huggingface_bert_key='cxr'): 
    """
    FUNCTION: run_single_prediction
    --------------------------------------
    This function will make probability predictions for a single template
    (i.e. "has {}"). 
    
    args: 
        * cxr_labels - list, labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        * template - string, template to input into model. 
        * model - PyTorch model, trained clip model
        * loader - PyTorch data loader, loads in cxr images
        * softmax_eval (optional) - Use +/- softmax method for evaluation 
        * context_length (optional) - int, max number of tokens of text inputted into the model.
        
    Returns list, predictions from the given template. 
    """
    cxr_phrase = [template]
    zeroshot_weights = zeroshot_classifier(cxr_labels, cxr_phrase, model, context_length=context_length, 
                                           use_huggingface_bert=use_huggingface_bert, 
                                           huggingface_bert_key=huggingface_bert_key)
    y_pred = predict(loader, model, zeroshot_weights, softmax_eval=softmax_eval)
    return y_pred

def run_softmax_eval(model, loader, eval_labels: list, pair_template: tuple, context_length: int = 77, use_huggingface_bert=False, 
                     huggingface_bert_key='cxr'): 
    """
    Run softmax evaluation to obtain a single prediction from the model.
    """
     # get pos and neg phrases
    pos = pair_template[0]
    neg = pair_template[1]

    # get pos and neg predictions, (num_samples, num_classes)
    pos_pred = run_single_prediction(eval_labels, pos, model, loader, 
                                     softmax_eval=True, context_length=context_length, 
                                     use_huggingface_bert=use_huggingface_bert, 
                                     huggingface_bert_key=huggingface_bert_key) 
    neg_pred = run_single_prediction(eval_labels, neg, model, loader, 
                                     softmax_eval=True, context_length=context_length, 
                                     use_huggingface_bert=use_huggingface_bert, 
                                     huggingface_bert_key=huggingface_bert_key) 

    # compute probabilities with softmax
    sum_pred = np.exp(pos_pred) + np.exp(neg_pred)
    y_pred = np.exp(pos_pred) / sum_pred
    return y_pred

def make_true_labels(
    cxr_true_labels_path: str, 
    cxr_labels: List[str],
    cutlabels: bool = True,
    vindr_labels = False,
): 
    """
    Loads in data containing the true binary labels
    for each pathology in `cxr_labels` for all samples. This
    is used for evaluation of model performance.

    args: 
        * cxr_true_labels_path - str, path to csv containing ground truth labels
        * cxr_labels - List[str], subset of label columns to select from ground truth df 
        * cutlabels - bool, if True, will keep columns of ground truth labels that correspond
            with the labels inputted through `cxr_labels`. Otherwise, drop the first column and keep remaining.

    Returns a numpy array of shape (# samples, # labels/pathologies)
        representing the binary ground truth labels for each pathology on each sample.
    """
    # create ground truth labels
    full_labels = pd.read_csv(cxr_true_labels_path)
    # reorder labels if dataset is vindr
    if vindr_labels:
        # get proper order of labels from file paths csv
        order = pd.read_csv('data/vindr_cxr_paths.csv')['Path']
        # filter out path and file extension to just have file names
        order = [o.split('/')[-1].split('.')[0] for o in order]
        # reorder full labels according to order
        full_labels = full_labels.set_index('image_id')
        full_labels = full_labels.loc[order]
        full_labels = full_labels.reset_index()
    if cutlabels: 
        full_labels = full_labels.loc[:, cxr_labels]
    else: 
        full_labels.drop(full_labels.columns[0], axis=1, inplace=True)

    y_true = full_labels.to_numpy()
    return y_true

def get_biovil_transform():
    TRANSFORM_RESIZE = 256 # 512
    TRANSFORM_CENTER_CROP_SIZE = 224 # 480
    
    biovil_transform = create_chest_xray_transform_for_inference(
        resize=TRANSFORM_RESIZE,
        center_crop_size=TRANSFORM_CENTER_CROP_SIZE,
    )
    
    return biovil_transform

def make(
    image_tower_type: str,
    model_path: str, 
    cxr_filepath: str, 
    pretrained: bool = True, 
    context_length: bool = 77, 
    use_huggingface_bert=False,
    huggingface_bert_key='cxr',
    image_csv_path=None,
    use_vindr=False,
):
    """
    FUNCTION: make
    -------------------------------------------
    This function makes the model, the data loader, and the ground truth labels. 
    
    args: 
        * model_path - String for directory to the weights of the trained clip model. 
        * context_length - int, max number of tokens of text inputted into the model. 
        * cxr_filepath - String for path to the chest x-ray images. 
        * cxr_labels - Python list of labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        * pretrained - bool, whether or not model uses pretrained clip weights
        * cutlabels - bool, if True, will keep columns of ground truth labels that correspond
        with the labels inputted through `cxr_labels`. Otherwise, drop the first column and keep remaining.
    
    Returns model, data loader. 
    """
    # load model
    model = load_clip(
        image_tower_type,
        model_path=model_path, 
        pretrained=pretrained, 
        context_length=context_length,
        use_huggingface_bert=use_huggingface_bert,
        huggingface_bert_key=huggingface_bert_key
    )

    use_biovision = image_tower_type == "biovision"
    
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
        transform = Compose(transformations)
    else:
        transform = get_biovil_transform()
        print(f"Using BioViL transforms: {transform}")
    
    # create dataset
    torch_dset = CXRTestDataset(
        img_path=cxr_filepath if not use_biovision else image_csv_path,
        transform=transform, 
        use_biovision=use_biovision,
        use_vindr=use_vindr
    )
    loader = torch.utils.data.DataLoader(torch_dset, shuffle=False)
    
    return model, loader

## Run the model on the data set using ensembled models
def ensemble_models(
    image_tower_type: str, # Only supports one image tower type for the entire ensemble
    model_paths: List[str], 
    cxr_filepath: str, 
    cxr_labels: List[str], 
    cxr_pair_template: Tuple[str], 
    cache_dir: str = None, 
    save_name: str = None,
    use_huggingface_bert: bool = False,
    huggingface_bert_key: str = 'cxr',
    image_csv_path = None,
    use_vindr=False
) -> Tuple[List[np.ndarray], np.ndarray]: 
    """
    Given a list of `model_paths`, ensemble model and return
    predictions. Caches predictions at `cache_dir` if location provided.

    Returns a list of each model's predictions and the averaged
    set of predictions.
    """

    predictions = []
    model_paths = sorted(model_paths) # ensure consistency of 
    for path in model_paths: # for each model
        model_name = Path(path).stem

        # load in model and `torch.DataLoader`
        model, loader = make(
            image_tower_type,
            model_path=path, 
            cxr_filepath=cxr_filepath, 
            use_huggingface_bert=use_huggingface_bert,
            huggingface_bert_key=huggingface_bert_key,
            image_csv_path=image_csv_path,
            use_vindr=use_vindr
        ) 
        
        # path to the cached prediction
        if cache_dir is not None:
            if save_name is not None: 
                cache_path = Path(cache_dir) / f"{save_name}_{model_name}.npy"
            else: 
                cache_path = Path(cache_dir) / f"{model_name}.npy"

        # if prediction already cached, don't recompute prediction
        if cache_dir is not None and os.path.exists(cache_path): 
            print("Loading cached prediction for {}".format(model_name))
            y_pred = np.load(cache_path)
        else: # cached prediction not found, compute preds
            print("Inferring model {}".format(path))
            y_pred = run_softmax_eval(model, loader, cxr_labels, cxr_pair_template, use_huggingface_bert=use_huggingface_bert, 
                                      huggingface_bert_key=huggingface_bert_key)
            if cache_dir is not None: 
                Path(cache_dir).mkdir(exist_ok=True, parents=True)
                np.save(file=cache_path, arr=y_pred)
        predictions.append(y_pred)
    
    # compute average predictions
    y_pred_avg = np.mean(predictions, axis=0)
    
    return predictions, y_pred_avg
