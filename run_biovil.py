import tempfile
from pathlib import Path
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

from eval import evaluate

from health_multimodal.text import get_cxr_bert_inference
from health_multimodal.image import get_biovil_resnet_inference
from health_multimodal.vlp import ImageTextInferenceEngine

def get_biovil_model():
    image_text_inference = ImageTextInferenceEngine(
        image_inference_engine=get_biovil_resnet_inference(),
        text_inference_engine=get_cxr_bert_inference()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device {device}")
    
    image_text_inference.to(device)
    
    return image_text_inference

"""
Example of how to use the image_text_inference_model
image_paths = [
    Path("../CheXzero/cxr_data/files/p10/p10000764/s57375967/096052b7-d256dc40-453a102b-fa7d01c6-1b22c6b4.jpg"),
    Path("../CheXzero/cxr_data/files/p10/p10000898/s50771383/0c4eb1e1-b801903c-bcebe8a4-3da9cd3c-3b94a27c.jpg")
]
pathology = "pneumonia"
positive_text_prompt = f"Findings suggesting {pathology}"
negative_text_prompt = f"No evidence of {pathology}"

for im_path in image_paths:
    pos_sim_score = image_text_inference.get_similarity_score_from_raw_data(im_path, positive_text_prompt)
    neg_sim_score = image_text_inference.get_similarity_score_from_raw_data(im_path, negative_text_prompt)

    print(pos_sim_score, neg_sim_score)
"""

def get_single_prediction(image_text_inference, image_path, pathology, biovil_prompts=False):
    if not isinstance(image_path, Path):
        image_path = Path(image_path)
    
    if not biovil_prompts:
        positive_text_prompt = pathology
        negative_text_prompt = f"No {pathology}"
    else:
        positive_text_prompt = f"Findings suggesting {pathology}"
        negative_text_prompt = f"No evidence of {pathology}"
    
    pos_pred = image_text_inference.get_similarity_score_from_raw_data(image_path, positive_text_prompt)
    neg_pred = image_text_inference.get_similarity_score_from_raw_data(image_path, negative_text_prompt)
    
    sum_pred = np.exp(pos_pred) + np.exp(neg_pred)
    y_pred = np.exp(pos_pred) / sum_pred
    return y_pred

def get_test_data(path):
    """Returns dataframe and pathology labels"""
    df = pd.read_csv(path)
    df = df[["view1" in x for x in df["Path"]]]
    assert len(df) == 500, "This is specific to the CheXpert dataset, remove if using another dataset!"
    cxr_labels = df.columns[1:].tolist()
    return df, cxr_labels

def adjust_path(path):
    return os.path.join("../OLD_REPO/cs197-final-project/data/CheXpert", path)

def generate_preds(model, data, pathologies, biovil_prompts=False):
    pred_df = data.copy()
    for pathology in pathologies:
        preds = []
        for path in tqdm(data["Path"]):
            pred = get_single_prediction(model, adjust_path(path), pathology, biovil_prompts=biovil_prompts)
            preds.append(pred)
        pred_df[pathology] = preds
    return pred_df

def save_preds(preds, filename="biovil_chexpert_preds", biovil_prompts=False):
    if biovil_prompts:
        filename += "_biovil_prompts"
    save_path = f"results/{filename}.csv"
    preds.to_csv(save_path, index=False)
    print(f"Saved predictions at {save_path}")

def get_mean_auc(pred_df, label_df, pathologies):
    y_true = label_df.iloc[:, 1:].to_numpy()
    y_pred = pred_df.iloc[:, 1:].to_numpy()

    return evaluate(y_pred, y_true, cxr_labels=pathologies)

# image_embeddings = {}
# image_model = image_text_inference.image_inference_engine
# for path in paths:
#     image_embeddings[path] = image_model.get_projected_global_embedding(path)
    
# pos_prompts = cxr_labels
# neg_prompts = [f"No {pathology}" for pathology in cxr_labels]    

# def get_text_embeddings(prompts):
#     import torch.nn.functional as F
#     text_model = image_text_inference.text_inference_engine
#     text_emb = text_model.get_embeddings_from_prompt(prompts, normalize=False)
#     text_emb = F.normalize(text_emb.mean(dim=0), dim=0, p=2)
#     return text_emb

# pos_emb = get_text_embeddings(pos_prompts)
# neg_emb = get_text_embeddings(neg_prompts)

if __name__ == "__main__":
    biovil_prompts = True
    model = get_biovil_model()
    test_csv_path = "../OLD_REPO/cs197-final-project/data/CheXpert/test_labels.csv"
    data_df, cxr_labels = get_test_data(test_csv_path)
    pred_df = generate_preds(model, data_df, cxr_labels, biovil_prompts=biovil_prompts)
    save_preds(pred_df, biovil_prompts=biovil_prompts)
    print(get_mean_auc(pred_df, data_df, cxr_labels))