import zero_shot
import pandas as pd
import numpy as np
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxr_filepath', type=str, default='data/cxr.h5', help="Directory to load chest x-ray image data from.")
    parser.add_argument('--cxr_labels', type=str, default='data/CheXpert/test_labels.csv', help="True labels for zeroshot.")
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--use_cxrbert', action='store_true')
    parser.add_argument('--use_biovision', action='store_true')
    parser.add_argument('--image_csv_path', type=str, default='data/CheXpert/test_labels.csv')
    parser.add_argument('--results_dir', type=str, default="predictions")
    args = parser.parse_args()
    
    assert args.model_path.split("/")[0] == "checkpoints", "Checkpoint must be in checkpoints folder"
    
    if args.use_biovision and not args.image_csv_path:
        raise ValueError("Must define image_csv_path if use_biovision=True")
    
    return args

def generate_predictions(config, model_paths):
    
    cxr_filepath = config.cxr_filepath
    cxr_labels = pd.read_csv(config.cxr_labels).columns.tolist()[2:]
   
    # PRESET variables
    cxr_pair_template = ("{}", "no {}")
    cache_dir = None
    use_cxrbert = config.use_cxrbert
    use_biovision = config.use_biovision
    image_csv_path = config.image_csv_path
    
    predictions, _ = zero_shot.ensemble_models(
        model_paths=model_paths, 
        cxr_filepath=cxr_filepath, 
        cxr_labels=cxr_labels, 
        cxr_pair_template=cxr_pair_template, 
        cache_dir=cache_dir,
        use_cxrbert=use_cxrbert,
        use_biovision=use_biovision,
        image_csv_path=image_csv_path
    )
    
    for model_path, pred in zip(model_paths, predictions):
        split = model_path.split("/")
        split[0] = config.results_dir
        split[-1] = split[-1].split(".")[0]
        save_name_np = "/".join(split)
        split[-1] = split[-1] + ".csv"
        save_name_csv = "/".join(split)
    
        parent_dir = os.path.dirname(save_name_np)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        np.save(save_name_np, pred)
        pd.DataFrame(columns = cxr_labels, data = pred).to_csv(save_name_csv, index=False)

        print(f"Finished saving to {save_name_np} and {save_name_csv} files")

if __name__ == "__main__":
    config = parse_args()
    if not config.model_path:
        raise ValueError("model_path must be specified")
    generate_predictions(config, [config.model_path])