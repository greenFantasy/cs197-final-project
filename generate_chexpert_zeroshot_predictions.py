import zero_shot
import pandas as pd
import numpy as np
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxr_filepath', type=str, default='data/test_cxr.h5', help="Directory to load chest x-ray image data from.")
    parser.add_argument('--cxr_labels', type=str, default='data/CheXpert/test_labels.csv', help="True labels for zeroshot.")
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--image_tower_type', type=str, required=True)
    parser.add_argument('--use_huggingface_bert', action='store_true')
    parser.add_argument('--huggingface_bert_key', type=str, action='store', default='cxr')
    parser.add_argument('--image_csv_path', type=str, default='data/CheXpert/test_labels.csv')
    parser.add_argument('--results_dir', type=str, default="predictions")
    parser.add_argument('--dataset', type=str, default="chexpert")
    parser.add_argument('--group', action='store_true')
    args = parser.parse_args()
    
    assert args.model_path.split("/")[0] == "checkpoints", "Checkpoint must be in checkpoints folder"
    
    if args.image_tower_type=="biovision" and not args.image_csv_path:
        raise ValueError("Must define image_csv_path if use_biovision=True")
    
    return args

def generate_predictions(config, model_paths):
    
    cxr_filepath = config.cxr_filepath
    
    if config.dataset == 'chexpert':
        cxr_labels = pd.read_csv(config.cxr_labels).columns.tolist()[2:]
    elif config.dataset == 'vindr':
        cxr_labels = pd.read_csv(config.cxr_labels).columns.tolist()[1:-1]
    else:
        raise ValueError('Must select a valid dataset for evaluation')
   
    # PRESET variables
    cxr_pair_template = ("{}", "no {}")
    cache_dir = None
    image_csv_path = config.image_csv_path
    
    if config.group:
        print("Ignoring provided image_tower_type!")
        image_tower_types = []
        use_huggingface_berts = []
        huggingface_bert_keys = []
        image_types = ["biovision", "mococxr", "medaug"]
        text_types = ["clinical", "blue", "cxr"]
        for path in model_paths:
            model_name = path.split("/")[-1].split(".")[0]
            
            if model_name == "baseline":
                image_tower_types.append("clip")
                use_huggingface_berts.append(False)
                huggingface_bert_keys.append("cxr")
                continue
            
            for type in image_types:
                if type in model_name:
                    image_tower_types.append(type)
                    use_huggingface_berts.append(False)
                    huggingface_bert_keys.append("cxr")
                    break
            else:
                image_tower_types.append("clip")
                for type in text_types:
                    if type in model_name:
                        use_huggingface_berts.append(True)
                        huggingface_bert_keys.append(type)
                        break
                else:
                    raise ValueError("Shouldnt hit this point")
            
            print(f"Predicting {path} as {image_tower_types[-1], huggingface_bert_keys[-1]}")
    else:
        image_tower_types = [config.image_tower_type]
        use_huggingface_berts = [config.use_huggingface_bert]
        huggingface_bert_keys = [config.huggingface_bert_key]
    
    for i, model_path in enumerate(model_paths):
        try:
            split = model_path.split("/")
            split[0] = config.results_dir
            split[-1] = split[-1].split(".")[0]
            split[-1] = config.dataset + '/' + split[-1]
            save_name_np = "/".join(split)
            split[-1] = split[-1] + ".csv"
            save_name_csv = "/".join(split)
            
            if os.path.exists(save_name_csv):
                print(f"Already have this model, skipping {model_path}")
                continue
            
            predictions, all_preds = zero_shot.ensemble_models(
                image_tower_types=image_tower_types[i:i+1],
                model_paths=model_paths[i:i+1], 
                cxr_filepath=cxr_filepath, 
                cxr_labels=cxr_labels, 
                cxr_pair_template=cxr_pair_template, 
                cache_dir=cache_dir,
                use_huggingface_berts=use_huggingface_berts[i:i+1],
                huggingface_bert_keys=huggingface_bert_keys[i:i+1],
                image_csv_path=image_csv_path,
                use_vindr=config.dataset=='vindr'
            )
            
            pred = predictions[0]
            
            parent_dir = os.path.dirname(save_name_np)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            np.save(save_name_np, pred)
            pd.DataFrame(columns = cxr_labels, data = pred).to_csv(save_name_csv, index=False)

            print(f"Finished saving to {save_name_np} and {save_name_csv} files")
        except Exception as e:
            print(f"Failed generation for {model_path}")
            continue

if __name__ == "__main__":
    config = parse_args()
    if not config.model_path:
        raise ValueError("model_path must be specified")
    if not config.group:
        generate_predictions(config, [config.model_path])
    else:
        import os
        filenames = os.listdir(config.model_path)
        full_paths = [os.path.join(config.model_path, fname) for fname in filenames]
        print(f"Predicting on models {filenames}")
        generate_predictions(config, full_paths)