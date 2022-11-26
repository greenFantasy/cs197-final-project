import zero_shot
import eval
import pandas as pd
import pdb
import argparse
import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxr_filepath', type=str, default='data/cxr.h5', help="Directory to load chest x-ray image data from.")
    parser.add_argument('--cxr_labels', type=str, default='data/CheXpert/test_labels.csv', help="True labels for zeroshot.")
    parser.add_argument('--model_dir', type=str, default="fixed_stored_UU")
    parser.add_argument('--use_cxrbert', action='store_true')
    parser.add_argument('--results_dir', type=str, default="bootvals")
    args = parser.parse_args()
    return args

def model_zeroshot_confidence_interval(config, model_paths):
    
    cxr_filepath = config.cxr_filepath
    cxr_labels = pd.read_csv(config.cxr_labels).columns.tolist()[2:]
   
    # PRESET variables
    cxr_pair_template = ("{}", "no {}")
    cache_dir = None
    use_cxrbert = True
    
    predictions, y_pred_avg = zero_shot.ensemble_models(
        model_paths=model_paths, 
        cxr_filepath=cxr_filepath, 
        cxr_labels=cxr_labels, 
        cxr_pair_template=cxr_pair_template, 
        cache_dir=cache_dir,
        use_cxrbert=use_cxrbert
    )

    cxr_true_labels_path = "/home/ec2-user/cheXpert-test-set-labels/groundtruth.csv"

    # loads in ground truth labels into memory
    test_pred = y_pred_avg
    test_true = zero_shot.make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

    # boostrap evaluations for 95% confidence intervals
    bootstrap_results = eval.bootstrap(test_pred, test_true, cxr_labels) # (df of results for each bootstrap, df of CI)

    return bootstrap_results[1]

if __name__ == "__main__":
    config = parse_args()
    files = glob.glob(config.model_dir + "/*.pt")
    print(files)
    
    for epoch in range(1, 1+len(files)):
        for file in files:
            if f"epoch{epoch}" in file:
                print("Running zeroshot for: ", file)
                result = model_zeroshot_confidence_interval(config, [file])
                print("Finished running boostrap")
                filename = f"{config.results_dir}/{config.model_dir}_epoch{epoch}.csv"
                print("Savings results to file: ", filename)
                result.to_csv(filename)
                