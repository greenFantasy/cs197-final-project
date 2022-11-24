import zero_shot
import eval
import pandas as pd
import pdb


model_paths = ['./checkpoints/pt-imp/checkpoint_11900_CheXzero.pt']
cxr_filepath = '../cs197-final-project/data/cxr.h5'
cxr_labels = pd.read_csv("../cs197-final-project/data/CheXpert/test_labels.csv").columns.tolist()[2:]
cxr_pair_template = ("{}", "no {}")
cache_dir = None

predictions, y_pred_avg = zero_shot.ensemble_models(
    model_paths=model_paths, 
    cxr_filepath=cxr_filepath, 
    cxr_labels=cxr_labels, 
    cxr_pair_template=cxr_pair_template, 
    cache_dir=cache_dir,
)

cxr_true_labels_path = "/home/ec2-user/cheXpert-test-set-labels/groundtruth.csv"

# loads in ground truth labels into memory
test_pred = y_pred_avg
test_true = zero_shot.make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

# evaluate model, no bootstrap
cxr_results: pd.DataFrame = eval.evaluate(test_pred, test_true, cxr_labels) # eval on full test datset

# boostrap evaluations for 95% confidence intervals
bootstrap_results = eval.bootstrap(test_pred, test_true, cxr_labels) # (df of results for each bootstrap, df of CI)