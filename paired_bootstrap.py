import argparse
import numpy as np
import pandas as pd

from eval import paired_bootstrap
from eval import plot_paired_bootstrap
from zero_shot import make_true_labels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_names', type=str, default="")
    parser.add_argument('--pred_csv_paths', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--true_label_path', type=str, default="../cheXpert-test-set-labels/groundtruth.csv")
    parser.add_argument('--table', action='store_true')
    parser.add_argument('--eval_vindr', action='store_true')
    args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    pred_csv_paths = args.pred_csv_paths.split(",")
    if args.model_names:
        model_names = args.model_names.split(",")    
    else:
        model_names = [p.split("/")[-1].split(".")[0] for p in pred_csv_paths]
    assert len(model_names) == len(pred_csv_paths)
    
    print(f"Model names: {model_names}", end="\n\n")
    
    y_preds = [pd.read_csv(path) for path in pred_csv_paths]
    pathologies = y_preds[0].columns
    
    print(f"Computing Paired AUC Bootstrap for {list(pathologies)} pathologies", end="\n\n")
    
    y_true = make_true_labels(cxr_true_labels_path=args.true_label_path, cxr_labels=pathologies, vindr_labels=args.eval_vindr)
    
    y_preds = [y.to_numpy() for y in y_preds]
    paired_dfs, _ = paired_bootstrap(model_names, y_preds, y_true, pathologies, n_samples=args.num_samples)
    
    print("Finished generating bootstraps", end='\n\n')
    
    if len(paired_dfs) == 2:
        model1_name = model_names[0]
        model2_name = model_names[1]
        plot_paired_bootstrap(model1_name, model2_name, paired_dfs[model1_name], paired_dfs[model2_name])
    else:
        print(f"Not plotting, number of things to compare is {len(paired_dfs)}")
        
    if args.table:
        data = [[] for _ in model_names]
        for i, model_name in enumerate(model_names):
            numpy_data = paired_dfs[model_name].to_numpy()
            means = np.nanmean(numpy_data, axis=0).tolist()
            lowers = np.nanquantile(numpy_data, 0.05, axis=0).tolist()
            uppers = np.nanquantile(numpy_data, 0.95, axis=0).tolist()
            assert len(lowers) == len(means), "Mean and lower quantile should have same length"
            data[i] = [f"{m:.3f} ({l:.3f}, {u:.3f})" for m, l, u in zip(means, lowers, uppers)]
        table = pd.DataFrame(columns=pathologies, data=data)
        table["Configuration"] = model_names
        table = table[[ *(["Configuration"] + list(pathologies)) ]]
        split_point = 7
        print(table.iloc[:, :split_point].to_latex(index=False))
        print(table.iloc[:, split_point:].to_latex(index=False))
        table.to_csv("results/table.csv", index=False)