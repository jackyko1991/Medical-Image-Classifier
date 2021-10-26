import pandas as pd
import os
from tqdm import tqdm

def main():
    metrics_dir = "/mnt/DIIR-JK-NAS/data/carotid/results/metrics"
    output = "/mnt/DIIR-JK-NAS/data/carotid/results/metrics_summary.csv"

    # switch binary to true if only one class is used for training
    binary = False

    # resnet_image_fold_0_LR_0.0001_2_xent_mom-0.9_test_1500
    if binary:
        results = {
            "network": [],
            "dataset": [],
            "fold": [],
            "learning rate": [],
            "network config": [],
            "loss function": [],
            "momentum": [],
            "phase": [],
            "step": [],
            "micro auc": [],
            "macro auc": [],
            "sensitivity": [],
            "specificity": [],
            "f1-score": []
        }
    else:
        results = {
            "network": [],
            "dataset": [],
            "fold": [],
            "learning rate": [],
            "network config": [],
            "loss function": [],
            "momentum": [],
            "phase": [],
            "step": [],
            "micro avg auc": [],
            "macro avg auc": [],
            "macro avg precision": [],
            "macro avg sensitivity": [],
            "macro avg f1-score": [],
            "weighted avg auc": [],
            "weighted avg precision": [],
            "weighted avg sensitivity": [],
            "weighted avg f1-score": []
        }

    pbar = tqdm(os.listdir(metrics_dir))
    for file in pbar:
        filename = os.path.splitext(file)[0]

        pbar.set_description(filename)

        filename_split = filename.split("_")

        if filename_split[2] == "mag":
            filename_split[1] = filename_split[1] + "_" + filename_split[2]
            filename_split.pop(2)
        
        results["network"].append(filename_split[0])
        results["dataset"].append(filename_split[1])
        results["fold"].append(filename_split[3])
        results["learning rate"].append(filename_split[5])
        results["network config"].append(filename_split[6])
        results["loss function"].append(filename_split[7])
        results["momentum"].append(filename_split[8].split("-")[1])
        results["phase"].append(filename_split[9])
        results["step"].append(filename_split[10])

        # read csv file
        metric = pd.read_csv(os.path.join(metrics_dir,file),index_col=0)

        print(file)
        print(metric)

        if binary:
            results["auc"].append(metric.iloc[0]["auc"])
            results["sensitivity"].append(metric.iloc[0]["sensitivity"])
            results["specificity"].append(metric.iloc[0]["specificity"])
            results["f1-score"].append(metric.iloc[0]["f1-score"])
        else:
            results["micro avg auc"].append(metric.loc["micro avg"]["auc"])
            results["macro avg auc"].append(metric.loc["macro avg"]["auc"])
            results["macro avg precision"].append(metric.loc["macro avg"]["precision"])
            results["macro avg sensitivity"].append(metric.loc["macro avg"]["sensitivity"])
            results["macro avg f1-score"].append(metric.loc["macro avg"]["f1-score"])
            results["weighted avg auc"].append(metric.loc["weighted avg"]["auc"])
            results["weighted avg precision"].append(metric.loc["weighted avg"]["precision"])
            results["weighted avg sensitivity"].append(metric.loc["weighted avg"]["sensitivity"])
            results["weighted avg f1-score"].append(metric.loc["weighted avg"]["f1-score"])

    results_df = pd.DataFrame(data=results)
    results_df.to_csv(output,index=False)


if __name__ == "__main__":
    main()