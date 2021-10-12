import os
import pandas as pd
import argparse
import json
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def get_parser():
        # create parser object
    parser = argparse.ArgumentParser(
        description='Benchmark the prediction result',
        epilog='For questions and bug reports, contact Jacky Ko <jkmailbox1991@gmail.com>')

    # add arguments
    parser.add_argument(
        '--ground_truth',
        dest='ground_truth',
        help='Ground truth CSV file location',
        type=str,
        default='label.csv', 
        metavar='FILENAME'
        )
    parser.add_argument(
        '--predict',
        dest='predict',
        help='Predict output CSV file location',
        type=str,
        default='predict.csv', 
        metavar='FILENAME'
        )
    parser.add_argument(
        '--output',
        dest='output',
        help='Benchmark result output file location',
        type=str,
        default='benchmark.csv', 
        metavar='FILENAME'
        )
    parser.add_argument(
        '--plot_dir',
        dest='plot_dir',
        help='Plotting output directory',
        type=str,
        default='./benchmark_plot', 
        metavar='DIR'
        )
    parser.add_argument(
        '--config_json',
        dest='config_json',
        help='JSON file location for model configuration',
        type=str,
        default='config.json', 
        metavar='FILENAME'
        )

    args = parser.parse_args()

    return args

def main(args):
    # read ground truth and predict csv files
    gt = pd.read_csv(args.ground_truth)
    pred = pd.read_csv(args.predict)

    # load the configuration json
    with open(args.config_json) as f: 
        config = json.load(f)
    columns = ["case"]
    classnames = config["TrainingSetting"]["Data"]["ClassNames"]
    columns.extend(classnames)

    # select the gt columns
    gt = gt[columns]
    gt = gt[gt.case.isin(pred["case"])]

    # sorting
    gt = gt.sort_values(by=["case"])
    pred = pred.sort_values(by=["case"])

    y_true = np.argmax(gt[classnames].to_numpy(),axis=1)
    y_pred = np.argmax(pred[classnames].to_numpy(),axis=1)

    if config["TrainingSetting"]["LossFunction"]["Multiclass/Multilabel"] == "Multiclass":
        # confusion matrix
        cnf_matrix = metrics.confusion_matrix(y_true,y_pred)
    
        print("confusion matrix:")
        print(cnf_matrix)

        # plot confusion matrix 
        fig_cm, ax_cm = plt.subplots(1,1,figsize=(9,6))
        disp = metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap=plt.cm.Blues, ax=ax_cm)
        ax_cm.set_xticklabels(classnames)
        ax_cm.set_yticklabels(classnames)

        # roc
        fig_roc, ax_roc = plt.subplots(1,1,figsize=(9,6))
        ax_roc.plot([0,1],[0,1], linestyle="--", lw=2, color="r", label="", alpha=0.8)
        ax_roc.set(
            xlim=[0.0, 1.05],
            ylim=[0.0, 1.05],
            title="Receiver operating characteristics",
            xlabel='False Positive Rate',
            ylabel='True Positive Rate'
        )

        fpr = dict()
        tpr = dict()
        auc = dict()
        mean_fpr = np.linspace(0,1,100)
        tpr_interp = []

        for classname in classnames:
            fpr[classname], tpr[classname], thresholds = metrics.roc_curve(gt[classname], pred[classname])
            interp_tpr = np.interp(mean_fpr, fpr[classname], tpr[classname])
            interp_tpr[0] = 0.0
            tpr_interp.append(interp_tpr)

            auc[classname] = metrics.auc(fpr[classname],tpr[classname])
            ax_roc.plot(fpr[classname],tpr[classname],label="{} (AUC = {:.2f})".format(classname, auc[classname]),alpha=0.3,lw=1)

        # macro average roc and auc 
        tpr["macro"] = np.mean(tpr_interp,axis=0)
        tpr["macro"][-1] = 1.0
        auc["macro"] = metrics.auc(mean_fpr, tpr["macro"])
        fpr["macro"] = mean_fpr
        ax_roc.plot(fpr["macro"], tpr["macro"],color="b",label="Macro Average (AUC = {:.2f})".format(auc["macro"]),lw=2,alpha=0.8,)
        
        # micro average roc and auc
        fpr["micro"], tpr["micro"], thresholds = metrics.roc_curve(gt[classnames].to_numpy().ravel(), pred[classnames].to_numpy().ravel())
        auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

        ax_roc.plot(fpr["micro"], tpr["micro"],color="darkorange",label="Micro Average (AUC = {:.2f})".format(auc["micro"]),lw=2,alpha=0.8,)

        ax_roc.legend(loc="lower right")
        
        # benchmarks: precision, recall, f1, sensitivity, specificity
        report = metrics.classification_report(y_true,y_pred, target_names=classnames,output_dict=True)

        print(report)

        print("AUC: {}".format(auc["macro"]))
        print("precision: {}".format(report["macro avg"]["precision"]))
        print("sensitivity: {}".format(report["macro avg"]["recall"]))
        print("fscore: {}".format(report["macro avg"]["f1-score"]))

        output_df = pd.DataFrame(columns= ["class","auc","precision","sensitivity","f1-score"])

        for classname in classnames:
            output_df = output_df.append({
                "class": classname, 
                "auc": auc[classname], 
                "precision": report[classname]["precision"],
                "sensitivity": report[classname]["recall"],
                "f1-score": report[classname]["f1-score"]
            },ignore_index=True)

        output_df = output_df.append({
            "class": "macro avg", 
            "auc": auc["macro"], 
            "precision": report["macro avg"]["precision"],
            "sensitivity": report["macro avg"]["recall"],
            "f1-score": report["macro avg"]["f1-score"]
        },ignore_index=True)

        output_df.to_csv(args.output,index=False)
        

    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)
    fig_cm.savefig(os.path.join(args.plot_dir,"confusion_matrix.png"))
    fig_roc.savefig(os.path.join(args.plot_dir,"roc.png"))
    print("save plot at {} complete".format(args.plot_dir))

if __name__ == "__main__":
    args = get_parser()
    main(args)