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

    if len(classnames) > 1:
        y_true = np.argmax(gt[classnames].to_numpy(),axis=1)
        y_pred = np.argmax(pred[classnames].to_numpy(),axis=1)
    else:
        y_true = gt[classnames].to_numpy().ravel()
        # need to set threshold by youden index, current set threshold at 0.5
        y_pred = np.round(pred[classnames].to_numpy().ravel())

    if config["TrainingSetting"]["LossFunction"]["Multiclass/Multilabel"] == "Multiclass":
        # confusion matrix
        cnf_matrix = metrics.confusion_matrix(y_true,y_pred)
    
        print("confusion matrix:")
        print(cnf_matrix)

        # plot confusion matrix 
        fig_cm, ax_cm = plt.subplots(1,1,figsize=(9,6))
        disp = metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap=plt.cm.Blues, ax=ax_cm)
        if len(classnames) > 1:
            ax_cm.set_xticklabels(classnames)
            ax_cm.set_yticklabels(classnames)
        else:
            ax_cm.set_xticklabels([0,1])
            ax_cm.set_yticklabels([0,1])

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

        weighted_auc = 0
        weight = 0
        for classname in classnames:
            fpr[classname], tpr[classname], thresholds = metrics.roc_curve(gt[classname], pred[classname])
            interp_tpr = np.interp(mean_fpr, fpr[classname], tpr[classname])
            interp_tpr[0] = 0.0
            tpr_interp.append(interp_tpr)

            auc[classname] = metrics.auc(fpr[classname],tpr[classname])
            weighted_auc += auc[classname]*np.sum(gt[classname].to_numpy())
            weight += np.sum(gt[classname].to_numpy())

            if len(classnames) > 1:
                ax_roc.plot(fpr[classname],tpr[classname],label="{} (AUC = {:.2f})".format(classname, auc[classname]),alpha=0.3,lw=1)
            else:
                ax_roc.plot(fpr[classname],tpr[classname],label="AUC = {:.2f}".format(auc[classname]),alpha=0.3,lw=1)

        weighted_auc = weighted_auc/weight
        auc["weighted"] = weighted_auc

        # macro average roc and auc 
        tpr["macro"] = np.mean(tpr_interp,axis=0)
        tpr["macro"][-1] = 1.0
        auc["macro"] = metrics.auc(mean_fpr, tpr["macro"])
        fpr["macro"] = mean_fpr
        if len(classnames) > 1:
            ax_roc.plot(fpr["macro"], tpr["macro"],color="b",label="Macro Average (AUC = {:.2f})".format(auc["macro"]),lw=2,alpha=0.8,)
        
        # micro average roc and auc
        fpr["micro"], tpr["micro"], thresholds = metrics.roc_curve(gt[classnames].to_numpy().ravel(), pred[classnames].to_numpy().ravel())
        auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

        if len(classnames) > 1:
            ax_roc.plot(fpr["micro"], tpr["micro"],color="darkorange",label="Micro Average (AUC = {:.2f})".format(auc["micro"]),lw=2,alpha=0.8,)

        ax_roc.legend(loc="lower right")
        
        # benchmarks: precision, recall, f1, sensitivity (recall of positive class), specificity (recall of negative class)
        if len(classnames) > 1:
            report = metrics.classification_report(y_true,y_pred, target_names=classnames,output_dict=True)
            print(metrics.classification_report(y_true,y_pred, target_names=classnames))
            print("micro AUC: {}".format(auc["micro"]))
            print("macro AUC: {}".format(auc["macro"]))
            print("precision: {}".format(report["macro avg"]["precision"]))
            print("sensitivity: {}".format(report["macro avg"]["recall"]))
            print("fscore: {}".format(report["macro avg"]["f1-score"]))
        else:
            report = metrics.classification_report(y_true,y_pred,output_dict=True)
            #print(metrics.classification_report(y_true,y_pred,output_dict=False))
            print("AUC: {}".format(auc[classnames[0]]))
            print("precision: {}".format(report["0"]["precision"]))
            print("sensitivity: {}".format(report["1"]["recall"]))
            print("fscore: {}".format(report["1"]["f1-score"]))

        if len(classnames) > 1:
            output_df = pd.DataFrame(columns= ["class","auc","precision","sensitivity","f1-score"])
        else:
            output_df = pd.DataFrame(columns= ["class","auc","sensitivity","specificity","f1-score"])

        if len(classnames) > 1:
            for classname in classnames:
                output_df = output_df.append({
                    "class": classname, 
                    "auc": auc[classname], 
                    "precision": report[classname]["precision"],
                    "sensitivity": report[classname]["recall"],
                    "f1-score": report[classname]["f1-score"]
                },ignore_index=True)

            # for multi class macro and micro average are the same, only macro average is reported
            output_df = output_df.append({
                "class": "micro avg",
                "auc": auc["micro"],
            },ignore_index=True)

            output_df = output_df.append({
                "class": "macro avg", 
                "auc": auc["macro"], 
                "precision": report["macro avg"]["precision"],
                "sensitivity": report["macro avg"]["recall"],
                "f1-score": report["macro avg"]["f1-score"]
            },ignore_index=True)

            output_df = output_df.append({
                "class": "weighted avg", 
                "auc": auc["weighted"], 
                "precision": report["weighted avg"]["precision"],
                "sensitivity": report["weighted avg"]["recall"],
                "f1-score": report["weighted avg"]["f1-score"]
            },ignore_index=True)
        else:
            output_df = output_df.append({
                "class": classnames[0], 
                "auc": auc[classnames[0]], 
                "specificity": report["0"]["precision"],
                "sensitivity": report["1"]["recall"],
                "f1-score": report["1"]["f1-score"]
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