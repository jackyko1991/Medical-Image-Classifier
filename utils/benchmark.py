import os
import pandas as pd
import argparse


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
        help='Benchmark result CSV output file location',
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

    args = parser.parse_args()

    return args

def main(args):
    # read ground truth and predict csv files
    gt = pd.read_csv(args.ground_truth)
    pred = pd.read_csv(args.predict)
    

    return

if __name__ == "__main__":
    args = get_parser()
    main(args)