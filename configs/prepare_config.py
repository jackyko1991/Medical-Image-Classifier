import os
import json
import argparse
from jinja2 import Template

def get_parser():
    # create parser object
    parser = argparse.ArgumentParser(
        description='Prepare configuration files for medical imaging using Tensorflow',
        epilog='For questions and bug reports, contact Jacky Ko <jkmailbox1991@gmail.com>')

    # add arguments
    # parser.add_argument(
    # 	'-p','--phase', 
    # 	dest='phase', 
    # 	help='Training phase (default=TRAIN)',
    # 	choices=['TRAIN','PREDICT'],
    # 	default='TRAIN',
    # 	metavar='[TRAIN PREDICT]')
    parser.add_argument(
        '--tmp_json',
        dest='tmp_json',
        help='Template JSON file location for model configuration',
        type=str,
        default='config_sample.json', 
        metavar='FILENAME'
        )
    parser.add_argument(
        '--output',
        dest='output',
        help='Output JSON file location',
        type=str,
        default='config_out.json', 
        metavar='FILENAME'
        )
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        default='./data',
        type=str,
        help='Specify data directory',
        metavar='DIR'
        )
    parser.add_argument(
        '--fold',
        dest='fold',
        default=0,
        type=int,
        help="Fold number",
        metavar='INT'
        )
    parser.add_argument(
        '--ckpt_dir',
        dest='ckpt_dir',
        default='./ckpt',
        type=str,
        help='Specify checkpoint directory',
        metavar='DIR'
        )
    parser.add_argument(
        '--log_dir',
        dest='log_dir',
        default='./log',
        type=str,
        help='Specify tensorboard log directory',
        metavar='DIR'
        )
    parser.add_argument(
        '--epoches',
        dest='epoches',
        default=80,
        type=int,
        help='Training epoches',
        metavar='DIR'
        )

    args = parser.parse_args()

    return args

def main(args):
    print("loading template file: {}".format(args.tmp_json))
    with open(args.tmp_json) as json_temp:
       config = json.load(json_temp)

    config["TrainingSetting"]["Data"]["TrainingDataDirectory"] = os.path.join(args.data_dir,"fold_{}".format(args.fold),"train")
    config["TrainingSetting"]["Data"]["TestingDataDirectory"] = os.path.join(args.data_dir,"fold_{}".format(args.fold),"test")
    config["TrainingSetting"]["LogDir"] = os.path.join(args.log_dir,"mixed_mag_resnet_fold_{}_LR-4_2_xent_mom-0.9".format(args.fold),"log")
    config["TrainingSetting"]["CheckpointDir"] = os.path.join(args.ckpt_dir,"mixed_mag_resnet_fold_{}_LR-4_2_xent_mom-0.9".format(args.fold),"ckpt")
    # config["TrainingSetting"]["LogDir"] = os.path.join(args.log_dir)
    # config["TrainingSetting"]["CheckpointDir"] = os.path.join(args.ckpt_dir)
    config["TrainingSetting"]["Epoches"] = args.epoches

    print("saving config json file: {}".format(args.output))
    with open(args.output, 'w') as outfile:
        json.dump(config, outfile)

if __name__=="__main__":
    args = get_parser()
    main(args)
