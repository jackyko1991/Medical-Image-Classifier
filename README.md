# Medical-Image-Classifier
Neural network classifier for medical image using Tensorflow

## Content
1. [Introduction](#introduction)
2. [Features](#features)
3. [Networks](#networks)
4. [Usage](#usage)
	1. [Required Libraries](#required-libraries)
	2. [Software Configuration](#sofware-configuration)
	3. [Example usage](#example-usage)
		1. [Folder Hierarchy](#folder-hierarchy)
		2. [Data preprocessing](#data-preprocessing)
		3. [Training](#training)
		4. [Evaluate](#evaluate)
5. [References](#references)
6. [Authors](#authors)

## Introduction
This is a Tensorflow implementation for medical image classification tasks. The repository integrates a pre-processing pipeline optimized for 3D imaging data that suitable for multiple images input with multi-class output. The original application of this framework is to detect intracranial hemorrhage types with 3D CT images, however with slightly modification the framework is general applicable for all medical image classification tasks.

## Features
- 3D data processing ready
- Augumented patching technique, requires less image input for training
- Multichannel input and multiclass output
- Generic image reader with SimpleITK support (Currently only support .nii/.nii.gz format for convenience, easy to expand to DICOM, tiff and jpg format)
- Medical image pre-post processing with SimpleITK filters
- Easy network replacement structure

## Networks
Currently the classifier provides following classification networks:
- LeNet
- AlexNet
- GoogLeNetv1
- Vgg
- ResNet

## Usage
### Required Libraries
Known good dependencies:
- Anaconda 3.6
- Tensorflow 1.15
- SimpleITK
- tqdm
- pylatex (for PDF report output)

### Software Configuration
All necessary network configurations are stored in `config.json` except for GPU specification. Modify the `config.json` file to fit your application.

### Example usage
#### Folder Hierarchy
All training, testing and evaluation data should put in the directory that specified in `config.json`. Note that you should also provide the class labels in CSV format file.

In the default `config.json`, the data folder structure is as following:
```
.                                       # Repository root
├── ...
├── data                      
│   └── example                         # Data directory
│       ├── training                    # Put all training data here
│       │   ├── case1                   # foldername for the cases is arbitrary
│       │   |   └── image_brain_mni.nii # The image name is specified in config.json
│       │   ├── case2
│       │   |   └── image_brain_mni.nii
│       │   ├──	...
│       ├── testing                     # Put all testing data here
│       │   ├── case1                   # follow the same folder structure as the training one
│       │   |   └── image_brain_mni.nii
│       │   ├── case2
│       │   |   └── image_brain_mni.nii
│       │   ├── ...
│       ├── evaluation                  # Put all evaluation data here
│       │   ├── case1                   # follow the same folder structure as the training one
│       │   |   └── image_brain_mni.nii
│       │   ├── case2
│       │   |   └── image_brain_mni.nii
│       │   ├──	...
│       └── labels.csv                  # CSV file stores all labels for training and testing
├── tmp
│   ├── cktp                            # Tensorflow checkpoints
|   └── log                             # Tensorboard logging folder
├── ...
```

In training phase, the the program will automatically scan all the data in training and testing folder. The case/subject name is identified by subfolder's name and all of them should also be listed in `labels.csv`. Filenames could be altered in `config.csv`.

An example dataset folder is provided in [./data/example](./data/example).

#### Data preprocessing
The framework provides 3D image preprocessing pipeline for end-to-end training. You may modify `self.train_transforms` with SimpleITK backed classes in `NiftiDataset.py`.

The example cases are used to classify intracranial hemorrhage types. To provide a normalized spacing and orientation for all data from different CT machines, the data are registered to [MNI-spacing](https://www.lead-dbs.org/about-the-mni-spaces/) as one of the standard neurological coordinate system. Here we provide MNI templates in [./data/atlas](./data/atlas) for gantry tile correction and coordinate system normalization. Cerebral tissues are pre-extracted for a faster training time. The pre-processing procedures are completed by [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki).

#### GPU selection
By default the software use the 0th GPU in the system. Use argument option `--gpu 1` to perform on GPU 1.

#### Training
1. Put all the data in accordance to [Folder Hierarchy](#folder-hierarchy)
2. Run the command:
	```bash
	python main.py -p TRAIN --gpu 0
	```
3. Open Tensorboard:
	```bash
	tensorboard --logdir="./tmp/log"
	```

Note:
- If you want to resume training from previous checkpoint, set `"Restore": true` in `config.json`.
- By default all the input images are in MNI space with size (91x109x91), voxel size = (2x2x2)mm^3.
- Change batch size to optimize the training according to your GPU memory. You may check GPU memory usage with `nvidia-smi`

#### Evaluate

Edit `PredictionSetting` in `./configs/config.json`, then run with following command:

```bash
$ python main.py --phase PREDICT --config_json ./configs/config.json
```

## Grid Search Optimal Hyperparameters
It is advisible to use [Guild AI](https://guild.ai/) for machine learning model tracking. For detail check [this](./GUILD_AI.md) file for further information.

## References
- [About the MNI space(s)](https://www.lead-dbs.org/about-the-mni-spaces/)
- [FMRIB Software Library v6.0](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki)

## Authors
Jacky Ko [jackkykokoko@gmail.com](mailto:jackkykokoko@gmail.com)
