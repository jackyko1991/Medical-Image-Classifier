# Medical-Image-Classifier
Neural network classifier for medical image using Tensorflow

## Introduction
This is a Tensorflow implementation for medical image classification tasks. The repository integrates a pre-processing pipeline optimized for 3D imaging data that suitable for multiple images input with multi-class output. The original application of this framework is to detect intracranial hemorrhage types with 3D CT images, however with slightly modification the framework is general applicable for all medical image classification tasks.

### Features
- 3D data processing ready
- Augumented patching technique, requires less image input for training
- Multichannel input and multiclass output
- Generic image reader with SimpleITK support (Currently only support .nii/.nii.gz format for convenience, easy to expand to DICOM, tiff and jpg format)
- Medical image pre-post processing with SimpleITK filters
- Easy network replacement structure

### Networks
Currently the classifier provides following classification networks:
- LeNet

Following networks are to be developed:
- AlexNet
- VGG
- ResNet

## Usage
### Required Libraries
Known good dependencies:
- Anaconda 3.6
- Tensorflow 1.14
- SimpleITK

### Software Configuration
All necessary network configurations are stored in `config.json` except for GPU specification. Modify the `config.json` file to fit your application.

### Example usage
#### Folder Hierarchy
All training, testing and evaluation data should put in the directory that specified in `config.json`. Note that you should also provide the class labels in CSV format file.

In the default `config.json`, the data folder structure is as following:
```
.										# Repository root
├── ...
├── data                      
│   └── example							# Data directory
│       ├── training 					# Put all training data here
│       │   ├── case1            		# foldername for the cases is arbitrary
│       │   |   └── image_brain_mni.nii # The image name is specified in config.json
│       │   ├── case2
│       │   |   └── image_brain_mni.nii
│       │   ├──	...
│       ├── testing              		# Put all testing data here
│       │   ├── case1            		# follow the same folder structure as the training one
│       │   |   └── image_brain_mni.nii
│       │   ├── case2
│       │   |   └── image_brain_mni.nii
│       │   ├── ...
│       ├── evaluation					# Put all evaluation data here
│       │   ├── image1.nii.gz           # The image name are arbitrary
│       │   ├──	image2.nii.gz
│       │   ├──	...
│       └── labels.csv					# CSV file stores all labels for training and testing
├── tmp
│   ├── cktp                  			# Tensorflow checkpoints
|   └── log                   			# Tensorboard logging folder
├── ...
```

In training phase, the the program will automatically scan all the data in training and testing folder. The case/subject name is identified by subfolder's name and all of them should also be listed in `labels.csv`. Filenames could be altered in `config.csv`.

An example dataset folder is provided in [./data/example](./data/example).

#### Data preprocessing

#### Training

#### Evaluate

## References

## Author
Jacky Ko [jackkykokoko@gmail.com](mailto:jackkykokoko@gmail.com)