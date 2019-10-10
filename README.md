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
### Folder Hierarchy
All training, testing and evaluation data should put in the directory that specified in `config.json`. Note that you should also provide the class labels in CSV format file.

In the default `config.json`, the data folder structure is as following:
```
.									# Repository root
├── ...
├── data                      
│  	└── dataset						# Data directory
│   	├── training 				# Put all training data here
|   	├── case1            
|   	|   └── image_brain_mni.nii # The image name is specified in config.json
|       ├── case2
│		|   └── image_brain_mni.nii
|   	├──	...
│   ├── training              # Put all training data here
|   |   ├── case1             # foldername for the cases is arbitary
|   |   |   ├── img.nii.gz    # Image for training
|   |   |   └── label.nii.gz  # Corresponding label for training
|   |   ├── case2
|   |   ├──...
│   └── evaluation            # Put all evaluation data here
|   |   ├── case1             # foldername for the cases is arbitary
|   |   |   └── img.nii.gz    # Image for evaluation
|   |   ├── case2
|   |   ├──...
├── tmp
|   ├── cktp                  # Tensorflow checkpoints
|   └── log                   # Tensorboard logging folder
├── ...
```

#### Data preprocessing

#### Training

#### Evaluate

## References

## Author
Jacky Ko [jackkykokoko@gmail.com](mailto:jackkykokoko@gmail.com)