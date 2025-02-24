
# Custom CNN for the CIFAR-10 dataset

## Overview
In this project, I designed a convolutional neural network inspired by the [VGG](https://arxiv.org/abs/1409.1556) architecture for classifying 32x32 images into 10 different categories from The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. Additionally, I participated in a Kaggle competition where the final score was determined by dividing the **average accuracy by the average latency**, making it crucial to optimize both accuracy and inference speed. I have finished third.

---
## how to run:
### Clone The Repo:
```bash
git clone https://github.com/markshperkin/CustomCNNforCIFAR-10.git
cd CustomCNNforCIFAR-10
```
### *RECOMMENDED* Create a virtual environment with environment.yml file using conda
```bash
conda env create -f environment.yml
conda activate assignment1_2
```
### *OPTIONAL* install and use Cuda for Nvidia GPU

### Download the dataset and split the dataset into train, validation, and test datasets.
```bash
python .\get_dataset.py
```
this will create two new directories. the split dataset will be in competition_data

### Enter the functions directory
```bash
cd scripts
```

### Start training
```bash
python .\train.py
```

### Start testing
```bash
python .\test.py
```
---
## Class Project

This project was developed as part of the Edge Neumorphic computing class under the instruction of [Professor Ramtin Zand](https://sc.edu/study/colleges_schools/engineering_and_computing/faculty-staff/zand.php) and Teacher assistant [Peyton Chandarana](https://www.peytonsc.com/) at the University of South Carolina.


