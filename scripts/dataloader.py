import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import random

# File paths
TRAIN_FILE_PATH = "../competition_data./train.csv"
VAL_FILE_PATH = "../competition_data./val.csv"
TEST_FILE_PATH = "testKaggle.csv" # when runnning test.py, results are wrong because it uses different solutions. to get accurate solutions, link dataset who comes with get_dataset.py in this location "../competition_data./test.csv"
SOLUTION_FILE_PATH = "../competition_data./solution.csv"

# TEST_FILE_PATH = "../competition_data./test.csv"

# adding 70000 more augmented images
NUM_TRANSLATION_AUG = 7000
NUM_ROTATION_AUG = 7000
NUM_FLIP_AUG = 7000
NUM_MINOR_SHIFT_AUG = 7000
NUM_COLOR_AUG = 7000
NUM_PERSPECTIVE_AUG = 7000




class DatasetLoader:
    def __init__(self):
        self.train_images, self.train_labels = self._load_dataset(TRAIN_FILE_PATH, augment=True)
        self.val_images, self.val_labels = self._load_dataset(VAL_FILE_PATH, augment=False)
        self.test_images, self.test_labels, self.test_ids = self._load_test_dataset()

    # function to read csv files, extract images and labels, notmilizee the image data and apply augmentation to the training dataset
    def _load_dataset(self, file_path, augment=False):

        file = pd.read_csv(file_path)

        # extract labels
        labels = file.iloc[:, 1].values

        # extract image data
        images = file.iloc[:, 2:].values.reshape(-1, 32, 32, 3).astype(np.float32)

        # normalize pixel values to [0,1]
        images /= 255.0

        # convert to PyTorch format
        images = np.transpose(images, (0, 3, 1, 2))

        # convert to tensors
        data_tensor = torch.tensor(images, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        if augment: # augment dataset. applies only to training dataset
            data_tensor, labels_tensor = self._apply_augmentation(data_tensor, labels_tensor)

        return data_tensor, labels_tensor

    # function to apply all the augmentations. it copies the passed in dataset, applies augmentation to random indexes from the specified amount and appends them to the copied data set and returns it.
    def _apply_augmentation(self, images, labels):
        augmented_images = []
        augmented_labels = []

        translation_transform = transforms.RandomAffine(degrees=0, translate=(4/32, 4/32))
        rotation_transform = transforms.RandomAffine(degrees=30, translate=(0, 0))
        flip_transform = transforms.RandomHorizontalFlip(p=0.5)
        minor_shift_transform = transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))
        color_jitter_transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        perspective_transform = transforms.RandomPerspective(distortion_scale=0.1, p=0.5)

        translation_indices = random.sample(range(len(images)), NUM_TRANSLATION_AUG)
        rotation_indices = random.sample(range(len(images)), NUM_ROTATION_AUG)
        flip_indices = random.sample(range(len(images)), NUM_FLIP_AUG)
        minor_shift_indices = random.sample(range(len(images)), NUM_MINOR_SHIFT_AUG)
        color_jitter_indices = random.sample(range(len(images)), NUM_COLOR_AUG)
        perspective_indices = random.sample(range(len(images)), NUM_PERSPECTIVE_AUG)

        augmented_images.extend(images)
        augmented_labels.extend(labels)

        for idx in translation_indices:
            augmented_images.append(translation_transform(images[idx]))
            augmented_labels.append(labels[idx])

        for idx in rotation_indices:
            augmented_images.append(rotation_transform(images[idx]))
            augmented_labels.append(labels[idx])

        for idx in flip_indices:
            augmented_images.append(flip_transform(images[idx]))
            augmented_labels.append(labels[idx])

        for idx in minor_shift_indices:
            augmented_images.append(minor_shift_transform(images[idx]))
            augmented_labels.append(labels[idx])

        for idx in color_jitter_indices:
            augmented_images.append(color_jitter_transform(images[idx]))
            augmented_labels.append(labels[idx])

        for idx in perspective_indices:
            augmented_images.append(perspective_transform(images[idx]))
            augmented_labels.append(labels[idx])


        return torch.stack(augmented_images), torch.tensor(augmented_labels, dtype=torch.long)

    # function to load the test dataset
    def _load_test_dataset(self):

        test_file = pd.read_csv(TEST_FILE_PATH)
        test_ids = test_file.iloc[:, 0].values
        images = test_file.iloc[:, 1:].values.reshape(-1, 32, 32, 3).astype(np.float32)

        images /= 255.0
        images = np.transpose(images, (0, 3, 1, 2))

        test_images = torch.tensor(images, dtype=torch.float32)

        solution_df = pd.read_csv(SOLUTION_FILE_PATH)
        actual_labels = solution_df.iloc[:, 1].values
        test_labels = torch.tensor(actual_labels, dtype=torch.long)

        return test_images, test_labels, test_ids