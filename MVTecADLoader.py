#!/usr/bin/env python
# coding: utf-8

# # Generative Adversarial Networks for Automated Defect Detection and Quality Control in Industrial Manufacturing
# 

# Loading Dataset 

# In[15]:


# MVTecADLoader.py

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, LSTM, GRU, Reshape, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class MVTecADLoader:
    def __init__(self, data_dir, image_size=(128, 128)):
        self.data_dir = data_dir
        self.image_size = image_size
        self.train_images = None
        self.train_labels = None
        self.val_images = None
        self.val_labels = None

        self._configure_gpu()
        self._load_data()

    def _configure_gpu(self):
        # Ensure TensorFlow uses GPU if available
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU configuration completed.")

    def _is_image_file(self, file_name):
        # Only accept files with image extensions
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        return any(file_name.lower().endswith(ext) for ext in valid_extensions)

    def _load_data(self):
        categories = os.listdir(self.data_dir)
        images = []
        labels = []

        for label, category in enumerate(categories):
            category_dir = os.path.join(self.data_dir, category)
            if not os.path.isdir(category_dir):
                continue  # Skip if it's not a directory

            for root, _, files in os.walk(category_dir):
                for file_name in files:
                    if not self._is_image_file(file_name):
                        print(f"Skipping non-image file: {file_name}")
                        continue

                    image_path = os.path.join(root, file_name)
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Warning: Could not read image {image_path}. Skipping...")
                        continue

                    image = cv2.resize(image, self.image_size)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    images.append(image)
                    labels.append(label)

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int')

        if len(images) == 0:
            raise ValueError("No valid images were found. Please check your dataset directory structure and content.")

        # Normalize images to range [0, 1]
        images /= 255.0

        # Convert labels to one-hot encoding
        labels = to_categorical(labels, num_classes=len(categories))

        # Split into training and validation sets
        self.train_images, self.val_images, self.train_labels, self.val_labels = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )

        print("Dataset loaded successfully")
        print(f"train_images shape: {self.train_images.shape}")
        print(f"train_labels shape: {self.train_labels.shape}")
        print(f"val_images shape: {self.val_images.shape}")
        print(f"val_labels shape: {self.val_labels.shape}")

    def get_data(self):
        return self.train_images, self.train_labels, self.val_images, self.val_labels


# In[ ]:





# In[ ]:




