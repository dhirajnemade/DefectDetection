#!/usr/bin/env python
# coding: utf-8

# # CNN RNN hybrid model

# 

# In[1]:


import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, TimeDistributed, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class CNN_RNN_Model:
    def __init__(self, data_dir, image_size=(128, 128), sequence_length=5):
        self.data_dir = data_dir
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.train_images = None
        self.train_labels = None
        self.val_images = None
        self.val_labels = None
        self.train_sequences = None
        self.train_seq_labels = None
        self.val_sequences = None
        self.val_seq_labels = None
        self.model = None

    def is_image_file(self, file_name):
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        return any(file_name.lower().endswith(ext) for ext in valid_extensions)

    def load_data(self):
        categories = os.listdir(self.data_dir)
        images = []
        labels = []

        for label, category in enumerate(categories):
            category_dir = os.path.join(self.data_dir, category)
            if not os.path.isdir(category_dir):
                continue

            for root, _, files in os.walk(category_dir):
                for file_name in files:
                    if not self.is_image_file(file_name):
                        print(f"Skipping non-image file: {file_name}")
                        continue

                    image_path = os.path.join(root, file_name)
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Warning: Could not read image {image_path}. Skipping...")
                        continue

                    image = cv2.resize(image, self.image_size)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    labels.append(label)

        images = np.array(images, dtype='float32') / 255.0
        labels = to_categorical(np.array(labels, dtype='int'), num_classes=len(categories))

        self.train_images, self.val_images, self.train_labels, self.val_labels = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )

        self.train_sequences, self.train_seq_labels = self.create_image_sequences(self.train_images, self.train_labels)
        self.val_sequences, self.val_seq_labels = self.create_image_sequences(self.val_images, self.val_labels)

        print("Dataset loaded successfully")
        print(f"train_sequences shape: {self.train_sequences.shape}")
        print(f"train_seq_labels shape: {self.train_seq_labels.shape}")
        print(f"val_sequences shape: {self.val_sequences.shape}")
        print(f"val_seq_labels shape: {self.val_seq_labels.shape}")

    def create_image_sequences(self, images, labels):
        sequences = []
        sequence_labels = []
        for i in range(len(images) - self.sequence_length + 1):
            sequences.append(images[i:i+self.sequence_length])
            sequence_labels.append(labels[i + self.sequence_length - 1])
        return np.array(sequences), np.array(sequence_labels)

    def create_cnn_rnn_model(self):
        sequence_length = self.sequence_length
        input_shape = (self.image_size[0], self.image_size[1], 3)
        num_classes = self.train_seq_labels.shape[1]

        model = Sequential()
        model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(sequence_length, *input_shape)))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(128, return_sequences=False))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model
        print("Model created successfully")
        self.model.summary()

    def train_model(self, epochs=10, batch_size=32):
        if self.model is None:
            raise ValueError("Model is not created. Call `create_cnn_rnn_model` first.")

        history = self.model.fit(
            self.train_sequences, self.train_seq_labels,
            validation_data=(self.val_sequences, self.val_seq_labels),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        self.model.save("cnn_rnn_model.h5")
        print("Model saved as cnn_rnn_model.h5")
        return history

    def evaluate_model(self):
        if self.model is None:
            raise ValueError("Model is not created. Call `create_cnn_rnn_model` first.")

        val_loss, val_accuracy = self.model.evaluate(self.val_sequences, self.val_seq_labels, verbose=1)
        print(f"Validation Loss: {val_loss}")
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

        return val_loss, val_accuracy

    def plot_training_history(self, history):
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], marker='o', linestyle='-', color='teal', label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], marker='o', linestyle='-', color='orange', label='Validation Accuracy')
        plt.title('Model Accuracy', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], marker='o', linestyle='-', color='teal', label='Train Loss')
        plt.plot(history.history['val_loss'], marker='o', linestyle='-', color='orange', label='Validation Loss')
        plt.title('Model Loss', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix_and_report(self, class_names):
        if self.model is None:
            raise ValueError("Model is not created. Call `create_cnn_rnn_model` first.")

        val_predictions = self.model.predict(self.val_sequences)
        val_predictions_classes = np.argmax(val_predictions, axis=1)
        true_classes = np.argmax(self.val_seq_labels, axis=1)

        conf_matrix = confusion_matrix(true_classes, val_predictions_classes)

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu', cbar=False, 
                    xticklabels=class_names, yticklabels=class_names, linewidths=0.5, linecolor='gray')
        plt.title('Confusion Matrix', fontsize=18, fontweight='bold')
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        print("Classification Report:")
        print(classification_report(true_classes, val_predictions_classes, target_names=class_names))


# In[ ]:




