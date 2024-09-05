#!/usr/bin/env python
# coding: utf-8

# # CNN Model

# #install all this matplotlib seaborn scikit-learn using following cammand
# 
# #install using jupiter notebook
# 
# !pip install matplotlib seaborn scikit-learn
# 
# #install using termial
# 
# pip install matplotlib seaborn scikit-learn
# 

# In[5]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class CNNModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._create_cnn_model()

    def _create_cnn_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_images, train_labels, val_images, val_labels, epochs=10, batch_size=32):
        history = self.model.fit(
            train_images, train_labels,
            validation_data=(val_images, val_labels),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        self.model.save("cnn_model.h5")
        return history

    def plot_training_history(self, history):
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], marker='o', linestyle='-', color='dodgerblue', label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], marker='o', linestyle='-', color='coral', label='Validation Accuracy')
        plt.title('Model Accuracy', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], marker='o', linestyle='-', color='dodgerblue', label='Train Loss')
        plt.plot(history.history['val_loss'], marker='o', linestyle='-', color='coral', label='Validation Loss')
        plt.title('Model Loss', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix_and_report(self, val_images, val_labels, class_names):
        val_predictions = self.model.predict(val_images)
        val_predictions_classes = np.argmax(val_predictions, axis=1)
        true_classes = np.argmax(val_labels, axis=1)
        conf_matrix = confusion_matrix(true_classes, val_predictions_classes)

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
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




