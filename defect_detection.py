#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from sklearn.model_selection import train_test_split

class DefectDetection:
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size

    def load_images_from_directory(self, directory, label):
        images = []
        labels = []
        for image_name in os.listdir(directory):
            image_path = os.path.join(directory, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, self.target_size)
                image = img_to_array(image) / 255.0  # Normalize the image
                images.append(image)
                labels.append(label)
            else:
                print(f"Warning: Failed to load image {image_name}")
        return np.array(images), np.array(labels)

    def build_cnn_model(self, input_shape):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification: Good or Defected

        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_on_images(self, good_images_dir, defected_images_dir, model_save_path, epochs=10, batch_size=32):
        # Load good and defected images
        good_images, good_labels = self.load_images_from_directory(good_images_dir, label=0)
        defected_images, defected_labels = self.load_images_from_directory(defected_images_dir, label=1)

        # Combine the data
        X = np.concatenate([good_images, defected_images], axis=0)
        y = np.concatenate([good_labels, defected_labels], axis=0)

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        model = self.build_cnn_model((self.target_size[0], self.target_size[1], 3))

        # Using data augmentation to increase variety
        datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, 
                                     shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

        # Train the model with validation
        model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                  validation_data=(X_val, y_val), epochs=epochs)

        model.save(model_save_path)
        print(f"Model trained and saved to {model_save_path}")

    def classify_test_images(self, test_images_dir, model_path):
        model = load_model(model_path)
        print("Model loaded successfully.")

        for image_name in os.listdir(test_images_dir):
            print(f"Processing {image_name}")
            image_path = os.path.join(test_images_dir, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                print(f"Image loaded: {image_name}")
                # Preprocess the image
                image_resized = cv2.resize(image, self.target_size)
                image_resized = img_to_array(image_resized) / 255.0
                image_resized = np.expand_dims(image_resized, axis=0)

                # Classify the image
                prediction = model.predict(image_resized)[0][0]
                label = 'Good' if prediction < 0.5 else 'Defected'

                print(f"Prediction for {image_name}: {label}")
                cv2.putText(image, f"Classification: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (0, 255, 0) if label == 'Good' else (0, 0, 255), 2)

                cv2.imshow(f"Result - {image_name}", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"Warning: Failed to load image {image_name}")


