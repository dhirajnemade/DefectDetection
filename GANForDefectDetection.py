#!/usr/bin/env python
# coding: utf-8

# # GNN CNN-RNN

# ..

# In[16]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, LeakyReLU, Flatten, Reshape, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

class GANForDefectDetection:
    def __init__(self, data_dir, image_size=(64, 64), noise_dim=100):
        self.data_dir = data_dir
        self.image_size = image_size
        self.noise_dim = noise_dim
        self.optimizer = Adam(0.0002, 0.5)
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan(self.generator, self.discriminator)
        self.gan.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        
    def load_data(self):
        images = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(('jpg', 'png', 'jpeg', 'bmp', 'tiff')):
                    image_path = os.path.join(root, file)
                    image = cv2.imread(image_path)
                    if image is not None:
                        image = cv2.resize(image, self.image_size)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        images.append(image)
        images = np.array(images, dtype='float32')
        images = (images - 127.5) / 127.5  # Normalize images to [-1, 1]
        return images

    def build_generator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.noise_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(np.prod(self.image_size) * 3, activation='tanh'))
        model.add(Reshape((*self.image_size, 3)))
        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=(*self.image_size, 3), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        return model

    def build_gan(self, generator, discriminator):
        discriminator.trainable = False
        gan_input = Input(shape=(self.noise_dim,))
        img = generator(gan_input)
        gan_output = discriminator(img)
        gan = Model(gan_input, gan_output)
        return gan

    def train(self, epochs=10000, batch_size=32, save_interval=200):
        images = self.load_data()
        half_batch = int(batch_size / 2)
        
        for epoch in range(epochs):
            # Train Discriminator
            idx = np.random.randint(0, images.shape[0], half_batch)
            real_images = images[idx]

            noise = np.random.normal(0, 1, (half_batch, self.noise_dim))
            fake_images = self.generator.predict(noise)

            real_labels = np.ones((half_batch, 1))
            fake_labels = np.zeros((half_batch, 1))

            d_loss_real = self.discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            valid_y = np.ones((batch_size, 1))

            g_loss = self.gan.train_on_batch(noise, valid_y)

            # Print the progress
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}] [G loss: {g_loss}]")

            # Save generated image samples
            if epoch % save_interval == 0:
                self.save_images(epoch)

    def save_images(self, epoch, examples=5):
        noise = np.random.normal(0, 1, (examples, self.noise_dim))
        gen_images = self.generator.predict(noise)
        gen_images = 0.5 * gen_images + 0.5  # Rescale images from [-1, 1] to [0, 1]

        fig, axs = plt.subplots(1, examples, figsize=(15, 3))
        for i in range(examples):
            axs[i].imshow(gen_images[i])
            axs[i].axis('off')
        plt.show()
        fig.savefig(f"gan_generated_image_epoch_{epoch}.png")
        plt.close()
        
    def evaluate(self):
        images = self.load_data()
        noise = np.random.normal(0, 1, (images.shape[0], self.noise_dim))
        fake_images = self.generator.predict(noise)
        
        real_labels = np.ones((images.shape[0], 1))
        fake_labels = np.zeros((images.shape[0], 1))
        
        real_loss, real_acc = self.discriminator.evaluate(images, real_labels, verbose=0)
        fake_loss, fake_acc = self.discriminator.evaluate(fake_images, fake_labels, verbose=0)
        
        print(f"Real Image Evaluation -> Loss: {real_loss:.4f}, Accuracy: {100 * real_acc:.2f}%")
        print(f"Fake Image Evaluation -> Loss: {fake_loss:.4f}, Accuracy: {100 * fake_acc:.2f}%")
        return real_loss, real_acc, fake_loss, fake_acc




# In[ ]:





# In[ ]:




