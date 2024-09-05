#!/usr/bin/env python
# coding: utf-8

# # GAN 

# In[18]:


import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Conv2D, Conv2DTranspose, Activation, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

class GAN:
    def __init__(self, latent_dim, img_shape):
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.channels = img_shape[-1]

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # Build and compile the GAN model
        self.gan = self.build_gan()

    def build_generator(self):
        model = Sequential()

        # Initial dense layer
        model.add(Dense(256 * 8 * 8, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((8, 8, 256)))

        # Upsample to 16x16
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())

        # Upsample to 32x32
        model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())

        # Upsample to 64x64
        model.add(Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())

        # Final upsample to 128x128
        model.add(Conv2DTranspose(self.channels, (4, 4), strides=(2, 2), padding='same'))
        model.add(Activation('tanh'))

        return model

    def build_discriminator(self):
        model = Sequential()

        # First Conv layer
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        # Second Conv layer
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        # Third Conv layer
        model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        # Fourth Conv layer
        model.add(Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        # Flatten and output
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        return model

    def build_gan(self):
        self.discriminator.trainable = False  # Freeze discriminator during GAN training

        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        model.compile(optimizer=Adam(), loss='binary_crossentropy')

        return model

    def train(self, train_images, epochs, batch_size):
        half_batch = batch_size // 2

        for epoch in range(epochs):
            # Train the discriminator with real images
            idx = np.random.randint(0, train_images.shape[0], half_batch)
            real_images = train_images[idx]
            real_labels = np.ones((half_batch, 1))

            # Generate fake images
            noise = np.random.normal(0, 1, (half_batch, self.latent_dim))
            fake_images = self.generator.predict(noise)
            fake_labels = np.zeros((half_batch, 1))

            # Train the discriminator (real classified as 1 and fake as 0)
            d_loss_real = self.discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator via the GAN model (discriminator's weights are frozen)
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            valid_labels = np.ones((batch_size, 1))  # We want the generator to fool the discriminator
            g_loss = self.gan.train_on_batch(noise, valid_labels)

            # Print progress
            if epoch % 100 == 0:
                print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {g_loss}]")

            # Save generated images at certain intervals
            if epoch % 1000 == 0:
                self.save_generated_images(epoch)

    def save_generated_images(self, epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
        noise = np.random.normal(0, 1, (examples, self.latent_dim))
        generated_images = self.generator.predict(noise)
        generated_images = 0.5 * generated_images + 0.5  # Rescale images to 0 - 1

        plt.figure(figsize=figsize)
        for i in range(examples):
            plt.subplot(dim[0], dim[1], i + 1)
            plt.imshow(generated_images[i])
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(f"gan_generated_image_epoch_{epoch}.png")
        plt.close()


# In[ ]:





# In[ ]:




