import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, Conv2DTranspose, Flatten, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.datasets import mnist

class SimpleAAN:
    def __init__(self, latent_dim=100, image_shape=(28, 28, 1), learning_rate=0.0002, beta_1=0.5):
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()
   
    def build_generator(self):
        input_layer = Input(shape=(self.latent_dim,))
        x = Dense(128 * 7 * 7, activation="relu")(input_layer)
        x = Reshape((7, 7, 128))(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation="tanh")(x)

        model = Model(input_layer, x)
        return model

    def build_discriminator(self):
        input_layer = Input(shape=self.image_shape)
        x = Conv2D(64, kernel_size=4, strides=2, padding="same")(input_layer)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Flatten()(x)
        x = Dense(100, activation="tanh")(x)
        x = Dense(1, activation="sigmoid")(x)

        model = Model(input_layer, x)
        model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate, beta_1=self.beta_1), metrics=["accuracy"])
        return model


    def build_gan(self):
        self.discriminator.trainable = False
        gan_input = Input(shape=(self.latent_dim,))
        x = self.generator(gan_input)
        gan_output = self.discriminator(x)

        model = Model(gan_input, gan_output)
        model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate, beta_1=self.beta_1))
        return model

    def train(self, X_train, epochs, batch_size=128):
        # Normalize the input images
        X_train = X_train / 127.5 - 1.0
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        for epoch in range(epochs):
            # Train the discriminator
            # Select a random batch of real images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_images = X_train[idx]

            # Generate a batch of fake images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_images = self.generator.predict(noise)

            # Train the discriminator on real and fake images
            d_loss_real = self.discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, real_labels)

            # Print the progress
            print(f"Epoch {epoch + 1}/{epochs} | D loss: {d_loss[0]} | D acc: {d_loss[1] * 100} | G loss: {g_loss}")

    def generate_images(self, num_images=1):
        # Sample noise from a normal distribution
        noise = np.random.normal(0, 1, (num_images, self.latent_dim))

        # Generate images using the trained generator
        generated_images = self.generator.predict(noise)

        # Rescale the images from the range [-1, 1] back to [0, 255]
        generated_images = 0.5 * generated_images + 0.5
        generated_images = (generated_images * 255).astype(np.uint8)

        # Display the generated images
        import matplotlib.pyplot as plt

        nrows, ncols = 2, (num_images + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
        for i, ax in enumerate(axes.flat):
            if i < num_images:
                ax.imshow(generated_images[i, :, :, 0], cmap='gray')
            ax.axis('off')

        plt.show()



# Load the MNIST dataset
(X_train, _), (_, _) = mnist.load_data()
X_train = np.expand_dims(X_train, axis=-1)

# Instantiate and train the SimpleAAN model
simple_aan = SimpleAAN(learning_rate=0.001)
simple_aan.train(X_train, epochs=10000, batch_size=128,)

# Generate and display images using the trained generator
simple_aan.generate_images(num_images=10)