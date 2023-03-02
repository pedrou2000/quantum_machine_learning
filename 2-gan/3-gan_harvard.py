# source: https://harvard-iacs.github.io/2019-CS109B/labs/lab11/GANS-sol/

import tensorflow as tf
import numpy
from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout
from keras.layers import LSTM, SimpleRNN, Input
from keras.layers import Flatten
from keras.preprocessing import sequence
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import numpy as np
import matplotlib.pyplot as plt
import scipy
# fix random seed for reproducibility
numpy.random.seed(1)



def generate_data(mean, stddev, n_samples = 10000, n_dim=1):
    #data = np.random.rand(n_samples, n_dim)
    data =  tf.random.normal([n_samples,n_dim], mean=mean, stddev=stddev).numpy()
    #print(data)
    return data  

def set_model(input_dim, output_dim, hidden_dim=64,n_layers = 1,activation='tanh',optimizer='adam', loss = 'binary_crossentropy'):
  model = Sequential()
  model.add(Dense(hidden_dim,input_dim=input_dim,activation=activation))
  
  for _ in range(n_layers-1):
    model.add(Dense(hidden_dim),activation=activation)
  model.add(Dense(output_dim))
  
  model.compile(loss=loss, optimizer=optimizer)
  print(model.summary())
  return model

def get_gan_network(discriminator, random_dim, generator, optimizer = 'adam'):
  discriminator.trainable = False
  gan_input = Input(shape=(random_dim,))
  x = generator(gan_input)
  gan_output = discriminator(x)
  gan = Model(inputs = gan_input,outputs=gan_output)
  gan.compile( loss='binary_crossentropy', optimizer=optimizer)
  return gan

NOISE_DIM = 10
DATA_DIM = 1
G_LAYERS = 1
D_LAYERS = 1

def train_gan(mean, stddev, epochs=1,batch_size=128):
  x_train = generate_data(mean, stddev,n_samples=12800,n_dim=DATA_DIM)
  batch_count = x_train.shape[0]/batch_size
  
  generator = set_model(NOISE_DIM, DATA_DIM, n_layers=G_LAYERS, activation='tanh',loss = 'mean_squared_error')
  discriminator = set_model(DATA_DIM, 1, n_layers= D_LAYERS, activation='sigmoid')
  gan = get_gan_network(discriminator, NOISE_DIM, generator, 'adam')
  
  for e in range(1,epochs+1):   
    
    # Noise is generated from a uniform distribution
    noise = np.random.rand(batch_size,NOISE_DIM)
    true_batch = x_train[np.random.choice(x_train.shape[0], batch_size, replace=False), :]
    
    generated_values = generator.predict(noise)
    X = np.concatenate([generated_values,true_batch])
    
    y_dis = np.zeros(2*batch_size)
    
    #One-sided label smoothing to avoid overconfidence. In GAN, if the discriminator depends on a small set of features to detect real images, 
    #the generator may just produce these features only to exploit the discriminator. 
    #The optimization may turn too greedy and produces no long term benefit.
    #To avoid the problem, we penalize the discriminator when the prediction for any real images go beyond 0.9 (D(real image)>0.9). 
    y_dis[:batch_size] = 0.9
    
    discriminator.trainable = True
    disc_history = discriminator.train_on_batch(X, y_dis)
    discriminator.trainable = False

    # Train generator
    # Noise is generated from a uniform distribution
    noise = np.random.rand(batch_size,NOISE_DIM)
    y_gen = np.zeros(batch_size)    
    gan.train_on_batch(noise, y_gen)  
 
    
  return generator, discriminator


mean = 0
stddev = 1

generator, discriminator = train_gan(mean, stddev)

noise = np.random.rand(10000,NOISE_DIM)
generated_values = generator.predict(noise)
plt.hist(generate_data(mean, stddev, n_samples=12800,n_dim=DATA_DIM),bins=100, color='blue')
plt.hist(generated_values,bins=100, color='red')
plt.show()
