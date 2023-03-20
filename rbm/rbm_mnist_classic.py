import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load and preprocess MNIST data
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data = train_data / 255.0
test_data = test_data / 255.0
train_data = train_data.reshape(-1, 28*28)
test_data = test_data.reshape(-1, 28*28)

# Hyperparameters
learning_rate = 0.1
n_epochs = 25
batch_size = 100
n_hidden = 80

# RBM model
class RBM:
    def __init__(self, visible_size, hidden_size):
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.W = tf.Variable(tf.random.normal([visible_size, hidden_size], stddev=0.01), name="W")
        self.bv = tf.Variable(tf.zeros([visible_size]), name="bv")
        self.bh = tf.Variable(tf.zeros([hidden_size]), name="bh")

    def sample(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs))))

    def forward_pass(self, v):
        return tf.nn.sigmoid(tf.matmul(v, self.W) + self.bh)

    def backward_pass(self, h):
        return tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.bv)

# Training the RBM
rbm = RBM(28*28, n_hidden)

optimizer = tf.keras.optimizers.Adam(learning_rate)

for epoch in range(n_epochs):
    np.random.shuffle(train_data)
    for i in range(0, train_data.shape[0], batch_size):
        batch = train_data[i:i+batch_size]
        with tf.GradientTape() as tape:
            v0 = tf.constant(batch, dtype=tf.float32)
            h0 = rbm.sample(rbm.forward_pass(v0))
            v1 = rbm.sample(rbm.backward_pass(h0))
            h1 = rbm.sample(rbm.forward_pass(v1))

            positive_phase = tf.matmul(tf.transpose(v0), h0)
            negative_phase = tf.matmul(tf.transpose(v1), h1)
            loss = tf.reduce_mean(positive_phase - negative_phase)

        gradients = tape.gradient(loss, [rbm.W, rbm.bv, rbm.bh])
        optimizer.apply_gradients(zip(gradients, [rbm.W, rbm.bv, rbm.bh]))
    print("Epoch: {}, Batch: {}, Loss: {}".format(epoch + 1, i // batch_size + 1, loss.numpy()))
