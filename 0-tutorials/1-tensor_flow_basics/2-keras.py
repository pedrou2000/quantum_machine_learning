import tensorflow as tf
from datetime import datetime



class FlexibleDense(tf.keras.layers.Layer):
  # Note the added `**kwargs`, as Keras supports many arguments
  def __init__(self, out_features, **kwargs):
    super().__init__(**kwargs)
    self.out_features = out_features

  def build(self, input_shape):  # Create the state of the layer (weights)
    print(input_shape)
    self.w = tf.Variable(tf.random.normal([input_shape[-1], self.out_features]), name='w')
    self.b = tf.Variable(tf.zeros([self.out_features]), name='b')

  def call(self, inputs):  # Defines the computation from inputs to outputs
    return tf.matmul(inputs, self.w) + self.b

# Create the instance of the layer
flexible_dense = FlexibleDense(out_features=3)

print(flexible_dense.variables)
print("Model results:", flexible_dense(tf.constant([[2.0, 2.0], [7.0, 8.0],[7.0, 8.0]])))
print(tf.constant([[2.0, 2.0], [7.0, 8.0]]).shape)
