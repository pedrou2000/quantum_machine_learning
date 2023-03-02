import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
# Call model on a test input
x = tf.ones((2, 3))
y = model(x)


# Create 3 layers
layer1 = layers.Dense(2, activation="relu", name="layer1")
layer2 = layers.Dense(3, activation="relu", name="layer2")
layer3 = layers.Dense(4, name="layer3")

# Call layers on a test input
x = tf.ones((4, 3))
print(x)
y = layer1(x)
print(layer1.weights)
print(y)
y = layer2(y)
print(layer2.weights)
print(y)
y = layer3(y)
print(layer3.weights)
print(y)



#print(layers.Dense(2, activation="relu", name="layer1"))
"""
print(x)
print(model.layers[0](x))
print(model.layers[0].weights)
print()
print()
print(y)
print(model.weights)
"""
