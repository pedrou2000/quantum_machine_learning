import tensorflow as tf
from datetime import datetime
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
# Use export TF_CPP_MIN_LOG_LEVEL="2" to disable warnings.

class Dense(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(tf.random.normal([in_features, out_features]), name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')
        
    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)

class SequentialModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.dense_1 = Dense(in_features=3, out_features=3)
        self.dense_2 = Dense(in_features=3, out_features=2)

    def __call__(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)

# You have made a model!
my_model = SequentialModule(name="the_model")

# Call it, with random results
print("Model results:", my_model(tf.constant([[2.0, 2.0, 2.0]])))

print("Submodules:", my_model.submodules)

for var in my_model.variables:
    print(var, "\n")  

chkp_path = "my_checkpoint/a"
checkpoint = tf.train.Checkpoint(model=my_model)
checkpoint.write(chkp_path)