#os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
# Use export TF_CPP_MIN_LOG_LEVEL="2" to disable warnings.
# In order to use the GPU use: conda activate tf_gpu_env


# Sanity check for validating 
# visibility of the GPUs to TF
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))