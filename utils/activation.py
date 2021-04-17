import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import  LeakyReLU, ReLU, PReLU

def act_type(features, type):
    if type == 'swish':
        if backend.backend() == 'tensorflow':
            try:
                return backend.tf.nn.swish(features)  # backend.
            except AttributeError:
                return features * backend.sigmoid(features)
    elif type == 'hswish':
        return features * tf.nn.relu6(features + 3) / 6
    elif type == 'relu6':
        return tf.nn.relu6(features)
    elif type == 'relu':
        return ReLU()(features)
    elif type == 'prelu':
        return PReLU()(features)  # many parameter
    elif type == 'mish':  # OOM
        return features * backend.tanh(backend.softplus(features))  # ln(1+ex)
    elif type == 'linear':
        return features
    elif type == 'leaky':
        return LeakyReLU()(features)