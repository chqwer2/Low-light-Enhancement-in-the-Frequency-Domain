import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation, Dense, Reshape, Multiply
import tensorflow.keras.backend as K


def Atten(inputs):
    inputs_channels = int(inputs.shape[-1])
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(int(inputs_channels / 4))(x)
    x = Activation('relu')(x)
    x = Dense(int(inputs_channels))(x)
    x = Activation('softmax')(x)
    x = Reshape((1, 1, inputs_channels))(x)
    x = Multiply()([inputs, x])
    return x


def hard_swish(x):
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0


def relu6(x):
    return tf.nn.relu(x, max_value=6)


def squeeze(inputs):
    input_channels = int(inputs.shape[-1])
    x = GlobalAveragePooling2D()(inputs)

    x = Dense(int(input_channels / 4))(x)
    x = Activation(relu6)(x)

    x = Dense(input_channels)(x)
    x = Activation(hard_swish)(x)

    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])
    return x