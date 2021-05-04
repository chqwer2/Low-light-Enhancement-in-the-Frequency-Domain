import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, \
    GlobalAveragePooling2D, AveragePooling2D, MaxPool2D, UpSampling2D, \
    BatchNormalization, Activation, ReLU, Flatten, Dense, Input, \
    Add, Multiply, Concatenate, Softmax, LeakyReLU
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax

tf.keras.backend.set_image_data_format('channels_last')
import tensorflow.keras.backend as K
from .attention import Atten


def concat(layers):
    return tf.concat(layers, axis=-1)

class Rec_Conv_block(tf.keras.layers.Layer):
    def __init__(self, num_filters=200, kernel_size=3, use_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.conv_0 = Conv2D(filters=self.num_filters, kernel_size=1, use_bias=use_bias, padding='same')
        self.conv_1 = Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, use_bias=use_bias, padding='same')
        self.conv_2 = Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, use_bias=use_bias, padding='same')
        self.conv_3 = Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, use_bias=use_bias, padding='same')
        self.conv_4 = Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, use_bias=use_bias, padding='same')
        momentum = 0.99
        self.bn_0 = BatchNormalization(momentum=momentum)
        self.bn_1 = BatchNormalization(momentum=momentum)
        self.bn_2 = BatchNormalization(momentum=momentum)
        self.bn_3 = BatchNormalization(momentum=momentum)
        self.bn_4 = BatchNormalization(momentum=momentum)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size
        })
        return config

    def call(self, x):
        x = self.conv_0(x)
        x = self.bn_0(x)
        x = ReLU()(x)

        X = self.conv_1(x)
        X = self.bn_1(X)
        X = ReLU()(X)
        X = self.conv_2(x + X)
        X = self.bn_2(X)
        X = ReLU()(X)
        X = self.conv_3(x + X)  # add an Conv
        X = self.bn_3(X)
        X = ReLU()(X)
        # X = self.conv_4(x + X)   #
        # X = self.bn_4(X)
        # X = ReLU()(X)

        return X + x


class Asy_Rec_Conv_block(tf.keras.layers.Layer):
    def __init__(self, num_filters=200, kernel_size=3, use_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.conv_0 = Conv2D(filters=self.num_filters, kernel_size=1, use_bias=use_bias, padding='same')
        self.conv_out = Conv2D(filters=self.num_filters, kernel_size=1, use_bias=use_bias, padding='same')
        self.conv_1 = Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, use_bias=use_bias, padding='same')
        self.conv_2 = Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, use_bias=use_bias, padding='same')
        self.conv_3 = Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, use_bias=use_bias, padding='same')
        self.conv_4 = Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, use_bias=use_bias, padding='same')

        self.conv_1h = Conv2D(filters=self.num_filters, kernel_size=[self.kernel_size, 1], use_bias=use_bias,
                              padding='same')
        self.conv_2h = Conv2D(filters=self.num_filters, kernel_size=[self.kernel_size, 1], use_bias=use_bias,
                              padding='same')
        self.conv_3h = Conv2D(filters=self.num_filters, kernel_size=[self.kernel_size, 1], use_bias=use_bias,
                              padding='same')
        self.conv_4h = Conv2D(filters=self.num_filters, kernel_size=[self.kernel_size, 1], use_bias=use_bias,
                              padding='same')
        self.conv_1v = Conv2D(filters=self.num_filters, kernel_size=[1, self.kernel_size], use_bias=use_bias,
                              padding='same')
        self.conv_2v = Conv2D(filters=self.num_filters, kernel_size=[1, self.kernel_size], use_bias=use_bias,
                              padding='same')
        self.conv_3v = Conv2D(filters=self.num_filters, kernel_size=[1, self.kernel_size], use_bias=use_bias,
                              padding='same')
        self.conv_4v = Conv2D(filters=self.num_filters, kernel_size=[1, self.kernel_size], use_bias=use_bias,
                              padding='same')
        momentum = 0.8  # 0.99->0.8
        self.bn_0 = BatchNormalization(momentum=momentum)
        self.bn_out = BatchNormalization(momentum=momentum)
        self.bn_output = BatchNormalization(momentum=momentum)
        self.bn_1 = BatchNormalization(momentum=momentum)
        self.bn_2 = BatchNormalization(momentum=momentum)
        self.bn_3 = BatchNormalization(momentum=momentum)
        self.bn_4 = BatchNormalization(momentum=momentum)

        self.bn_1v = BatchNormalization(momentum=momentum)
        self.bn_2v = BatchNormalization(momentum=momentum)
        self.bn_3v = BatchNormalization(momentum=momentum)
        self.bn_4v = BatchNormalization(momentum=momentum)

        self.bn_1h = BatchNormalization(momentum=momentum)
        self.bn_2h = BatchNormalization(momentum=momentum)
        self.bn_3h = BatchNormalization(momentum=momentum)
        self.bn_4h = BatchNormalization(momentum=momentum)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size
        })
        return config

    def call(self, x):
        x = self.conv_0(x)
        x = self.bn_0(x)
        x = ReLU()(x)

        X = self.conv_1(x)
        X = self.bn_1(X)
        X = ReLU()(X)
        X = self.conv_2(x + X)
        X = self.bn_2(X)
        X = ReLU()(X)
        X = self.conv_3(x + X)  # add an Conv
        X = self.bn_3(X)
        X = ReLU()(X)
        # X = self.conv_4(x + X)   #
        # X = self.bn_4(X)
        # X = ReLU()(X)

        Xv = self.conv_1v(x)
        Xh = self.conv_1h(x)
        Xv = self.bn_1v(Xv + Xh)
        Xv = ReLU()(Xv)
        Xv = self.conv_2v(x + Xv)
        Xh = self.conv_2h(x + Xv)
        Xv = self.bn_2v(Xv + Xh)
        Xv = ReLU()(Xv)
        Xv = self.conv_3v(x + Xv)  # add an Conv
        Xh = self.conv_3h(x + Xv)
        Xv = self.bn_3v(Xv + Xh)
        Xv = ReLU()(Xv)

        X = self.conv_out(X + Xv)
        X = self.bn_out(X)
        X = ReLU()(X)

        return self.bn_output(X + x)


Conv_block = Rec_Conv_block
# Conv_block = Asy_Rec_Conv_block

class DWT_downsampling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        """
        The following calculations for DWT are inspired from,
        https://github.com/AureliePeng/Keras-WaveletTransform/blob/master/models/DWT.py
        """
        x1 = x[:, 0::2, 0::2, :]  # x(2i−1, 2j−1)
        x2 = x[:, 1::2, 0::2, :]  # x(2i, 2j-1)
        x3 = x[:, 0::2, 1::2, :]  # x(2i−1, 2j)
        x4 = x[:, 1::2, 1::2, :]  # x(2i, 2j)

        x_LL = x1 + x2 + x3 + x4
        x_LH = -x1 - x3 + x2 + x4
        x_HL = -x1 + x3 - x2 + x4
        x_HH = x1 - x3 - x2 + x4

        return Concatenate(axis=-1)([x_LL, x_LH, x_HL, x_HH])  # channel*4

class IWT_upsampling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        """
        The following calculations for IWT are inspired from,
        https://github.com/AureliePeng/Keras-WaveletTransform/blob/master/models/DWT.py
        """
        x_LL = x[:, :, :, 0:x.shape[3] // 4]
        x_LH = x[:, :, :, x.shape[3] // 4:x.shape[3] // 4 * 2]
        x_HL = x[:, :, :, x.shape[3] // 4 * 2:x.shape[3] // 4 * 3]
        x_HH = x[:, :, :, x.shape[3] // 4 * 3:]

        x1 = (x_LL - x_LH - x_HL + x_HH) / 4
        x2 = (x_LL - x_LH + x_HL - x_HH) / 4
        x3 = (x_LL + x_LH - x_HL - x_HH) / 4
        x4 = (x_LL + x_LH + x_HL + x_HH) / 4

        y1 = K.stack([x1, x3], axis=2)
        y2 = K.stack([x2, x4], axis=2)
        shape = K.shape(x)
        return K.reshape(K.concatenate([y1, y2], axis=-1),
                         K.stack([shape[0], shape[1] * 2, shape[2] * 2, shape[3] // 4]))


