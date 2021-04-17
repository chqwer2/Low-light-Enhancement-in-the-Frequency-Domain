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


class Nor_Conv_block(tf.keras.layers.Layer):
    def __init__(self, num_filters=200, kernel_size=3, use_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_1 = Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, use_bias=use_bias, padding='same')
        self.conv_2 = Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, use_bias=use_bias, padding='same')
        self.conv_3 = Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, use_bias=use_bias, padding='same')
        self.conv_4 = Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, use_bias=use_bias, padding='same')

        self.bn_1 = BatchNormalization(momentum=0.8)
        self.bn_2 = BatchNormalization(momentum=0.8)
        self.bn_3 = BatchNormalization(momentum=0.8)
        self.bn_4 = BatchNormalization(momentum=0.8)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size
        })
        return config

    def call(self, X):
        X = self.conv_1(X)
        X = self.bn_1(X)
        X = ReLU()(X)
        X = self.conv_2(X)
        X = self.bn_2(X)
        X = ReLU()(X)
        X = self.conv_3(X)
        X = self.bn_3(X)
        X = ReLU()(X)
        # X = self.conv_4(X)
        # # X = self.bn_4(X)
        # X = ReLU()(X)

        return X


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


#
# Conv_block = Rec_Conv_block
Conv_block = Nor_Conv_block


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


def MWCNN(input_L, input_R, input_img):
    # tf.keras.backend.clear_session()
    use_bias = False  #
    input = concat([input_R, input_L])  # （400， 600） , input_img
    with tf.compat.v1.variable_scope('RelightNet'):
        cb_1 = Conv_block(num_filters=64)(input)
        dwt_1 = DWT_downsampling()(cb_1)

        cb_2 = Conv_block(num_filters=128)(dwt_1)  # 256
        dwt_2 = DWT_downsampling()(cb_2)

        cb_3 = Conv_block(num_filters=256)(dwt_2)
        dwt_3 = DWT_downsampling()(cb_3)

        # cb_4 = Conv_block(num_filters=64)(dwt_3)
        # dwt_4 = DWT_downsampling()(cb_4)      # 37 vs 38

        cb_5 = Conv_block(num_filters=512)(dwt_3)
        cb_5 = BatchNormalization(momentum=0.8)(cb_5)
        cb_5 = ReLU()(cb_5)
        cb_5 = Conv_block(num_filters=512)(cb_5)
        cb_5 = Conv2D(filters=1024, kernel_size=3, strides=1, use_bias=use_bias, activation=tf.nn.relu, padding='same')(
            cb_5)

        # up = IWT_upsampling()(cb_5)
        # up = Conv_block(num_filters=64)(Add()([up, cb_3]))
        # up = Conv2D(filters=256, kernel_size=3, strides=1, use_bias=use_bias, activation=tf.nn.relu, padding='same')(up)

        up = IWT_upsampling()(cb_5)  # 1024
        up = Conv_block(num_filters=256)(Add()([up, cb_3]))
        up = Conv2D(filters=512, kernel_size=3, strides=1, use_bias=use_bias, activation=tf.nn.relu, padding='same')(up)

        up = IWT_upsampling()(up)
        up = Conv_block(num_filters=256)(Add()([up, cb_2]))
        up = Conv2D(filters=256, kernel_size=3, strides=1, use_bias=use_bias, activation=tf.nn.relu, padding='same')(up)

        up = IWT_upsampling()(up)
        up = Conv_block(num_filters=256)(Add()([up, cb_1]))
        up = Conv2D(filters=256, kernel_size=3, strides=1, use_bias=use_bias, padding='same')(up)

        out = Conv2D(filters=1, kernel_size=(1, 1), use_bias=use_bias, padding="same")(Atten(up))

    return tf.sigmoid(out)


def MWCNN_Decom(input_img, layer_num):
    channel = 64
    kernel_size = 3
    use_bias = False  #
    input_max = tf.reduce_max(input_img, axis=3, keepdims=True)  # maximum
    input_im = concat([input_img, input_max])

    with tf.compat.v1.variable_scope('DecomNet', reuse=tf.compat.v1.AUTO_REUSE):
        conv = tf.compat.v1.layers.conv2d(input_im, channel, kernel_size,
                                          padding='same', activation=None,
                                          name="shallow_feature_extraction")
        conv1 = conv
        for idx in range(layer_num):
            conv = tf.compat.v1.layers.conv2d(conv, channel, kernel_size, use_bias=False,
                                              # activation=tf.nn.relu,
                                              # kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                                              padding='same', name='activated_layer_%d' % idx)

            if idx == 4:
                conv = Add()([conv, conv1])

            BN = tf.compat.v1.layers.BatchNormalization(momentum=0.8, trainable=True)(conv)
            # conv = tf.nn.relu(BN, 'relu')
            conv = LeakyReLU()(BN)

        conv = Conv2D(filters=4, kernel_size=1, strides=1, use_bias=use_bias, padding='same')(conv)
        features = conv

        # remove noise  Nor_
        cb_1 = Conv_block(num_filters=64)(input_im)
        dwt_1 = DWT_downsampling()(cb_1)

        # cb_2 = Conv_block(num_filters=64)(dwt_1)  # 256
        # dwt_2 = DWT_downsampling()(cb_2)

        cb_5 = Conv_block(num_filters=256)(dwt_1)
        cb_5 = BatchNormalization(momentum=0.8)(cb_5)
        cb_5 = ReLU()(cb_5)
        cb_5 = Conv_block(num_filters=256)(cb_5)
        cb_5 = BatchNormalization(momentum=0.8)(cb_5)
        cb_5 = ReLU()(cb_5)

        # up = IWT_upsampling()(cb_5)  #1024

        # up = Conv_block(num_filters=64)(Add()([up, cb_2]))
        # up = Conv2D(filters=256, kernel_size=3, strides=1, use_bias=use_bias, activation=tf.nn.relu, padding='same')(up)

        up = IWT_upsampling()(cb_5)
        up = Conv_block(num_filters=64)(Add()([up, cb_1]))

        up = Conv2D(filters=4, kernel_size=3, strides=1, use_bias=use_bias, padding='same')(up)

        out = Conv2D(filters=4, kernel_size=(1, 1), use_bias=use_bias, padding="same")(features - up)
        # out = Conv2D(filters=4, kernel_size=(1, 1), use_bias=use_bias, padding="same")()

    R = tf.sigmoid(out[:, :, :, 0:3])  # R o L
    L = tf.sigmoid(out[:, :, :, 3:4])

    return R, L


def MWCNN_Color(input_img, input_I):
    channel = 64
    kernel_size = 3
    use_bias = False  #

    with tf.compat.v1.variable_scope('ColorNet', reuse=tf.compat.v1.AUTO_REUSE):
        input_max = tf.reduce_max(input_img, axis=3, keepdims=True)  # maximum
        input_im = concat([input_img, input_I, input_max])
        cb_1 = Conv_block(num_filters=64)(input_im)
        dwt_1 = DWT_downsampling()(cb_1)

        cb_2 = Conv_block(num_filters=128)(dwt_1)  # 256
        dwt_2 = DWT_downsampling()(cb_2)

        cb_3 = Conv_block(num_filters=128)(dwt_2)
        dwt_3 = DWT_downsampling()(cb_3)

        # cb_4 = Conv_block(num_filters=64)(dwt_3)
        # dwt_4 = DWT_downsampling()(cb_4)      # 37 vs 38

        cb_5 = Conv_block(num_filters=256)(dwt_3)
        cb_5 = Conv_block(num_filters=256)(cb_5)
        cb_5 = Conv2D(filters=512, kernel_size=3, strides=1, use_bias=use_bias, activation=tf.nn.relu, padding='same')(
            cb_5)

        up = IWT_upsampling()(cb_5)  # 1024
        up = Conv_block(num_filters=256)(Add()([up, cb_3]))
        up = Conv2D(filters=512, kernel_size=3, strides=1, use_bias=use_bias, activation=tf.nn.relu, padding='same')(up)

        up = IWT_upsampling()(up)
        up = Conv_block(num_filters=256)(Add()([up, cb_2]))
        up = Conv2D(filters=256, kernel_size=3, strides=1, use_bias=use_bias, activation=tf.nn.relu, padding='same')(up)

        up = IWT_upsampling()(up)
        up = Conv_block(num_filters=256)(Add()([up, cb_1]))
        up = Conv2D(filters=256, kernel_size=3, strides=1, use_bias=use_bias, padding='same')(up)

        out = Conv2D(filters=3, kernel_size=(1, 1), use_bias=use_bias, padding="same")(up)

    return tf.sigmoid(out)
