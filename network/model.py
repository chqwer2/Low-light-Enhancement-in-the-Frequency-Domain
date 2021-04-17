from R2_MWCNN import *
from attention import *
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, ReLU, Add, PReLU
from tensorflow.keras.models import Sequential, Model


def Model(use_bias=False):
    input_img = tf.keras.Input(shape=(None, None, 3))  # 150*150*1  Input placeholder

    # input_max = tf.reduce_max(input_img, axis=3, keepdims=True)  # maximum
    input_im = input_img   # concat([input_img, input_max])
    with tf.compat.v1.variable_scope('R2-MWCNN Net'):
        # MWCNN
        cb_1 = Conv_block(num_filters=32)(input_im)
        cb_1 = Conv_block(num_filters=32)(cb_1)
        dwt_1 = DWT_downsampling()(cb_1)

        cb_2 = Conv_block(num_filters=64)(dwt_1)  # 256
        dwt_2 = DWT_downsampling()(cb_2)

        cb_3 = Conv_block(num_filters=64)(dwt_2)
        dwt_3 = DWT_downsampling()(cb_3)

        cb_5 = Conv_block(num_filters=128)(dwt_3)
        cb_5 = Conv_block(num_filters=128)(cb_5)
        cb_5 = Conv2D(filters=256, kernel_size=3, strides=1, use_bias=use_bias, activation=tf.nn.relu,
                      padding='same')(cb_5)
        cb_5 = BatchNormalization(momentum=0.8)(cb_5)
        cb_5 = ReLU()(cb_5)

        up = IWT_upsampling()(cb_5)  # 1024
        up = Conv_block(num_filters=128)(Add()([up, cb_3]))
        up = Conv2D(filters=256, kernel_size=3, strides=1, use_bias=use_bias, activation=tf.nn.relu,
                    padding='same')(up)

        up = IWT_upsampling()(up)
        up = Conv_block(num_filters=128)(Add()([up, cb_2]))
        up = Conv2D(filters=128, kernel_size=3, strides=1, use_bias=use_bias, activation=tf.nn.relu,
                    padding='same')(up)

        up = IWT_upsampling()(up)
        up = Conv_block(num_filters=128)(Add()([up, cb_1]))
        up = Conv2D(filters=128, kernel_size=3, strides=1, use_bias=use_bias, padding='same')(up)
        up = BatchNormalization(momentum=0.8)(up)
        up = ReLU()(up)
        up = Conv2D(filters=128, kernel_size=3, strides=1, use_bias=use_bias, padding='same')(up)

        out = Conv2D(filters=3, kernel_size=(1, 1), use_bias=use_bias, padding="same")(Atten(up))  # features
    model = Model(inputs=input_img, outputs=tf.sigmoid(out), name='R2-MWCNN Net')

    print(model.summary())
    return model