from .R2_MWCNN import *
from .attention import *
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, ReLU, Add, PReLU
from tensorflow.keras.models import Sequential, Model


def R2MWCNN(use_bias=False):
    input_img = tf.keras.Input(shape=(None, None, 3))  # 150*150*1  Input placeholder
    input_im = input_img

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
    model = Model(inputs=input_img, outputs=tf.sigmoid(out), name='R2-MWCNN')

    print(model.summary())
    return model



def modified_model(use_bias=False):
    input_img = tf.keras.Input(shape=(None, None, 3))  # 150*150*1  Input placeholder
    input_im = input_img

    # MWCNN
    cb_1 = Conv_block(num_filters=32)(input_im)
    cb_1 = Conv_block(num_filters=32)(cb_1)

    dwt_1 = DWT_downsampling()(cb_1)
    dwt1_con = Conv2D(filters=32, kernel_size=2, strides=2, use_bias=use_bias)(cb_1)
    dwt_1 = concat([dwt_1, dwt1_con])

    dwt_1 = Conv2D(filters=64, kernel_size=1, strides=1, use_bias=use_bias, padding='same')(dwt_1)
    dwt_1 = BatchNormalization(momentum=0.9)(dwt_1)
    dwt_1 = ReLU()(dwt_1)
    cb_2 = Conv_block(num_filters=64)(dwt_1)  # 256


    dwt_2 = DWT_downsampling()(cb_2)
    dwt2_con = Conv2D(filters=32, kernel_size=2, strides=2, use_bias=use_bias)(cb_2)
    dwt_2 = concat([dwt_2, dwt2_con])

    dwt_2 = Conv2D(filters=64, kernel_size=1, strides=1, use_bias=use_bias, padding='same')(dwt_2)
    dwt_2 = BatchNormalization(momentum=0.9)(dwt_2)
    dwt_2 = ReLU()(dwt_2)
    cb_3 = Conv_block(num_filters=64)(dwt_2)

    dwt_3 = DWT_downsampling()(cb_3)
    dwt3_con = Conv2D(filters=32, kernel_size=2, strides=2, use_bias=use_bias)(cb_3)
    dwt_3 = concat([dwt_3, dwt3_con])

    cb_5 = Conv_block(num_filters=128)(dwt_3)
    cb_5 = Conv_block(num_filters=128)(cb_5)

    cb_5 = Conv2D(filters=256, kernel_size=3, strides=1, use_bias=use_bias,  # activation=tf.nn.relu,
                  padding='same')(cb_5)
    cb_5 = BatchNormalization(momentum=0.9)(cb_5)
    cb_5 = ReLU()(cb_5)

    up = IWT_upsampling()(cb_5)  # 1024
    up = Conv_block(num_filters=128)(Add()([up, cb_3]))
    up = Conv2D(filters=256, kernel_size=3, strides=1, use_bias=use_bias, activation=None,
                padding='same')(up)
    up = BatchNormalization(momentum=0.9)(up)
    up = ReLU()(up)

    up = IWT_upsampling()(up)
    up = Conv_block(num_filters=128)(Add()([up, cb_2]))
    up = Conv2D(filters=128, kernel_size=3, strides=1, use_bias=use_bias,       # activation=tf.nn.relu,
                padding='same')(up)
    up = BatchNormalization(momentum=0.9)(up)
    up = ReLU()(up)

    up = IWT_upsampling()(up)
    up = Conv_block(num_filters=128)(Add()([up, cb_1]))
    up = Conv2D(filters=128, kernel_size=3, strides=1, use_bias=use_bias, padding='same')(up)
    up = BatchNormalization(momentum=0.9)(up)
    up = ReLU()(up)  # LR
    # up = Dropout(0.2)(up)

    up = Conv2D(filters=64, kernel_size=3, strides=1, use_bias=use_bias, padding='same')(up)
    up = BatchNormalization(momentum=0.9)(up)
    up = ReLU()(up)
    out = Conv2D(filters=3, kernel_size=(1, 1), use_bias=use_bias, padding="same")(Atten(up))  # features

    model = Model(inputs=input_img, outputs=tf.sigmoid(out), name='Nnet')

    print(model.summary())
    return model
