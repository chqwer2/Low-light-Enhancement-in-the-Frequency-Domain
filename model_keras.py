import matplotlib.pyplot as plt
from utils import mkdir, load_images, fft_np, ifft_np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.keras import Input, Model, layers
import numpy as np


def concat(layers):
    return tf.concat(layers, axis=3)


def exponential_decay_with_warmup(global_step, ):
    warmup_step = 10
    learning_rate_base = 0.01  #0.1
    learning_rate_step = 1
    learning_rate_decay = 0.8  # 0.96 for 200, 0.8 for 50
    staircase = False
    with tf.name_scope("exponential_decay_with_warmup"):
        decayed_lr = learning_rate_schedule.ExponentialDecay(learning_rate_base, learning_rate_step,
                                                             learning_rate_decay, staircase=staircase)
        linear_increase = learning_rate_base * tf.cast(global_step / warmup_step, tf.float32)

        exponential_decay = decayed_lr(global_step - 10)
        learning_rate = tf.cond(global_step <= warmup_step,
                                lambda: linear_increase.numpy(),
                                lambda: exponential_decay.numpy())

        learning_rate = tf.cond(learning_rate <= tf.constant(0.001),
                                lambda: tf.constant(0.001),
                                lambda: learning_rate)
        return learning_rate


def keras_RelightNet(channel=64, kernel_size=3):  # u-net
    input = Input(shape=[None, None, 1])

    conv0 = layers.Conv2D(channel, kernel_size, padding='same')(input)
    conv0 = layers.BatchNormalization(momentum=0.8, trainable=True)(conv0)
    conv0 = layers.ReLU()(conv0)

    conv1 = layers.Conv2D(channel, kernel_size, strides=2, padding='same',
                          activation=tf.nn.relu)(conv0, )
    conv1 = layers.BatchNormalization(momentum=0.8, trainable=True)(conv1)
    conv1 = layers.ReLU()(conv1)

    conv2 = layers.Conv2D(channel, kernel_size, strides=2, padding='same')(conv1, )
    conv2 = layers.BatchNormalization(momentum=0.8, trainable=True)(conv2)
    conv2 = layers.ReLU()(conv2)

    conv3 = layers.Conv2D(channel, kernel_size, strides=2, padding='same', )(conv2, )
    conv3 = layers.BatchNormalization(momentum=0.8, trainable=True)(conv3)
    conv3 = layers.ReLU()(conv3)

    up1 = tf.compat.v1.image.resize_nearest_neighbor(conv3, (tf.shape(conv2)[1], tf.shape(conv2)[2]))
    deconv1 = layers.Conv2D(channel, kernel_size, padding='same')(up1, )
    deconv1 = layers.BatchNormalization(momentum=0.8, trainable=True)(deconv1)
    deconv1 = layers.ReLU()(deconv1)+ conv2

    up2 = tf.compat.v1.image.resize_nearest_neighbor(deconv1, (tf.shape(conv1)[1], tf.shape(conv1)[2]))
    deconv2 = layers.Conv2D(channel, kernel_size, padding='same')(up2, )
    deconv2 = layers.BatchNormalization(momentum=0.8, trainable=True)(deconv2)
    deconv2 = layers.ReLU()(deconv2) + conv1

    up3 = tf.compat.v1.image.resize_nearest_neighbor(deconv2, (tf.shape(conv0)[1], tf.shape(conv0)[2]))
    deconv3 = layers.Conv2D(channel, kernel_size, padding='same')(up3, )
    deconv3 = layers.BatchNormalization(momentum=0.8, trainable=True)(deconv3)
    deconv3 = layers.ReLU()(deconv3) + conv0

    deconv1_resize = tf.compat.v1.image.resize_nearest_neighbor(deconv1,
                                                                (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
    deconv2_resize = tf.compat.v1.image.resize_nearest_neighbor(deconv2,
                                                                (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
    feature_gather = concat([deconv1_resize, deconv2_resize, deconv3])
    feature_fusion = layers.Conv2D(channel, 1, padding='same', activation=tf.nn.relu)(feature_gather, )
    output_mag = layers.Conv2D(1, 1, padding='same', activation=tf.nn.relu)(feature_fusion, )

    model = Model(input, output_mag)
    return model


def fft_train():
    model = keras_RelightNet()

    mag_h = np.load('./Data/high/high_mag.npz')['arr_0']
    ang_h = np.load('./Data/high/high_ang.npz')['arr_0']
    mag_l = np.load('./Data/low/low_mag.npz')['arr_0']
    ang_l = np.load('./Data/low/low_ang.npz')['arr_0']
    # pic = np.max(np.load('./Data/high/high.npz')['arr_0'], axis=3)
    # min, max-min
    scaler_l = np.load('./Data/low/low_scaler.npy')
    scaler_h = np.load('./Data/high/high_scaler.npy')

    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, decay=1e-4, epsilon=1e-7),
                  loss=keras.losses.MeanSquaredError(), metrics=['mse'])
    # reduceLR = [LearningRateScheduler(exponential_decay_with_warmup)]
    model.fit(x=mag_l, y=mag_h, epochs=100, batch_size=16, validation_split=0.02,  # callbacks=reduceLR,
              shuffle=True, workers=4, use_multiprocessing=True)

    mag = np.squeeze(model.predict(mag_l))
    print(scaler_l) # not so much difference
    print(scaler_h)
    print(mag_l[0, 0, 0:3])
    print(mag[0, 0, 0:3]) # 0.018
    print(mag_h[0, 0, 0:3]) # 0.32
    pic_recover = ifft_np(mag[:10] * scaler_l[1] + scaler_l[0], ang_l[:10] * scaler_l[3] + scaler_l[2])
    pic = ifft_np(mag_h[:10] * scaler_h[1] + scaler_h[0], ang_h[:10] * scaler_h[3] + scaler_h[2])
    return pic, pic_recover


def img_train():
    model = keras_RelightNet()

    pic_h = np.max(np.load('./Data/high/high.npz')['arr_0'], axis=3)
    pic_l = np.max(np.load('./Data/low/low.npz')['arr_0'], axis=3)

    model.compile(optimizer=keras.optimizers.Adam(lr=0.0002, decay=1e-4, epsilon=1e-7),
                  loss=keras.losses.MeanSquaredError(), metrics=['mse'])
    reduceLR = [LearningRateScheduler(exponential_decay_with_warmup)]
    model.fit(x=pic_l, y=pic_h, epochs=100, batch_size=16, validation_split=0.02,   callbacks=reduceLR,
              shuffle=True, workers=4, use_multiprocessing=True)

    pic_out = np.squeeze(model.predict(pic_l))

    return pic_h*255, pic_out*255

if __name__ == '__main__':
    pic, pic_recover = img_train()  #-e03,  +e02
    for i in range(4):
        print(pic_recover[i].shape, pic[i].shape)
        out = np.concatenate([pic_recover[i], pic[i]], 1)
        print(out)
        plt.imshow(out)
        # plt.imsave('/content/drive/MyDrive/How_to_see_in_the_Dark/output/img_{}.jpg'.format(i), out)
        plt.imsave('output/img_{}.jpg'.format(i), pic_recover[i], cmap='gray')
        plt.imsave('output/target_{}.jpg'.format(i), pic[i], cmap='gray')