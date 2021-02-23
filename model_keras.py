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
    learning_rate_base = 0.1
    learning_rate_step = 1
    learning_rate_decay = 0.8  # 0.96 for 200, 0.8 for 50
    staircase = False
    with tf.name_scope("exponential_decay_with_warmup"):
        decayed_lr = learning_rate_schedule.ExponentialDecay(learning_rate_base, learning_rate_step,
                                                             learning_rate_decay, staircase=staircase)
        linear_increase = learning_rate_base * tf.cast(global_step / warmup_step, tf.float32)

        exponential_decay = decayed_lr(global_step-10)
        learning_rate = tf.cond(global_step <= warmup_step,
                                lambda: linear_increase.numpy(),
                                lambda: exponential_decay.numpy())
        return learning_rate


def keras_RelightNet(channel=64, kernel_size=3):  # u-net
    input = Input(shape=[None, None, 1])

    conv0 = layers.Conv2D(channel, kernel_size, padding='same', activation=tf.nn.relu)(input)
    conv1 = layers.Conv2D(channel, kernel_size, strides=2, padding='same',
                          activation=tf.nn.relu)(conv0, )
    conv2 = layers.Conv2D(channel, kernel_size, strides=2, padding='same',
                          activation=tf.nn.relu)(conv1, )
    conv3 = layers.Conv2D(channel, kernel_size, strides=2, padding='same',
                          activation=tf.nn.relu)(conv2, )

    up1 = tf.compat.v1.image.resize_nearest_neighbor(conv3, (tf.shape(conv2)[1], tf.shape(conv2)[2]))
    deconv1 = layers.Conv2D(channel, kernel_size, padding='same', activation=tf.nn.relu)(up1, ) + conv2
    up2 = tf.compat.v1.image.resize_nearest_neighbor(deconv1, (tf.shape(conv1)[1], tf.shape(conv1)[2]))
    deconv2 = layers.Conv2D(channel, kernel_size, padding='same', activation=tf.nn.relu)(up2, ) + conv1
    up3 = tf.compat.v1.image.resize_nearest_neighbor(deconv2, (tf.shape(conv0)[1], tf.shape(conv0)[2]))
    deconv3 = layers.Conv2D(channel, kernel_size, padding='same', activation=tf.nn.relu)(up3, ) + conv0

    deconv1_resize = tf.compat.v1.image.resize_nearest_neighbor(deconv1,
                                                                (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
    deconv2_resize = tf.compat.v1.image.resize_nearest_neighbor(deconv2,
                                                                (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
    feature_gather = concat([deconv1_resize, deconv2_resize, deconv3])
    feature_fusion = layers.Conv2D(channel, 1, padding='same', activation=None)(feature_gather, )
    output_mag = layers.Conv2D(1, 1, padding='same', activation=None)(feature_fusion, )

    model = Model(input, output_mag)
    return model



if __name__ == '__main__':
    model = keras_RelightNet()

    mag_h = np.load('/content/drive/MyDrive/LOL/LOLDataset/our485/high_mag.npz')['arr_0']
    mag_l = np.load('/content/drive/MyDrive/LOL/LOLDataset/our485/low_mag.npz')['arr_0']
    ang_l = np.load('/content/drive/MyDrive/LOL/LOLDataset/our485/low_ang.npz')['arr_0']
    pic = np.max(np.load('/content/drive/MyDrive/LOL/LOLDataset/our485/high.npz')['arr_0'], axis=3)
    # min, max-min
    scaler = np.load('/content/drive/MyDrive/LOL/LOLDataset/our485/low_scaler.npy')['arr_0']

    model.compile(optimizer='adam', loss='mse')
    reduceLR = [LearningRateScheduler(exponential_decay_with_warmup)]
    model.fit(x=mag_l, y=mag_h, epochs=100, batch_size=16, validation_split=0.02,
                   callbacks=reduceLR, shuffle=True, workers=4, use_multiprocessing=True)


    mag = model.predict(mag_l)
    pic_recover = ifft_np(mag[:10] * scaler[1] + scaler[0], ang_l[:10] * scaler[3] + scaler[2])

    while True:
        i = int(input())
        print(pic_recover[i].shape, pic[i].shape)
        out = np.concatenate([pic_recover[i], pic[i]], 1)
        plt.imshow(out)
        plt.imsave('/content/drive/MyDrive/How_to_see_in_the_Dark/img_{}.jpg'.format(i), out)

