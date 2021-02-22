import matplotlib.pyplot as plt
from utils import mkdir, load_images, fft_np, ifft_np
import tensorflow as tf
from tensorflow.keras import Input, Model, layers
import numpy as np

def concat(layers):
    return tf.concat(layers, axis=3)

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

    mag_h = np.load('E:/LOLdataset/our485/high_mag.npy')
    mag_l = np.load('E:/LOLdataset/our485/low_mag.npy')
    ang_l = np.load('E:/LOLdataset/our485/low_ang.npy')
    pic = np.load('E:/LOLdataset/our485/high.npy')

    model.compile(optimizer='adam', loss='mse')
    model.fit(x=mag_l, y=mag_h, epochs=10, batch_size=8)

    mag = model.predict(mag_l)
    pic_recover = ifft_np(mag, ang_l)
    ax1 = plt.subplot(2, 1, 1)

    print(mag[0].shape, mag_h[0].shape)
    plt.imshow(np.concatenate([mag[0], mag_h[0]], 1))
    ax2 = plt.subplot(2, 1, 2)
    while True:
        i = int(input())
        print(pic_recover[i].shape, np.max(pic, axis=3)[i].shape)
        plt.imshow(np.concatenate([pic_recover[i], np.max(pic, axis=3)[i]], 1))
        plt.show()
