import matplotlib.pyplot as plt
from utils import mkdir, load_images, fft_np, ifft_np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.keras import Input, Model, layers
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
import cv2

def concat(layers):
    return tf.concat(layers, axis=-1)


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

he_initializer = tf.keras.initializers.VarianceScaling(scale=2.0,  mode='fan_in', distribution='truncated_normal')   # stddev = sqrt(scale / n), 'uniform'

def conv_bn_block(input, channel, kernel_size):
    conv = layers.Conv2D(channel, kernel_size, padding='same',use_bias=False,
                         kernel_regularizer=keras.regularizers.l2(1e-4),
                         bias_initializer=tf.zeros_initializer(),
                         kernel_initializer=he_initializer, trainable=True)(input)
    conv = layers.BatchNormalization(momentum=0.8, trainable=True)(conv)
    conv = layers.ReLU()(conv)  #inplace=True？

    # conv1 = conv3x3(inplanes, planes, stride)
    # bn1 = norm_layer(planes)
    # relu = nn.ReLU(inplace=True)
    # conv2 = conv3x3(planes, planes)
    # bn2 = norm_layer(planes)

    return conv

def conv_block():
    pass

def _keras_RelightNet(channel=64, kernel_size=3):  # u-net
    input = Input(shape=[None, None, 1])

    conv0 = conv_bn_block(input, channel, kernel_size)

    conv1 = conv_bn_block(conv0, channel, kernel_size)

    conv2 = conv_bn_block(conv1, channel, kernel_size)

    conv3 = conv_bn_block(conv2, channel, kernel_size)

    up1 = tf.compat.v1.image.resize_nearest_neighbor(conv3, (tf.shape(conv2)[1], tf.shape(conv2)[2]))
    deconv1 = layers.Add()([conv_bn_block(up1, channel, kernel_size), conv2])

    up2 = tf.compat.v1.image.resize_nearest_neighbor(deconv1, (tf.shape(conv1)[1], tf.shape(conv1)[2]))
    deconv2 = layers.Add()([conv_bn_block(up2, channel, kernel_size), conv1])

    up3 = tf.compat.v1.image.resize_nearest_neighbor(deconv2, (tf.shape(conv0)[1], tf.shape(conv0)[2]))
    deconv3 = layers.Add()([conv_bn_block(up3, channel, kernel_size), conv0])

    deconv1_resize = tf.compat.v1.image.resize_nearest_neighbor(deconv1,
                                                                (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
    deconv2_resize = tf.compat.v1.image.resize_nearest_neighbor(deconv2,
                                                                (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))

    feature_gather = concat([deconv1_resize, deconv2_resize, deconv3, input]) # conv0
    feature_fusion = layers.Conv2D(96, 1, padding='same', activation=tf.nn.relu)(feature_gather, ) #

    output_mag = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(feature_fusion, )   # input, remove relu

    model = Model(input, output_mag)
    return model


def keras_RelightNet(channel=64, kernel_size=3):  # u-net, kernel=3:0.02
    input = Input(shape=[None, None, 1])

    conv0 = conv_bn_block(input, channel, kernel_size)

    conv1 = conv_bn_block(conv0, channel, kernel_size)

    conv2 = conv_bn_block(conv1, channel, kernel_size)

    conv3 = conv_bn_block(conv2, channel, kernel_size) + input

    output_mag = layers.Conv2D(1, 1, padding='same')(conv3 )   # input, remove relu
    output_mag = layers.ReLU(max_value=1)(output_mag)


    model = Model(input, output_mag)
    return model

def gradient(input_tensor, direction):
    smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
    smooth_kernel_y = tf.transpose(smooth_kernel_x, [1, 0, 2, 3])
    # filter = tf.Variable(tf.constant(tf.constant([[0, 0], [-1, 1]], tf.float32),
    #                                  shape=[2, 2, 1, 1]))
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    return tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=1, padding='SAME'))

def ave_gradient(input_tensor, direction):
    return tf.compat.v1.layers.average_pooling2d(gradient(input_tensor, direction), pool_size=3, strides=1, padding='SAME')


def smooth(input_I, input_R=None):
    ##* tf.exp(-10 * ave_gradient(input_R, "y"))
    return tf.reduce_mean(gradient(input_I, "x") + gradient(input_I, "y") )

#  loss
def loss_function(real, pred):
    # recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_3 -  input_low))
    # recon_loss_high = tf.reduce_mean(tf.abs(R_high * I_high_3 - input_high))
    #
    # recon_loss_mutal_low = tf.reduce_mean(tf.abs(R_high * I_low_3 - input_low))
    # recon_loss_mutal_high = tf.reduce_mean(tf.abs(R_low * I_high_3 - input_high))
    #
    # equal_R_loss = tf.reduce_mean(tf.abs(R_low - R_high))
    relight_loss = tf.reduce_mean(tf.abs(real - pred))
    # Ismooth_loss_low = smooth(I_low, R_low)
    # Ismooth_loss_high = smooth(I_high, R_high)
    Ismooth_loss_delta = smooth(tf.expand_dims(pred, axis=-1))

    # loss_Decom = recon_loss_low + recon_loss_high + 0.001 * recon_loss_mutal_low + 0.001 * recon_loss_mutal_high + 0.1 * Ismooth_loss_low + 0.1 * Ismooth_loss_high + 0.01 * equal_R_loss
    loss_Relight = relight_loss #+ 3 * Ismooth_loss_delta
    return loss_Relight



def fft_train():
    model = keras_RelightNet()

    mag_h = np.load('./Data/high/high_mag.npz')['arr_0']
    ang_h = np.load('./Data/high/high_ang.npz')['arr_0']
    mag_l = np.load('./Data/low/low_mag.npz')['arr_0']

    # pic = np.max(np.load('./Data/high/high.npz')['arr_0'], axis=3)
    # min, max-min
    scaler_l = np.load('./Data/low/low_scaler.npy')
    scaler_h = np.load('./Data/high/high_scaler.npy')

    model.compile(optimizer=keras.optimizers.Adam(lr=0.00025, decay=1e-4, epsilon=1e-7),  #0.0002
                  loss=loss_function, metrics=['mse'])  #keras.losses.MeanSquaredError()

    # reduceLR = [LearningRateScheduler(exponential_decay_with_warmup)]
    print("train loss:", tf.reduce_mean(tf.pow(mag_l - mag_h, 2)))   # mse = 0.00315
    model.fit(x=mag_l, y=mag_h, epochs=50, batch_size=16, #validation_split=0.02,  # callbacks=reduceLR,
              shuffle=True, workers=4, use_multiprocessing=True)

    mag = np.squeeze(model.predict(mag_l))

    print(scaler_l) # not so much difference
    print(scaler_h)
    print(mag_l[0, 0, 0:3])
    print(mag[0, 0, 0:3]) # 0.018
    print(mag_h[0, 0, 0:3]) # 0.32
    pic_recover = ifft_np(mag[:10] * scaler_h[1] + scaler_h[0], ang_h[:10] * scaler_h[3] + scaler_h[2])
    pic = ifft_np(mag_h[:10] * scaler_h[1] + scaler_h[0], ang_h[:10] * scaler_h[3] + scaler_h[2])
    return pic, pic_recover


def img_train():
    model = keras_RelightNet()
    pic_h = np.load('./Data/high/high.npz')['arr_0']
    pic_l = np.load('./Data/low/low.npz')['arr_0']

    # pic_h_hsv = cv2.cvtColor(pic_h, cv2.COLOR_BGR2HSV)
    # pic_l_hsv = cv2.cvtColor(pic_l, cv2.COLOR_BGR2HSV)
    pic_h = np.max(pic_h, axis=3)  # 0.8115, I
    pic_l = np.max(pic_l, axis=3)   # 0.047

    model.compile(optimizer=keras.optimizers.Adam(lr=0.0002, decay=1e-4, epsilon=1e-7),
                  loss=loss_function, metrics=['mse'])
    reduceLR = [LearningRateScheduler(exponential_decay_with_warmup)]

    print(pic_h[0]-pic_l[0])
    print("data difference:", tf.reduce_mean(tf.pow(pic_h - pic_l, 2)))
    train_on_epoch = False
    epochs = 10
    batch_size = 64
    if train_on_epoch:
        for epoch in range(100 * epochs):
            idx = np.random.randint(0, pic_l.shape[0], batch_size)
            idx_val = np.random.randint(0, 5, batch_size)
            input_imgs = pic_l[idx]
            input_label = pic_h[idx]
            # input_h_hsv = pic_h_hsv[idx]
            # input_l_hsv = pic_l_hsv[idx]

            loss_real = model.train_on_batch(input_imgs, input_label)  # [0] for loss, [1] for accuracy
            pred_mutau = model.predict(input_imgs[:3])
            # MSE
            y_pred = ops.convert_to_tensor(pred_mutau)
            y_true = math_ops.cast(input_label[:3], y_pred.dtype)
            loss_pred = K.mean(K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1))
            # Reconstruct
              #
            # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


            print("%d [train loss: %f, ] [val loss: %f]" % (epoch, loss_real[0], loss_pred))
    else:
        print("Train process")
        model.fit(x=pic_l, y=pic_h, epochs=100, batch_size=16, validation_split=0.02,   callbacks=reduceLR,
        shuffle=True, workers=4, use_multiprocessing=True)


    pic_out = np.squeeze(model.predict(pic_l))

    return pic_h*255, pic_out*255, pic_l*255

if __name__ == '__main__':
    pic, pic_recover, pic_l = img_train()  #-e03,  +e02
    # pic, pic_recover = fft_train()
    for i in range(4):
        print(pic_recover[i].shape, pic[i].shape)
        out = np.concatenate([pic_recover[i], pic[i]], 1)
        print(out)
        plt.imshow(out)
        # plt.imsave('/content/drive/MyDrive/How_to_see_in_the_Dark/output/img_{}.jpg'.format(i), out)
        plt.imsave('output/img_{}.jpg'.format(i), pic_recover[i], cmap='gray')
        plt.imsave('output/target_{}.jpg'.format(i), pic[i], cmap='gray')
        plt.imsave('output/origin_{}.jpg'.format(i), pic_l[i], cmap='gray')