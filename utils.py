import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt


def fft(illumination):
    illumination = tf.cast(illumination, tf.complex64)
    fft_y = tf.signal.fft2d(illumination)  # a+bj
    fft_y = tf.signal.fftshift(fft_y)  # cmap='gray'
    mag = tf.math.log(tf.math.abs(fft_y))
    ang = tf.math.angle(fft_y)
    return mag, ang

def ifft(mag, ang):
    # xf1.*cos(yf2)+xf1.*sin(yf2).*i
    mag = tf.math.exp(mag)
    i = tf.complex(0.0, 1.0)
    ifft = tf.cast(mag, tf.complex64) * (tf.cast(tf.math.cos(ang), tf.complex64) + tf.cast(tf.math.sin(ang), tf.complex64) * i)
    ifft = tf.signal.ifftshift(ifft)
    ifft = tf.signal.ifft2d(ifft)
    ifft = tf.cast(ifft, tf.float32)  #tf.math.abs(
    return ifft

def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)

def load_images(file):
    im = Image.open(file)
    return np.array(im, dtype="float32") / 255.0

def save_images(filepath, result_1, result_2 = None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)

    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis = 1)

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')


if __name__ == '__main__':
    img_value = tf.compat.v1.read_file('22.png')


    img = tf.image.decode_jpeg(img_value, channels=3)
    print(img.shape)   #(400, 600, 3)
    input_max = tf.reduce_max(img, axis=2, keepdims=True)

    mag, ang = fft(input_max)
    ifft_mag = ifft(mag, ang)

    ax1 = plt.subplot(2, 2, 1)
    plt.imshow(img.numpy())
    ax2 = plt.subplot(2, 2, 2)
    plt.imshow(input_max.numpy())
    ax3 = plt.subplot(2, 2, 3)
    plt.imshow(mag.numpy())
    ax4 = plt.subplot(2, 2, 4)
    plt.imshow(ifft_mag.numpy())

    plt.show()
