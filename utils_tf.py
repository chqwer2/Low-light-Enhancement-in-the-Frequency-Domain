import tensorflow as tf
import matplotlib.pyplot as plt

def fft_tf(illumination):
    illumination = tf.cast(illumination, tf.complex64)
    fft_y = tf.signal.fft2d(illumination)  # a+bj
    fft_y = tf.signal.fftshift(fft_y)  # cmap='gray'
    mag = tf.math.log(tf.math.abs(fft_y)+1)
    ang = tf.math.angle(fft_y)
    return mag, ang

def ifft_tf(mag, ang):
    # xf1.*cos(yf2)+xf1.*sin(yf2).*i
    mag = tf.math.exp(mag)-1
    i = tf.complex(0.0, 1.0)
    ifft = tf.cast(mag, tf.complex64) * (tf.cast(tf.math.cos(ang), tf.complex64) + tf.cast(tf.math.sin(ang), tf.complex64) * i)
    ifft = tf.signal.ifftshift(ifft)
    ifft = tf.signal.ifft2d(ifft)
    ifft = tf.cast(ifft, tf.float32)  #tf.math.abs(
    return ifft


if __name__ == '__main__':
    img_value = tf.compat.v1.read_file('22.png')

    img = tf.image.decode_jpeg(img_value, channels=3)
    print(img.shape)   #(400, 600, 3)
    input_max = tf.reduce_max(img, axis=2, keepdims=True)

    mag, ang = fft_tf(input_max)
    ifft_mag = ifft_tf(mag, ang)

    ax1 = plt.subplot(2, 2, 1)
    plt.imshow(img.numpy())
    ax2 = plt.subplot(2, 2, 2)
    plt.imshow(input_max.numpy())
    ax3 = plt.subplot(2, 2, 3)
    plt.imshow(mag.numpy())
    ax4 = plt.subplot(2, 2, 4)
    plt.imshow(ifft_mag.numpy())

    plt.show()
