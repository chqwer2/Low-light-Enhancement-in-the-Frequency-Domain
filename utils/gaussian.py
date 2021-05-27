import tensorflow as tf
import numpy as np
# from scipy.ndimage.filters import gaussian_filter
# from ops import concat


def gauss_kernel_fixed(sigma, N):
    # Non-Adaptive kernel size
    if sigma == 0:
        return np.eye(2 * N + 1)[N]
    x = np.arange(-N, N + 1, 1.0)
    g = np.exp(-x * x / (2 * sigma * sigma))
    g = g / np.sum(np.abs(g))
    return g


def gaussian_blur(image, kernel, kernel_size, cdim=3):
    # kernel as placeholder variable, so it can change
    outputs = []
    pad_w = (kernel_size - 1) // 2
    padded = tf.pad(image, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], mode='REFLECT')
    for channel_idx in range(cdim):
        data_c = padded[:, :, :, channel_idx:(channel_idx + 1)]
        g = tf.reshape(kernel, [kernel_size, kernel_size, 1, 1])
        data_c = tf.nn.conv2d(data_c, g, [1, 1, 1, 1], 'VALID')  # g: filter same..
        # print(data_c)
        # g = tf.reshape(kernel, [kernel_size, 1, 1, 1])
        # data_c = tf.nn.conv2d(data_c, g, [1, 1, 1, 1], 'VALID')
        outputs.append(data_c)

    return tf.concat(outputs, axis=3)


def gauss_kernel(sigma, eps, truncate):
    # Adaptive kernel size based on sigma,
    # for fixed kernel size, hardcode N
    # truncate limits kernel size as in scipy's gaussian_filter

    N = np.clip(np.ceil(sigma * np.sqrt(2 * np.log(1 / eps))), 1, truncate)
    x = np.arange(-N, N + 1, 1.0)
    g = np.exp(-x * x / (2 * sigma * sigma))
    g = g / np.sum(np.abs(g))
    return g


def gaussian_blur_adaptive(image, sigma, eps=0.01, img_width=32, cdim=3):
    if sigma == 0:
        return image
    outputs = []
    kernel = gauss_kernel(sigma, eps, img_width - 1)
    pad_w = (kernel.shape[0] - 1) // 2
    padded = tf.pad(image, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], mode='REFLECT')
    for channel_idx in range(cdim):
        data_c = padded[:, :, :, channel_idx:(channel_idx + 1)]
        g = np.expand_dims(kernel, 0)
        g = np.expand_dims(g, axis=2)
        g = np.expand_dims(g, axis=3)
        data_c = tf.nn.conv2d(data_c, g, [1, 1, 1, 1], 'VALID')
        g = np.expand_dims(kernel, 1)
        g = np.expand_dims(g, axis=2)
        g = np.expand_dims(g, axis=3)
        data_c = tf.nn.conv2d(data_c, g, [1, 1, 1, 1], 'VALID')
        outputs.append(data_c)
    return tf.concat(outputs, axis=3)


def gauss_2d_kernel(kernel_size=3, sigma=0):
    # tf.constant([[0, 0], [-1, 1]], tf.float32)
    kernel = np.zeros([kernel_size, kernel_size])
    center = (kernel_size - 1) / 2
    if sigma == 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = 2 * (sigma ** 2)
    sum_val = 0
    print(sum_val)
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
            sum_val += kernel[i, j]
    sum_val = 1 / sum_val

    return kernel * sum_val



if __name__ == '__main__':
    output = []
    gauss_filter = gauss_2d_kernel(3, 0.3)
    gauss_filter = gauss_filter.astype(dtype=np.float32)
    gauss_filter = tf.convert_to_tensor(gauss_filter, dtype=tf.float32)
    gray = gaussian_blur(tf.expand_dims(output, 0), gauss_filter, 3, cdim=1)


