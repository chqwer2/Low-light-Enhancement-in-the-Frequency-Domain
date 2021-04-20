import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from .content_loss import VGG_loss

def SmoothL1(true, pred, delta=0.05):
    y_pred = ops.convert_to_tensor(pred)
    y_true = math_ops.cast(true, y_pred.dtype)
    diff = tf.abs(y_true - y_pred)
    huber_loss = tf.where(
        tf.less(diff, delta),        # modified smoothL1
        0.5 * tf.pow(diff, 2),
        delta * diff - 0.5 * tf.pow(delta, 2)
    )
    return K.mean(huber_loss, axis=-1)


def ssim_loss(true, pred):
    y_pred = ops.convert_to_tensor(pred)
    y_true = math_ops.cast(true, y_pred.dtype)

    ssim = tf.reduce_mean(1 - tf.image.ssim(y_pred, y_true, max_val=1))       # slow
    return ssim

def light_region_loss(y_true, y_pred, percent = 0.3):
    # shape = 400 * 600
    index = int(400 * 600 * percent - 1)
    gray1 = 0.39 * y_true[:, :, :, 0] + 0.5 * y_true[:, :, :, 1] + 0.11 * y_true[:, :, :, 2]  # light
    gray = tf.reshape(gray1, [-1, 400 * 600])
    gray_sort = tf.nn.top_k(gray, 400 * 600)[0]   # input, k
    yu = gray_sort[:, index]
    yu = tf.expand_dims(tf.expand_dims(yu, -1), -1)
    mask = tf.compat.v1.to_float(gray1 >= yu)
    mask1 = tf.expand_dims(mask, -1)
    mask = tf.concat([mask1, mask1, mask1], -1)

    low_fake_clean = tf.multiply(mask, y_pred[:, :, :, :3])
    high_fake_clean = tf.multiply(1 - mask, y_pred[:, :, :, :3])
    low_clean = tf.multiply(mask, y_true[:, :, :, :])
    high_clean = tf.multiply(1 - mask, y_true[:, :, :, :])
    Region_loss = SmoothL1(low_fake_clean, low_clean) * 4 + SmoothL1(high_fake_clean, high_clean)
    return Region_loss

def sobel_edge_loss(true, pred):
    #
    pred = tf.image.sobel_edges(pred)
    true = tf.image.sobel_edges(true)

    edge_loss1 = SmoothL1(pred[..., 0], true[..., 0])
    edge_loss2 = SmoothL1(pred[..., 1], true[..., 1])
    return edge_loss1 + edge_loss2

vgg = VGG_loss()
def loss_function(true, pred):
    mse = SmoothL1(pred, true)     # SmoothL1
    ssim = ssim_loss(pred, true)
    region = light_region_loss(true, pred)
    VGG = vgg(pred, true)

    edge_loss = sobel_edge_loss(true, pred)

    return 1.5 * mse + 1.0 * ssim + 1.0 * region + 0.1 * VGG + 0.2 * edge_loss