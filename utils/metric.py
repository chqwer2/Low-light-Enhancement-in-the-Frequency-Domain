import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

class SSIM_metric(tf.keras.metrics.Metric):
    def __init__(self, name="PSNR", **kwargs):
        super(SSIM_metric, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")
        from SSIM import sk_psnr
        self.psnr = sk_psnr
        self.values = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        err = tf.reduce_mean(tf.losses.mean_squared_error(y_true, y_pred), axis=1) #for i,j in enumerate(y_true)]

        psnr = 10 * tf.math.log((1.0 ** 2) / err) / tf.math.log(10.)
        # self.values.append(tf.reduce_mean(psnr))
        self.true_positives.assign_add(tf.reduce_mean(psnr))  # assign_add

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        # self.values = []
        self.true_positives.assign(0.0)
