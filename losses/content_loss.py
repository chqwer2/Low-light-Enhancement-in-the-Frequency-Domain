from tensorflow.keras.applications import VGG19
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import Model
import tensorflow as tf


class VGG_loss():
    def __init__(self):
        from tensorflow.keras.applications import VGG19
        self.input = tf.keras.Input(shape=(400, 600, 3))
        VGG = VGG19(include_top=False, input_tensor=self.input)
        self.model = Model(inputs=VGG.input, outputs=VGG.get_layer('block3_conv4').output)
        self.model.compile("adam", 'mse')
        self.model.trainable = False
        self.vgg_mean = (0.485, 0.456, 0.406)
        self.vgg_std = (0.229, 0.224, 0.225)

    def shift_forward(self, data):
        # data = (data - self.vgg_mean) / self.vgg_std
        return self.model(data, )  # , steps_per_epoch=1)

    def __call__(self, true, pred):
        print(true.shape)
        true_vgg = self.shift_forward(true)
        pred_vgg = self.shift_forward(pred)

        return tf.keras.losses.MeanSquaredError()(true_vgg, pred_vgg)


