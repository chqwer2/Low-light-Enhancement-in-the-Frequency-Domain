# This script is the demo for Newron Model contest, source code from ATD kara
# Author: Hao, 2020/12/23
# Modified: Calvin. 2020/12/24

# Not included in ML LH Project
import itertools
import sys, os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, ReLU, Concatenate, \
                                    MaxPooling2D, Dense, Reshape, Flatten, Cropping2D, Add, PReLU, UpSampling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, LambdaCallback
import matplotlib.pyplot as plt
from LambdaLR import exponential_decay_with_warmup, LambdaLR
from PIL import Image
from skimage.measure import compare_ssim, compare_psnr
import random
from utils import *
# from U_Net import att_r2_unet, attention_up_and_concate
from MWCNN import *
from glob import  glob
from utils import load_images
from attention import Atten
from SSIM import sk_psnr, compute_ssim

Batch_size = 2   #2

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




# He initializer
glorot = tf.keras.initializers.GlorotUniform(seed=None)     # scale = 1.0, 'glorot'
# random = tf.random_normal_initializer(stddev=0.01)
he_initializer = tf.keras.initializers.VarianceScaling(scale=2.0,  mode='fan_in', distribution='truncated_normal')   # stddev = sqrt(scale / n), 'uniform'


class NewronModel(object):
    def __init__(self, X, y, momentum, Learning_rate=0.1, Batch_size=32, Batch_num=101):
        self.X = X
        self.y = y
        self.Learning_rate = Learning_rate
        # self.Batch_size = Batch_size
        self.batch_size = Batch_size
        self.validation_split = 0.001
        self.epoch = Batch_num
        self.out_dir = 'Newron_output'
        self.act = 'leaky' #'relu' #
        self.momentum = momentum  #0.8
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.vgg = self.VGG_loss()
        self.test_low_data_name = glob('/home/calvchen/study/LOL/data/LOLv2/test/low/*.*')
        self.test_high_data_name = glob('/home/calvchen/study/LOL/data/LOLv2/test/high/*.*')
        # test_low_data_name = glob('/home/calvchen/study/LOL/data/LOLv2/train/low/*.*')
        # test_high_data_name = glob('/home/calvchen/study/LOL/data/LOLv2/train/high/*.*')

        self.test_high_data = []  # np.array(io.ImageCollection(test_low_data_name)) / 255
        self.test_low_data = []  # test_high_data

        for idx in range(len(self.test_low_data_name)):
            low_im = load_images(self.test_low_data_name[idx])
            self.test_low_data.append(low_im)
            high_im = load_images(self.test_high_data_name[idx])
            self.test_high_data.append(high_im)

    # Choose Activate Function
    def act_type(self, features, type):
        if type == 'swish':
            if backend.backend() == 'tensorflow':
                try:
                    return backend.tf.nn.swish(features)  # backend.
                except AttributeError:
                    return features * backend.sigmoid(features)
        elif type == 'hswish':
            return features * tf.nn.relu6(features + 3) / 6
        elif type == 'relu6':
            return tf.nn.relu6(features)
        elif type == 'relu':
            return ReLU()(features)
        elif type == 'prelu':
            return PReLU()(features)    # many parameter
        elif type == 'mish':   # OOM
            return features * backend.tanh(backend.softplus(features))   #ln(1+ex)
        elif type == 'linear':
            return features
        elif type == 'leaky':
            return LeakyReLU()(features)


    def Model_(self, use_bias=False):
        input_img = tf.keras.Input(shape=(None, None, 3))  # 150*150*1  Input placeholder

        # input_max = tf.reduce_max(input_img, axis=3, keepdims=True)  # maximum
        input_im = input_img   # concat([input_img, input_max])
        channel, kernel_size, layer_num = 64, 3, 3
        with tf.compat.v1.variable_scope('EndNet'):
            # MWCNN
            cb_1 = Conv_block(num_filters=64)(input_im)
            cb_1 = Conv_block(num_filters=64)(cb_1)
            dwt_1 = DWT_downsampling()(cb_1)

            cb_2 = Conv_block(num_filters=128)(dwt_1)  # 256
            dwt_2 = DWT_downsampling()(cb_2)

            cb_3 = Conv_block(num_filters=128)(dwt_2)
            dwt_3 = DWT_downsampling()(cb_3)

            # cb_4 = Conv_block(num_filters=64)(dwt_3)
            # dwt_4 = DWT_downsampling()(cb_4)      # 37 vs 38

            cb_5 = Conv_block(num_filters=256)(dwt_3)
            cb_5 = Conv_block(num_filters=256)(cb_5)
            cb_5 = Conv2D(filters=512, kernel_size=3, strides=1, use_bias=use_bias, activation=tf.nn.relu,
                          padding='same')(cb_5)
            cb_5 = BatchNormalization(momentum=0.8)(cb_5)
            cb_5 = ReLU()(cb_5)
            # up = IWT_upsampling()(cb_5)
            # up = Conv_block(num_filters=64)(Add()([up, cb_3]))
            # up = Conv2D(filters=256, kernel_size=3, strides=1, use_bias=use_bias, activation=tf.nn.relu, padding='same')(up)

            up = IWT_upsampling()(cb_5)  # 1024
            up = Conv_block(num_filters=256)(Add()([up, cb_3]))
            up = Conv2D(filters=512, kernel_size=3, strides=1, use_bias=use_bias, activation=tf.nn.relu,
                        padding='same')(up)

            up = IWT_upsampling()(up)
            up = Conv_block(num_filters=256)(Add()([up, cb_2]))
            up = Conv2D(filters=256, kernel_size=3, strides=1, use_bias=use_bias, activation=tf.nn.relu,
                        padding='same')(up)

            up = IWT_upsampling()(up)
            up = Conv_block(num_filters=256)(Add()([up, cb_1]))
            up = Conv2D(filters=256, kernel_size=3, strides=1, use_bias=use_bias, padding='same')(up)
            up = BatchNormalization(momentum=0.8)(up)
            up = ReLU()(up)   #LR
            up = Conv2D(filters=256, kernel_size=3, strides=1, use_bias=use_bias, padding='same')(up)

            out = Conv2D(filters=3, kernel_size=(1, 1), use_bias=use_bias, padding="same")(Atten(up))  # features
            # out = Conv2D(filters=3, kernel_size=(1, 1), use_bias=use_bias, padding="same")())
        model = Model(inputs=input_img, outputs=tf.sigmoid(out), name='R2MWCNN_net')

        print(model.summary())
        return model

    def Model(self, use_bias=False):
        input_img = tf.keras.Input(shape=(None, None, 3))  # 150*150*1  Input placeholder

        # input_max = tf.reduce_max(input_img, axis=3, keepdims=True)  # maximum
        input_im = input_img   # concat([input_img, input_max])
        channel, kernel_size, layer_num = 64, 3, 3
        with tf.compat.v1.variable_scope('EndNet'):
            # MWCNN
            cb_1 = Conv_block(num_filters=32)(input_im)
            cb_1 = Conv_block(num_filters=32)(cb_1)
            dwt_1 = DWT_downsampling()(cb_1)

            cb_2 = Conv_block(num_filters=64)(dwt_1)  # 256
            dwt_2 = DWT_downsampling()(cb_2)

            cb_3 = Conv_block(num_filters=64)(dwt_2)
            dwt_3 = DWT_downsampling()(cb_3)

            # cb_4 = Conv_block(num_filters=64)(dwt_3)
            # dwt_4 = DWT_downsampling()(cb_4)      # 37 vs 38

            cb_5 = Conv_block(num_filters=128)(dwt_3)
            cb_5 = Conv_block(num_filters=128)(cb_5)
            cb_5 = Conv2D(filters=256, kernel_size=3, strides=1, use_bias=use_bias, activation=tf.nn.relu,
                          padding='same')(cb_5)
            cb_5 = BatchNormalization(momentum=0.8)(cb_5)
            cb_5 = ReLU()(cb_5)
            # up = IWT_upsampling()(cb_5)
            # up = Conv_block(num_filters=64)(Add()([up, cb_3]))
            # up = Conv2D(filters=256, kernel_size=3, strides=1, use_bias=use_bias, activation=tf.nn.relu, padding='same')(up)

            up = IWT_upsampling()(cb_5)  # 1024
            up = Conv_block(num_filters=128)(Add()([up, cb_3]))
            up = Conv2D(filters=256, kernel_size=3, strides=1, use_bias=use_bias, activation=tf.nn.relu,
                        padding='same')(up)

            up = IWT_upsampling()(up)
            up = Conv_block(num_filters=128)(Add()([up, cb_2]))
            up = Conv2D(filters=128, kernel_size=3, strides=1, use_bias=use_bias, activation=tf.nn.relu,
                        padding='same')(up)

            up = IWT_upsampling()(up)
            up = Conv_block(num_filters=128)(Add()([up, cb_1]))
            up = Conv2D(filters=128, kernel_size=3, strides=1, use_bias=use_bias, padding='same')(up)
            up = BatchNormalization(momentum=0.8)(up)
            up = ReLU()(up)   #LR
            up = Conv2D(filters=128, kernel_size=3, strides=1, use_bias=use_bias, padding='same')(up)

            out = Conv2D(filters=3, kernel_size=(1, 1), use_bias=use_bias, padding="same")(Atten(up))  # features
            # out = Conv2D(filters=3, kernel_size=(1, 1), use_bias=use_bias, padding="same")())
        model = Model(inputs=input_img, outputs=tf.sigmoid(out), name='Newron_net')

        print(model.summary())
        return model

    class VGG_loss():
        def __init__(self):
            from tensorflow.keras.applications import VGG19
            self.input = tf.keras.Input(shape=(400, 600, 3))
            VGG = VGG19(include_top=False,
                        weights='/home/calvchen/lithotuner/study/ml_lh/model_train/study/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        input_tensor=self.input)
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

    def SmoothL1(self, true, pred, delta=0.05):
        y_pred = ops.convert_to_tensor(pred)
        y_true = math_ops.cast(true, y_pred.dtype)
        diff = tf.abs(y_true - y_pred)
        huber_loss = tf.where(
            tf.less(diff, delta),  # modified smoothL1
            0.5 * tf.pow(diff, 2),
            delta * diff - 0.5 * tf.pow(delta, 2)
        )
        return K.mean(huber_loss, axis=-1)

    def ssim_calculate(self, true, pred):
        y_pred = ops.convert_to_tensor(pred)
        y_true = math_ops.cast(true, y_pred.dtype)

        ssim = tf.reduce_mean(1 - tf.image.ssim(y_pred, y_true, max_val=1)) # slow
        return ssim

    def region_loss(self, y_true, y_pred, percent = 0.3):
        # 400 * 600
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
        Region_loss = self.SmoothL1(low_fake_clean, low_clean) * 4 + self.SmoothL1(high_fake_clean, high_clean)
        return Region_loss

    def edge_loss(self, true, pred):
        # tf.math.reduce_sum(
        pred = tf.image.sobel_edges(pred)
        true = tf.image.sobel_edges(true)

        edge_loss1 = self.SmoothL1(pred[..., 0], true[..., 0])
        edge_loss2 = self.SmoothL1(pred[..., 1], true[..., 1])
        return edge_loss1 + edge_loss2


    def ssim_loss(self, true, pred):
        mse = self.SmoothL1(pred, true) # SmoothL1
        ssim = self.ssim_calculate(pred, true)  #
        region = self.region_loss(true, pred)   #
        VGG = self.vgg(pred, true)

        edge_loss = self.edge_loss(true, pred)

        return 1.5 * mse + 1.0 * ssim + 1.0 * region + 0.1 * VGG + 0.2 * edge_loss    # 0.1 *


    def data_generator(self):
        for i in itertools.count(self.epoch * 100):
            patch_size = 144  # 128
            h, w, = 400, 600


            x = random.randint(0, h - patch_size )
            y = random.randint(0, w - patch_size )
            id = random.randint(0, self.X.shape[0]-1)#, random.randint(0, self.y.shape[0])]
            # list(range(self.y.shape[0]))
            # random.shuffle(id)

            rand_mode = 0  # random.randint(0, 0)  # ---> 4?
            batch_input_low = self.X[id, x: x + patch_size, y: y + patch_size, :]
            batch_input_high= self.y[id, x: x + patch_size, y: y + patch_size, :],
            # print(batch_input_high.shape)
            yield batch_input_low, batch_input_high

    def TrainModel(self):

        dataset = tf.data.Dataset.from_generator(self.data_generator,
                                            (tf.float16, tf.float16), output_shapes=None).batch(16)

        # self.model = att_r2_unet()
        self.model = self.Model()
        print(self.model.summary())

        self.model.compile(optimizer=keras.optimizers.Adam(lr=self.Learning_rate,  epsilon=1e-7), # decay=1e-5,
                           loss=self.ssim_loss, metrics=['mse' , SSIM_metric()])  #
        # key method
        # Learning Rate Warm up and Exp Decay
        training_log = os.path.join(self.out_dir, 'training.csv')
        csv_logger = tf.keras.callbacks.CSVLogger(training_log)
        callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='mse', factor=0.2, patience=10, verbose=1,
                                             min_delta=1e-4, cooldown=5, min_lr=1e-8),
                     # LambdaCallback(on_epoch_begin=None, on_epoch_end=self.lambda_end),
                     csv_logger]

                     # tf.keras.callbacks.ModelCheckpoint(
                     #     './savecheckpoint', monitor='val_loss', verbose=0, save_best_only=True,
                     #     save_weights_only=False, mode='auto', save_freq='epoch', options=None, ),
                     # ]

        #
        # LearningRateScheduler(exponential_decay_with_warmup(num_epochs=self.epoch, min_lr=1e-5)),  # 1e-5, 2e-5

        # validation
        test_low_data_name = glob('/home/calvchen/study/LOL/data/LOLv2/test/low/*.*')[32:35]
        test_high_data_name = glob('/home/calvchen/study/LOL/data/LOLv2/test/high/*.*')[32:35]
        # 722
        val_data_x = []  # np.array(io.ImageCollection(test_low_data_name)) / 255
        val_data_y = []

        for idx in range(len(test_low_data_name)):
            low_im = load_images(test_low_data_name[idx])
            val_data_x.append(low_im)
            high_im = load_images(test_high_data_name[idx])
            val_data_y.append(high_im)

        val_data_x = np.array(val_data_x)
        val_data_y = np.array(val_data_y)
        print("GT shape:", self.y.shape)

        self.model.fit(self.X, self.y, epochs=self.epoch,
                       batch_size=self.batch_size, verbose=1, #validation_split=self.validation_split,
                       validation_data=(val_data_x, val_data_y),  # (x_val, y_val)
                       callbacks=callbacks, shuffle=True, workers=4, use_multiprocessing=True)


        # self.model.fit_generator(dataset, epochs=self.epoch, verbose=1, steps_per_epoch=100,
        #                          callbacks=callbacks, shuffle=True, workers=4, use_multiprocessing=True)

        self.model.save(os.path.join(self.out_dir, 'cnn_mutau_model_saved.h5'))
        print("Finish saving")

    def evaluation(self):
        output = []
        if not os.path.exists('./test_results'):
            os.mkdir('./test_results')
        PSNR = []
        SSIM = []
        # test_low_data, test_high_data = self.X, self.y
        for i in range(len(self.test_low_data)):
            y_predict = self.model.predict(np.expand_dims(self.test_low_data[i], 0))
            output.append(y_predict)
            PSNR.append(sk_psnr(y_predict[0], self.test_high_data[i]))
            SSIM.append(compute_ssim(y_predict, tf.expand_dims(self.test_high_data[i], 0)))
            print("High File: {}".format(self.test_high_data_name[i]))
            print("File {} , SSIM {}, PSNR {}\n".format(self.test_low_data_name[i], SSIM[-1], PSNR[-1]))

        print("Mean PSNR :", np.mean(PSNR))
        print("Mean SSIM :{}\n".format(np.mean(SSIM)))
        return lambda: 1

    def lambda_end(self, epoch, logs):
        every_epoch = 20
        tf.cond( tf.cast(epoch%every_epoch==0 , tf.bool)
                 , true_fn=self.evaluation(), false_fn=lambda:1)



    def TestModel(self):

        # self.model =  keras.models.load_model(os.path.join('Newron_output',
        #                                                    'cnn_mutau_model_saved_loss_22.83027167823021.h5'), compile=False,
        #                                     custom_objects={'Rec_Conv_block':Rec_Conv_block,
        #                                                     'DWT_downsampling':DWT_downsampling,
        #                                                     'Nor_Conv_block': Nor_Conv_block,
        #                                                     'IWT_upsampling':IWT_upsampling})


        # test_low_data_name = glob('../data/eval15/low/*.*')
        # test_high_data_name = glob('../data/eval15/high/*.*')
        # self.test_low_data_name = glob('/home/calvchen/study/LOL/data/LOLv2/train/low/*.*')
        # self.test_high_data_name = glob('/home/calvchen/study/LOL/data/LOLv2/train/high/*.*')
        # self.test_low_data = []
        # self.test_high_data = []
        # for idx in range(len(self.test_low_data_name)):
        #     low_im = load_images(self.test_low_data_name[idx])
        #     self.test_low_data.append(low_im)
        #     high_im = load_images(self.test_high_data_name[idx])
        #     self.test_high_data.append(high_im)

        # np.save(os.path.join(self.out_dir, 'test_predict_result.npy'), y_predict)
        # PSNR
        output = []
        from SSIM import sk_psnr, compute_ssim
        if not os.path.exists('./test_results'):
            os.mkdir('./test_results')
        PSNR = []
        SSIM = []
        # test_low_data, test_high_data = self.X, self.y
        for i in range(len(self.test_low_data)):
            y_predict = self.model.predict(np.expand_dims(self.test_low_data[i], 0))
            output.append(y_predict)
            PSNR.append(sk_psnr(y_predict[0], self.test_high_data[i]))
            SSIM.append(compute_ssim(y_predict, tf.expand_dims(self.test_high_data[i], 0)))
            print("High File: {}".format(self.test_high_data_name[i]))
            print("File {} , SSIM {}, PSNR {}\n".format(self.test_low_data_name[i], SSIM[-1], PSNR[-1]))

            save_images(os.path.join('test_results', 'eval_%s_%d_%d.png' % (self.test_low_data_name[i].split('/')[-1].split('.')[0], SSIM[-1], PSNR[-1])),
                        y_predict, self.test_high_data[i])

        print("Mean PSNR :", np.mean(PSNR))
        print("Mean SSIM :", np.mean(SSIM))
        self.model.save(os.path.join(self.out_dir, 'cnn_mutau_model_saved_loss_{}.h5'.format(np.mean(PSNR))))
        np.savez_compressed(os.path.join('Newron_output', 'predict_result_test_v2'), np.array(output, np.float16))

    def TestModel_no_pair(self):
        self.model =  keras.models.load_model(os.path.join('Newron_output',
                                                           'cnn_mutau_model_saved_loss_22.83027167823021.h5'), compile=False,
                                            custom_objects={'Rec_Conv_block':Rec_Conv_block,
                                                            'Nor_Conv_block': Nor_Conv_block,
                                                            'DWT_downsampling':DWT_downsampling,
                                                            'IWT_upsampling':IWT_upsampling})


        test_low_data_name = glob('/home/calvchen/study/LOL/data/ExDark/*.*')  ####LLLIME   Test/MEF
        test_low_data = []  # test_high_data

        for idx in range(len(test_low_data_name)):
            low_im = np.array(load_images(test_low_data_name[idx]))
            test_low_data.append(low_im)



        # np.save(os.path.join(self.out_dir, 'test_predict_result.npy'), y_predict)
        if not os.path.exists('./test_results'):
            os.mkdir('./test_results')
        for i in range(len(test_low_data)):
            shape = np.array(test_low_data[i]).shape
            x_crop = shape[0] % 8
            y_crop = shape[1] % 8
            print("Shape: ", x_crop, y_crop, shape)
            img = np.array(test_low_data[i])[x_crop:, y_crop:, :]

            y_predict = self.model.predict(np.expand_dims(img, 0))
            save_images(os.path.join('test_results', 'eval_%s.png' %
                                     (test_low_data_name[i].split('/')[-1].split('.')[0])),
                        img, y_predict)



##########################################################################################################################

def run():

    print("Loading...")
    InputImage = np.load("../data/LOLv2_low_compress.npz")['arr_0']  # 1.4W * 150 * 150, v2
    GroundTruth = np.load("../data/LOLv2_high_compress.npz")['arr_0'] # 70 * 70

    print("Finish Loading")
    # InputImage = InputImage[:, :, :, np.newaxis]
    print("Shape:", InputImage.shape)

    # 20000 vs. 16000
    # Data 150*150

    print(InputImage.shape, GroundTruth.shape)
    f = NewronModel(InputImage, GroundTruth, 0.8, Learning_rate=0.0001, Batch_size=Batch_size, Batch_num=220)   #80
    # f.PredictModel()
    # f.TestModel_no_pair()
    f.TrainModel()
    f.TestModel()



#  bsub -I -q gpu -app gpu   /nfs/DEV/PWC/fdu/software/anaconda3/bin/python3 NewronModel.py
# bsub -XF -I -q gpu-short -app gpu
if __name__ == '__main__':
    run()  # 0.8-0.6
