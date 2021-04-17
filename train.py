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
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, ReLU,  \
                                    MaxPooling2D, Dense, Reshape, Flatten, Add, PReLU
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, LambdaCallback
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim, compare_psnr
from utils import *
from .network.R2_MWCNN import *
from .utils.utils import load_images
from .utils.activation import act_type
from .utils.metric import SSIM_metric
from .network.model import Model
from .losses.losses import loss_function
from SSIM import sk_psnr, compute_ssim


Batch_size = 2   #2

# He initializer
glorot = tf.keras.initializers.GlorotUniform(seed=None)
he_initializer = tf.keras.initializers.VarianceScaling(scale=2.0,  mode='fan_in', distribution='truncated_normal')   # stddev = sqrt(scale / n), 'uniform'


class run(object):
    def __init__(self, args, momentum=0.8,):
        self.args = args
        self.X, self.y = load_images(args.train)
        self.Learning_rate = args.lr
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.out_dir = args.out_dir
        self.act = args.act
        self.momentum = momentum
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.test_low_data, self.test_high_data = load_images(args.test)
        self.val_data_x, self.val_data_y = self.test_low_data[0:10], self.test_high_data[0:10]


    def callbacks(self):
        training_log = os.path.join(self.out_dir, 'training.csv')
        csv_logger = tf.keras.callbacks.CSVLogger(training_log)
        callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='mse', factor=0.2, patience=10, verbose=1,
                                                          min_delta=1e-4, cooldown=5, min_lr=1e-8),
                     # LambdaCallback(on_epoch_begin=None, on_epoch_end=self.lambda_end),
                     csv_logger]
        return callbacks


    def TrainModel(self):

        self.model = Model()
        self.model.compile(optimizer=keras.optimizers.Adam(lr=self.Learning_rate,  epsilon=1e-7), # decay=1e-5,
                           loss=loss_function, metrics=['mse' , SSIM_metric()])  #

        self.model.fit(self.X, self.y, epochs=self.epoch, batch_size=self.batch_size, verbose=1,
                       validation_data=(self.val_data_x, self.val_data_y),  # (x_val, y_val)
                       callbacks=self.callbacks(), shuffle=True, workers=4, use_multiprocessing=True)

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

def train_(args):
    """ npz files loading """
    # InputImage = np.load("../data/LOLv2_low_compress.npz")['arr_0']
    # GroundTruth = np.load("../data/LOLv2_high_compress.npz")['arr_0']
    """ img files loading """

    fun = run(args)   #80
    # f.PredictModel()
    # f.TestModel_no_pair()
    fun.TrainModel()
    fun.TestModel()


