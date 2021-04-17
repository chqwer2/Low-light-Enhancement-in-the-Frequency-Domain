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

from network.R2_MWCNN import *
from utils.utils import load_images, save_images
from utils.activation import act_type
from utils.metric import SSIM_metric, sk_psnr, compute_ssim
from network.model import Model
from losses.losses import loss_function

# He initializer
glorot = tf.keras.initializers.GlorotUniform(seed=None)
he_initializer = tf.keras.initializers.VarianceScaling(scale=2.0,  mode='fan_in', distribution='truncated_normal')   # stddev = sqrt(scale / n), 'uniform'


class run(object):
    def __init__(self, args, momentum=0.8,):
        self.args = args
        self.Learning_rate = args.lr
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.out_dir = args.out_dir
        self.act = args.act
        self.momentum = momentum
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.test_low_data, self.test_high_data, self.test_low_data_name, self.test_high_data_name = load_images(args.test)
        self.val_data_x, self.val_data_y = self.test_low_data[0:10], self.test_high_data[0:10]

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        if not os.path.exists(self.args.out_dir):
            os.mkdir(self.args.out_dir)


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
        self.X, self.y, _, _ = load_images(self.args.train)

        # self.X = np.load("../data/LOLv2_low_compress.npz")['arr_0']
        # self.y = np.load("../data/LOLv2_high_compress.npz")['arr_0']

        self.model.fit(self.X, self.y, epochs=self.epoch, batch_size=self.batch_size, verbose=1,
                       validation_data=(self.val_data_x, self.val_data_y),  # (x_val, y_val)
                       callbacks=self.callbacks(), shuffle=True, workers=4, use_multiprocessing=True)

        self.model.save(os.path.join(self.out_dir, 'cnn_mutau_model_saved.h5'))
        print("Finish saving")

    def evaluation(self):
        output = []
        PSNR = []
        SSIM = []

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

        if self.args.phase == 'test':
            self.model =  keras.models.load_model(os.path.join(self.args.out_dir,
                                                'cnn_mutau_model_saved.h5'), compile=False,
                                                custom_objects={'Rec_Conv_block':Rec_Conv_block,
                                                                'DWT_downsampling':DWT_downsampling,
                                                                'Nor_Conv_block': Nor_Conv_block,
                                                                'IWT_upsampling':IWT_upsampling})

        PSNR, SSIM = [], []
        for i in range(len(self.test_low_data)):
            y_predict = self.model.predict(np.expand_dims(self.test_low_data[i], 0))
            PSNR.append(sk_psnr(y_predict[0], self.test_high_data[i]))
            SSIM.append(compute_ssim(y_predict, tf.expand_dims(self.test_high_data[i], 0)))
            print("High File: {}".format(self.test_high_data_name[i]))
            print("File {} , SSIM {}, PSNR {}\n".format(self.test_low_data_name[i], SSIM[-1], PSNR[-1]))

            save_images(os.path.join('test_results', 'eval_%s_%d_%d.png' % (self.test_low_data_name[i].split('/')[-1].split('.')[0], SSIM[-1], PSNR[-1])),
                        y_predict, self.test_high_data[i])

        print("Mean PSNR :", np.mean(PSNR))
        print("Mean SSIM :", np.mean(SSIM))
        if self.args.phase == 'train':
            self.model.save(os.path.join(self.out_dir, 'cnn_mutau_model_saved_psnr_{}.h5'.format(np.mean(PSNR))))


    def TestModel_no_pair(self):
        self.model =  keras.models.load_model(os.path.join('Newron_output',
                                                           'cnn_mutau_model_saved_loss_22.83027167823021.h5'), compile=False,
                                            custom_objects={'Rec_Conv_block':Rec_Conv_block,
                                                            'Nor_Conv_block': Nor_Conv_block,
                                                            'DWT_downsampling':DWT_downsampling,
                                                            'IWT_upsampling':IWT_upsampling})

        from glob import glob
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
    fun = run(args)
    if args.phase == 'train':
        fun.TrainModel()
    elif args.phase == 'test':
        # f.TestModel_no_pair()
        fun.TestModel()


