# Author: Calvin. 2020/12/24

import os
import numpy as np
from tensorflow import keras
from network.R2_MWCNN import *
from utils.utils import load_images, save_images
from utils.activation import act_type
from utils.metric import PSNR_metric, sk_psnr, compute_ssim
from network.model import R2MWCNN
from losses.losses import loss_function
from PIL import Image


def TestModel(test_dir):
    low_data, high_data, low_data_name, high_data_name = load_images('/content/drive/MyDrive/Data/LOL/test')

    model = keras.models.load_model('model/model_saved.h5', compile=False,
                                    custom_objects={'Rec_Conv_block': Rec_Conv_block,
                                                    'DWT_downsampling': DWT_downsampling,
                                                    'IWT_upsampling': IWT_upsampling})
    if not os.path.exists('test_results'):
        os.mkdir('test_results')
    PSNR, SSIM = [], []
    for i in range(len(low_data)):
        y_predict = model.predict(np.expand_dims(low_data[i], 0))
        PSNR.append(sk_psnr(y_predict[0], high_data[i]))
        SSIM.append(compute_ssim(y_predict, tf.expand_dims(high_data[i], 0)))
        print("High File: {}".format(high_data_name[i]))
        print("File {} , SSIM {}, PSNR {}\n".format(low_data_name[i], SSIM[-1], PSNR[-1]))

        save_images(os.path.join('test_results', 'eval_%s_%d_%d.png' %
                                 (low_data_name[i].split('/')[-1].split('.')[0], SSIM[-1], PSNR[-1])),
                    y_predict, high_data[i])
        result_1 = np.squeeze(y_predict)

        im = Image.fromarray(np.clip(result_1 * 255.0, 0, 255.0).astype('uint8'))
        im.save(os.path.join('test_results', 'eval_%s_%d_%d_enhn.png' % (
            low_data_name[i].split('/')[-1].split('.')[0], SSIM[-1], PSNR[-1])), 'png')

    print("Mean PSNR :", np.mean(PSNR))
    print("Mean SSIM :", np.mean(SSIM))

if __name__ == '__main__':
    TestModel(test_dir=0)
