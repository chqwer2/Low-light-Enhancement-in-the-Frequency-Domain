import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from datetime import datetime
from functools import wraps


def time_log(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        begin = datetime.now()
        fun = func(*args, **kwargs)
        end = datetime.now()
        print("===time cost: {} costs:{}".format(func.__name__, end - begin))
        return fun
    return wrapper

def img_normalization(array):
    array = (array - np.min(array)) / (np.max(array) - np.min(array))
    return array

def fft_np(illumination):
    fft_y = np.fft.fft2(illumination)  # a+bj
    fft_y = np.fft.fftshift(fft_y)  # cmap='gray'
    mag = np.log(np.abs(fft_y)+1)
    ang = np.angle(fft_y)
    return mag, ang


def ifft_np(mag, ang):
    # xf1.*cos(yf2)+xf1.*sin(yf2).*i
    i = complex(0.0, 1.0)
    ifft = (np.exp(mag)-1) * (np.cos(ang) + np.sin(ang) * i)
    ifft = np.fft.ifftshift(ifft)
    ifft = np.abs(np.fft.ifft2(ifft))
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



def mkdir(dir):
    if not os.path.exists(dir):
            os.makedirs(dir)

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



