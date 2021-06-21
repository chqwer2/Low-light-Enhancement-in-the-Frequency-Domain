from PIL import Image
import numpy as np
from glob import glob
import os

def normalize_images(file):
    im = Image.open(file)
    return np.array(im, dtype="float16") / 255.0  #32

def save_images(filepath, result_1, result_2 = None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)

    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis = 1)

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')

def img_normalization(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

def mkdir(dir):
    if not os.path.exists(dir):
            os.makedirs(dir)

def load_images(dir):
    print("Loading data from {}...".format(dir))
    low_data_name = glob(os.path.join(dir, 'low/*.*'))
    high_data_name = glob(os.path.join(dir, 'high/*.*'))
    low_data_name.sort()
    high_data_name.sort()
    high_data = []
    low_data = []

    for idx in range(len(low_data_name)):
        low_im = normalize_images(low_data_name[idx])
        low_data.append(low_im)
        high_im = normalize_images(high_data_name[idx])
        high_data.append(high_im)
    high_data = np.array(high_data)
    low_data = np.array(low_data)
    print("Finish Loading...")
    print("Image Shape:", high_data.shape)

    return low_data, high_data, low_data_name, high_data_name