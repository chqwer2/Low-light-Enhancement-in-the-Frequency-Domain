import numpy as np
from utils import mkdir, fft_np, time_log, img_normalization
from tqdm import tqdm
import joblib
import skimage.io as io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skimage import data_dir
import matplotlib.pyplot as plt


@time_log
def compress_img(input_img_repository, output_img_repository, output_name):
    mkdir(output_img_repository)
    coll = io.ImageCollection(input_img_repository)
    # show pic
    # io.imshow(coll[10])
    # io.show()
    coll = np.array(coll).astype(np.float16) / 255  # Normalize
    print(coll.shape)
    np.savez_compressed(output_img_repository + output_name + '.npz', coll)


@time_log
def compress_img_fft(input_img_repository, output_img_repository, output_name):
    mkdir(output_img_repository)

    coll = io.ImageCollection(input_img_repository)
    coll = np.array(coll)
    print("fft transform")
    mag, ang = fft_np(coll)
    print("finish transform")
    print(mag.shape)

    scaler = [np.min(mag), np.max(mag) - np.min(mag), np.min(ang), np.max(ang) - np.min(ang)]  # min, max-min
    # Normalization Standardize
    mag = img_normalization(mag).astype(np.float16)
    ang = img_normalization(ang).astype(np.float16)

    np.savez_compressed(output_img_repository + output_name + '_mag', mag)
    np.savez_compressed(output_img_repository + output_name + '_ang', ang)
    np.savez_compressed(output_img_repository + output_name + '_scaler', scaler)


if __name__ == '__main__':
    print("==begin process")
    Img_high = '/content/drive/MyDrive/Data/LOL/LOLDataset/our485/high/*.png'
    Img_low = '/content/drive/MyDrive/Data/LOL/LOLDataset/our485/low/*.png'
    # high_compress
    compress_img(Img_high, '/content/drive/MyDrive/LOL/LOLDataset/our485/', 'high')
    compress_img_fft(Img_high, '/content/drive/MyDrive/LOL/LOLDataset/our485/', 'high')
    # low_compress
    compress_img(Img_low, '/content/drive/MyDrive/LOL/LOLDataset/our485/', 'low')
    compress_img_fft(Img_low, '/content/drive/MyDrive/LOL/LOLDataset/our485/', 'low')
