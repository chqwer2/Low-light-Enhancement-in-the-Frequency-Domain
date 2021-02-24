import numpy as np
from utils import mkdir, fft_np, ifft_np, time_log, img_normalization
from tqdm import tqdm
import joblib
import skimage.io as io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skimage import data_dir
import matplotlib.pyplot as plt


@time_log
def compress_img(input_img, output_img_repository, output_name):
    mkdir(output_img_repository)
    coll = input_img
    # show pic
    # io.imshow(coll[10])
    # io.show()
    coll = coll / 255  # Normalize
    print(coll.shape)
    np.savez_compressed(output_img_repository + output_name + '.npz', coll)


@time_log
def compress_img_fft(input_img, output_img_repository, output_name):
    mkdir(output_img_repository)

    coll = np.max(input_img, axis=3)

    print("fft transform")
    mag, ang = fft_np(coll)
    print("finish transform")

    scaler = [np.min(mag), np.max(mag) - np.min(mag), np.min(ang), np.max(ang) - np.min(ang)]  # min, max-min

    # Normalization Standardize
    mag = img_normalization(mag)#.astype(np.float16)
    ang = img_normalization(ang)#.astype(np.float16)

    np.savez_compressed(output_img_repository + output_name + '_mag', mag)
    np.savez_compressed(output_img_repository + output_name + '_ang', ang)
    np.save(output_img_repository + output_name + '_scaler.npy', scaler)
    print("===saving succeed")

if __name__ == '__main__':
    print("==begin process")
    # Img_high = '/content/drive/MyDrive/Data/LOL/LOLDataset/eval15/high/*.png'
    # Img_low = '/content/drive/Myhigh/Drive/Data/LOL/LOLDataset/eval15/low/*.png'
    Img_high = './Data/high/*.png'
    Img_low = './Data/low/*.png'

    Img_high = np.array(io.ImageCollection(Img_high))
    Img_low = np.array(io.ImageCollection(Img_low))
    print("Shape:", Img_high.shape, Img_low.shape)
    print("High img：", Img_high[0][0][0:5])
    print("Low img：", Img_low[0][0][0:5])
    # high_compress
    # compress_img(Img_high, '/content/drive/MyDrive/LOL/LOLDataset/our485/', 'high')
    # compress_img_fft(Img_high, '/content/drive/MyDrive/LOL/LOLDataset/our485/', 'high')
    # low_compress
    # compress_img(Img_low, '/content/drive/MyDrive/LOL/LOLDataset/our485/', 'low')
    # compress_img_fft(Img_low, '/content/drive/MyDrive/LOL/LOLDataset/our485/', 'low')

    # high_compress
    compress_img(Img_high, './Data/high/', 'high')
    compress_img_fft(Img_high, './Data/high/', 'high')
    # low_compress
    compress_img(Img_low, './Data/low/', 'low')
    compress_img_fft(Img_low, './Data/low/', 'low')

