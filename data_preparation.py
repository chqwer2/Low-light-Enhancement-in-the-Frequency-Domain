import numpy as np
from utils import mkdir, fft_np, time_log, img_normalization
from tqdm import tqdm
import joblib
import skimage.io as io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skimage import data_dir

@time_log
def compress_img(input_img_repository, output_img_repository, output_name):
    mkdir(output_img_repository)
    coll = io.ImageCollection(input_img_repository)
    #show pic
    # io.imshow(coll[10])
    # io.show()
    coll = np.array(coll) / 255   # Normalize
    print(coll.shape)
    np.save(output_img_repository+output_name+'.npy', coll)

@time_log
def compress_img_fft(input_img_repository, output_img_repository, output_name):
    mkdir(output_img_repository)

    coll = io.ImageCollection(input_img_repository)
    coll = np.array(coll)
    print("fft transform")
    mag, ang = fft_np(coll)
    print("finish transform")
    print(mag.shape)

    scaler = [np.min(mag), np.max(mag)-np.min(mag), np.min(ang), np.max(ang)-np.min(ang)]  #min, max-min
    # Normalization Standardize
    mag = img_normalization(mag)
    ang = img_normalization(ang)
    print(ang)
    np.save(output_img_repository+output_name+'_mag.npy', mag)
    np.save(output_img_repository+output_name+'_ang.npy', ang)
    np.save(output_img_repository + output_name + '_scaler.npy', scaler)


if __name__ == '__main__':
    print("==begin process")
    Img_high = 'F:/LowLightDataset/LOLdataset/our485/high/*.png'
    Img_low = 'F:/LowLightDataset/LOLdataset/our485/low/*.png'
    # high_compress
    compress_img(Img_high, 'E:/LOLdataset/our485/', 'high')
    compress_img_fft(Img_high, 'E:/LOLdataset/our485/', 'high')
    # low_compress
    compress_img(Img_low, 'E:/LOLdataset/our485/', 'low')
    compress_img_fft(Img_low, 'E:/LOLdataset/our485/', 'low')

