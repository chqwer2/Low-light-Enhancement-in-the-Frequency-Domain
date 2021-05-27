from PIL import Image
import numpy as np
from glob import glob
import os
import skimage.io as io


def path2npz(path='', save_path=''):
    array = glob(os.path.join(path, '*.*'))
    print(len(array))
    # Pic = []
    # for i in array:
    #     Pic.append(np.array(Image.open(i), dtype=np.float16))

    coll = io.ImageCollection(os.path.join(path, '*.*'))

    Pic = np.array(coll, dtype=np.float16)
    print(Pic.shape)
    np.savez_compressed(save_path, Pic)


path2npz('/content/drive/MyDrive/Data/LOL/Our_low', '/content/drive/MyDrive/Data/Our_low.npz')
path2npz('/content/drive/MyDrive/Data/LOL/Our_normal', '/content/drive/MyDrive/Data/Our_normal.npz')