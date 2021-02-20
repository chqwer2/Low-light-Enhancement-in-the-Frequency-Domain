import numpy as np
from PIL import Image

def fft(illumination):
    fft_y = np.fft.fft2(illumination)  # a+bj
    fft_y = np.fft.fftshift(fft_y)  # cmap='gray'
    mag = np.log(np.abs(fft_y))
    ang = np.angle(fft_y)
    return mag, ang

def ifft(mag, ang):
    # xf1.*cos(yf2)+xf1.*sin(yf2).*i
    mag = np.exp(mag)

    ifft = mag * (np.cos(ang) + np.sin(ang) * complex(0, 1))
    ifft = np.fft.ifft2(ifft)
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

