import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from skimage.measure import compare_ssim, compare_psnr
import cv2
from glob import glob


def _as_floats(image0, image1):
    """
    Promote im1, im2 to nearest appropriate floating point precision.
    """
    float_type = np.result_type(image0.dtype, image1.dtype, np.float32)
    image0 = np.asarray(image0, dtype=float_type)
    image1 = np.asarray(image1, dtype=float_type)
    return image0, image1


def PSNR(y_true, y_pred):
    # err = mean_squared_error(image_true, image_test)
    y_true, y_pred = _as_floats(y_true, y_pred)
    out = 10 * np.log10((255 ** 2) / np.mean(np.square(y_pred - y_true), dtype=np.float64))
    return np.mean(out)


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    # x = np.squeeze(x)
    out = np.zeros(x.shape)

    for i in range(3):
        out[0, :, :, i] = convolve2d(x[0, :, :, i], np.rot90(kernel, 2), mode='same')
    return out


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=1):  # not 255
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")

    M, N = im1.shape[1:3]
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))


def sk_psnr(im1, im2):
    # blur = cv2.GaussianBlur(origin, (5, 5), 0)
    psnr_dark = compare_psnr(im1, im2, data_range=1)  # not 255
    return psnr_dark


def sk_ssim(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    ssim_dark = compare_ssim(im1, im2, data_range=1, multichannel=True)  # , full=True
    return ssim_dark


if __name__ == "__main__":

    # Pred_files = glob("/content/CVPR-2020-Semi-Low-Light/DRBL-stage1/experiment/Full_results/results/*HR_Pred.png")
    PSNR = []
    SSIM = []
    sk_SSIM = []
    im1_array = []
    im2_array = []

    im1 = np.array(Image.open('../images/compare/normal.png'), np.float16) / 255
    im2 = np.array(Image.open('../images/compare/RetinexNet.png'), np.float16) / 255
    im1_array.append(im1)
    im2_array.append(im2)

    PSNR.append(sk_psnr(np.array(im1), np.array(im2)))
    SSIM.append(compute_ssim(np.expand_dims(np.array(im1), axis=0), np.expand_dims(np.array(im2), axis=0)))
    sk_SSIM.append(sk_ssim(np.array(im1), np.array(im2)))

    # print("sk SSIM score:", sk_SSIM[-1])
    # print("SSIM score:", SSIM[-1])
    print("psnr:", PSNR[-1], "ssim:", SSIM[-1])
    # print("PSNR score:", PSNR(np.expand_dims(np.array(im1), axis=0), np.expand_dims(np.array(im2), axis=0)))

    print("Mean PSNR :", np.mean(PSNR))
    print("Mean SSIM :", np.mean(SSIM))
    print("Mean sk SSIM :", np.mean(sk_SSIM))

    from importlib import import_module
    fid = import_module('utils.FID').fid
    eval_fid = fid()
    FID = eval_fid.eval(np.squeeze(im1_array), np.squeeze(im2_array))
    print("Mean FID:", np.mean(FID), FID)
    import tensorflow as tf
    # lpips
    # from LPIPS import lpips
    #
    # sess = tf.compat.v1.Session()
    # eval_lpips = lpips()
    # LPIPS = eval_lpips.eval(np.array(np.squeeze(im1_array), dtype=np.float32), np.array(im2_array, dtype=np.float32))
    # # a series of output
    # print("Mean LPIPS:", sess.run(tf.reduce_mean(LPIPS)))

