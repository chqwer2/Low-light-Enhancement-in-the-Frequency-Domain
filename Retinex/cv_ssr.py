import numpy as np
import cv2
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt

print("cv2:",cv2.__version__)

def singleScaleRetinex(img, sigma):

    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))    # S - F*S

    return retinex

def multiScaleRetinex(img, sigma_list):

    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)

    retinex = retinex / len(sigma_list)

    return retinex

def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

def SSR(src_img, size=3):
    L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)

    dst_Img = cv2.log(img/255.0)
    dst_Lblur = cv2.log(L_blur/255.0)
    dst_IxL = cv2.multiply(dst_Img,dst_Lblur)
    log_R = cv2.subtract(dst_Img, dst_IxL)

    dst_R = cv2.normalize(log_R,None,0,255,cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8

def MSR(img):
    weight = 1 / 3.0
    scales = [15, 101, 301]
    scales_size = len(scales)
    h, w = img.shape[:2]
    log_R = np.zeros((h, w), dtype=np.float32)

    for i in range(scales_size):
        img = replaceZeroes(img)
        L_blur = cv2.GaussianBlur(img, (scales[i], scales[i]), 0)  # SSR ?
        L_blur = replaceZeroes(L_blur)
        dst_Img = cv2.log(img/255.0)   # S
        dst_Lblur = cv2.log(L_blur/255.0)   # F
        dst_Ixl = cv2.multiply(dst_Img, dst_Lblur)
        log_R += weight * cv2.subtract(dst_Img, dst_Ixl)     # Retinex S - S*F

    log_L = np.log(img) - log_R

    dst_R = cv2.normalize(log_R,None, 0, 255, cv2.NORM_MINMAX)
    R = cv2.convertScaleAbs(dst_R)
    dst_L = cv2.normalize(log_L, None, 0, 255, cv2.NORM_MINMAX)
    L = cv2.convertScaleAbs(dst_L)

    return R, L

def Decom(img):
    src_img = cv2.imread(img)
    (b_gray, g_gray, r_gray) = cv2.split(src_img)
    b_R, b_L = MSR(b_gray)  #
    g_R, g_L = MSR(g_gray)
    r_R, r_L = MSR(r_gray)
    result_R = cv2.merge([b_R, g_R, r_R])
    result_L = cv2.merge([b_L, g_L, r_L])
    cv2.imshow('L', result_L)
    cv2.imshow('img',src_img)
    cv2.imshow('R',result_R)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.imshow(result_L, cmap='gray')
    plt.show()
    return result_R, result_L




if __name__ == '__main__':
    img = '../figures/ExDark/1.PNG'
    result_R, result_L = Decom(img)




