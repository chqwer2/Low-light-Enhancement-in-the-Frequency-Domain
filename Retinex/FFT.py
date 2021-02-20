from scipy.fftpack import fft2,ifft2
import numpy as np
import matplotlib.pyplot as plt
import cv2


def FFT(img):   #cv2.dft()，cv2.idft()
    fft_y = np.fft.fft2(img)         # a+bj
    fft_y = np.fft.fftshift(fft_y)   # cmap='gray'
    mag = np.log(np.abs(fft_y))
    ang = np.angle(fft_y)   #*180/np.pi
    print(ang)
    return mag, ang

def IFFT(mag, ang):
    # xf1.*cos(yf2)+xf1.*sin(yf2).*i
    mag = np.exp(mag)

    ifft = mag * (np.cos(ang) + np.sin(ang) * complex(0, 1))
    ifft = np.fft.ifft2(ifft)
    return ifft

if __name__ == '__main__':
    # img = '1_his.PNG'
    # img = cv2.imread(img)
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # mag, ang = FFT(hsv[:,:,2])
    name = 'Lighthouse_under_V'  #
    img = cv2.imread(name+'.jpg')
    print(img)
    mag, ang = FFT(img[:, :,0])

    ifft = IFFT(mag, ang)
    print(ifft)
    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(mag, cmap='gray')
    ax[1].imshow(ang, cmap='gray')
    # ax[1].title("Normal")
    ax[2].imshow(np.abs(ifft), cmap='gray')
    print(mag)
    plt.imsave(name+'_fft.jpg', np.round(mag), cmap='gray')
    ax[3].imshow(img, cmap='gray')  #hsv[:,:,2]

    plt.title('Dark vs. Normal')
    plt.show()