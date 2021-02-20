import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb2hsv(img):
    h = img.shape[0]
    w = img.shape[1]
    H = np.zeros((h,w),np.float32)
    S = np.zeros((h, w), np.float32)
    V = np.zeros((h, w), np.float32)
    r,g,b = cv2.split(img)
    r, g, b = r/255.0, g/255.0, b/255.0
    for i in range(0, h):
        for j in range(0, w):
            mx = max((b[i, j], g[i, j], r[i, j]))
            mn = min((b[i, j], g[i, j], r[i, j]))
            dt=mx-mn

            if mx == mn:
                H[i, j] = 0
            elif mx == r[i, j]:
                if g[i, j] >= b[i, j]:
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / dt)
                else:
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / dt)+360
            elif mx == g[i, j]:
                H[i, j] = 60 * ((b[i, j]) - r[i, j]) / dt + 120
            elif mx == b[i, j]:
                H[i, j] = 60 * ((r[i, j]) - g[i, j]) / dt+ 240
            H[i,j] =int( H[i,j] / 2)

            #S
            if mx == 0:
                S[i, j] = 0
            else:
                S[i, j] =int( dt/mx*255)
            #V
            V[i, j] =int( mx*255)

    return H, S, V


flags = [i for i in dir(cv2) if i.startswith('COLOR_')]   #show all convertion type
print(flags)

img=cv2.imread("../figures/780.png")
#change into HSV type
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)#
rgb = cv2.cvtColor(hls,cv2.COLOR_HLS2RGB)
#change BGR to RGB
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#get h s v channels
h,s,v = rgb2hsv(img)
merged = cv2.merge([h,s,v]) #前面分离出来的三个通道
merged=np.array(merged,dtype='uint8')

plt.figure(1)
#第一行第一列图形
ax1 = plt.subplot(3,4,1)
plt.sca(ax1)
plt.imshow(img)
plt.title("RGB")
#第一行第二列图形
ax2 = plt.subplot(3,4,2)
plt.sca(ax2)
plt.imshow(img[:,:,0],cmap="gray")
plt.title("R")
#第一行第3列图形
ax3 = plt.subplot(3,4,3)
plt.sca(ax3)
plt.imshow(img[:,:,1],cmap="gray")
plt.title("G")
#第一行第4列图形
ax4 = plt.subplot(3,4,4)
plt.sca(ax4)
plt.imshow(img[:,:,2],cmap="gray")
plt.title("B")
#第2行第1列图形
ax5 = plt.subplot(3,4,5)
plt.sca(ax5)
plt.imshow(hls, cmap='hsv_r')
plt.title("hls")
#第2行第2列图形
ax6 = plt.subplot(3,4,6)
plt.sca(ax6)
plt.imshow(hls[:,:,0],cmap="gray")
plt.title("H")
#第2行第3列图形
ax7 = plt.subplot(3,4,7)
plt.sca(ax7)
plt.imshow(hls[:,:,1],cmap="gray")
plt.title("L")
#第2行第4列图形
ax8 = plt.subplot(3,4,8)
plt.sca(ax8)
plt.imshow(hls[:,:,2],cmap="gray")
plt.title("S")
#第一行第一列图形
ax9 = plt.subplot(3,4,9)
plt.sca(ax9)
plt.imshow(merged)
plt.title("my hls")
#第一行第二列图形
ax12 = plt.subplot(3,4,10)
plt.sca(ax12)
plt.imshow(merged[:,:,0],cmap="gray")
plt.title("h")
#第一行第3列图形
ax13 = plt.subplot(3,4,11)
plt.sca(ax13)
plt.imshow(merged[:,:,1],cmap="gray")
plt.title("s")
#第一行第4列图形
ax14 = plt.subplot(3,4,12)
plt.sca(ax14)
plt.imshow(merged[:,:,2],cmap="gray")
plt.title("v")

plt.show()
plt.imshow(hls[:,:,1],cmap="gray")
plt.show()