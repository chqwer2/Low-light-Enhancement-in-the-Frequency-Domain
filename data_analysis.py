import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from utils.gaussian import *

a = np.array(Image.open(r'C:\Users\calvchen\Downloads\stage1\Our_low\low00001.png'), dtype=np.float32) / 255
print(a.shape)

gray = 1 - np.max(a, axis=2, keepdims=True)# np.expand_dims(,  -1)
# b1 = np.multiply(a, np.expand_dims(gray,  -1))
# b = np.array([a[...,0] * gray, a[...,1] * gray, a[...,2] * gray])
# b = b.transpose(1,2,0)
gauss_filter = gauss_2d_kernel(3, 0.01)
gauss_filter = gauss_filter.astype(dtype=np.float32)
gauss_filter = tf.convert_to_tensor(gauss_filter, dtype=tf.float32)
gray = gaussian_blur(tf.expand_dims(gray, 0), gauss_filter, 3)[0]

# plt.imshow(np.concatenate([a, b, b1], 1))
# plt.imshow(np.concatenate([b[..., 0], b[...,1], b[...,2]], 1))
plt.imshow(gray)
plt.show()

