import numpy as np
import matplotlab.pyplot as plt
import cv2



def SSR(Image):
    # Single Scale Retinex
    # S = R*L
    Input_C = 1 # ?
    # Solve Lambda
    # Lambda = ?
    # F = Lambda * exp()
    Image_log = np.log(Image)    # log 3D-Image

    # Reflectance = Image_log - np.log(F*S)
    # Illumination =





