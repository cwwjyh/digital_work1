# -*- coding: utf-8 -*-
# @Time    : 2020/9/27 18:54
# @Author  : weiwei cai
# @File    : HSI.py
# @Software: PyCharm

# subject : equalizeHist for rgb image from hsi space converted by rgb.
# note that sharing code online
import cv2
import numpy as np

def RGB2HSI(rgb_img):
    row = np.shape(rgb_img)[0]
    col = np.shape(rgb_img)[1]
    hsi_img = rgb_img.copy()
    B, G, R = cv2.split(rgb_img)
    [B, G, R] = [i / 255.0 for i in ([B, G, R])]
    H = np.zeros((row, col))
    I = (R + G + B) / 3.0
    S = np.zeros((row, col))
    for i in range(row):
        den = np.sqrt((R[i] - G[i]) ** 2 + (R[i] - B[i]) * (G[i] - B[i]))
        thetha = np.arccos(0.5 * (R[i] - B[i] + R[i] - G[i]) / den)
        h = np.zeros(col)
        h[B[i] <= G[i]] = thetha[B[i] <= G[i]]
        h[G[i] < B[i]] = 2 * np.pi - thetha[G[i] < B[i]]
        h[den == 0] = 0
        H[i] = h / (2 * np.pi)
    for i in range(row):
        min = []
        for j in range(col):
            arr = [B[i][j], G[i][j], R[i][j]]
            min.append(np.min(arr))
        min = np.array(min)
        S[i] = 1 - min * 3 / (R[i] + B[i] + G[i])
        S[i][R[i] + B[i] + G[i] == 0] = 0
    hsi_img[:, :, 0] = H * 255
    hsi_img[:, :, 1] = S * 255
    hsi_img[:, :, 2] = I * 255
    return hsi_img


def HSI2RGB(hsi_img):
    row = np.shape(hsi_img)[0]
    col = np.shape(hsi_img)[1]
    rgb_img = hsi_img.copy()
    H, S, I = cv2.split(hsi_img)
    [H, S, I] = [i / 255.0 for i in ([H, S, I])]
    R, G, B = H, S, I
    for i in range(row):
        h = H[i] * 2 * np.pi
        a1 = h >= 0
        a2 = h < 2 * np.pi / 3
        a = a1 & a2
        tmp = np.cos(np.pi / 3 - h)
        b = I[i] * (1 - S[i])
        r = I[i] * (1 + S[i] * np.cos(h) / tmp)
        g = 3 * I[i] - r - b
        B[i][a] = b[a]
        R[i][a] = r[a]
        G[i][a] = g[a]
        a1 = h >= 2 * np.pi / 3
        a2 = h < 4 * np.pi / 3
        a = a1 & a2
        tmp = np.cos(np.pi - h)
        r = I[i] * (1 - S[i])
        g = I[i] * (1 + S[i] * np.cos(h - 2 * np.pi / 3) / tmp)
        b = 3 * I[i] - r - g
        R[i][a] = r[a]
        G[i][a] = g[a]
        B[i][a] = b[a]
        a1 = h >= 4 * np.pi / 3
        a2 = h < 2 * np.pi
        a = a1 & a2
        tmp = np.cos(5 * np.pi / 3 - h)
        g = I[i] * (1 - S[i])
        b = I[i] * (1 + S[i] * np.cos(h - 4 * np.pi / 3) / tmp)
        r = 3 * I[i] - g - b
        B[i][a] = b[a]
        G[i][a] = g[a]
        R[i][a] = r[a]
    rgb_img[:, :, 0] = B * 255
    rgb_img[:, :, 1] = G * 255
    rgb_img[:, :, 2] = R * 255
    return rgb_img

raw = cv2.imread('input/onion.png')

eq = RGB2HSI(raw)
channels = cv2.split(eq)
channels[0] = cv2.equalizeHist(channels[0])
# channels[1] = cv2.equalizeHist(channels[1])
# channels[2] = cv2.equalizeHist(channels[2])
channelsnew = cv2.merge(channels)

histhsi = HSI2RGB(channelsnew)

res = np.hstack((raw, histhsi))
cv2.imshow('equalizeHist', res)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite('output/onion_hsi.png', res)

