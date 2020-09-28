# -*- coding: utf-8 -*-
# @Time    : 2020/9/27 18:54
# @Author  : weiwei cai
# @File    : RGB.py
# @Software: PyCharm

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('input/canoe.tif')
#split to 3channels b,g,r
B, G, R = cv2.split(img)

#equalizeHist for b,g,r respectly
EB = cv2.equalizeHist(B)
EG = cv2.equalizeHist(G)
ER = cv2.equalizeHist(R)

#when finish the operation then merge them into together
bgrhist = cv2.merge((EB, EG, ER))

equ = np.hstack((img, bgrhist))

cv2.imshow('Histogram equalized', equ)
cv2.waitKey(0)
cv2.imwrite('output/canoe.tif', equ)
