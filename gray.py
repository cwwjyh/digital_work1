# -*- coding: utf-8 -*-
# @Time    : 2020/9/27 18:54
# @Author  : weiwei cai
# @File    : gray.py
# @Software: PyCharm

# subject : equalizeHist for gray image.
import cv2
import numpy as np
import matplotlib.pyplot as plt

#直方图均衡化是用来改善图像的全局亮度和对比度。均衡化用来使图像的直方图分布更加均匀，提升亮度和对比度
img = cv2.imread('input/coins.png', 0)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
equ = clahe.apply(img)
cv2.imshow('equakuzation', np.hstack((img, equ))) #并排显示
cv2.waitKey(0)
#注意：cv2.equalizeHist进行直方图均衡化处理，只能对单通道进行处理

cv2.imwrite('output/coins.png', equ)

