# -*- coding: utf-8 -*-

import cv2
import os

for i in range (10):
    orig=cv2.imread("C:/Users/roboworks/Documents/Python/sm64_fonts/"+str(i)+".png", cv2.IMREAD_GRAYSCALE)
    print(str(i)+".png")
    cv2.imwrite(str(i)+".png",orig)