# -*- coding: utf-8 -*-
#OpenCVとosをインポート
import cv2
import os
 



TARGET_FILE = "test_0.png"
IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '/sm64_fonts/'
IMG_SIZE = (200, 200)

# CLAHEオブジェクトの生成
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))

#print(type(IMG_DIR))
target_img_path = IMG_DIR + TARGET_FILE
print(target_img_path)
#ターゲット画像をグレースケールで読み出し
target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
target_img = clahe.apply(target_img)
#ターゲット画像を200px×200pxに変換
zoom_diam = 6
target_img = cv2.resize(target_img, dsize=None, fx=zoom_diam, fy=zoom_diam)

# BFMatcherオブジェクトの生成
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# AKAZEを適用、特徴点を検出
detector = cv2.AKAZE_create()
(target_kp, target_des) = detector.detectAndCompute(target_img, None)

print('TARGET_FILE: %s' % (TARGET_FILE))

files = os.listdir(IMG_DIR)
for file in files:
    if file == '.DS_Store' or file == TARGET_FILE:
        continue
    #比較対象の写真の特徴点を検出
    comparing_img_path = IMG_DIR + file
    try:
        comparing_img = cv2.imread(comparing_img_path, cv2.IMREAD_GRAYSCALE)
        comparing_img = cv2.resize(comparing_img, IMG_SIZE)
        (comparing_kp, comparing_des) = detector.detectAndCompute(comparing_img, None)

        matches = bf.match(target_des, comparing_des)
        #特徴量の距離を出し、平均を取る
        dist = [m.distance for m in matches]
        ret = sum(dist) / len(dist)
    except cv2.error:
        # cv2がエラーを吐いた場合の処理
        ret = 100000
    print(file, ret)