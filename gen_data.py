# ダウンロードした画像をファイルに格納し、トレーニング用とテスト用にデータを分割するためのファイル

from PIL import Image  # ファイルを扱うためにos、ファイルの一覧を取得するためのパッケージglob
import os, glob
import numpy as np
from sklearn import model_selection # ファイルを分割する

# パラメタ指定
classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

# 画像の読み込み
X = []  # 画像データを格納
Y = []  # ラベルデータを格納

for index, classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")  # パターン一致でファイル一覧を取得する
    for i, file in enumerate(files):  # 写真に番号を付加
        if i >= 200: break
        image = Image.open(file)
        image = image.convert("RGB")  # RGBの256段階で示す数字に変換される
        image = image.resize((image_size, image_size))  # 画像サイズの指定
        data = np.asarray(image)  # 画像データを数字の配列に入れる
        X.append(data)  # リストの最後に格納
        Y.append(index)

X = np.array(X) # numpyが扱えるようにnumpyの配列に変換する
Y = np.array(Y)

# トレーニング用とテスト用にデータを分割する
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,Y)
# 分割して４つをまとめて変数に入れてファイルに保存し、プログラムから参照できるようにする
xy = (X_train,X_test,y_train,y_test)
# これによってそれぞれのデータに対して150個のトレーニングデータと５０個の正解ラベルに分けられる(3:1の割合になる)
np.save("./animal.npy",xy)


