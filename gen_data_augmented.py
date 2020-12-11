# gen_data.pyからさらに画像データを増幅(augment)させたファイル

from PIL import Image  # ファイルを扱うためにos、ファイルの一覧を取得するためのパッケージglob
import os, glob
import numpy as np
from sklearn import model_selection # ファイルを分割する

# パラメタ指定
classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50
num_testdata = 100

# 画像の読み込み
X_train = []  # 画像データを格納
X_test = []
Y_train = []  # ラベルデータを格納
Y_test = []

for index, classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")  # パターン一致でファイル一覧を取得する
    for i, file in enumerate(files):  # 写真に番号を付加
        if i >= 200: break
        image = Image.open(file)
        image = image.convert("RGB")  # RGBの256段階で示す数字に変換される
        image = image.resize((image_size, image_size))  # 画像サイズの指定
        data = np.asarray(image)  # 画像データを数字の配列に入れる

        # 200個以下であればまずは優先的にテストデータへ入れて、
        # 余裕があればテストデータへ入れる
        if i < num_testdata:
            X_test.append(data)
            Y_test.append(index)
        else:
            X_train.append(data)
            Y_train.append(index)

            # 画像を回転させてテストデータ数を増やしていく
            for angle in range(-20,20,5): #今回は−２０度から２０度まで５度単位で回転させる
                # 回転させる処理
                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                Y_train.append(index)

                # 反転させる処理
                img_trans = image.transpose(Image.FLIP_LEFT_RIGHT) #左右方向に反転
                data = np.asarray(img_trans)
                X_train.append(data)
                Y_train.append(index)

# X.append(data)  # リストの最後に格納
 # Y.append(index)

X_train = np.array(X_train) # numpyが扱えるようにnumpyの配列に変換する
X_test = np.array(X_test)
y_train = np.array(Y_train)
y_test = np.array(Y_test)


# トレーニング用とテスト用にデータを分割する
# X_train,X_test,y_train,y_test = model_selection.train_test_split(X,Y)
# 分割して４つをまとめて変数に入れてファイルに保存し、プログラムから参照できるようにする
xy = (X_train,X_test,y_train,y_test)
# これによってそれぞれのデータに対して150個のトレーニングデータと５０個の正解ラベルに分けられる(3:1の割合になる)
np.save("./animal_aug.npy",xy)


