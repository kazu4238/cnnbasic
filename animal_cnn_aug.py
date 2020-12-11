
from keras.models import Sequential  # ニューラルネットワークのモデル定義する際に利用
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D  # 畳み込みの処理、プーリングをする
from keras.utils import np_utils
import keras
import numpy as np
import tensorflow

# パラメタ指定
classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50  # px

# メインの関数を定義する
def main():
    # ファイルからデータを配列に読み込む
    X_train, X_test, y_train, y_test = np.load("./animal.npy",allow_pickle=True)
    # データの正規化を行う（NNで計算する場合には0~1の方が計算しやすいため）
    X_train = X_train.astype("float") /256
    X_test = X_test.astype("float") /256
    # one-hot-vector:正解は１、他は０を表す行列に変換する monkeyの場合は[1,0,0] boarの場合は[0,1,0] crowの場合は[0,0,1]となる
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    # パラメタ一覧を更新して保存する
    model = model_train(X_train,y_train)
    model_eval(model,X_test,y_test) # eval=評価

#     モデルの作成
def model_train(X,y):
    model = Sequential()
    # ここから層をどんどん足していく
    #　畳み込み結果が同じサイズになるようにsameで設定し、X_train.shapeの中身[450,50,50,3]のうちの[50,50,3]を利用するため、
    # X_train.shape[1:]で配列の２番目以降を取り出す
    # ３２個のカーネルを持ち、サイズを３X３とする
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=X.shape[1:]))
    model.add(Activation('relu')) # 活性化関数で正のところを通過させて、その他を０にするためのreluを指定する
    model.add(Conv2D(32, (3, 3))) #２層目
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # プーリング層を追加、一番大きい値を取り出すことでより特徴を際立たせて表示させる
    model.add(Dropout(0.25)) # 25%を捨てて、データの偏りを減らす

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    # プーリング層を追加、一番大きい値を取り出すことでより特徴を際立たせて表示させる
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 全結合を行う
    model.add(Flatten()) # データを一列に並べるFlatten
    model.add(Dense(512)) # Dence(全結合層)
    model.add(Activation('relu')) # 負の値を処分する
    model.add(Dropout(0.5)) # 半分残す
    model.add(Dense(3)) # 最後の出力層のノードは今回３クラスあるので３つに今回設定する
    # その層から出力されたすべての値に指数関数をかけ（すなわちすべて正にし）、その和を1に正規化する。
    # これによって和は1だしどの値も正なので、確率として判断することができる
    model.add(Activation('softmax'))

    # 最適化の処理 トレーニング時の更新アルゴリズム lrはlearning rate,decayは学習率を下げる
    opt = tensorflow.keras.optimizers.RMSprop(lr=0.0001,decay=1e-6)

    # modelの最適化(評価手法)を宣言 lossは損失関数で正解値と推定値の誤差
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    # 一回のトレーニング(エポック)で利用するデータ数,それを何セット行うのかを明記する
    # データサイズに応じてここの値を調整すると良い
    model.fit(X,y,batch_size=80,epochs=100)

    # modelの保存
    model.save("./animal_cnn_aug.h5")

    return model

def model_eval(model,X,y):
    scores = model.evaluate(X,y,verbose=1) # 評価結果を代入 verboseで途中の経過を表示する
    print('Test Loss: ',scores[0])
    print('Test Accuracy:' , scores[1])

if __name__ == "__main__":
    main()




