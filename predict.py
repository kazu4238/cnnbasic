# 画像を判別する推定プログラムを作成する

from keras.models import Sequential,load_model # ニューラルネットワークのモデル定義する際に利用
# 畳み込みの処理、プーリングをする
from keras.layers import Dense, Dropout, Activation, Flatten,Conv2D, MaxPooling2D
from keras.utils import np_utils
import keras,sys
import numpy as np
from PIL import Image  # ファイルを扱うためにos、ファイルの一覧を取得するためのパッケージglob
import tensorflow

# パラメタ指定
classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50  # px

def build_model():
    model = Sequential()
    # ここから層をどんどん足していく
    # 　畳み込み結果が同じサイズになるようにsameで設定し、X_train.shapeの中身[450,50,50,3]のうちの[50,50,3]を利用するため、
    # X_train.shape[1:]で配列の２番目以降を取り出す
    # ３２個のカーネルを持ち、サイズを３X３とする
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(50,50,3))) # 手動で入れていく
    model.add(Activation('relu'))  # 活性化関数で正のところを通過させて、その他を０にするためのreluを指定する
    model.add(Conv2D(32, (3, 3)))  # ２層目
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # プーリング層を追加、一番大きい値を取り出すことでより特徴を際立たせて表示させる
    model.add(Dropout(0.25))  # 25%を捨てて、データの偏りを減らす

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    # プーリング層を追加、一番大きい値を取り出すことでより特徴を際立たせて表示させる
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 全結合を行う
    model.add(Flatten())  # データを一列に並べるFlatten
    model.add(Dense(512))  # Dence(全結合層)
    model.add(Activation('relu'))  # 負の値を処分する
    model.add(Dropout(0.5))  # 半分残す
    model.add(Dense(3))  # 最後の出力層のノードは今回３クラスあるので３つに今回設定する
    # その層から出力されたすべての値に指数関数をかけ（すなわちすべて正にし）、その和を1に正規化する。
    # これによって和は1だしどの値も正なので、確率として判断することができる
    model.add(Activation('softmax'))
    # 最適化の処理 トレーニング時の更新アルゴリズム lrはlearning rate,decayは学習率を下げる
    opt = tensorflow.keras.optimizers.RMSprop(lr=0.0001,decay=1e-6)

    # modelの最適化(評価手法)を宣言 lossは損失関数で正解値と推定値の誤差
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    # 一回のトレーニング(エポック)で利用するデータ数,それを何セット行うのかを明記する
    # データサイズに応じてここの値を調整すると良い

    # modelのロード
    model=load_model("./animal_cnn_aug.h5")

    return model

def main():
    image = Image.open(sys.argv[1]) # コマンドラインの２番目の引数(全体では３番目の引数)
    image = image.convert('RGB') # グレー系の画像もRGBで表示するように設定
    image = image.resize((image_size,image_size)) # 画像のサイズを揃える
    data = np.asarray(image)
    X = []
    X.append(data)
    X = np.array(X)
    model = build_model()

    result = model.predict([X])[0] # 推定結果を格納する
    predicted = result.argmax()  # 配列の中で一番推定確率が高いものを取り出す
    # それを元に確率とラベル名を表示する(%表示で)
    percentage = int(result[predicted] * 100)
    print("{0}({1} %)".format(classes[predicted],percentage))

if __name__ == "__main__":
    main()
