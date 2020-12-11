import os
from flask import Flask, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from keras.models import Sequential,load_model # ニューラルネットワークのモデル定義する際に利用

import numpy as np
from PIL import Image  # ファイルを扱うためにos、ファイルの一覧を取得するためのパッケージglob

# パラメタ指定
classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50  # px

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png','jpg','gif']) # 3つの拡張子のファイルを許可する

app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ファイルのアップロード可否判定関数
def allowed_file(filename):
    # 拡張子があるか、また拡張子は上の３種類のどれかであるかを判定
    # 両方OKなら１、そうでないなら０を返す
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files: # requestの中にファイルがあるかどうか
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename): # 両方が満たされていたら
            filename = secure_filename(file.filename) # サニタイズ処理(危険な文字等を削除)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            model = load_model("./animal_cnn_aug.h5") # 利用するモデルを指定

            # 利用するイメージを指定
            image = Image.open(filepath) # コマンドラインの２番目の引数(全体では３番目の引数)
            image = image.convert('RGB') # グレー系の画像もRGBで表示するように設定
            image = image.resize((image_size,image_size)) # 画像のサイズを揃える
            data = np.asarray(image)
            X = []
            X.append(data)
            X = np.array(X)

            result = model.predict([X])[0]  # 推定結果を格納する
            predicted = result.argmax()  # 配列の中で一番推定確率が高いものを取り出す
            # それを元に確率とラベル名を表示する(%表示で)
            percentage = int(result[predicted] * 100)
            print("{0}({1} %)".format(classes[predicted], percentage))

            return classes[predicted] + str(percentage) + "%"

            # アップロード後のページに転送
            # return redirect(url_for('upload_file',filename=filename))
    return '''
    <!doctype html>
    <html>
    <head>
    <meta charset="UTF-8">
    <title>ファイルをアップロードして判定しよう</title>
    <body>
    </head>
    <h1>ファイルをアップロードして判定しよう</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file></p>
      <input type=submit value=Upload>
    </form>
    </body>
    </html>
    '''


from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

if __name__ == '__main__':
    app.run()




