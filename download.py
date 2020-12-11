# FluckerAPIから画像をダウンロードするファイル

from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

# APIキーの情報

key = "cb7a6e6cde7a30be3178e1f3e1f3a5f3"
secret = "4bd23c87c8af0f74"
# 連続利用による負荷軽減のための待ち時間を設定
wait_time = 1

# pythonファイルを呼び出すための保存フォルダの指定
# argv[1]はコマンドラインの引数の２つ目であることを意味する
animalname = sys.argv[1]
savedir = "./" + animalname

# 取得したデータを保存するための変数宣言 formatで値をどのように受け取るかを指定する
flickr = FlickrAPI(key, secret, format='parsed-json')
result = flickr.photos.search(
    # 検索時のパラメタを入力 　
    text=animalname,
    per_page=400,  # 取得数
    media='photos',  # 検索するデータの種類
    sort='relevance',  # 関連順に並べる
    safe_search=1,  # UIコンテンツ非表示
    extras =  'url_q, licence' #返り値に取得したいデータを含める
)

photos = result['photos']
# pprint(photos)

# enumurateでループ回数をiに代入して番号を取得
for i,photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = savedir + '/' + photo['id'] + 'jpg'
    # ファイルの重複を確認 重複していなければ格納する
    if os.path.exists(filepath): continue
    urlretrieve(url_q,filepath)
    time.sleep(wait_time)