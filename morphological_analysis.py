# pythonで自然言語を入力として受け取り、ライブラリmecabを用いて形態素解析を行う関数
# その後、形態素解析の結果を用いて、単語の出現頻度を計算し、
# その結果を出力する関数を定義する
# また、その結果を用いて、単語の出現頻度をグラフ化する関数を定義する

# ライブラリのインポート
import MeCab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

# 形態素解析を行う関数
def mecab_analysis(text):
    # 形態素解析の結果を格納するリスト
    result = []  
    # 形態素解析の実行
    tagger = MeCab.Tagger() # MeCabの初期化
    tagger.parse('') # 形態素解析の初期化
    node = tagger.parseToNode(text) # 形態素解析の実行
    while node:
        # 形態素解析の結果をリストに格納
        result.append(node.surface) # 単語を格納
        node = node.next  # 次の単語に移動
    # 形態素解析の結果を返す
    return result

# 単語の出現頻度を計算する関数
def word_frequency(text):
    # 単語の出現頻度を格納する辞書
    word_freq = {}
    # 形態素解析の実行
    result = mecab_analysis(text)
    # 形態素解析の結果から、単語の出現頻度を計算
    for word in result:
        if word in word_freq:
            word_freq[word] += 1  # 単語がすでに辞書にある場合は、出現頻度を1増やす
        else:
            word_freq[word] = 1  # 単語が辞書にない場合は、出現頻度を1にする
    # 単語の出現頻度を返す
    return word_freq

# 単語の出現頻度をグラフ化する関数
def graph_word_frequency(text):
    # 単語の出現頻度を計算
    word_freq = word_frequency(text)
    # 単語の出現頻度を降順にソート
    word_freq = sorted(word_freq.items(), key=lambda x:x[1], reverse=True)
    # グラフの描画
    plt.rcParams['font.family'] = 'Hiragino Sans'  ## 日本語フォントの設定(ユーザーによる追加)
    plt.figure(figsize=(10, 5)) ## グラフのサイズを指定
    plt.bar(np.arange(len(word_freq)), [freq[1] for freq in word_freq])  ## 棒グラフの描画
    plt.xticks(np.arange(len(word_freq)), [freq[0] for freq in word_freq], rotation=90) ## 棒グラフのラベルを設定
    plt.show() ## グラフの表示

# テキストの入力
text = input('テキストを入力してください：')

# 単語の出現頻度を計算
word_freq = word_frequency(text)
# 単語の出現頻度を降順にソート
word_freq = sorted(word_freq.items(), key=lambda x:x[1], reverse=True)
# 単語の出現頻度を出力
print(word_freq)

# 単語の出現頻度をグラフ化
graph_word_frequency(text)
