# pythonで自然言語を入力として受け取り、ライブラリMecabを用いて形態素解析を行うプログラム
# その後、形態素解析結果を用いて、単語の出現頻度を計算し、頻度順に並べ替える
# その後、出現頻度の高い単語を上位10個表示する

# coding: utf-8
import MeCab
import sys
import re
import collections


# テキストファイルを読み込み、文字列として返す関数
def read_text(filename):
    with open(filename, "r") as f:
        return f.read()


# 形態素解析を行う関数
def morphological_analysis(text):
    # MeCabのオブジェクトを生成
    tagger = MeCab.Tagger()
    # 形態素解析を行い、結果を取得
    result = tagger.parse(text)
    # 形態素解析結果を改行で分割し、リストに格納
    result = result.split("")
    # 形態素解析結果を格納するリスト
    morphological_analysis_result = []
    # 形態素解析結果を1行ずつ処理
    for line in result:
        # 行が空の場合は処理をスキップ
        if line == "":
            continue
        # 行をタブで分割し、リストに格納
        line = line.split("\t")
        # 行の要素数が2つでない場合は処理をスキップ
        if len(line) != 2:
            continue
        # 行の2つ目の要素をカンマで分割し、リストに格納
        line = line[1].split(",")
        # 行の2つ目の要素の1つ目の要素を取得
        morphological_analysis_result.append(line[0])
    # 形態素解析結果を返す
    return morphological_analysis_result


# 単語の出現頻度を計算する関数
def count_word(morphological_analysis_result):
    # 単語の出現頻度を格納する辞書
    word_count = {}
    # 形態素解析結果を1つずつ処理
    for word in morphological_analysis_result:
        # 単語が辞書に存在する場合
        if word in word_count:
            # 単語の出現頻度を1増やす
            word_count[word] += 1
        # 単語が辞書に存在しない場合
        else:
            # 単語をキーにして、出現頻度を1とする
            word_count[word] = 1
    # 単語の出現頻度を返す
    return word_count


# 単語の出現頻度を頻度順に並べ替える関数
def sort_word(word_count):
    # 単語の出現頻度を頻度順に並べ替える
    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    # 単語の出現頻度を返す
    return word_count


# 単語の出現頻度を表示する関数
def print_word(word_count):
    # 単語の出現頻度を1つずつ処理
    for word in word_count:
        # 単語と出現頻度を表示
        print(word[0], word[1])


# テキストファイルを読み込み、文字列として返す
text = read_text("test.txt")
# 形態素解析を行い、結果を取得
morphological_analysis_result = morphological_analysis(text)
# 単語の出現頻度を計算し、結果を取得
word_count = count_word(morphological_analysis_result)
# 単語の出現頻度を頻度順に並べ替える
word_count = sort_word(word_count)
# 単語の出現頻度を表示する
print_word(word_count)
