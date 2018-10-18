"""modelA
・学習対象
Appendix : A
dictionary : D

・ハイパーパラメータ
L1ノルムの正則化係数lambda
L2ノルムの正則化係数Gamma
A (overcomplete representation)の要素数

・目的関数
復元誤差 + 正則化係数(L1 + L2)

・更新アルゴリズム
Adagrad
"""
import argparse
import gzip
import math
import numpy
import re
import sys
import numpy as np
from copy import deepcopy
import codecs

import util
import model
import param

"""処理"""
def main():
    """コマンドライン引数の設定"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=None, help="Input word vecs")
    parser.add_argument("-o", "--output", type=str, help="Output word vecs")
    parser.add_argument("-n", "--numiter", type=int, default=10, help="Num iterations")
    args = parser.parse_args()

    """ベクトルの初期化"""
    factor = 10  # Aの要素数 (= factor * word2vecの要素数)
    wordVecs = util.ReadVecsFromFile(args.input) # (vec_len, vec_len)
    newVecs = args.output # A : (factor * vec_len, vec_len)
    vocab = {} # D : (vector_len, factor * vec_len)
    print("Input vector length : {}".format(len(wordVecs[0])))
    print("Output vector length : {}".format(factor * len(wordVecs[0])))

    """ハイパーパラメータ設定"""
    numIter = int(args.numiter) # trainのiter数
    l1_reg = 0.5 # AのL1ノルムの正則化項
    l2_reg = 1e-5 # DのL1ノルムの正則化項
    print("num_iter : {}".format(numIter))
    print("L1 Reg (Atom) : {}".format(l1_reg))
    print("L2 Reg (Dict) : {}".format(l2_reg))

    """model"""
    trainer = model.Model(factor, len(wordVecs[0]), len(wordVecs[0]))

    """train"""
    trainer.train(wordVecs, newVecs, numIter, l1_reg, l2_reg, factor, vocab)

if __name__ == '__main__':
    main()
