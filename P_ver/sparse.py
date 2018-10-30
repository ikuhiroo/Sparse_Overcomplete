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
    data = util.Data()
    # wordVecsのread
    wordVecs = data.ReadVecsFromFile(args.input)
    vocab_len = len(wordVecs)
    print("wordVecs vocab_len : {}".format(vocab_len))
    for key in wordVecs.keys():
        vec_len = len(wordVecs[key])
        print("wordVecs per example : {}".format(vec_len))
        break
    # AとDの初期化
    Dict, Atom = data.CreateVecs(wordVecs, factor, vec_len)
    for key in Atom.keys():
        print("A : {}".format(len(Atom[key])))
        print("D : {}".format(Dict.shape))
        print("DA : {}".format(np.dot(Dict, Atom[key]).shape))
        break

    """ハイパーパラメータ設定"""
    numIter = int(args.numiter) # trainのiter数
    l1_reg = 0.5 # AのL1ノルムの正則化項
    l2_reg = 1e-5 # DのL1ノルムの正則化項
    print("num_iter : {}".format(numIter))
    print("L1 Reg (Atom) : {}".format(l1_reg))
    print("L2 Reg (Dict) : {}".format(l2_reg))

    """model"""
    trainer = model.Model(wordVecs, Dict, Atom, factor, vec_len, vocab_len)

    """train"""
    trainer.train(numIter, l1_reg, l2_reg)

if __name__ == '__main__':
    main()
