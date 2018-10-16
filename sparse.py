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
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=None, help="Input word vecs")
    parser.add_argument("-o", "--output", type=str, help="Output word vecs")
    parser.add_argument("-n", "--numiter", type=int, default=10, help="Num iterations")
    args = parser.parse_args()

    # ハイパーパラメータ設定
    wordVecs = util.ReadVecsFromFile(args.input)
    outFileName = args.output
    vocab = {} # D
    numIter = int(args.numiter)
    factor = 10 # Aの要素数 = factor * word2vecの要素数
    l1_reg = 0.5 
    l2_reg = 1e-5
    print("Input vector length : {}".format(len(wordVecs[0])))
    print("Output vector length : {}".format(factor * len(wordVecs[0])))
    print("L1 Reg (Atom) : {}".format(l1_reg))
    print("L2 Reg (Dict) : {}".format(l2_reg))
    print("num_iter : {}".format(numIter))

    # model
    trainer = model(factor, len(wordVecs[0]), wordVecs.size)

    # train
    trainer.train(wordVecs, outFileName, numIter, l1_reg, l2_reg, factor, vocab)

if __name__ == '__main__':
    main()
