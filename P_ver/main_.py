import math
import re
import sys
import numpy as np
np.random.seed(100)
import argparse
from memory_profiler import profile
import gc

import util
import param
from hyperparameter import numIter, l1_reg, l2_reg, factor, rate

from sklearn.externals import joblib
# from zpkobj import ZpkObj  # zpkobjというファイル名にした場合
# import pickle
# import bz2
# PROTOCOL = pickle.HIGHEST_PROTOCOL
# class ZpkObj:
#     """
#     obj = ZpkObj(obj)  # 圧縮するとき
#     obj = obj.load()  # 解凍するとき

#     zobj1 = ZpkObj(obj1)
#     del obj1 #違う名前に代入するなら必須
#     """
#     def __init__(self, obj):
#         self.zpk_object = bz2.compress(pickle.dumps(obj, PROTOCOL), 9)

#     def load(self):
#         return pickle.loads(bz2.decompress(self.zpk_object))

"""init_vec（vectors.modelなど）の読み込み
# wordVecs : (V, L)
# メモリ計算式：単語数 * 初期vecの次元数 * 0.000004[MB]
= 754069 * 300 * 0.000002 = 452.4414[MB]
"""
"""AとDの初期化
# A : (V, K), A[key] : (K,)
# 初期値の係数 : 0.6*(1/np.sqrt(vec_len*factor)
# メモリ計算式：単語数 * SOVのvecの次元数 * 0.000004[MB]
= 754069 * 900 * 0.000002 = 1357.3242[MB]

# D : (L, K)
# 初期値の係数 : 0.6*(1/np.sqrt(vec_len+vec_len*factor))
# メモリ計算式：初期vecの次元数 * SOVのvecの次元数 * 0.000004[MB]
= 300 * 900 * 0.000002 = 0.54[MB]
"""
"""Optimizerの初期化
# self.atom = Atom  # (L, V)
# メモリ計算式：初期vecの次元数 * 単語数 * 0.000004[MB]
= 754069 * 300 * 0.000002 = 452.4414[MB]

# self.dict = Dict  # (L, K)
# メモリ計算式：初期vecの次元数 * SOVのvecの次元数 * 0.000004[MB]
= 300 * 900 * 0.000002 = 0.54[MB]

# self._del_grad_D  # (L, K)
# メモリ計算式：初期vecの次元数 * SOVのvecの次元数 * 0.000004[MB]
= 300 * 900 * 0.000002 = 0.54[MB]

# self._grad_sum_D  # (L, K)
# メモリ計算式：初期vecの次元数 * SOVのvecの次元数 * 0.000004[MB]
= 300 * 900 * 0.000002 = 0.54[MB]

# self._del_grad_A[key] # (K,)
# メモリ計算式：単語数 * SOVのvecの次元数 * 0.000004[MB]
= 754069 * 900 * 0.000002 = 1357.3242[MB]

# self._grad_sum_A[key] # (K,)
# メモリ計算式：単語数 * SOVのvecの次元数 * 0.000004[MB]
= 754069 * 900 * 0.000002 = 1357.3242[MB]

合計 : 4979.0154[MB]
"""

# @profile
def main():
    # init_vec（vectors.modelなど）の読み込み
    data = util.Data()
    wordVecs = data.ReadVecsFromFile('../word2vec/vectors.model')
    vocab_len = np.array(len(list(wordVecs.keys())), dtype=np.int16)
    for key in wordVecs.keys():
        vec_len = np.array(len(wordVecs[key][0]), dtype=np.int16)
        break

    print("\n----------------")
    print("word_num: {}".format(len(wordVecs.keys())))
    print("Input vector length: {}".format(vec_len))
    print("Output vector length:: {}".format(vec_len * factor))
    print("L2 Reg(Dict): {}".format(l2_reg))
    print("L1 Reg(Atom): {}".format(l1_reg))
    print("----------------\n")

    # AとDの初期化
    atom = {}
    # 3.6MB (32bit, factor=10)増える
    for key in wordVecs.keys():
        atom[key] = 0.6*(1/np.sqrt(factor*vec_len, dtype=np.float16)) * \
            np.random.randn(1, factor*vec_len).astype(np.float16)

    # 3.6MB (32bit, factor=10)増える
    Dict = (0.6*(1/np.sqrt(vec_len + factor*vec_len).astype(np.float16)) *
            np.random.randn(vec_len, factor*vec_len).astype(np.float16))
    
    # Optimizerの初期化
    #  L*V*1 + L*K*3 + V*K*2
    # 90000 + 2700000 + 1800000 = 18.36MB
    Optimizer = param.Param(atom, Dict, vocab_len, vec_len)
    for time in range(1, numIter):
        num_words = 0  # 更新単語数
        # total_error = np.array(0, dtype=np.float16)  # 総ロス
        # atom_l1_norm = np.array(0, dtype=np.float16)  # Aに関するノルム値
        # adaptiveな手続き, A[key]を対象
        for key in wordVecs.keys():
            """error算出"""
            # predict i-th word, DとAの内積を計算, (1, L)
            pred_vec = np.dot(atom[key].astype(np.float64), Dict.astype(
                np.float64).T).astype(np.float16)
            # true_vec - pred_vecの復元誤差, (1, L)
            diff_vec = wordVecs[key] - pred_vec

            """AとDの更新
            全単語共通でtimeごとに値を保持している
            -> 単語ごとに値を保持するべき

            atom[key].UpdateParams(time, diff_vec)
            """
            Optimizer.UpdateParams(time, key, diff_vec, vec_len)

            num_words += 1  # 更新単語数
            # diff_vec = np.clip(diff_vec, -1, 1)
            # print("Error per example : {}".format(
            #     np.sum(np.square(diff_vec), dtype=np.float64)))
            # print(Optimizer.atom[key][0])
            # print(Optimizer.Dict) # Dictに問題なし

            # total_error += error
            # atom_l1_norm += np.sum(Optimizer.atom[key][0])
        # total_error = np.clip(total_error, -1, 1)
        # print("Dict L2 norm : {}".format(np.linalg.norm(Dict, ord=2)))
        # print("Avg Atom L1 norm : {}\n".format(atom_l1_norm/num_words))

        # 保存先，オブジェクトで保存したらファイル保存をする必要がない
        print(Optimizer.atom['放送'])
        joblib.dump(Optimizer.atom, "./newvec_{}.pkl".format(time))
        joblib.dump(Optimizer.Dict, "./_dict_{}.pkl".format(time))
        # output = '../sample_Sparse/newvec_{}.model'.format(time)
        # """save（sparse）A and D"""
        # data.WriteVectorsToFile(Optimizer.atom, str(output))
        # data.WriteDictToFile(Optimizer.dict, str(output+'_dict'))

        # """save（sparse + binary）A and D"""
        # # Sparse + Binary
        # data.WriteVectorsToFile_non(self.atom, str(output + '_non'))
        # # dict
        # data.WriteDictToFile(self.dict, str(output+'_dict'))
if __name__ == '__main__':
    main()
