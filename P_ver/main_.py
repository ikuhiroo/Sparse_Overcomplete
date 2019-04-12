import math
import re
import sys
import numpy as np
import pandas as pd

np.random.seed(100)
import argparse
from memory_profiler import profile
import gc

import util
import param
from hyperparameter import numIter, l1_reg, l2_reg, factor, rate

from sklearn.externals import joblib
import logging

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
    wordVecs = data.ReadVecsFromFile("../word2vec/vectors.model")

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
        atom[key] = (
            0.6
            * (1 / np.sqrt(factor * vec_len, dtype=np.float16))
            * np.random.randn(1, factor * vec_len).astype(np.float16)
        )

    # 3.6MB (32bit, factor=10)増える
    Dict = (
        0.6
        * (1 / np.sqrt(vec_len + factor * vec_len).astype(np.float16))
        * np.random.randn(vec_len, factor * vec_len).astype(np.float16)
    )

    # atom = joblib.load(
    #     '/Users/1-10robotics/Desktop/Sparse_Overcomplete/P_ver/trained_model/numIter = 6_l1_reg = 0.5_l2_reg = 1e-5_factor = 3_rate = 0.05_learning rate/newvec_5.pkl')

    # Dict = joblib.load(
    #     '/Users/1-10robotics/Desktop/Sparse_Overcomplete/P_ver/trained_model/numIter = 6_l1_reg = 0.5_l2_reg = 1e-5_factor = 3_rate = 0.05_learning rate/_dict_5.pkl')

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
            pred_vec = np.dot(
                atom[key].astype(np.float64), Dict.astype(np.float64).T
            ).astype(np.float16)
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
        joblib.dump(Optimizer.atom, "./newvec_{}.pkl".format(time))
        joblib.dump(Optimizer.Dict, "./_dict_{}.pkl".format(time))
        joblib.dump(Optimizer._del_grad_A, "./_del_grad_A{}.pkl".format(time))
        joblib.dump(Optimizer._grad_sum_A, "./_grad_sum_A{}.pkl".format(time))
        joblib.dump(Optimizer._del_grad_D, "./_del_grad_D{}.pkl".format(time))
        joblib.dump(Optimizer._grad_sum_D, "./_grad_sum_D{}.pkl".format(time))
        # output = '../sample_Sparse/newvec_{}.model'.format(time)
        # """save（sparse）A and D"""
        # data.WriteVectorsToFile(Optimizer.atom, str(output))
        # data.WriteDictToFile(Optimizer.dict, str(output+'_dict'))

        # """save（sparse + binary）A and D"""
        # # Sparse + Binary
        # data.WriteVectorsToFile_non(self.atom, str(output + '_non'))
        # # dict
        # data.WriteDictToFile(self.dict, str(output+'_dict'))

        try:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)

            h = logging.FileHandler("./logtest_{}.log".format(time))
            logger.addHandler(h)

            logger.info(Optimizer.atom["放送"])

            # 分析
            newvec_word = list(Optimizer.atom.keys())
            newvec_dimention = list(Optimizer.atom.values())

            # dataframe作成
            f_word = pd.DataFrame(newvec_word, columns=["word"])

            # limit次元まで集計
            var_per_dimention = {}
            limit_num = 100
            for v in range(len(newvec_dimention)):
                for j in range(len(newvec_dimention[v][0])):
                    if j < limit_num:
                        try:
                            var_per_dimention[j].append(newvec_dimention[v][0][j])
                        except:
                            var_per_dimention[j] = [newvec_dimention[v][0][j]]
                    else:
                        break

            # sparse rate > 90を求める
            memo = []
            for i in range(len(list(var_per_dimention.values()))):
                x = list(var_per_dimention.values())[i]
                if (x.count(0) / len(x)) * 100 > 90 and (
                    x.count(0) / len(x)
                ) * 100 < 95:
                    memo.append(i)
            logger.info("len(memo): {}".format(len(memo)))
            logger.info("memo: {}".format(memo))

            # 特定の次元の平均を求める
            for d in memo:
                # i番目の次元のベクトルに注目
                x = list(var_per_dimention.values())[d]
                # 平均を求める
                mean_x = np.average(x)
                logger.info("d: {}, mean: {}\n".format(d, mean_x))

                # 求めた平均から各単語の分散を求める
                var_x = [math.sqrt((mean_x - x[i]) ** 2) for i in range(len(x))]

                # 分散の大きさ順にソートし，indexを返す
                index_x_sorted = sorted(
                    range(len(var_x)), key=lambda k: var_x[k], reverse=True
                )

                logger.info(
                    "d: {}, sorted_value: {}\n".format(d, sorted(var_x, reverse=True))
                )

                # indexの上位5個
                target = index_x_sorted[:10]
                logger.info("d: {}, target_index: {}\n".format(d, target))

                # indexの上位5個の単語を返す
                logger.info("d: {}, target_word: {}\n".format(d, f_word.iloc[target]))
        except:
            logger.info("error")


if __name__ == "__main__":
    main()
