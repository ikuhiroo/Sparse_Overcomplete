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

from tqdm import tqdm


def main():
    vec_len = 300
    vocab_len = 192953
    print("Input vector length: {}".format(vec_len))  # 300
    print("Output vector length:: {}".format(vec_len * factor))  # 900
    print("L2 Reg(Dict): {}".format(l2_reg))
    print("L1 Reg(Atom): {}".format(l1_reg))

    # AとDの初期化
    atom = joblib.load(
        "/Users/1-10robotics/Desktop/Sparse_Overcomplete/P_ver/trained_model/newvec_9.pkl"
    )
    Dict = joblib.load(
        "/Users/1-10robotics/Desktop/Sparse_Overcomplete/P_ver/trained_model/_dict_9.pkl"
    )

    # Optimizerの初期化
    Optimizer = param.Param(atom, Dict, vocab_len, vec_len)
    Optimizer._del_grad_A = joblib.load(
        "/Users/1-10robotics/Desktop/Sparse_Overcomplete/P_ver/trained_model/_del_grad_A9.pkl"
    )
    Optimizer._grad_sum_A = joblib.load(
        "/Users/1-10robotics/Desktop/Sparse_Overcomplete/P_ver/trained_model/_grad_sum_A9.pkl"
    )
    Optimizer._del_grad_D = joblib.load(
        "/Users/1-10robotics/Desktop/Sparse_Overcomplete/P_ver/trained_model/_del_grad_D9.pkl"
    )
    Optimizer._grad_sum_D = joblib.load(
        "/Users/1-10robotics/Desktop/Sparse_Overcomplete/P_ver/trained_model/_grad_sum_D9.pkl"
    )

    try:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        h = logging.FileHandler("./logtest.log")
        logger.addHandler(h)
        logger.info("log出力")

        logger.info(Optimizer.atom["放送"])

        # indexの単語
        newvec_word = list(Optimizer.atom.keys())
        # indexの単語のベクトル
        newvec_dimention = list(Optimizer.atom.values())

        # dataframe作成, idと単語
        f_word = pd.DataFrame(newvec_word, columns=["word"])

        # limit次元まで集計
        var_per_dimention = {}
        limit_num = 100
        # v: 単語
        for v in range(len(newvec_dimention)):
            # j: 次元
            for j in range(len(newvec_dimention[v][0])):
                # jがlimit_num以下の時
                if j < limit_num:
                    try:
                        var_per_dimention[j].append(newvec_dimention[v][0][j])
                    except:
                        var_per_dimention[j] = [newvec_dimention[v][0][j]]
                else:
                    break

        # 次元ごとに0占有率に関するリストを作成する
        memo_80 = []
        memo_80_85 = []
        memo_85_90 = []
        memo_90_95 = []
        memo_95 = []
        memo_96 = []
        memo_97 = []
        memo_98 = []
        memo_99 = []
        memo_other = []
        # i: 次元
        for i in range(len(list(var_per_dimention.values()))):
            # 100次元までをそれぞれ見る
            x = list(var_per_dimention.values())[i]
            if (x.count(0) / len(x)) * 100 < 80:
                memo_80.append(i)
            elif (x.count(0) / len(x)) * 100 < 85:
                memo_80_85.append(i)
            elif (x.count(0) / len(x)) * 100 < 90:
                memo_85_90.append(i)
            elif (x.count(0) / len(x)) * 100 < 95:
                memo_90_95.append(i)
            elif (x.count(0) / len(x)) * 100 < 96:
                memo_95.append(i)
            elif (x.count(0) / len(x)) * 100 < 97:
                memo_96.append(i)
            elif (x.count(0) / len(x)) * 100 < 98:
                memo_97.append(i)
            elif (x.count(0) / len(x)) * 100 < 99:
                memo_98.append(i)
            elif (x.count(0) / len(x)) * 100 < 100:
                memo_99.append(i)
            else:
                memo_other.append(i)

        logger.info("~80: {}".format(len(memo_80)))
        logger.info("80~85: {}".format(len(memo_80_85)))
        logger.info("85~90: {}".format(len(memo_85_90)))
        logger.info("90~95: {}".format(len(memo_90_95)))
        logger.info("95: {}".format(len(memo_95)))
        logger.info("96: {}".format(len(memo_96)))
        logger.info("97: {}".format(len(memo_97)))
        logger.info("98: {}".format(len(memo_98)))
        logger.info("99: {}".format(len(memo_99)))
        logger.info("other: {}".format(len(memo_other)))

        try:
            # 特定の次元の平均を求める
            for d in memo_80:
                # i番目の次元のベクトルに注目
                x = list(var_per_dimention.values())[d]
                # 平均を求める
                mean_x = np.average(x)
                logger.info("[memo_80] d: {}, mean: {}\n".format(d, mean_x))

                # 求めた平均から各単語の分散を求める
                var_x = [math.sqrt((mean_x - x[i]) ** 2) for i in range(len(x))]

                # 分散の大きさ順にソートし，indexを返す
                index_x_sorted = sorted(
                    range(len(var_x)), key=lambda k: var_x[k], reverse=True
                )

                logger.info(
                    "[memo_80] d: {}, max: {}\n".format(
                        d, sorted(var_x, reverse=True)[0]
                    )
                )

                # indexの上位5個
                target = index_x_sorted[:10]

                # indexの上位5個の単語を返す
                logger.info(
                    "[memo_80] d: {}, target_word: {}\n".format(d, f_word.iloc[target])
                )
        except:
            pass

        try:
            # 特定の次元の平均を求める
            for d in memo_95:
                # i番目の次元のベクトルに注目
                x = list(var_per_dimention.values())[d]
                # 平均を求める
                mean_x = np.average(x)
                logger.info("[memo_95] d: {}, mean: {}\n".format(d, mean_x))

                # 求めた平均から各単語の分散を求める
                var_x = [math.sqrt((mean_x - x[i]) ** 2) for i in range(len(x))]

                # 分散の大きさ順にソートし，indexを返す
                index_x_sorted = sorted(
                    range(len(var_x)), key=lambda k: var_x[k], reverse=True
                )

                logger.info(
                    "[memo_95] d: {}, max: {}\n".format(
                        d, sorted(var_x, reverse=True)[0]
                    )
                )

                # indexの上位5個
                target = index_x_sorted[:10]

                # indexの上位5個の単語を返す
                logger.info(
                    "[memo_95] d: {}, target_word: {}\n".format(d, f_word.iloc[target])
                )
        except:
            pass
        # 特定の次元の平均を求める
        logger.info("memo_99")
        for d in memo_99:
            # i番目の次元のベクトルに注目
            x = list(var_per_dimention.values())[d]
            # 平均を求める
            mean_x = np.average(x)
            logger.info("[memo_99] d: {}, mean: {}\n".format(d, mean_x))

            # 求めた平均から各単語の分散を求める
            var_x = [math.sqrt((mean_x - x[i]) ** 2) for i in range(len(x))]

            # 分散の大きさ順にソートし，indexを返す
            index_x_sorted = sorted(
                range(len(var_x)), key=lambda k: var_x[k], reverse=True
            )

            logger.info(
                "[memo_99] d: {}, max: {}\n".format(d, sorted(var_x, reverse=True)[0])
            )

            # indexの上位5個
            target = index_x_sorted[:10]

            # indexの上位5個の単語を返す
            logger.info(
                "[memo_99] d: {}, target_word: {}\n".format(d, f_word.iloc[target])
            )
    except:
        logger.info("error")


if __name__ == "__main__":
    main()
