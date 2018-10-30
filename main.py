"""modelA"""
import argparse
import sys
import numpy as np

import util
import model
import param
from hyperparameter import numIter, l1_reg, l2_reg, factor, rate

def main():
    """コマンドライン引数の設定
    $ python main.py -i ./sample/sample_vecs.txt -o ./newvec.txt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=None, help="Input word vecs")
    parser.add_argument("-o", "--output", type=str, help="Output word vecs")
    args = parser.parse_args()

    """init_vecの読み込み"""
    data = util.Data()
    # wordVecsのread
    wordVecs = data.ReadVecsFromFile(args.input)
    # word2vecのkey数
    vocab_len = len(list(wordVecs.keys()))
    # print("wordVecs vocab_len : {}".format(vocab_len))
    # word2vecのvalueの次元数out
    for key in wordVecs.keys():
        vec_len = len(wordVecs[key][0])
        break

    """model"""    
    trainer = model.Model(wordVecs, vocab_len, vec_len)
    """train"""
    trainer.Sparse_Overfitting()

    """save"""
    #  Non Binarizing Transformation
    data.WriteVectorsToFile(trainer.atom, str(args.output))
    #  Binarizing Transformation
    # data.WriteVectorsToFile_non(trainer.atom, str(args.output))

if __name__ == '__main__':
    main()
