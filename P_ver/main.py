import argparse
import sys
import numpy as np

import util
import model
import param
from hyperparameter import numIter, l1_reg, l2_reg, factor, rate

from memory_profiler import profile
import gc

# @profile
def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--input", type=str, default=None, help="Input word vecs")
    # # parser.add_argument("-o", "--output", type=str, help="Output word vecs")
    # args = parser.parse_args()

    """init_vec（vectors.modelなど）の読み込み"""
    data = util.Data()
    # wordVecsのread
    # wordVecs = data.ReadVecsFromFile('../sample/sample_vecs.txt')
    # wordVecs = data.ReadVecsFromFile('../word2vec/vectors.model')
    wordVecs = data.ReadVecsFromFile('../word2vec/vectors_300.model')

    # word2vecの単語数：vocab_len
    vocab_len = np.array(len(list(wordVecs.keys())), dtype=np.int16)
    # word2vecのvalueの次元数：vec_len
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

    """model
    実際，18.1MB増えている．
    self.wordVecs : 300 * 300 * 4B = 360,000B = 0.36MB
    self.atom : 3000 * 300 * 4B = 3600,000B = 3.6MB
    self.dict : 3000 * 300 * 4B = 3600,000B = 3.6MB
    self.vocab_len : 4B
    self.vec_len : 4B
    -> 大体，7.56MB増えるはず
    """    
    trainer = model.Model(wordVecs, vocab_len, vec_len)

    """train"""
    trainer.Sparse_Overfitting(wordVecs, vocab_len, vec_len)
    del wordVecs, vocab_len, vec_len

if __name__ == '__main__':
    main()
