import argparse
import gzip
import math
import numpy
import re
import sys
import numpy as np
from copy import deepcopy
import codecs

"""vectorsのread + normalize"""
def ReadVecsFromFile(filename):
    wordVectors = {}
    # ファイル読み込み
    if filename.endswith('.gz'):
        fileObject = gzip.open(filename, 'r')
    else:
        fileObject = codecs.open(filename, "r", "utf-8", 'ignore')

    for line in fileObject:
        # line = line.strip().lower()
        line = line.strip()
        word = line.split()[0]
        wordVectors[word] = numpy.zeros(len(line.split())-1, dtype=float)
        for index, vecVal in enumerate(line.split()[1:]):
            wordVectors[word][index] = float(vecVal)
        """normalize weight vector"""
        wordVectors[word] /= math.sqrt((wordVectors[word]**2).sum() + 1e-6)

    sys.stderr.write("Vectors read from: "+filename+" \n")
    return wordVectors


"""vector A の書き込み"""
def WriteVectorsToFile():
    pass

"""dict D の書き込み"""
def WriteDictToFile():
    pass
