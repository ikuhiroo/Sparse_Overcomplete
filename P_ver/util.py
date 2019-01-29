import gzip
import math
import re
import sys
import numpy as np
from copy import deepcopy
import codecs

from memory_profiler import profile
import gc


# @profile
class Data:
    def __init__(self):
        pass

    """ファイルの中身を１行ずつ返すジェネレーター"""
    def file_generator(self, filename):
        cnt = 0
        with open(filename, encoding="utf-8", errors='ignore') as infile:
            for line in infile:
                cnt += 1
                if cnt == 1:
                    pass
                else:
                    yield line

    """vectorsのread + normalize"""
    def ReadVecsFromFile(self, filename):
        sys.stderr.write("Vectors read from: "+filename+" \n")
        print('wordVecsのread中')
        wordVectors = {}
        # 以下，yierdによる処理
        gen = self.file_generator(filename)
        for line in gen:
            line = line.strip()
            word = line.split()[0]
            wordVectors[word] = np.zeros(len(line.split())-1, dtype=np.float16)  # (L,)
            for index, vecVal in enumerate(line.split()[1:]):
                wordVectors[word][index] = float(vecVal)
            del index, vecVal
            """normalize"""
            # wordVectors[word] /= [math.sqrt((wordVectors[word]**2).sum() + 1e-6)]
            wordVectors[word] = np.array([wordVectors[word]], dtype=np.float16)  # (1, L)
        print('wordVecsのread完了')
        return wordVectors

    # """vectorsのread + normalize"""
    # # @profile
    # def ReadVecsFromFile(self, filename):
    #     wordVectors = {}
    #     # ファイル読み込み
    #     if filename.endswith('.gz'):
    #         infile = gzip.open(filename, 'r')
    #     else:
    #         infile = codecs.open(filename, "r", "utf-8", 'ignore')

    #     print('wordVecsのread中')
    #     cnt = 0
    #     for line in infile:
    #         if cnt == 0:
    #             pass
    #         else:
    #             line = line.strip()
    #             word = line.split()[0]
    #             wordVectors[word] = np.zeros(len(line.split())-1, dtype=np.float16)  # (L,)
    #             for index, vecVal in enumerate(line.split()[1:]):
    #                 wordVectors[word][index] = float(vecVal)
    #             del index, vecVal
    #             """normalize"""
    #             # wordVectors[word] /= [math.sqrt((wordVectors[word]**2).sum() + 1e-6)]
    #             wordVectors[word] = np.array([wordVectors[word]], dtype=np.float16) # (1, L)
    #         cnt += 1
    #     del line, cnt
    #     gc.collect()
    #     print('wordVecsのread完了')
    #     sys.stderr.write("Vectors read from: "+filename+" \n")
    #     infile.close()
    #     return wordVectors

    """vector A の書き込み"""
    def WriteVectorsToFile(self, newvec, outFileName):
        """Write word vectors to file"""
        sys.stderr.write('\nWriting down the vectors in '+outFileName+'\n')
        outFile = open(outFileName, 'w')
        for word in newvec.keys():
            outFile.write(word+' ')
            for val in newvec[word][0]:
                outFile.write('%.3f' % (val)+' ')
            outFile.write('\n')
        del word, val
        gc.collect()
        outFile.close()

    """vector A non の書き込み"""
    def WriteVectorsToFile_non(self, newvec, outFileName):
        """binary + Write word vectors to file"""
        sys.stderr.write('\nWriting down the vectors in '+outFileName+'\n')
        outFile = open(outFileName, 'w')
        for word in newvec.keys():
            outFile.write(word+' ')
            for val in newvec[word][0]:
                if val > 0:
                    val = 1
                else:
                    val = 0
                outFile.write(str(val)+' ')
            outFile.write('\n')
        del word, val
        gc.collect()
        outFile.close()

    """dict D の書き込み"""
    def WriteDictToFile(self, dic, outFileName):
        print(dic.shape)
        """Write dict to file"""
        sys.stderr.write('\nWriting down the vectors in '+outFileName+'\n')
        outFile = open(outFileName, 'w')
        for i in range(len(dic)):
            for j in range(len(dic[i])):
                outFile.write(str(dic[i][j])+' ')
        del i, j
        gc.collect()
        outFile.close()
