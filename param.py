"""Param update"""

import argparse
import gzip
import math
import numpy
import re
import sys
import numpy as np
from copy import deepcopy
import codecs

class Param:
    def __init__(self, rows, cols):
        # var = (0.6 / sqrt(rows)), Random(rows, 1))
        _del_grad=0
        _grad_sum=0

    # 係数毎の計算：絶対値の二乗
    def cwiseAbs2():
        pass

    # 係数毎の計算：平方根
    def cwiseSqrt():
        pass

    # A.array() / A2.array()
    def cwiseQuotient(A, A2):
        return A / A2

    """sgn
    value > 0 -> 1
    otherwise -> 0
    """
    def sgn():
        pass

    def AdagradUpdate(rate, grad):
        _del_grad += cwiseAbs2(grad)
        _grad_sum += grad
        var -= rate * cwiseQuotient(grad, cwiseSqrt(_del_grad))

    def AdagradUpdateWithL1Reg(rate, grad, l1_reg):
        _update_num += 1
        _del_grad += cwiseAbs2(grad)
        _grad_sum += grad
        for i in range(rows):
            for j in range(cols):
                diff=abs(_grad_sum(i, j)) - _update_num * l1_reg
                if diff <= 0:
                    var[i][j]=0
                else:
                    var[i][j] = -sgn(_grad_sum(i, j)) * rate * \
                        diff / sqrt(_del_grad[i][j])

    # def AdagradUpdateWithL1RegNonNeg():
    #     pass
