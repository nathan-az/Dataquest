import numpy as np
import pandas as pd
import math


def entropy(column):
    length = len(column)
    elements, counts = np.unique(column, return_counts=True)
    # if statement in comprehension stops nan result since 0*log2(x) is undefined, returns 0. in this case,
    # 1*log2(1) + 0*log2(0) = 0. zero entropy result, zero uncertainty is consistent with theory
    entropy = np.sum(
        [-(counts[i] / length) * np.log2(counts[i] / length) if counts[i] > 0 else 0 for i in range(length)])
    return entropy


def information_gain(data, split_name, target_name):
    target_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_name], return_index=True)
    target_length = len(data[split_name])
    # below line changed to use np.where so numpy arrays can be used -- .loc solution only worked with pandas data
    weighted_entropy = np.sum(
        [counts[i] / target_length * entropy(data.where(data[split_name] == vals[i])[target_name]) for i in
         range(target_length)])
    return target_entropy - weighted_entropy


def id3():
    return


def predict():
    return


class Tree()
    def __init__(self, left=None, label=None, number=None, right=None):
        self.left = left
        self.label = label
        self.number = number
        self.right = right


class DecisionTree():
    def __init__(self):
        pass
