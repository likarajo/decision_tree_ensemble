#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 19:48:26 2019

@author: likarajo
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from random import seed
from random import random
from random import randrange
from random import randint
import copy 
global global_attributes

class DataSet:
    def __init__(self, data_set):
        self.name = data_set

        # The training and test labels
        self.labels = {'train': None, 'test': None}

        # The training and test examples
        self.examples = {'train': None, 'test': None}

        # Load all the data for this data set
        for data in ['train', 'test']:
            self.load_file(data)

        # The shape of the training and test data matrices
        self.num_train = self.examples['train'].shape[0]
        self.num_test = self.examples['test'].shape[0]
        self.dim = self.examples['train'].shape[1]

    def load_file(self, dset_type):
        path = './data/{0}.{1}'.format(self.name, dset_type)
        try:
            file_contents = np.genfromtxt(path, missing_values=0, skip_header=0, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19,20,21), dtype=int, delimiter=',')
            self.labels[dset_type] = file_contents[:, 0]
            self.examples[dset_type] = file_contents[:, 1:]

        except RuntimeError:
            print('ERROR: Unable to load file ''{0}''. Check path and try again.'.format(path))


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)
    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    x_dict = {}
    unique_x = np.unique(x)

    for x_attr in unique_x:
        x_dict[x_attr] = np.where(x == x_attr)[0]

    return x_dict

def get_best_attribute(x, y, attributes):
    """
    This function is used for normal id3 implementation
    """
    best = attributes[0]
    maxGain = 0.0
    
    attr = 0
    bestAttr = 0
    for x_col in x.T:
        if not attr in attributes:
            attr = attr + 1
            continue
        gain = mutual_information(x_col, y)
        
        if gain >= maxGain:
            maxGain = gain
            bestAttr = attr
        attr = attr + 1

    return bestAttr

def id3(x, y, attributes, max_depth, depth=0):
    if np.unique(y).size == 1:
        return y[0]
    elif (len(attributes) == 0) or (depth == max_depth) or (len(x) == 0):
        counts = np.bincount(y)
        return np.argmax(counts)
    else:
        best_attr = get_best_attribute(x, y, attributes)
        tree = {}
        x_set = partition(x[:,best_attr])

        for x_val in x_set:
            new_attr = attributes[:]
            new_attr.remove(best_attr)
            idx = x_set[x_val]
            
            subtree = id3(x[idx], y[idx], new_attr, max_depth, depth+1)

            if (best_attr, x_val) in tree:
                tree[best_attr, x_val].append(subtree)
            else:
                tree[best_attr, x_val] = subtree

    return tree


def entropy(y, weight=None):
    """
    Computing the entropy of given labels
    
    """
    if weight is None:
        weight = np.ones((len(y), 1), dtype=int)
    h_entropy = 0
    label = {0: 0, 1: 0}
    y_len = len(y)
    if y_len != 0:
        for i in range(y_len):
            if y[i] == 0:
                label[0] = label[0] + weight[i]
            elif y[i] == 1:
                label[1] = label[1] + weight[i]
        sum = label[0] + label[1]
        for j in range(len(label)):
            label[j] = label[j]/sum
            if label[j] != 0:
                h_entropy = label[j] * np.log2(label[j]) + h_entropy
        return (-1)*h_entropy
    else:
        return 0


def mutual_information(x, y, weight=None):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.
    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    if weight is None:
        weight = np.ones((len(y), 1), dtype=int)
    h_y = entropy(y, weight)
    x_partition = partition(x)
    temp = 0
    total_weight = 0
    for j in x_partition:
        weight_i = np.sum(weight[x_partition[j]])
        temp = ((weight_i) * entropy(y[x_partition[j]], weight[x_partition[j]])) + temp
        total_weight = weight_i + total_weight
    h_y_x = temp / total_weight
    info_gain = h_y - h_y_x
    if (info_gain) < 0:
        info_gain = 0
    return (info_gain)

def get_best_attribute_modified(x, y, attributes, weights = None):
    """
    Get the best attribute from the list of attributes
    """
    best = attributes[0]
    if weights is None:
        weights = [1.] * len(x)
    
    maxGain = 0
    bestAttr = 0
    for attr_pair in attributes:
        x_attr = attr_pair[0]
        unique_value = attr_pair[1]
        x_col = copy.deepcopy(x[:,x_attr])
        x_eq_idx = np.where(x_col == unique_value)[0]
        x_neq_idx = np.where(x_col != unique_value)[0]
        
        x_col[x_eq_idx] = 1
        
        x_col[x_neq_idx] = 0
        gain = mutual_information(x_col, y, weights)
        
        if gain >= maxGain:
            maxGain = gain
            bestAttr = attr_pair

    return bestAttr

def id3Modified(x, y, weights, attributes, max_depth, depth=0):
    """
    This is a modified implementation of id3 that will create a binary decision tree.
    Now the tree will have the structure like tree[best_attribute, best_attribute_value, {true/false}]
        
    """
    if np.unique(y).size == 1:
        return y[0]
    elif (len(attributes) == 0) or (depth == max_depth) or (len(x) == 0):
        counts = np.bincount(y)
        if len(counts) == 0:
            return 1
        else:
            return np.argmax(counts)
    else:
        best_attr = get_best_attribute_modified(x, y, attributes, weights)
        tree = {}
        new_attr1 = attributes[:]
        new_attr1.remove(best_attr)
        new_attr2 = attributes[:]
        new_attr2.remove(best_attr)
        x_col = x[:,best_attr[0]]
        idx_true = np.where(x_col == best_attr[1])[0]
        idx_false = np.where(x_col != best_attr[1])[0]
        tree[(best_attr[0], best_attr[1], "true")] = id3Modified(x[idx_true], y[idx_true], weights, new_attr1, max_depth, depth+1)
        tree[(best_attr[0], best_attr[1], "false")] = id3Modified(x[idx_false], y[idx_false], weights, new_attr2, max_depth, depth+1)
            
    return tree

def bagging(x, y, max_depth, num_trees):
    """
    This is implementation of bagging algorithm. We will learn multiple trees and use them to implement
    bagging algorithm.
    """
    modelList = list()
    index = 0
    seed(1)
    weights = list()
    for sam in range(num_trees):
        samples = bootstrapSamples(x, 1)
        training_set = x[samples]
        bagging_attr = list()
        for attr in global_attributes:
            x_col = training_set[:,attr]
            unique_x = np.unique(x_col)
            for x_val in unique_x:
                bagging_attr.append([attr, x_val])
        weights = np.ones((len(y), 1), dtype=int)
        tree = id3Modified(x[samples], y[samples], weights, bagging_attr, max_depth, depth=0)
        modelList.append([1, tree])
        index = index + 1
    
    return modelList

def boosting(x, y, max_depth, num_stumps):
    """
    This is implementation of boosting algorithm. We will learn the stumps and calculate hypothesis weights
        
    """
    treeList = [None] * num_stumps
    alpha = [0] * num_stumps
    boosting_attr = list()
    weight = []
    alpha_i = 0
    row, cols = np.shape(x)

    for attr in global_attributes:
        x_col = x[:,attr]
        unique_x = np.unique(x_col)
        for x_val in unique_x:
            boosting_attr.append([attr, x_val])
    
    for num in range(0, num_stumps):
        if num == 0:
            d = 1/row
            weight = np.full((row, 1), d)
        else:
            previous_weight = weight
            weight = []
            for i in range(row):
                if y[i] == trn_pred[i]:
                    weight.append(previous_weight[i] * np.exp(-1*alpha_i))
                else:
                    weight.append(previous_weight[i] * np.exp(alpha_i))
            d_total = np.sum(weight)
            weight = weight / d_total

        tree = id3Modified(x, y, weight, boosting_attr, max_depth, depth=0)
        trn_pred = [predict_example_utility(x[i, :], tree) for i in range(row)]
        wsum = 0
        for i in range(row):
            if(trn_pred[i] != y[i]):
                wsum += weight[i]
                
        error = (1/(np.sum(weight))) * wsum
        alpha_i = 0.5 * np.log((1 - error)/error)
        alpha[num] = alpha_i
        treeList[num] = tree
        
    finalHypothesis = [alpha, treeList]
    return finalHypothesis

def normalize(weights):
    """
    Normalize the given dataset
    """
    norm = sum(weights)
    weights = [x / norm for x in weights]
    return weights

def bootstrapSamples(dataset, ratio=1.0):
    """
    Create random samples with size equal to dataset when ratio = 1
    """
    sample = list()
    n_sample = round(len(dataset) * ratio)
    for i in range(len(dataset)):
        index = randrange(0, len(dataset))
        sample.append(index)
    return sample

def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.
    Returns the predicted label of x according to tree
    """
    for index, val in enumerate(x):
        key = (index, val)

        if key in list(tree.keys()):
            try:
                result = tree[key]
            except:
                return default_output

            result = tree[key]

            if isinstance(result, dict):
                return predict_example(x, result)
            else:
                return result

def predict_example_bagging(x, modelList):
    """
    Predict a given example for bagging using majority voting
    """
    resultList = list()
    
    for tree in modelList:
        result = predict_example_utility(x, tree[1])
        resultList.append(result)

    counts = np.bincount(resultList)
    if len(counts) == 0:
        return 1
    else:
        return np.argmax(counts)

def predict_example_utility(x, tree):
    """
    Predict a given example x using the binary tree returned from id3modified
    """
    default_output = 1
    values = next(iter(tree))

    if (x[values[0]] == values[1]):
        if (tree[values] == 0 or tree[values] == 1):
            return tree[values]
        else:
            return predict_example_utility(x, tree[values])
    elif (x[values[0]] != values[1]):
        keys = (values[0], values[1], 'false')
        if (tree[keys] == 0 or tree[keys] == 1):
            return tree[keys]
        else:
            return predict_example_utility(x, tree[keys])

    return default_output


def compute_error(y_true, y_pred, weights=None):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)
    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    if weights is None:
        weights = [1] * len(y_true)
    
    weights = np.array(weights)
    wsum = np.sum(weights)
    error_idx = np.where(y_true != y_pred)
    error_val = np.sum(weights[error_idx])
    return error_val/wsum

def confusion_matrix(true, predicted):
    """
    Create a confusion matrix using the true and predicted labels
    """
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    w, h = 2, 2
    matrix = [[0 for x in range(w)] for y in range(h)]
    for t, p in zip(true, predicted):
        if t == 1 and p == 1:
            true_positive =  true_positive + 1
        elif t == 0 and p == 0:
            true_negative = true_negative + 1
        elif t == 1 and p == 0:
            false_negative = false_negative + 1
        elif t == 0 and p == 1:
            false_positive = false_positive + 1

    matrix[0][0] = true_positive
    matrix[0][1] = false_negative
    matrix[1][0] = false_positive
    matrix[1][1] = true_negative
    return matrix

def visualize(tree, depth=0):
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))

def predict_example_boosting(x, Hypothesis):
    alpha = normalize(Hypothesis[0])
    treeList = Hypothesis[1]
    total = 0
    for i in range(len(treeList)):
        prediction = predict_example_utility(x, treeList[i])
        total = total + alpha[i] * prediction

    if total > 0.5 :
        return 1
    else:
        return 0
    
if __name__ == '__main__':

    # Load a data set
    data = DataSet('mushroom')

    # Get a list of all the attribute indices
    attribute_idx = np.array(range(data.dim))
    global_attributes = attribute_idx
    run_boost = 1
    run_bag = 1

    if run_bag == 1:
    # Runnning bagging
        depth = [3, 5]
        num_trees = [10, 20]
        for (index, dep) in enumerate(depth):
            for (idx, num) in enumerate(num_trees):
                modelList = bagging(data.examples['train'], data.labels['train'], dep, num)
                tst_pred = [predict_example_bagging(data.examples['test'][i, :], modelList) for i in range(data.num_test)]
                tst_err = compute_error(data.labels['test'], tst_pred, None)
                matrix = confusion_matrix(data.labels['test'], tst_pred)
                print("Confusion Matrix for bagging")
                print(str(matrix[0][0])+" "+str(matrix[0][1]))
                print(str(matrix[1][0])+" "+str(matrix[1][1]))
                print('d={0} bagsize={1} test error={2}'.format(dep, num, tst_err))
                
    if run_boost == 1:
    # Running boosting
        depth = [1, 2]
        num_trees = [20, 40]
        for (index, dep) in enumerate(depth):
            for (idx, num) in enumerate(num_trees):
                finalHypothesis = boosting(data.examples['train'], data.labels['train'], dep, num)
                tst_pred = [predict_example_boosting(data.examples['test'][i, :], finalHypothesis) for i in range(data.num_test)]
                tst_err = compute_error(data.labels['test'], tst_pred, None)
                matrix = confusion_matrix(data.labels['test'], tst_pred)
                print("Confusion Matrix for boosting")
                print(str(matrix[0][0])+" "+str(matrix[0][1]))
                print(str(matrix[1][0])+" "+str(matrix[1][1]))
                print('d={0} bagsize={1} test error={2}'.format(dep, num, tst_err))
