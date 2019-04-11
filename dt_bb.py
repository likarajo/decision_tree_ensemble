import numpy as np
import math
import matplotlib.pyplot as plt
from random import seed
from random import random
from random import randrange
from random import randint
import copy 

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

def entropy(y, w=None):

    if w is None:
        w = np.ones((len(y), 1), dtype=int)
        
    hy = 0
    p = {0: 0, 1: 0}
    
    if len(y) == 0:
        return 0
    else:
        for i in range(len(y)):
            if y[i] == 0:
                p[0] = p[0] + w[i]
            elif y[i] == 1:
                p[1] = p[1] + w[i]
                
        for j in range(len(p)):
            p[j] = p[j]/(p[0] + p[1])
            if p[j] != 0:
                hy = hy - (p[j] * np.log2(p[j]))
        return hy


def mutual_information(x, y, w=None):
    
    if w is None:
        w = np.ones((len(y), 1), dtype=int)
        
    hy = entropy(y, w)
    x_list = partition(x)
    
    w_entropy = 0
    total_weight = 0
    for i in x_list:
        wi = np.sum(w[x_list[i]])
        w_entropy = w_entropy + (wi * entropy(y[x_list[i]], w[x_list[i]]))
        total_weight = total_weight + wi
        
    hyx = w_entropy / total_weight
    
    mi = hy - hyx
    
    if (mi) < 0:
        mi = 0
        
    return (mi)

def get_best_attribute(x, y, attribute_value_pairs, w=None):
    """
    For obtaining the best attribute with max info gain from the list of attribute-value pairs
    """
    if w is None:
        w = [1.] * len(x)
    
    maxGain = 0
    
    for pair in attribute_value_pairs:
        attr = pair[0]
        val = pair[1]
        x_vec = copy.deepcopy(x[:,attr])
        
        x_vec[np.where(x_vec == val)[0]] = 1
        x_vec[np.where(x_vec != val)[0]] = 0
        
        gain = mutual_information(x_vec, y, w)
        
        if gain >= maxGain:
            maxGain = gain
            best_attr = pair

    return best_attr

def id3(x, y, attribute_value_pairs=None, max_depth=5, depth=0, w=None):
    
    if np.unique(y).size == 1:
        return y[0]
    
    elif (len(attribute_value_pairs) == 0) or (depth == max_depth) or (len(x) == 0):
        count = np.bincount(y)
        if len(count) == 0:
            return 1
        else:
            return np.argmax(count)
    else:
         # Find the best attribute that has maximum Mutual Information
        best_attr = get_best_attribute(x, y, attribute_value_pairs, w)
        
        # Delete the best attribute from the attribute-value-pairs list
        avpair_true = attribute_value_pairs[:]
        avpair_true.remove(best_attr)
        avpair_false = attribute_value_pairs[:]
        avpair_false.remove(best_attr)
        
        tree = {}
        
        idx_true = np.where(x[:,best_attr[0]] == best_attr[1])[0]
        idx_false = np.where(x[:,best_attr[0]] != best_attr[1])[0]
        tree[(best_attr[0], best_attr[1], "true")] = id3(x[idx_true], y[idx_true], avpair_true, max_depth, depth+1, w)
        tree[(best_attr[0], best_attr[1], "false")] = id3(x[idx_false], y[idx_false], avpair_false, max_depth, depth+1, w)
            
    return tree

def bagging(x, y, max_depth, num_trees):
    """
    Learn multiple deep trees
    """
    ensemble = list()

    for i in range(num_trees):
        indices = list()
        for n in range(len(x)): # with replacement
            k = randrange(0, len(x)) # generate random indices using bootstrap 
            indices.append(k)

        _x = x[indices]
        
        bagging_attribute_value_pair = list()
        for attr in np.array(range(x.shape[1])):
            col = _x[:,attr]
            for val in np.unique(col):
                bagging_attribute_value_pair.append([attr, val])
                
        weights = np.ones((len(y), 1), dtype=int) # in bagging all instances have same weight
        h_i = id3(x[indices], y[indices], bagging_attribute_value_pair, max_depth, 0, weights)
        
        ensemble.append([1, h_i]) # alpha is 1 (same) for all trees
    
    return ensemble

def boosting(x, y, max_depth, num_stumps):
    """
    Learn multiple stumps   
    """
    ensemble = list()
    
    # initialize empty weights and stumps
    alpha = [0] * num_stumps
    h = [None] * num_stumps
    
    alpha_i = 0
    
    boosting_attribute_value_pair = list()
    for attr in np.array(range(x.shape[1])):
        col = x[:,attr]
        val_list = np.unique(col)
        for val in val_list:
            boosting_attribute_value_pair.append([attr, val])
    
    rows, cols = np.shape(x)
    
    weights = [] 
    for i in range(num_stumps):
        if i == 0:
            # for the first time weigh all data points equally
            d = 1/rows
            weights = np.full((rows, 1), d)
        else:
            # for subsequent runs save previous weights, and update weights of datapoints based on prediction
            prev_weights = weights
            weights = []
            for n in range(rows):
                if y[n] == classification[i]:
                    # for correct classification reduce weight
                    weights.append(prev_weights[n] * np.exp(-1*alpha_i))
                else:
                    # for incorrect classification increase weight
                    weights.append(prev_weights[n] * np.exp(alpha_i))
                    
            weights = weights / np.sum(weights)

        h_i = id3(x, y, boosting_attribute_value_pair, max_depth, 0, weights)
        
        classification = [predict_example_utility(x[k, :], h_i) for k in range(rows)]
        wsum = 0
        for n in range(rows):
            if(classification[n] != y[n]):
                wsum = wsum + weights[n]
                
        error = (1/(np.sum(weights))) * wsum
        alpha_i = 0.5 * np.log((1 - error)/error)
        
        alpha[i] = alpha_i
        h[i] = h_i
        
    ensemble = [alpha, h]
    return ensemble

def normalize(weights):
    """
    Normalize the given dataset
    """
    norm = sum(weights)
    weights = [x / norm for x in weights]
    return weights

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

    # Load the training and testing data
    Mtrn = np.genfromtxt('./data/mushroom.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = Mtrn[:, 0]
    Xtrn = Mtrn[:, 1:]

    Mtst = np.genfromtxt('./data/mushroom.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = Mtst[:, 0]
    Xtst = Mtst[:, 1:]

    # Get a list of all the attribute indices 
    global_attributes = np.array(range(Xtrn.shape[1]))
    
    run_boost = 1
    run_bag = 1

    if run_bag == 1:
    # Runnning bagging
        modelList = bagging(Xtrn, ytrn, 3, 5)
        tst_pred = [predict_example_bagging(Xtst[i, :], modelList) for i in range(Xtst.shape[0])]
        tst_err = compute_error(ytst, tst_pred, None)
        print('test error={}'.format(tst_err))
                
    if run_boost == 1:
    # Running boosting
        finalHypothesis = boosting(Xtrn, ytrn, 1, 5)
        tst_pred = [predict_example_boosting(Xtst[i, :], finalHypothesis) for i in range(Xtst.shape[0])]
        tst_err = compute_error(ytst, tst_pred, None)
        print('test error={}'.format(tst_err))
