# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework 3 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import os
import graphviz

from pprint import pprint

import matplotlib.pyplot as plt

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split

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
    
    # INSERT YOUR CODE HERE
    
    x_list = {}
    unique_x = np.unique(x)

    for x_attr in unique_x:
        x_list[x_attr] = np.where(x == x_attr)[0]

    return x_list


def entropy(y, w=None):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    
    if w is None:
        w = np.ones((len(y), 1), dtype=int)
        
    hy = 0.0
    
    _y = {0: 0, 1: 0}
    
    if len(y) == 0:
        return 0
    
    else:
        
        for i in range(len(y)):
            if y[i] == 0:
                _y[0] = _y[0] + w[i]
            elif y[i] == 1:
                _y[1] = _y[1] + w[i]

        for j in range(len(_y)):
            _y[j] = _y[j]/(_y[0] + _y[1])
            
            if _y[j] != 0:
                hy = hy - (_y[j] * np.log2(_y[j]))
                
        return hy
        

    

def mutual_information(x, y, w=None):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # INSERT YOUR CODE HERE
    
    if w is None:
        w = np.ones((len(y), 1), dtype=int)
    h_y = entropy(y, w)
    x_partition = partition(x)
    temp = 0
    total_weight = 0
    for j in x_partition:
        weight_i = np.sum(w[x_partition[j]])
        temp = ((weight_i) * entropy(y[x_partition[j]], w[x_partition[j]])) + temp
        total_weight = weight_i + total_weight
    h_y_x = temp / total_weight
    info_gain = h_y - h_y_x
    if (info_gain) < 0:
        info_gain = 0
    return (info_gain)
    
def best_attribute(x, y, attribute_value_pairs, w=None):
    
    if w is None:
        w = [1.] * len(x)
    
    max_info_gain = 0
    bestAttr = 0
    
    for attr_pair in attribute_value_pairs:
        
        x_attr = attr_pair[0]
        x_value = attr_pair[1]
        
        x_col = copy.deepcopy(x[:,x_attr])
        
        xi = np.where(x_col == x_value)[0]
        xni = np.where(x_col != x_value)[0]
        
        x_col[xi] = 1
        x_col[xni] = 0
        
        gain = mutual_information(x_col, y, w)
        
        if gain >= max_info_gain:
            max_info_gain = gain
            bestAttr = attr_pair

    return bestAttr


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5, w=None):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    
    md = max_depth
    
    if w is None:
        weights = [1.] * len(x)
    else:
        weights=w

    tree = {}

    # list of unique values in y
    y_unique = partition(y)

    # counts of unique values in y
    y_counts = []
    for k in y_unique.keys():
        y_counts.append(len(y_unique[k]))

    # When y is Pure, i.e. only one unique value in the set, return the label
    if len(list(y_unique.keys())) == 1:
        return y[0]
    
    # Forming the attribute-value pairs list
    if attribute_value_pairs is None:
        attribute_value_pairs = []
        # for each of the attributes
        for attr in range(x.shape[1]):
            # for each of the unique values
            for val in partition(x[:, attr]).keys():
                # form attribute value pair
                attribute_value_pairs.append((attr, val))
        # Convert list to array
        attribute_value_pairs = np.array(attribute_value_pairs)

    # When no attribute-value pairs left to split on, return the majority label
    if len(attribute_value_pairs) == 0:
        max_count = 0
        majority_label = 0
        for l, c in y_unique.items():
            if len(c) >= max_count:
                max_count = len(c)
                majority_label = l
        return majority_label

    # When the Tree has grown to maximum depth, return the majority label
    if md==0:
        max_count = 0
        majority_label = 0
        for l, c in y_unique.items():
            if len(c) >= max_count:
                max_count = len(c)
                majority_label = l
        return majority_label

    # Find the best attribute that has maximum Mutual Information
    mi_max = 0.0
    attr = 0
    value = 0
    for (a, v) in attribute_value_pairs:
        mi = mutual_information(np.array(x[:, a] == v), y, w)
        if mi >= mi_max:
            mi_max = mi
            attr = a
            value = v

    # Delete the best attribute from the attribute-value-pairs list
    attribute_value_pairs = np.delete(attribute_value_pairs, attribute_value_pairs[np.where(attribute_value_pairs == (attr, value))], 0)

    # Split on the best attribute
    splits = partition(np.array(x[:, attr] == value).astype(int))

    for split_value, indices in splits.items():
        x_subset = x.take(indices, axis=0)
        y_subset = y.take(indices, axis=0)
        decision = bool(split_value)
        tree[(attr, value, decision)] = id3(x=x_subset, y=y_subset, attribute_value_pairs=attribute_value_pairs, max_depth = md - 1, w=weights)

    return tree

    raise Exception('Function not yet implemented!')


def predict(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    for nodes, subtrees in tree.items():
        _feature = nodes[0]
        _value = nodes[1]
        _decision = nodes[2]
        
        if _decision == (x[_feature] == _value):
            if type(subtrees) is dict:
                predicted_label = predict(x, subtrees)
            else:
                predicted_label = subtrees
            
            return predicted_label

    #raise Exception('Function not yet implemented!')
    
def normalize(weights):
    """
    Normalize the given dataset
    """
    norm = sum(weights)
    weights = [x / norm for x in weights]
    return weights

    
def predict_example(x, h_ens):
    
    if np.unique(h_ens[0]).count == 1:
      
        resultList = list()
        
        for tree in h_ens:
            result = predict(x, tree[1])
            resultList.append(result)
    
        counts = np.bincount(resultList)
        if len(counts) == 0:
            return 1
        else:
            return np.argmax(counts)
    
    else:
    
        alpha = normalize(h_ens[0])
        h = h_ens[1]
        total = 0
        for i in range(len(h)):
            prediction = predict(x, h[i])
            total = total + (alpha[i] * prediction)
    
        if total > 0.5 :
            return 1
        else:
            return 0


def compute_error(y_true, y_pred, w=None):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    
    if w is None:
        w = [1] * len(y_true)
    
    weights = np.array(w)
    wsum = np.sum(weights)
    i = np.where(y_true != y_pred)
    v = np.sum(weights[i])
    return v/wsum

    raise Exception('Function not yet implemented!')


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid
 
# Function to print and plot the confusion matrix (customized and edited the sklearn's plot_confusion_matrix function)
def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()
    fig.savefig('./'+title+'.jpg')
    return ax
'''
def boosting(x, y, max_depth, num_stumps):
    
    treeList = [None] * num_stumps # initialize an empty list of trees
    alpha = [0] * num_stumps # initilaize an empty list of weights
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
'''

def bagging(x, y, max_depth, num_trees):
    sampleTrees = list()
    size = len(x)
    for i in range(num_trees):
        sample_indices = np.random.choice(size, size, replace=True)
        sample_x, sample_y = x[sample_indices], y[sample_indices]
        #(sample_x)
        #print(sample_y)
        dtree = id3(x=sample_x, y=sample_y, max_depth=max_depth)
        #pprint(dtree)
        #print("================")
        sampleTrees.append([1, dtree])
        
    return sampleTrees

if __name__ == '__main__':
    # Load the training and testing data
    Mtrn = np.genfromtxt('./data/mushroom.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = Mtrn[:, 0]
    Xtrn = Mtrn[:, 1:]

    Mtst = np.genfromtxt('./data/mushroom.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = Mtst[:, 0]
    Xtst = Mtst[:, 1:]
    
    baggingDepths = np.array([3,5])
    boostingDepths = np.array([1,2])
    ensembleSize = np.array([5,10])
    
    runBagging = True
    runBoosting = False
    
    if (runBagging):
        treeList = bagging(Xtrn, ytrn, 3, 5)
        tst_pred = predict_example(Xtrn, treeList)
        tst_err = compute_error(ytst, tst_pred, None)
        print('test error={2}'.format(tst_err))
    
'''
#=========Test in mushroom dataset=================================
    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3, w=weights)

    # Pretty print it to console
    pprint(decision_tree)
    #pretty_print(decision_tree)
    
    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree)
    render_dot_file(dot_str, './my_learned_tree')

    # Compute the train and test error
    y_pred = [predict_example(x, decision_tree) for x in Xtrn]
    trn_err = compute_error(ytrn, y_pred)
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print('Train Error = {0:4.2f}%.'.format(trn_err * 100))
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
        
#===========For mushrooms, learned decision tree using self implemented classifier and scikit's version
#           and found the confusion matrix on the test set for depth = 1, 3, 5========================
    
    print("Self-implementation on monks-1")
    for d in range(1, 6, 2):
        # Learn classifier of depth d
        self_clf = id3(Xtrn, ytrn, max_depth=d)
        print("For depth = ",d)
#        # Pretty print decision tree to console 
#        pprint(self_clf)
#        #pretty_print(self_clf)
        # Visualize the tree and save it as a PNG image
        dot_str = to_graphviz(self_clf)
        render_dot_file(dot_str, './my_learned_tree_monks1_depth{0}'.format(d))
        os.remove('./my_learned_tree_monks1_depth{0}'.format(d))
        # Predict using test set
        self_y_pred = [predict_example(x, self_clf) for x in Xtst]
        # Compute and plot the confusion matrix
        tn, fp, fn, tp = confusion_matrix(ytst, self_y_pred).ravel()
        print("(tn, fp, fn, tp) = ", (tn, fp, fn, tp))
        plot_confusion_matrix(ytst, self_y_pred, normalize=False,
                      title='my_learned_tree_monks1_depth{0}_cm'.format(d))
        
    print("Scikit-learn's implementation on monks-1")
    for d in range(1, 6, 2):
        # Learn classifier of depth d
        sklearn_clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
        sklearn_clf.fit(Xtrn, ytrn)
        print("For depth = ",d)
        # Visualize the tree and save it as a PNG image
        dot_data = export_graphviz(sklearn_clf)  
        render_dot_file(dot_data, './sklearn_tree_monks1_depth{0}'.format(d))
        os.remove("./sklearn_tree_monks1_depth{0}".format(d))
        # Predict using test set
        sklearn_y_pred = sklearn_clf.predict(Xtst, check_input=True)
        # Compute and plot the confusion matrix (reference from sklearn.plot_confusion_matrix)
        tn, fp, fn, tp = confusion_matrix(ytst, sklearn_y_pred).ravel()
        print("(tn, fp, fn, tp) = ", (tn, fp, fn, tp))
        plot_confusion_matrix(ytst, sklearn_y_pred, normalize=False,
                      title='sklearn_tree_monks1_depth{0}_cm'.format(d)) 
'''