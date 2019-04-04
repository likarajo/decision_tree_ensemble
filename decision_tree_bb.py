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
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split

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
    
    x_dict = {}
    j = 0
    for i in x:
        if i not in x_dict:
            x_dict[i] = []
            x_dict[i].append(j)
        else:
            x_dict[i].append(j)
        j = j + 1

    return x_dict

    #return {c: (x==c).nonzero()[0] for c in np.unique(x)}
    
    raise Exception('Function not yet implemented!')

#x1 = [1,2,3,2]
#print(partition(x1))

def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE
    
    z = partition(y)
    h = 0.0

    for k, v in z.items():
        # p = length of the list of indices for each item in z divided by the total length of vector y
        p = len(v) / len(y)
        # update the entropy with plog(p).
        h = h - ( p * np.log2(p))

    return h

    raise Exception('Function not yet implemented!')


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # INSERT YOUR CODE HERE

    # list of unique values in x
    x_unique = partition(x)

    # counts of the unique values in x
    x_counts = []
    for k in x_unique.keys():
        x_counts.append(len(x_unique[k]))

    # probabilty of the unique values in x
    probs_x = []
    for i in x_counts:
        probs_x.append(i / len(x))
        
    # Entropy
    hy = entropy(y)

    # Conditional entropy H(y | x)
    hyx = 0.0

    # Weighted average entropy over all possible splits
    j = 0
    for i, v in x_unique.items():
        hyx = hyx + probs_x[j] * entropy(y[v])
        j += 1

    # Mutual Information or Information Gain, MI = H(y) - H( y | x)
    mi = hy - hyx

    return mi

    raise Exception('Function not yet implemented!')


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
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
        mi = mutual_information(np.array(x[:, a] == v), y)
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
        tree[(attr, value, decision)] = id3(x_subset, y_subset, attribute_value_pairs=attribute_value_pairs, max_depth = md - 1)

    return tree

    raise Exception('Function not yet implemented!')


def predict_example(x, tree):
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
                predicted_label = predict_example(x, subtrees)
            else:
                predicted_label = subtrees
            
            return predicted_label

    #raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    misclassifications = 0
    n = len(y_true)
    
    for i in range(n):
        if(y_pred[i] != y_true[i]):
            misclassifications += 1
    
    return misclassifications/n

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

if __name__ == '__main__':
    # Load the training and testing data
    Mtrn1 = np.genfromtxt('./data/monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn1 = Mtrn1[:, 0]
    Xtrn1 = Mtrn1[:, 1:]

    Mtst1 = np.genfromtxt('./data/monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst1 = Mtst1[:, 0]
    Xtst1 = Mtst1[:, 1:]
    
    Mtrn2 = np.genfromtxt('./data/monks-2.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn2 = Mtrn2[:, 0]
    Xtrn2 = Mtrn2[:, 1:]

    Mtst2 = np.genfromtxt('./data/monks-2.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst2 = Mtst2[:, 0]
    Xtst2 = Mtst2[:, 1:]
    
    Mtrn3 = np.genfromtxt('./data/monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn3 = Mtrn3[:, 0]
    Xtrn3 = Mtrn3[:, 1:]

    Mtst3 = np.genfromtxt('./data/monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst3 = Mtst3[:, 0]
    Xtst3 = Mtst3[:, 1:]
    
    # function to change the strings in dataset to integers
    convertfunc = lambda symbol: 1 if symbol=='x' else 2 if symbol=='o' else 3 if symbol=='b' else 1 if symbol=='positive' else 0 if symbol=='negative' else 1      
            
    y_ttt = np.genfromtxt('./data/tic-tac-toe.txt', missing_values=0, skip_header=0, delimiter=',', dtype=None, encoding='ascii',
                          converters={9: convertfunc}, usecols=9)
    X_ttt = np.genfromtxt('./data/tic-tac-toe.txt', missing_values=0, skip_header=0, delimiter=',', dtype=None, encoding='ascii',
                          converters={0:convertfunc,
                                      1:convertfunc,
                                      2:convertfunc,
                                      3:convertfunc,
                                      4:convertfunc,
                                      5:convertfunc,
                                      6:convertfunc,
                                      7:convertfunc,
                                      8:convertfunc}, usecols=range(9))

    X_ttt_trn, X_ttt_tst, y_ttt_trn, y_ttt_tst = train_test_split(X_ttt, y_ttt, test_size=0.3, random_state=42)
 

#=========Test in monks-1=================================
    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn1, ytrn1, max_depth=3)

    # Pretty print it to console
    pprint(decision_tree)
    #pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree)
    render_dot_file(dot_str, './my_learned_tree')

    # Compute the train and test error
    y_pred = [predict_example(x, decision_tree) for x in Xtrn1]
    trn_err = compute_error(ytrn1, y_pred)
    y_pred = [predict_example(x, decision_tree) for x in Xtst1]
    tst_err = compute_error(ytst1, y_pred)

    print('Train Error = {0:4.2f}%.'.format(trn_err * 100))
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))

#=========For depth = 1 to 10, learn decision trees 
#         and compute the average training and test errors on each of 
#         the three MONK’s problems===================================

    trn_err1 = {}
    tst_err1 = {}
    
    trn_err2 = {}
    tst_err2 = {}
    
    trn_err3 = {}
    tst_err3 = {}    
    
    for d in range (1,11):
        # Learn a decision tree of depth d
        decision_tree1 = id3(Xtrn1, ytrn1, max_depth=d)
        decision_tree2 = id3(Xtrn2, ytrn2, max_depth=d)
        decision_tree3 = id3(Xtrn3, ytrn3, max_depth=d)
        
#        # Visualize the tree and save it as a PNG image
#        dot_str = to_graphviz(decision_tree1)
#        render_dot_file(dot_str, './my_learned_tree_monks1_depth'+str(d))
#        dot_str = to_graphviz(decision_tree2)
#        render_dot_file(dot_str, './my_learned_tree_monks2_depth'+str(d))
#        dot_str = to_graphviz(decision_tree3)
#        render_dot_file(dot_str, './my_learned_tree_monks3_depth'+str(d))
        
        # Predict using train set
        y_pred1 = [predict_example(x, decision_tree1) for x in Xtrn1]
        y_pred2 = [predict_example(x, decision_tree2) for x in Xtrn2]
        y_pred3 = [predict_example(x, decision_tree3) for x in Xtrn3]
        
        # Compute and store the training errors
        trn_err1[d] = compute_error(ytrn1, y_pred1)
        trn_err2[d] = compute_error(ytrn2, y_pred2)
        trn_err3[d] = compute_error(ytrn3, y_pred3)
        
        # Predict using test set
        y_pred1 = [predict_example(x, decision_tree1) for x in Xtst1]
        y_pred2 = [predict_example(x, decision_tree2) for x in Xtst2]
        y_pred3 = [predict_example(x, decision_tree3) for x in Xtst3]
        
        # Compute and store the test errors 
        tst_err1[d] = compute_error(ytst1, y_pred1)
        tst_err2[d] = compute_error(ytst2, y_pred2)
        tst_err3[d] = compute_error(ytst3, y_pred3)
      
    # Plot the training and Testing errors with varying depth
    fig1 = plt.figure()
    fig1.suptitle('Error on monks-1')
    plt.plot(trn_err1.keys(), trn_err1.values(), marker='s', linewidth=2, markersize=7)
    plt.plot(tst_err1.keys(), tst_err1.values(), marker='o', linewidth=2, markersize=7)
    plt.xlabel('Tree depth', fontsize=10)
    plt.ylabel('Trn/Test error', fontsize=10)
    plt.xticks(list(trn_err1.keys()), fontsize=10)
    plt.legend(['Training Error', 'Test Error'], fontsize=10)
    fig1.savefig('./monks1_error.jpg')
    
    fig2 = plt.figure()
    fig2.suptitle('Error on monks-2')
    plt.plot(trn_err2.keys(), trn_err2.values(), marker='s', linewidth=2, markersize=7)
    plt.plot(tst_err2.keys(), tst_err2.values(), marker='o', linewidth=2, markersize=7)
    plt.xlabel('Tree depth', fontsize=10)
    plt.ylabel('Trn/Test error', fontsize=10)
    plt.xticks(list(trn_err2.keys()), fontsize=10)
    plt.legend(['Training Error', 'Test Error'], fontsize=10)
    fig2.savefig('./monks2_error.jpg')

    fig3 = plt.figure()
    fig3.suptitle('Error on monks-3')
    plt.plot(trn_err3.keys(), trn_err3.values(), marker='s', linewidth=2, markersize=7)
    plt.plot(tst_err3.keys(), tst_err3.values(), marker='o', linewidth=2, markersize=7)
    plt.xlabel('Tree depth', fontsize=10)
    plt.ylabel('Trn/Test error', fontsize=10)
    plt.xticks(list(trn_err3.keys()), fontsize=10)
    plt.legend(['Training Error', 'Test Error'], fontsize=10)
    fig3.savefig('./monks3_error.jpg')
    
    
#===========For monks-1, learned decision tree using self implemented classifier and scikit's version
#           and found the confusion matrix on the test set for depth = 1, 3, 5========================
    
    print("Self-implementation on monks-1")
    for d in range(1, 6, 2):
        # Learn classifier of depth d
        self_clf = id3(Xtrn1, ytrn1, max_depth=d)
        print("For depth = ",d)
#        # Pretty print decision tree to console 
#        pprint(self_clf)
#        #pretty_print(self_clf)
        # Visualize the tree and save it as a PNG image
        dot_str = to_graphviz(self_clf)
        render_dot_file(dot_str, './my_learned_tree_monks1_depth{0}'.format(d))
        os.remove('./my_learned_tree_monks1_depth{0}'.format(d))
        # Predict using test set
        self_y_pred = [predict_example(x, self_clf) for x in Xtst1]
        # Compute and plot the confusion matrix
        tn, fp, fn, tp = confusion_matrix(ytst1, self_y_pred).ravel()
        print("(tn, fp, fn, tp) = ", (tn, fp, fn, tp))
        plot_confusion_matrix(ytst1, self_y_pred, normalize=False,
                      title='my_learned_tree_monks1_depth{0}_cm'.format(d))
        
    print("Scikit-learn's implementation on monks-1")
    for d in range(1, 6, 2):
        # Learn classifier of depth d
        sklearn_clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
        sklearn_clf.fit(Xtrn1, ytrn1)
        print("For depth = ",d)
        # Visualize the tree and save it as a PNG image
        dot_data = export_graphviz(sklearn_clf)  
        render_dot_file(dot_data, './sklearn_tree_monks1_depth{0}'.format(d))
        os.remove("./sklearn_tree_monks1_depth{0}".format(d))
        # Predict using test set
        sklearn_y_pred = sklearn_clf.predict(Xtst1, check_input=True)
        # Compute and plot the confusion matrix (reference from sklearn.plot_confusion_matrix)
        tn, fp, fn, tp = confusion_matrix(ytst1, sklearn_y_pred).ravel()
        print("(tn, fp, fn, tp) = ", (tn, fp, fn, tp))
        plot_confusion_matrix(ytst1, sklearn_y_pred, normalize=False,
                      title='sklearn_tree_monks1_depth{0}_cm'.format(d)) 

#===========For personal dataset, learned decision tree using self implemented classifier and scikit's version
#           and found the confusion matrix on the test set for depth = 1, 3, 5========================
   
    print("Self implementation on tic-tac-toe data")
    for d in range(1, 6, 2):
        # Learn classifier of depth d
        self_clf = id3(X_ttt_trn, y_ttt_trn, max_depth=d)
        print("For depth = ",d)
#        # Pretty print decision tree to console 
#        pprint(self_clf)
#        #pretty_print(self_clf)
        # Visualize the tree and save it as a PNG image
        dot_str = to_graphviz(self_clf)
        render_dot_file(dot_str, './my_learned_tree_ttt_depth{0}'.format(d))
        os.remove('./my_learned_tree_ttt_depth{0}'.format(d))
        # Predict using test set
        self_y_pred = [predict_example(x, self_clf) for x in X_ttt_tst]
        # Compute and plot the confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_ttt_tst, self_y_pred).ravel()
        print("(tn, fp, fn, tp) = ", (tn, fp, fn, tp))
        plot_confusion_matrix(y_ttt_tst, self_y_pred, normalize=False,
                      title='my_learned_tree_ttt_depth{0}_cm'.format(d))
        
    print("Scikit-learn's implementation on tic-tac-toe data")
    for d in range(1, 6, 2):
        # Learn classifier of depth d
        sklearn_clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
        sklearn_clf.fit(X_ttt_trn, y_ttt_trn)
        print("For depth = ",d)
        # Visualize the tree and save it as a PNG image
        dot_data = export_graphviz(sklearn_clf)  
        render_dot_file(dot_data, './sklearn_tree_ttt_depth{0}'.format(d))
        os.remove("./sklearn_tree_ttt_depth{0}".format(d))
        # Predict using test set
        sklearn_y_pred = sklearn_clf.predict(X_ttt_tst, check_input=True)
        # Compute the confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_ttt_tst, sklearn_y_pred).ravel()
        print("(tn, fp, fn, tp) = ", (tn, fp, fn, tp))
        plot_confusion_matrix(y_ttt_tst, sklearn_y_pred, normalize=False,
                      title='sklearn_tree_ttt_depth{0}_cm'.format(d))
