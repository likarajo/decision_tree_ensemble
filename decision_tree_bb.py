import numpy as np
from random import randint
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

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
    arr, count = np.unique(x, return_counts=True)
    for i in arr:
        x_dict[i] = []
        for j in range(len(x)):
            if i == x[j]:
                x_dict[i].append(j)
    return x_dict



def entropy(y, w=None):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z
    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    if w is None:
        w = np.ones((len(y), 1), dtype=int)
    
    hy = 0.0
    p = {0: 0, 1: 0}
    
    y_len = len(y)
    if y_len != 0:
        for i in range(y_len):
            if y[i] == 0:
                p[0] = p[0] + w[i]
            elif y[i] == 1:
                p[1] = p[1] + w[i]
        sum = p[0] + p[1]
        for j in range(len(p)):
            p[j] = p[j]/sum
            if p[j] != 0:
                hy = hy - (p[j] * np.log2(p[j]))
        return hy
    else:
        return 0

def mutual_information(x, y, w=None):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.
    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    
    if w is None:
        w = np.ones((len(y), 1), dtype=int)
    
    hy = entropy(y, w)
    x_partition = partition(x)
    
    w_entropy = 0
    total_weight = 0
    for j in x_partition:
        weight_i = np.sum(w[x_partition[j]])
        w_entropy = w_entropy + ((weight_i) * entropy(y[x_partition[j]], w[x_partition[j]]))
        total_weight = weight_i + total_weight
        
    hyx = w_entropy / total_weight
    return (hy - hyx)

def id3(x, y, attribute_value_pairs=None, max_depth=5, depth=0, w=None):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of attributes
    to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attributes is empty (there is nothing to split on), then return the most common value of y
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y
    Otherwise the algorithm selects the next best attribute using INFORMATION GAIN as the splitting criterion and
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.
    See https://gkunapuli.github.io/files/cs6375/04-DecisionTrees.pdf (Slide 18) for more details.
    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current level.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1): 1,
     (4, 2): {(3, 1): 0,
              (3, 2): 0,
              (3, 3): 0},
     (4, 3): {(5, 1): 0,
              (5, 2): 0},
     (4, 4): {(0, 1): 0,
              (0, 2): 0,
              (0, 3): 1}}
    """


    tree = {}
    arr, count = np.unique(y, return_counts=True)
    
    if len(attribute_value_pairs) == 0 or depth == max_depth or len(x) == 0:
        return arr[np.argmax(count)]
    
    elif len(arr) == 1:
        return arr[0]
    
    else:
        # calculate information gain of attribute-value pairs
        infoGain = getInfoGain(x, y, attribute_value_pairs, w)
        
        # find best attribute-value pair based on information gain
        bestAttr, bestValue = best_attribute_value_pair(infoGain)
        
        val_list = partition(x[:,bestAttr])
        
        # remove best attribute-value pair from the attribute-value pairs list
        new_attributes = list(filter(lambda x: x!= (bestAttr, bestValue), attribute_value_pairs))
        
        remaining_indicies = []
        for i in val_list:
            if i != bestValue:
                remaining_indicies.extend(val_list[i])
                      
        for i in range(2):
            if i == 0:
                index = val_list[bestValue]
                new_x = x[index]
                new_y = y[index]
                tree[bestAttr, bestValue, 'true'] = id3(new_x, new_y, new_attributes, max_depth, depth+1, w)
            else:
                new_x = x[remaining_indicies]
                new_y = y[remaining_indicies]
                tree[bestAttr, bestValue, 'false'] = id3(new_x, new_y, new_attributes, max_depth, depth+1, w)
    
    return tree

def best_attribute_value_pair(infoGain):
    maxGain = 0
    bestAttrValue = 0
    keys = list(infoGain.keys())
    for key in keys:
        gain = infoGain[key]
        if(gain >= maxGain):
            maxGain = gain
            bestAttrValue = key
    #print('maxgain',maxGain)
    return bestAttrValue

def getInfoGain(x, y, attributes, weight):
    
    infoGain = {}
    row , col = np.shape(x)

    for attr in range(0, col):
        x_partition = partition(x[:, attr])
        array = x_partition.keys();
        for attribute in attributes:
            x_vec = []
            key , value = attribute
            if(attr == key) and (value in array):
                indexes = x_partition[value]
                for i in range(0, row):
                    if i in indexes:
                        x_vec.append(1)
                    else:
                        x_vec.append(0)
                infoGain[(attr, value)] = mutual_information(x_vec, y, weight)
#                if(infoGain[(attr, value)] < 0):
#                    print('negative infoGain', infoGain[(attr, value)])
    return infoGain


def predict_label(x, tree):
    keys = list(tree.keys())
    for key in keys:
        attr, value, bool = key
        for i in range(0, len(x)):
            if i == attr:
                if x[i] == value:
                    if type(tree[key]) is dict:
                        return predict_label(x, tree[key])
                    else:
                        return tree[key]
                else:
                    newKey = (attr, value, 'false')
                    if type(tree[newKey]) is dict:
                        return predict_label(x, tree[newKey])
                    else:
                        return tree[newKey]

def predict_example(x, h_ens):
    """
    For predicting examples with bagging we recursively descend through the tree until a label/leaf node is reached.
    If the node has multiple labels, we returns the label based on majority count.

    For predicting examples with boosting we multiply the predicion with the normalized weights based on correctness.
    Based on this, we returns the predicted label.
    """  
    y_predict = []
    total_alpha = 0

    for i in h_ens:
        alpha, tree = h_ens[i]
        y = predict_label(x, tree)
        y_predict.append(y*alpha)
        total_alpha += alpha
    
    if total_alpha == len(h_ens): 
        # i.e. alpha = 1 for each tree => Bagging
        arr, count = np.unique(y_predict, return_counts=True)
        return arr[np.argmax(count)]
    
    # else Boosting  
    predictValue = np.sum(y_predict) / total_alpha
    if(predictValue >= 0.5):
        return 1
    else:
        return 0

def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)
    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    count = 0;
    label_len = len(y_true)
    for i in range(0, label_len):
        if y_pred[i] != y_true[i]:
            count+=1
    return count / label_len

def bootstrap(x):
    indexList= []
    length = len(x[:, 1])
    for i in range(0, length):
        index = randint(0, length-1)
        indexList.append(index)
    return indexList

def bagging(x, y, maxdepth, num_trees):
    h_i = {}
    attributes =[]
    rows, cols = np.shape(x)
    for i in range(cols):
        arr = np.unique(x[:, i])
        for value in arr:
            attributes.append((i, value))

    weight = np.ones((rows, 1), dtype=int)
    alpha_i = 1
    for i in range(0, num_trees):
        radIndexes = bootstrap(x)
        tree = id3(x[radIndexes], y[radIndexes], attributes, maxdepth, 0, weight)
        h_i[i] = (alpha_i, tree)
    return h_i


def boosting(x, y, max_depth, num_stumps):
    rows, cols = np.shape(x)
    weight = []
    h_ens = {}
    alpha_i = 0
    trn_pred = []
    attributes = []
    for i in range(cols):
        arr = np.unique(x[:, i])
        for value in arr:
            attributes.append((i, value))

    for stump in range(0, num_stumps):
        if stump == 0:
            d = (1/rows)
            weight = np.full((rows, 1), d)
        else:
            pre_weight = weight
            weight = []
            for i in range(rows):
                if y[i] == trn_pred[i]:
                    weight.append(pre_weight[i] * np.exp(-1*alpha_i))
                else:
                    weight.append(pre_weight[i] * np.exp(alpha_i))
            d_total = np.sum(weight)
            weight = weight / d_total
        tree = id3(x, y, attributes, max_depth, 0, weight)

        trn_pred = [predict_label(x[i, :], tree) for i in range(rows)]
        temp = 0
        for i in range(rows):
            if(trn_pred[i] != y[i]):
                temp += weight[i]
        err = (1/(np.sum(weight))) * temp
        alpha_i = 0.5 * np.log((1-err)/err)
        h_ens[stump] = (alpha_i, tree)
    return h_ens

def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    Function to print and plot the confusion matrix 
    Customized and edited the sklearn's plot_confusion_matrix function
    """
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
     
    # Only use the labels that appear in the data
    #classes = unique_labels(y_true, y_pred)

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           #xticklabels=classes, yticklabels=classes,
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
    Mtrn = np.genfromtxt('./data/mushroom.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = Mtrn[:, 0]
    Xtrn = Mtrn[:, 1:]

    Mtst = np.genfromtxt('./data/mushroom.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = Mtst[:, 0]
    Xtst = Mtst[:, 1:]
    
    _bagging = True
    _boosting = True
    
    _self = True
    _scikit = True
    
    tree_depth = np.array([3,5])
    stump_depth = np.array([1,2])
    ens_size = np.array([5,10])

    if (_self):
        print('Self-implementation')
        if(_bagging):
            print('Bagging')
            for n in ens_size:
                for d in tree_depth:
                    print('Bag Size {0} | Tree Depth {1}'.format(n, d))
                    modelList = bagging(Xtrn, ytrn, d, n)
                    self_y_pred = [predict_example(Xtst[i, :], modelList) for i in range(Xtst.shape[0])]
                    plot_confusion_matrix(ytst, self_y_pred, normalize=False, title='self_bagging_depth{0}_size{1}_cm'.format(d,n))
                    tn, fp, fn, tp = confusion_matrix(ytst, self_y_pred).ravel()
                    print("(tn, fp, fn, tp) = ",(tn, fp, fn, tp))
                    
        if(_boosting):
            print('Boosting')
            for n in ens_size:
                for d in stump_depth:
                    print('Ensemble Size {0} | Stump Depth {1}'.format(n, d)) 
                    model = boosting(Xtrn, ytrn, d, n)
                    self_y_pred = [predict_example(Xtst[i, :], model) for i in range(Xtst.shape[0])]
                    plot_confusion_matrix(ytst, self_y_pred, normalize=False, title='self_boosting_depth{0}_size{1}_cm'.format(d,n))
                    tn, fp, fn, tp = confusion_matrix(ytst, self_y_pred).ravel()
                    print("(tn, fp, fn, tp) = ",(tn, fp, fn, tp))
                    
                
    if (_scikit):
        print('Scikit implementation') 
        if(_bagging):
            print('Bagging')
            for n in ens_size:
                for d in tree_depth:     
                    print('Bag Size {0} | Tree Depth {1}'.format(n, d))
                    dtree = DecisionTreeClassifier(criterion='entropy', max_depth=d)
                    modelList = BaggingClassifier(base_estimator=dtree, n_estimators=n, bootstrap=True)
                    modelList.fit(Xtrn, ytrn)
                    self_y_pred = modelList.predict(Xtst)
                    plot_confusion_matrix(ytst, self_y_pred, normalize=False, title='sklearn_bagging_depth{0}_size{1}_cm'.format(d,n))
                    tn, fp, fn, tp = confusion_matrix(ytst, self_y_pred).ravel()
                    print("(tn, fp, fn, tp) = ", (tn, fp, fn, tp))
                    
        if(_boosting):
            print('Boosting')
            for n in ens_size:
                for d in stump_depth:
                    print('Ensemble Size {0} | Stump Depth {1}'.format(n, d))
                    dtree = DecisionTreeClassifier(criterion='entropy', max_depth=d)
                    model = AdaBoostClassifier(base_estimator=dtree, n_estimators=n)
                    model.fit(Xtrn, ytrn)
                    self_y_pred = model.predict(Xtst)
                    plot_confusion_matrix(ytst, self_y_pred, normalize=False, title='sklearn_boosting_depth{0}_size{1}_cm'.format(d,n))
                    tn, fp, fn, tp = confusion_matrix(ytst, self_y_pred).ravel()
                    print("(tn, fp, fn, tp) = ", (tn, fp, fn, tp))
                
