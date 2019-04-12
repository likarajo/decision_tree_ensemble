import numpy as np

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from random import randint


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)
    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    temp = {
    }
    arr, count = np.unique(x, return_counts=True)
    for i in arr:
        temp[i] = []
        for j in range(len(x)):
            if i == x[j]:
                temp[i].append(j)
    return temp



def entropy(y, weight):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z
    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    h = 0
    temp = {
        0: 0,
        1: 0
        }
    y_len = len(y)
    if y_len != 0:
        for i in range(y_len):
            if y[i] == 0:
                temp[0] = temp[0] + weight[i]
            elif y[i] == 1:
                temp[1] = temp[1] + weight[i]
        sum = temp[0] + temp[1]
        for j in range(len(temp)):
            temp[j] = temp[j]/sum
            if temp[j] != 0:
                h = temp[j] * np.log2(temp[j]) + h
        return -h
    else:
        return 0

def mutual_information(x, y, weight):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.
    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    h_y = entropy(y, weight)
    x_partition = partition(x)
    temp = 0
    total_weight = 0
    for j in x_partition:
        weight_i = np.sum(weight[x_partition[j]])
        temp = ((weight_i) * entropy(y[x_partition[j]], weight[x_partition[j]])) + temp
        total_weight = weight_i + total_weight
    h_y_of_x = temp / total_weight
    return (h_y - h_y_of_x)

def id3(x, y, attributes, max_depth, weight, depth=0):
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
    ''' return the max of label if attribute array is empty or depth of the current itireation
        reaches the specified max depth of the tree or when input array is empty'''
    if len(attributes) == 0 or depth == max_depth or len(x) == 0:
        return arr[np.argmax(count)]
    elif len(arr) == 1:
        ''' return 1 when all the values of label list are one'''
        return arr[0]
    else:
        ''' if none of the above cases matches then find the best attribute to split and call id3 recursively'''
        informationGain = get_mutual_information(x, y, attributes, weight)
        bestAttr, bestValue = choose_attribute(informationGain)
        a = partition(x[:,bestAttr])
        new_attributes = list(filter(lambda x: x!= (bestAttr, bestValue), attributes))
        non_best_indicies = []
        for i in a:
            if i != bestValue:
                non_best_indicies.extend(a[i])
        depth+=1
        for i in range(0,2):
            if i == 0:
                index = a[bestValue]
                new_x = x[index]
                new_y = y[index]
                tree[bestAttr, bestValue, 'true'] = id3(new_x, new_y, new_attributes, max_depth,weight, depth)
            else:
                new_x = x[non_best_indicies]
                new_y = y[non_best_indicies]
                tree[bestAttr, bestValue, 'false'] = id3(new_x, new_y, new_attributes, max_depth,weight, depth)
    return tree


"""
    choose_attribute is to choose the best attribute which has maximum gain
"""
def choose_attribute(infoGain):
    maxGain = 0
    bestAttrVlaue = 0
    keys = list(infoGain.keys())
    for key in keys:
        gain = infoGain[key]
        if(gain >= maxGain):
            maxGain = gain
            bestAttrVlaue = key
    print('maxgain',maxGain)
    return bestAttrVlaue

def get_mutual_information(x, y, attributes, weight):
    infoGain = {}
    row , col = np.shape(x)

    for attr in range(0, col):
        x_partition = partition(x[:, attr])
        array = x_partition.keys();
        for attribute in attributes:
            temp = []
            key , value = attribute
            if(attr == key) and (value in array):
                indexes = x_partition[value]
                for i in range(0, row):
                    if i in indexes:
                        temp.append(1)
                    else:
                        temp.append(0)
                infoGain[(attr, value)] = mutual_information(temp, y, weight)
                if(infoGain[(attr, value)] < 0):
                    print('negative info', infoGain[(attr, value)])
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
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.
    Returns the predicted label of x according to tree
    """
    y_preict = []

    for numHypo in h_ens:
        alpha, tree = h_ens[numHypo]
        y = predict_label(x, tree)
        y_preict.append(y)
    arr, count = np.unique(y_preict, return_counts=True)
    return arr[np.argmax(count)]

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

def randomFunction(x):
    indexList= []
    length = len(x[:, 1])
    for i in range(0, length):
        index = randint(0, length-1)
        indexList.append(index)
    return indexList

def bagging(x, y, maxdepth, numtrees):
    h_i = {}
    attributes =[]
    rows, cols = np.shape(x)
    for i in range(cols):
        arr = np.unique(x[:, i])
        for value in arr:
            attributes.append((i, value))

    weight = np.ones((rows, 1), dtype=int)
    alpha_i = 1
    for i in range(0, numtrees):
        radIndexes = randomFunction(x)
        tree = id3(x[radIndexes], y[radIndexes], attributes, maxdepth, weight)
        h_i[i] = (alpha_i, tree)
    return h_i

def predict_boosting_example(x, h_ens):
    """
    For prediciting exampls with boosting alogirthm where we multiply the precition with respecte to the alpha and normalize it with
    total alpha vlaues
    Returns the predicted label of x according to tree
    """
    y_preict = []
    total_alpha = 0

    for numHypo in h_ens:
        alpha, tree = h_ens[numHypo]
        y = predict_label(x, tree)
        y_preict.append(y*alpha)
        total_alpha += alpha
    predictValue = np.sum(y_preict) / total_alpha

    if(predictValue >= 0.5):
        return 1
    else:
        return 0

def boosting(x, y, max_depth,num_stumps):
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
        tree = id3(x, y, attributes, max_depth, weight)

        trn_pred = [predict_label(x[i, :], tree) for i in range(rows)]
        temp = 0
        for i in range(rows):
            if(trn_pred[i] != y[i]):
                temp += weight[i]
        err = (1/(np.sum(weight))) * temp
        alpha_i = 0.5 * np.log((1-err)/err)
        h_ens[stump] = (alpha_i, tree)
    return h_ens

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
    _scikit = False
    
    tree_depth = np.array([3])
    stump_depth = np.array([2])
    ens_size = np.array([5])

    if (_self):
        print('Self-implementation')
        if(_bagging):
            print('Bagging')
            for n in ens_size:
                for d in tree_depth:
                    print('Size {1} Depth {0}'.format(d,n))
                    modelList = bagging(Xtrn, ytrn, d, n)
                    self_y_pred = [predict_example_bagging(Xtst[i, :], modelList) for i in range(Xtst.shape[0])]
                    # plot_confusion_matrix(ytst, self_y_pred, normalize=False, title='self_bagging_depth{0}_size{1}_cm'.format(d,n))
                    tn, fp, fn, tp = confusion_matrix(ytst, self_y_pred).ravel()
                    print("(tn, fp, fn, tp) = ", (tn, fp, fn, tp))
                    
        if(_boosting):
            print('Boosting')
            for n in ens_size:
                for d in stump_depth:
                    print('Size {1} Depth {0}'.format(d,n)) 
                    model = boosting(Xtrn, ytrn, d, n)
                    self_y_pred = [predict_example_boosting(Xtst[i, :], model) for i in range(Xtst.shape[0])]
                    # plot_confusion_matrix(ytst, self_y_pred, normalize=False, title='self_boosting_depth{0}_size{1}_cm'.format(d,n))
                    tn, fp, fn, tp = confusion_matrix(ytst, self_y_pred).ravel()
                    print("(tn, fp, fn, tp) = ", (tn, fp, fn, tp))
                    
                
    if (_scikit):
        print('Scikit implementation') 
        if(_bagging):
            print('Bagging')
            for n in ens_size:
                for d in tree_depth:     
                    print('Size {1} Depth {0}'.format(d,n))
                    dtree = DecisionTreeClassifier(criterion='entropy', max_depth=d)
                    modelList = BaggingClassifier(base_estimator=dtree, n_estimators=n, bootstrap=True)
                    modelList.fit(Xtrn, ytrn)
                    self_y_pred = modelList.predict(Xtst)
                    # plot_confusion_matrix(ytst, self_y_pred, normalize=False, title='sklearn_bagging_depth{0}_size{1}_cm'.format(d,n))
                    tn, fp, fn, tp = confusion_matrix(ytst, self_y_pred).ravel()
                    print("(tn, fp, fn, tp) = ", (tn, fp, fn, tp))
                    
        if(_boosting):
            print('Boosting')
            for n in ens_size:
                for d in stump_depth:
                    print('Size {1} Depth {0}'.format(d,n))
                    dtree = DecisionTreeClassifier(criterion='entropy', max_depth=d)
                    model = AdaBoostClassifier(base_estimator=dtree, n_estimators=n)
                    model.fit(Xtrn, ytrn)
                    self_y_pred = model.predict(Xtst)
                    # plot_confusion_matrix(ytst, self_y_pred, normalize=False, title='sklearn_boosting_depth{0}_size{1}_cm'.format(d,n))
                    tn, fp, fn, tp = confusion_matrix(ytst, self_y_pred).ravel()
                    print("(tn, fp, fn, tp) = ", (tn, fp, fn, tp))
                
