import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from random import randrange

import copy 

def partition(x):

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

def normalize(w):

    tot = sum(w)
    w = [x / tot for x in w]
    return w

def predict_example_bagging(x, h_ens):

    resultList = list()   
    for tree in h_ens:
        prediction = predict_example_utility(x, tree[1])
        resultList.append(prediction)

    counts = np.bincount(resultList)
    if len(counts) == 0:
        return 1
    else:
        return np.argmax(counts)
    
def predict_example_boosting(x, h_ens):
    
    alpha = normalize(h_ens[0])
    h = h_ens[1]
    
    total = 0
    for i in range(len(h)):
        prediction = predict_example_utility(x, h[i])
        total = total + (alpha[i] * prediction)

    if total > 0.5 :
        return 1
    else:
        return 0
    
def majority_label(_map):
    
    max_count = -1
    major_label = None
    for label in _map:
        e = sum(_map[label])
        if e > max_count:
            max_count = e
            major_label = label
            
    return major_label
        
def predict_example_utility(x, tree):

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
                
