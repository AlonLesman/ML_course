import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 1.0
    # collect and count the data labels
    labels = {}
    for instance in data:
        label = instance[-1]
        if label in labels:
            labels[label] += 1
        else:
            labels[label] = 1
    num_of_instances = len(data)
    for label in labels:
        label_frequency = labels[label] / num_of_instances
        gini -= label_frequency ** 2
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    # collect and count the data labels
    labels = {}
    for instance in data:
        label = instance[-1]
        if label in labels:
            labels[label] += 1
        else:
            labels[label] = 1
    num_of_instances = len(data)
    for label in labels:
        label_frequency = labels[label] / num_of_instances
        entropy -= label_frequency * np.log(label_frequency)
    return entropy

def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    goodness = 0
    groups = {} # groups[feature_value] = data_subset
    source_impurity = impurity_func(data)
    # create partition to groups by the given feature
    for feature_value in np.unique(data[:,feature]):
         data_subset = data [data[:,feature] == feature_value]
         groups[feature_value] = data_subset
    
    groups_impurity = sum([impurity_func(groups[feature]) * len(groups[feature]) for feature in groups])
    goodness = source_impurity - groups_impurity
    if(gain_ratio):
        # must check info gain
        split_info = sum([-(len(groups[feature])/len(data)) * np.log2(len(groups[feature])/len(data)) for feature in groups])
        if split_info == 0:
            return 0, groups
    
    print("goodness has ended!")
    return goodness, groups
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     goodness=0
#     groups = {}# groups [feature_value] = data_subset
#     for feature_value in np. unique (data[:, feature]):
#         data_subset = data[data[:, feature] == feature_value]
#         groups[feature_value] = data_subset
# # Calculate the initial impurity of the whole dataset
#         phi_of_s = impurity_func (data)
# # Calculate the weighted average of the impurity of each subset 
#         weighted_avg_split = sum([len (groups [feature_value]) / len(data) * impurity_func(data)])
# # Calculate the goodness of split value
#         goodness = phi_of_s - weighted_avg_split
# # Calculate the split information value
#         split_info = sum([-(len(groups[feature])/len(data)) * np.log2(len(groups[feature])/len(data)) for feature in groups])
# # Calculate the gain ratio, if specified
#         if gain_ratio and split_info != 0:
#             goodness = goodness / split_info
#     return goodness, groups
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class DecisionNode:

    def __init__(self, data, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        print("new desicion node created !")
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio 
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        labels = [row[-1] for row in self.data]
        label_counts = {label: labels.count(label) for label in labels}
        return max(label_counts,key=lambda label : label_counts[label])
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
     
    def split(self, impurity_func):

        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.
        
        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        # 1-extracts the best feature
        # 2-split the data into childs bast on the first step
        # 2.1-set the feature attribute to be the best feature
        # 2.2-for each group in groups create new child and add it to self_child
        # 2.3-for each new child determind if is it terminal,namly leaf
        print("split has begun")
        # create {feature : goodness} dictionary
        features_goodness = {index : goodness_of_split(self.data,index,impurity_func)[0] for index in range(self.data.shape[1]-1)}
        # extract the best feature 
        max_goodness_feature = max(features_goodness, key=lambda index : features_goodness[index])
        # self.feature = self.data.loc[:,max_goodness_feature]
        self.feature = np.where(self.data == max_goodness_feature)[0] 
        print("the best feature for split is: " + str(max_goodness_feature))    
        self.children_values = list(goodness_of_split(self.data, self.feature, impurity_func)[1].values())
        # err - dict/list
        # for child in children:
        #     # child_node = DecisionNode(children[child])
        #     # child_node = build_tree(children[child])
        #     # children[child].pop()
        print("split has ended")

            

    

def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = DecisionNode(data)  
    # update node terminal attribute
   
    root.terminal = (impurity(data) == 0)  
    # stop condition
    if root.terminal or max_depth == 0:
        print("terminal node")
        return
    root.split(impurity)
    for child in root.children_values:
        # create an object of decision node and add it to children
        child_node = build_tree(root.children_values[child],impurity,max_depth-1)
        child_value = root.children_values[child]
        root.add_child(child_node,child_value) 
        print("add child !")   
    return root

def predict(root, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    pred = None
    while (root.children is not []):
        for child in root.children:
            if child.data[(0,root.feature)] == instance [root.feature]:
                root = child
    return root.pred

def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    correct_prediction = 0
    for instance in dataset:
        if instance[-1] == predict(node,instance):
            correct_prediction += 1
    accuracy = (correct_prediction / len(dataset) * 100)
    return accuracy

def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output: the training and testing accuracies per max depth
    """
    training = []
    testing  = []
    # to be update
    best_impurity_function = calc_entropy() 
    gain_ratio = False
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        root = build_tree(X_train, best_impurity_function, gain_ratio, max_depth=max_depth)
        training.append(calc_accuracy(root, X_train))
        testing.append(calc_accuracy(root, X_test))

    return training, testing


def chi_pruning(X_train, X_test):

    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc  = []
    depth = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return chi_training_acc, chi_testing_acc, depth

def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of nodes in the tree.
    """
    n_nodes = 0
    if node.children == []:
        return 1
    for child in node.children:
        n_nodes += count_nodes(child)
    return 1 + n_nodes







