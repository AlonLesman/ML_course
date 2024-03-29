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
    return goodness, groups

class DecisionNode:

    def __init__(self, data, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
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
        # create {feature_index : goodness} dictionary
        features_goodness = {index : goodness_of_split(self.data,index,impurity_func)[0] for index in range(self.data.shape[1]-1)}
        # extract the best feature and update the relevent field
        self.feature = max(features_goodness, key=lambda index : features_goodness[index])           
        groups = goodness_of_split(self.data, self.feature, impurity_func)[1]
        self.chi = self.chi_square(groups)
        # if the split doesnt signficiant by chi test, set this node as terminal 
        if (not self.chi_square_test(groups,self.chi) or len(groups) <= 1):
            self.terminal = True
            return        
        for key,value in groups.items():
            child_node = DecisionNode(value)
            self.add_child(child_node,key)
            child_node.terminal = (impurity_func(value) == 0)
    
    def chi_square(self, groups):
        """
        Calculate the chi-squared statistic for a given node's split.

        Input: groups: {feature_value : data_subset} dictionary
        
        Output: chi_squared_statistic(int)
        """
        total_instances = len(self.data)
        labels_count = labels_counter(self.data)
        chi_square_statistic = 0

        for group in groups.values():
            subset_size = len(group)
            subset_labels_count = labels_counter(group)
            for label, count in labels_count.items():
                expected_count = subset_size * count / total_instances
                observed_count = subset_labels_count.get(label, 0)
                chi_square_statistic += ((observed_count - expected_count) ** 2) / expected_count
        
        return chi_square_statistic
    
    def chi_square_test(self, groups, chi):
        """
        Return True if the chi-squared test for the node's split is significant.

        Input:
        - node: the current node being evaluated.
        - groups: a dictionary holding the data after splitting according to the feature values.
        - chi: the chi-squared significance threshold.

        Output: True if the chi-squared test is significant, False otherwise.
        """
        # If chi is set to 1, the chi-squared test is always significant
        if chi == 1:
            return True
        # Calculate the chi-squared statistic for the current split
        chi_squared_statistic = self.chi_square(groups)
        degrees_of_freedom = len(groups) - 1
        # Look up the threshold chi-squared value based on the degrees of freedom and significance level
        threshold = chi_table.get(degrees_of_freedom, {}).get(chi, 100000)
        # If the chi-squared statistic is greater than or equal to the threshold value, the test is significant
        return chi_squared_statistic >= threshold
    
    
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
    print("new tree created")  
    root = DecisionNode(data)  
    # update node terminal attribute
    root.terminal = (impurity(data) == 0)
    built_tree_helper(root,impurity,gain_ratio,max_depth) 
    return root

def built_tree_helper(node, impurity, gain_ratio, max_depth):
    if node.terminal or max_depth == 0:
        return
    node.split(impurity)
    for child in node.children:
        built_tree_helper(child, impurity, gain_ratio, max_depth-1)        

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
    while not root.terminal and root.children != []:
        for child in root.children:
            if child.data[(0,root.feature)] == instance[root.feature]:
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
    best_impurity_function = calc_entropy
    gain_ratio = False
    for max_depth in range(1,10):
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

def labels_counter(data):
        """
        create a {label : counter of label} dictionary for a given data

        Input: data where the last column holds the labels

        Output: dictionary of {label : counter of label}
        """
        labels_count = {}
        for label in data[-1]:
            if label not in labels_count:
                labels_count[label] = 1
            else:
                labels_count[label] += 1

        return labels_count







