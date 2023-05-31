import numpy as np
import matplotlib.pyplot as plt

# ***  208000588, 208608018 ***
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

def labels_counter(labels):
    """
    Returns a dictionary of label counts for a given set of labels.

    Parameters:
    labels (numpy.ndarray): An array of labels.

    Returns:
    dict: A dictionary where the keys are the unique labels and the values
          are the counts of each label in the input array.
    """
    return {label: np.sum(labels == label) for label in np.unique(labels)}


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
    labels_counter = {}
    entropy = 0.0

    # Count the frequency of each label
    for row in data:
        label = row[-1]
        if label in labels_counter:
            labels_counter[label] += 1
        else:
            labels_counter[label] = 1

    # Calculate the entropy using the formula
    for label in labels_counter:
        prob = float(labels_counter[label]) / len(data)
        entropy -= prob * np.log2(prob)

    return entropy

def split_information(data, groups):
    """
    Calculate the split information for a given split of a dataset.

    Input:
    - data: any dataset.
    - groups: a dictionary of group labels and the corresponding subset of the data.

    Returns:
    - split_info: The split information value.
    """
    data_subsets = list(groups.values())
    size = len(data)
    # Calculate the split information value using the formula:
    # - sum[(|S_a|/|S|) * log2(|S_a|/|S|) for S_a in data_subsets]
    split_info = - sum((len(Sa)/size * np.log2(len(Sa)/size)) for Sa in data_subsets)
    return split_info


    
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
    groups = {}# groups[feature_value] = data_subset
   
    for feature_value in np.unique(data[:, feature]):
        data_subset = data[data[:, feature] == feature_value]
        groups[feature_value] = data_subset
        # Calculate the initial impurity of the whole dataset
        phi_of_s = impurity_func(data)
        # Calculate the weighted average of the impurity of each subset
        weighted_avg_split = sum([len(groups[feature_value]) / len(data) * impurity_func(groups[feature_value]) for feature_value in groups])
        # Calculate the goodness of split value
        goodness = phi_of_s - weighted_avg_split
        # Calculate the split information value
        split_info = split_information(data, groups)
        # Calculate the gain ratio, if specified
        if gain_ratio and split_info != 0:
            goodness = goodness / split_info

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
        pred_dictionary = labels_counter(self.data[:,-1])       
        pred = max(pred_dictionary, key=pred_dictionary.get)
        return pred


        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
     
    def chi_squared(self, groups):
        """
        Calculate the chi-squared statistic for the node's split.

        Args:
        - groups: dictionary where keys are feature values and values are subsets of the data that have that feature value
        
        Returns:
        - chi_squared_statistic: the chi-squared statistic for the node's split
        """

        num_of_instances = len(self.data)
        num_of_labels = labels_counter(self.data[:, -1])
        chi_squared_statistic = 0

        for _, subset in groups.items():
            subset_num_of_instances = len(subset)
            subset_num_of_labels = labels_counter(subset[:, -1])

            for label, amount in num_of_labels.items():   
                X_count = subset_num_of_instances * (amount / num_of_instances)
                Y_count = subset_num_of_labels.get(label, 0)
                chi_squared_statistic += ((Y_count - X_count) ** 2) / X_count
        
        return chi_squared_statistic

    
    def split(self, impurity_func):

        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        groups = {}
        max_score = 0.0
        # extract the feature index of the feature with the best goodness of split score
        for feature_index in range(self.data.shape[1]-1):  
            goodness_score, temp_groups = goodness_of_split(self.data, feature_index, impurity_func, self.gain_ratio)
            if goodness_score > max_score:
                max_score = goodness_score
                groups = temp_groups
                self.feature = feature_index
        # check the stop conditions
        if (self.depth < self.max_depth) and (len(groups) > 1) and (chi_squared_test(self, groups, self.chi)):
            for key, node_data in groups.items():
                new_node = DecisionNode(node_data, depth=self.depth+1, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
                self.add_child(new_node, key)
        else:
            self.terminal = True
            
    def is_terminal(self):
        """
        Check if the node's data contains more than one label, indicating it is incomplete.

        Returns:
        - bool: True if the node's data contains more than one label, False otherwise.
        """
        return (len(labels_counter(self.data[:,-1])) > 1)
        
    

       

def chi_squared_test(node, groups, chi):
    """
    Return True if the chi-squared test for the node's split is significant.

    Input:
    - node: the current node being evaluated.
    - groups: a dictionary holding the data after splitting according to the feature values.
    - chi: the chi-squared significance threshold.

    Output: True if the chi-squared test is significant, False otherwise.
    """
    if chi == 1:
        return True
    chi_squared_statistic = node.chi_squared(groups)
    degrees_of_freedom = len(groups) - 1
    threshold = chi_table.get(degrees_of_freedom, {}).get(chi, 100000)
    return chi_squared_statistic >= threshold

        
def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a for  using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = DecisionNode(data, chi=chi, max_depth=max_depth, gain_ratio=gain_ratio)
    node_queue = []
    if root.is_terminal():
        node_queue.append(root)
    while len(node_queue) > 0:
        current_node = node_queue.pop(0)
        current_node.split(impurity)
        for node in current_node.children:
            if not node.is_terminal():
                node.terminal = True
            if node.is_terminal():
                node_queue.append(node)
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
    current = root
    # Traverse the decision tree until a terminal node is reached
    while not current.terminal:
        feature_value = instance[current.feature]
        if feature_value not in current.children_values:
            return current.pred
        child_index = current.children_values.index(feature_value)
        current = current.children[child_index]
    pred = current.pred
    return pred

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
    accuracy = (correct_prediction / len(dataset))
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

    # Iterate over each maximum depth value and build a decision tree with that depth
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        current_tree_root = build_tree(X_train, impurity=calc_entropy, gain_ratio=True, max_depth = max_depth)
        # Calculate the accuracy of the decision tree on the training and testing datasets
        training.append(calc_accuracy(current_tree_root, X_train))
        testing.append(calc_accuracy(current_tree_root, X_test))
    return training, testing



def get_depth(node):
    """
    Recursively get the maximum depth of a decision tree starting from a given node.

    Input:
    - node: the starting node of the tree.

    Output: the maximum depth of the tree.
    """
    # If the current node has no children, its depth is 1
    if not node.children:
        return 0
    # Otherwise, recursively get the maximum depth of each child and return the maximum depth + 1
    return 1 + max(get_depth(child) for child in node.children)


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
     
    # Define the chi values to try
    chi_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]
    chi_training_acc = []
    chi_testing_acc = []
    depths = []

    # best impurity function and gain_ratio flag here
    impurity_func = calc_entropy
    gain_ratio = True

    # For each chi value, build a decision tree and calculate the training and testing accuracies
    for chi in chi_values:
        tree = build_tree(X_train, impurity_func, gain_ratio=gain_ratio, chi=chi)
        train_acc = calc_accuracy(tree, X_train)
        test_acc = calc_accuracy(tree, X_test)
        depth = get_depth(tree)

        # Append the accuracies and depth to their respective lists
        chi_training_acc.append(train_acc)
        chi_testing_acc.append(test_acc)
        depths.append(depth)

    return chi_training_acc, chi_testing_acc, depths

def count_nodes(node):
    """
    Count the number of nodes in a given decision tree.

    Input:
    - node: a node in the decision tree.

    Output: the number of nodes in the tree.
    """
    if node is None or node.terminal:
        return 1
    counter = 1
    # Recursively count the number of nodes 
    for child in node.children:
        counter += count_nodes(child)
    return counter




        


