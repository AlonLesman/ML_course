###### Your ID ######
# ID1: 208608018
# ID2: 208000588
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """

    X = (X - X.mean(axis = 0)) / (X.max(axis = 0) - X.min(axis = 0))
    y = (y - y.mean(axis = 0)) / (y.max(axis = 0) - y.min(axis = 0))

    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    
    ones = np.ones((len(X), 1))
    X = np.column_stack((ones,X))
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the cost.
    temp = np.square(np.dot(X, theta) - y)
    J = temp.mean() / 2
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    for i in range(num_iters):
        theta = theta - ((alpha * np.dot(np.dot(X, theta) - y, X)) / len(y))  
        #X has m X n dimension while thetha is n X 1. that is, computing simultaneusly all h(x)
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    pinv_theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T), y)
    
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    for i in range(num_iters):
        J_history.append(compute_cost(X,y,theta))
        if i >= 1 and J_history[i-1] - J_history[i] < 1e-8:
            break
        h = np.dot(X, theta)
        gradients = np.dot(X.T, h - y) / len(y)
        # Update theta using gradients and learning rate alpha
        theta -= alpha * gradients 
    return theta, J_history
    
def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    random_thetas = np.random.random(X_train.shape[1])
    thetas =[]
    validation_losses=[]

    # Runs on each alpha and trains it
    for alpha in alphas:
        thetas.append(efficient_gradient_descent(X_train, y_train, random_thetas, alpha, iterations)[0])
    
    # Calculate the cost function value for each parameter vector
    for theta in thetas:
        validation_losses.append(compute_cost(X_val,y_val,theta))

    alpha_dict = {a:vl for a,vl in zip(alphas, validation_losses)} #{alpha_value: validation_loss}
    return alpha_dict
    

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    
    selected_features = []
    n = X_train.shape[1]
    feature_list = [i for i in range(n)] # Indexes of features

    # Runs 5 times on each feature that is not yet in selected_features
    for i in range(5): 
        np.random.seed(42)
        random_thetas = np.random.random(size=i+2)
        min_costs={}
        for feature in feature_list:
            selected_features.append(feature)
            tempX_train = apply_bias_trick(X_train[:,selected_features]) # Extracts the training data for the selected features.
            tempX_val = apply_bias_trick(X_val[:,selected_features])
            current_theta, _ = efficient_gradient_descent(tempX_train,y_train,random_thetas,best_alpha,iterations)
            min_costs[feature] = compute_cost(tempX_val,y_val,current_theta) # Computes the cost of the validation set
            selected_features.remove(feature)

        # Adds the feature with the lowest cost to the list of selected features 
        min_index = min(min_costs,key=min_costs.get)
        selected_features.append(min_index)
        # Removes the feature from the list
        feature_list.remove(min_index)

    
    return selected_features


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    # Column names from the input dataframe
    features = df_poly.columns
    n = df_poly.shape[1]
    new_poly_cols = []

    # Runs on pairs of columns and creates new polynomial features
    for i in range(n):
        for j in range(i,n):
            name_i = features[i]  # The i-th feature
            name_j = features[j]  # The j-th feature
            
            # Creates a name for the new feature according to the names of the original features
            if i==j:
                name = name_i + "^2" 
            else:
                name = name_i + "*" + name_j
            
            # Creates the new feature column and adds it to the new polynomial features list
            new_col = df_poly[name_i] * df_poly[name_j]
            new_col.name = name
            new_poly_cols.append(new_col)
    
    # Concatenates the original dataframe with the new polynomial feature columns list
    df_poly = pd.concat([df_poly] + new_poly_cols, axis=1)
    
    return df_poly