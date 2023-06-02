import numpy as np

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state) 
        X = np.c_[np.ones((X.shape[0], 1)), X]
        # initialize random theta 
        self.theta = np.random.randn(X.shape[1])
        self.thetas.append(self.theta)
        # cost fuction that represent the liklyhood
        cost = self.cost(X, y)
        # gradient decsent loop
        for _ in range(self.n_iter):
            # compute gradient vector
            gradient = np.dot((self.sigmoid(X) - y), X)
            # update theta
            self.theta -= self.eta * gradient
            # recalculate the cost fuction
            cost = self.cost(X, y)
            # tracks costs and thetas
            self.Js.append(cost)
            self.thetas.append(self.theta)
            # stop condition- check if the improvement is smaller then epsilon
            if len(self.Js) < 2:
                continue
            improvement = abs(self.Js[-2] - self.Js[-1])
            if(improvement < self.eps):
                break
    
    def cost(self, X, y):
        """TC"""   
        h_theta = self.sigmoid(X)
        total_cost = (-y * np.log(h_theta) - (1 - y) * np.log(1 - h_theta)).mean()
        return total_cost
                         
    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        X = np.c_[np.ones((X.shape[0], 1)), X]
        # compute S(h(x)) for each instance in X
        h_theta = self.sigmoid(X)
        # for each instance, if h_thete bigger then 0.5 predict class 1 else predist 0
        preds = np.where(h_theta >= 0.5, 1, 0)
        return preds
       
    
    def sigmoid(self, X):
        """TC"""
        theta_x = np.dot(X, self.theta)
        h_theta = 1 / (1 + np.exp(-theta_x))
        return h_theta
  

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = 0
    # set random seed
    np.random.seed(random_state)
    # shuffle the data that concatenatedwith the labels
    XandY = np.append(X, y.reshape(-1, 1), axis=1)
    np.random.shuffle(XandY)
    # splits the data into folds
    test_size = X.shape[0] // folds
    for i in range(folds):
        t1, test, t2 = np.vsplit(XandY, [i * test_size, (i+1) * test_size])
        train = np.concatenate((t1,t2),axis=0)
        algo.fit(train[:,:-1], train[:,-1])
        prediction = algo.predict(test[:,:-1])
        cv_accuracy += np.sum(prediction == test[:,-1]) / len(prediction)
    cv_accuracy = cv_accuracy / folds
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = np.exp(-((data-mu)**2) / (2*(sigma**2))) / np.sqrt(2*np.pi*(sigma**2))
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        self.weights = np.ones(self.k)/self.k
        self.mus = np.random.rand(self.k)
        self.sigmas = np.random.rand(self.k)
        self.costs = []
     
   
    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        self.responsibilities = np.zeros((self.k, len(data)))
        # calculate and update resposibilities for each cluster/gaussian
        for k in range(self.k):
            self.responsibilities[k] = self.weights[k] * (norm_pdf(data, self.mus[k], self.sigmas[k]).flatten()) 
        # normalizing responsibilities 
        self.responsibilities /= self.responsibilities.sum(0)

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        resposibility_sum = self.responsibilities.sum(0)
        
        self.weights = (1 / len(data)) * resposibility_sum

        for k in range(self.k):
            weights_N = self.weights[k] * len(data)
            self.mus[k] = sum(data[i] * self.responsibilities[k][i] for i in range(len(data))) / weights_N
            self.sigmas[k] = np.sqrt(sum(self.responsibilities[k][i] * (data[i]-self.mus[k])**2 for i in range(len(data))) / weights_N)             

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)
        for i in range(self.n_iter):
            self.expectation(data)
            self.maximization(data)
            self.costs.append(self.cost(data))
            if len(self.costs) >= 2 and abs(self.costs[-1] - self.costs[-2]) < self.eps:
                break
    
    def cost(self, data):
        """TC"""
        cost = 0
        for x in data:
            likelihood = sum(self.weights[k] * norm_pdf(x, self.mus[k], self.sigmas[k]) for k in range(self.k))
            cost += -np.log2(likelihood)
        return cost
    
    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = sum (weights[i] * norm_pdf(data,  mus[i], sigmas[i]) for i in range(len(weights)))
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        # Determine unique class labels
        self.classes = np.unique(y)
        self.priors = []
        self.parameters = []
        
        for i, label in enumerate(self.classes):
            # creates subset of instance for the current class 
            class_subset = X[y == label]
            # set prior liklihood for the current class
            self.priors.append(class_subset.shape[0] / X.shape[0])
            self.parameters.append([])
            for feature in class_subset.T:
                em = EM(k=self.k, random_state=self.random_state)
                em.fit(feature)
                weight, mu, sigma = em.get_dist_params()
                self.parameters[i].append((weight, mu, sigma)) 



    def predict(self, X):
        """
        Return the predicted class labels for each given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []
        for instance in X:
            posteriors = []
            for class_i,_ in enumerate(self.classes):
                # prior liklihood
                class_likelihood = self.priors[class_i]
                # features responsibilities
                for feature in instance:
                    weight, mu, sigma = self.parameters[class_i][int(feature)]
                    # compute respon. with the parameter we found in the EM 
                    resposibility = gmm_pdf(instance,weight,mu,sigma)
                    # multiply each resposibility, assuming iid
                    class_likelihood *= resposibility 
                
                posteriors.append(class_likelihood) 
            preds.append(self.classes[np.argmax(posteriors)])                  

        return preds

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }