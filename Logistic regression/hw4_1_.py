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
        # Add a column of ones to X for the bias term
        X = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.random.randn(X.shape[1])
        self.thetas.append(self.theta)  # initialize theta
        for _ in range(self.n_iter):
            # calculate the gradient
            z = np.dot(X, self.theta)
            h = 1 / (1 + np.exp(-z))
            gradient = np.dot(X.T, (h - y))
            self.theta -= self.eta * gradient
            self.thetas.append(self.theta)

            # Compute cost
            cost = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
            self.Js.append(cost)

            if len(self.Js) > 1 and abs(self.Js[-2] - self.Js[-1]) < self.eps:
                break

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        X = np.c_[np.ones((X.shape[0], 1)), X]
        z = np.dot(X, self.theta)
        # Apply the sigmoid function
        values = 1 / (1 + np.exp(-z))
        # preds = [1 if value > 0.5 else 0 for value in values]
        preds = np.where(values >= 0.5, 1, 0)
        return preds


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

    cv_accuracy = None
    sum_acc = 0
    # set random seed
    np.random.seed(random_state)
    X_y = np.append(X, y.reshape(-1, 1), axis=1)
    np.random.shuffle(X_y)

    # create folds
    test_size = len(y) // folds
    for i in range(folds):
        p1, test, p2 = np.vsplit(X_y, [i * test_size, (i+1) * test_size])
        train = np.concatenate((p1, p2), axis=0)
        algo.fit(train[:, :-1], train[:, -1])
        preds = algo.predict(test[:, :-1])
        accuracy = np.sum(preds == test[:, -1]) / len(preds)
        sum_acc += accuracy

    cv_accuracy = sum_acc/folds
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

# def multi_normal_pdf(x, mean, cov):
#     """
#     Calculate multi variable normal desnity function for a given x, mean and covarince matrix.

#     Input:
#     - x: A value we want to compute the distribution for.
#     - mean: The mean vector of the distribution.
#     - cov:  The covariance matrix of the distribution.

#     Returns the normal distribution pdf according to the given mean and var for the given x.
#     """
#     det = np.linalg.det(cov)
#     inv = np.linalg.inv(cov)
#     exp = np.exp((-0.5) * np.dot((x-mean).T, np.dot(inv, (x-mean))))
#     denominator = np.sqrt((2 * np.pi) ** len(mean) * det)
#     pdf = exp / denominator
#     return pdf


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
        self.costs = []

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        # self.weights = np.ones(self.k) / self.k
        # self.mus = data[np.random.choice(data.shape[0], self.k, replace=False)]
        # self.sigmas = np.random.random_sample(self.k
        # initialize weights to be all equal
        self.weights = np.ones(self.k) / self.k
        # initialize mus with random numbers
        self.mus = np.random.rand(self.k)
        # initialize sigmas with random numbers
        self.sigmas = np.random.rand(self.k)

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        self.responsibilities = np.zeros((self.k, len(data)))
        for k in range(self.k):
            self.responsibilities[k] = self.weights[k] * \
                (norm_pdf(data, self.mus[k], self.sigmas[k])).flatten()

        self.responsibilities /= self.responsibilities.sum(0)

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        self.weights = self.responsibilities.sum(1)/len(data)
        for k in range(self.k):
            new_weight_N = self.weights[k]*len(data)
            self.mus[k] = sum(data[i] * self.responsibilities[k][i]
                              for i in range(len(data))) / new_weight_N
            self.sigmas[k] = np.sqrt(sum(self.responsibilities[k][i] * (data[i]-self.mus[k])**2
                                         for i in range(len(data))) / new_weight_N)

    def compute_cost(self, data):
        cost = 0
        for x in data:
            likelihood = sum(self.weights[k] * norm_pdf(x, self.mus[k],
                                                        self.sigmas[k]) for k in range(self.k))
            cost += -np.log2(likelihood)
        return cost

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
        for _ in range(self.n_iter):
            self.expectation(data)
            self.maximization(data)
            cost = self.compute_cost(data)
            self.costs.append(cost)
            if len(self.costs) > 1 and np.abs(self.costs[-1] - self.costs[-2]) < self.eps:
                break

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
    pdf = sum(weights[i] * norm_pdf(data,  mus[i], sigmas[i])
              for i in range(len(weights)))
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
        self.prior = {}
        self.gmm = {}

    # def fit(self, X, y):
    #     """
    #     Fit training data.

    #     Parameters
    #     ----------
    #     X : array-like, shape = [n_examples, n_features]
    #       Training vectors, where n_examples is the number of examples and
    #       n_features is the number of features.
    #     y : array-like, shape = [n_examples]
    #       Target values.
    #     """
    #     # Calculate the prior probabilities
    #     unique_labels, counts = np.unique(y, return_counts=True)
    #     self.prior = dict(zip(unique_labels, counts / len(y)))

    #     # Calculate the GMM for each class
    #     for label in unique_labels:
    #         class_data = X[y == label]
    #         self.gmm[label] = EM(k=self.k, random_state=self.random_state)
    #         self.gmm[label].fit(class_data)

    # def predict(self, X):
    #     """
    #     Return the predicted class labels for a given instance.
    #     Parameters
    #     ----------
    #     X : {array-like}, shape = [n_examples, n_features]
    #     """
    #     # Store the final predictions
    #     preds = []

    #     # For each instance in X, calculate the posterior probabilities and choose the class with the highest probability
    #     for x in X:
    #         posteriors = {}
    #         for label in self.gmm:
    #             likelihood = gmm_pdf(x, *self.gmm[label].get_dist_params())
    #             posteriors[label] = self.prior[label] * likelihood
    #         preds.append(max(posteriors, key=posteriors.get))

    #     return np.array(preds)
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

        for i, c in enumerate(self.classes):
            X_c = X[y == c]

            self.priors.append(X_c.shape[0] / X.shape[0])
            self.parameters.append([])

            # For each feature calculate mean and variance for each class
            for column in X_c.T:
                em = EM(k=self.k, random_state=self.random_state)
                em.fit(column)
                w, m, s = em.get_dist_params()
                self.parameters[i].append((w, m, s))

    def predict(self, X):
        """
        Return the predicted class labels for each given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []

        for x in X:

            posteriors = []

            for class_i, c in enumerate(self.classes):

                class_likelihood = 1

                for column_i, column in enumerate(x):
                    w, m, s = self.parameters[class_i][column_i]
                    gmm = gmm_pdf(column, w, m, s)
                    class_likelihood *= gmm

                posteriors.append(self.priors[class_i] * class_likelihood)

            preds.append(self.classes[np.argmax(posteriors)])

        return preds


def accuracy(y_pred, y_real):
    return np.sum(y_pred == y_real) / len(y_pred)


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

    lor = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor.fit(x_train, y_train)
    lor_train_preds = lor.predict(x_train)
    lor_train_acc = accuracy(lor_train_preds, y_train)
    lor_test_preds = lor.predict(x_test)
    lor_test_acc = accuracy(lor_test_preds, y_test)

    bayes = NaiveBayesGaussian(k=k)
    bayes.fit(x_train, y_train)
    bayes_train_preds = bayes.predict(x_train)
    bayes_train_acc = accuracy(bayes_train_preds, y_train)
    bayes_test_preds = bayes.predict(x_test)
    bayes_test_acc = accuracy(bayes_test_preds, y_test)

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

    # Dataset A: independent features, favorable for Naive Bayes
    mean1 = [0, 0, 0]
    cov1 = np.eye(3)  # Independent features

    mean2 = [5, 5, 5]
    cov2 = np.eye(3)  # Independent features

    class1_samples = multivariate_normal.rvs(mean1, cov1, size=1000)
    class2_samples = multivariate_normal.rvs(mean2, cov2, size=1000)

    dataset_a_features = np.vstack([class1_samples, class2_samples])
    dataset_a_labels = np.hstack([np.zeros(1000), np.ones(1000)])

    # Dataset B: correlated features, favorable for Logistic Regression
    mean1 = [0, 0, 0]
    cov1 = [[1, 0.8, 0.8], [0.8, 1, 0.8], [0.8, 0.8, 1]]  # Correlated features

    mean2 = [5, 5, 5]
    cov2 = [[1, 0.8, 0.8], [0.8, 1, 0.8], [0.8, 0.8, 1]]  # Correlated features

    class1_samples = multivariate_normal.rvs(mean1, cov1, size=1000)
    class2_samples = multivariate_normal.rvs(mean2, cov2, size=1000)

    dataset_b_features = np.vstack([class1_samples, class2_samples])
    dataset_b_labels = np.hstack([np.zeros(1000), np.ones(1000)])

    return {
        'dataset_a_features': dataset_a_features,
        'dataset_a_labels': dataset_a_labels,
        'dataset_b_features': dataset_b_features,
        'dataset_b_labels': dataset_b_labels
    }
