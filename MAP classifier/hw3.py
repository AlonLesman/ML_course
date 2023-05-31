import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0,
            (0, 1): 0.3,
            (1, 0): 0.3,
            (1, 1): 0.4
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0,
            (0, 1): 0.3,
            (1, 0): 0.5,
            (1, 1): 0.2
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0,
            (0, 1): 0.3,
            (1, 0): 0.5,
            (1, 1): 0.2
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0,
            (0, 0, 1): 0.18,
            (0, 1, 0): 0.0,
            (0, 1, 1): 0.12,
            (1, 0, 0): 0,
            (1, 0, 1): 0.12,
            (1, 1, 0): 0.5,
            (1, 1, 1): 0.08,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y

        # Calculate P(X, Y) for each possible value of X and Y
        for x in X:
            for y in Y:
                if not np.isclose(X_Y[(x, y)], X[x]*Y[y]):
                    return True  
        return False
        
            
    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        for c in C:
            for x in X:
                for y in Y:
                    if not np.isclose(X_Y_C[(x, y, c)], (X_C[(x,c)]*Y_C[(y,c)])/C[c]):
                        return False
        return True


def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = np.log((rate**k * np.exp(1)**-rate)/np.math.factorial(k))
    return log_p

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = np.array([sum(poisson_log_pmf(sample, rate) for sample in samples) for rate in rates])

    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates) # might help
    max_index_likelihoods = np.argmax(likelihoods)
    rate = rates[max_index_likelihoods]

    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = np.mean(samples)
    
    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """

    p = (np.exp(-((x - mean)**2) / (2 * std**2))) / (np.sqrt(2 * np.pi) * std)
    return np.prod(p, axis=0)
    

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        class_rows = dataset[dataset[:, -1] == class_value, :-1]
        
        # calculate mean and std for each feature
        self.mean = np.mean(class_rows, axis=0)
        self.std = np.std(class_rows, axis=0)
        self.dataset = dataset
        self.class_value = class_value
            

    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        num_samples = self.dataset.shape[0]
        class_count = np.count_nonzero(self.dataset[:, -1] == self.class_value)
        prior = class_count / num_samples
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        
        likelihood = normal_pdf(x, self.mean, self.std)
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        return posterior

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pp_0 = self.ccd0.get_instance_posterior(x)
        pp_1 = self.ccd1.get_instance_posterior(x)
        pred = int(pp_1 > pp_0)
        
        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    
    count = 0
    for x in test_set:
        if(map_classifier.predict(x[0:len(x)-1]) == x[len(x)-1]):
            count+=1
    acc = count / test_set.shape[0]

    return acc

    
    

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    d = len(mean)
    x_minus_mean = np.subtract(x, mean)
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)

    coef = 1 / ((2 * np.pi) ** (d / 2) * (cov_det ** 0.5))
    exponent = -0.5 * np.dot(x_minus_mean.T, np.dot(cov_inv, x_minus_mean))

    pdf = coef * np.exp(exponent)
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        class_rows = dataset[dataset[:, -1] == class_value, :-1]
        
        # calculate mean and std for each feature
        self.mean = np.mean(class_rows, axis=0)
        self.cov = np.cov(class_rows, rowvar=False)
        self.dataset = dataset
        self.class_value = class_value

        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """

        num_samples = self.dataset.shape[0]
        class_count = np.count_nonzero(self.dataset[:, -1] == self.class_value)
        prior = class_count / num_samples
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = multi_normal_pdf(x, self.mean, self.cov)
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        return posterior

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pp_0 = self.ccd0.get_prior()
        pp_1 = self.ccd1.get_prior()
        pred = int(pp_1 > pp_0)
        
        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pp_0 = self.ccd0.get_instance_likelihood(x)
        pp_1 = self.ccd1.get_instance_likelihood(x)
        pred = int(pp_1 > pp_0)
        
        
        return pred
        return pred

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

def get_epsilon_vector(vector,is_in_data):
    epsilon_vector = []
    i = 0
    for val in vector:
        if is_in_data[i] == 0:
            epsilon_vector = np.append(epsilon_vector,EPSILLON)
        else:
            epsilon_vector = np.append(epsilon_vector,val)
        i+=1 
    return epsilon_vector


class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        sub_data=dataset[(dataset[:, -1] == class_value), :]
        self.sub_data=sub_data[:, :-1]
        self.class_value = class_value
        self.dataset= dataset
        self.n_i = sub_data.shape[0]
        self.v_j =  np.array([len(np.unique(self.sub_data[:, col])) for col in range(self.sub_data.shape[1])])
    
    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        sub_data=self.dataset[(self.dataset[:, -1] == self.class_value), :]
        prior = sub_data.shape[0]/self.dataset.shape[0]
        return prior
    
    def get_n_i_j(self, x):
        # Initialize an empty NumPy array to store the counts
        counts = np.array([])
        # Iterate over the values in the vector
        trans_data = np.transpose(self.sub_data)
        for i in range(len(x)):
            # Count the occurrences of the value in the corresponding column
            count = np.sum(x[i] == trans_data[i])
            # Append the count to the NumPy array
            counts = np.append(counts, count)
        return counts

    def get_is_in_data(self,x):
        # Initialize an empty list to store the counts
        counts = np.array([])
        # Iterate over the values in the vector
        trans_data= np.transpose(self.dataset)
        for i in range(len(x)):
            count = np.sum(x[i]==trans_data[i])
            counts= np.append(counts,count)
        return counts
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        v_j = self.v_j
        n_i_j = self.get_n_i_j(x)
        n_i = self.n_i
        is_in_data= self.get_is_in_data(x)
        laplace_likelihood_vector = (n_i_j+1)/(n_i+v_j)
        epsilon_vector = get_epsilon_vector(laplace_likelihood_vector,self.get_is_in_data(x))
        likelihood = np.prod(epsilon_vector)
        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_prior() * self.get_instance_likelihood(x)
        return posterior



class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """

        pp_0 = self.ccd0.get_instance_posterior(x)
        pp_1 = self.ccd1.get_instance_posterior(x)
        pred = int(pp_1 > pp_0)
        
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        correct_count = 0
        row = 0
        class_index = test_set.shape[1] - 1 
        for x in test_set[:, :-1]:
            if self.predict(x) == test_set[row][class_index]:
                correct_count += 1
            row += 1
        return correct_count / row


