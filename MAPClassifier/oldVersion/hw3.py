import numpy as np
np.random.seed(42)

####################################################################################################
#                                            Part A
####################################################################################################


class NaiveNormalClassDistribution:
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.

        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """

        self.dataset = dataset
        self.class_value = class_value

    def get_prior(self):
        """
        Returns the prior probability of the class according to the dataset distribution.
        """
        classCounter = np.count_nonzero(self.dataset[:, -1])
        datasize = self.dataset.shape[0]
        return (classCounter / datasize) if self.class_value == 1 else ((datasize - classCounter) / datasize)

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood probability of the instance under the class according to the dataset distribution.
        """
        likelihoodProbability = 1
        # data set of only the class value we want
        datasetByClass = self.dataset[self.dataset[:, -1] == self.class_value]

        for i in range(2):
            mean = np.mean(datasetByClass[:, i])
            std = np.std(datasetByClass[:, i])
            likelihoodProbability *= normal_pdf(x[i], mean, std)

        return likelihoodProbability

    def get_instance_posterior(self, x):
        """
        Returns the posterior probability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_prior() * self.get_instance_likelihood(x)


class MultiNormalClassDistribution:

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditional multi normal
        distribution. The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """

        X = dataset[dataset[:, -1] == class_value][:, 0:-1]
        self.mean = X.mean(axis=0)
        self.cov = np.cov(X.T)
        self.lenByClass = len(X)
        self.datasetLen = len(dataset)

    def get_prior(self):
        """
        Returns the prior probability of the class according to the dataset distribution.
        """
        return self.lenByClass / self.datasetLen

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood probability of the instance under the class according to the dataset distribution.
        """
        return multi_normal_pdf(x, self.mean, self.cov)

    def get_instance_posterior(self, x):
        """
        Returns the posterior probability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_prior() * self.get_instance_likelihood(x)


def normal_pdf(x, mean, std):
    """
    Calculate normal density function for a given x, mean and standard deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * std ** 2))


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variance normal density function for a given x, mean and covariance matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """

    return float(1 / (((2 * np.pi) ** (len(mean) / 2)) * (np.linalg.det(cov) ** (1 / 2))) *
                 np.exp((-1 / 2) * ((x - mean).T.dot(np.linalg.inv(cov))).dot((x - mean))))


####################################################################################################
#                                            Part B
####################################################################################################

EPSILLON = 1e-6         # == 0.000001 It could happen that a certain value will only occur in the test set.
                        # In case such a thing occur the probability for that value will EPSILLON.


class DiscreteNBClassDistribution:
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilities for a discrete naive bayes
        distribution for a specific class. The probabilities are computed with la place smoothing.
        
        Input
        - dataset: The dataset from which to compute the probabilities (Numpy Array).
        - class_value : Compute the relevant parameters only for instances from the given class.
        """
        # X is the dataset with only the class value we want and with out the last column of spotted
        self.X = dataset[dataset[:, -1] == class_value][:, 0:-1]
        self.lenByClass = len(self.X)
        self.datasetLen = len(dataset)
        # where v_j is the number of possible values of the relevant attribute
        self.v_j = []
        for data in dataset[:, 0:-1].T:
            self.v_j.append(len(np.unique(data, return_counts=True)[1]))
        self.v_j = np.array(self.v_j)

    def get_prior(self):
        """
        Returns the prior probability of the class according to the dataset distribution.
        """
        return self.lenByClass / self.datasetLen

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood probability of the instance under the class according to the dataset distribution.
        """

        laplace = ((self.X == x).sum(axis=0) + 1) / (self.lenByClass + self.v_j)
        val_arr = []
        for data in range(len(x)):
            print(laplace)
            print(self.X[:, data])
            if laplace[data] == self.X[:, data]:
                val_arr[data] = laplace[data]
            else:
                val_arr[data] = EPSILLON
        return val_arr.prod()

    def get_instance_posterior(self, x):
        """
        Returns the posterior probability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_prior() * self.get_instance_likelihood(x)


####################################################################################################
#                                            General
####################################################################################################

class MAPClassifier:
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posterior classifier.
        This class will hold 2 class distribution, one for class 0 and one for class 1, and will predict and instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object containing the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object containing the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
        
        Input
            - An instance to predict.
            
        Output
            - 0 if the posterior probability of class 0 is higher 1 otherwise.
        """
        return 1 if self.ccd0.get_instance_posterior(x) < self.ccd1.get_instance_posterior(x) else 0


def compute_accuracy(testset, map_classifier):
    """
    Compute the accuracy of a given a testset and using a map classifier object.
    
    Input
        - testset: The test for which to compute the accuracy (Numpy array).
        - map_classifier : A MAPClassifier object capable of predicating the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / #testset size
    """
    correctly_classified = 0
    testset_size = len(testset)

    for instance in testset:
        if map_classifier.predict(instance[0: -1]) == instance[-1]:
            correctly_classified = correctly_classified + 1

    return correctly_classified / testset_size
