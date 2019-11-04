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
        # Given data set and class values
        self.dataset = dataset
        self.class_value = class_value

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        # count of the number of instances from the chosen class
        classcounter = np.count_nonzero(self.dataset[:, -1])
        # total number of instances
        datasize = self.dataset.shape[0]
        # returns the prior given the class
        return (classcounter / datasize) if self.class_value == 1 else ((datasize - classcounter) / datasize)

    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihoodProbability = 1
        # the data set only with the relevant class
        datasetbyclass = self.dataset[self.dataset[:, -1] == self.class_value]

        # iterate on the data set to find the mean and std of the data
        for i in range(self.dataset.shape[1] - 1):
            mean = np.mean(datasetbyclass[:, i])
            std = np.std(datasetbyclass[:, i])
            likelihoodProbability *= normal_pdf(x[i], mean, std)

        return likelihoodProbability

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_prior() * self.get_instance_likelihood(x)


class MultiNormalClassDistribution:
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.

        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """

        # X is the data set with only the chosen class value
        X = dataset[dataset[:, -1] == class_value][:, 0:-1]
        self.mean = X.mean(axis=0)
        self.cov = np.cov(X.T)
        self.lenByClass = len(X)
        self.datasetLen = len(dataset)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.lenByClass / self.datasetLen

    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        return multi_normal_pdf(x, self.mean, self.cov)

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_prior() * self.get_instance_likelihood(x)


def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.

    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mean and var for the given x.
    """
    return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * std ** 2))


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variante normal desnity function for a given x, mean and covarince matrix.

    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mean and var for the given x.
    """
    x = x[:-1]
    return ((((2 * np.pi) ** (len(mean) / -2)) * (np.linalg.det(cov) ** (-0.5))) *
            (np.exp((-0.5) * ((np.transpose(x - mean).T.dot(np.linalg.inv(cov))).dot(x - mean)))))


####################################################################################################
#                                            Part B
####################################################################################################
EPSILLON = 1e-6     # == 0.000001 It could happen that a certain value will only occur in the test set.
                    # In case such a thing occur the probability for that value will EPSILLON.


class DiscreteNBClassDistribution:
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes
        distribution for a specific class. The probabilites are computed with la place smoothing.

        Input
        - dataset: The dataset from which to compute the probabilites (Numpy Array).
        - class_value : Compute the relevant parameters only for instances from the given class.
        """
        # the data set
        self.dataset = dataset
        # data set by class
        self.X = dataset[dataset[:, -1] == class_value][:, 0:-1]
        # length of data by class and the full data
        self.lenByClass = len(self.X)
        self.datasetLen = len(dataset)
        # conversion to an numpy array named X
        self.X = np.array(self.X)
        # number of training instances in the given class
        self.ni_instances = self.X.shape[0]
        # the number of possible values of the relevant class
        self.vj = None
        # the number of training instances from the given class and the value x_j in the relevant class
        self.nij_instances = None

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.lenByClass / self.datasetLen

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood probability of the instance under the class according to the dataset distribution.
        """
        likelihoodProbability = 1
        for i in range(self.dataset.shape[1] - 1):
            self.nij_instances = (self.X[:, i] == x[i]).sum()
            self.vj = len(np.unique(self.X[:, i]))
            probability = self.laplace_estimation()
            likelihoodProbability *= probability

        return likelihoodProbability

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_prior() * self.get_instance_likelihood(x)

    def laplace_estimation(self):
        return EPSILLON if self.nij_instances == 0 else (self.nij_instances + 1) / (self.ni_instances + self.vj)

####################################################################################################
#                                            General
####################################################################################################


class MAPClassifier:
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a postreiori classifier.
        This class will hold 2 class distribution, one for class 0 and one for class 1, and will predicit and instance
        by the class that outputs the highest posterior probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        # class distribution of class 0
        self.ccd0 = ccd0
        # class distribution of class 1
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
        - map_classifier : A MAPClassifier object capable of p rediciting the class for each instance in the testset.

    Ouput
        - Accuracy = #Correctly Classified / #testset size
    """
    # number of classes that were correctly classified
    correctly_classified = 0
    # size of the test data
    testset_size = len(testset)

    # iterate over the test data and add each correctly classified class to the correctly classified
    for instance in testset:
        if map_classifier.predict(instance) == instance[-1]:
            correctly_classified = correctly_classified + 1

    return correctly_classified / testset_size
