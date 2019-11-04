import numpy as np

np.random.seed(42)


####################################################################################################
#                                            Part A
####################################################################################################

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        A = dataset[dataset[:, -1] == class_value][:, 0:-1]
        self.mean = A.mean(axis=0)
        self.std = A.std(axis=0)
        self.Alen = len(A)
        self.datasetlen = len(dataset)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.Alen / self.datasetlen

    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        return normal_pdf(x, self.mean, self.std).prod()

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """

        return self.get_prior() * self.get_instance_likelihood(x)


class MultiNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        A = dataset[dataset[:, -1] == class_value][:, 0:-1]
        self.mean = A.mean(axis=0)
        self.cov = np.cov(A.T)
        self.Alen = len(A)
        self.datasetlen = len(dataset)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.Alen / self.datasetlen

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
    return (1 / (np.sqrt(2 * np.pi * np.square(std)))) * np.exp(-1 * (np.square(x - mean) / (2 * np.square(std))))


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variante normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    return np.power((2 * np.pi), (len(mean) / -2)) * np.power(np.linalg.det(cov), -0.5) * \
           np.exp(-0.5 * ((x - mean).T).dot(np.linalg.inv(cov)).dot(x - mean))


####################################################################################################
#                                            Part B
####################################################################################################
EPSILLON = 1e-6  # == 0.000001 It could happen that a certain value will only occur in the test set.


# In case such a thing occur the probability for that value will EPSILLON.


class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with la place smoothing.
        
        Input
        - dataset: The dataset from which to compute the probabilites (Numpy Array).
        - class_value : Compute the relevant parameters only for instances from the given class.
        """
        self.A = dataset[dataset[:, -1] == class_value][:, 0:-1]
        self.n_i = len(self.A)
        self.v_j = []
        for data in dataset[:, 0:-1].T:
            self.v_j.append(len(np.unique(data, return_counts=True)[1]))
        self.v_j = np.array(self.v_j)
        self.datasetlen = len(dataset)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.n_i / self.datasetlen

    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        pxa = ((self.A == x).sum(axis=0) + 1) / (self.n_i + self.v_j)
        val_arr = []
        for data in range(len(x)):
            if pxa[data] == self.A[:, data]:
                val_arr = pxa[data]
            else:
                val_arr = EPSILLON
        return val_arr.prod()

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_prior() * self.get_instance_likelihood(x)


####################################################################################################
#                                            General
####################################################################################################            
class MAPClassifier():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a postreiori classifier. 
        This class will hold 2 class distribution, one for class 0 and one for class 1, and will predicit and instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd1 = ccd1
        self.ccd0 = ccd0

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
        
        Input
            - An instance to predict.
            
        Output
            - 0 if the posterior probability of class 0 is higher 1 otherwise.
        """

        if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x):
            return 0
        else:
            return 1


def compute_accuracy(testset, map_classifier):
    """
    Compute the accuracy of a given a testset and using a map classifier object.
    
    Input
        - testset: The test for which to compute the accuracy (Numpy array).
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / #testset size
    """
    accuracy = 0.0
    # for each sample, if the predict equal the right label
    # then add 1 to tha accuracy, and at the end calculate
    # the percentage
    for data in testset:
        if map_classifier.predict(data[0:-1]) == data[-1]:
            accuracy += 1

    accuracy = accuracy / len(testset)
    return accuracy
