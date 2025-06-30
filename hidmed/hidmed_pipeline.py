import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from scipy.special import expit, logit

from .pmr import ProximalMultiplyRobust
from .hidmed_data import HidMedDataset

import pickle

class HidMedPipeline():
    """
    A class that takes as input prespecified data and estimates the counterfactual
    mediation term E[Y(a', M(a))].

    Some assumptions in this class: A is binary, a'=1, a=0, dimension of A, Y, W, and
    Z are all 1 whereas the dimension of X can vary
    """

    def __init__(self, data, var_name_dict, propensity_score_model=None, folds=2, num_runs=200, seed=0):
        """
        Constructor
        data: a pandas dataframe containing the dataset with a hidden mediator
        var_name_dict: a dictionary that maps string to string where the key is
        the variable name(s) in the canonical graph (one of A, X, Y, W, Z) and the
        value is the corresponding name of the string in the pandas dataframe
        propensity_score_model: an sklearn Pipeline object containing the propensity
        score model
        folds: the number of cross-validation folds when fitting the proximal estimator
        num_runs: the number of runs to use when fitting hyperparameters in the proximal estimator
        seed: the desired seed
        """
        # set the seed
        np.random.seed(seed)

        # save the parameters as attributes
        self.data = data
        self.var_name_dict = var_name_dict
        self.propensity_score_model = propensity_score_model
        self.folds = folds
        self.num_runs = num_runs
        self.hyperparams = None

        # save the columns A, W, Z, and Y in the format required for the 
        # hidden mediation dataset
        A = np.array(data[var_name_dict['A']])
        Y = np.array(data[var_name_dict['Y']])
        W = np.array(data[var_name_dict['W']])
        Z = np.array(data[var_name_dict['Z']])

        # save the all the columns corresponding to X in the format required
        # for the hidden mediation dataset
        X = np.array(data[var_name_dict['X']])

        # create dummy data for U and M as they are hidden
        U = np.array(np.random.normal(0, 1, len(data)))
        U = U.reshape(-1, 1)
        M = np.array(np.random.normal(0, 1, len(data)))
        M = U.reshape(-1, 1)

        # set up the hidden mediator dataset as an attribute in the class
        self.hid_med_data = HidMedDataset(X, U, A, M, W, Z, Y)

    def learn_propensity_score_model(self, model=None):
        """
        Use the pandas data to learn the propensity score model and update the
        attribute

        The model the class learns will be a linear logistic regression model

        If a model is passed in, then save that directly
        """
        if model is None:
            # estimate treatment probabilities using sklearn and pipeline
            # turn off regularization
            model = Pipeline([("logistic", LogisticRegression(penalty=None))])
            # save the predictors as an np array
            predictors = np.array(self.data[self.var_name_dict['X']])
            # the first parameter is the covariates and the second parameter is the outcome
            # the ravel function flattens the one-dimensional outcome variable
            model.fit(predictors, np.array(self.data[self.var_name_dict['A']]).ravel())
            # save the fitted model as the propensity score model
            self.propensity_score_model = model
        else:
            self.propensity_score_model = model

        # return the fitted model
        return model
    
    def setup_hyperparameters(self, learn_new_params, filename):
        """
        Setup the hyperparameters needed for the proximal estimation procedure. If learn_new_params
        is True, then learn new hyperparameters and save them using filename.
        """
        # make sure that we have a propensity score model
        assert self.propensity_score_model != None, "Error in tune_hyperparameters: need a propensity score model"

        if learn_new_params is True:
            # declare an object of type ProximalMultiplyRobustBase, setting setup="a" estimates
            # psi_1
            proximal_estimator = ProximalMultiplyRobust(setup="a", folds=self.folds, num_runs=self.num_runs)
            # fit the proximal estimator, which involves fitting the h and q bridge functions;
            # pass in model_star to reflect that we are in an MSM and a seed for reproducibility
            proximal_estimator.fit(self.hid_med_data, seed=0, treatment=self.propensity_score_model)

            # save the learned hyperparameters as an attribute in the class
            self.hyperparams = proximal_estimator.param_dict

            # save the tuned parameters into a pickle file
            file = open(filename, 'wb')
            pickle.dump(proximal_estimator.param_dict, file)
            file.close()
        else:
            # read hyperparameters from the pickle file
            file = open(filename, 'rb')
            tuned_param_dict = pickle.load(file)
            file.close()

            # save the learned hyperparameters as an attribute in the class
            self.hyperparams = tuned_param_dict

        return self.hyperparams
    
    def estimate_mediation_term(self):
        """
        Estimate the counterfactual mediation term.
        """
        assert self.hyperparams != None, "Error in estimate_mediation_term: setup hyperparameters first"

        # declare an object of type ProximalMultiplyRobustBase, setting setup="a" estimates
        # psi_1
        # pass in the parameter dictionary that was previously computed for reproducibility
        proximal_estimator = ProximalMultiplyRobust(setup="a", folds=self.folds, param_dict=self.hyperparams, num_runs=self.num_runs)
        # fit the proximal estimator, which involves fitting the h and q bridge functions;
        # pass in model_star to reflect that we are in an MSM and a seed for reproducibility
        proximal_estimator.fit(self.hid_med_data, seed=0, treatment=self.propensity_score_model)

        # compute the pseudo-outcome for each row of data, pass in reduce as False so we
        # get the whole array of outputs
        res = proximal_estimator.evaluate(self.hid_med_data, reduce=True)

        return res
    
# code for testing the class
if __name__ == "__main__":
    # set the seed
    np.random.seed(0)

    # define the sample size
    n = 1000

    # define the standard deviation
    sd = 1

    # generate U as a standard normal, but none of the other variables
    # will depend on it since I don't want latent confounding in this sim
    U = np.random.normal(0, sd, n)

    # generate X as a standard normal
    X = np.random.normal(0, sd, n)
    X2 = np.random.normal(0, 0.1*sd, n)

    # generate A as a Bernoulli random variable
    A = np.random.binomial(1, expit(X), n)

    # generate M as a standard normal
    M = X + A + np.random.normal(0, 0.1*sd, n)

    # generate Y as a standard normal
    Y = A + X + M + np.random.normal(0, 0.1*sd, n)

    # generate Z and W as standard normal random variables; in this
    # sim, Z and W are children of only M for simplicity
    Z = M + np.random.normal(0, 0.1*sd, n)
    W = M + np.random.normal(0, 0.1*sd, n)

    # create a dataframe
    df = pd.DataFrame({'A': A, 'X': X, 'X2': X2, 'Y': Y, 'Z': Z, 'W': W})

    # create a dictionary mapping canonical variable names to actual variable names
    var_name_dict = {'A': ['A'], 'X': ['X', 'X2'], 'Y': ['Y'], 'Z': ['Z'], 'W': ['W']}

    # create an object of type HidMedPipeline
    hid_med_pipeline = HidMedPipeline(df, var_name_dict, folds=5)

    # learn the propensity score model and print it
    print(hid_med_pipeline.learn_propensity_score_model())

    # learn the hyperparameters and print them
    print(hid_med_pipeline.setup_hyperparameters(learn_new_params=True, filename='test_hyperparams.pkl'))

    # evaluate the mediation term and print it
    print(hid_med_pipeline.estimate_mediation_term())
