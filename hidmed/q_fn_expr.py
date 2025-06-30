import numpy as np
import pandas as pd
from scipy.special import expit, logit
import statsmodels.api as sm
from statsmodels.formula.api import glm
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from .pipw import ProximalInverseProbWeightingBase
from .pmr import ProximalMultiplyRobustBase
from .pmr import ProximalMultiplyRobust
from .hidmed_data import HidMedDataset

import pickle

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

    # generate potential outcome random variables so that we know what
    # values should be in ground truth
    M_A0 = X + 0 + np.random.normal(0, 0.1*sd, n)
    Y_A1_M_A0 = 1 + X + M_A0 + np.random.normal(0, 0.1*sd, n)

    # save a pandas dataframe to train the treatment model
    pandas_data = pd.DataFrame({'A': A, 'X': X})

    # reshape the variables into an (n, 1) matrix, which is required for
    # the class HidMedDataset
    U = U.reshape(-1, 1)
    X = X.reshape(-1, 1)
    A = A.reshape(-1, 1)
    M = M.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    Z = Z.reshape(-1, 1)
    W = W.reshape(-1, 1)

    # put the generated data into an object of type HidMedDataset
    # each input needs to be a 2D array as the original code is general enough
    # to account for each variable having multiple dimensions
    data = HidMedDataset(X, U, A, M, W, Z, Y)
    print(data)

    # split the dataset into two parts
    datasets = data.split(2, 0)
    print(datasets[0])
    print(datasets[1])

    # generate A_star, a randomized version of A
    A_star = np.random.binomial(1, 0.5, n)
    # compute the empirical probability of observing A_star=1
    p_A_star = np.array([np.mean(A_star)]*n)

    # estimate treatment probabilities using sklearn and pipeline
    # turn off regularization
    model = Pipeline([("logistic", LogisticRegression(penalty=None))])
    # the first parameter is the covariates and the second parameter is the outcome
    predictors = pandas_data.drop('A', axis=1)
    # pass in values without column names
    model.fit(predictors.values, pandas_data['A'].values)
    # make predictions using values without column names and keep only the first column
    # as the first column is the prediction for p(A=1)
    model_pred = model.predict_proba(predictors.values)[:, 0]

    # compute weights where the numerator is the probability of success in A_sta
    # weights = (pandas_data['A']*p_A_star + (1-pandas_data['A'])*(1-p_A_star)) / (pandas_data['A']*model_pred + (1-pandas_data['A'])*(1-model_pred))
    weights = 0.5 / (pandas_data['A']*model_pred + (1-pandas_data['A'])*(1-model_pred))
    # standardize the weights by dividing by the sum of the weights and multiplying by the size of
    # the dataset; this makes sure that the weights sum to n
    weights_stand = weights / np.sum(weights) * n

    # train a dummy classifier for estimating A_star that makes predictions for
    # each possible class with equal probability
    model_star = Pipeline([("logistic", DummyClassifier(strategy="prior"))])
    predictors = pandas_data.drop('A', axis=1)
    # use the dummy model to make predictions for the treatment A; the predictions should
    # be just all 1s
    model_star.fit(predictors.values, pandas_data['A'].values)

    ### hyperparameter saving
    # # print the tuned parameters
    # print(proximal_estimator.param_dict)
    # # save the tuned parameters into a pickle file
    # file = open('hyperparams.pkl', 'wb')
    # pickle.dump(proximal_estimator.param_dict, file)
    # file.close()

    ### hyper parameter reading
    # read hyperparameters from the pickle file
    file = open('hyperparams.pkl', 'rb')
    tuned_param_dict = pickle.load(file)
    file.close()

    # declare an object of type ProximalMultiplyRobustBase, setting setup="a" estimates
    # psi_1
    # pass in the parameter dictionary that was previously computed for reproducibility
    proximal_estimator = ProximalMultiplyRobust(setup="a", folds=2, param_dict=tuned_param_dict, num_runs=200)
    # fit the proximal estimator, which involves fitting the h and q bridge functions;
    # pass in model_star to reflect that we are in an MSM and a seed for reproducibility
    proximal_estimator.fit(data, seed=0, treatment=model_star)

    # compute the pseudo-outcome for each row of data, pass in reduce as False so we
    # get the whole array of outputs
    res = proximal_estimator.evaluate(data, reduce=False)
    # reweight the pseudo-outcome by the MSM weights
    res_reweight = res[0] * weights_stand

    # according to Corollary 1, the estimate for the mediation term is the empirical average
    # of the output res for each row of the data
    print("ground truth Y(a', M(a)):", np.mean(Y_A1_M_A0))
    print('estimate of mediation term:', np.mean(res_reweight))