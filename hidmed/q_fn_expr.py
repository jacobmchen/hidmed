import numpy as np
import pandas as pd
from scipy.special import expit, logit
import statsmodels.api as sm
from statsmodels.formula.api import glm
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from .pipw import ProximalInverseProbWeightingBase
from .pmr import ProximalMultiplyRobustBase
from .hidmed_data import HidMedDataset

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
    M = X + 2*A + np.random.normal(0, 0.1*sd, n)

    # generate Y as a standard normal
    Y = 2*A + X + M + np.random.normal(0, 0.1*sd, n)

    # generate Z and W as standard normal random variables; in this
    # sim, Z and W are children of only M for simplicity
    Z = M + np.random.normal(0, 0.1*sd, n)
    W = M + np.random.normal(0, 0.1*sd, n)

    # generate potential outcome random variables so that we know what
    # values should be in ground truth
    M_A0 = X + 2*0 + np.random.normal(0, 0.1*sd, n)
    Y_A1_M_A0 = 2*1 + X + M_A0 + np.random.normal(0, 0.1*sd, n)

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

    # estimate treatment probabilities using sklearn and pipeline
    # turn off regularization
    model = Pipeline([("logistic", LogisticRegression(penalty=None))])
    # the first parameter is the covariates and the second parameter is the outcome
    predictors = pandas_data.drop('A', axis=1)
    # pass in values without column names
    model.fit(predictors.values, pandas_data['A'].values)
    # make predictions using values without column names
    model_pred = model.predict_proba(predictors.values)
    # print the learned coefficients of the linear logistic regression model
    print(model[0].intercept_)
    print(model[0].coef_)

    # declare an object of type ProximalMultiplyRobustBase, setting setup="a" estimates
    # psi_1
    proximal_estimator = ProximalMultiplyRobustBase(setup="a", num_runs=200)
    # fit the proximal estimator, which involves fitting the h and q bridge functions
    proximal_estimator.fit(datasets[0], datasets[1], treatment=model)
    res = proximal_estimator.evaluate(data)

    # according to Corollary 1, the estimate for the mediation term is the empirical average
    # of the output res for each row of the data
    print("ground truth Y(a', M(a)):", np.mean(Y_A1_M_A0))
    print('estimate of mediation term:', np.mean(res))