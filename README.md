# hidmed
Causal Inference with Hidden Mediators

This repository is a fork from the hidden mediators repository written by Alan Yang.

The goal of this fork is to modify the repository to create a class that is able to learn the bridge function $q_a(Z, X)$ and return outputs for the bridge function. Furthermore, this fork will identify which files are relevant to this task and keep only those files.

Before running, need to install the HOLA package in Python that helps with hyperparameter tuning: https://github.com/blackrock/HOLA. 

Some notes on the edits I make in this repository:
0. The file q_fn_expr.py contains experiments for learning the syntax for constructing code that will give estimates for the mediation term.
1. Proposition 1 shows the integral that we must solve for in order to learn the function $a_a$. Note that it takes as input the variables $Z, W, X,$ and $A$. 
2. The file pipw.py contains code that builds the ProximalInverseProbWeightingBase class that we will edit. We basically only need the fit and evaluate functions.
3. The fit function takes in two arguments: fit_data and val_data. The val_data is needed for hyperparameter tuning. We need to pass in the parameter "q" for the parameter which and supply our own treatment_prob.
4. The datasets passed into the functions defined in ProximalInverseProbWeightingBase are of type HidMedDataset, which is a wrapper class for a dataset used in these experiments.
5. In pipw.py, the method fit_treatment_probability() returns a 3-tuple, the first element of which is passed into the fit_bridge method. The first element is of type Pipeline[(LogisticRegression)] from the sklearn package. Thus, in my implementation, I will do the same and pass in a Pipeline object for the parameter treatment.
6. There is a long run time and made my laptop pretty hot. Perhaps there is a hyperparameter to tune somewhere in the kkt_solve() method in bridge_q.py file that will reduce the number of iterations or cross validations that the function runs. Update: when declaring an estimator object, there is a parameter called num_runs that specifies how many times to tune hyperparameters in the fit_bridge function. By default, it is 200, but I turned it down so that the run-time is more tractable.
7. Have successfully set up a pipeline for the pipw estimator, but it seems to be biased. However, even in the paper the pipw estimator seems to be more biased than the doubly robust estimator. So, we are going to try setting up the multiply robust estimator.
8. In the ProximalEstimatorBase class, there are methods for fitting two other functions: h and eta. By default, num_runs is set to 200 for tuning hyperparameters for these two functions. However, these do not seem to affect the amount of time it takes to run the experiments.
9. With certain DGPs, the estimating the parameters for the function eta seems not to converge, giving biased estimates for the mediation term. This happens especially as a result of the q function not converging likely because of setting too little runs for tuning hyperparameters.
10. According to Remark 2, in practice we avoid estimating p(W | A, X) directly since W may be a continuous variable and we want to avoid estimating arbitrary conditional distributions. However, doing so means that we only have 2 conditions to satisfy to satisfy the multiply robustness property.
11. Eta is a conditional mean that depends on the bridge function h. When the function eta is not correctly learned, which is indicated by the r2 value outputted after training eta, is not close 1, the estimates for the mediation term is biased. However, when the function eta is correctly learned, the estimates for the mediation term are unbiased.
12. For small n, even after turning the number of runs up for learning the function q up, it still seems to be incorrectly specified. However, the function h seems to converge pretty consistently.
13. Now that we've set up a rough pipeline in q_fn_expr.py, we will want to write a class that takes as input a pandas dataframe, does pre-processing, and returns estimates for the mediation term.
