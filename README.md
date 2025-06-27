# hidmed
Causal Inference with Hidden Mediators

This repository is a fork from the hidden mediators repository written by Alan Yang.

The goal of this fork is to modify the repository to create a class that is able to learn the bridge function $q_a(Z, X)$ and return outputs for the bridge function. Furthermore, this fork will identify which files are relevant to this task and keep only those files.

Before running, need to install the HOLA package in Python that helps with hyperparameter tuning: https://github.com/blackrock/HOLA. 

Some notes on the construction of this repository:
1. Proposition 1 shows the integral that we must solve for in order to learn the function $a_a$. Note that it takes as input the variables $Z, W, X,$ and $A$. 
2. The file pipw.py contains code that builds the ProximalInverseProbWeightingBase class that we will edit. We basically only need the fit and evaluate functions.
3. The fit function takes in two arguments: fit_data and val_data. The val_data is needed for hyperparameter tuning. We need to pass in the parameter "q" for the parameter which and supply our own treatment_prob.
4. The datasets passed into the functions defined in ProximalInverseProbWeightingBase are of type HidMedDataset, which is a wrapper class for a dataset used in these experiments.
5. In pipw.py, the method fit_treatment_probability() returns a 3-tuple, the first element of which is passed into the fit_bridge method. The first element is of type Pipeline[(LogisticRegression)] from the sklearn package. Thus, in my implementation, I will do the same and pass in a Pipeline object for the parameter treatment.
6. There is a long run time and made my laptop pretty hot. Perhaps there is a hyperparameter to tune somewhere in the kkt_solve() method in bridge_q.py file that will reduce the number of iterations or cross validations that the function runs. Update: when declaring an estimator object, there is a parameter called num_runs that specifies how many times to tune hyperparameters in the fit_bridge function. By default, it is 200, but I turned it down so that the run-time is more tractable.
