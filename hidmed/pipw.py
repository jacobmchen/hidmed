import numpy as np

from .proximal_estimator_base import ProximalEstimatorBase
from .cross_fit_base import CrossFittingEstimatorBase


class ProximalInverseProbWeightingBase(ProximalEstimatorBase):
    """PIPW estimator based on Theorem 2"""

    def fit(self, fit_data, val_data, treatment=None):
        """Fit the PIPW estimator
        JMC edit: added another parameter treatment so that I can self-specify
        treatment probabilities
        """
        # estimate treatment probability
        if treatment is None:
            if self.dgp is None:
                treatment, treatment_params, _ = self.fit_treatment_probability(
                    fit_data,
                    val_data,
                )
                self.treatment = treatment
                self.params["treatment"] = treatment_params
            else:
                treatment = self.dgp
                self.treatment = treatment
        else:
            # JMC: if treatment is not None, then we can pass it in directly to the
            # fit_bridge method and also set the attribute to the passed in treatment
            self.treatment = treatment

        # fit bridge functions
        q_fn, q_params, _ = self.fit_bridge(fit_data, val_data, which="q", treatment_prob=treatment)
        self.q_fn = q_fn
        self.params["q"] = q_params

        return self
    
    def evaluate_q_fn(self, eval_data):
        """Evaluate the learned q_fn and return its outputs on the evaluation data"""
        # JMC: eval_data is an object of type HidMedDataset, a class that wraps the
        # dataset.
        q_eval = self.q_fn(np.hstack((eval_data.z, eval_data.x))).flatten()

        return q_eval

    def evaluate(self, eval_data):
        """Evaluate the PIPW estimator pointwise on the evaluation data"""
        # JMC: this function gives a vector/list of the estimator phi_1 given in 
        # theorem 2, one could then take the empirical mean as an esimate for
        # the target estimand
        loc1 = eval_data.a[:, 0] == 1
        a_eval = loc1.astype(float).flatten()
        y_eval = eval_data.y[:, 0].flatten()
        q_eval = self.q_fn(np.hstack((eval_data.z, eval_data.x))).flatten()
        prop_score = self.treatment.predict_proba(eval_data.x)[:, 1]

        # JMC: setup is a parameter you have to specify when you declare an object of type
        # ProximalEstimatorBase
        if self.setup == "a":
            return a_eval * y_eval * q_eval / prop_score
        
        return a_eval * y_eval * q_eval


class ProximalInverseProbWeighting(CrossFittingEstimatorBase):
    """PIPW estimator with cross-fitting"""

    base_estimator = ProximalInverseProbWeightingBase
