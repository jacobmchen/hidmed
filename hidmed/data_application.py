import numpy as np
import pandas as pd

from .hidmed_pipeline import HidMedPipeline

# use the real data and run the hidden mediator pipeline

if __name__ == "__main__":
    # read the dataset
    analysis_data = pd.read_csv('../aki_mediation/analysis_data_stand_07022025.csv')

    # subset the columns that we need
    columns_analysis = ['creatlst', 'age', 'gender', 'bmi',
                                   'hypertn', 'dm', 'copd', 'chf', 'prior_mi',
                                   'hct', 'hdef', 'statin', 'acearb', 'betablocker',
                                   'rbc_transfusion',
                                   'xclamp_duration', 'nadirDO2',
                                   'delta_KIM.1', 'delta_MCP.1', 'delta_NGAL', 'delta_YKL.40',
                                   'aki']
    analysis_data = analysis_data[columns_analysis]

    # keep only the complete cases
    analysis_data = analysis_data.dropna()
    n = len(analysis_data)
    print(n)

    # binarize xclamp_duration and nadirDO2 since the implementation of the hidden
    # mediator pipeline requires binary treatment
    # median of standardized xclamp_duration is 2.63
    analysis_data['xclamp_duration_binary'] = \
        np.array([1 if val > 2.63 else 0 for val in analysis_data['xclamp_duration'].values])
    analysis_data['nadirDO2_binary'] = \
        np.array([1 if val  > 2.22 else 0 for val in analysis_data['nadirDO2'].values])

    # create a dictionary mapping canonical variable names to actual variable
    # names
    var_name_dict = {'A': ['xclamp_duration_binary'],
                     'X': ['creatlst', 'age', 'gender', 'bmi',
                                   'hypertn', 'dm', 'copd', 'chf', 'prior_mi',
                                   'hct', 'hdef', 'statin', 'acearb', 'betablocker',
                                   'rbc_transfusion'],
                     'Y': ['aki'],
                     'Z': ['delta_KIM.1'],
                     'W': ['delta_MCP.1']
                     }
    
    # create a pipeline object
    hid_med_pipeline = HidMedPipeline(analysis_data, var_name_dict, folds=5)

    # learn a propensity score model
    hid_med_pipeline.learn_propensity_score_model()

    # learn the hyperparameters and save them
    hid_med_pipeline.setup_hyperparameters(learn_new_params=True, filename='data_hyperparams.pkl')

    # estimate the mediation term
    print(hid_med_pipeline.estimate_mediation_term())