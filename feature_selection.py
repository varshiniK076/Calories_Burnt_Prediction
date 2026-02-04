import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import sklearn
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
from log_code import setup_logging
logger = setup_logging('feature_selection')
from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold
reg_constant = VarianceThreshold(threshold=0.0)
reg_quasi_constant = VarianceThreshold(threshold=0.1)

class COMPLETE_FEATURE_SELECTION():
    def feature_selection(X_train_numeric,X_test_numeric,y_train):
        try:
            logger.info(f"{X_train_numeric.columns} -> {X_train_numeric.shape}")
            logger.info(f"{X_test_numeric.columns} -> {X_test_numeric.shape}")
        #constant technique
            reg_constant.fit(X_train_numeric)
            logger.info(f'Columns we need to remove because std.deviation is 0:{X_train_numeric.columns[~reg_constant.get_support()]}')
            good_features_train = reg_constant.transform(X_train_numeric)
            good_features_test = reg_constant.transform(X_test_numeric)
            X_train_numeric_fs = pd.DataFrame(data=good_features_train, columns=X_train_numeric.columns[reg_constant.get_support()])
            X_test_numeric_fs = pd.DataFrame(data= good_features_test, columns=X_test_numeric.columns[reg_constant.get_support()])
            #logger.info(X_train_numeric_fs.head(5))
            #logger.info(X_test_numeric_fs.head(5))

        #Quasi-constant techinique
            reg_quasi_constant.fit(X_train_numeric_fs)
            logger.info(f'Columns we need to remove from quasi_constant_technique : {X_train_numeric_fs.columns[~reg_quasi_constant.get_support()]}')
            good_feature_train1 = reg_quasi_constant.transform(X_train_numeric_fs)
            good_feature_test1 = reg_quasi_constant.transform(X_test_numeric_fs)
            X_train_numeric_fs_1 = pd.DataFrame(data=good_feature_train1, columns=X_train_numeric_fs.columns[reg_quasi_constant.get_support()])
            X_test_numeric_fs_2 = pd.DataFrame(data=good_feature_test1, columns=X_test_numeric_fs.columns[reg_quasi_constant.get_support()])
            #logger.info(X_train_numeric_fs_1.head(5))
            #logger.info(X_test_numeric_fs_2.head(5))
       #Hypothesis Testing
            logger.info(f"Before hypothesis testing : {X_train_numeric_fs_1.columns} -> {X_train_numeric_fs_1.shape}")
            logger.info(f"Before hypothesis testing : {X_test_numeric_fs_2.columns} -> {X_test_numeric_fs_2.shape}")
            value = 0.05
            p_values = []
            for i in X_train_numeric_fs_1.columns:
                correlation, p_value = pearsonr(X_train_numeric_fs_1[i], y_train)
                p_values.append(p_value)
            for i in p_values:
                logger.info(f"values: {i}")
            p_values = pd.Series(p_values, index=X_train_numeric_fs_1.columns)
            features_to_remove = []
            for i in p_values.index:
                if p_values[i] > value:
                    features_to_remove.append(i)
            logger.info(f"Features removed by hypothesis testing: {features_to_remove}")
            X_train_numeric_fs_1 = X_train_numeric_fs_1.drop(columns=features_to_remove)
            X_test_numeric_fs_2 = X_test_numeric_fs_2.drop(columns=features_to_remove)

            logger.info(f"After hypothesis testing : {X_train_numeric_fs_1.columns} -> {X_train_numeric_fs_1.shape}")
            logger.info(f"After hypothesis testing : {X_test_numeric_fs_2.columns} -> {X_test_numeric_fs_2.shape}")
            return X_train_numeric_fs_1, X_test_numeric_fs_2

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

