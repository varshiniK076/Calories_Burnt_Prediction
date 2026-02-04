import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sklearn
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
from log_code import setup_logging
logger = setup_logging('imbalanced_data')

from sklearn.preprocessing import StandardScaler
import pickle

class SCALE_DATA():
    def scale(X_train, X_test):
        try:
            logger.info('Balancing data')
            if 'User_ID' in X_train.columns:
                X_train = X_train.drop(columns=['User_ID'])
                X_test = X_test.drop(columns=['User_ID'])
            logger.info(f'Before balancing and Scaling Data : {X_train.shape}')
            logger.info(f'Before balancing and Scaling Data : {X_test.shape}')
            logger.info(f'{X_train.sample(10)}')
            logger.info(f'{X_test.sample(10)}')
            scaling = StandardScaler()
            X_train_scaled = pd.DataFrame(scaling.fit_transform(X_train),columns=X_train.columns,index=X_train.index)
            X_test_scaled = pd.DataFrame(scaling.transform(X_test),columns=X_test.columns,index=X_test.index)
            logger.info(f'After balancing and Scaling Data : {X_train_scaled.shape}')
            logger.info(f'After balancing and Scaling Data : {X_test_scaled.shape}')
            logger.info(f'{X_train_scaled.sample(10)}')
            logger.info(f'{X_test_scaled.sample(10)}')




            with open('scaling.pkl', 'wb') as f:
                pickle.dump(scaling, f)

            logger.info("Scaling completed successfully")

            return X_train_scaled, X_test_scaled

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')