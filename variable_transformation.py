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
logger = setup_logging('variable_transformation')

class VARIABLE_TRANSFORMATION():
    def __init__(self):
        pass
    def variable_trans(X_train_numeric,X_test_numeric):
        try:
            logger.info(f"{X_train_numeric.columns} -> {X_train_numeric.shape}")
            logger.info(f"{X_test_numeric.columns} -> {X_test_numeric.shape}")
            PLOT_PATH = "plot_path"
            logger.info(f'Before Variable Transformation')
            # Plotting Outliers Before
            for i in X_train_numeric.columns:
                plt.figure()
                X_train_numeric[i].plot(kind='kde', color='r')
                plt.title(f'KDE-{i}')
                plt.savefig(f'{PLOT_PATH}/kde_{i}.png')
                plt.close()
            for i in X_train_numeric.columns:
                plt.figure()
                sns.boxplot(x=X_train_numeric[i])
                plt.title(f'Boxplot-{i}')
                plt.savefig(f'{PLOT_PATH}/boxplot_{i}.png')
                plt.close()

            for col in X_train_numeric.columns:
                 # 1️⃣ Log transform (BEST for calories-like features)
                X_train_numeric[col] = np.log1p(X_train_numeric[col])
                X_test_numeric[col] = np.log1p(X_test_numeric[col])

                # 2️⃣ Quantile capping (from train only)
                lower = X_train_numeric[col].quantile(0.01)
                upper = X_train_numeric[col].quantile(0.99)

                X_train_numeric[col] = X_train_numeric[col].clip(lower, upper)
                X_train_numeric[col] = X_train_numeric[col].clip(lower, upper)

            logger.info(f'{X_train_numeric.shape}')
            logger.info(f"{X_test_numeric.columns} -> {X_test_numeric.shape}")

            #Plotting Outliers After
            logger.info(f'After Variable Transformation')
            logger.info(f'{X_train_numeric.columns}')
            for i in X_train_numeric.columns:
                plt.figure()
                X_train_numeric[i].plot(kind='kde', color='r')
                plt.title(f'KDE-{i}')
                plt.savefig(f'{PLOT_PATH}/kde_{i}_new.png')
                plt.close()
            for i in X_train_numeric.columns:
                plt.figure()
                sns.boxplot(x=X_train_numeric[i])
                plt.title(f'Boxplot-{i}')
                plt.savefig(f'{PLOT_PATH}/boxplot_{i}_new.png')
                plt.close()

            return X_train_numeric, X_test_numeric

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

if __name__ == "__main__":
    pass