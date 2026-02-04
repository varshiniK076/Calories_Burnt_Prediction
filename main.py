'''
In this Project we are finding the calories burnt prediction using regression models
'''

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
logger = setup_logging('main')
from sklearn.model_selection import train_test_split
from random_sample_imputataion import RSI_tecnique
from variable_transformation import VARIABLE_TRANSFORMATION
from feature_selection import COMPLETE_FEATURE_SELECTION
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from Scaling import SCALE_DATA
from model_training import REGRESSION


class CALORIES_BURNT_PREDICTION():
    def __init__(self,path1,path2):
        try:
            self.path1 = pd.read_csv(path1)
            self.path2 = pd.read_csv(path2)
            logger.info(f'Total Rows and columns in exercise.csv:{self.path1.shape}')
            logger.info(f'Total Rows and columns in calories.csv:{self.path2.shape}')
            self.df = pd.merge(self.path1,self.path2,on="User_ID",how="inner" ) # inner means Keeps only matching User_IDs (most common)
            logger.info(self.df.head(5))
            logger.info(self.df.head(5))
            #check for null values
            logger.info(f'Total no.of null values in data:{self.df.isnull().sum()}')

            self.X = self.df.iloc[:,:-1]
            self.y = self.df.iloc[:,-1]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            logger.info(f'{self.X_train.columns}')
            #logger.info(self.y_train.info())

            logger.info(f'{self.X_train.head(5)}')
            logger.info(f'{self.y_train.head(5)}')

            logger.info(f'Training data size : {self.X_train.shape}')
            logger.info(f'Testing data size : {self.X_test.shape}')
            logger.info(f'========================================')
            for i in self.df.columns:
                if self.df[i].isnull().sum() > 0:
                    logger.info(f'{i} -> {self.df[i].dtype}')
                    if self.df[i].dtype == 'object':
                        self.df[i] = pd.to_numeric(self.df[i])
                        logger.info(f'{i} -> {self.df[i].dtype}')
                    else:
                        pass

            #for i in self.df.columns:
                #logger.info(f'{[i] -> {self.df[i].dtype}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
    def missing_values(self):
        try:
            if self.X_train.isnull().sum().all() > 0 or self.X_test.isnull().sum().all() > 0:
                self.X_train,self.X_test = RSI_tecnique.random_sample_imputataion(self.X_train, self.X_test)
            else:
                logger.info(f'There are no null values in data:{self.X_train.isnull().sum()}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
    def var_transformation(self):
        try:
            logger.info(f'columns of X_train:{self.X_train.columns}')
            logger.info(f'columns of X_test:{self.X_test.columns}')
            self.X_train_numeric = self.X_train.select_dtypes(exclude='object')
            self.X_train_categorical = self.X_train.select_dtypes(include='object')
            self.X_test_numeric = self.X_test.select_dtypes(exclude='object')
            self.X_test_categorical = self.X_test.select_dtypes(include='object')
            logger.info(f'{self.X_train_numeric.columns}')
            logger.info(f'{self.X_train_categorical.columns}')
            logger.info(f'{self.X_test_numeric.columns}')
            logger.info(f'{self.X_test_categorical.columns}')
            logger.info(f'{self.X_train_numeric.shape}')
            logger.info(f'{self.X_train_categorical.shape}')
            logger.info(f'{self.X_test_numeric.shape}')
            logger.info(f'{self.X_test_categorical.shape}')
            self.X_train_numeric, self.X_test_numeric = VARIABLE_TRANSFORMATION.variable_trans(self.X_train_numeric, self.X_test_numeric)
            logger.info(f"{self.X_train_numeric.columns} -> {self.X_train_numeric.shape}")
            logger.info(f"{self.X_test_numeric.columns} -> {self.X_test_numeric.shape}")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
    '''
    def f_selection(self):
        try:
            logger.info(f" Before : {self.X_train_numeric.columns} -> {self.X_train_numeric.shape}")
            logger.info(f"Before : {self.X_test_numeric.columns} -> {self.X_test_numeric.shape}")
            self.X_train_numeric,self.X_test_numeric = COMPLETE_FEATURE_SELECTION.feature_selection(self.X_train_numeric, self.X_test_numeric,self.y_train)
            logger.info(f" After : {self.X_train_numeric.columns} -> {self.X_train_numeric.shape}")
            logger.info(f"After : {self.X_test_numeric.columns} -> {self.X_test_numeric.shape}")
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
    '''
    def cat_to_num(self):
        try:
            logger.info('Categorical to Numerical')
            logger.info(f'{self.X_train_categorical.columns}')
            logger.info(f'{self.X_test_categorical.columns}')

            for i in self.X_train_categorical.columns:
                logger.info(f'{i} --> {self.X_train_categorical[i].unique()}')

            logger.info(f'Before Converting : {self.X_train_categorical}')
            logger.info(f'Before Converting : {self.X_test_categorical}')

            #One-Hot Encoding
            one_hot = OneHotEncoder(drop = 'first')
            one_hot.fit(self.X_train_categorical[['Gender']])
            res = one_hot.transform(self.X_train_categorical[['Gender']]).toarray()

            f = pd.DataFrame(data = res, columns = one_hot.get_feature_names_out())
            self.X_train_categorical.reset_index(drop = True, inplace = True)
            f.reset_index(drop = True, inplace = True)

            self.X_train_categorical = pd.concat([self.X_train_categorical, f],axis = 1)
            self.X_train_categorical = self.X_train_categorical.drop(['Gender'], axis = 1)

            res1 = one_hot.transform(self.X_test_categorical[['Gender']]).toarray()
            f1 = pd.DataFrame(data = res1, columns = one_hot.get_feature_names_out())

            self.X_test_categorical.reset_index(drop=True, inplace=True)
            f1.reset_index(drop=True, inplace=True)

            self.X_test_categorical = pd.concat([self.X_test_categorical, f1],axis = 1)
            self.X_test_categorical = self.X_test_categorical.drop(['Gender'], axis = 1)

            logger.info(f'{self.X_train_categorical.columns}')
            logger.info(f'{self.X_test_categorical.columns}')

            logger.info(f"After Converting : {self.X_train_categorical}")
            logger.info(f"After Converting : {self.X_test_categorical}")

            logger.info(f"{self.X_train_categorical.shape}")
            logger.info(f"{self.X_test_categorical.shape}")

            logger.info(f"{self.X_train_categorical.isnull().sum()}")
            logger.info(f"{self.X_test_categorical.isnull().sum()}")

            self.X_train_numeric.reset_index(drop=True, inplace=True)
            self.X_test_numeric.reset_index(drop=True, inplace=True)

            self.X_train_categorical.reset_index(drop=True, inplace=True)
            self.X_test_categorical.reset_index(drop=True, inplace=True)

            self.training_data = pd.concat([self.X_train_numeric, self.X_train_categorical], axis=1)
            self.testing_data = pd.concat([self.X_test_numeric, self.X_test_categorical], axis=1)

            logger.info(f"{self.training_data.shape}")
            logger.info(f"{self.testing_data.shape}")

            logger.info(f"{self.training_data.isnull().sum()}")
            logger.info(f"{self.testing_data.isnull().sum()}")

            logger.info(f"=======================================================")

            logger.info(f"Training Data : {self.training_data.sample(10)}")
            logger.info(f"Testing Data : {self.testing_data.sample(10)}")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def data_balance(self):
        try:
            logger.info('Scaling Data Before Regression')
            self.X_train_numeric, self.X_test_numeric = SCALE_DATA.scale(self.X_train_numeric, self.X_test_numeric)
            logger.info('Scaling Data After Regression')
            self.X_train = pd.concat([self.X_train_numeric, self.X_train_categorical], axis = 1)
            self.X_test = pd.concat([self.X_test_numeric, self.X_test_categorical], axis = 1)
            self.X_train = self.X_train.drop(['User_ID'], axis=1)
            self.X_test = self.X_test.drop(['User_ID'], axis=1)
            logger.info(f'{self.X_train.shape} -> {self.X_train.columns}')
            logger.info(f'{self.X_test.shape} -> {self.X_test.columns}')


            REGRESSION.linear_regression(self.X_train, self.y_train, self.X_test, self.y_test)
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')





        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')












if __name__ == "__main__":
    try:
        obj = CALORIES_BURNT_PREDICTION('C:\\Users\\VARSHINI\\Downloads\\Calories_Burnt_Prediction\\exercise.csv','C:\\Users\\VARSHINI\\Downloads\\Calories_Burnt_Prediction\\calories.csv')
        obj.missing_values()
        obj.var_transformation()
        #obj.f_selection()
        obj.cat_to_num()
        obj.data_balance()
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')