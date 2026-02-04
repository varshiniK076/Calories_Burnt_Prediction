import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('model_training')
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
import pickle

class REGRESSION:

    def linear_regression(X_train, y_train, X_test, y_test):
        try:
            #Linear Regression
            logger.info('Linear Regression')
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            logger.info(f'Intercept : {reg.intercept_}')
            logger.info(f'Coefficient : {reg.coef_}')
            y_train_pred_LR = reg.predict(X_train)
            y_test_pred_LR = reg.predict(X_test)
            c = pd.DataFrame({'y_train': y_train, 'y_train_pred_LR': y_train_pred_LR})
            logger.info(f'{c.sample(10)}')
            logger.info(f'Training Accuracy (r2_score) Using Linear Regression : {r2_score(y_train, y_train_pred_LR)}')
            logger.info(f'Training Loss (Mean_squared_error) Using Linear Regression : {mean_squared_error(y_train, y_train_pred_LR)}')
            logger.info(f'Test Accuracy (r2_score) Using Linear Regression : {r2_score(y_test, y_test_pred_LR)}')
            logger.info(f'Test Loss (Mean_squared_error) Using Linear Regression : {mean_squared_error(y_test, y_test_pred_LR)}')


            with open("calories.pkl", 'wb') as f:
                pickle.dump(reg, f)

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def gd(X_train, y_train, X_test, y_test):
        try:
            logger.info('Gradient Descent Regression')
            X_train_b = LR_MODEL.add_bias(X_train)
            X_test_b = LR_MODEL.add_bias(X_test)
            theta, cost_history = LR_MODEL.gradient_descent(
                X_train_b, y_train,
                lr=0.001,
                epochs=5000,
                l2=0.1
            )
            y_train_pred = X_train_b.dot(theta)
            y_test_pred = X_test_b.dot(theta)
            '''
            plt.plot(cost_history)
            plt.xlabel("Epochs")
            plt.ylabel("Cost")
            plt.title("Gradient Descent Convergence")
            plt.show()
            '''
            logger.info(f'Gradient Descent Coefficients: {theta}')
            logger.info(f'Gradient Descent Train Accuracy (r2_score): {r2_score(y_train, y_train_pred)}')
            logger.info(f'Gradient Descent Test R2: {r2_score(y_test, y_test_pred)}')

            adj_r2 = LR_MODEL.r2_adjusted(y_train, y_train_pred, X_train.shape[1])
            logger.info(f'Gradient Descent Adjusted R2: {adj_r2}')

            logger.info(f'Final Cost: {cost_history[-1]}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    @staticmethod
    def add_bias(X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    @staticmethod
    def cost_function(X, y, theta):
        m = len(y)
        y_pred = X.dot(theta)
        return (1 / (2 * m)) * np.sum((y_pred - y) ** 2)

    @staticmethod
    def gradient_descent(X, y, lr=0.01, epochs=1000, l2=0.0):
        m, n = X.shape
        theta = np.zeros(n)
        cost_history = []

        for _ in range(epochs):
            y_pred = X.dot(theta)
            error = y_pred - y

            gradient = (1 / m) * X.T.dot(error)
            gradient[1:] += (l2 / m) * theta[1:]
            theta -= lr * gradient

            cost = LR_MODEL.cost_function(X, y, theta)
            cost_history.append(cost)

        return theta, cost_history

    @staticmethod
    def r2_adjusted(y, y_pred, p):
        r2 = r2_score(y, y_pred)
        n = len(y)
        return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)