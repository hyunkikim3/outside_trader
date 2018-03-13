import numpy as np
import pandas as pd
import json
from sklearn.metrics import classification_report, confusion_matrix, \
                            mean_squared_error
from sklearn.ensemble import RandomForestClassifier, \
                            GradientBoostingClassifier, BaggingClassifier

class BST_model:

    def __init__(self, x_train, y_train, x_valid, y_valid, x_test, \
                 y_test, test_in):
        '''
        Construct a boosting model on the data passed. 

        Attributes:
          x_train: dataframe, training set for predictors
          y_train: series, training set for dependet variable
          x_valid: dataframe, validation set for predictors
          y_valid: series, validation set for dependent variable
          x_test: dataframe, testing set for predictors
          y_test: series, testing set for dependent variable
          test_in: series, price percentage increase in testing set
          x_total_train: the whole 'training' set for x to get prediction 
                         combining training set and validation set
          x_total_train: the whole 'training' set for y to get prediction 
                         combining training set and validation set
          var_importance: a seires recording the importance of variables 
                          in boosting model's learning process
          y_pred: series, best prediction made based on all given attributes
          confusion_matrix: numpy array, the confusion matrix for y_pred and 
                            y_test
          report: string, the classification report 
          prot_shape: integer, the size of portfolio predicted from testing 
                      set
          exp_rv: float, the expected percentage increase from the portfolio
        '''

        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test
        self.test_in = test_in
        self.x_total_train = pd.concat([x_train, x_valid]).reset_index(drop=\
                                                                     True)
        self.y_total_train = pd.concat([y_train, y_valid]).reset_index(drop=\
                                                                     True)
        
        self.var_importance = self.get_importance(x_train, y_train)
        self.y_pred = self.get_prediction(self.x_total_train, \
                                          self.y_total_train, x_test)
        self.confusion_matrix = self.get_confusion_matrix(self.y_test, \
                              self.y_pred)
        self.report = self.get_precision_table(self.y_test, self.y_pred)
        self.port_shape, self.exp_rv = \
                   self.get_portfolio_detail(self.test_in, self.y_pred)


    def get_importance(self, x_train, y_train):
        '''
        Return a series recording importance of x variables from the training
        set. 

        Input:
          x_train: dataframe, training set for x
          y_train: series, training set for y

        Return: a series
        '''

        boost = GradientBoostingClassifier(n_estimators = 500, \
                                           learning_rate = 0.01, \
                                           max_depth = 2, random_state=1)
        boost.fit(x_train, y_train)
        feature_importance = boost.feature_importances_*100
        var_impor = pd.Series(feature_importance, index = \
                              x_train.columns).sort_values(inplace=False)

        return var_impor  


    def get_prediction(self, x_train, y_train, x_test):
        '''
        Given the number of max_samples, training and testing sets to
        get the prediction for y.

        Input:
          x_train: dataframe, training set for x
          y_train: series, training set for y
          x_test: dataframe, testing set for x

        Return: series
        '''

        boost = GradientBoostingClassifier(n_estimators = 500, \
                                           learning_rate = 0.01, \
                                           max_depth = 2, random_state=1)
        boost.fit(x_train, y_train)
        y_pred = boost.predict(x_test)

        return y_pred

    def get_confusion_matrix(self, y_test, y_pred):
        '''
        Given the testing set and prediction for y (categorical) and return
        a confusion matrix.

        Input:
          y_test: series, testing set for y
          y_pred: series, prediction for y

        Return: numpy array
        '''

        result_matrix = confusion_matrix(y_test, y_pred)

        return result_matrix

    def get_precision_table(self, y_test, y_pred):
        '''
        Given the testing set and prediction for y (categorical) and return
        print out a classification report table.

        Input:
          y_test: series, testing set for y
          y_pred: series, prediction for y

        Return
        '''
        table = classification_report(y_test, y_pred)

        return table

    def get_portfolio_detail(self, test_in, y_pred):
        '''
        Given the price percentage increase for testing set and prediction, 
        return the size of portfolio and expected return. 

        Input:
          test_in: series, price percentage increase for testing set
          y_pred: series, prediction for buying or not 

        Return: tuple
        '''
        
        y_port = test_in[y_pred == 1]

        return y_port.shape[0], y_port.mean()
