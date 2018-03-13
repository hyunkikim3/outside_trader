import numpy as np
import pandas as pd
import json
from sklearn.metrics import classification_report, confusion_matrix, \
                            mean_squared_error
from sklearn.ensemble import RandomForestClassifier, \
                            GradientBoostingClassifier, BaggingClassifier

#Bagging
class BAGGING_mod:

    def __init__(self, min_num, max_num, x_train, y_train, x_valid, y_valid, \
                 x_test, y_test, test_in):
        '''
        Construct a random forest model on the data passed. 

        Input:
          min_num: integer, minimum number of samples to draw in each estimator
          max_num: integer, maximum number of samples to draw in each estimator
          x_train: dataframe, training set for predictors
          y_train: series, training set for dependet variable
          x_valid: dataframe, validation set for predictors
          y_valid: series, validation set for dependent variable
          x_test: dataframe, testing set for predictors
          y_test: series, testing set for dependent variable
          test_in: series, price percentage increase in testing set

        Return:
          x_total_train: the whole 'training' set for x to get prediction 
                         combining training set and validation set
          x_total_train: the whole 'training' set for y to get prediction 
                         combining training set and validation set
          best_n: integer, the number of maximum feature to get minimum error 
                  rate for y = 1 comparing training set and validation set
          min_error: float, the minimum error rate mentioned above
          y_pred: series, best prediction made based on all given attributes
          confusion_matrix: numpy array, the confusion matrix for y_pred and 
                            y_test
          report: string, the classification report 
          prot_shape: integer, the size of portfolio predicted from testing 
                      set
          exp_rv: float, the expected percentage increase from the portfolio
        '''
        
        self.min_num = min_num
        self.max_num = max_num
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
        self.best_n, self.min_error = self.get_best_sample(min_num, max_num, \
                                                           x_train, y_train, \
                                                           x_valid, y_valid)
        self.y_pred = self.get_prediction(self.best_n, self.x_total_train, \
                                        self.y_total_train, x_test)
        self.confusion_matrix = self.get_confusion_matrix(self.y_test, \
                              self.y_pred)
        self.report = self.get_precision_table(self.y_test, self.y_pred)
        self.port_shape, self.exp_rv = \
                   self.get_portfolio_detail(self.test_in, self.y_pred)


    def get_best_sample(self, min_num, max_num, x_train, y_train, x_valid, \
                        y_valid):
        '''
        Return the number of sample size in the bagging model and the 
        smallest value of the error rate for y = 1 it gets from the 
        training set and validation set.

        Inputs:
          min_num: integer, the minimum number of sample size
          max_num: integer, the maximum number of sample size
          x_train: dataframe, the data for x training set
          y_train: series, the data for y training set
          x_valid: dataframe, the data for x validation set
          y_valid: series, the data for y validation set

        Return: tuple
        '''

        min_error = 1
        f_num = 1
        if (min_num < max_num) and (max_num < x_train.shape[0]):
            for i in range(min_num, max_num + 1):
                bag_tree = BaggingClassifier(n_estimators = 500, \
                                             max_samples = i,
                                             max_features = \
                                             x_train.shape[1], \
                                             bootstrap = True, \
                                             oob_score = True, \
                                             random_state = 0)
                bag_tree.fit(X_train, y_train)
                y_pred = bag_tree.predict(X_test)
                error = (y_test != y_pred).astype(int)
                avg = error[y_pred == 1].mean()
                if avg < min_error:
                    f_num = i
                    min_error = avg
        
        return f_num, min_error


    def get_prediction(self, num, x_train, y_train, x_test):
        '''
        Given the number of max_samples, training and testing sets to
        get the prediction for y.

        Input:
          num: integer, the maximum number of neighbours
          x_train: dataframe, training set for x
          y_train: series, training set for y
          x_test: dataframe, testing set for x

        Return: series
        '''

        bag_tree = BaggingClassifier(n_estimators = 500, \
                                     max_features = x_train.shape[1],
                                     max_samples = num, bootstrap = True, \
                                     oob_score = True, random_state = 0)
        bag_tree.fit(x_train, y_train)
        y_pred = bag_tree.predict(x_test)

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


BAG = BAGGING_mod(1, X_TRAIN.shape[0], X_TRAIN, Y_TRAIN, X_VALID, Y_VALID, \
                  X_TEST, Y_TEST, TEST_IN)
BAG_model = pd.DataFrame(BAG.y_pred, columns = ['BAGGING'])
BAG_model.to_json('BAGGING_mod.json', orient='values')
