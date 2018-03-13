import numpy as np
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, confusion_matrix, \
                            classification_report
from sklearn import neighbors

class KNN_model:

    def __init__(self, max_num, x_train, y_train, x_valid, y_valid, x_test, \
                 y_test, test_in):
        '''
        Construct a KNN model on the data passed. 

        Attributes:
          max_num: integer, maximum number of neighbour (arbitary)
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
          best_n: integer, the number of neighbours to get minimum error rate 
                  for y = 1 comparing training set and validation set
          min_error: float, the minimum error rate mentioned above
          y_pred: series, best prediction made based on all given attributes
          confusion_matrix: numpy array, the confusion matrix for y_pred and 
                            y_test
          report: string, the classification report 
          prot_shape: integer, the size of portfolio predicted from testing 
                      set
          exp_rv: float, the expected percentage increase from the portfolio
        '''
        
        self.max_num = max_num
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test
        self.test_in = test_in
        self.x_total_train = pd.concat([x_train, x_valid]).reset_index(drop = \
        	                                                           True)
        self.y_total_train = pd.concat([y_train, y_valid]).reset_index(drop = \
        	                                                           True)
        self.best_n, self.min_error = self.get_best_neighbour(max_num, \
        	                          x_train, y_train, x_valid, y_valid)
        self.y_pred = self.get_prediction(self.best_n, self.x_total_train, \
        	                              self.y_total_train, x_test)
        self.confusion_matrix = self.get_confusion_matrix(self.y_test, \
        	                    self.y_pred)
        self.report = self.get_precision_table(self.y_test, self.y_pred)
        self.port_shape, self.exp_rv = self.get_portfolio_detail(self.test_in,\
        	                           self.y_pred)


    def get_neighbour_mean(self, num, x_train, y_train, x_valid, y_valid):
        '''
        Return the error rate of y = 1 for the validation set
        given the number of neighbours defined in KNN method. 

        Input:
          num: integer, from 1 to 100
          x_train: dataframe, the data for x training set
          y_train: series, the data for y training set
          x_valid: dataframe, the data for x validation set
          y_valid: series, the data for y validation set

        Return: float
        '''

        knn = neighbors.KNeighborsClassifier(n_neighbors = num)
        y_pred = knn.fit(x_train, y_train).predict(x_valid)
        error = (y_valid != y_pred).astype(int)
        rv = error[y_pred == 1].mean()
    
        return rv    


    def get_best_neighbour(self, num, x_train, y_train, x_valid, y_valid):
        '''
        Return the number of neighbours with the minimum error rates of y = 1 
        for the validation set given the maximum number of neighbours, 
        training set and validation set.

        Input: 
          num: integer, the maximum number of neighbours
          x_train: dataframe, training set for x
          y_train: series, training set for y
          x_test: dataframe, testing set for x
          y_test: series, testing set for y

        Return: tuple, number of mimimum error rates and the minimum error 
                rates.
        '''

        rv_min = 1
        min_id = 1
        for i in range(1, num + 1):
            value = self.get_neighbour_mean(i, x_train, y_train, \
            	                       x_valid, y_valid)
            if (value != None) and (value < rv_min):
                min_id = i
                rv_min = value
            #print(i, value) you can print out the result for detailed info
        return min_id, rv_min


    def get_prediction(self, num, x_train, y_train, x_test):
        '''
        Given the number of neighbours, training and testing sets to get the 
        prediction for y.

        Input:
          num: integer, the maximum number of neighbours
          x_train: dataframe, training set for x
          y_train: series, training set for y
          x_test: dataframe, testing set for x

        Return: series
        '''

        knn = neighbors.KNeighborsClassifier(n_neighbors = num)
        y_pred = knn.fit(x_train, y_train).predict(x_test)

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
