#documentation: https://github.com/JWarmenhoven/ISLR-python/blob/master/Notebooks/Chapter%206.ipynb

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

#SVM
class SVM_mod:

    def __init__(self, params, x_train, y_train, x_valid, y_valid, x_test, \
                 y_test, test_in):
        '''
        Construct a SVM model on the data passed. 

        Input:
          params: a list with a dictionary inside mapping C, a list of floats,
                  which is penalty for support vectors in the wrong side, 
                  and gamma, list of float, which is negatively related 
                  with variance of sv
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
          best_C: float, the best c for least mean square error
          best_gamma: float, the best gamma for least mean square error
          y_pred: series, best prediction made based on all given attributes
          confusion_matrix: numpy array, the confusion matrix for y_pred and 
                            y_test
          report: string, the classification report 
          prot_shape: integer, the size of portfolio predicted from testing 
                      set
          exp_rv: float, the expected percentage increase from the portfolio
        '''
        
        self.params = params
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
        self.best_params, self.min_error = self.get_best_para(params, \
        	                    x_train, y_train, x_valid, y_valid)
        self.best_C = self.best_params['C']
        self.best_gamma = self.best_params['gamma']
        self.y_pred = self.get_prediction(self.best_C, self.best_gamma, \
        	                              self.x_total_train, \
        	                              self.y_total_train, x_test)
        self.confusion_matrix = self.get_confusion_matrix(self.y_test, \
        	                    self.y_pred)
        self.report = self.get_precision_table(self.y_test, self.y_pred)
        self.port_shape, self.exp_rv = self.get_portfolio_detail(self.test_in,\
        	                           self.y_pred)

    
    #'MODIFIED'(all others original)
    def get_best_para(self, params, x_train, y_train, x_valid, y_valid):
        '''
        Get the best C and gamma and store them in a dictionary. 

        Input:
          params: a list storing a dictionary mapping lists of potentail 
                  Cs and gammas
          x_train: dataframe, the data for x training set
          y_train: series, the data for y training set
          x_valid: dataframe, the data for x validation set
          y_valid: series, the data for y validation set

        Return: a dictionary
        '''

        svm = GridSearchCV(SVC(kernel='rbf'), params, cv=10, \
        	  scoring='accuracy', return_train_score=True)
        svm.fit(x_train, y_train)
        rv = svm.best_params_
        min_error =  1 - svm.best_estimator_.score(x_valid, y_valid)

        return rv, min_error   


    def get_prediction(self, C_, gamma_, x_train, y_train, x_test):
        '''
        Given the parameters and data for trainig and testing
        set, make prediction.

        Input:
          C: float, the C parameter in rbf svm
          gamma: float, the gamma parameter in rbf svm
          x_train: dataframe, training set for x
          y_train: series, training set for y
          x_test: dataframe, testing set for x

        Return: series
        '''

        svm = SVC(C = C_, kernel = 'rbf', gamma = gamma_)
        svm.fit(x_train, y_train)
        y_pred = svm.predict(x_test)

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


#write best prediction to file
#however, the best prediction might all be 0
#you can hard code to get prediction by using the get_prediction method
par_list = [{'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.5, 1, 2, 3, 4]}]
svm = SVM_mod(par_list, X_TRAIN, Y_TRAIN, X_VALID, Y_VALID, X_TEST, Y_TEST, \
	          TEST_IN)
model = pd.DataFrame(svm.y_pred, columns = ['SVM'])
model.to_json('SVM_mod.json', orient='values')
