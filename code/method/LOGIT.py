import numpy as np
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, confusion_matrix, \
                            classification_report
from sklearn.linear_model import LogisticRegression

#logistic

class LOGISTIC_mod:

    def __init__(self, x_train, y_train, x_valid, y_valid, x_test, \
                 y_test, valid_in, test_in):
        '''
        Construct a logistic model on the data passed. 

        Input:
          x_train: dataframe, training set for predictors
          y_train: series, training set for dependet variable
          x_valid: dataframe, validation set for predictors
          y_valid: series, validation set for dependent variable
          x_test: dataframe, testing set for predictors
          y_test: series, testing set for dependent variable
          valid_in: series, price percentage increase in validation set
          test_in: series, price percentage increase in testing set

        Return:
          best_var: string, the variable which can get the lowest error rate
                    for y = 1validation set
          best_vars: list, the list of variables found out sequentially which 
                     can the lowest error rate for y = 1
          min_error: float, the minimum error rate mentioned above
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
        self.valid_in = valid_in
        self.test_in = test_in
        self.best_var = self.find_first_mod(x_train, y_train, x_valid, \
                                           y_valid, valid_in)
        self.best_vars = self.get_best_list([self.best_var], x_train, \
                                            y_train, x_valid, y_valid, \
                                            valid_in)

        self.y_pred = self.get_prediction(self.x_train[self.best_vars],\
                                          self.y_train, \
                                          x_test[self.best_vars])
        self.confusion_matrix = self.get_confusion_matrix(self.y_test, \
                              self.y_pred)
        self.report = self.get_precision_table(self.y_test, self.y_pred)
        self.port_shape, self.exp_rv = \
                    self.get_portfolio_detail(self.test_in, self.y_pred)


    def find_first_mod(self, x_train, y_train, x_valid, y_valid, valid_in):
        '''
        Return the oredictor which gets lowest error rate of y = 1 for the 
        validation set the dataframes, it's a simple linear model.

        Input:
          x_train: dataframe, the data for x training set
          y_train: series, the data for x training set
          x_valid: dataframe, the data for x validation set
          y_valid: series, the data for x validation set
          valid_in: series, price percentage increase in validation set

        Return: string
        '''
        rv = 1
        for var in x_train.columns:
            x_train_sub = x_train[var]
            log = LogisticRegression(solver='newton-cg')            
            log.fit(x_train_sub.reshape(-1, 1), y_train)
            y_pred = log.predict(x_valid[var].reshape(-1, 1))
            error = (y_valid != y_pred).astype(int)
            avg = error[y_pred == 1].mean()
            y_port = valid_in[y_pred == 1] 
            size = y_port.shape[0]
            if (avg < rv) and (size >= 10):
                max_rv = avg
                var_best = var
            
        return var_best


    def find_best_var(self, var_list, x_train, y_train, x_valid, \
                      y_valid, valid_in, mse_min=1):
        '''
        Return the variables not in the varlist with the lowest error rates 
        for y = 1 together with variables in var_list all put in the model, 
        and together with the smallest mean square error.

        Inputs:
          var_list: list, the list to add each variable not included to 
                    evaluate a new model
          x_train: dataframe, the data for x training set
          y_train: series, the data for y training set
          x_valid: dataframe, the data for x validation set
          y_valid: series, the data for y validation set
          valid_in: series, price percentage increase in validation set
          mse_min: float, the cut-off mean square error passed, the result 
                   should be smaller than mse_min, default = 1


        Return: tuple
        '''

        best_var = None
        for var in x_train.columns:
            if var not in var_list:
                var_list.append(var)
                x_train_sub = x_train[var_list]
                log = LogisticRegression(solver='newton-cg')            
                log.fit(x_train_sub, y_train)
                y_pred = log.predict(x_valid[var_list])
                y_port = valid_in[y_pred == 1]
                size = y_port.shape[0]
                if size >= 30:
                    error = (y_valid != y_pred).astype(int)
                    avg = error[y_pred == 1].mean()
                    if mse_min > avg:
                        best_var = var
                        mse_min = avg
                var_list = var_list[:-1]
    
        return best_var, mse_min     


    def get_best_list(self, var_list, x_train, y_train, x_valid, \
                      y_valid, valid_in, mse_min=1):
        '''
        Get the best list of variables which get lowest error rates for
        y = 1 given the training and validation set. 
        
        Inputs:
          var_list: list, the list to add each variable not included to 
                    evaluate a new model
          x_train: dataframe, the data for x training set
          y_train: series, the data for y training set
          x_valid: dataframe, the data for x validation set
          y_valid: series, the data for y validation set
          valid_in: series, price percentage increase in validation set
          mse_min: float, the cut-off mean square error passed, the result 
                   should be smaller than mse_min, default = 1

        Return: a list
        '''
    
        num = 0
        mse_min=1
        while num < x_train.shape[1]:
            best_var, mse_min_new = self.find_best_var(var_list, x_train, \
                                                       y_train, x_valid, \
                                                       y_valid, valid_in, \
                                                       mse_min)
            if best_var != None:
                if mse_min_new <= mse_min:
                    mse_min = mse_min_new
                    var_list.append(best_var)
                    print(var_list, num, mse_min)
                else:
                    return var_list
            else:
                return var_list
            num += 1

        return var_list


    def get_prediction(self, x_train, y_train, x_test):
        '''
        Given the training and testing sets to get the prediction for y.

        Input:

          x_train: dataframe, training set for x
          y_train: series, training set for y
          x_test: dataframe, testing set for x

        Return: series
        '''

        log = LogisticRegression(solver='newton-cg')
        log.fit(x_train, y_train)
        y_pred = log.predict(x_test)

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


logistic = LOGISTIC_mod(X_TRAIN, Y_TRAIN, X_VALID, Y_VALID, X_TEST, \
                        Y_TEST, VALID_IN, TEST_IN)
logistic_mod = pd.DataFrame(logistic.y_pred, columns = ['Logistic'])
logistic_mod.to_json('Logistic_mod.json', orient='values')
