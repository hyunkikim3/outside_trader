#run this file and can get json file for prediction for bese logistic
#model inside a json file
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, confusion_matrix, \
                            classification_report
from sklearn.linear_model import LogisticRegression
import json

with open("df_1hour_Feb.json", 'r', encoding='UTF-8') as f:
    F_data = json.load(f)

with open("df_1hour_Mar_07.json", 'r', encoding='UTF-8') as f:
    M_data = json.load(f)

COLUMNS = ['name', 'code', 'time', 'price', 'time_1', 'price_1', \
           'price_dif_1', 'sell_1', 'buy_1', 'volume_1', 'variation_1', \
           'post_num_1', 'unique_id_1', 'click_1', 'like_1', 'dislike_1', \
           'time_2', 'price_2', 'price_dif_2', 'sell_2', 'buy_2', \
           'volume_2', 'variation_2', 'post_num_2', 'unique_id_2', \
           'click_2', 'like_2', 'dislike_2', 'time_3', 'price_3', \
           'price_dif_3', 'sell_3', 'buy_3', 'volume_3', 'variation_3', \
           'post_num_3', 'unique_id_3', 'click_3', 'like_3', 'dislike_3', \
           'mkt_cap', 'kospi', 'kosdaq', 'trash', 'yesterday_closing_price', \
           'is_maximum', 'is_minimum', 'price_volatility', 'price_trend', \
           'average_price_volatility', 'sell_minus_buy_1', \
           'sell_minus_buy_2', 'sell_minus_buy_3', 'is_price_gap_stable', \
           'price_gap_volatility', 'is_like_higher', 'volume_trend', \
           'post_num_trend', 'unique_id_trend', 'click_trend', \
           'price_increase', 'did_price_increase', 'did_price_033', \
           'did_price_100', 'did_price_150', 'kospi_ind', 'kosdaq_ind', \
           'time_slot', 'ko_inter', 'early_mor', 'morning', 'lunch', \
           'afternoon', 'late', 'mkt_change', 'alpha', 'per_now', \
           'kospi_1', 'kospi_2', 'kospi_3', 'kospi_answer', 'kosdaq_1', \
           'kosdaq_2', 'kosdaq_3', 'kosdaq_answer', 'kospi_trend', \
           'kosdaq_trend', 'kospi_increase', 'kosdaq_increase', \
           'market_increase', 'did_opening_price_increase', 'price_1_sq', \
           'price_dif_1_sq', 'sell_1_sq', 'buy_1_sq', 'volume_1_sq', \
           'variation_1_sq', 'post_num_1_sq', 'unique_id_1_sq', \
           'click_1_sq', 'like_1_sq', 'dislike_1_sq', 'price_2_sq', \
           'price_dif_2_sq', 'sell_2_sq', 'buy_2_sq', 'volume_2_sq', \
           'variation_2_sq', 'post_num_2_sq', 'unique_id_2_sq', \
           'click_2_sq', 'like_2_sq', 'dislike_2_sq', 'price_3_sq', \
           'price_dif_3_sq', 'sell_3_sq', 'buy_3_sq', 'volume_3_sq', \
           'variation_3_sq', 'post_num_3_sq', 'unique_id_3_sq', \
           'click_3_sq', 'like_3_sq', 'dislike_3_sq', 'mkt_cap_sq', \
           'yesterday_closing_price_sq', 'price_volatility_sq', \
           'price_trend_sq', 'average_price_volatility_sq', \
           'sell_minus_buy_1_sq', 'sell_minus_buy_2_sq', \
           'sell_minus_buy_3_sq', 'price_gap_volatility_sq', \
           'volume_trend_sq', 'post_num_trend_sq', 'unique_id_trend_sq', \
           'click_trend_sq', 'kospi_ind_sq', 'kosdaq_ind_sq', \
           'time_slot_sq', 'ko_inter_sq', 'mkt_change_sq', 'alpha_sq', \
           'per_now_sq', 'kospi_1_sq', 'kospi_2_sq', 'kospi_3_sq', \
           'kosdaq_1_sq', 'kosdaq_2_sq', 'kosdaq_3_sq', 'kospi_trend_sq', \
           'kosdaq_trend_sq']

TO_DEL = ['name', 'code', 'time', 'price', 'time_1', 'time_2', 'time_3', \
          'price_increase', 'did_price_increase', 'did_price_033', \
          'did_price_100', 'did_price_150', 'kospi_answer', \
          'kosdaq_answer', 'kospi_increase', 'kosdaq_increase', \
          'market_increase']

X_COL = COLUMNS
for col in TO_DEL:
  X_COL = list(X_COL)
  X_COL.remove(col)

  
DF_F = pd.DataFrame(F_data, columns = COLUMNS)
DF_M = pd.DataFrame(M_data, columns = COLUMNS)
DF = pd.concat([DF_F, DF_M])
DF = DF.dropna(axis=0, how='any')

#define filter for trainig set, validation test and testing set
TIME_FILTER_TRAIN = (DF['time'].str.startswith("2018-02-21")) | \
                    (DF['time'].str.startswith("2018-02-20")) | \
                    (DF['time'].str.startswith("2018-02-14"))
        
TIME_FILTER_VALID = (DF['time'].str.startswith("2018-02-22")) | \
                    (DF['time'].str.startswith("2018-02-23")) | \
                    (DF['time'].str.startswith("2018-02-26")) 

TIME_FILTER_TEST = (DF['time'].str.startswith("2018-02-28")) | \
                   (DF['time'].str.startswith("2018-03-02")) | \
                   (DF['time'].str.startswith("2018-03-05")) | \
                   (DF['time'].str.startswith("2018-02-27")) | \
                   (DF['time'].str.startswith("2018-03-06")) | \
                   (DF['time'].str.startswith("2018-03-07"))

TRAIN_DF = DF[TIME_FILTER_TRAIN].reset_index(drop = True)
VALID_DF = DF[TIME_FILTER_VALID].reset_index(drop = True)
TEST_DF = DF[TIME_FILTER_TEST].reset_index(drop = True)

#split X and y dataframe, hardcoding y variable is 'did_price_033'
X = DF[X_COL]
Y = DF['did_price_033']
X_TRAIN = TRAIN_DF[X_COL]
Y_TRAIN = TRAIN_DF['did_price_033']
X_VALID = VALID_DF[X_COL]
Y_VALID = VALID_DF['did_price_033']
X_TEST = TEST_DF[X_COL]
Y_TEST = TEST_DF['did_price_033']
TEST_IN = TEST_DF['price_increase']
VALID_IN = VALID_DF['price_increase']


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
        while num < x_train.shape[0]:
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

'''
Logistic_mod = pd.DataFrame(prob, columns = ['Logistic'])
Logistic_mod.to_json('Logistic_mod.json', orient='values')
'''
