#run this file and can get json file for prediction for PCR, PLS
#model inside a json file
#documentation: https://github.com/JWarmenhoven/ISLR-python/blob/master/Notebooks/Chapter%206.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, classification_report,\
                            confusion_matrix
from sklearn.cross_decomposition import PLSRegression

from sklearn.linear_model import LogisticRegression
from pylab import rcParams
from sklearn.preprocessing import scale 
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
                    (DF['time'].str.startswith("2018-02-14")) | \
                    (DF['time'].str.startswith("2018-02-22")) | \
                    (DF['time'].str.startswith("2018-02-23")) | \
                    (DF['time'].str.startswith("2018-02-26")) 

TIME_FILTER_TEST = (DF['time'].str.startswith("2018-02-28")) | \
                   (DF['time'].str.startswith("2018-03-02")) | \
                   (DF['time'].str.startswith("2018-03-05")) | \
                   (DF['time'].str.startswith("2018-02-27")) | \
                   (DF['time'].str.startswith("2018-03-06")) | \
                   (DF['time'].str.startswith("2018-03-07"))

TRAIN_DF = DF[TIME_FILTER_TRAIN].reset_index(drop = True)
TEST_DF = DF[TIME_FILTER_TEST].reset_index(drop = True)

#split X and y dataframe, hardcoding y variable is 'did_price_033'
X = DF[X_COL]
Y = DF['did_price_033']
X_TRAIN = TRAIN_DF[X_COL]
Y_TRAIN = TRAIN_DF['did_price_033']
X_TEST = TEST_DF[X_COL]
Y_TEST = TEST_DF['did_price_033']
TEST_IN = TEST_DF['price_increase']

#PCR
class PCR_mod:

    def __init__(self, k, x_train, y_train, x_test, y_test, test_in):
        '''
        Construct a principal component logistic regression model on the 
        data passed. 

        Input:
          k: integer, >=2, parameter for cross validation to calculate 
             mean square error for each principal component
          x_train: dataframe, training set for predictors
          y_train: series, training set for dependet variable
          x_test: dataframe, testing set for predictors
          y_test: series, testing set for dependent variable
          test_in: series, price percentage increase in testing set

        Return:
          mse: list, the list of mean square error for each principal 
               component
          best_n: the number of component with least mean square error
          min_error: the minimum mean square error
          y_pred: series, best prediction made based on all given attributes
          confusion_matrix: numpy array, the confusion matrix for y_pred and 
                            y_test
          report: string, the classification report 
          prot_shape: integer, the size of portfolio predicted from testing 
                      set
          exp_rv: float, the expected percentage increase from the portfolio
        '''

        self.k_fold = k
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.test_in = test_in
        self.mse, self.best_n, self.min_error = self.find_best_pca(x_train, \
                                                                   y_train)
        self.graph = self.draw_pca_graph(self.mse)
        self.y_pred = self.get_prediction(self.best_n, x_train, y_train, 
        	                              x_test)
        self.confusion_matrix = self.get_confusion_matrix(self.y_test, \
                              self.y_pred)
        self.report = self.get_precision_table(self.y_test, self.y_pred)
        self.port_shape, self.exp_rv = \
                    self.get_portfolio_detail(self.test_in, self.y_pred)


    #'MODIFIED'
    def find_best_pca(self, x_train, y_train):
        '''
        Get the number of componet which give minumum mean square error
        from the data for x and y

        Input:
          x_trian: x_train: dataframe, the data for x training set
          y_train: series, the data for y training set

        Return: list, integer
        '''
        pca = PCA()
        x_reduced_train = pca.fit_transform(scale(x_train))
        n = len(x_reduced_train)
        kf = KFold(n_splits= self.k_fold, shuffle=False, random_state=1)
        mse = []
        log = LogisticRegression(solver='newton-cg')    
        score = -1 * cross_val_score(log, np.ones((n,1)), y_train, cv=kf, \
                                     scoring='neg_mean_squared_error').mean()    
        mse.append(score) 
        for i in np.arange(1, x_train.shape[1] + 1):
            score = abs(cross_val_score(log, x_reduced_train[:,:i], y_train, \
                        cv=kf, scoring='neg_mean_squared_error')).mean()
            mse.append(score)
        min_error = min(mse)
    
        return mse, mse.index(min_error), min_error  
    
    #"MODIFIED"
    def get_prediction(self, num, x_train, y_train, x_test):
        '''
        Get the prediction based from the training set and testing set

        Input:
          num: integer, number of principal components
          x_train: dataframe, the data for x training set
          y_train: series, the data for x training set
          x_test: dataframe, the data for x testing set

        Return: series
        '''
        pca = PCA()
        x_reduced_train = pca.fit_transform(scale(x_train))
        x_reduced_test = pca.transform(scale(x_test))[:,:num]
        log = LogisticRegression(solver='newton-cg')
        model = log.fit(x_reduced_train[:,:num], y_train)
        y_pred = log.predict(x_reduced_test)

        return y_pred

    #'MODIFIED'
    def draw_pca_graph(self, mse):
        '''
        Given the list of mean square error and plot it

        Input:
          mse: list, mean square errors of each component

        Return
        '''

        plt.plot(np.array(mse), '-v')
        plt.xlabel('Number of principal components in regression')
        plt.ylabel('MSE')
        plt.title('Pricipal component and mean square errors')
        plt.xlim(xmin=-1)
        plt.show()
    
        return        



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
        shape = y_port.shape[0]

        return y_port.shape[0], y_port.mean()


#write pcr prediction to json file, you can harcode y_pred
#by using get_prediction function and setting differnt number of pc
PCR = PCR_mod(10, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, TEST_IN)
y_pred_pcr = PCR.y_pred
PCR_mod = pd.DataFrame(y_pred_pcr, columns = ['PCR'])
PCR_mod.to_json('PCR_mod.json', orient='values')




#PLS
class PLS_mod:

    def __init__(self, k, x_train, y_train, x_test, y_test, test_in):
        '''
        Construct a partial least square regression model on the 
        data passed. 

        Input:
          k: integer, >=2, parameter for cross validation to calculate 
             mean square error for each principal component
          x_train: dataframe, training set for predictors
          y_train: series, training set for dependet variable
          x_test: dataframe, testing set for predictors
          y_test: series, testing set for dependent variable
          test_in: series, price percentage increase in testing set

        Return:
          mse: list, the list of mean square error for each principal 
               component
          best_n: the number of component with least mean square error
          min_error: the minimum mean square error
          y_pred: series, best prediction made based on all given attributes
          confusion_matrix: numpy array, the confusion matrix for y_pred and 
                            y_test
          report: string, the classification report 
          prot_shape: integer, the size of portfolio predicted from testing 
                      set
          exp_rv: float, the expected percentage increase from the portfolio
        '''

        self.k_fold = k
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.test_in = test_in
        self.mse, self.best_n, self.min_error = self.find_best_pls(x_train, \
                                                                   y_train)
        self.graph = self.draw_pls_graph(self.mse)
        self.y_pred = self.get_prediction(self.best_n, x_train, y_train, \
        	                              x_test)
        self.port_shape, self.exp_rv = \
                    self.get_portfolio_detail(self.test_in, self.y_pred)


    #'MODIFIED'
    def find_best_pls(self, x_train, y_train):
        '''
        Get the number of componet which give minumum mean square error
        from the data for x and y

        Input:
          x_trian: x_train: dataframe, the data for x training set
          y_train: series, the data for y training set

        Return: list, integer
        '''

        kf_10 = KFold(n_splits= self.k_fold, shuffle=False, random_state=1)
        mse = []
        for i in np.arange(1, x_train.shape[1]):
            pls = PLSRegression(n_components=i)
            score = cross_val_score(pls, scale(x_train), y_train, \
                                cv=kf_10, scoring='neg_mean_squared_error').mean()
            mse.append(-score)
        min_error = min(mse)
    
        return mse, mse.index(min_error) + 1, min_error  

    
    #"MODIFIED"
    def get_prediction(self, num, x_train, y_train, x_test, y_test):
        '''
        Get the prediction based from the training set and testing set

        Input:
          num: integer, the number of principal components to make prediction
          x_train: dataframe, the data for x training set
          y_train: series, the data for x training set
          x_test: dataframe, the data for x testing set
          y_test: series, the data for x testing set

        Return: series
        '''
    
        pls = PLSRegression(n_components = num)
        pls.fit(scale(X_train), y_train)
        y_pred = pls.predict(scale(x_test))
        y_pred = np.reshape(y_pred, y_pred.shape[0])

        return y_pred
        '''
        pls = PLSRegression(n_components = n_pls + 1) #100
        pls.fit(scale(X_train), y_train)
        y_pred_pls = pls.predict(scale(X_test))
        y_pred_pls = np.reshape(y_pred_pls, y_pred_pls.shape[0])
        y_port = y_test_in[y_pred_pls >= .5]
        '''

    #'MODIFIED'
    def draw_pls_graph(self, mse):
        '''
        Given the list of mean square error and plot it

        Input:
          mse: list, mean square errors of each component

        Return
        '''

        plt.plot(np.array(mse), '-v')
        plt.xlabel('Number of principal components in regression')
        plt.ylabel('MSE')
        plt.title('Pricipal component and mean square errors')
        plt.xlim(xmin=-1)
        plt.show()
    
        return        


    def get_portfolio_detail(self, test_in, y_pred):
        '''
        Given the price percentage increase for testing set and prediction, 
        return the size of portfolio and expected return. 

        Input:
          test_in: series, price percentage increase for testing set
          y_pred: series, prediction for buying or not 

        Return: tuple
        '''
        
        y_port = test_in[y_pred_pls >= .5]

        return y_port.shape[0], y_port.mean()

'''
#write pls prediction to json file, you can hardcode number of principal
#component by using get_prediction function
PLS = PLS_mod(10, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, TEST_IN)
y_pred_pls = PLS.y_pred
PLS_mod = pd.DataFrame(y_pred_pls, columns = ['PLS'])
PLS_mod['PLS'] = (PLS_mod['PLS'] >= 0.5).astype(int)
PLS_mod.to_json('PLS_mod.json', orient='values')
'''





