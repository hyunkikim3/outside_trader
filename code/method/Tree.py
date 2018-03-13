import numpy as np
import pandas as pd
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
import graphviz

class Tree_model:
    
    def __init__(self, max_depth, min_samples, x_train, y_train, \
                 x_test, y_test):
        '''
        Construct a binary tree model for the training and testing sets
        
        Attributes:
          max_depth: the maximum depth of the tree
          min_samples: the minimum number of observations in each branch
          x_train: dataframe, the training set for x
          y_train: series, the training set for y
          x_test: dataframe, the testing set for x
          y_test: series, the testing set for y
        '''
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.y_pred = self.get_best_prediction(x_test)
    
    def get_prediction(self, x_train, y_train, x_test):
        '''
        Get the prediction based on the model.
        
        Inputs:
          x_train: dataframe, training set for x
          y_test: series, training set for y
          x_test: dataframe, testing set for x
        
        Return: series
        '''
        df_tree = DecisionTreeClassifier(max_depth=self.max_depth, \
                                         min_samples_leaf=self.min_samples)
        df_tree.fit(X_train, y_train)
        
        return df_tree.predict(X_test)
    
    def get_mean_squared_error(self, y_test, y_pred):
        '''
        Get the mean square error of y for testing set and y prediction
        
        Inputs:
          y_test: series
          y_pred: series
          
        Return: float
        '''
        return mean_squared_error(y_test, y_pred)
        
    def get_best_prediction(self, df):
        '''
        Get the best prediction we modified from looking at previous results
        
        Input:
          df: a dataframe
        
        Return: a dataframe
        '''
        return ((df["price_volatility"] > 0.5) & \
                (df["average_price_volatility"] <= 0) &\
                (df["alpha"] <= -1) & \
                (df["variation_2_sq"] <= 200)).astype(int)
    
    def visualize_tree(self, filename):
        '''
        Visualize the tree model and save it in a file
        
        Input:
          string: a file name
        
        Return: None
        '''
        df_tree_viz = export_graphviz(df_tree, out_file=None, \
                                      feature_names=X, \
                                      rounded=True, filled=True)
        graph = graphviz.Source(df_tree_viz)
        graph.render(filename)
        return None
