import numpy as np
import pandas as pd
import json
from sklearn.tree import DecisionTreeClassifier

class Tree_model:
    
    def __init__(self, max_depth, min_samples, x_test):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.x_test = x_test
        self.y_pred = self.best_prediction(x_test)
    
    def best_prediction(df):
        return ((df["price_volatility"] > 0.5) & \
                (df["average_price_volatility"] <= 0) &\
                (df["alpha"] <= -1) & \
                (df["variation_2_sq"] <= 200)).astype(int)
