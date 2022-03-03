# File Name:  problem2.py
# Author:     Stephen Campbell, Alexandra Danhof, Kristi Doleh
# NetID:      sac170630, and170130, kxd170004
# Team:       ML Group 4
# Date:       03/03/2022
# Version:    Python 3.9
#
# Copyright:  2022, All Rights Reserved
#
# Description:
#    Written for ML Group 4 Homework 1 Problem 2.
#    This file reads in a dataset for Boston housing prices
#    and performs prediction using Linear Regression using the
#    13 features. The program will create a simple model using
#    only a single feature before moving to create multiple
#    regression models using a combination of the given features.
#    The program will display the mse vs features curves for the
#    different models.
#


# import all necessary files
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from util import save_fig_to_images

# load necessary data for dataset into training and
# test sets
X, y = load_boston(return_X_y=True)

trn_X, tst_X, trn_y, tst_y = train_test_split(X, y, random_state=0)


#
# Function Name: determine_mse_for_n_features
# Description: Function creating Linear Regression model
# using n amount of features. It will display the
# accuracy (mse) and plot results.
# Inputs: features n
# Outputs: accuracy (mse) and plot results
#
def determine_mse_for_n_features(n):
    clf = LinearRegression().fit(trn_X[:, :n], trn_y)
    tst_pred_y = clf.predict(tst_X[:, :n])
    mse = mean_squared_error(tst_y, tst_pred_y)
    return mse


# function call to get results from part 1
print(f'MSE using 1 feature : {determine_mse_for_n_features(1)}')


# function call to get results from part 2
print(f'MSE using 13 features : {determine_mse_for_n_features(13)}')


#
# Function Name: part_iii
# Description: Function creating 13 MLR models
# by incrementally changing number of features.
# It will display the accuracy (mse) vs features curve.
# Inputs: none
# Outputs: mse vs features curve
#
def part_iii():
    num_features = list(range(1, 14))
    mse_array = [determine_mse_for_n_features(i) for i in num_features]
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_title(
        'Boston Housing Market\nMean Squared Error vs. Number of Features')
    ax.scatter(num_features, mse_array)
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Mean Squared Error')
    save_fig_to_images('MSEvsNumFeatures.png')


# function call to get results from part 3
part_iii()
