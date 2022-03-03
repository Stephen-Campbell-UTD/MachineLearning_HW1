# File Name:  problem1.py
# Author:     Stephen Campbell, Alexandra Danhof, Kristi Doleh
# NetID:      sac170630, and170130, kxd170004
# Team:       ML Group 4
# Date:       03/03/2022
# Version:    Python 3.9
#
# Copyright:  2022, All Rights Reserved
#
# Description:
#    Written for ML Group 4 Homework 1 Problem 1.
#    This file reads in a dataset for Wisconson Breast
#    Cancer and divides into training, test, and validation sets.
#    A Logistical Regression (LogReg) model is created to perform binary
#    classification on the dataset. The program then repeat this
#    process using an SVM classifier with both linear and gaussian
#    kernels.
#


# import all necessary files
from sklearn import svm
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from util import save_fig_to_images


# load necessary data for dataset into training and
# test sets
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


#
# Function Name: part_i
# Description: Function creating LogReg model to
# perform binary classification and print the
# recall/precision values and confusion matrix.
# Inputs: clf and name of model
# Outputs: recall/precision values and confusion matrix
#
def part_i(clf, model_name):
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(X_train, y_train)
    y_test_pred = pipe.predict(X_test)
    test_precision, tst_recall, _, _ = precision_recall_fscore_support(
        y_test, y_test_pred, labels=pipe.classes_, average='binary')

    print(model_name)
    print('labels', pipe.classes_)
    print('Precision w/ Test Set:', test_precision)
    print('Recall w/ Test Set', tst_recall)

    tst_confusion_matrix = confusion_matrix(
        y_test, y_test_pred, labels=pipe.classes_)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(
        confusion_matrix=tst_confusion_matrix, display_labels=pipe.classes_)
    ax.set_title(f'Confusion Matrix of {model_name}')
    disp.plot(ax=ax)
    save_fig_to_images(f'ConfusionMatrix-{model_name}.png')


# function call to get results from part 1
part_i(LogisticRegression(), 'Logistic Regression')


#
# Function Name: part_ii
# Description: Function creating 30 LogReg model
# by incrementally changing the number of features.
# Then plotting the recall vs features & precision vs
# features plots.
# Inputs: clf and name of model
# Outputs: plots for recall vs features and precision vs features
#
def part_ii(clf, model_name):
    precisions = []
    recalls = []
    num_features = []
    for i in range(30):
        X_train_sliced = X_train[:, :i+1]
        X_test_sliced = X_test[:, :i+1]
        pipe = make_pipeline(StandardScaler(), clf)
        pipe.fit(X_train_sliced, y_train)
        y_test_pred = pipe.predict(X_test_sliced)
        precision, recall, _, _ = precision_recall_fscore_support(
            y_test, y_test_pred, labels=pipe.classes_, average='binary')
        num_features.append(i+1)
        recalls.append(recall)
        precisions.append(precision)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    ax1.plot(num_features, precisions)
    ax1.grid(True)
    ax1.set_title('Precision')
    ax2.plot(num_features, recalls)
    ax2.set_title('Recall')
    ax2.grid(True)
    ax2.set_xlabel('Number of Features')
    fig.suptitle(f'{model_name}\nMetrics vs. Number of Features', fontsize=16)
    save_fig_to_images(f'MetricVsFeatures-{model_name}.png')


# function call to get results from part 2
part_ii(LogisticRegression(), 'Logistic Regression')


# (iii) Repeat (i) and (ii) using an SVM classifier with linear kernel

# call both part 1 and part 2 with linear kernel
part_i(svm.SVC(kernel='linear'), 'SVM - Linear Kernel')
part_ii(svm.SVC(kernel='linear'), 'SVM - Linear Kernel')
# %%


# (iv) Repeat (i) and (ii) using an SVM classifier with gaussian kernel

# call both part 1 and part 2 with linear kernel
part_i(svm.SVC(kernel='rbf'), 'SVM - Gaussian Kernel')
part_ii(svm.SVC(kernel='rbf'), 'SVM - Gaussian Kernel')
