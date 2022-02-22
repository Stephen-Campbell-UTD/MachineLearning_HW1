# %%
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# %%


def extract_wdbc_data(path):
    df = pd.read_csv(path, sep=',', header=None)
    y = df.loc[:, 0]
    X = df.loc[:, 1:]
    return y, X


trn_y, trn_X = extract_wdbc_data('data/wdbc_trn.csv')
tst_y, tst_X = extract_wdbc_data('data/wdbc_trn.csv')
val_y, val_X = extract_wdbc_data('data/wdbc_trn.csv')

# %% [markdown]
#  (i)
# Create a Logistic Regression (LogReg) model using all the features to perform
# binary classification # on this dataset.

# Use the code in the notebook file on eLearning for Logistic Regression as a hint.

# Split the data into training and test sets and use the training set to train
# your model.

# Print the recall and precision values on test the test set. Use the test set to plot confusion matrix as well.
# (hint: SVM example on eLearning)
# %%


def part_i(clf, model_name):
    clf.fit(trn_X, trn_y)
    trn_y_pred = clf.predict(trn_X)
    trn_precision, trn_recall, _, _ = precision_recall_fscore_support(
        trn_y, trn_y_pred, labels=clf.classes_)

    print(model_name)
    print('training labels', clf.classes_)
    print('training precision', trn_precision)
    print('training recall', trn_recall)

    trn_confusion_matrix = confusion_matrix(
        trn_y, trn_y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=trn_confusion_matrix, display_labels=clf.classes_)
    disp.plot()


part_i(LogisticRegression(), 'Logistic Regression')

# %% [markdown]
# (ii) Create 30 LogReg models by incrementally changing the number of features.
# %%


def part_ii(clf, model_name):
    trn_precisions = []
    trn_recalls = []
    num_features = []
    for i in range(30):
        trn_X_sliced = trn_X.iloc[:, :i+1]
        clf.fit(trn_X_sliced, trn_y)
        trn_y_pred = clf.predict(trn_X_sliced)
        trn_precision, trn_recall, _, _ = precision_recall_fscore_support(
            trn_y, trn_y_pred, labels=clf.classes_, average='macro')
        num_features.append(i+1)
        trn_recalls.append(trn_recall)
        trn_precisions.append(trn_precision)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    ax1.plot(num_features, trn_precisions)
    ax1.set_title('Precision')
    ax2.plot(num_features, trn_recalls)
    ax2.set_title('Recall')
    ax2.set_xlabel('Number of Features')
    fig.suptitle(f'{model_name}\nMetrics vs. Number of Features', fontsize=16)


part_ii(LogisticRegression(), 'Logistic Regression')
# %%

# %% [markdown]

# (iii) Repeat (i) and (ii) using an SVM classifier with linear kernel
# %%
part_i(svm.SVC(kernel='linear'), 'SVM - Linear Kernel')
part_ii(svm.SVC(kernel='linear'), 'SVM - Linear Kernel')
# %%

# %% [markdown]
# (iv) Repeat (i) and (ii) using an SVM classifier with gaussian kernel

part_i(svm.SVC(kernel='rbf'), 'SVM - Gaussian Kernel')
part_ii(svm.SVC(kernel='rbf'), 'SVM - Guassian Kernel')
# %%
