# %%
from sklearn import svm
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


from util import save_fig_to_images
# %%
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

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
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(X_train, y_train)
    y_test_pred = pipe.predict(X_test)
    test_precision, tst_recall, _, _ = precision_recall_fscore_support(
        y_test, y_test_pred, labels=pipe.classes_)

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


part_i(LogisticRegression(), 'Logistic Regression')

# %% [markdown]
# (ii) Create 30 LogReg models by incrementally changing the number of features.
# %%


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
part_ii(svm.SVC(kernel='rbf'), 'SVM - Gaussian Kernel')
# %%
