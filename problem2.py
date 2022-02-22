# %%
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
# %%
X, y = load_boston(return_X_y=True)
NUM_TST = 101
trn_X = X[:-NUM_TST]
trn_y = y[:-NUM_TST]
tst_X = X[-NUM_TST:]
tst_y = y[-NUM_TST:]

# %% [markdown]
# Create a simple Linear Regression model using only the first feature. Print
# the accuracy (mse) and  plot the results similar to the example found below
#  (you will see this example in the class):
# %%
clf = LinearRegression().fit(trn_X[:, 0], trn_y)

# %%