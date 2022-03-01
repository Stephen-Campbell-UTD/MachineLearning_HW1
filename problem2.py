# %%
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from util import save_fig_to_images
# %%
X, y = load_boston(return_X_y=True)
NUM_TST = 101
trn_X = X[:-NUM_TST, :]
trn_y = y[:-NUM_TST]
tst_X = X[-NUM_TST:, :]
tst_y = y[-NUM_TST:]

# %% [markdown]
# Create a simple Linear Regression model using only the first feature. Print
# the accuracy (mse) and  plot the results similar to the example found below
#  (you will see this example in the class):
# %%


def determine_mse_for_n_features(n):
    clf = LinearRegression().fit(trn_X[:, :n], trn_y)
    tst_pred_y = clf.predict(tst_X[:, :n])
    mse = mean_squared_error(tst_y, tst_pred_y)
    return mse


print(f'MSE using 1 feature : {determine_mse_for_n_features(1)}')
# %%

print(f'MSE using 13 features : {determine_mse_for_n_features(13)}')

# %%


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


part_iii()

# %%
