from sklearn.datasets import load_breast_cancer
import pandas as pd

pd.set_option('display.max_columns', 35)

data, target = load_breast_cancer(return_X_y = True, as_frame=True)

# See what the features are
print(data.columns)

# Get an impression of the target distribution in our data set to check that it is balanced
print((target.value_counts(normalize=True)))

# Distribution  of features
print(data.describe())

# Look at the dtypes. Comment: No NaNs. All values are floats, so no encoding needed. 
print(data.info())

