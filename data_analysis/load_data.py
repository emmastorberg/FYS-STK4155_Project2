from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', 35)

data, target = load_breast_cancer(return_X_y = True, as_frame=True) # set as_frame = True if you don´t want columns as series

target = target.map({1: 0, 0:1}) # We want B=0, M=1
# See what the features are 
print(data.columns)

# Get an impression of the target distribution in our data set to check that it is balanced
#print((target.value_counts(normalize=True)))
print(target.value_counts(normalize = True))

# Distribution  of features
#print(data.describe())

# Look at the dtypes. 
#print(data.info()) #Comment: No NaNs. All values are floats, so no encoding needed. 

# Correlation

# Calculate the pairwise correlation between each feature and the target
correlation_with_target = data.apply(lambda x: x.corr(target))

# Plotting the heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_with_target.to_frame(), annot=True, cmap='coolwarm', cbar=True)
plt.title('Correlation of Features with Target')
#plt.show()
