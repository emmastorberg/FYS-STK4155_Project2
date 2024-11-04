from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
import git

pd.set_option('display.max_columns', 35)

data, target = load_breast_cancer(return_X_y = True, as_frame=True) # set as_frame = True if you don´t want columns as series

target = target.map({1: 0, 0:1}) # We want B=0, M=1
# See what the features are 
#print(data.columns)

# Get an impression of the target distribution in our data set to check that it is balanced
print((target.value_counts(normalize=True)))

# Distribution  of features
print(data.describe())

# Look at the dtypes. 
print(data.info()) #Comment: No NaNs. All values are floats, so no encoding needed. 

# Correlation

# Calculate the pairwise correlation between each feature and the target
correlation_with_target = data.apply(lambda x: x.corr(target))

print(correlation_with_target.to_frame().index)
# Plotting the heatmap
plt.figure(figsize=(10, 10))

ax = sns.heatmap(correlation_with_target.to_frame(), annot=True, cmap='coolwarm', xticklabels=False, yticklabels=correlation_with_target.to_frame().index)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')  # Rotate y-axis labels

plt.title('Correlation of features with target in breast cancer data')
plt.tight_layout()

PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
plt.savefig(PATH_TO_ROOT+'/figures/target_corr.png')

#print(data)