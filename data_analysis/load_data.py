from sklearn.datasets import load_breast_cancer
import pandas as pd
data = load_breast_cancer(as_frame=True)

# Get an impression of our data set
print((data.target.value_counts(normalize=True)))