# %%
import pandas as pd

# Load data
data = pd.read_csv('data_validasi.csv')
print(data.head())

# %%
value_counts = data.iloc[:, 1:].apply(pd.value_counts)
print(value_counts)

# %%
print(data.dtypes)

# %%
