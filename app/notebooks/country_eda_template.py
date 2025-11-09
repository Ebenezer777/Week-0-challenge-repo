# cell 1: imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# cell 2: load data
raw_path = "data/benin_raw.csv"  # rename as appropriate
df = pd.read_csv(raw_path, parse_dates=['Timestamp'])

# cell 3: initial report
print(df.info())
display(df.describe())
display(df.isna().sum())

# cell 4: convert numerics
num_cols = ['GHI','DNI','DHI','ModA','ModB','Tamb','RH','WS','WSgust','BP']
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# cell 5: missing value strategy
missing = df.isna().mean()*100
print("Columns with >5% missing:\n", missing[missing>5])

# cell 6: impute medians for key columns
for c in ['GHI','DNI','DHI','ModA','ModB','Tamb']:
    df[c].fillna(df[c].median(), inplace=True)

# cell 7: z-score outliers
cols_to_z = ['GHI','DNI','DHI','ModA','ModB','WS','WSgust']
z = np.abs(stats.zscore(df[cols_to_z].fillna(df[cols_to_z].median())))
outliers = (z>3).any(axis=1)
print("Outliers flagged:", outliers.sum())
df['outlier_flag'] = outliers

# cell 8: time series plot
df.set_index('Timestamp', inplace=True)
df[['GHI','DNI','DHI']].resample('1H').mean().plot(figsize=(12,5))
plt.title("Hourly mean irradiance")
plt.show()

# cell 9: cleaning impact
if 'Cleaning' in df.columns:
    grp = df.groupby('Cleaning')[['ModA','ModB']].mean()
    display(grp)

# cell 10: correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[['GHI','DNI','DHI','TModA','TModB','Tamb','RH']].corr(), annot=True, fmt=".2f")
plt.show()

# cell 11: export
df.reset_index().to_csv("data/benin_clean.csv", index=False)
print("Saved data/benin_clean.csv (do NOT commit)")
