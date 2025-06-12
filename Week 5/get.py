import pandas as pd
from sklearn.datasets import fetch_california_housing

cal = fetch_california_housing()
df = pd.DataFrame(cal.data, columns=cal.feature_names)
df['PRICE'] = cal.target
df.to_csv("california_housing.csv", index=False)
