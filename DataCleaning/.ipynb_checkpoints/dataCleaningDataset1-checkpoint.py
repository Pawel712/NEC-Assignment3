import pandas as pd
from sklearn.preprocessing import StandardScaler

with open('../Datasets/OriginalData/Dataset1Original/A3-dataset1.txt', 'r') as file:
    df = pd.read_csv('../Datasets/OriginalData/Dataset1Original/A3-dataset1.txt', sep=',')

#z-score normalization except the last column
scaler = StandardScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

df.to_csv("../Datasets/A3-dataset1Modified.csv", sep=",", index=False)
