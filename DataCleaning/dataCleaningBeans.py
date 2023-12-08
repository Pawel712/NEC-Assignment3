import pandas as pd
from sklearn.preprocessing import StandardScaler

with open('../Datasets/DryBeansOriginalData/A3-dryBeans.csv', 'r') as file:
    df = pd.read_csv('../Datasets/DryBeansOriginalData/A3-dryBeans.csv', sep=',')

#Replacing the class names with numbers
df['Class'] = df['Class'].replace({'SEKER': 0, 'BARBUNYA': 1, 'BOMBAY': 2, 'CALI': 3, 'DERMASON': 4, 'HOROZ': 5, 'SIRA': 6})

#z-score normalization except the last column
scaler = StandardScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

df.to_csv("../Datasets/A3-dryBeansModified.csv", sep=",", index=False)

