import pandas as pd


df = pd.read_csv("DNA_classification.csv")

# saber qué tipos de datos son y si existen valores nulos
print(df.info())

print(df.head())