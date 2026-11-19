import pandas as pd

df_train = pd.read_csv('DNA_data/train.csv')
df_test = pd.read_csv('DNA_data/test.csv')
df_val = pd.read_csv('DNA_data/validation.csv')

df_completo = pd.concat([df_train, df_test, df_val], ignore_index=True)

nombre_archivo_completo = 'DNA_classification.csv'
df_completo.to_csv(nombre_archivo_completo, index=False)

