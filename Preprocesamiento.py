import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack

df = pd.read_csv("DNA_classification.csv")


#eliminar columna que no aporta información real
df = df.drop(columns=['Unnamed: 0'])

#se eliminan los caracteres <> de la columna NucleotideSequence 
df['NucleotideSequence'] = df['NucleotideSequence'].str.replace(
    r'[<>]', "", regex=True)

#Solo con clases que tengan más de 5 
conteo = df['GeneType'].value_counts()
clases_validas = conteo[conteo >= 5].index
df = df[df['GeneType'].isin(clases_validas)].copy()

#label encoder para codificar la variable objetivo
le = LabelEncoder()
y = le.fit_transform(df['GeneType'])

#kmers para dividir en secuencias de 6
def kmers(sequence, size=6):
    return ' '.join([sequence[i:i+size] for i in range(len(sequence)-size+1)])

df['Kmers'] = df['NucleotideSequence'].apply(kmers)

#matriz de características
X = df[['Kmers', 'Description', 'GeneGroupMethod']]

#división de los datos en 70% entrenamiento, 30% validación y prueba
X_train_df, X_temp_df, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

#división de los datos de pureba y validación
X_val_df, X_test_df, y_val, y_test = train_test_split(
    X_temp_df, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

#CountVectorizer: letras a números
vectorizer = CountVectorizer()
X_train_kmers = vectorizer.fit_transform(X_train_df['Kmers'])    #70%
X_val_kmers = vectorizer.transform(X_val_df['Kmers'])            #15%
X_test_kmers = vectorizer.transform(X_test_df['Kmers'])          #15%

#TfidVectorizaer: transformar cadenas de texto en número
tfidf = TfidfVectorizer(max_features=100)
X_train_desc = tfidf.fit_transform(X_train_df['Description'].fillna(''))
X_val_desc = tfidf.transform(X_val_df['Description'].fillna(''))
X_test_desc = tfidf.transform(X_test_df['Description'].fillna(''))

#OneHotEncoder: codificar variables en 0 y 1
encoder = OneHotEncoder(handle_unknown='ignore')
X_train_group = encoder.fit_transform(X_train_df[['GeneGroupMethod']])
X_val_group = encoder.transform(X_val_df[['GeneGroupMethod']])
X_test_group = encoder.transform(X_test_df[['GeneGroupMethod']])

#se unen las columnas en una sola matriz
X_train = hstack([X_train_kmers, X_train_desc, X_train_group])
X_val = hstack([X_val_kmers, X_val_desc, X_val_group])
X_test = hstack([X_test_kmers, X_test_desc, X_test_group])

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)