import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack


def load_and_preprocess_data(config):
    #configuración de yaml
    data_path = config["data_path"]
    test_size = config["test_size"]
    random_state = config["random_state"]
    kmer_size = config["kmer_size"]
    min_samples = config["min_samples"]
    max_features = config["tfidf_max_features"]

    #eliminar columna que no aporta información real
    df = pd.read_csv(data_path)
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
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    #CountVectorizer: letras a números
    vectorizer = CountVectorizer()
    vectorizer = CountVectorizer()
    X_train_kmers = vectorizer.fit_transform(X_train["Kmers"])
    X_test_kmers = vectorizer.transform(X_test["Kmers"])

    #TfidVectorizaer: transformar cadenas de texto en número
    tfidf = TfidfVectorizer(max_features=100)
    X_train_desc = tfidf.fit_transform(X_train['Description'].fillna(''))
    X_test_desc = tfidf.transform(X_test['Description'].fillna(''))

    #OneHotEncoder: codificar variables en 0 y 1
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_train_group = encoder.fit_transform(X_train[['GeneGroupMethod']])
    X_test_group = encoder.transform(X_test[['GeneGroupMethod']])

    #se unen las columnas en una sola matriz
    X_train = hstack([X_train_kmers, X_train_desc, X_train_group])
    X_test = hstack([X_test_kmers, X_test_desc, X_test_group])



print(X_train.shape)
print(X_test.shape)

