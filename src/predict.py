import joblib
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack

BASE_DIR = Path(__file__).resolve().parent.parent

def kmers(sequence, size=6):
    return ' '.join([sequence[i:i+size] for i in range(len(sequence)-size+1)])

def predict_new_sequence(nucleotide_sequence, description, gene_group):
    # 1. Cargar el modelo y los artefactos (Manejo de errores crucial aquí)
    try:
        model = joblib.load(BASE_DIR / "models/RandomForest.pkl") # o el que quieras usar
        le = joblib.load(BASE_DIR / "models/label_encoder.pkl")
        vectorizer = joblib.load(BASE_DIR / "models/vectorizer.pkl")
        tfidf = joblib.load(BASE_DIR / "models/tfidf.pkl")
        encoder = joblib.load(BASE_DIR / "models/onehot.pkl")
    except FileNotFoundError:
        print("Error: No se encontraron los modelos/artefactos. Por favor ejecuta 'python -m src.main' primero.")
        return

    # 2. Preprocesar la entrada (imitar el flujo del data_loader)
    sec_limpia = nucleotide_sequence.replace("<", "").replace(">", "")
    kmer_seq = kmers(sec_limpia)
    
    x_kmers = vectorizer.transform([kmer_seq])
    x_desc = tfidf.transform([description])
    x_group = encoder.transform(pd.DataFrame({'GeneGroupMethod': [gene_group]}))
    
    # 3. Unir y predecir
    X_new = hstack([x_kmers, x_desc, x_group])
    prediccion_num = model.predict(X_new)
    
    # 4. Decodificar la predicción
    clase_predicha = le.inverse_transform(prediccion_num)
    print(f"La secuencia pertenece a la clase: {clase_predicha[0]}")

# Ejemplo de uso:
if __name__ == "__main__":
    predict_new_sequence("ATGCTAGCTAG", "Example gene description", "MethodA")