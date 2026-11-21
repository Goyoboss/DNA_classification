import yaml
from src.data_loader import load_and_preprocess_data
from src.model_trainer import train_and_save_model
# (importa también train_and_save_model si la tienes en otro archivo)

# 1. Abrir el archivo config.yaml y cargarlo en la variable 'config'
with open("config.yaml", "r") as archivo:
    config = yaml.safe_load(archivo)

# 2. Ahora 'config' ya existe en memoria y se lo puedes pasar a tus funciones
X_train, X_test, y_train, y_test = load_and_preprocess_data(config)
metricas = train_and_save_model(X_train, y_train, X_test, y_test, config)

print("Métricas obtenidas:", metricas)