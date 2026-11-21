from pathlib import Path
import json
import yaml
import joblib

from src.data_loader import load_and_preprocess_data
from src.model_trainer import train_and_save_model


# Ruta principal del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent


# Leer la configuracion de YAML
def cargar_configuracion():
    ruta_config = BASE_DIR / "config" / "params.yaml"

    with open(ruta_config, "r", encoding="utf-8") as archivo:
        configuracion = yaml.safe_load(archivo)

    return configuracion


# Crear las carpetas donde se guardaran los resultados
def preparar_directorios(configuracion):
    (BASE_DIR / configuracion["models_path"]).mkdir(
        parents=True, exist_ok=True
    )

    (BASE_DIR / configuracion["outputs_path"]).mkdir(
        parents=True, exist_ok=True
    )


# Funcion principal que coordina el pipeline
def ejecutar_pipeline():

    # Cargar los parametros
    configuracion = cargar_configuracion()

    # Preparar carpetas
    preparar_directorios(configuracion)

    # Obtener la ruta del dataset
    ruta_datos = BASE_DIR / configuracion["data_path"]

    # Verificar que el dataset existe
    if not ruta_datos.is_file():
        raise FileNotFoundError(
            f"No se encontro el dataset: {ruta_datos}"
        )

    configuracion["data_path"] = str(ruta_datos)

    print("Iniciando pipeline de clasificacion de ADN...")
    print(f"Dataset encontrado: {ruta_datos.name}")

    print(f"Porcentaje de prueba: {configuracion['test_size']}")
    print(f"Semilla aleatoria: {configuracion['random_state']}")

    # Ejecutar el preprocesamiento
    print("Iniciando preprocesamiento de datos...")

    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        configuracion
    )

    print("Preprocesamiento completado correctamente.")

    print(f"Registros de entrenamiento: {X_train.shape[0]}")
    print(f"Registros de prueba: {X_test.shape[0]}")

    # Entrenar los tres modelos
    for nombre_modelo in ["RandomForest", "KNN", "DecisionTree"]:

        configuracion["model_name"] = nombre_modelo

        print(f"\nEntrenando modelo: {nombre_modelo}")

        modelo, metricas = train_and_save_model(
            X_train,
            y_train,
            X_test,
            y_test,
            configuracion
        )

        # Guardar el modelo entrenado
        ruta_modelo = (
            BASE_DIR
            / configuracion["models_path"]
            / f"{nombre_modelo}.pkl"
        )

        joblib.dump(modelo, ruta_modelo)

        print(f"Modelo guardado en: {ruta_modelo}")

        # Guardar las metricas obtenidas
        ruta_metricas = (
            BASE_DIR
            / configuracion["outputs_path"]
            / f"{nombre_modelo}_metricas.json"
        )

        with open(ruta_metricas, "w", encoding="utf-8") as archivo:
            json.dump(metricas, archivo, indent=4)

        print(f"Metricas guardadas en: {ruta_metricas}")

    print("\nPipeline finalizado correctamente.")


if __name__ == "__main__":
    ejecutar_pipeline()

