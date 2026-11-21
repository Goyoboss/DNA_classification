from pathlib import Path
import yaml
BASE_DIR = Path(__file__).resolve().parent.parent #guardala ruta principal del proyecto


def cargar_configuracion():
    ruta_config = BASE_DIR / "config" / "params.yaml"

    with open(ruta_config, "r", encoding="utf-8") as archivo:
        configuracion = yaml.safe_load(archivo)

    return configuracion


def preparar_directorios(configuracion):
    (BASE_DIR / configuracion["paths"]["models"]).mkdir(parents=True, exist_ok=True)
    (BASE_DIR / configuracion["paths"]["outputs"]).mkdir(parents=True, exist_ok=True)




def ejecutar_pipeline():
    configuracion = cargar_configuracion()

    preparar_directorios(configuracion)

    ruta_datos = BASE_DIR / configuracion["paths"]["data"]

    if not ruta_datos.is_file():
        raise FileNotFoundError(f"No se encontró el dataset: {ruta_datos}")

    print(f"Dataset encontrado: {ruta_datos.name}")

    print("Iniciando pipeline de clasificación de ADN...")

    test_size = configuracion["data"]["test_size"]
    random_state = configuracion["data"]["random_state"]

    print(f"Porcentaje de prueba: {test_size}")
    print(f"Semilla aleatoria: {random_state}")

# Siguiente etapa: preparar los datos del ADN
# El Data Engineer proporcionará preparar_datos(ruta_datos, test_size, random_state)
# La función devolverá X_train, X_test, y_train, y_test

    parametros_modelos = configuracion["models"]

    print("Modelos que se entrenarán:")

    for nombre_modelo in parametros_modelos: #extrae la seccion de modelos de yaml
        print(f"- {nombre_modelo}")

if __name__ == "__main__":
    ejecutar_pipeline();