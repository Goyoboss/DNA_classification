import yaml
from pathlib import Path
from src.data_loader import load_and_preprocess_data

BASE_DIR = Path(__file__).resolve().parent.parent

def test_config_exists():
    config_path = BASE_DIR / "config/params.yaml"
    assert config_path.exists(), "El archivo params.yaml no se encuentra."

def test_data_loader_returns_correct_shapes():
    with open(BASE_DIR / "config/params.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Truco para que apunte bien al archivo de pruebas
    config["data_path"] = str(BASE_DIR / config["data_path"]) 
    
    # Asumiendo que modificaste data_loader para retornar los encoders
    X_train, X_test, y_train, y_test, _, _, _, _ = load_and_preprocess_data(config)
    
    assert X_train.shape[0] > 0, "X_train está vacío"
    assert X_test.shape[0] > 0, "X_test está vacío"
    assert len(y_train) == X_train.shape[0], "Desajuste de dimensiones entre X e y en entrenamiento"