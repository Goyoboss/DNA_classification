from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score

from src.Preprocesamiento import load_and_preprocess_data


def train_and_save_model(X_train, y_train, X_test, y_test, config):
    model = config["model_name"]
    #si se escogió el modelo de árboles de decisión
    if model=="DecisionTree":
        model = DecisionTreeClassifier(**config["DecisionTree"])
    #si se escogió el modelo de bosques aleatorios
    elif model=="RandomForest":
        model = RandomForestClassifier(**config["RandomForest"])
    #si se escogió el modelo de knn
    elif model=="KNN":
        model = KNeighborsClassifier(**config["KNN"])
    else:
        print(f"No se admite {model}, solo: DecisionTree, RandomForest y KNN")

    #se entrena el modelo
    model.fit(X_train, y_train)

    #se generan las predicciones del modelo
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    #-----Métricas-----
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    pr_train = precision_score(y_train, y_pred_train, average='weighted', zero_division=1)
    pr_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=1)

    rec_train = recall_score(y_train, y_pred_train, average='weighted', zero_division=1)
    rec_test = recall_score(y_test, y_pred_test, average='weighted', zero_division=1)

    f1_train = f1_score(y_train, y_pred_train, average='weighted', zero_division=1)
    f1_test = f1_score(y_test, y_pred_test, average='weighted', zero_division=1)

    cm = confusion_matrix(y_test, y_pred_test)

    print("-"*10, f"METRICAS EN {model}", "-"*10)
    print(f"Accuracy en entrenamiento: {acc_train:.2f}\nAccuracy en prueba: {acc_test:.2f}\n")
    print(f"Precision en entrenamiento: {pr_train:.2f}\nPrecision en prueba: {pr_test:.2f}\n")
    print(f"Recall en entrenamiento: {rec_train:.2f}\nRecall en prueba: {rec_test:.2f}\n")
    print(f"F1 en entrenamiento: {f1_train:.2f}\nF1 en prueba: {f1_test:.2f}\n")
    print("Matriz de confusión:\n", cm)

    #diccionario de métricas
    metricas = {
        "model": config["model_name"],
        "accuracy_train": float(acc_train),
        "accuracy_test": float(acc_test),
        "precision_train": float(pr_train),
        "precision_test": float(pr_test),
        "recall_train": float(rec_train),
        "recall_test": float(rec_test),
        "f1_train": float(f1_train),
        "f1_test": float(f1_test)
    }

    return model, metricas