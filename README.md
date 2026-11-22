#**Clasificación de ADN a través del reconocimiento de patrones de secuencias nucleotídicas**

___

##**Objetivo del proyecto**

Construir un pipeline de Machine Learning modular, reproducible y colaborativo para predecir el tipo de gen de acuerdo a patrones encontrados en las secuencias nucleotídicas.

___

##**Dataset**

**Fuente:** [DNA sequence prediction](https://www.kaggle.com/datasets/harshvardhan21/dna-sequence-prediction)

**Archivos:** test.csv, train.csv, validation.csv

**Archivo final:** DNA_classification.csv

___

##**Estructuras de carpetas**

```
DNA_classification/
|-- config/
|    |_ params.yaml          # configuración centralizada
|-- DNA_data/                # conjuntos de datos
|    |_train.csv
|    |_test.csv
|    |_validation.csv
|    |_union_data.csv        # unión de los tres datasets en uno solo
|-- DNA_classification.csv   # conjunto de datos final
|-- src/
|    |_Model_trainer.py      # entrenamiento de los modelo | Rol: ML Engineer
|    |_data_loader.py        # limpieza y preprocesamiento de los datos | Rol: Data Engineer
|    |_main.py               # Rol: MLOps Engineer
|    |_predict.py            # Rol: QA Engineer
|-- tests.py                 # Rol: QA Engineer
|-- requirements.txt         # dependencias
|-- .gitignore
|-- README.md
```

