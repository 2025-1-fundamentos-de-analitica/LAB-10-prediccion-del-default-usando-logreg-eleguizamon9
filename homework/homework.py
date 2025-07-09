# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#

# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
import gzip
import json
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

# Paso 1: Limpieza de datos
def load_and_clean_data():
    """Carga y limpia los datasets"""
    # Cargar datos
    train_data = pd.read_csv("files/input/train_data.csv.zip", compression='zip')
    test_data = pd.read_csv("files/input/test_data.csv.zip", compression='zip')
    
    # Renombrar columna y remover ID
    train_data = train_data.rename(columns={"default payment next month": "default"})
    test_data = test_data.rename(columns={"default payment next month": "default"})
    
    train_data = train_data.drop("ID", axis=1)
    test_data = test_data.drop("ID", axis=1)
    
    # Eliminar registros con información no disponible
    train_data = train_data.dropna()
    test_data = test_data.dropna()
    
    # Agrupar valores > 4 en EDUCATION como "others" (valor 4)
    train_data.loc[train_data["EDUCATION"] > 4, "EDUCATION"] = 4
    test_data.loc[test_data["EDUCATION"] > 4, "EDUCATION"] = 4
    
    return train_data, test_data

# Paso 2: Dividir datasets
def split_data(train_data, test_data):
    """Divide los datasets en X e y"""
    x_train = train_data.drop("default", axis=1)
    y_train = train_data["default"]
    x_test = test_data.drop("default", axis=1)
    y_test = test_data["default"]
    
    return x_train, y_train, x_test, y_test

# Paso 3: Crear pipeline
def create_pipeline():
    """Crea el pipeline de preprocessing y modelo"""
    # Identificar columnas categóricas y numéricas
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    
    # Crear preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
            ('scaler', MinMaxScaler(), ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                                       'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                                       'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'])
        ]
    )
    
    # Crear pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(score_func=f_classif)),
        ('classifier', LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced'))
    ])
    
    return pipeline

# Paso 4: Optimizar hiperparámetros
def optimize_pipeline(pipeline, x_train, y_train):
    """Optimiza hiperparámetros usando GridSearchCV"""
    # Definir parámetros para búsqueda
    param_grid = {
        'selector__k': [10, 15, 20, 25, 'all'],
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear', 'saga'],
        'classifier__class_weight': ['balanced', None]
    }
    
    # Crear GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring='balanced_accuracy',
        n_jobs=-1
    )
    
    # Ajustar modelo
    grid_search.fit(x_train, y_train)
    
    return grid_search

# Paso 5: Guardar modelo
def save_model(model, filename):
    """Guarda el modelo comprimido"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with gzip.open(filename, 'wb') as f:
        pickle.dump(model, f)

# Paso 6 y 7: Calcular métricas y matrices de confusión
def calculate_and_save_metrics(model, x_train, y_train, x_test, y_test):
    """Calcula métricas y matrices de confusión"""
    os.makedirs("files/output", exist_ok=True)
    
    # Predicciones
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    metrics = []
    
    # Métricas para entrenamiento
    train_metrics = {
        'type': 'metrics',
        'dataset': 'train',
        'precision': precision_score(y_train, y_train_pred),
        'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1_score': f1_score(y_train, y_train_pred)
    }
    metrics.append(train_metrics)
    
    # Métricas para prueba
    test_metrics = {
        'type': 'metrics',
        'dataset': 'test',
        'precision': precision_score(y_test, y_test_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred)
    }
    metrics.append(test_metrics)
    
    # Matriz de confusión para entrenamiento
    cm_train = confusion_matrix(y_train, y_train_pred)
    train_cm = {
        'type': 'cm_matrix',
        'dataset': 'train',
        'true_0': {
            'predicted_0': int(cm_train[0, 0]),
            'predicted_1': int(cm_train[0, 1])
        },
        'true_1': {
            'predicted_0': int(cm_train[1, 0]),
            'predicted_1': int(cm_train[1, 1])
        }
    }
    metrics.append(train_cm)
    
    # Matriz de confusión para prueba
    cm_test = confusion_matrix(y_test, y_test_pred)
    test_cm = {
        'type': 'cm_matrix',
        'dataset': 'test',
        'true_0': {
            'predicted_0': int(cm_test[0, 0]),
            'predicted_1': int(cm_test[0, 1])
        },
        'true_1': {
            'predicted_0': int(cm_test[1, 0]),
            'predicted_1': int(cm_test[1, 1])
        }
    }
    metrics.append(test_cm)
    
    # Guardar métricas
    with open("files/output/metrics.json", "w", encoding="utf-8") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + "\n")

# Función principal
def main():
    """Función principal que ejecuta todo el proceso"""
    # Paso 1: Cargar y limpiar datos
    train_data, test_data = load_and_clean_data()
    
    # Paso 2: Dividir datos
    x_train, y_train, x_test, y_test = split_data(train_data, test_data)
    
    # Paso 3: Crear pipeline
    pipeline = create_pipeline()
    
    # Paso 4: Optimizar hiperparámetros
    best_model = optimize_pipeline(pipeline, x_train, y_train)
    
    # Paso 5: Guardar modelo
    save_model(best_model, "files/models/model.pkl.gz")
    
    # Paso 6 y 7: Calcular y guardar métricas
    calculate_and_save_metrics(best_model, x_train, y_train, x_test, y_test)
    
    print("Proceso completado exitosamente!")
    print(f"Mejor score de validación cruzada: {best_model.best_score_:.4f}")
    print(f"Mejores parámetros: {best_model.best_params_}")

if __name__ == "__main__":
    main()