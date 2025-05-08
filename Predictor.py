import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

def select_csv_file():
    """
    Abre un selector de archivos para que el usuario elija un archivo CSV
    y lo carga como DataFrame de pandas.
    
    Returns:
        pandas.DataFrame: Los datos del archivo CSV seleccionado
        str: Ruta del archivo seleccionado
    """
    root = tk.Tk()
    root.withdraw()  # Oculta la ventana principal de tkinter
    
    # Muestra el diálogo para seleccionar un archivo
    file_path = filedialog.askopenfilename(
        title="Selecciona un archivo CSV",
        filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
    )
    
    if file_path:
        try:
            # Cargar el archivo CSV como un DataFrame de pandas
            df = pd.read_csv(file_path)
            print(f"Archivo cargado correctamente: {os.path.basename(file_path)}")
            print(f"Dimensiones: {df.shape}")
            return df, file_path
        except Exception as e:
            print(f"Error al cargar el archivo: {e}")
            return None, file_path
    else:
        print("No se seleccionó ningún archivo.")
        return None, None
def preprocess_data(data):
    """
    Preprocesa los datos eliminando columnas no numéricas y filas con valores nulos.
    
    Args:
        data (pandas.DataFrame): DataFrame a preprocesar
    
    Returns:
        pandas.DataFrame: DataFrame preprocesado
    """
    # Eliminar columnas no numéricas
    data = data.select_dtypes(include=[np.number])
    
    # Eliminar filas con valores nulos
    data = data.dropna()
    print(f"Datos preprocesados: {data.shape[0]} filas restantes después de eliminar nulos.")

    
    return data
def split_data(data, target_column, test_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba teniendo en cuenta que es una serie temporal.
    
    Args:
        data (pandas.DataFrame): DataFrame con los datos
        target_column (str): Nombre de la columna objetivo
        test_size (float): Proporción del conjunto de prueba
        random_state (int): Semilla para la aleatoriedad
    
    Returns:
        tuple: Conjuntos de entrenamiento y prueba (X_train, X_test, y_train, y_test)
    """
    # se crea el conjunto de datos de entrenamiento y prueba sabiendo que es una serie temporal
    X = pd.Series(list(range(0, len(data), 1)))
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)
    print(f"Conjunto de entrenamiento: {X_train.shape[0]} filas, Conjunto de prueba: {X_test.shape[0]} filas")
    # se eliminan los índices de las series para que no interfieran en el entrenamiento del modelo
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    # se convierten las series en dataframes para que el modelo pueda entrenar correctamente
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    return X_train, X_test, y_train, y_test
def train_model(X_train, y_train, model_type='lasso'):
    """
    Entrena un modelo de regresión según el tipo especificado.
    
    Args:
        X_train (pandas.DataFrame): Conjunto de entrenamiento de características
        y_train (pandas.Series): Conjunto de entrenamiento de la variable objetivo
        model_type (str): Tipo de modelo ('lasso', 'ridge', 'elasticnet', 'svr', 'sgd')
    
    Returns:
        model: Modelo entrenado
    """
    if model_type == 'lasso':
        model = LassoCV(cv=5).fit(X_train, y_train)
    elif model_type == 'ridge':
        model = RidgeCV(cv=5).fit(X_train, y_train)
    elif model_type == 'elasticnet':
        model = ElasticNetCV(cv=5).fit(X_train, y_train)
    elif model_type == 'svr':
        model = SVR(kernel='linear').fit(X_train, y_train)
    elif model_type == 'sgd':
        model = SGDRegressor().fit(X_train, y_train)
    else:
        raise ValueError("Tipo de modelo no soportado.")
    
    return model



# Ejemplo de uso:
if __name__ == "__main__":
    print("Por favor, selecciona un archivo CSV...")
    data, filepath = select_csv_file()
    
    if data is not None:
        print("\nPrimeras 5 filas del DataFrame:")
        print(data.head())
        print("\nPreprocesando datos...")
        data = preprocess_data(data)
        print("Datos preprocesados.")
        print("\nPrimeras 5 filas del DataFrame preprocesado:")
        print(data.head())
        print("\nSelecciona la columna objetivo (target) para la predicción:")
        target_column = input("Columna objetivo: ")
        if target_column not in data.columns:
            print(f"Error: La columna '{target_column}' no se encuentra en el DataFrame.")
        else:
            print("\nDividiendo los datos en conjuntos de entrenamiento y prueba...")
            X_train, X_test, y_train, y_test = split_data(data, target_column)
            print("Datos divididos.")
            
            print("\nEntrenando el modelo...")
            model_type = input("Selecciona el tipo de modelo (lasso, ridge, elasticnet, svr, sgd): ").lower()
            model = train_model(X_train.values.reshape(-1, 1), y_train.values, model_type=model_type)
            print(f"Modelo {model_type} entrenado.")
            
            # Predicción
            y_pred = model.predict(X_test.values.reshape(-1, 1))
            
            # Evaluación del modelo
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\nError cuadrático medio (MSE): {mse}")
            print(f"R^2: {r2}")
            print("\nPredicciones:")
            print(y_pred)
            print("\nValores reales:")
            print(y_test.values)

            # se abre una ventana para graficar los resultados
            plt.figure(figsize=(10, 5))
            plt.plot(X_test, y_test, label='Valores reales', color='blue')
            plt.plot(X_test, y_pred, label='Predicciones', color='red')
            plt.title('Predicciones vs Valores reales')
            plt.xlabel('Índice')
            plt.ylabel('Valor')
            plt.legend()
            plt.show()
            
            print("\nFin del programa.")
    else:
        print("No se pudo cargar el archivo CSV. Asegúrate de que el archivo sea válido.")
    print("Fin del programa.")