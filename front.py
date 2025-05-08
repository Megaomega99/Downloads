import pandas as pd
import numpy as np
import os
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import flet as ft
from flet.matplotlib_chart import MatplotlibChart

class PredictorApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "Predictor de Series Temporales"
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.padding = 20
        
        # Variables de estado
        self.data = None
        self.filepath = None
        self.processed_data = None
        self.target_column = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        
        # Control de navegación
        self.current_view = "main"  # Puede ser "main" o "results"
        
        # Componentes de la UI
        self.setup_ui()
        
    def setup_ui(self):
        # Título principal
        self.title = ft.Text("Predictor de Series Temporales", size=30, weight=ft.FontWeight.BOLD)
        
        # Configuración de la página principal
        self.setup_main_view()
        
        # Configuración de la página de resultados
        self.setup_results_view()
        
        # Mostrar la vista principal inicialmente
        self.show_view("main")
    
    def setup_main_view(self):
        # Sección 1: Carga de datos
        self.file_path_text = ft.Text("Ningún archivo seleccionado", color=ft.colors.GREY)
        self.select_file_button = ft.ElevatedButton(
            text="Seleccionar archivo CSV",
            icon=ft.icons.UPLOAD_FILE,
            on_click=self.pick_files
        )
        
        # Sección 2: Vista previa de datos
        self.data_preview_title = ft.Text("Vista previa de datos", size=20, weight=ft.FontWeight.BOLD)
        self.data_preview = ft.DataTable(
            columns=[ft.DataColumn(ft.Text("Sin datos"))],
            rows=[]
        )
        self.data_info = ft.Text("Sin datos cargados", color=ft.colors.GREY)
        
        # Sección 3: Preprocesamiento
        self.preprocess_button = ft.ElevatedButton(
            text="Preprocesar datos",
            icon=ft.icons.CLEANING_SERVICES,
            on_click=self.handle_preprocess,
            disabled=True
        )
        
        # Sección 4: Selección de columna objetivo
        self.target_column_dropdown = ft.Dropdown(
            label="Columna objetivo",
            disabled=True,
            on_change=self.handle_target_change
        )
        
        # Sección 5: Entrenamiento del modelo
        self.model_type_dropdown = ft.Dropdown(
            label="Tipo de modelo",
            options=[
                ft.dropdown.Option("lasso", "Lasso"),
                ft.dropdown.Option("ridge", "Ridge"),
                ft.dropdown.Option("elasticnet", "ElasticNet"),
                ft.dropdown.Option("svr", "SVR"),
                ft.dropdown.Option("sgd", "SGD")
            ],
            disabled=True,
            value="lasso"
        )
        self.train_button = ft.ElevatedButton(
            text="Entrenar modelo",
            icon=ft.icons.MODEL_TRAINING,
            on_click=self.handle_train,
            disabled=True
        )
        
        # Construir la vista principal
        self.main_view = ft.Column(
            controls=[
                self.title,
                ft.Divider(),
                
                # Sección 1: Carga de datos
                ft.Row([self.select_file_button, self.file_path_text]),
                ft.Divider(),
                
                # Sección 2: Vista previa de datos
                self.data_preview_title,
                self.data_info,
                self.data_preview,
                ft.Divider(),
                
                # Sección 3 y 4: Preprocesamiento y selección de columna objetivo
                ft.Row([
                    self.preprocess_button,
                    self.target_column_dropdown,
                ]),
                ft.Divider(),
                
                # Sección 5: Entrenamiento del modelo
                ft.Row([
                    self.model_type_dropdown,
                    self.train_button,
                ]),
            ],
            scroll=ft.ScrollMode.AUTO
        )
    
    def setup_results_view(self):
        # Título de resultados
        self.results_title = ft.Text("Resultados del Modelo", size=30, weight=ft.FontWeight.BOLD)
        
        # Métricas de rendimiento
        self.results_metrics = ft.Text("Sin resultados disponibles", color=ft.colors.GREY)
        
        # Gráfico
        self.chart_container = ft.Container(
            content=ft.Text("El gráfico se mostrará aquí después del entrenamiento"),
            alignment=ft.alignment.center,
            height=400
        )
        
        # Detalles del modelo
        self.model_details = ft.Text("", size=16)
        
        # Botón para volver a la página principal
        self.back_button = ft.ElevatedButton(
            text="Volver al inicio",
            icon=ft.icons.ARROW_BACK,
            on_click=lambda e: self.show_view("main")
        )
        
        # Construir la vista de resultados
        self.results_view = ft.Column(
            controls=[
                self.results_title,
                ft.Divider(),
                
                ft.Container(
                    content=ft.Column([
                        ft.Text("Métricas del modelo", size=20, weight=ft.FontWeight.BOLD),
                        self.results_metrics,
                    ]),
                    padding=20,
                    border=ft.border.all(1, ft.colors.GREY_300),
                    border_radius=10,
                    margin=10
                ),
                
                ft.Divider(),
                ft.Text("Gráfico de predicciones", size=20, weight=ft.FontWeight.BOLD),
                self.chart_container,
                ft.Divider(),
                
                self.model_details,
                ft.Divider(),
                
                ft.Row([self.back_button], alignment=ft.MainAxisAlignment.CENTER)
            ],
            scroll=ft.ScrollMode.AUTO
        )
    
    def show_view(self, view_name):
        """Cambia entre las diferentes vistas de la aplicación"""
        self.current_view = view_name
        
        # Limpiar la página
        self.page.controls.clear()
        
        # Mostrar la vista correspondiente
        if view_name == "main":
            self.page.add(self.main_view)
        elif view_name == "results":
            self.page.add(self.results_view)
        
        self.page.update()
    
    def pick_files(self, e):
        def on_dialog_result(e: ft.FilePickerResultEvent):
            if e.files:
                self.filepath = e.files[0].path
                self.file_path_text.value = f"Archivo seleccionado: {os.path.basename(self.filepath)}"
                
                try:
                    self.data = pd.read_csv(self.filepath)
                    self.data_info.value = f"Dimensiones: {self.data.shape[0]} filas x {self.data.shape[1]} columnas"
                    self.update_data_preview()
                    self.preprocess_button.disabled = False
                    self.page.update()
                except Exception as ex:
                    self.data_info.value = f"Error al cargar el archivo: {ex}"
                    self.page.update()
            else:
                self.page.update()
        
        picker = ft.FilePicker(on_result=on_dialog_result)
        self.page.overlay.append(picker)
        self.page.update()
        picker.pick_files(
            dialog_title="Selecciona un archivo CSV",
            allowed_extensions=["csv"],
            initial_directory=os.path.expanduser("~")
        )
    
    def update_data_preview(self):
        if self.data is None:
            return
        
        # Limitar la vista previa a 5 filas y 10 columnas para mejor rendimiento
        preview_data = self.data.iloc[:5, :10] if self.data.shape[1] > 10 else self.data.iloc[:5, :]
        
        # Crear columnas
        columns = [
            ft.DataColumn(ft.Text(str(col)))
            for col in preview_data.columns
        ]
        
        # Crear filas
        rows = []
        for _, row in preview_data.iterrows():
            cells = [ft.DataCell(ft.Text(str(val))) for val in row]
            rows.append(ft.DataRow(cells=cells))
        
        self.data_preview.columns = columns
        self.data_preview.rows = rows
    
    def handle_preprocess(self, e):
        if self.data is not None:
            self.processed_data = self.preprocess_data(self.data)
            self.data_info.value = f"Datos preprocesados: {self.processed_data.shape[0]} filas x {self.processed_data.shape[1]} columnas"
            
            # Actualizar la vista previa con los datos preprocesados
            self.data = self.processed_data
            self.update_data_preview()
            
            # Actualizar el dropdown de columnas objetivo
            self.target_column_dropdown.options = [
                ft.dropdown.Option(col, col) for col in self.processed_data.columns
            ]
            self.target_column_dropdown.disabled = False
            self.page.update()
    
    def handle_target_change(self, e):
        self.target_column = e.data
        self.model_type_dropdown.disabled = False  # Habilitar el dropdown de modelos
        self.train_button.disabled = False
        self.page.update()
    
    def handle_train(self, e):
        if self.processed_data is not None and self.target_column:
            try:
                # Dividir los datos
                self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(
                    self.processed_data, self.target_column
                )
                
                # Entrenar el modelo
                model_type = self.model_type_dropdown.value
                self.model = self.train_model(
                    self.X_train.values.reshape(-1, 1), 
                    self.y_train.values, 
                    model_type=model_type
                )
                
                # Predecir
                self.y_pred = self.model.predict(self.X_test.values.reshape(-1, 1))
                
                # Evaluar
                mse = mean_squared_error(self.y_test, self.y_pred)
                r2 = r2_score(self.y_test, self.y_pred)
                
                # Mostrar resultados
                self.results_metrics.value = f"""
MSE (Error Cuadrático Medio): {mse:.4f}
R² (Coeficiente de determinación): {r2:.4f}
                """
                
                # Mostrar detalles del modelo
                self.model_details.value = f"""
Modelo: {model_type.upper()}
Columna objetivo: {self.target_column}
Tamaño del conjunto de entrenamiento: {self.X_train.shape[0]} filas
Tamaño del conjunto de prueba: {self.X_test.shape[0]} filas
                """
                
                # Crear y mostrar gráfico
                self.update_chart()
                
                # Cambiar a la vista de resultados
                self.show_view("results")
                
            except Exception as ex:
                self.results_metrics.value = f"Error durante el entrenamiento: {ex}"
                self.page.update()
    
    def update_chart(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.X_test, self.y_test, label='Valores reales', color='blue')
        ax.plot(self.X_test, self.y_pred, label='Predicciones', color='red')
        ax.set_title('Predicciones vs Valores reales')
        ax.set_xlabel('Índice')
        ax.set_ylabel('Valor')
        ax.legend()
        
        # Actualizar contenedor con gráfico
        chart = MatplotlibChart(fig, expand=True)
        self.chart_container.content = chart
    
    # Funciones de procesamiento (las mismas que en el código original)
    def preprocess_data(self, data):
        # Eliminar columnas no numéricas
        data = data.select_dtypes(include=[np.number])
        
        # Eliminar filas con valores nulos
        data = data.dropna()
        
        return data
    
    def split_data(self, data, target_column, test_size=0.2, random_state=42):
        # Se crea el conjunto de datos de entrenamiento y prueba
        X = pd.Series(list(range(0, len(data), 1)))
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        # Resetear índices
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        
        # Convertir a DataFrames
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, model_type='lasso'):
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

def main(page: ft.Page):
    app = PredictorApp(page)

# Iniciar la aplicación
if __name__ == "__main__":
    ft.app(target=main)