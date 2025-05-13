import pandas as pd
import numpy as np
import os
import io
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import flet as ft
from flet.matplotlib_chart import MatplotlibChart

class PredictorApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "Predictor de Series Temporales"
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.padding = 20
        self.page.bgcolor = "#F5F5F7"  # Color de fondo moderno
        self.page.window_width = 1000  # Ancho de ventana predeterminado
        self.page.window_height = 800  # Alto de ventana predeterminado
        
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
        self.show_preview = True  # Variable para controlar la visibilidad de la vista previa
        
        # Control de navegación
        self.current_view = "main"  # Puede ser "main" o "results"
        
        # Variables para el easter egg
        self.easter_egg_clicks = 0
        self.last_click_time = 0
        
        # Componentes de la UI
        self.setup_ui()
        
    def setup_ui(self):
        # Título principal con easter egg
        self.logo = ft.Container(
            content=ft.Text(
                "⚡ Predictor ML", 
                size=40, 
                weight=ft.FontWeight.BOLD, 
                color="#1E88E5"
            ),
            on_click=self.handle_logo_click,
            margin=ft.margin.only(bottom=10),
        )
        
        # Logo Uniandes en esquina superior derecha
        self.uniandes_logo = ft.Image(src="logo_uniandes.PNG", width=100, height=100, fit=ft.ImageFit.CONTAIN)
        
        # Configuración de la página principal
        self.setup_main_view()
        
        # Configuración de la página de resultados
        self.setup_results_view()
        
        # Mostrar la vista principal inicialmente
        self.show_view("main")
    
    def setup_main_view(self):
        # Sección 1: Carga de datos
        self.file_path_text = ft.Text(
            "Ningún archivo seleccionado", 
            color=ft.colors.GREY_600,
            size=16
        )
        
        self.select_file_button = ft.ElevatedButton(
            text="Seleccionar CSV",
            icon=ft.icons.UPLOAD_FILE,
            on_click=self.pick_files,
            style=ft.ButtonStyle(
                color=ft.colors.WHITE,
                bgcolor="#1E88E5",
                elevation=5,
                shape=ft.RoundedRectangleBorder(radius=10),
                padding=15
            )
        )
        
        # Sección 2: Vista previa de datos
        self.data_preview_title = ft.Text(
            "Vista previa de datos", 
            size=22, 
            weight=ft.FontWeight.BOLD,
            color="#333333"
        )
        
        self.toggle_preview_button = ft.IconButton(
            icon=ft.icons.VISIBILITY_OFF,
            tooltip="Ocultar vista previa",
            on_click=self.toggle_data_preview,
            icon_color="#1E88E5",
            icon_size=24,
        )
        
        self.data_preview = ft.DataTable(
            columns=[ft.DataColumn(ft.Text("Sin datos"))],
            rows=[],
            border=ft.border.all(1, ft.colors.GREY_400),
            border_radius=8,
            vertical_lines=ft.border.BorderSide(1, ft.colors.GREY_300),
            horizontal_lines=ft.border.BorderSide(1, ft.colors.GREY_300),
            bgcolor=ft.colors.WHITE
        )
        
        self.data_preview_container = ft.Container(
            content=self.data_preview,
            padding=ft.padding.only(top=10),
            visible=self.show_preview
        )
        
        self.data_info = ft.Text(
            "Sin datos cargados", 
            color=ft.colors.GREY_600,
            size=16
        )
        
        # Botón para retroceder/reiniciar
        self.reset_button = ft.ElevatedButton(
            text="Reiniciar",
            icon=ft.icons.REFRESH,
            on_click=self.handle_reset,
            style=ft.ButtonStyle(
                color=ft.colors.WHITE,
                bgcolor="#FF5252",
                elevation=5,
                shape=ft.RoundedRectangleBorder(radius=10),
                padding=15
            )
        )
        
        # Sección 3: Preprocesamiento
        self.preprocess_button = ft.ElevatedButton(
            text="Preprocesar datos",
            icon=ft.icons.CLEANING_SERVICES,
            on_click=self.handle_preprocess,
            disabled=True,
            style=ft.ButtonStyle(
                color=ft.colors.WHITE,
                bgcolor="#4CAF50",
                elevation=5,
                shape=ft.RoundedRectangleBorder(radius=10),
                padding=15
            )
        )
        
        # Sección 4: Selección de columna objetivo
        self.target_column_dropdown = ft.Dropdown(
            label="Columna objetivo",
            disabled=True,
            on_change=self.handle_target_change,
            width=300,
            border_radius=10,
            filled=True,
            bgcolor=ft.colors.WHITE,
            border_color=ft.colors.GREY_400
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
            value="lasso",
            width=300,
            border_radius=10,
            filled=True,
            bgcolor=ft.colors.WHITE,
            border_color=ft.colors.GREY_400
        )
        
        # Modifica la definición del botón de entrenamiento para hacerlo más visible
        self.train_button = ft.ElevatedButton(
            text="ENTRENAR MODELO",  # Texto en mayúsculas para mayor énfasis
            icon=ft.icons.MODEL_TRAINING,
            on_click=self.handle_train,
            disabled=True,
            style=ft.ButtonStyle(
                color=ft.colors.WHITE,
                bgcolor="#FF5722",  # Color más llamativo (naranja intenso)
                elevation=8,  # Mayor elevación para destacarlo
                shape=ft.RoundedRectangleBorder(radius=10),
                padding=20,  # Padding más grande
            ),
            width=300,  # Ancho fijo para hacerlo más grande
            height=60,  # Altura específica para hacerlo más visible
        )
        
        # Construir la vista principal con tarjetas (Cards) para un aspecto más moderno
        self.main_view = ft.Column(
            controls=[
                # Encabezado con logo Uniandes
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Row([self.logo, ft.Text("Predictor de Series Temporales", size=30)], spacing=10),
                            self.uniandes_logo
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN
                    ),
                    margin=ft.margin.only(bottom=20)
                ),
                
                # Tarjeta 1: Selección de archivo
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text("Cargar datos", size=18, weight=ft.FontWeight.BOLD),
                            ft.Row([self.select_file_button, self.file_path_text]),
                            ft.Row([self.reset_button], alignment=ft.MainAxisAlignment.END)
                        ]),
                        padding=20
                    ),
                    elevation=5,
                    margin=ft.margin.only(bottom=20)
                ),
                
                # Tarjeta 2: Vista previa de datos
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            # Fila con título y botón de alternar
                            ft.Row([
                                self.data_preview_title,
                                self.toggle_preview_button
                            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                            self.data_info,
                            self.data_preview_container  # Contenedor con control de visibilidad
                        ]),
                        padding=20
                    ),
                    elevation=5,
                    margin=ft.margin.only(bottom=20)
                ),
                
                # Tarjeta 3: Opciones de procesamiento
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text("Configuración del modelo", size=18, weight=ft.FontWeight.BOLD),
                            ft.Container(
                                content=ft.Column([
                                    ft.Row(
                                        [self.preprocess_button],
                                        alignment=ft.MainAxisAlignment.CENTER
                                    ),
                                    ft.Row(
                                        [
                                            ft.Container(
                                                content=self.target_column_dropdown,
                                                margin=ft.margin.only(right=10)
                                            ),
                                            ft.Container(
                                                content=self.model_type_dropdown,
                                                margin=ft.margin.only(left=10)
                                            )
                                        ],
                                        alignment=ft.MainAxisAlignment.CENTER
                                    ),
                                    ft.Row(
                                        [self.train_button],
                                        alignment=ft.MainAxisAlignment.CENTER
                                    )
                                ]),
                                padding=ft.padding.only(top=20)
                            )
                        ]),
                        padding=20
                    ),
                    elevation=5
                )
            ],
            scroll=ft.ScrollMode.AUTO
        )
    
    def setup_results_view(self):
        # Título de resultados
        self.results_title = ft.Text(
            "Resultados del Modelo", 
            size=30, 
            weight=ft.FontWeight.BOLD,
            color="#1E88E5"
        )
        
        # Métricas de rendimiento
        self.results_metrics = ft.Text(
            "Sin resultados disponibles", 
            color=ft.colors.GREY_600,
            size=16
        )
        
        # Gráfico
        self.chart_container = ft.Container(
            content=ft.Text("El gráfico se mostrará aquí después del entrenamiento"),
            alignment=ft.alignment.center,
            height=400,
            border=ft.border.all(1, ft.colors.GREY_300),
            border_radius=8,
            bgcolor=ft.colors.WHITE
        )
        
        # Detalles del modelo
        self.model_details = ft.Text("", size=16)
        
        # Botones de navegación
        self.back_button = ft.ElevatedButton(
            text="Volver al inicio",
            icon=ft.icons.ARROW_BACK,
            on_click=lambda e: self.show_view("main"),
            style=ft.ButtonStyle(
                color=ft.colors.WHITE,
                bgcolor="#1E88E5",
                elevation=5,
                shape=ft.RoundedRectangleBorder(radius=10),
                padding=15
            )
        )
        
        # Construir la vista de resultados con tarjetas
        self.results_view = ft.Column(
            controls=[
                # Encabezado con logo Uniandes
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Row([self.logo, self.results_title], spacing=10),
                            self.uniandes_logo
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN
                    ),
                    margin=ft.margin.only(bottom=20)
                ),
                # Fila con métricas y detalles del modelo
                ft.Row(
                    [
                        # Tarjeta Métricas
                        ft.Card(
                            content=ft.Container(
                                content=ft.Column([
                                    ft.Text("Métricas del modelo", size=18, weight=ft.FontWeight.BOLD),
                                    ft.Container(
                                        content=self.results_metrics,
                                        padding=15,
                                        bgcolor="#F9F9F9",
                                        border_radius=8
                                    )
                                ]),
                                padding=20
                            ),
                            elevation=5,
                            expand=True,
                            margin=ft.margin.only(right=10)
                        ),
                        # Tarjeta Detalles del modelo
                        ft.Card(
                            content=ft.Container(
                                content=ft.Column([
                                    ft.Text("Detalles del modelo", size=18, weight=ft.FontWeight.BOLD),
                                    ft.Container(
                                        content=self.model_details,
                                        padding=15,
                                        bgcolor="#F9F9F9",
                                        border_radius=8
                                    )
                                ]),
                                padding=20
                            ),
                            elevation=5,
                            expand=True,
                            margin=ft.margin.only(left=10)
                        )
                    ],
                    spacing=20,
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN
                ),
                # Tarjeta Gráfico de predicciones
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text("Visualización de predicciones", size=18, weight=ft.FontWeight.BOLD),
                            self.chart_container
                        ]),
                        padding=20
                    ),
                    elevation=5,
                    margin=ft.margin.only(bottom=20)
                ),
                # Navegación
                ft.Row([self.back_button], alignment=ft.MainAxisAlignment.CENTER)
            ],
            scroll=ft.ScrollMode.AUTO
        )
    
    def handle_reset(self, e):
        """Reinicia la aplicación al estado inicial"""
        # Restablecer variables de estado
        self.data = None
        self.filepath = None
        self.processed_data = None
        self.target_column = None
        self.model = None
        
        # Restablecer UI
        self.file_path_text.value = "Ningún archivo seleccionado"
        self.data_info.value = "Sin datos cargados"
        self.data_preview.columns = [ft.DataColumn(ft.Text("Sin datos"))]
        self.data_preview.rows = []
        
        # Deshabilitar botones
        self.preprocess_button.disabled = True
        self.target_column_dropdown.disabled = True
        self.target_column_dropdown.options = []
        self.model_type_dropdown.disabled = True
        self.train_button.disabled = True
        
        # Actualizar la página
        self.page.update()
    
    def handle_logo_click(self, e):
        """Maneja los clics en el logo para activar el easter egg"""
        current_time = time.time()
        # Contar clics dentro de un periodo de 5 segundos
        if current_time - self.last_click_time <= 5:
            self.easter_egg_clicks += 1
        else:
            self.easter_egg_clicks = 1
        self.last_click_time = current_time
        # Activar el easter egg al alcanzar 5 clics
        if self.easter_egg_clicks >= 5:
            self.easter_egg_clicks = 0
            self.show_easter_egg()

    def show_easter_egg(self):
        """Muestra el easter egg 'MEGAOMEGA' en una vista separada"""
        if not hasattr(self, 'easter_egg_view'):
            # Crear la vista del easter egg si aún no existe
            self.setup_easter_egg_view()
        
        # Cambiar a la vista del easter egg
        self.show_view("easter_egg")

    def setup_easter_egg_view(self):
        """Configura la vista para el easter egg"""
        # Título del easter egg
        self.easter_egg_title = ft.Text(
            "¡MEGAOMEGA!", 
            size=40, 
            color="#FF5252", 
            weight=ft.FontWeight.BOLD,
            text_align=ft.TextAlign.CENTER
        )
        
        # Mensaje del easter egg
        self.easter_egg_message = ft.Text(
            "¡Has descubierto el easter egg!", 
            size=24, 
            text_align=ft.TextAlign.CENTER
        )
        
        # Créditos
        self.easter_egg_credits = ft.Text(
            "Desarrollado con 💙 por los mejores", 
            size=20, 
            text_align=ft.TextAlign.CENTER
        )
        
        # GIF animado
        self.easter_egg_animation = ft.Image(
            src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOGRlMHoydXB5bHl0dXFienhhbmw3OXR3Y2NzN2ZseW44dGJqaDR1ZyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/GeimqsH0TLDt4tScGw/giphy.gif",
            width=400,
            height=300,
            fit=ft.ImageFit.CONTAIN
        )
        
        # Botón para volver
        self.easter_egg_back_button = ft.ElevatedButton(
            text="Volver al inicio",
            icon=ft.icons.ARROW_BACK,
            on_click=lambda e: self.show_view("main"),
            style=ft.ButtonStyle(
                color=ft.colors.WHITE,
                bgcolor="#1E88E5",
                elevation=5,
                shape=ft.RoundedRectangleBorder(radius=10),
                padding=15
            )
        )
        
        # Construir la vista completa con una tarjeta y header Uniandes
        self.easter_egg_view = ft.Column(
            controls=[
                # Contenido principal
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Row([self.easter_egg_title, self.uniandes_logo], spacing=10, alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                            ft.Divider(),
                            self.easter_egg_message,
                            self.easter_egg_credits,
                            ft.Container(
                                content=self.easter_egg_animation,
                                alignment=ft.alignment.center,
                                padding=20
                            ),
                        ]),
                        padding=40,
                        alignment=ft.alignment.center
                    ),
                    elevation=10,
                    margin=ft.margin.only(top=50, bottom=30),
                ),
                # Botón para volver
                ft.Container(
                    content=self.easter_egg_back_button,
                    alignment=ft.alignment.center
                )
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
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
        elif view_name == "easter_egg":
            self.page.add(self.easter_egg_view)
        
        self.page.update()
    
    def pick_files(self, e):
        def on_dialog_result(e: ft.FilePickerResultEvent):
            if e.files and hasattr(e.files[0], 'path') and e.files[0].path is not None:
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
                self.file_path_text.value = "Ningún archivo seleccionado"
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
            # Mostrar indicador de carga
            self.preprocess_button.text = "Procesando..."
            self.preprocess_button.disabled = True
            self.page.update()
            
            # Procesar datos
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
            
            # Restaurar botón
            self.preprocess_button.text = "Preprocesar datos"
            self.preprocess_button.disabled = False
            
            self.page.update()
    
    def handle_target_change(self, e):
        self.target_column = e.data
        self.model_type_dropdown.disabled = False  # Habilitar el dropdown de modelos
        self.train_button.disabled = False
        self.page.update()
    
    def handle_train(self, e):
        if self.processed_data is not None and self.target_column:
            try:
                # Mostrar indicador de carga
                self.train_button.text = "Entrenando..."
                self.train_button.disabled = True
                self.page.update()
                
                # Dividir los datos
                self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(
                    self.processed_data, self.target_column
                )
                
                # Entrenar el modelo
                model_type = self.model_type_dropdown.value
                # Aplanar y_train para sklearn
                y_train_flat = self.y_train.values.ravel()
                self.model = self.train_model(
                    self.X_train.values.reshape(-1, 1),
                    y_train_flat,
                    model_type=model_type
                )
                
                # Predecir
                self.y_pred = self.model.predict(self.X_test.values.reshape(-1, 1))
                
                # Evaluar
                mse = mean_squared_error(self.y_test, self.y_pred)
                r2 = r2_score(self.y_test, self.y_pred)
                
                # Mostrar resultados con formato mejorado
                self.results_metrics.value = f"""
📊 **Métricas de evaluación:**

🔹 **MSE (Error Cuadrático Medio):** {mse:.4f}
🔹 **R² (Coeficiente de determinación):** {r2:.4f}

Una R² más cercana a 1 indica un mejor ajuste del modelo.
                """
                
                # Mostrar detalles del modelo
                self.model_details.value = f"""
📋 **Información del modelo:**

🔸 **Tipo de modelo:** {model_type.upper()}
🔸 **Columna objetivo:** {self.target_column}
🔸 **Tamaño del conjunto de entrenamiento:** {self.X_train.shape[0]} filas
🔸 **Tamaño del conjunto de prueba:** {self.X_test.shape[0]} filas
                """
                
                # Crear y mostrar gráfico
                self.update_chart()
                
                # Restaurar botón
                self.train_button.text = "Entrenar modelo"
                self.train_button.disabled = False
                
                # Cambiar a la vista de resultados
                self.show_view("results")
                
            except Exception as ex:
                self.train_button.text = "Entrenar modelo"
                self.train_button.disabled = False
                self.results_metrics.value = f"Error durante el entrenamiento: {ex}"
                self.page.update()
    
    def update_chart(self):
        # Crear figura y ejes
        plt.style.use('seaborn-v0_8')  # estilo moderno
        fig, ax = plt.subplots(figsize=(12, 6))
        # Graficar valores reales y predicciones
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.plot(self.X_test, self.y_test, 'o', label='Valores reales', color='#1E88E5', alpha=0.8, markersize=5)
        ax.plot(self.X_test, self.y_pred, 'o-', label='Predicciones', color='#FFA726', alpha=0.8, markersize=5)
        # Calcular proyección en el mismo eje
        last_index = len(self.processed_data) - 1
        future_end = last_index * 2
        future_x = np.arange(last_index + 1, future_end + 1)
        future_y = self.model.predict(future_x.reshape(-1, 1))
        ax.plot(future_x, future_y, '--', label='Proyección', color='#66BB6A', alpha=0.8)
        # Ajustar título, etiquetas y leyenda
        ax.set_title('Predicciones y proyección hasta doble índice', fontsize=16, fontweight='bold')
        ax.set_xlabel('Ciclos de carga', fontsize=12)
        ax.set_ylabel(f'Valor de {self.target_column}', fontsize=12)
        ax.legend(loc='best', frameon=True, fontsize=10)
        plt.tight_layout()
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
            X, y, test_size=test_size, random_state=random_state, shuffle=True
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

    def toggle_data_preview(self, e):
        """Alterna la visibilidad de la vista previa de datos"""
        self.show_preview = not self.show_preview
        
        # Actualizar el ícono según el estado
        if self.show_preview:
            self.toggle_preview_button.icon = ft.icons.VISIBILITY_OFF
            self.toggle_preview_button.tooltip = "Ocultar vista previa"
        else:
            self.toggle_preview_button.icon = ft.icons.VISIBILITY
            self.toggle_preview_button.tooltip = "Mostrar vista previa"
        
        # Actualizar la visibilidad del contenedor
        self.data_preview_container.visible = self.show_preview
        
        self.page.update()

def main(page: ft.Page):
    app = PredictorApp(page)

# Iniciar la aplicación
if __name__ == "__main__":
    ft.app(target=main)