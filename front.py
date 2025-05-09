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
        self.page.title = "Predictor de Series Temporales - Edición Automotriz Premium"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.padding = 0
        self.page.bgcolor = ft.LinearGradient(
            begin=ft.alignment.top_left,
            end=ft.alignment.bottom_right,
            colors=["#181A20", "#23272F", "#2C2F38"]
        )
        self.page.window_width = 1100
        self.page.window_height = 850
        self.page.fonts = {
            "luxury": "https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap"
        }
        self.page.font_family = "luxury"
        
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
        # Encabezado premium con logo automotriz
        self.logo = ft.Container(
            content=ft.Row([
                ft.Icon(ft.icons.DIRECTIONS_CAR_FILLED, size=48, color="#FFD700"),
                ft.Text(
                    "AutoPredictor ML", 
                    size=44, 
                    weight=ft.FontWeight.BOLD, 
                    color="#FFD700",
                    font_family="luxury"
                )
            ]),
            on_click=self.handle_logo_click,
            margin=ft.margin.only(bottom=10, top=20),
            padding=ft.padding.symmetric(horizontal=20),
            border_radius=20,
            bgcolor=ft.LinearGradient(
                begin=ft.alignment.top_left,
                end=ft.alignment.bottom_right,
                colors=["#23272F", "#181A20"]
            ),
            shadow=ft.BoxShadow(blur_radius=16, color="#FFD70033", offset=ft.Offset(0,8))
        )
        
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
            color="#B0B0B0",
            size=18,
            font_family="luxury"
        )
        
        self.select_file_button = ft.ElevatedButton(
            text="Seleccionar CSV",
            icon=ft.icons.UPLOAD_FILE,
            on_click=self.pick_files,
            style=ft.ButtonStyle(
                color="#23272F",
                bgcolor="#FFD700",
                elevation=8,
                shape=ft.RoundedRectangleBorder(radius=16),
                padding=20
            ),
            tooltip="Sube tus datos automotrices"
        )
        
        # Sección 2: Vista previa de datos
        self.data_preview_title = ft.Text(
            "Vista previa de datos", 
            size=24, 
            weight=ft.FontWeight.BOLD,
            color="#FFD700",
            font_family="luxury"
        )
        
        self.toggle_preview_button = ft.IconButton(
            icon=ft.icons.VISIBILITY_OFF,
            tooltip="Ocultar vista previa",
            on_click=self.toggle_data_preview,
            icon_color="#FFD700",
            icon_size=28,
            style=ft.ButtonStyle(bgcolor="#23272F", shape=ft.CircleBorder(), elevation=4)
        )
        
        self.data_preview = ft.DataTable(
            columns=[ft.DataColumn(ft.Text("Sin datos", color="#FFD700"))],
            rows=[],
            border=ft.border.all(2, "#FFD700"),
            border_radius=12,
            vertical_lines=ft.border.BorderSide(1, "#444444"),
            horizontal_lines=ft.border.BorderSide(1, "#444444"),
            bgcolor="#23272F"
        )
        
        self.data_preview_container = ft.Container(
            content=self.data_preview,
            padding=ft.padding.only(top=10),
            visible=self.show_preview,
            border_radius=12,
            shadow=ft.BoxShadow(blur_radius=12, color="#FFD70022", offset=ft.Offset(0,4))
        )
        
        self.data_info = ft.Text(
            "Sin datos cargados", 
            color="#B0B0B0",
            size=18,
            font_family="luxury"
        )
        
        # Botón para retroceder/reiniciar
        self.reset_button = ft.ElevatedButton(
            text="Reiniciar",
            icon=ft.icons.REFRESH,
            on_click=self.handle_reset,
            style=ft.ButtonStyle(
                color="#23272F",
                bgcolor="#FFD700",
                elevation=8,
                shape=ft.RoundedRectangleBorder(radius=16),
                padding=20
            )
        )
        
        # Sección 3: Preprocesamiento
        self.preprocess_button = ft.ElevatedButton(
            text="Preprocesar datos",
            icon=ft.icons.CLEANING_SERVICES,
            on_click=self.handle_preprocess,
            disabled=True,
            style=ft.ButtonStyle(
                color="#23272F",
                bgcolor="#FFD700",
                elevation=8,
                shape=ft.RoundedRectangleBorder(radius=16),
                padding=20
            )
        )
        
        # Sección 4: Selección de columna objetivo
        self.target_column_dropdown = ft.Dropdown(
            label="Columna objetivo",
            disabled=True,
            on_change=self.handle_target_change,
            width=320,
            border_radius=14,
            filled=True,
            bgcolor="#23272F",
            border_color="#FFD700",
            color="#FFD700",
            label_style=ft.TextStyle(color="#FFD700", font_family="luxury")
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
            width=320,
            border_radius=14,
            filled=True,
            bgcolor="#23272F",
            border_color="#FFD700",
            color="#FFD700",
            label_style=ft.TextStyle(color="#FFD700", font_family="luxury")
        )
        
        self.train_button = ft.ElevatedButton(
            text="ENTRENAR MODELO",
            icon=ft.icons.MODEL_TRAINING,
            on_click=self.handle_train,
            disabled=True,
            style=ft.ButtonStyle(
                color="#23272F",
                bgcolor="#FFD700",
                elevation=12,
                shape=ft.RoundedRectangleBorder(radius=18),
                padding=28,
            ),
            width=340,
            height=70,
            tooltip="Entrena tu modelo premium"
        )
        
        # Construir la vista principal con tarjetas (Cards) para un aspecto más moderno
        self.main_view = ft.Column(
            controls=[
                # Encabezado
                ft.Container(
                    content=ft.Row([
                        self.logo,
                        ft.Text("Predictor de Series Temporales para Autos de Lujo", size=32, color="#FFD700", font_family="luxury", weight=ft.FontWeight.BOLD)
                    ]),
                    margin=ft.margin.only(bottom=24)
                ),
                
                # Tarjeta 1: Selección de archivo
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text("Cargar datos automotrices", size=20, weight=ft.FontWeight.BOLD, color="#FFD700", font_family="luxury"),
                            ft.Row([self.select_file_button, self.file_path_text]),
                            ft.Row([self.reset_button], alignment=ft.MainAxisAlignment.END)
                        ]),
                        padding=28,
                        bgcolor=ft.LinearGradient(
                            begin=ft.alignment.top_left,
                            end=ft.alignment.bottom_right,
                            colors=["#23272F", "#181A20"]
                        ),
                        border_radius=18
                    ),
                    elevation=10,
                    margin=ft.margin.only(bottom=24),
                    shadow_color="#FFD70033"
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
                            self.data_preview_container
                        ]),
                        padding=28,
                        bgcolor=ft.LinearGradient(
                            begin=ft.alignment.top_left,
                            end=ft.alignment.bottom_right,
                            colors=["#23272F", "#181A20"]
                        ),
                        border_radius=18
                    ),
                    elevation=10,
                    margin=ft.margin.only(bottom=24),
                    shadow_color="#FFD70033"
                ),
                
                # Tarjeta 3: Opciones de procesamiento
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text("Configuración del modelo premium", size=20, weight=ft.FontWeight.BOLD, color="#FFD700", font_family="luxury"),
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
                                                margin=ft.margin.only(right=12)
                                            ),
                                            ft.Container(
                                                content=self.model_type_dropdown,
                                                margin=ft.margin.only(left=12)
                                            )
                                        ],
                                        alignment=ft.MainAxisAlignment.CENTER
                                    ),
                                    ft.Row(
                                        [self.train_button],
                                        alignment=ft.MainAxisAlignment.CENTER
                                    )
                                ]),
                                padding=ft.padding.only(top=24)
                            )
                        ]),
                        padding=28,
                        bgcolor=ft.LinearGradient(
                            begin=ft.alignment.top_left,
                            end=ft.alignment.bottom_right,
                            colors=["#23272F", "#181A20"]
                        ),
                        border_radius=18
                    ),
                    elevation=10,
                    shadow_color="#FFD70033"
                )
            ],
            scroll=ft.ScrollMode.AUTO
        )
    
    def setup_results_view(self):
        self.results_title = ft.Text(
            "Resultados del Modelo Premium", 
            size=34, 
            weight=ft.FontWeight.BOLD,
            color="#FFD700",
            font_family="luxury"
        )
        self.results_metrics = ft.Text(
            "Sin resultados disponibles", 
            color="#B0B0B0",
            size=18,
            font_family="luxury"
        )
        self.chart_container = ft.Container(
            content=ft.Text("El gráfico se mostrará aquí después del entrenamiento", color="#FFD700"),
            alignment=ft.alignment.center,
            height=350,  # Reducido de 420 a 300
            border=ft.border.all(2, "#FFD700"),
            border_radius=14,
            bgcolor="#23272F",
            shadow=ft.BoxShadow(blur_radius=12, color="#FFD70022", offset=ft.Offset(0,4))
        )
        self.model_details = ft.Text("", size=18, color="#FFD700", font_family="luxury")
        self.back_button = ft.ElevatedButton(
            text="Volver al inicio",
            icon=ft.icons.ARROW_BACK,
            on_click=lambda e: self.show_view("main"),
            style=ft.ButtonStyle(
                color="#23272F",
                bgcolor="#FFD700",
                elevation=8,
                shape=ft.RoundedRectangleBorder(radius=16),
                padding=20
            )
        )
        self.results_view = ft.Column(
            controls=[
                ft.Container(
                    content=ft.Row([self.logo, self.results_title]),
                    margin=ft.margin.only(bottom=24)
                ),
                ft.Row([
                    ft.Card(
                        content=ft.Container(
                            content=ft.Column([
                                ft.Text("Métricas del modelo", size=20, weight=ft.FontWeight.BOLD, color="#FFD700", font_family="luxury"),
                                ft.Container(
                                    content=self.results_metrics,
                                    padding=18,
                                    bgcolor="#23272F",
                                    border_radius=10
                                )
                            ]),
                            padding=24
                        ),
                        elevation=10,
                        expand=True,
                        margin=ft.margin.only(right=12),
                        shadow_color="#FFD70033"
                    ),
                    ft.Card(
                        content=ft.Container(
                            content=ft.Column([
                                ft.Text("Detalles del modelo", size=20, weight=ft.FontWeight.BOLD, color="#FFD700", font_family="luxury"),
                                ft.Container(
                                    content=self.model_details,
                                    padding=18,
                                    bgcolor="#23272F",
                                    border_radius=10
                                )
                            ]),
                            padding=24
                        ),
                        elevation=10,
                        expand=True,
                        margin=ft.margin.only(left=12),
                        shadow_color="#FFD70033"
                    )
                ], spacing=24, alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text("Visualización de predicciones", size=20, weight=ft.FontWeight.BOLD, color="#FFD700", font_family="luxury"),
                            self.chart_container
                        ]),
                        padding=24
                    ),
                    elevation=10,
                    margin=ft.margin.only(bottom=24),
                    shadow_color="#FFD70033"
                ),
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
        self.easter_egg_title = ft.Text(
            "¡MEGAOMEGA!", 
            size=44, 
            color="#FFD700", 
            weight=ft.FontWeight.BOLD,
            text_align=ft.TextAlign.CENTER,
            font_family="luxury"
        )
        self.easter_egg_message = ft.Text(
            "¡Has descubierto el easter egg premium!", 
            size=28, 
            text_align=ft.TextAlign.CENTER,
            color="#FFD700",
            font_family="luxury"
        )
        self.easter_egg_credits = ft.Text(
            "Desarrollado con 💙 para el mundo automotriz de lujo", 
            size=22, 
            text_align=ft.TextAlign.CENTER,
            color="#FFD700",
            font_family="luxury"
        )
        self.easter_egg_animation = ft.Image(
            src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOGRlMHoydXB5bHl0dXFienhhbmw3OXR3Y2NzN2ZseW44dGJqaDR1ZyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/GeimqsH0TLDt4tScGw/giphy.gif",
            width=420,
            height=320,
            fit=ft.ImageFit.CONTAIN
        )
        self.easter_egg_back_button = ft.ElevatedButton(
            text="Volver al inicio",
            icon=ft.icons.ARROW_BACK,
            on_click=lambda e: self.show_view("main"),
            style=ft.ButtonStyle(
                color="#23272F",
                bgcolor="#FFD700",
                elevation=8,
                shape=ft.RoundedRectangleBorder(radius=16),
                padding=20
            )
        )
        self.easter_egg_view = ft.Column(
            controls=[
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            self.easter_egg_title,
                            ft.Divider(color="#FFD700"),
                            self.easter_egg_message,
                            self.easter_egg_credits,
                            ft.Container(
                                content=self.easter_egg_animation,
                                alignment=ft.alignment.center,
                                padding=24
                            ),
                        ]),
                        padding=48,
                        alignment=ft.alignment.center,
                        bgcolor=ft.LinearGradient(
                            begin=ft.alignment.top_left,
                            end=ft.alignment.bottom_right,
                            colors=["#23272F", "#181A20"]
                        ),
                        border_radius=24
                    ),
                    elevation=14,
                    margin=ft.margin.only(top=60, bottom=36),
                    shadow_color="#FFD70033"
                ),
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
        plt.clf()
        plt.cla()
        plt.close()
        
        # Usar un estilo sofisticado
        plt.style.use('dark_background')
        
        # Crear figura con dimensiones elegantes
        fig, ax = plt.subplots(figsize=(10, 5.5))
        
        # Colores premium
        bg_color = '#1A1A2E'  # Azul oscuro elegante
        grid_color = '#333366'  # Azul medio para la cuadrícula
        accent_gold = '#B8860B'  # Dorado oscuro elegante (más sofisticado)
        teal_color = '#20B2AA'  # Turquesa elegante
        
        # Configurar fondos
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        # Cuadrícula sutil
        ax.grid(True, linestyle='--', linewidth=0.5, color=grid_color, alpha=0.5)
        
        # Datos reales (puntos dorados sutiles)
        ax.scatter(
            self.X_test, self.y_test, 
            label='Valores reales', 
            color=accent_gold, 
            edgecolor=bg_color,
            s=50,
            alpha=0.85,
            zorder=3
        )
        
        # Línea de predicción (línea turquesa elegante)
        ax.plot(
            self.X_test, self.y_pred, 
            label='Predicción', 
            color=teal_color, 
            linewidth=2.5, 
            marker=None,
            alpha=0.9, 
            zorder=2
        )
        
        # Eliminar bordes de ejes para un look más limpio
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Etiquetas elegantes
        ax.set_title('Predicción Premium', fontsize=16, color='white', pad=10, fontweight='normal')
        ax.set_xlabel('Índice', fontsize=12, color='white', labelpad=8)
        ax.set_ylabel(f'{self.target_column}', fontsize=12, color='white', labelpad=8)
        
        # Ticks sutiles
        ax.tick_params(axis='both', colors='#CCCCCC', labelsize=10)
        
        # Leyenda elegante en la esquina superior derecha
        legend = ax.legend(
            loc='upper right',
            framealpha=0.7,
            facecolor=bg_color,
            edgecolor='#444444',
            fontsize=10
        )
        
        for text in legend.get_texts():
            text.set_color('white')
        
        plt.tight_layout()
        
        # Convertir a widget de Flet
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