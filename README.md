# AutoPredictor ML - Predictor de Series Temporales

![Predictor Premium](https://img.shields.io/badge/Aplicación-Premium-gold)
![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Flet](https://img.shields.io/badge/Flet-UI-purple.svg)

## Descripción

AutoPredictor ML es una aplicación de escritorio elegante y potente especializada en la predicción de series temporales para el sector automotriz de lujo. La aplicación permite a los usuarios cargar sus propios datos en formato CSV, preprocesarlos automáticamente, seleccionar variables objetivo y entrenar diferentes modelos de machine learning para generar predicciones precisas.

## Características

- **Interfaz Premium**: Diseño elegante con una paleta de colores exclusiva en tonos oscuros y dorados.
- **Carga de datos**: Soporte para archivos CSV con detección automática de columnas numéricas.
- **Preprocesamiento automático**: Elimina columnas no numéricas y filas con valores nulos.
- **Selección flexible**: Permite elegir la columna objetivo para las predicciones.
- **Múltiples modelos**: Soporta 5 algoritmos diferentes de regresión:
  - Lasso
  - Ridge
  - ElasticNet
  - SVR (Support Vector Regression)
  - SGD (Stochastic Gradient Descent)
- **Visualización premium**: Gráficos elegantes que muestran valores reales vs. predicciones.
- **Métricas de evaluación**: Cálculo automático de MSE y R² para evaluar la precisión del modelo.
- **Experiencia de usuario mejorada**: Incluye un easter egg secreto (¡descúbrelo haciendo clic repetidamente en el logo!).

## Requisitos

Ver [requirements.txt](requirements.txt) para la lista completa de dependencias.

## Instalación

1. Clona este repositorio o descarga los archivos.
2. Crea un entorno virtual (recomendado):

```powershell
python -m venv venv
.\venv\Scripts\activate
```

3. Instala las dependencias:

```powershell
pip install -r requirements.txt
```

## Uso

Para ejecutar la aplicación:

```powershell
python front.py
```

### Flujo de uso

1. Inicia la aplicación.
2. Haz clic en "Seleccionar CSV" para cargar tu conjunto de datos.
3. Haz clic en "Preprocesar datos" para limpiar automáticamente tus datos.
4. Selecciona la columna objetivo de la lista desplegable.
5. Elige el tipo de modelo que deseas entrenar.
6. Haz clic en "ENTRENAR MODELO" para iniciar el entrenamiento.
7. Visualiza los resultados en la pantalla de métricas y gráficos.

## Estructura de datos recomendada

Para obtener mejores resultados, se recomienda utilizar archivos CSV con:
- Variables numéricas
- Sin valores faltantes (aunque la app los maneja automáticamente)
- Datos de series temporales ordenados cronológicamente

## Contribuir

Si deseas contribuir a este proyecto, por favor:

1. Haz un fork del repositorio
2. Crea una rama para tu característica (`git checkout -b feature/AmazingFeature`)
3. Realiza tus cambios y haz commit (`git commit -m 'Añadir característica increíble'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

## Contacto

Email: tu-email@ejemplo.com

---

*Desarrollado con 💙 para el mundo automotriz de lujo*
