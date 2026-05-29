# ATP Sports Analytics Platform

¡Bienvenido a la **ATP Sports Analytics Platform**! Este proyecto es un pipeline avanzado de Ciencia de Datos y Machine Learning que analiza el rendimiento histórico del circuito profesional de la ATP, aplicando reducción dimensional (PCA) por superficies y generando reportes analíticos automatizados mediante Inteligencia Artificial.

🌐 **[¡Haz clic aquí para ver el Dashboard Interactivo en vivo!]*(AQUÍ PEGAS TU LINK DE GITHUB PAGES)***

---

## Características Principales

**Pipeline de Datos Completo:** Extracción, limpieza y análisis de características avanzadas a partir de datasets históricos de la ATP.
**Análisis Multivariante (PCA):** Reducción de dimensionalidad interactiva que separa el comportamiento de los jugadores de forma Global y desglosada por superficies (**Clay, Hard, Grass**).
**Insights Automatizados con IA:** Integración con la API de **Groq Cloud** utilizando el modelo **Llama-3.3-70b-versatile** para generar reportes técnicos detallados basados en los resultados matemáticos.
**Interfaz Web Moderna:** Dashboard multipestaña (*Tabs náticos*) construido puramente con HTML5, CSS3 y gráficos dinámicos de **Plotly Core**.

---

## Tecnologías Utilizadas

* **Lenguaje:** Python 3.12+
* **Análisis de Datos:** Pandas, NumPy, SciPy
* **Machine Learning:** Scikit-Learn (StandardScaler, PCA)
* **Visualización:** Plotly Core (Gráficos interactivos de dispersión, biplots, mapas de calor y barras)
* **Uso de IA:** Groq Cloud SDK (Llama 3.3)
* **Diseño Web:** HTML5 & CSS3 (Diseño modular, interactividad sin JS externo)

---

## Estructura del Proyecto

```text
Proyecto_ATP/
├── data/
│   └── raw/               # Archivos CSV con los datos crudos de la ATP
├── src/
│   ├── Cargar_data.py     # Módulo de lectura y unificación de archivos
│   ├── Procesamiento.py   # Limpieza y filtrado de datos analíticos
│   ├── Features.py        # Cálculo de métricas avanzadas y scores de dominancia
│   └── plots.py           # Construcción de gráficos interactivos con Plotly
├── Main.py                # Script principal que ejecuta todo el pipeline
├── interfaz.html          # Estructura del dashboard (Template de salida)
├── interfaz_decoracion.css# Estilos visuales y lógica de pestañas
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Portada del proyecto (Este archivo)