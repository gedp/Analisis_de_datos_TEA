# Análisis Comparativo de Detección de Autismo (TEA) por Grupos Etarios (AQ-10) 🧩📊🧠

## Descripción ℹ️

Este repositorio contiene un análisis exploratorio y comparativo de datos de detección del Trastorno del Espectro Autista (TEA) para tres grupos etarios distintos: niños, adolescentes y adultos. El análisis se basa en los resultados del cuestionario AQ-10 y variables demográficas asociadas.

## Objetivo 🎯

El objetivo principal de este proyecto es investigar y visualizar cómo varían los patrones de detección del TEA, las puntuaciones del test AQ-10, la influencia de factores demográficos (como género, etnia, historia familiar) y los perfiles sintomáticos entre:

1.  **Niños** 🧒 (4-11 años)
2.  **Adolescentes** 🧑 (12-16 años)
3.  **Adultos** 👨‍👩‍👧‍👦 (18+ años)

Se busca identificar diferencias significativas y patrones específicos de cada grupo que puedan tener implicaciones para la detección temprana y el desarrollo de herramientas de screening adaptadas. 🤔

## Datos Utilizados 💾📄

Se emplearon tres conjuntos de datos distintos, cada uno correspondiente a un grupo etario, obtenidos originalmente para la investigación sobre métodos de screening de TEA.

* **Fuente Original:** Fadi Fayez Thabtah, Department of Digital Technology, Manukau Institute of Technology, Auckland, New Zealand. 🧑‍🔬
* **Metodología Base:** Cuestionario AQ-10 (10 preguntas sobre comportamientos asociados al TEA ❓) complementado con 10 características individuales/demográficas.
* **Atributos:** Cada registro contiene 21 atributos 🔢, incluyendo respuestas a las 10 preguntas (A1-A10), edad, género, etnia, país, historial de ictericia al nacer, historial familiar de TEA (PDD), quién completó el test, uso previo de apps de screening, y la puntuación final del test.
* **Tamaño de las Muestras:**
    * Niños: 292 registros
    * Adolescentes: 104 registros
    * Adultos: 704 registros
* **Tarea Principal (en origen):** Clasificación (predicción de posible TEA).
* **Nota:** Los conjuntos de datos originales pueden contener valores faltantes. ⚠️

## Análisis Realizado 📈🔬

El análisis presentado en este repositorio (visualizado a través del dashboard `index.html`) incluye:

* Introducción al TEA y al test AQ-10.
* Descripción detallada de las muestras por grupo etario.
* Comparación de tasas de detección positiva (diagnóstico) entre grupos.
* Análisis de la distribución de las puntuaciones totales del AQ-10.
* Análisis por género: distribución dentro de cada grupo y tasas de detección comparadas.
* Evaluación de factores de riesgo asociados (historia familiar de TEA, ictericia neonatal).
* Análisis de ítems específicos del AQ-10 para determinar su poder discriminativo en cada grupo.
* Identificación de perfiles de síntomas característicos por edad.
* Exploración de modelos predictivos (Regresión lineal) y la importancia relativa de las variables.
* Conclusiones principales y recomendaciones derivadas del análisis. 💡

## Visualización Interactiva (Dashboard) 🖥️✨

Los resultados completos del análisis, junto con gráficos interactivos, se encuentran en el archivo `index.html`. Este dashboard permite explorar visualmente:

* Tasas de detección por edad y género.
* Distribución de puntuaciones.
* Impacto de factores de riesgo.
* Análisis detallado por ítem del AQ-10.
* Comparativa de perfiles sintomáticos.
* Rendimiento de modelos predictivos.

**Para ver el dashboard:** Simplemente descarga el repositorio y abre el archivo `index.html` o el `reporte_profiling_asd_en.html` en tu navegador web preferido.

## Aplicación Interactiva Streamlit 🚀

¡Explora el análisis de forma interactiva aquí! 👉 <a href="https://adata-tea-bcf.streamlit.app/" target="_blank" rel="noopener noreferrer">adata-tea-bcf.streamlit.app</a>

Esta aplicación web, desarrollada con Streamlit, te permite navegar y visualizar de manera dinámica los hallazgos clave de este estudio sobre el Trastorno del Espectro Autista (TEA) 🧠.

**¿Qué puedes hacer en la aplicación?**

* 📊 **Visualizar Datos Clave:** Explora gráficos interactivos sobre tasas de detección, distribución de puntuaciones AQ-10, análisis por género y factores de riesgo (como ictericia o historial familiar) para cada grupo etario (niños 🧒, adolescentes 🧑, adultos 👨‍👩‍👧‍👦).
* 🔬 **Comparar Grupos:** Observa fácilmente las diferencias y similitudes en los perfiles de síntomas y la relevancia de distintas preguntas del test AQ-10 entre los diferentes grupos de edad.
* 💡 **Entender los Resultados:** La aplicación presenta los resultados del análisis comparativo de una forma accesible y fácil de interpretar.
* 🖥️ **Interfaz Amigable:** Interactúa con los datos y gráficos para profundizar en los aspectos que más te interesen del análisis.

## Tecnologías Utilizadas 💻🌐

*Jupyter Notebook para el código en python
* HTML5
* CSS3 (Tailwind CSS 🎨)
* JavaScript (Chart.js para las gráficas 📊)

## Fuente Original de los Datos y Citación 📚🙏

Los datos fueron proporcionados por Fadi Fayez Thabtah (fadi.fayez@manukau.ac.nz 📧). Si utilizas estos datos o el análisis derivado, por favor considera citar los trabajos originales del autor:

* Tabtah, F. (2017). Autism Spectrum Disorder Screening: Machine Learning Adaptation and DSM-5 Fulfillment. *Proceedings of the 1st International Conference on Medical and Health Informatics 2017*, pp.1-6. Taichung City, Taiwan, ACM.
* Thabtah, F. (2017). *ASDTests. A mobile app for ASD screening*. Disponible en: www.asdtests.com
* Thabtah, F. (2017). Machine Learning in Autistic Spectrum Disorder Behavioural Research: A Review. *Informatics for Health and Social Care Journal*.
