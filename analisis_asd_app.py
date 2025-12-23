# -*- coding: utf-8 -*-
"""
Aplicación Streamlit para el Análisis exploratorio y comparativo 
de Datos sobre ASD en diferentes grupos de edad

Autor: [Gerlyn Eduardo Duarte]
Fecha: 2025-04-21 
Descripción: Carga datos de niños, adolescentes y adultos, realiza limpieza, 
             análisis exploratorio (EDA), comparaciones entre grupos, entrena 
             un modelo predictivo básico y presenta los resultados de forma 
             interactiva usando Streamlit.
"""

# ==============================================================================
# PASO 1: CONFIGURACIÓN DEL ENTORNO Y CARGA DE DATOS
# ==============================================================================

# --- 1.1 Importación de Librerías ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from ydata_profiling import ProfileReport
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import os # Para manejar rutas de archivos
# Ignorar advertencias futuras de Pandas (opcional, pero limpia la consola)
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# --- Configuración General de la Página Streamlit ---
st.set_page_config(
    page_title="Análisis ASD por grupo de edad",
    page_icon="🧠", 
    layout="wide", 
    initial_sidebar_state="expanded" 
)

# --- 1.2 Definición de Nombres de Archivo (CORREGIDO) ---
# Usamos os.path.join para que funcione en diferentes sistemas operativos
# Asume que el script está en la misma carpeta que los CSVs
# Usamos realpath para asegurar que funcione incluso si se ejecuta desde otro dir
try:
    script_dir = os.path.dirname(os.path.realpath(__file__)) 
except NameError: # __file__ no está definido si se ejecuta interactivamente (ej. IDE)
    script_dir = os.getcwd() # Usar directorio actual como fallback

archivo_ninos = os.path.join(script_dir, 'datos_ninos.csv') # <<< CORREGIDO
archivo_adolescentes = os.path.join(script_dir, 'datos_adolescentes.csv') # <<< CORREGIDO
archivo_adultos = os.path.join(script_dir, 'datos_adultos.csv') # <<< CORREGIDO

# --- Función para Cargar y Pre-procesar Inicialmente ---
@st.cache_data # Cachear los datos cargados para mejorar rendimiento
def cargar_y_preparar_datos(nombre_archivo, grupo):
    """Carga un archivo CSV, añade grupo_edad y maneja errores básicos."""
    if not os.path.exists(nombre_archivo):
         st.error(f"Error: No se pudo encontrar el archivo CSV: {nombre_archivo}. Asegúrate de que está en la misma carpeta que el script .py.")
         # Mostrar directorio actual para ayudar a depurar
         st.info(f"Directorio actual buscado: {os.path.dirname(nombre_archivo)}")
         st.stop() 
         return None 
         
    try:
        # Especificamos quotechar ya que lo identificamos antes
        df = pd.read_csv(nombre_archivo, sep=',', quotechar='"') 
        df['grupo_edad'] = grupo
        return df
    except Exception as e:
        st.error(f"Error al cargar o procesar el archivo {os.path.basename(nombre_archivo)}: {e}")
        st.stop()
        return None

# --- 1.3 Carga de Datos usando la Función ---
df_ninos = cargar_y_preparar_datos(archivo_ninos, 'Niño')
df_adolescentes = cargar_y_preparar_datos(archivo_adolescentes, 'Adolescente')
df_adultos = cargar_y_preparar_datos(archivo_adultos, 'Adulto')

# Verificar si todos los dataframes se cargaron
if df_ninos is not None and df_adolescentes is not None and df_adultos is not None:
    # --- 1.4 Concatenar los DataFrames ---
    df_completo = pd.concat([df_ninos, df_adolescentes, df_adultos], ignore_index=True)
    st.sidebar.success(f"Datos cargados ({df_completo.shape[0]} filas)") # Mensaje en sidebar
    
    # ==============================================================================
    # PASO 2: LIMPIEZA Y PREPROCESAMIENTO DE DATOS
    # ==============================================================================
    @st.cache_data # Cachear también el resultado de la limpieza
    def limpiar_preprocesar_datos(df_input):
        """Realiza la limpieza y preprocesamiento completo."""
        df_limpio = df_input.copy()
        print("Iniciando limpieza...") # Para consola local

        # Renombrar columnas
        df_limpio.rename(columns={'Class/ASD': 'Class_ASD', 'austim': 'autism_family_hist'}, inplace=True) 

        # Eliminar 'row ID'
        if 'row ID' in df_limpio.columns:
            df_limpio.drop(columns=['row ID'], inplace=True)
        
        # Manejar NaNs en 'age' (Eliminar filas)
        filas_antes_drop_age = df_limpio.shape[0]
        df_limpio.dropna(subset=['age'], inplace=True)
        filas_despues_drop_age = df_limpio.shape[0]
        num_filas_eliminadas_age = filas_antes_drop_age - filas_despues_drop_age
        print(f"Filas eliminadas por 'age' faltante: {num_filas_eliminadas_age}")

        # Imputar 'ethnicity' y 'relation' usando .loc para intentar evitar warnings
        df_limpio.loc[df_limpio['ethnicity'].isnull(), 'ethnicity'] = 'Desconocido'
        df_limpio.loc[df_limpio['relation'].isnull(), 'relation'] = 'Desconocido'
        # Rellenar cualquier otro NaN restante en estas columnas (por si acaso)
        df_limpio['ethnicity'].fillna('Desconocido', inplace=True)
        df_limpio['relation'].fillna('Desconocido', inplace=True)
        
        # Corregir tipos numéricos
        # Edad: Asegurar que es int después de quitar NaNs
        if pd.api.types.is_float_dtype(df_limpio['age']):
             df_limpio['age'] = df_limpio['age'].astype(int)
        elif not pd.api.types.is_integer_dtype(df_limpio['age']):
             print(f"Advertencia: tipo inesperado para age: {df_limpio['age'].dtype}. Intentando convertir a int.")
             df_limpio['age'] = pd.to_numeric(df_limpio['age'], errors='coerce').fillna(0).astype(int)

        score_cols = [f'A{i}_Score' for i in range(1, 11)]
        for col in score_cols:
            df_limpio[col] = pd.to_numeric(df_limpio[col], errors='coerce').fillna(0).astype(int)
        df_limpio['result'] = pd.to_numeric(df_limpio['result'], errors='coerce').fillna(0).astype(int)

        # Convertir yes/no a 1/0 de forma robusta
        map_yes_no = {'yes': 1, 'no': 0}
        cols_yes_no = ['jundice', 'autism_family_hist', 'used_app_before'] 
        for col in cols_yes_no:
            if col in df_limpio.columns:
                # Usamos .loc para la asignación para intentar evitar warnings
                df_limpio_col_str = df_limpio[col].astype(str).str.lower().str.strip()
                df_limpio_col_mapped = df_limpio_col_str.map(map_yes_no)
                
                if df_limpio_col_mapped.isnull().any():
                    moda = df_limpio_col_mapped.dropna().mode() 
                    if not moda.empty:
                        moda_value = moda.iloc[0] 
                        # Usamos fillna aquí ya que .loc puede ser complejo con boolean indexing y NaNs
                        df_limpio_col_mapped.fillna(moda_value, inplace=True)
                        print(f"Valores no reconocidos en '{col}' imputados con moda ({moda_value}).")
                    else:
                        df_limpio_col_mapped.fillna(0, inplace=True) 
                        print(f"Advertencia: No se pudo calcular moda para '{col}'. Imputado con 0.")
                
                # Convertir a Int64 y asignar de vuelta
                # Asegurarse que no queden NaNs antes de convertir a Int64
                if df_limpio_col_mapped.isnull().any():
                     print(f"Advertencia: Aún quedan NaNs en la columna {col} antes de convertir a Int64. Imputando con 0.")
                     df_limpio_col_mapped.fillna(0, inplace=True)
                     
                df_limpio[col] = df_limpio_col_mapped.astype('Int64')

        # Procesar Class_ASD (DECISIÓN: Eliminar filas con valores no válidos)
        map_asd = {'YES': 1, 'NO': 0}
        rows_before_asd_clean = df_limpio.shape[0]
        if 'Class_ASD' in df_limpio.columns:
            # Guardamos original para diagnóstico
            df_limpio['Class_ASD_original_str'] = df_limpio['Class_ASD'].astype(str) 
            # Procesamos: a string, mayúsculas, quitamos espacios
            df_limpio['Class_ASD_proc'] = df_limpio['Class_ASD_original_str'].str.upper().str.strip()
            # Mapeamos a 1/0, otros serán NaN
            df_limpio['Class_ASD'] = df_limpio['Class_ASD_proc'].map(map_asd)
            
            nan_count_asd = df_limpio['Class_ASD'].isnull().sum()
            if nan_count_asd > 0:
                print(f"ADVERTENCIA: Se encontraron {nan_count_asd} valores inválidos en 'Class_ASD'. Eliminando estas filas.")
                # Guardamos temporalmente los valores eliminados para mostrar advertencia
                valores_eliminados = df_limpio.loc[df_limpio['Class_ASD'].isnull(), 'Class_ASD_proc'].unique()
                # Mostramos advertencia en la app Streamlit
                st.warning(f"Se eliminaron {nan_count_asd} filas por tener valores inválidos en la columna de diagnóstico 'Class_ASD' (distintos de 'YES' o 'NO'). Valores encontrados: {valores_eliminados[:10]} {'...' if len(valores_eliminados)>10 else ''}")
                # Eliminamos las filas con NaN en Class_ASD
                df_limpio.dropna(subset=['Class_ASD'], inplace=True) # <<<--- DROPNA ACTIVO ---
                
            # Convertir a Int64 ahora que no debería haber NaNs
            if not df_limpio['Class_ASD'].isnull().any():
                 df_limpio['Class_ASD'] = df_limpio['Class_ASD'].astype('Int64')
            else:
                 # Si aún quedan NaNs (muy improbable), convertir a Float64
                 df_limpio['Class_ASD'] = df_limpio['Class_ASD'].astype('Float64')
                 print("Advertencia: Aún quedan NaNs en Class_ASD después de dropna. Convertido a Float64.")

            # Eliminar columnas temporales
            if 'Class_ASD_proc' in df_limpio.columns:
                df_limpio.drop(columns=['Class_ASD_proc'], inplace=True)
            if 'Class_ASD_original_str' in df_limpio.columns:
                 df_limpio.drop(columns=['Class_ASD_original_str'], inplace=True)
        else:
             print("Columna 'Class_ASD' no encontrada.")
        
        rows_after_asd_clean = df_limpio.shape[0]
        print(f"Filas eliminadas por Class_ASD inválida: {rows_before_asd_clean - rows_after_asd_clean}")

        # Convertir categóricas a tipo 'category'
        cols_categoricas = ['gender', 'ethnicity', 'contry_of_res', 'age_desc', 'relation', 'grupo_edad']
        for col in cols_categoricas:
             if col in df_limpio.columns:
                # Asegurar que no haya NaNs antes de convertir a category
                if df_limpio[col].isnull().any():
                     if not pd.api.types.is_string_dtype(df_limpio[col]) and not pd.api.types.is_categorical_dtype(df_limpio[col]):
                          df_limpio[col] = df_limpio[col].astype(str)
                     df_limpio[col].fillna('Desconocido', inplace=True)
                try:
                    df_limpio[col] = df_limpio[col].astype('category')
                except Exception as e:
                     print(f"Error al convertir '{col}' a category: {e}. Se dejará como object.")
                     df_limpio[col] = df_limpio[col].astype(object) # Fallback

        # Manejar Outliers de Edad
        edad_maxima_razonable = 100 
        filas_antes_drop_outlier = df_limpio.shape[0]
        df_limpio = df_limpio[df_limpio['age'] <= edad_maxima_razonable]
        filas_despues_drop_outlier = df_limpio.shape[0]
        num_filas_eliminadas_outlier = filas_antes_drop_outlier - filas_despues_drop_outlier
        if num_filas_eliminadas_outlier > 0:
             print(f"Filas eliminadas por outlier de edad (> {edad_maxima_razonable}): {num_filas_eliminadas_outlier}")
             st.warning(f"Se eliminaron {num_filas_eliminadas_outlier} filas por tener una edad outlier (> {edad_maxima_razonable} años).")

        # Eliminar 'age_desc' si aún existe
        if 'age_desc' in df_limpio.columns:
            df_limpio.drop(columns=['age_desc'], inplace=True)
        
        print(f"Limpieza completada. Forma final: {df_limpio.shape}")
        return df_limpio

    # Ejecutar la función de limpieza
    # Esta función ahora está cacheada por Streamlit
    df_final = limpiar_preprocesar_datos(df_completo)

    # Verificar si df_final quedó vacío después de la limpieza
    if df_final.empty:
        st.error("¡Error Crítico! El DataFrame quedó vacío después del proceso de limpieza. Revisa los datos de entrada y los pasos de limpieza en el código.")
        st.stop() # Detener la ejecución de la app si no hay datos

    # ==============================================================================
    # PASO 3: GENERACIÓN DE PERFIL DETALLADO (ydata-profiling)
    # ==============================================================================
    # Generar y guardar el reporte (se mostrará en su sección)
    @st.cache_resource # Cachear el reporte generado
    def generar_guardar_profiling(df):
        """Genera informe ydata-profiling, intenta español, fallback a inglés."""
        print("Generando informe ydata-profiling...") 
        # Definir nombres de archivo usando script_dir para guardarlos en la carpeta del script
        nombre_archivo_es = os.path.join(script_dir, "reporte_profiling_asd_es.html")
        nombre_archivo_en = os.path.join(script_dir, "reporte_profiling_asd_en.html")
        
        # Intentar generar en español
        try:
            profile = ProfileReport(df, 
                                    title="Informe de Perfilado - Datos ASD (Español)", 
                                    explorative=True, 
                                    minimal=False, 
                                    locale='es') # Intentamos en español
            profile.to_file(nombre_archivo_es)
            print(f"Informe ydata-profiling (Español) guardado como {nombre_archivo_es}")
            return nombre_archivo_es, True 
        except Exception as e_es:
            print(f"Error generando reporte en español: {e_es}. Intentando en inglés.")
            st.warning("No se pudo generar el informe de perfilado en español (puede ser por falta de archivos de idioma en el entorno), se generará en inglés.")
            # Fallback a Inglés
            try:
                 profile_en = ProfileReport(df, title="Profiling Report - ASD Data", explorative=True, minimal=False)
                 profile_en.to_file(nombre_archivo_en)
                 print(f"Informe ydata-profiling (Inglés) guardado como {nombre_archivo_en}")
                 return nombre_archivo_en, True 
            except Exception as e_en:
                 print(f"Error crítico generando reporte incluso en inglés: {e_en}")
                 st.error(f"No se pudo generar el informe de perfilado: {e_en}")
                 return None, False 

    # Solo generar si no se ha hecho antes en esta sesión de Streamlit
    if 'reporte_generado' not in st.session_state:
         if not df_final.empty:
             st.session_state.nombre_archivo_reporte, st.session_state.reporte_ok = generar_guardar_profiling(df_final)
             st.session_state.reporte_generado = True 
         else:
             st.session_state.reporte_generado = False
             st.session_state.reporte_ok = False
             print("DataFrame vacío, no se genera informe de profiling.")

    # ==============================================================================
    # PASO 7: INTERFAZ DE STREAMLIT (Definición de Secciones y Contenido)
    # ==============================================================================

    # --- Barra Lateral (Sidebar) ---
    st.sidebar.title("Navegación 🧭")
    opciones_sidebar = ["🏠 Inicio", 
                        "📊 Exploración de datos", 
                        "🆚 Comparación entre grupos", 
                        "📄 Informe Profiling detallado",
                        "🤖 Modelo predictivo (Básico)", 
                        "💻 Código fuente"]
                        
    reporte_disponible = st.session_state.get('reporte_ok', False)                    
    if not reporte_disponible:
         if "📄 Informe Profiling detallado" in opciones_sidebar:
              opciones_sidebar.remove("📄 Informe Profiling Detallado")
              
    pagina_seleccionada = st.sidebar.radio("Selecciona una sección:", options=opciones_sidebar)
    st.sidebar.markdown("---")
    st.sidebar.info("Proyecto Bootcamp Análisis de Datos\nGerlyn Eduardo Duarte")

    # --- Contenido Principal ---
    
    # -- Sección: Inicio --
    if pagina_seleccionada == "🏠 Inicio":
        st.title("🧠 Análisis de Datos para Cribado ASD")
        st.markdown("""
        Bienvenido al análisis interactivo de datos sobre el Trastorno del Espectro Autista (TEA), 
        comparando características entre **Niños (4-11 años)**, **Adolescentes (12-16 años)** y **Adultos (17+ años)**.

        **Contexto:** El TEA es una condición del neurodesarrollo cuyo diagnóstico temprano es clave. Este proyecto analiza datos de cuestionarios de cribado (AQ-10) y características demográficas para entender mejor los patrones asociados al TEA en diferentes edades.

        **Objetivo:** Explorar los datos, identificar diferencias y similitudes entre los grupos de edad, y construir un modelo predictivo básico para la clasificación de ASD.

        **Navegación:** Usa el menú de la izquierda para explorar las secciones.
        """)
        
        st.markdown("---")
        st.header("Resumen de los Datos Limpios:")
        
        if 'df_final' in locals() and not df_final.empty:
            col1, col2, col3 = st.columns(3)
            col1.metric("Registros Analizados", f"{df_final.shape[0]:,}")
            col2.metric("Variables (Columnas)", f"{df_final.shape[1]}")
            
            if 'Class_ASD' in df_final.columns and pd.api.types.is_numeric_dtype(df_final['Class_ASD']) and df_final['Class_ASD'].count() > 0:
                 asd_positivos = df_final['Class_ASD'].sum() 
                 total_validos_asd = df_final['Class_ASD'].count() 
                 asd_positivo_pct = (asd_positivos / total_validos_asd) * 100
                 col3.metric("Casos ASD Positivos (%)", f"{asd_positivo_pct:.1f}%")
            else:
                 col3.metric("Casos ASD Positivos (%)", "N/A")

            st.markdown("**Estadísticas descriptivas generales:**")
            try:
                st.dataframe(df_final.describe(include='all')) 
            except Exception as e_desc:
                st.warning(f"No se pudieron mostrar todas las estadísticas descriptivas: {e_desc}")
                st.dataframe(df_final.describe()) 
            
            st.markdown("**Primeras filas de datos limpios:**")
            st.dataframe(df_final.head())
        else:
            st.warning("Los datos limpios no están disponibles o están vacíos.")

    # -- Sección: Exploración de datos --
    elif pagina_seleccionada == "📊 Exploración de datos":
        st.header("📊 Análisis exploratorio de datos (EDA)")
        st.markdown("Exploración visual de las variables más importantes.")

        if 'df_final' in locals() and not df_final.empty:
            st.subheader("Distribuciones Univariadas")

            # Edad General y por Grupo (Plotly)
            with st.container(): 
                 st.markdown("**Distribución de edad (general y por grupo)**")
                 fig_edad_plotly = px.histogram(df_final, x="age", color="grupo_edad", 
                                                 marginal="box", 
                                                 histnorm='density',
                                                 barmode='overlay',
                                                 category_orders={"grupo_edad": ["Niño", "Adolescente", "Adulto"]},
                                                 title="Distribución de edad por grupo (con Boxplot marginal)")
                 fig_edad_plotly.update_layout(bargap=0.1)
                 st.plotly_chart(fig_edad_plotly, use_container_width=True)

            # Puntuación Total 'result' (Plotly)
            with st.container():
                st.markdown("**Distribución de la puntuación total (result)**")
                fig_result_plotly = px.histogram(df_final, x="result", nbins=11, 
                                                title="Distribución de la puntuación total (result)",
                                                labels={'result':'Puntuación total'})
                fig_result_plotly.update_layout(bargap=0.2, xaxis = dict(tickmode = 'linear', dtick = 1)) 
                st.plotly_chart(fig_result_plotly, use_container_width=True)

            # Columnas para otras distribuciones
            col1, col2 = st.columns(2)

            with col1:
                # Diagnóstico 'Class_ASD' (Plotly)
                st.markdown("**Distribución de casos ASD**")
                asd_labels_st = df_final['Class_ASD'].map({0: 'NO', 1: 'YES', 0.0: 'NO', 1.0: 'YES'}).fillna('Desconocido')
                fig_asd_plotly = px.histogram(df_final, x=asd_labels_st, color=asd_labels_st,
                                              title="Distribución de casos ASD",
                                              labels={'x':'Diagnóstico ASD'},
                                              category_orders={"x": ["NO", "YES"]}) 
                st.plotly_chart(fig_asd_plotly, use_container_width=True)
            
            with col2:
                # Género (Plotly)
                st.markdown("**Distribución por género**")
                fig_gender_plotly = px.histogram(df_final, x="gender", color="gender",
                                                title="Distribución por género")
                st.plotly_chart(fig_gender_plotly, use_container_width=True)

            st.subheader("Relaciones bivariadas")

            # Edad vs Resultado (Plotly)
            with st.container():
                st.markdown("**Edad vs. Puntuación total (Coloreado por ASD)**")
                color_map = {0: 'NO', 1: 'YES', 0.0: 'NO', 1.0: 'YES'}
                color_discrete_map_plotly = {'NO': 'rgba(50, 100, 200, 0.7)', 'YES': 'rgba(200, 50, 50, 0.7)'}
                
                fig_edad_result_plotly = px.scatter(df_final, x='age', y='result', 
                                                    color=df_final['Class_ASD'].map(color_map), 
                                                    title='Edad vs. Puntuación total',
                                                    labels={'age': 'Edad (años)', 'result': 'Puntuación', 'color': 'ASD'},
                                                    hover_data=['grupo_edad', 'gender'], 
                                                    color_discrete_map=color_discrete_map_plotly,
                                                    opacity=0.7) 
                st.plotly_chart(fig_edad_result_plotly, use_container_width=True)

            # Resultado vs Diagnóstico (Boxplot con Plotly)
            with st.container():
                st.markdown("**Puntuación total por diagnóstico ASD**")
                fig_res_asd_plotly = px.box(df_final, x=asd_labels_st, y='result', 
                                            color=asd_labels_st,
                                            title="Distribución de puntuación por diagnóstico ASD",
                                            labels={'x': 'Diagnóstico ASD', 'result': 'Puntuación total'},
                                            category_orders={"x": ["NO", "YES"]}, 
                                            points="all", 
                                            color_discrete_map=color_discrete_map_plotly) 
                st.plotly_chart(fig_res_asd_plotly, use_container_width=True)
            
            # Correlación Scores (Heatmap con Plotly - CORREGIDO)
            with st.container():
                st.markdown("**Correlación entre Scores individuales y resultados**")
                score_cols = [f'A{i}_Score' for i in range(1, 11)] # Asegurar que está definida
                columnas_scores_y_result = score_cols + ['result']
                cols_num_exist = [col for col in columnas_scores_y_result if col in df_final.columns and pd.api.types.is_numeric_dtype(df_final[col])]
                if cols_num_exist and len(cols_num_exist) > 1:
                     matriz_correlacion = df_final[cols_num_exist].corr()
                     fig_corr_plotly = px.imshow(matriz_correlacion, text_auto=".2f", 
                                                  aspect="auto",
                                                  color_continuous_scale='RdBu', # <<<--- CORREGIDO AQUÍ
                                                  title="Mapa de calor de correlación")
                     st.plotly_chart(fig_corr_plotly, use_container_width=True)
                else:
                     st.warning("No se pudieron calcular correlaciones.")
                     
        else:
            st.warning("No hay datos limpios disponibles para mostrar la exploración.")

    # -- Sección: Comparación entre Grupos --
    elif pagina_seleccionada == "🆚 Comparación entre grupos":
        st.header("🆚 Comparación formal entre grupos de edad")
        st.markdown("Análisis estadístico y visual para identificar diferencias entre niños, adolescentes y adultos.")

        if 'df_final' in locals() and not df_final.empty:
            st.subheader("Comparación de Puntuación Total (result)")
            # Boxplot Plotly
            fig_comp_res_plotly = px.box(df_final, x='grupo_edad', y='result', 
                                         color='grupo_edad',
                                         title="Comparación de puntuación por grupo de edad",
                                         labels={'grupo_edad': 'Grupo de Edad', 'result': 'Puntuación total'},
                                         category_orders={"grupo_edad": ["Niño", "Adolescente", "Adulto"]},
                                         points="all")
            st.plotly_chart(fig_comp_res_plotly, use_container_width=True)

            # Prueba Kruskal-Wallis
            st.markdown("**Prueba estadística (Kruskal-Wallis) para 'result':**")
            try:
                group_n = df_final[df_final['grupo_edad'] == 'Niño']['result'].dropna()
                group_a = df_final[df_final['grupo_edad'] == 'Adolescente']['result'].dropna()
                group_d = df_final[df_final['grupo_edad'] == 'Adulto']['result'].dropna()
                
                if not group_n.empty and not group_a.empty and not group_d.empty:
                    stat_res, p_res = stats.kruskal(group_n, group_a, group_d)
                    st.write(f"* Estadístico H: {stat_res:.3f}")
                    st.write(f"* P-valor: {p_res:.3g}")
                    if p_res < 0.05:
                        st.success("    Conclusión: Diferencias significativas en 'result' entre grupos (p < 0.05).")
                    else:
                        st.warning("    Conclusión: No hay diferencias significativas en 'result' entre grupos (p >= 0.05).")
                else:
                     st.warning("Uno o más grupos no tienen datos válidos para 'result'. No se pudo realizar la prueba Kruskal-Wallis.")
            except Exception as e:
                st.error(f"Error durante la prueba Kruskal-Wallis para 'result': {e}")

            st.subheader("Comparación de diagnóstico ASD (Class_ASD)")
            # Filtramos NaNs aquí también para la prueba y tabla
            df_test_asd = df_final.dropna(subset=['Class_ASD'])
            if not df_test_asd.empty and df_test_asd['Class_ASD'].nunique() > 1: 
                # Tabla de contingencia (proporciones)
                st.markdown("**Tabla de proporciones (%)**")
                asd_map_display = {0: 'NO', 1: 'YES', 0.0: 'NO', 1.0: 'YES'}
                tabla_contingencia_asd = pd.crosstab(df_test_asd['grupo_edad'], 
                                                     df_test_asd['Class_ASD'].map(asd_map_display), 
                                                     normalize='index') * 100
                st.dataframe(tabla_contingencia_asd.style.format("{:.1f}%"))
                
                # Gráfico de barras agrupadas con Plotly
                tabla_contingencia_asd_melt = tabla_contingencia_asd.reset_index().melt(id_vars='grupo_edad', var_name='Class_ASD', value_name='Porcentaje')
                fig_comp_asd_plotly = px.bar(tabla_contingencia_asd_melt, x='grupo_edad', y='Porcentaje', color='Class_ASD',
                                             barmode='group', title='Proporción de diagnóstico ASD por grupo de edad',
                                             category_orders={"grupo_edad": ["Niño", "Adolescente", "Adulto"]},
                                             labels={'grupo_edad':'Grupo de edad', 'Class_ASD':'Diagnóstico'})
                st.plotly_chart(fig_comp_asd_plotly, use_container_width=True)

                # Prueba Chi-Cuadrado
                st.markdown("**Prueba estadística (Chi-Cuadrado) para 'Class_ASD' vs 'grupo_edad':**")
                try:
                     contingencia_asd_abs = pd.crosstab(df_test_asd['grupo_edad'], df_test_asd['Class_ASD'])
                     if contingencia_asd_abs.shape[0] > 1 and contingencia_asd_abs.shape[1] > 1: 
                         chi2_asd, p_asd, dof_asd, expected_asd = stats.chi2_contingency(contingencia_asd_abs)
                         st.write(f"* Estadístico Chi2: {chi2_asd:.3f}")
                         st.write(f"* P-valor: {p_asd:.3g}")
                         if p_asd < 0.05:
                             st.success("    Conclusión: Asociación significativa entre grupo de edad y diagnóstico (p < 0.05).")
                         else:
                             st.warning("    Conclusión: No hay asociación significativa entre grupo de edad y diagnóstico (p >= 0.05).")
                     else:
                          st.warning("No hay suficientes datos o categorías (2x2 mínimo requerido) para realizar la prueba Chi-cuadrado para ASD.")
                except Exception as e:
                     st.error(f"Error durante la prueba Chi-cuadrado para ASD: {e}")
            else:
                 st.warning("No hay datos válidos o suficientes categorías en 'Class_ASD' para realizar la comparación.")
                 
            # Ejemplo comparación jundice
            st.subheader("Comparación de ictericia (jundice)")
            if 'jundice' in df_final.columns:
                 df_test_jundice = df_final.dropna(subset=['jundice']) 
                 if not df_test_jundice.empty and df_test_jundice['jundice'].nunique() > 1:
                     tabla_jun = pd.crosstab(df_test_jundice['grupo_edad'], df_test_jundice['jundice'].map({0:'No', 1:'Sí'}), normalize='index') * 100
                     st.dataframe(tabla_jun.style.format("{:.1f}%"))
                     # (Aquí se podría añadir gráfico Plotly y prueba Chi2 para jundice)
                 else:
                      st.warning("No hay datos válidos o suficientes categorías para 'jundice'.")

        else:
            st.warning("No hay datos limpios disponibles para mostrar comparaciones.")

    # -- Sección: Informe Profiling Detallado --
    elif pagina_seleccionada == "📄 Informe Profiling detallado":
        st.header("📄 Informe detallado de ydata-profiling")
        st.markdown("Informe interactivo con análisis exhaustivo de cada variable.")
        
        reporte_disponible_local = st.session_state.get('reporte_ok', False)
        if reporte_disponible_local:
            nombre_archivo_reporte_local = st.session_state.get('nombre_archivo_reporte', None)
            if nombre_archivo_reporte_local and os.path.exists(nombre_archivo_reporte_local):
                try:
                    with open(nombre_archivo_reporte_local, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=800, scrolling=True)
                    # Intenta ofrecer descarga
                    try:
                         with open(nombre_archivo_reporte_local, "rb") as fp:
                              st.download_button(
                                   label="Descargar informe en HTML",
                                   data=fp,
                                   file_name=os.path.basename(nombre_archivo_reporte_local),
                                   mime="text/html",
                              )
                    except Exception as down_e:
                         print(f"Error creando botón descarga: {down_e}") # Log error, no mostrar en app
                         st.info(f"Informe guardado como {os.path.basename(nombre_archivo_reporte_local)} en la carpeta del script.")

                except Exception as e:
                    st.error(f"Error al leer o mostrar el informe HTML ({nombre_archivo_reporte_local}): {e}")
            else:
                 st.error("El archivo del informe de perfilado no se encontró o no está disponible.")
        else:
             st.error("El informe de perfilado no pudo ser generado.")

    # -- Sección: Modelo Predictivo --
    elif pagina_seleccionada == "🤖 Modelo predictivo (Básico)":
        st.header("🤖 Modelo predictivo básico (Regresión logística)")
        st.markdown("""
        Entrenamiento de un modelo simple para predecir `Class_ASD` basado en las 
        otras características, como demostración para el Bootcamp. 
        **Nota:** Se utilizan solo las filas donde `Class_ASD` tenía un valor válido ('YES' o 'NO').
        """)

        if 'df_final' in locals() and not df_final.empty:
            try:
                # --- Preparación de Datos para el Modelo ---
                df_modelo = df_final.dropna(subset=['Class_ASD']).copy() 
                
                if not df_modelo.empty and df_modelo['Class_ASD'].nunique() > 1: # Necesitamos al menos 2 clases
                    st.subheader("Preparación de datos")
                    score_cols = [f'A{i}_Score' for i in range(1, 11)] # Redefinir por si acaso
                    features = score_cols + ['age', 'jundice', 'autism_family_hist', 'gender'] 
                    target = 'Class_ASD'
                    
                    features_existentes = [f for f in features if f in df_modelo.columns]
                    if len(features_existentes) < len(features):
                         st.warning(f"Faltan features para el modelo: {set(features) - set(features_existentes)}")
                    
                    if not features_existentes:
                         st.error("No hay features válidas disponibles para el modelo.")
                         st.stop()
                    else:
                        X = df_modelo[features_existentes]
                        y = df_modelo[target].astype(int) 

                        # One-Hot Encode 'gender' si existe
                        if 'gender' in X.columns:
                            X = pd.get_dummies(X, columns=['gender'], drop_first=True, dtype=int) 
                        
                        # Dividir datos
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y) 
                        st.write(f"Datos divididos: {X_train.shape[0]} entrenamiento, {X_test.shape[0]} prueba.")

                        # Escalar 'age' si existe
                        if 'age' in X_train.columns:
                            scaler = StandardScaler()
                            cols_a_escalar = ['age'] 
                            X_train = X_train.copy()
                            X_test = X_test.copy()
                            X_train.loc[:, cols_a_escalar] = scaler.fit_transform(X_train[cols_a_escalar])
                            X_test.loc[:, cols_a_escalar] = scaler.transform(X_test[cols_a_escalar])
                            st.write("Variable 'age' escalada (StandardScaler).")

                        # --- Entrenamiento del Modelo ---
                        st.subheader("Entrenamiento del modelo")
                        modelo = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced') 
                        modelo.fit(X_train, y_train)
                        st.success("Modelo de regresión logística entrenado.")

                        # --- Evaluación del Modelo ---
                        st.subheader("Evaluación del modelo (sobre datos de prueba)")
                        y_pred = modelo.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        cm = confusion_matrix(y_test, y_pred)
                        
                        # Obtener nombres de clases presentes en y_test o y_pred
                        present_classes_int = np.unique(np.concatenate((y_test, y_pred)))
                        target_names_report = [f"{'SI' if c==1 else 'NO'} ASD ({c})" for c in sorted(present_classes_int)]
                        if not target_names_report: target_names_report = ['Desconocido']
                        
                        report = classification_report(y_test, y_pred, target_names=target_names_report, zero_division=0)

                        st.metric("Accuracy (Precisión general)", f"{accuracy:.2%}")
                        
                        st.markdown("**Matriz de confusión:**")
                        # Usar Plotly para la matriz de confusión
                        try:
                            # Intentar obtener etiquetas correctas si ambas clases están presentes
                             labels_cm = ['NO ASD (0)', 'SI ASD (1)'] if len(present_classes_int) == 2 else [target_names_report[0]] # Si solo hay 1 clase
                             fig_cm_plotly = px.imshow(cm, text_auto=True, aspect="auto",
                                                       labels=dict(x="Predicción", y="Valor Real", color="Cantidad"),
                                                       x=[f"Pred {l}" for l in labels_cm],
                                                       y=[f"Real {l}" for l in labels_cm],
                                                       color_continuous_scale='Blues')
                             fig_cm_plotly.update_layout(title="Matriz de Confusión")
                             st.plotly_chart(fig_cm_plotly, use_container_width=False)
                        except Exception as cm_plot_e:
                             st.warning(f"No se pudo graficar la matriz de confusión: {cm_plot_e}")
                             st.text(f"Matriz:\n{cm}")


                        st.markdown("**Informe de clasificación:**")
                        st.text(report)
                else:
                     st.warning("No hay datos válidos con Class_ASD (o solo una clase) para entrenar/evaluar el modelo.")

            except Exception as e:
                st.error(f"Ocurrió un error durante el proceso de modelado: {e}")
                st.exception(e) 
        else:
            st.warning("No hay datos limpios disponibles para entrenar el modelo.")

    # -- Sección: Código Fuente --
    elif pagina_seleccionada == "💻 Código fuente":
        st.header("💻 Código fuente del análisis")
        st.markdown("Este es el script completo de Python utilizado para generar esta aplicación web.")
        
        try:
            with open(__file__, 'r', encoding='utf-8') as f:
                codigo = f.read()
            st.code(codigo, language='python')
        except NameError: 
             st.warning("No se pudo cargar automáticamente el código fuente (__file__ no definido). El código es el archivo .py que estás ejecutando.")
        except FileNotFoundError:
             st.warning("No se pudo encontrar este archivo de script para mostrar el código fuente.")
        except Exception as e:
            st.error(f"Error al leer el archivo de código fuente: {e}")

    # Mensaje final o pie de página (opcional)
    st.markdown("---")
    try:
        from datetime import datetime
        now = datetime.now()
        st.caption(f"Análisis generado el {now.strftime('%Y-%m-%d %H:%M:%S')} (Hora Local)") 
    except Exception: 
         st.caption(f"Análisis generado el {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
         
else:
     # Esto se mostraría si la carga inicial falló antes de concatenar
     st.error("La carga inicial de datos falló catastróficamente. Revisa los nombres/ubicación de los archivos CSV y la consola/terminal.")
