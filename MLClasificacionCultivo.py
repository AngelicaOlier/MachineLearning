# MLClasificacionCultivo.py

# --- Importaciones ---
import base64
from io import BytesIO # Para manejar buffers de bytes en memoria (Excel, Gráficos)
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier # El modelo de clasificación
from sklearn.preprocessing import StandardScaler, LabelEncoder # Para escalar datos numéricos y codificar categorías
from sklearn.model_selection import train_test_split # Para dividir datos en entrenamiento y prueba
from sklearn.metrics import ( # Métricas para evaluar el modelo
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)
import matplotlib.pyplot as plt # Para crear gráficos
import matplotlib
import seaborn as sns # Para mejorar la visualización de gráficos (matriz de confusión)
import joblib # Para guardar y cargar el modelo entrenado y los preprocesadores
import os # Para interactuar con el sistema operativo (crear directorios, construir rutas)
import io # (Ya importado arriba, pero para claridad en su uso con Excel)
import traceback # Para imprimir detalles completos de errores inesperados

# --- Configuración de Matplotlib ---
# Asegura que Matplotlib no intente usar una interfaz gráfica (necesario para servidores web)
matplotlib.use('Agg')

# --- Constantes ---
# Define el directorio donde se guardarán los artefactos del modelo
MODEL_DIR = "saved_models_es"
# Define las rutas completas para cada archivo a guardar
MODEL_PATH = os.path.join(MODEL_DIR, "adaboost_cultivo_modelo.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "escalador.joblib") # Guardar el escalador es crucial para preprocesar datos nuevos
ENCODER_PATH = os.path.join(MODEL_DIR, "codificador_etiquetas_suelo.joblib") # Guardar codificador de características
TARGET_ENCODER_PATH = os.path.join(MODEL_DIR, "codificador_etiquetas_cultivo.joblib") # Guardar codificador del objetivo (para decodificar predicciones)

# Crea el directorio para guardar modelos si no existe al iniciar el script
os.makedirs(MODEL_DIR, exist_ok=True)


# --- Función Auxiliar: Generar Gráfico de Matriz de Confusión ---
def generar_grafico_matriz_confusion(y_true, y_pred, nombres_clases):
    """
    Genera un gráfico de la matriz de confusión usando Seaborn.

    Args:
        y_true (array-like): Valores verdaderos de las etiquetas.
        y_pred (array-like): Valores predichos por el modelo.
        nombres_clases (array-like): Nombres de las clases para los ejes del gráfico.

    Returns:
        str: Una cadena de texto con la imagen del gráfico codificada en Base64,
             lista para ser incrustada en HTML (formato Data URI).
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6)) # Define el tamaño de la figura
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", # Dibuja el heatmap
                xticklabels=nombres_clases, yticklabels=nombres_clases)
    plt.title("Matriz de Confusión")
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout() # Ajusta el diseño para evitar solapamientos

    # Guarda el gráfico en un buffer de bytes en memoria en lugar de un archivo
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0) # Vuelve al inicio del buffer para leer su contenido

    # Codifica la imagen en Base64 y la convierte a string UTF-8
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close() # Cierra la figura para liberar memoria
    return f"data:image/png;base64,{img_str}" # Devuelve el Data URI


# --- Función Principal: Entrenamiento, Evaluación y Guardado del Modelo ---
def entrenar_evaluar_modelo_cultivo(file):
    """
    Proceso completo: Lee datos de un archivo Excel, preprocesa, entrena un
    modelo AdaBoost, lo evalúa, guarda el modelo/preprocesadores, y genera
    un archivo Excel con los resultados de la prueba.

    Args:
        file (FileStorage): Objeto archivo subido desde Flask (o ruta a un archivo).

    Returns:
        dict: Un diccionario con los resultados, incluyendo métricas de evaluación,
              la matriz de confusión como imagen base64, y el archivo Excel de
              resultados también como base64.

    Raises:
        FileNotFoundError: Si el archivo no se encuentra (menos probable con FileStorage).
        ValueError: Si faltan columnas requeridas o los datos no son válidos.
        Exception: Para cualquier otro error durante el procesamiento.
    """
    try:
        # Lee el archivo Excel usando pandas, especificando el motor
        df = pd.read_excel(file, engine='openpyxl')

        # --- 1. Validación de Datos ---
        print("Paso 1: Validación de Datos...")
        columnas_requeridas = ['tipo_suelo', 'humedad', 'temperatura', 'cultivo']
        # Verifica si todas las columnas necesarias están presentes
        if not all(c in df.columns for c in columnas_requeridas):
            faltantes = [c for c in columnas_requeridas if c not in df.columns]
            raise ValueError(f"Faltan columnas necesarias en el archivo: {', '.join(faltantes)}")

        # Elimina filas que tengan valores nulos (NaN) en cualquiera de las columnas requeridas
        df.dropna(subset=columnas_requeridas, inplace=True)
        # Verifica si el DataFrame quedó vacío después de eliminar NaNs
        if df.empty:
             raise ValueError("El archivo no contiene datos válidos después de eliminar filas vacías.")
        print(f"Datos validados. Número de filas iniciales (tras eliminar NaN): {len(df)}")

        # --- 2. Ingeniería de Características y Preprocesamiento ---
        print("Paso 2: Preprocesamiento...")
        # Codificar Característica Categórica: 'tipo_suelo'
        # LabelEncoder convierte cada categoría de texto en un número entero.
        le_suelo = LabelEncoder()
        df['tipo_suelo_encoded'] = le_suelo.fit_transform(df['tipo_suelo'])
        print(f"Clases encontradas para 'tipo_suelo': {le_suelo.classes_}")

        # Codificar Variable Objetivo: 'cultivo'
        # Hacemos lo mismo para la columna objetivo ('cultivo')
        le_cultivo = LabelEncoder()
        df['cultivo_encoded'] = le_cultivo.fit_transform(df['cultivo'])
        print(f"Clases encontradas para 'cultivo' (objetivo): {le_cultivo.classes_}")
        nombres_clases = le_cultivo.classes_ # Guardamos los nombres para usarlos después

        # Definir Características (X) y Objetivo (y) usando las versiones codificadas
        columnas_features = ['tipo_suelo_encoded', 'humedad', 'temperatura']
        X = df[columnas_features] # DataFrame de características
        y = df['cultivo_encoded'] # Serie del objetivo codificado

        # --- 3. División Entrenamiento/Prueba ---
        print("Paso 3: División Entrenamiento/Prueba...")
        # Divide los datos: 80% para entrenar el modelo, 20% para evaluarlo.
        # random_state=42 asegura que la división sea siempre la misma (reproducibilidad).
        # stratify=y asegura que la proporción de cada clase de 'cultivo' sea similar
        # tanto en el conjunto de entrenamiento como en el de prueba. Es importante
        # para problemas de clasificación, especialmente si las clases están desbalanceadas.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        # Guarda los índices originales del DataFrame que corresponden al conjunto de prueba.
        # Esto es necesario para luego poder recuperar las filas originales completas
        # del DataFrame `df` y crear el archivo Excel de resultados.
        indices_test = X_test.index
        print(f"Tamaño entrenamiento: {len(X_train)} filas. Tamaño prueba: {len(X_test)} filas.")

        # --- 4. Escalado de Características ---
        print("Paso 4: Escalado de Características...")
        # StandardScaler transforma los datos para que tengan media 0 y desviación estándar 1.
        # Es importante para muchos modelos de ML, incluyendo AdaBoost a veces.
        scaler = StandardScaler()
        # Ajusta el escalador SÓLO con los datos de entrenamiento (X_train).
        # Esto evita 'data leakage' (información del conjunto de prueba "filtrándose"
        # al proceso de entrenamiento).
        X_train_scaled = scaler.fit_transform(X_train)
        # Aplica la misma transformación (ya ajustada) a los datos de prueba.
        X_test_scaled = scaler.transform(X_test)

        # --- 5. Entrenamiento del Modelo ---
        print("Paso 5: Entrenamiento del Modelo AdaBoost...")
        # Inicializa el clasificador AdaBoost. n_estimators es el número de "weak learners".
        model = AdaBoostClassifier(n_estimators=100, random_state=42)
        # Entrena el modelo usando los datos de entrenamiento escalados y las etiquetas de entrenamiento.
        model.fit(X_train_scaled, y_train)
        print("Modelo entrenado.")

        # --- 6. Guardar Modelo y Preprocesadores ---
        print(f"Paso 6: Guardando modelo y preprocesadores en '{MODEL_DIR}'...")
        # Guarda el modelo entrenado, el escalador ajustado y los codificadores de etiquetas.
        # Es crucial guardar todos estos artefactos para poder hacer predicciones
        # consistentes sobre datos nuevos en el futuro.
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(le_suelo, ENCODER_PATH)
        joblib.dump(le_cultivo, TARGET_ENCODER_PATH)
        print("Artefactos guardados.")

        # --- 7. Predicción sobre el Conjunto de Prueba ---
        print("Paso 7: Realizando predicciones sobre el conjunto de prueba...")
        # Usa el modelo entrenado para predecir las etiquetas del conjunto de prueba (escalado).
        y_pred = model.predict(X_test_scaled) # Las predicciones estarán codificadas numéricamente

        # --- 8. Evaluación del Modelo ---
        print("Paso 8: Evaluando el modelo...")
        # Calcula las métricas de rendimiento comparando las etiquetas reales (y_test)
        # con las etiquetas predichas (y_pred) para el conjunto de prueba.
        accuracy = accuracy_score(y_test, y_pred)
        # average='weighted' calcula la métrica para cada clase y promedia ponderando
        # por el número de instancias de cada clase. Es útil para clases desbalanceadas.
        # zero_division=0 evita warnings si una clase no tiene predicciones (poco probable aquí).
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision (Weighted): {precision:.4f}")
        print(f"  Recall (Weighted): {recall:.4f}")

        # Genera el gráfico de la matriz de confusión usando la función auxiliar
        cm_plot_base64 = generar_grafico_matriz_confusion(y_test, y_pred, nombres_clases)
        print("Gráfico de matriz de confusión generado.")

        # --- 9. Preparar DataFrame de Resultados para Descargar ---
        print("Paso 9: Preparando DataFrame de resultados para Excel...")
        # Decodifica las etiquetas reales (y_test) y predichas (y_pred)
        # para volver a tener los nombres originales de los cultivos.
        y_test_decoded = le_cultivo.inverse_transform(y_test)
        y_pred_decoded = le_cultivo.inverse_transform(y_pred)

        # Selecciona las filas originales del DataFrame `df` que corresponden al conjunto de prueba
        # usando los índices guardados anteriormente. Se usa .copy() para evitar advertencias
        # sobre modificar una 'vista' en lugar de una copia del DataFrame.
        df_resultados_test = df.loc[indices_test].copy()

        # Añade las columnas con los nombres decodificados de los cultivos reales y predichos
        # a este DataFrame de resultados.
        df_resultados_test['cultivo_real_nombre'] = y_test_decoded
        df_resultados_test['cultivo_predicho_nombre'] = y_pred_decoded

        # Selecciona y reordena solo las columnas deseadas para el archivo Excel final.
        columnas_excel = ['tipo_suelo', 'humedad', 'temperatura', 'cultivo_real_nombre', 'cultivo_predicho_nombre']
        df_para_excel = df_resultados_test[columnas_excel]

        # --- 10. Crear Excel en Memoria y Codificar en Base64 ---
        print("Paso 10: Creando archivo Excel en memoria...")
        output_excel = BytesIO() # Crea un buffer de bytes en memoria
        # Escribe el DataFrame en el buffer en formato Excel (.xlsx)
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            df_para_excel.to_excel(writer, index=False, sheet_name='Resultados_Test')
        output_excel.seek(0) # Vuelve al inicio del buffer para poder leerlo

        # Codifica los bytes del archivo Excel a Base64 y luego a string UTF-8
        # para poder incluirlo en el diccionario de resultados y pasarlo a la plantilla HTML.
        excel_base64 = base64.b64encode(output_excel.getvalue()).decode('utf-8')
        print("Archivo Excel codificado en Base64.")

        # --- 11. Preparar Diccionario de Resultados Finales ---
        print("Paso 11: Compilando resultados finales...")
        # Agrupa todos los resultados importantes en un diccionario.
        results = {
            'message': f"Modelo entrenado y evaluado con éxito. Guardado en '{MODEL_DIR}'.",
            'accuracy': f"{accuracy:.4f}",
            'precision': f"{precision:.4f}",
            'recall': f"{recall:.4f}",
            'confusion_matrix_plot': cm_plot_base64, # Imagen Base64
            'num_train_samples': len(X_train),
            'num_test_samples': len(X_test),
            'model_path': MODEL_PATH,
            'class_distribution_test': pd.Series(y_test_decoded).value_counts().to_dict(), # Conteo de clases reales en test
            'predicted_distribution_test': pd.Series(y_pred_decoded).value_counts().to_dict(), # Conteo de clases predichas en test
            'excel_output_base64': excel_base64 # Excel Base64
        }
        print("Proceso completado con éxito.")
        return results

    # --- Manejo de Errores Específicos ---
    except FileNotFoundError:
         # Este error es menos probable si 'file' viene de request.files
         print("Error: El archivo no fue encontrado.")
         raise Exception("Error: El archivo no fue encontrado.")
    except ValueError as ve:
         # Errores relacionados con los datos (columnas faltantes, tipos incorrectos, etc.)
         print(f"Error de Valor/Datos: {str(ve)}")
         raise Exception(f"Error de datos: {str(ve)}")
    except Exception as e:
        # Captura cualquier otro error inesperado durante la ejecución.
        print("------ ERROR INESPERADO ------")
        # Imprime el traceback completo en la consola del servidor para depuración.
        traceback.print_exc()
        print("-----------------------------")
        # Lanza una nueva excepción con un mensaje más genérico para el usuario.
        raise Exception(f"Error procesando el archivo y entrenando el modelo: {str(e)}")


# --- Función para Predicción sobre Datos Nuevos (Usando Modelo Guardado) ---
def predecir_cultivo_nuevo(datos_nuevos_df):
    """
    Carga el modelo AdaBoost entrenado y los preprocesadores guardados
    para realizar predicciones sobre un nuevo conjunto de datos.

    Args:
        datos_nuevos_df (pd.DataFrame): Un DataFrame de Pandas que debe contener
                                        las columnas 'tipo_suelo', 'humedad',
                                        y 'temperatura'.

    Returns:
        np.ndarray: Un array de NumPy con los nombres de los cultivos predichos.

    Raises:
        FileNotFoundError: Si no se encuentran los archivos del modelo o preprocesadores.
        ValueError: Si faltan columnas en los datos nuevos o si 'tipo_suelo'
                    contiene categorías no vistas durante el entrenamiento.
        Exception: Para cualquier otro error durante la predicción.
    """
    print("Función predecir_cultivo_nuevo: Iniciando predicción...")
    try:
        # --- Cargar Artefactos Guardados ---
        print(f"Cargando modelo desde: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        print(f"Cargando escalador desde: {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)
        print(f"Cargando codificador de suelo desde: {ENCODER_PATH}")
        le_suelo = joblib.load(ENCODER_PATH)
        print(f"Cargando codificador de cultivo desde: {TARGET_ENCODER_PATH}")
        le_cultivo = joblib.load(TARGET_ENCODER_PATH) # Necesario para decodificar la salida

        # --- Preprocesar Datos Nuevos ---
        print("Preprocesando datos nuevos...")
        # Asegurar que las columnas requeridas para la predicción estén presentes
        columnas_requeridas = ['tipo_suelo', 'humedad', 'temperatura']
        if not all(c in datos_nuevos_df.columns for c in columnas_requeridas):
             faltantes = [c for c in columnas_requeridas if c not in datos_nuevos_df.columns]
             raise ValueError(f"Faltan columnas en los datos nuevos: {', '.join(faltantes)}")

        # Codificar 'tipo_suelo' usando el codificador CARGADO (le_suelo).
        # ¡Importante! Usar solo '.transform()', NUNCA '.fit()' o '.fit_transform()'
        # en datos nuevos, ya que debemos usar la codificación aprendida del entrenamiento.
        try:
             datos_nuevos_df['tipo_suelo_encoded'] = le_suelo.transform(datos_nuevos_df['tipo_suelo'])
        except ValueError as e:
            # Captura específicamente el error si LabelEncoder encuentra una categoría
            # en los datos nuevos que no existía en los datos de entrenamiento.
            if 'y contains previously unseen labels' in str(e):
                 label_no_vista = str(e).split(':')[-1].strip().replace("'", "").replace("[", "").replace("]", "")
                 raise ValueError(f"El tipo de suelo '{label_no_vista}' no fue visto durante el entrenamiento. No se puede predecir.")
            else:
                 raise e # Relanza otros errores de valor

        # Seleccionar las mismas columnas de características usadas en el entrenamiento
        columnas_features = ['tipo_suelo_encoded', 'humedad', 'temperatura']
        X_new = datos_nuevos_df[columnas_features]

        # Escalar las características usando el escalador CARGADO (scaler).
        # De nuevo, solo usar '.transform()'.
        X_new_scaled = scaler.transform(X_new)
        print("Datos nuevos preprocesados y escalados.")

        # --- Realizar Predicción ---
        print("Realizando predicción con el modelo cargado...")
        predicciones_codificadas = model.predict(X_new_scaled)

        # --- Decodificar Predicciones ---
        # Usa el codificador de etiquetas objetivo CARGADO (le_cultivo) para
        # convertir las predicciones numéricas de vuelta a los nombres de los cultivos.
        predicciones_decodificadas = le_cultivo.inverse_transform(predicciones_codificadas)
        print("Predicciones decodificadas.")

        return predicciones_decodificadas

    except FileNotFoundError:
        print("Error: No se encontró el modelo entrenado o los preprocesadores.")
        # Este error sugiere que el modelo no ha sido entrenado y guardado aún.
        raise Exception("Error: No se encontró el modelo entrenado o los preprocesadores. Entrene el modelo primero.")
    except ValueError as ve:
        # Errores durante el preprocesamiento de datos nuevos
         print(f"Error de Valor/Datos en predicción: {str(ve)}")
         raise Exception(f"Error de datos durante la predicción: {str(ve)}")
    except Exception as e:
        # Otros errores inesperados durante la predicción
        print("------ ERROR INESPERADO EN PREDICCIÓN ------")
        traceback.print_exc()
        print("-------------------------------------------")
        raise Exception(f"Error durante la predicción: {str(e)}")