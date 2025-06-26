import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os 

# --- 1. Configuración de la Página de Streamlit ---
st.set_page_config(page_title="Diagnóstico de Migraña", layout="wide")

# --- 2. Cargar Datos y Entrenar Modelo ---

csv_file_name = "migraine_symptom_classification.csv"

current_dir = os.getcwd()

try:
    df = pd.read_csv(csv_file_name, encoding='utf-8')
except FileNotFoundError:
    st.error(f"Error: El archivo '{csv_file_name}' no se encontró.")
    st.info(f"Asegúrate de que '{csv_file_name}' esté en la misma carpeta que este script.")
    st.stop()
except UnicodeDecodeError:
    st.error(f"Error de codificación al leer '{csv_file_name}'.")
    st.info("Intentando leer con 'latin-1' y convertir a UTF-8...")
    try:
        df = pd.read_csv(csv_file_name, encoding='latin-1')
        df.to_csv(csv_file_name, index=False, encoding='utf-8')
        st.warning("El archivo CSV se leyó con latin-1 y se sobrescribió como UTF-8 para futuros usos.")
    except Exception as e:
        st.error(f"No se pudo leer el archivo '{csv_file_name}' con codificaciones comunes. Error: {e}")
        st.stop()
except Exception as e:
    st.error(f"Ocurrió un error inesperado al cargar el CSV: {e}")
    st.info("Verifica el formato del CSV y los permisos de archivo.")
    st.stop()

if 'Type' not in df.columns:
    st.error("Error: La columna 'Type' (tipo de migraña) no se encontró en el CSV.")
    st.info("Asegúrate de que el nombre de la columna sea 'Type' (sensible a mayúsculas/minúsculas).")
    st.stop()

X = df.drop("Type", axis=1)
y = df["Type"]

if X.empty or X.shape[1] == 0:
    st.error("Error: El DataFrame de características (X) está vacío o no contiene columnas para entrenar el modelo.")
    st.info("Verifica que tu CSV contenga columnas de síntomas además de la columna 'Type'.")
    st.stop()

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# --- 3. Interfaz de Usuario con Streamlit ---
st.title("Asistente Inteligente de Diagnóstico de Migraña")
st.markdown("Este asistente predice el tipo de migraña basándose en los síntomas proporcionados.")
st.markdown("---")

st.sidebar.header("Síntomas del Paciente")
st.sidebar.markdown("Por favor, introduce los síntomas relevantes del paciente:")

# Los inputs que ya tenías
age = st.sidebar.slider("Edad", 10, 80, 30)
visual = st.sidebar.selectbox("Síntomas visuales (0=Ninguno, 1=Leve, ..., 4=Severo)", [0, 1, 2, 3, 4])
sensory = st.sidebar.selectbox("Alteraciones sensoriales (0=Ninguna, 1=Leve, 2=Moderada)", [0, 1, 2])
vertigo = st.sidebar.radio("Vértigo", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

# --- 4. Botón de Diagnóstico y Predicción ---
st.markdown("---")
if st.button("Diagnosticar"):
    # *** SECCIÓN CRÍTICA MODIFICADA PARA COINCIDIR CON TODAS LAS COLUMNAS DE X ***
    # Inicializa un diccionario con ceros para TODAS las columnas de entrenamiento (X.columns)
    input_data_dict = {col: 0 for col in X.columns}
    
    # Asigna los valores de los inputs del usuario a las columnas correspondientes
    # Si estas columnas no existen en tu CSV, el 'if' las ignorará
    if 'Age' in input_data_dict: input_data_dict['Age'] = age
    if 'Visual' in input_data_dict: input_data_dict['Visual'] = visual
    if 'Sensory' in input_data_dict: input_data_dict['Sensory'] = sensory
    if 'Vertigo' in input_data_dict: input_data_dict['Vertigo'] = vertigo
    
    # IMPORTANTE: Si hay OTRAS COLUMNAS en tu CSV que el modelo usó para entrenar
    # (ej. 'Duration', 'Location', 'Nausea', 'Vomiting', 'Phonophobia', etc.),
    # y NO las pides al usuario en la interfaz, su valor se mantendrá en 0 (el valor por defecto).
    # Si esto no es lo que esperas para tu modelo, deberías:
    # 1. Añadir más inputs en la barra lateral para esas columnas.
    # 2. O asignarles un valor predeterminado más apropiado (como la media o moda de tu dataset).
    # Por ahora, se asume que 0 es un valor aceptable para los síntomas no preguntados.
    
    # Crear un DataFrame de Pandas con la única fila de input, asegurando el ORDEN de las columnas
    # Esto es VITAL: las columnas deben estar en el mismo orden que X.columns
    input_df = pd.DataFrame([input_data_dict], columns=X.columns)

    try:
        prediction = model.predict(input_df)[0]
        st.subheader(f"Resultado del Diagnóstico: **{prediction}**")

        prediction_lower = prediction.lower() 

        if "hemiplegic" in prediction_lower:
            st.progress(70, text="Indicador de Posible Severidad")
            st.info("⚠️ Posible migraña hemipléjica. Requiere atención médica urgente.")
        elif "aura" in prediction_lower:
            st.progress(50, text="Indicador de Síntomas de Aura")
            st.info("✅ Síntomas de aura detectados. Común en algunos tipos de migraña.")
        else:
            st.progress(30, text="Diagnóstico Inicial")

        if vertigo == 1 and "basilar" not in prediction_lower:
            st.warning("⚠️ Vértigo presente pero no clasificado como migraña basilar. Considerar evaluación neurológica adicional.")

    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
        st.info("Asegúrate de que los datos de entrada sean consistentes con el entrenamiento del modelo. Detalles: " + str(e))

st.markdown("---")
st.caption("Disclaimer: Este es un asistente basado en Machine Learning y no reemplaza el diagnóstico médico profesional. Siempre consulta a un especialista para un diagnóstico preciso y tratamiento.")