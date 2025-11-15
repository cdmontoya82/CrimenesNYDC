import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np # Lo usaremos para np.nan

# 1. -----------Carga y Vistazo Inicial--------------
# Cargar dataset
try:
    df = pd.read_csv('/home/cristian/Documentos/MAchine_learning/NYPD_Arrests_Data__Historic_.csv')
    print("¡Dataset cargado exitosamente!")
except FileNotFoundError:
    print("Error: No se pudo encontrar el archivo CSV en la ruta especificada.")
    # Detener la ejecución si el archivo no se carga
    exit()


# Vistazo rápido a las primeras filas
print(df.head())

# Revisar los tipos de datos y si hay valores nulos
print("\n--- Información General (df.info()) ---")
df.info()
print("------------------------------------------")


# 2. ----- Diagnóstico de Columnas Numéricas (Tu pregunta) -----

# Columnas que deberían ser numéricas pero podrían tener texto
cols_a_revisar = ['Latitude', 'Longitude', 'PD_CD', 'KY_CD'] 

print(f"\nRevisando columnas: {cols_a_revisar}...")

for col in cols_a_revisar:
    if col in df.columns:
        if df[col].dtype == 'object':
            print(f"\nADVERTENCIA: La columna '{col}' es de tipo 'object' (texto).")
            
            # 1. Intentar convertir a número, los errores serán NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            print(f"-> Convertida a tipo numérico. Los textos no válidos ahora son NaN.")

        # 2. Ahora (o si ya era numérica), contamos los NaN
        nan_count = df[col].isnull().sum()
        if nan_count > 0:
            print(f"-> La columna '{col}' tiene {nan_count} valores NaN (nulos).")
        else:
            print(f"-> La columna '{col}' está limpia (sin NaN).")
    else:
        print(f"\nADVERTENCIA: La columna '{col}' no se encontró en el CSV.")

# 3. ----- Corrección de tu Análisis Exploratorio (EDA) -----

# 1. ¡Verificar el desbalanceo!
# Reemplazar 'Class' la variable objetivo 'LAW_CAT_CD'
if 'LAW_CAT_CD' in df.columns:
    print("\n--- Distribución de la Variable Objetivo (LAW_CAT_CD) ---")
    print(df['LAW_CAT_CD'].value_counts(normalize=True))
    print("-----------------------------------------------------")
else:
    print("\nERROR: No se encontró la columna 'LAW_CAT_CD' para analizar el desbalanceo.")

# 2. Convertir fechas (¡Importante!)

if 'ARREST_DATE' in df.columns:
    print("\nConvirtiendo 'ARREST_DATE' a formato datetime...")
    df['ARREST_DATE'] = pd.to_datetime(df['ARREST_DATE'], format='%m/%d/%Y')
    
    # 'describe()' SÍ es útil para la fecha
    print("\n--- Estadísticas Descriptivas de 'ARREST_DATE' ---")
    # llama .describe() sin argumentos
    print(df['ARREST_DATE'].describe())
    
    print("---------------------------------------------------")
else:
    print("\nERROR: No se encontró la columna 'ARREST_DATE'.")