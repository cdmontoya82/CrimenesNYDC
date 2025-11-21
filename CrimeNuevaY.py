import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np # Lo usaremos para np.nan
import os

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

 


# 1. Crear la "figura" y asignar a la variable 'fig'

fig = plt.figure(figsize=(15, 10)) 

# -----------------------------------------------------------
# GRÁFICA 1
# -----------------------------------------------------------
plt.subplot(2, 2, 1) 
ax1 = sns.countplot(x='LAW_CAT_CD', data=df, order=df['LAW_CAT_CD'].value_counts().index, palette='viridis')
plt.title('Distribución de Gravedad del Delito', fontsize=14)
plt.xlabel('Nivel de Delito')
plt.ylabel('Cantidad')

# Etiquetas de porcentaje
total = len(df)
for p in ax1.patches:
    height = p.get_height()
    if height > 0: # Evitar errores si hay barras vacías
        percentage = '{:.1f}%'.format(100 * height/total)
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = height
        ax1.annotate(percentage, (x, y), ha='center', va='bottom')

# -----------------------------------------------------------
# GRÁFICA 2
# -----------------------------------------------------------
plt.subplot(2, 2, 2) 
sns.countplot(x='ARREST_BORO', data=df, palette='magma', order=df['ARREST_BORO'].value_counts().index)
plt.title('Arrestos por Distrito', fontsize=14)
plt.xlabel('Distrito')
plt.ylabel('Cantidad')

# -----------------------------------------------------------
# GRÁFICA 3
# -----------------------------------------------------------
if 'ARREST_DATE' in df.columns:
    df['Year'] = df['ARREST_DATE'].dt.year
    plt.subplot(2, 1, 2) 
    sns.countplot(x='Year', data=df, palette='coolwarm')
    plt.title('Cantidad de Arrestos por Año', fontsize=14)
    plt.xlabel('Año')
    plt.ylabel('Cantidad')
    plt.xticks(rotation=45)

plt.tight_layout()

# -----------------------------------------------------------
# GUARDADO SEGURO
# -----------------------------------------------------------
ruta_guardado = '/home/cristian/Documentos/MAchine_learning/Proyecto_python/Graficas'

if not os.path.exists(ruta_guardado):
    os.makedirs(ruta_guardado)

nombre_archivo = 'analisis_inicial_nypd.png'
ruta_completa = os.path.join(ruta_guardado, nombre_archivo)


#Asegura que se guarde lo que está en la variable 'fig', no un lienzo vacío.
fig.savefig(ruta_completa, dpi=300, bbox_inches='tight')

print(f"¡Gráfica guardada (sin estar en blanco) en: {ruta_completa}")

# plt.show() 
plt.close(fig) # Buena práctica para liberar memoria RAM

# mapa de correlacion_________________


# 2. Preprocesamiento: Convertir texto a números para la correlación

df_corr = df.copy()

# Lista de columnas categóricas que queremos incluir en la correlación
columnas_categoricas = ['ARREST_BORO', 'AGE_GROUP', 'PERP_SEX', 'PERP_RACE', 'LAW_CAT_CD']

print("Codificando variables categóricas a numéricas...")
for col in columnas_categoricas:
    if col in df_corr.columns:
        # Convertimos cada texto único en un número (ej: 'F'->0, 'M'->1)
        df_corr[col] = df_corr[col].astype('category').cat.codes

#Aseguramos que la fecha sea útil (usaremos el Año y Mes)
df_corr['ARREST_DATE'] = pd.to_datetime(df_corr['ARREST_DATE'], format='%m/%d/%Y')
df_corr['Year'] = df_corr['ARREST_DATE'].dt.year
df_corr['Month'] = df_corr['ARREST_DATE'].dt.month

# Seleccionamos solo las columnas numéricas para la matriz
list
cols_finales = columnas_categoricas + ['Latitude', 'Longitude', 'Year', 'Month']
df_matriz = df_corr[cols_finales]

# 3. Calcular la correlación
matriz_correlacion = df_matriz.corr()

# 4. Graficar el Heatmap
fig = plt.figure(figsize=(12, 10))
sns.heatmap(matriz_correlacion, 
            annot=True,       # Mostrar los números en los cuadros
            fmt=".2f",        # 2 decimales
            cmap='coolwarm',  # Colores rojo/azul
            linewidths=0.5)

plt.title('Matriz de Correlación: Datos de Arrestos NYPD', fontsize=16)

# 5. Guardar la gráfica
ruta_guardado = '/home/cristian/Documentos/MAchine_learning/Proyecto_python/Graficas'
if not os.path.exists(ruta_guardado):
    os.makedirs(ruta_guardado)

ruta_completa = os.path.join(ruta_guardado, 'matriz_correlacion.png')
fig.savefig(ruta_completa, dpi=300, bbox_inches='tight')

print(f"Matriz de correlación guardada en: {ruta_completa}")
plt.close(fig)