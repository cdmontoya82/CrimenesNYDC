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

 

# 4. crear la graficas para su analisis

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
# GUARDADO SEGURO (Se guarda imagen si no se se puede visualizar la gráfica)
# -----------------------------------------------------------
ruta_guardado = '/home/cristian/Documentos/MAchine_learning/Proyecto_python/Graficas' # se debe modificar la ruta segun donde se este ejecutando

if not os.path.exists(ruta_guardado):
    os.makedirs(ruta_guardado)

nombre_archivo = 'analisis_inicial_nypd.png' # se nombra la gráfica, este puede modificarse a al que se considere mas adecuado
ruta_completa = os.path.join(ruta_guardado, nombre_archivo)


#Asegura que se guarde lo que está en la variable 'fig', no un lienzo vacío.
fig.savefig(ruta_completa, dpi=300, bbox_inches='tight')

print(f"¡Gráfica guardada (sin estar en blanco) en: {ruta_completa}")  # mensaje de que la gráfica no presento novedad al guardarla

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

# 5. ----- Ingeniería de Características (Feature Engineering) -----

# 1. LIMPIEZA DE NULOS CRÍTICOS
# Se eliminan filas si faltan Latitud/Longitud o la variable objetivo
print(f"Filas antes de limpiar: {len(df)}") # obtiene el número de filas que tiene el DataFrame inicial
df = df.dropna(subset=['Latitude', 'Longitude', 'LAW_CAT_CD'])   # en pandas sirve para eliminar filas que tengan valores nulos (NaN) en columnas específicas
print(f"Filas después de limpiar Lat/Lon: {len(df)}") # obtiene el número de filas que tiene el DataFrame despues de la limpieza

# 2. CREAR VARIABLE OBJETIVO BINARIA
# 1 = Felonía (Grave), 0 = Delito Menor/Violación
df['Target'] = df['LAW_CAT_CD'].apply(lambda x: 1 if x == 'F' else 0) # crea una nueva columna llamada target ( objetivo) y le asigna los valores binarios

print("\nVerificación de la Variable Objetivo (Target):")
print(df['Target'].value_counts(normalize=True))   # imprimimos lo que hay en la columna para verifficar que quede bien el target

# 3. EXTRAER DATOS DE LA FECHA (Tus X temporales)
# Transforma los valores de texto (strings), números o formatos ambiguos en un objeto datetime de pandas, que se entiende como fecha y hora reales
df['ARREST_DATE'] = pd.to_datetime(df['ARREST_DATE'])

df['Hour'] = df['ARREST_DATE'].dt.hour
df['Day'] = df['ARREST_DATE'].dt.dayofweek # 0=Lunes, 6=Domingo
df['Month'] = df['ARREST_DATE'].dt.month
df['Year'] = df['ARREST_DATE'].dt.year 

# 4. SELECCIONAR TUS VARIABLES FINALES
# selecionamos las columnas con las que vamos a trabajar en la cual nuestro objetivo es Target
columnas_a_usar = [
    'ARREST_BORO',  # Dónde (Distrito)
    'AGE_GROUP',    # Quién (Edad)
    'PERP_SEX',     # Quién (Sexo)
    'PERP_RACE',    # Quién (Raza)
    'Latitude',     # Dónde Exacto
    'Longitude',    # Dónde Exacto
    'Hour',         # Cuándo
    'Day',          # Cuándo
    'Month',        # Cuándo
    'Target'        # Lo que vamos predecir
]

# Creamos el DataFrame final limpio, el cual deberia funcionar para el siguiente paso
df_model = df[columnas_a_usar].copy()

print("\n--- Vistazo al Dataset listo para el Modelo ---")
print(df_model.head())
print("\nColumnas finales:", df_model.columns.tolist())