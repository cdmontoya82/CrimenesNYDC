import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import os

# ==============================================================================
# 1. CONFIGURACIÓN DE RUTAS Y CARGA DE DATOS
# ==============================================================================
# Definimos rutas absolutas para evitar errores de archivo no encontrado
ruta_csv = 'ruta del dataset'
ruta_graficas = 'ruta de graficas e infomees'

# Creamos la carpeta de salida si no existe (automatización de entorno)
if not os.path.exists(ruta_graficas):
    os.makedirs(ruta_graficas)

try:
    # Cargamos 250k filas para balancear representatividad y velocidad de procesamiento
    df = pd.read_csv(ruta_csv, low_memory=False, nrows=250000)
    print(f"¡Dataset cargado exitosamente! Filas: {len(df)}")
except FileNotFoundError:
    print(f"Error crítico: No se encontró el dataset en {ruta_csv}")
    exit()

# Estandarizamos nombres a minúsculas para facilitar la codificación
df.columns = [c.lower() for c in df.columns]

# ==============================================================================
# 2. LIMPIEZA Y FILTRADO ESTRATÉGICO (DATA WRANGLING)
# ==============================================================================
# Eliminamos duplicados para no sesgar el modelo con registros idénticos
df.drop_duplicates(inplace=True)

# Eliminamos filas con nulos en variables clave; el modelo no admite valores vacíos
cols_criticas = ['latitude', 'longitude', 'law_cat_cd', 'age_group', 'perp_sex', 'perp_race']
df.dropna(subset=cols_criticas, inplace=True)

# FILTRO GEOGRÁFICO: Eliminamos outliers fuera de NYC (coordenadas erróneas)
df = df[(df['latitude'] > 40.4) & (df['latitude'] < 40.9) & 
        (df['longitude'] > -74.3) & (df['longitude'] < -73.7)]

# INGENIERÍA DE VARIABLES (Feature Engineering)
df['arrest_date'] = pd.to_datetime(df['arrest_date'])
df['month'] = df['arrest_date'].dt.month
df['day_of_week'] = df['arrest_date'].dt.dayofweek

# VARIABLE OBJETIVO: 1 si es Felonía (Grave), 0 si es Misdemeanor/Violation (Menor)
df['target'] = (df['law_cat_cd'] == 'F').astype(int)

# ==============================================================================
# 3. VISUALIZACIONES ESTADÍSTICAS (EDA)
# ==============================================================================
print("Generando gráficas de diagnóstico inicial...")
fig1, axes = plt.subplots(1, 3, figsize=(22, 6))

# Distribución de Latitud con ajuste de curva Normal (Gauss)
sns.histplot(df['latitude'], kde=True, color='blue', stat="density", ax=axes[0])
mu, std = df['latitude'].mean(), df['latitude'].std()
x = np.linspace(df['latitude'].min(), df['latitude'].max(), 100)
axes[0].plot(x, stats.norm.pdf(x, mu, std), 'r', linewidth=2, label='Gauss')
axes[0].set_title('Distribución de Latitud vs Campana de Gauss')
axes[0].legend()

# Mapa de dispersión simplificado (Muestreo de 2000 puntos para rapidez)
sns.scatterplot(data=df.sample(2000), x='longitude', y='latitude', hue='law_cat_cd', alpha=0.5, ax=axes[1])
axes[1].set_title('Muestra de Distribución Geográfica')

# Estacionalidad por Mes
sns.countplot(x='month', data=df, hue='month', palette='rocket', ax=axes[2], legend=False)
axes[2].set_title('Frecuencia de Arrestos por Mes')

plt.tight_layout()
fig1.savefig(os.path.join(ruta_graficas, 'analisis_estadistico_gauss.png'), dpi=300)
plt.close(fig1)

# ==============================================================================
# 4. PREPARACIÓN PARA MACHINE LEARNING
# ==============================================================================
# Selección de predictores y conversión de categóricos a numéricos (One-Hot Encoding)
features = ['arrest_boro', 'age_group', 'perp_sex', 'perp_race', 'latitude', 'longitude', 'month', 'day_of_week']
X = pd.get_dummies(df[features], drop_first=True)
y = df['target']

# División Train/Test con estratificación (mantiene proporción de clases en ambos sets)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ESCALAMIENTO: Vital para que la magnitud de lat/long no domine sobre mes/día
scaler = StandardScaler()
num_vars = ['latitude', 'longitude', 'month', 'day_of_week']
X_train[num_vars] = scaler.fit_transform(X_train[num_vars])
X_test[num_vars] = scaler.transform(X_test[num_vars])

# BALANCEO DE CLASES: SMOTE genera ejemplos sintéticos de Felonías para igualar la muestra
print("Aplicando SMOTE para balancear clases y entrenando Random Forest...")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# ENTRENAMIENTO: Random Forest con profundidad limitada para evitar Sobreajuste (Overfitting)
rf = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
rf.fit(X_res, y_res)

# ==============================================================================
# 5. EVALUACIÓN Y MÉTRICAS DE IMPACTO
# ==============================================================================

fig2, ax2 = plt.subplots(figsize=(10, 6))
importancias = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
sns.barplot(x=importancias, y=importancias.index, hue=importancias.index, palette='viridis', ax=ax2, legend=False)
ax2.set_title('Top 10 Predictores más Influyentes')
fig2.savefig(os.path.join(ruta_graficas, 'importancia_variables.png'), dpi=300)
plt.close(fig2)

# Matriz de Confusión: Evaluación de aciertos vs errores
fig3, ax3 = plt.subplots(figsize=(8, 6))
y_pred = rf.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax3)
ax3.set_title('Matriz de Confusión: Rendimiento del Modelo')
ax3.set_xlabel('Predicción del Modelo')
ax3.set_ylabel('Realidad (NYPD)')
fig3.savefig(os.path.join(ruta_graficas, 'matriz_confusion_final.png'), dpi=300)
plt.close(fig3)

# ==============================================================================
# 6. MAPA PROFESIONAL Y EVIDENCIA FINAL
# ==============================================================================
# Generación de Mapa Dark Mode (Visualización de alta resolución)
plt.style.use('dark_background')
fig_map, ax_map = plt.subplots(figsize=(12, 12), facecolor='#1e2530')
sns.scatterplot(
    data=df.sample(min(80000, len(df))), 
    x='longitude', y='latitude', hue='law_cat_cd', 
    palette='YlOrBr', s=0.5, alpha=0.3, ax=ax_map
)
ax_map.set_title('MAPA DE CALOR: DENSIDAD DE ARRESTOS NYC', fontsize=18)
for spine in ax_map.spines.values(): spine.set_visible(False)
fig_map.savefig(os.path.join(ruta_graficas, 'mapa_densidad_pro_dark.png'), dpi=300, bbox_inches='tight')
plt.close()
plt.style.use('default')

# GENERACIÓN DE TABLA DE EVIDENCIA (Imagen de los datos procesados)
print("Generando imagen de evidencia técnica (Dataset Final)...")
cols_ev = [c for c in X_train.columns if 'boro' in c or 'sex' in c][:2] + num_vars
df_ev = X_train[cols_ev].head(10).copy()
df_ev['target'] = y_train.head(10).values

# Formateo manual para evitar errores de tipo de dato en la tabla visual
datos_texto = [[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row] for row in df_ev.values]
fig_tab, ax_tab = plt.subplots(figsize=(14, 5))
ax_tab.axis('off')
tabla = ax_tab.table(cellText=datos_texto, colLabels=df_ev.columns, cellLoc='center', loc='center')
tabla.auto_set_font_size(False); tabla.set_fontsize(9); tabla.scale(1, 2)
fig_tab.savefig(os.path.join(ruta_graficas, 'EVIDENCIA_DATASET_FINAL.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n PROCESO FINALIZADO CON ÉXITO.")
print(f"Archivos disponibles en: {ruta_graficas}")
print("\n--- REPORTE TÉCNICO DE CLASIFICACIÓN ---")
print(classification_report(y_test, y_pred))