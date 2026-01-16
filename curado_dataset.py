"""
Script para el curado y análisis exploratorio del dataset de Spotify
Incluye visualizaciones, detección de outliers y limpieza de datos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import ast
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")



print("analisis y curado del dataset")


# Cargar el dataset (probando diferentes codificaciones)
try:
    df = pd.read_csv('playlist_2010to2023.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv('playlist_2010to2023.csv', encoding='latin-1')
    except UnicodeDecodeError:
        df = pd.read_csv('playlist_2010to2023.csv', encoding='iso-8859-1')


print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")

print(df.head())
print(df.info())
print(df.describe())

print("\n Valores nulos")
print(df.isnull().sum())
print(f"\nPorcentaje de valores nulos por columna:")
print((df.isnull().sum() / len(df) * 100).round(2))


print("\nAnalisis de generos")


# Extraer géneros individuales
all_genres = []
for genres_str in df['artist_genres'].dropna():
    try:
        genres_list = ast.literal_eval(genres_str)
        all_genres.extend(genres_list)
    except:
        pass

genre_counts = pd.Series(all_genres).value_counts()
print(f"Total de géneros únicos: {len(genre_counts)}")
print(f"\nTop 20 géneros más frecuentes:")
print(genre_counts.head(20))

# Histograma de géneros (top 30)
plt.figure(figsize=(14, 8))
genre_counts.head(30).plot(kind='barh')
plt.title('Top 30 Géneros Musicales Más Frecuentes', fontsize=16, fontweight='bold')
plt.xlabel('Frecuencia', fontsize=12)
plt.ylabel('Género', fontsize=12)
plt.tight_layout()
plt.savefig('histograma_generos.png', dpi=300, bbox_inches='tight')
plt.close()

# Distribución de número de géneros por artista
df['num_genres'] = df['artist_genres'].apply(lambda x: len(ast.literal_eval(x)) if pd.notna(x) and x != '[]' else 0)
plt.figure(figsize=(10, 6))
df['num_genres'].hist(bins=20, edgecolor='black')
plt.title('Distribución del Número de Géneros por Artista', fontsize=14, fontweight='bold')
plt.xlabel('Número de Géneros', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.tight_layout()
plt.savefig('distribucion_num_generos.png', dpi=300, bbox_inches='tight')
print("✓ Distribución de número de géneros guardada como 'distribucion_num_generos.png'")
plt.close()


# Variables numéricas de interés
numeric_cols = ['danceability', 'energy', 'loudness', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'valence', 
                'tempo', 'duration_ms', 'track_popularity', 'artist_popularity']

print("\nVbles numericas")
print("-" * 80)

# Estadísticas descriptivas por variable
for col in numeric_cols:
    print(f"\n{col.upper()}:")
    print(f"  Media: {df[col].mean():.4f}")
    print(f"  Mediana: {df[col].median():.4f}")
    print(f"  Desv. Std: {df[col].std():.4f}")
    print(f"  Min: {df[col].min():.4f}")
    print(f"  Max: {df[col].max():.4f}")


print("\nHistogramas")


fig, axes = plt.subplots(4, 3, figsize=(15, 16))
axes = axes.ravel()

for idx, col in enumerate(numeric_cols):
    axes[idx].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribución de {col}', fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frecuencia')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('histogramas_variables.png', dpi=300, bbox_inches='tight')
plt.close()


print("\nBoxplots")
print("-" * 80)

fig, axes = plt.subplots(4, 3, figsize=(15, 16))
axes = axes.ravel()

for idx, col in enumerate(numeric_cols):
    axes[idx].boxplot(df[col].dropna(), vert=True)
    axes[idx].set_title(f'Boxplot de {col}', fontweight='bold')
    axes[idx].set_ylabel(col)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('boxplots_variables.png', dpi=300, bbox_inches='tight')
plt.close()


print("\nscatters")


# Scatterplots contra el año
fig, axes = plt.subplots(4, 3, figsize=(15, 16))
axes = axes.ravel()

for idx, col in enumerate(numeric_cols):
    axes[idx].scatter(df['year'], df[col], alpha=0.3, s=10)
    axes[idx].set_title(f'{col} vs Año', fontweight='bold')
    axes[idx].set_xlabel('Año')
    axes[idx].set_ylabel(col)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scatterplots_vs_year.png', dpi=300, bbox_inches='tight')
plt.close()

# Matriz de scatterplots entre variables musicales clave
music_features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness']
plt.figure(figsize=(14, 14))
for i, feat1 in enumerate(music_features):
    for j, feat2 in enumerate(music_features):
        plt.subplot(len(music_features), len(music_features), i * len(music_features) + j + 1)
        if i == j:
            plt.hist(df[feat1].dropna(), bins=30, edgecolor='black', alpha=0.7)
        else:
            plt.scatter(df[feat2], df[feat1], alpha=0.2, s=5)
        if i == len(music_features) - 1:
            plt.xlabel(feat2, fontsize=8)
        if j == 0:
            plt.ylabel(feat1, fontsize=8)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

plt.suptitle('Matriz de Scatterplots - Características Musicales', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('matriz_scatterplots.png', dpi=300, bbox_inches='tight')
plt.close()


print("\nOutliers")


# Método IQR (Interquartile Range)
outliers_info = {}

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outliers_info[col] = {
        'count': len(outliers),
        'percentage': (len(outliers) / len(df)) * 100,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }
    
    print(f"\n{col.upper()}:")
    print(f"  Rango válido: [{lower_bound:.4f}, {upper_bound:.4f}]")
    print(f"  Outliers detectados: {len(outliers)} ({outliers_info[col]['percentage']:.2f}%)")

# Visualización de outliers
fig, axes = plt.subplots(4, 3, figsize=(15, 16))
axes = axes.ravel()

for idx, col in enumerate(numeric_cols):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    normal_data = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    outlier_data = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    axes[idx].scatter(range(len(normal_data)), normal_data[col].values, 
                     alpha=0.5, s=10, label='Normal', color='blue')
    if len(outlier_data) > 0:
        # Crear índices para outliers
        outlier_indices = outlier_data.index
        axes[idx].scatter(outlier_indices, outlier_data[col].values, 
                         alpha=0.7, s=20, label='Outliers', color='red')
    
    axes[idx].axhline(y=lower_bound, color='green', linestyle='--', linewidth=1, alpha=0.7)
    axes[idx].axhline(y=upper_bound, color='green', linestyle='--', linewidth=1, alpha=0.7)
    axes[idx].set_title(f'{col} - Outliers en rojo', fontweight='bold', fontsize=10)
    axes[idx].set_ylabel(col)
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('deteccion_outliers.png', dpi=300, bbox_inches='tight')
plt.close()


print("\n Curado")

print(f"Filas originales: {len(df)}")

# Crear copia para curado
df_curado = df.copy()

# 1. Eliminar duplicados
df_curado = df_curado.drop_duplicates(subset=['track_id'], keep='first')
print(f"Filas después de eliminar duplicados: {len(df_curado)}")

# 2. Eliminar filas con valores nulos en columnas críticas
critical_cols = ['track_name', 'artist_name', 'danceability', 'energy', 
                 'valence', 'tempo', 'track_popularity']
df_curado = df_curado.dropna(subset=critical_cols)
print(f"Filas después de eliminar valores nulos críticos: {len(df_curado)}")

# 3. Eliminar outliers extremos (usando método IQR más permisivo - 3*IQR)
# Solo para variables que no deberían tener valores extremos naturalmente
outlier_cols = ['tempo', 'loudness', 'duration_ms']

for col in outlier_cols:
    Q1 = df_curado[col].quantile(0.25)
    Q3 = df_curado[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR  # Más permisivo con 3*IQR
    upper_bound = Q3 + 3 * IQR
    
    before = len(df_curado)
    df_curado = df_curado[(df_curado[col] >= lower_bound) & (df_curado[col] <= upper_bound)]
    removed = before - len(df_curado)
    print(f"  {col}: eliminadas {removed} filas con outliers extremos")

print(f"\nFilas después de eliminar outliers extremos: {len(df_curado)}")

# 4. Validar rangos de variables normalizadas (deben estar entre 0 y 1)
normalized_cols = ['danceability', 'energy', 'speechiness', 'acousticness', 
                   'instrumentalness', 'liveness', 'valence']

for col in normalized_cols:
    before = len(df_curado)
    df_curado = df_curado[(df_curado[col] >= 0) & (df_curado[col] <= 1)]
    removed = before - len(df_curado)
    if removed > 0:
        print(f"  {col}: eliminadas {removed} filas fuera del rango [0, 1]")

# 5. Validar que la duración sea razonable (entre 30 segundos y 20 minutos)
before = len(df_curado)
df_curado = df_curado[(df_curado['duration_ms'] >= 30000) & (df_curado['duration_ms'] <= 1200000)]
removed = before - len(df_curado)
print(f"  duration_ms: eliminadas {removed} filas con duración no razonable")

# 6. Validar que tempo sea positivo y razonable
before = len(df_curado)
df_curado = df_curado[(df_curado['tempo'] > 0) & (df_curado['tempo'] < 300)]
removed = before - len(df_curado)
print(f"  tempo: eliminadas {removed} filas con tempo no razonable")

# 7. Validar popularidad (0-100)
before = len(df_curado)
df_curado = df_curado[(df_curado['track_popularity'] >= 0) & (df_curado['track_popularity'] <= 100)]
df_curado = df_curado[(df_curado['artist_popularity'] >= 0) & (df_curado['artist_popularity'] <= 100)]
removed = before - len(df_curado)
print(f"  popularity: eliminadas {removed} filas con popularidad fuera de rango")
print(f"  Porcentaje retenido: {(len(df_curado) / len(df)) * 100:.2f}%")


df_curado.to_csv('playlist_2010to2023_curado.csv', index=False)




comparison_cols = ['danceability', 'energy', 'valence', 'tempo', 'track_popularity']

fig, axes = plt.subplots(len(comparison_cols), 2, figsize=(14, 15))

for idx, col in enumerate(comparison_cols):
    # Antes
    axes[idx, 0].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[idx, 0].set_title(f'{col} - ANTES (n={len(df)})', fontweight='bold')
    axes[idx, 0].set_xlabel(col)
    axes[idx, 0].set_ylabel('Frecuencia')
    axes[idx, 0].axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label='Media')
    axes[idx, 0].axvline(df[col].median(), color='green', linestyle='--', linewidth=2, label='Mediana')
    axes[idx, 0].legend()
    axes[idx, 0].grid(True, alpha=0.3)
    
    # Después
    axes[idx, 1].hist(df_curado[col].dropna(), bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[idx, 1].set_title(f'{col} - DESPUÉS (n={len(df_curado)})', fontweight='bold')
    axes[idx, 1].set_xlabel(col)
    axes[idx, 1].set_ylabel('Frecuencia')
    axes[idx, 1].axvline(df_curado[col].mean(), color='red', linestyle='--', linewidth=2, label='Media')
    axes[idx, 1].axvline(df_curado[col].median(), color='green', linestyle='--', linewidth=2, label='Mediana')
    axes[idx, 1].legend()
    axes[idx, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparacion_antes_despues.png', dpi=300, bbox_inches='tight')
plt.close()

# Estadísticas de comparación
print("\nEstadísticas comparativas:")
for col in comparison_cols:
    print(f"\n{col.upper()}:")
    print(f"  ANTES  - Media: {df[col].mean():.4f}, Mediana: {df[col].median():.4f}, Std: {df[col].std():.4f}")
    print(f"  DESPUÉS - Media: {df_curado[col].mean():.4f}, Mediana: {df_curado[col].median():.4f}, Std: {df_curado[col].std():.4f}")


# Matriz de correlación del dataset curado
correlation_cols = ['danceability', 'energy', 'loudness', 'speechiness', 
                   'acousticness', 'instrumentalness', 'liveness', 'valence', 
                   'tempo', 'track_popularity', 'artist_popularity']

correlation_matrix = df_curado[correlation_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlación - Dataset Curado', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('matriz_correlacion.png', dpi=300, bbox_inches='tight')
plt.close()

# Correlaciones más fuertes
print("\nCorrelaciones más fuertes (|r| > 0.3):")
correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > 0.3:
            correlations.append((correlation_matrix.columns[i], 
                               correlation_matrix.columns[j], 
                               corr_value))

correlations.sort(key=lambda x: abs(x[2]), reverse=True)
for var1, var2, corr in correlations:
    print(f"  {var1} <-> {var2}: {corr:.4f}")

