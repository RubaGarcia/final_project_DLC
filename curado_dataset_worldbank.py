"""
Script para el curado y análisis exploratorio del dataset del Banco Mundial
Datos de crecimiento poblacional por país (2000-2023)
Incluye visualizaciones, detección de outliers y limpieza de datos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# Cargar el dataset (probando diferentes codificaciones)
try:
    df = pd.read_csv('8fa87b74-34bf-4505-ac3f-1668554b3b6f_Data.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv('8fa87b74-34bf-4505-ac3f-1668554b3b6f_Data.csv', encoding='latin-1')
    except UnicodeDecodeError:
        df = pd.read_csv('8fa87b74-34bf-4505-ac3f-1668554b3b6f_Data.csv', encoding='iso-8859-1')


print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"\nPrimeras filas:")
print(df.head())


print(df.info())

# Identificar columnas de años
year_columns = [col for col in df.columns if '[YR' in col]


# Convertir a formato long para análisis más fácil


# Crear DataFrame en formato long
id_vars = ['Series Name', 'Series Code', 'Country Name', 'Country Code']
df_long = df.melt(id_vars=id_vars, var_name='Year', value_name='Value')

# Limpiar la columna Year
df_long['Year'] = df_long['Year'].str.extract(r'(\d{4})').astype(int)

# Convertir valores a numérico (manejar '..' como NaN)
df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')


print(f"  Dimensiones: {df_long.shape[0]} filas x {df_long.shape[1]} columnas")
print(f"\nPrimeras filas del formato long:")
print(df_long.head(10))


print(df_long['Value'].describe())

print(f"\nValores nulos:")
print(df_long.isnull().sum())
print(f"\nPorcentaje de valores nulos: {(df_long['Value'].isnull().sum() / len(df_long)) * 100:.2f}%")



# Países únicos
unique_countries = df_long['Country Name'].nunique()
print(f"Países únicos: {unique_countries}")

# Países con más datos
country_counts = df_long.groupby('Country Name')['Value'].count().sort_values(ascending=False)
print(f"\nTop 10 países con más datos:")
print(country_counts.head(10))

# Países con más datos faltantes
country_nulls = df_long.groupby('Country Name')['Value'].apply(lambda x: x.isnull().sum())
country_nulls_pct = (country_nulls / 24 * 100).sort_values(ascending=False)
print(f"\nTop 10 países con más datos faltantes (%):")
print(country_nulls_pct.head(10))



# Datos por año
year_stats = df_long.groupby('Year')['Value'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
print("\nEstadísticas por año:")
print(year_stats)



# Histograma de crecimiento poblacional
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histograma general
axes[0, 0].hist(df_long['Value'].dropna(), bins=100, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribución de Crecimiento Poblacional (%)', fontweight='bold', fontsize=12)
axes[0, 0].set_xlabel('Crecimiento Poblacional (%)')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].axvline(df_long['Value'].mean(), color='red', linestyle='--', linewidth=2, label='Media')
axes[0, 0].axvline(df_long['Value'].median(), color='green', linestyle='--', linewidth=2, label='Mediana')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Histograma sin outliers extremos
q1 = df_long['Value'].quantile(0.01)
q99 = df_long['Value'].quantile(0.99)
df_filtered = df_long[(df_long['Value'] >= q1) & (df_long['Value'] <= q99)]
axes[0, 1].hist(df_filtered['Value'].dropna(), bins=100, edgecolor='black', alpha=0.7, color='coral')
axes[0, 1].set_title('Distribución sin Outliers Extremos (1%-99%)', fontweight='bold', fontsize=12)
axes[0, 1].set_xlabel('Crecimiento Poblacional (%)')
axes[0, 1].set_ylabel('Frecuencia')
axes[0, 1].grid(True, alpha=0.3)

# Boxplot
axes[1, 0].boxplot(df_long['Value'].dropna(), vert=True)
axes[1, 0].set_title('Boxplot de Crecimiento Poblacional', fontweight='bold', fontsize=12)
axes[1, 0].set_ylabel('Crecimiento Poblacional (%)')
axes[1, 0].grid(True, alpha=0.3)

# Violin plot
axes[1, 1].violinplot(df_long['Value'].dropna(), vert=True, showmeans=True, showmedians=True)
axes[1, 1].set_title('Violin Plot de Crecimiento Poblacional', fontweight='bold', fontsize=12)
axes[1, 1].set_ylabel('Crecimiento Poblacional (%)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wb_distribucion_general.png', dpi=300, bbox_inches='tight')
print("✓ Distribuciones generales guardadas como 'wb_distribucion_general.png'")
plt.close()



# Tendencia promedio por año
avg_by_year = df_long.groupby('Year')['Value'].mean()

plt.figure(figsize=(14, 6))
plt.plot(avg_by_year.index, avg_by_year.values, marker='o', linewidth=2, markersize=8)
plt.title('Evolución del Crecimiento Poblacional Promedio Mundial (2000-2023)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Año', fontsize=12)
plt.ylabel('Crecimiento Poblacional Promedio (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
plt.tight_layout()
plt.savefig('wb_evolucion_temporal.png', dpi=300, bbox_inches='tight')
plt.close()



# Top y Bottom países promedio
country_avg = df_long.groupby('Country Name')['Value'].mean().sort_values()

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Top 20 países con mayor crecimiento
top_countries = country_avg.tail(20)
axes[0].barh(range(len(top_countries)), top_countries.values)
axes[0].set_yticks(range(len(top_countries)))
axes[0].set_yticklabels(top_countries.index, fontsize=9)
axes[0].set_title('Top 20 Países con Mayor Crecimiento Poblacional Promedio', 
                  fontweight='bold', fontsize=12)
axes[0].set_xlabel('Crecimiento Promedio (%)')
axes[0].grid(True, alpha=0.3, axis='x')

# Top 20 países con menor crecimiento (o decrecimiento)
bottom_countries = country_avg.head(20)
axes[1].barh(range(len(bottom_countries)), bottom_countries.values, color='coral')
axes[1].set_yticks(range(len(bottom_countries)))
axes[1].set_yticklabels(bottom_countries.index, fontsize=9)
axes[1].set_title('Top 20 Países con Menor Crecimiento Poblacional Promedio', 
                  fontweight='bold', fontsize=12)
axes[1].set_xlabel('Crecimiento Promedio (%)')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('wb_ranking_paises.png', dpi=300, bbox_inches='tight')
plt.close()


# Seleccionar algunos países importantes para visualización
selected_countries = ['United States', 'China', 'India', 'Brazil', 'Germany', 
                     'Japan', 'United Kingdom', 'France', 'Italy', 'Spain',
                     'Mexico', 'Russia', 'Canada', 'Australia', 'South Korea']

# Filtrar países que existan en el dataset
available_countries = [c for c in selected_countries if c in df['Country Name'].values]

if len(available_countries) > 0:
    df_selected = df[df['Country Name'].isin(available_countries)]
    
    # Preparar datos para heatmap
    heatmap_data = df_selected.set_index('Country Name')[year_columns]
    heatmap_data = heatmap_data.apply(pd.to_numeric, errors='coerce')
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(heatmap_data, cmap='RdYlGn', center=0, annot=False, 
                fmt='.2f', linewidths=0.5, cbar_kws={'label': 'Crecimiento (%)'})
    plt.title('Heatmap de Crecimiento Poblacional - Países Seleccionados (2000-2023)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Año', fontsize=12)
    plt.ylabel('País', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('wb_heatmap_paises.png', dpi=300, bbox_inches='tight')
    plt.close()



# Scatterplot: año vs crecimiento (muestra aleatoria para visualización)
sample_size = min(5000, len(df_long.dropna()))
df_sample = df_long.dropna().sample(n=sample_size, random_state=42)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Scatter general
axes[0, 0].scatter(df_sample['Year'], df_sample['Value'], alpha=0.3, s=10)
axes[0, 0].set_title('Distribución Temporal del Crecimiento', fontweight='bold')
axes[0, 0].set_xlabel('Año')
axes[0, 0].set_ylabel('Crecimiento (%)')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[0, 0].grid(True, alpha=0.3)

# Scatter por década
df_sample['Decade'] = (df_sample['Year'] // 10) * 10
for decade in df_sample['Decade'].unique():
    decade_data = df_sample[df_sample['Decade'] == decade]
    axes[0, 1].scatter(decade_data['Year'], decade_data['Value'], 
                       alpha=0.4, s=10, label=f'{decade}s')
axes[0, 1].set_title('Distribución por Década', fontweight='bold')
axes[0, 1].set_xlabel('Año')
axes[0, 1].set_ylabel('Crecimiento (%)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Tendencia por percentiles
percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
for p in percentiles:
    pct_by_year = df_long.groupby('Year')['Value'].quantile(p)
    axes[1, 0].plot(pct_by_year.index, pct_by_year.values, 
                    marker='o', label=f'P{int(p*100)}', linewidth=2)
axes[1, 0].set_title('Evolución de Percentiles', fontweight='bold')
axes[1, 0].set_xlabel('Año')
axes[1, 0].set_ylabel('Crecimiento (%)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

# Densidad por año (últimos 5 años)
recent_years = sorted(df_long['Year'].unique())[-5:]
for year in recent_years:
    year_data = df_long[df_long['Year'] == year]['Value'].dropna()
    axes[1, 1].hist(year_data, bins=50, alpha=0.4, label=str(year))
axes[1, 1].set_title('Distribución en Años Recientes', fontweight='bold')
axes[1, 1].set_xlabel('Crecimiento (%)')
axes[1, 1].set_ylabel('Frecuencia')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wb_scatterplots.png', dpi=300, bbox_inches='tight')
plt.close()


# Método IQR
Q1 = df_long['Value'].quantile(0.25)
Q3 = df_long['Value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_long[(df_long['Value'] < lower_bound) | (df_long['Value'] > upper_bound)]
print(f"Rango válido (IQR): [{lower_bound:.4f}, {upper_bound:.4f}]")
print(f"Outliers detectados: {len(outliers)} ({(len(outliers) / len(df_long.dropna())) * 100:.2f}%)")

# Outliers más extremos
print(f"\nTop 10 valores más extremos (positivos):")
top_outliers = df_long.nlargest(10, 'Value')[['Country Name', 'Year', 'Value']]
print(top_outliers.to_string(index=False))

print(f"\nTop 10 valores más extremos (negativos):")
bottom_outliers = df_long.nsmallest(10, 'Value')[['Country Name', 'Year', 'Value']]
print(bottom_outliers.to_string(index=False))

# Visualización de outliers
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Boxplot con outliers marcados
normal_data = df_long[(df_long['Value'] >= lower_bound) & (df_long['Value'] <= upper_bound)]['Value']
outlier_data = df_long[(df_long['Value'] < lower_bound) | (df_long['Value'] > upper_bound)]['Value']

axes[0, 0].boxplot([normal_data.dropna(), outlier_data.dropna()], 
                    labels=['Normales', 'Outliers'], vert=True)
axes[0, 0].set_title('Comparación: Datos Normales vs Outliers', fontweight='bold')
axes[0, 0].set_ylabel('Crecimiento (%)')
axes[0, 0].grid(True, alpha=0.3)

# Scatter con outliers marcados
sample_normal = normal_data.dropna().sample(n=min(3000, len(normal_data)), random_state=42)
axes[0, 1].scatter(range(len(sample_normal)), sample_normal.values, 
                   alpha=0.3, s=10, label='Normal', color='blue')
if len(outlier_data) > 0:
    axes[0, 1].scatter(range(len(outlier_data)), outlier_data.values, 
                       alpha=0.7, s=30, label='Outliers', color='red')
axes[0, 1].axhline(y=lower_bound, color='green', linestyle='--', linewidth=1, alpha=0.7)
axes[0, 1].axhline(y=upper_bound, color='green', linestyle='--', linewidth=1, alpha=0.7)
axes[0, 1].set_title('Distribución con Outliers Marcados', fontweight='bold')
axes[0, 1].set_ylabel('Crecimiento (%)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Histograma comparativo
axes[1, 0].hist(normal_data.dropna(), bins=100, alpha=0.7, label='Normal', color='blue')
axes[1, 0].hist(outlier_data.dropna(), bins=50, alpha=0.7, label='Outliers', color='red')
axes[1, 0].set_title('Distribución: Normal vs Outliers', fontweight='bold')
axes[1, 0].set_xlabel('Crecimiento (%)')
axes[1, 0].set_ylabel('Frecuencia')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Outliers por año
outliers_by_year = df_long[(df_long['Value'] < lower_bound) | 
                           (df_long['Value'] > upper_bound)].groupby('Year').size()
axes[1, 1].bar(outliers_by_year.index, outliers_by_year.values, color='coral')
axes[1, 1].set_title('Cantidad de Outliers por Año', fontweight='bold')
axes[1, 1].set_xlabel('Año')
axes[1, 1].set_ylabel('Número de Outliers')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('wb_deteccion_outliers.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Filas originales (formato long): {len(df_long)}")

# Crear copia para curado
df_curado = df_long.copy()

# 1. Eliminar valores nulos
df_curado = df_curado.dropna(subset=['Value'])
print(f"Filas después de eliminar valores nulos: {len(df_curado)}")

# 2. Eliminar outliers extremos (usando 3*IQR para ser más permisivo)
Q1 = df_curado['Value'].quantile(0.25)
Q3 = df_curado['Value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound_extreme = Q1 - 3 * IQR
upper_bound_extreme = Q3 + 3 * IQR

before = len(df_curado)
df_curado = df_curado[(df_curado['Value'] >= lower_bound_extreme) & 
                      (df_curado['Value'] <= upper_bound_extreme)]
removed = before - len(df_curado)
print(f"Filas después de eliminar outliers extremos (3*IQR): {len(df_curado)} (eliminadas: {removed})")

# 3. Validar rangos razonables (crecimiento poblacional entre -10% y +15%)
before = len(df_curado)
df_curado = df_curado[(df_curado['Value'] >= -10) & (df_curado['Value'] <= 15)]
removed = before - len(df_curado)
print(f"Filas después de validar rangos razonables: {len(df_curado)} (eliminadas: {removed})")

print(f"\nDATASET FINAL (long): {len(df_curado)} filas ({len(df_long) - len(df_curado)} eliminadas)")
print(f"  Porcentaje retenido: {(len(df_curado) / len(df_long)) * 100:.2f}%")

# Convertir de vuelta a formato wide para guardar
df_curado_wide = df_curado.pivot_table(
    index=['Series Name', 'Series Code', 'Country Name', 'Country Code'],
    columns='Year',
    values='Value'
).reset_index()

# Renombrar columnas al formato original
df_curado_wide.columns = [f'{col} [YR{col}]' if isinstance(col, int) else col 
                          for col in df_curado_wide.columns]

# Guardar
df_curado_wide.to_csv('8fa87b74_Data_curado.csv', index=False)
df_curado.to_csv('8fa87b74_Data_curado_long.csv', index=False)
print("\n✓ Dataset curado guardado:")
print("  - 8fa87b74_Data_curado.csv (formato wide)")
print("  - 8fa87b74_Data_curado_long.csv (formato long)")


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogramas comparativos
axes[0, 0].hist(df_long['Value'].dropna(), bins=100, alpha=0.7, 
                edgecolor='black', color='coral', label='Antes')
axes[0, 0].axvline(df_long['Value'].mean(), color='red', linestyle='--', 
                   linewidth=2, label='Media')
axes[0, 0].set_title(f'ANTES - Distribución (n={len(df_long.dropna())})', fontweight='bold')
axes[0, 0].set_xlabel('Crecimiento (%)')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(df_curado['Value'].dropna(), bins=100, alpha=0.7, 
                edgecolor='black', color='skyblue', label='Después')
axes[0, 1].axvline(df_curado['Value'].mean(), color='red', linestyle='--', 
                   linewidth=2, label='Media')
axes[0, 1].set_title(f'DESPUÉS - Distribución (n={len(df_curado)})', fontweight='bold')
axes[0, 1].set_xlabel('Crecimiento (%)')
axes[0, 1].set_ylabel('Frecuencia')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Boxplots comparativos
axes[1, 0].boxplot([df_long['Value'].dropna(), df_curado['Value'].dropna()],
                    labels=['Antes', 'Después'], vert=True)
axes[1, 0].set_title('Comparación Boxplot', fontweight='bold')
axes[1, 0].set_ylabel('Crecimiento (%)')
axes[1, 0].grid(True, alpha=0.3)

# Estadísticas comparativas
stats_comparison = pd.DataFrame({
    'Antes': [
        len(df_long.dropna()),
        df_long['Value'].mean(),
        df_long['Value'].median(),
        df_long['Value'].std(),
        df_long['Value'].min(),
        df_long['Value'].max()
    ],
    'Después': [
        len(df_curado),
        df_curado['Value'].mean(),
        df_curado['Value'].median(),
        df_curado['Value'].std(),
        df_curado['Value'].min(),
        df_curado['Value'].max()
    ]
}, index=['Count', 'Mean', 'Median', 'Std', 'Min', 'Max'])

axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=stats_comparison.round(4).values,
                         rowLabels=stats_comparison.index,
                         colLabels=stats_comparison.columns,
                         cellLoc='center',
                         loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
axes[1, 1].set_title('Estadísticas Comparativas', fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('wb_comparacion_antes_despues.png', dpi=300, bbox_inches='tight')
print("\n✓ Comparación antes/después guardada como 'wb_comparacion_antes_despues.png'")
plt.close()

print("\nEstadísticas comparativas:")
print(stats_comparison)


