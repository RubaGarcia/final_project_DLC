"""
Script para el curado y análisis exploratorio del dataset del Banco Mundial
Datos de indicadores mundiales (2000-2023)
Incluye visualizaciones, detección de outliers y limpieza de datos
Análisis de múltiples indicadores a nivel mundial
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
    df_full = pd.read_csv('P_Popular Indicators/f42e0515-61ae-4207-bcbf-ceea36d0c854_Data.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        df_full = pd.read_csv('P_Popular Indicators/f42e0515-61ae-4207-bcbf-ceea36d0c854_Data.csv', encoding='latin-1')
    except UnicodeDecodeError:
        df_full = pd.read_csv('P_Popular Indicators/f42e0515-61ae-4207-bcbf-ceea36d0c854_Data.csv', encoding='iso-8859-1')

# Limpiar el dataset eliminando filas vacías o de metadatos
df = df_full[df_full['Country Name'] == 'World'].copy()
df = df.dropna(subset=['Series Name'])
df = df[df['Series Name'].notna() & (df['Series Name'] != '')]



print(f"Dimensiones del dataset mundial: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"\nIndicadores disponibles:")
for i, indicator in enumerate(df['Series Name'].unique(), 1):
    print(f"{i:2d}. {indicator}")
print(f"\nPrimeras filas:")
print(df[['Series Name', 'Series Code']].head())


print(df.info())

# Identificar columnas de años (ajustar para el rango 2000-2023)
year_columns = [col for col in df.columns if '[YR' in col and any(str(year) in col for year in range(2000, 2024))]
print(f"\nColumnas de años encontradas: {len(year_columns)}")
print(f"Rango: {year_columns[0]} hasta {year_columns[-1]}")


# Convertir a formato long para análisis más fácil


# Crear DataFrame en formato long
id_vars = ['Series Name', 'Series Code', 'Country Name', 'Country Code']
df_long = df.melt(id_vars=id_vars, var_name='Year', value_name='Value')

# Limpiar la columna Year
df_long['Year'] = df_long['Year'].str.extract(r'(\d{4})').astype(int)

# Convertir valores a numérico (manejar '..' como NaN)
df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')

# Eliminar filas con valores nulos o indicadores vacíos
df_long = df_long.dropna(subset=['Value', 'Series Name'])
df_long = df_long[df_long['Series Name'].str.strip() != '']

print(f"  Dimensiones formato long: {df_long.shape[0]} filas x {df_long.shape[1]} columnas")
print(f"\nPrimeras filas del formato long:")
print(df_long.head(10))
print(f"\nRango de años: {df_long['Year'].min()} - {df_long['Year'].max()}")
print(f"Indicadores con datos: {df_long['Series Name'].nunique()}")


print(f"  Dimensiones formato long: {df_long.shape[0]} filas x {df_long.shape[1]} columnas")
print(f"\nPrimeras filas del formato long:")
print(df_long.head(10))
print(f"\nRango de años: {df_long['Year'].min()} - {df_long['Year'].max()}")


print(f"\nEstadísticas generales de todos los indicadores:")
print(df_long['Value'].describe())

print(f"\nValores nulos:")
print(df_long.isnull().sum())
print(f"\nPorcentaje de valores nulos: {(df_long['Value'].isnull().sum() / len(df_long)) * 100:.2f}%")

# Estadísticas por indicador
print(f"\nEstadísticas por indicador:")
indicator_stats = df_long.groupby('Series Name')['Value'].agg(['count', 'mean', 'std', 'min', 'max']).round(4)
print(indicator_stats)



# Datos por año
years_in_data = df_long['Year'].max() - df_long['Year'].min() + 1
year_stats = df_long.groupby('Year')['Value'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
print(f"\nEstadísticas por año ({years_in_data} años) - todos los indicadores:")
print(year_stats.round(4))



# Histograma de todos los valores (múltiples indicadores)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histograma general de todos los valores
axes[0, 0].hist(df_long['Value'].dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribución de Todos los Indicadores', fontweight='bold', fontsize=12)
axes[0, 0].set_xlabel('Valores')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].axvline(df_long['Value'].mean(), color='red', linestyle='--', linewidth=2, label='Media')
axes[0, 0].axvline(df_long['Value'].median(), color='green', linestyle='--', linewidth=2, label='Mediana')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Histograma sin outliers extremos
q1 = df_long['Value'].quantile(0.05)
q99 = df_long['Value'].quantile(0.95)
df_filtered = df_long[(df_long['Value'] >= q1) & (df_long['Value'] <= q99)]
axes[0, 1].hist(df_filtered['Value'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='coral')
axes[0, 1].set_title('Distribución sin Outliers Extremos (5%-95%)', fontweight='bold', fontsize=12)
axes[0, 1].set_xlabel('Valores')
axes[0, 1].set_ylabel('Frecuencia')
axes[0, 1].grid(True, alpha=0.3)

# Boxplot
axes[1, 0].boxplot(df_long['Value'].dropna(), vert=True)
axes[1, 0].set_title('Boxplot de Todos los Indicadores', fontweight='bold', fontsize=12)
axes[1, 0].set_ylabel('Valores')
axes[1, 0].grid(True, alpha=0.3)

# Conteo de indicadores por tipo (aproximado por rangos de valores)
value_ranges = pd.cut(df_long['Value'].dropna(), bins=10)
range_counts = value_ranges.value_counts().sort_index()
axes[1, 1].bar(range(len(range_counts)), range_counts.values, color='skyblue')
axes[1, 1].set_title('Distribución por Rangos de Valores', fontweight='bold', fontsize=12)
axes[1, 1].set_xlabel('Rangos de Valores')
axes[1, 1].set_ylabel('Frecuencia')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results_images/wb_distribucion_general.png', dpi=300, bbox_inches='tight')
plt.close()



# Evolución temporal de indicadores seleccionados
selected_indicators = [
    'GDP growth (annual %)',
    'Inflation, consumer prices (annual %)', 
    'Life expectancy at birth, total (years)',
    'Primary completion rate, total (% of relevant age group)',
    'Prevalence of underweight, weight for age (% of children under 5)'
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, indicator in enumerate(selected_indicators):
    if indicator in df_long['Series Name'].values:
        data = df_long[df_long['Series Name'] == indicator]
        if not data.empty:
            axes[i].plot(data['Year'], data['Value'], marker='o', linewidth=2, markersize=6)
            axes[i].set_title(f'{indicator}', fontsize=10, fontweight='bold')
            axes[i].set_xlabel('Año')
            axes[i].set_ylabel('Valor')
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)

# Gráfico general de tendencias (normalizado)
all_indicators_norm = []
for indicator in df_long['Series Name'].unique():
    if len(df_long[df_long['Series Name'] == indicator]) > 10:  # Solo indicadores con suficientes datos
        data = df_long[df_long['Series Name'] == indicator].sort_values('Year')
        # Normalizar valores entre 0 y 1 para comparación
        if data['Value'].std() > 0:
            data_norm = data.copy()
            data_norm['Value_norm'] = (data['Value'] - data['Value'].min()) / (data['Value'].max() - data['Value'].min())
            axes[5].plot(data_norm['Year'], data_norm['Value_norm'], alpha=0.7, linewidth=1, label=indicator[:20] + '...' if len(indicator) > 20 else indicator)

axes[5].set_title('Tendencias Normalizadas de Todos los Indicadores', fontsize=10, fontweight='bold')
axes[5].set_xlabel('Año')
axes[5].set_ylabel('Valor Normalizado (0-1)')
axes[5].grid(True, alpha=0.3)
axes[5].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('results_images/wb_evolucion_temporal.png', dpi=300, bbox_inches='tight')
plt.close()



# Ranking de indicadores por variabilidad
indicator_stats = df_long.groupby('Series Name')['Value'].agg(['mean', 'std', 'min', 'max']).round(4)
indicator_stats['range'] = indicator_stats['max'] - indicator_stats['min']
indicator_stats['cv'] = (indicator_stats['std'] / abs(indicator_stats['mean'])) * 100  # Coeficiente de variación

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top indicadores por rango de valores
top_range = indicator_stats.nlargest(10, 'range')
axes[0, 0].barh(range(len(top_range)), top_range['range'].values)
axes[0, 0].set_yticks(range(len(top_range)))
axes[0, 0].set_yticklabels([name[:30] + '...' if len(name) > 30 else name for name in top_range.index], fontsize=9)
axes[0, 0].set_title('Top 10 Indicadores por Rango de Valores', fontweight='bold', fontsize=12)
axes[0, 0].set_xlabel('Rango (Max - Min)')
axes[0, 0].grid(True, alpha=0.3, axis='x')

# Top indicadores por desviación estándar
top_std = indicator_stats.nlargest(10, 'std')
axes[0, 1].barh(range(len(top_std)), top_std['std'].values, color='coral')
axes[0, 1].set_yticks(range(len(top_std)))
axes[0, 1].set_yticklabels([name[:30] + '...' if len(name) > 30 else name for name in top_std.index], fontsize=9)
axes[0, 1].set_title('Top 10 Indicadores por Desviación Estándar', fontweight='bold', fontsize=12)
axes[0, 1].set_xlabel('Desviación Estándar')
axes[0, 1].grid(True, alpha=0.3, axis='x')

# Distribución de valores medios
axes[1, 0].hist(indicator_stats['mean'].dropna(), bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
axes[1, 0].set_title('Distribución de Valores Medios por Indicador', fontweight='bold', fontsize=12)
axes[1, 0].set_xlabel('Valor Medio')
axes[1, 0].set_ylabel('Frecuencia')
axes[1, 0].grid(True, alpha=0.3)

# Top indicadores por coeficiente de variación (solo valores finitos)
cv_finite = indicator_stats[np.isfinite(indicator_stats['cv'])].nlargest(10, 'cv')
axes[1, 1].barh(range(len(cv_finite)), cv_finite['cv'].values, color='gold')
axes[1, 1].set_yticks(range(len(cv_finite)))
axes[1, 1].set_yticklabels([name[:30] + '...' if len(name) > 30 else name for name in cv_finite.index], fontsize=9)
axes[1, 1].set_title('Top 10 Indicadores por Coeficiente de Variación', fontweight='bold', fontsize=12)
axes[1, 1].set_xlabel('Coeficiente de Variación (%)')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('results_images/wb_ranking_paises.png', dpi=300, bbox_inches='tight')
plt.close()


# Heatmap de correlaciones entre indicadores (requiere pivot)
# Crear matriz de correlación entre indicadores
pivot_data = df_long.pivot(index='Year', columns='Series Name', values='Value')
correlation_matrix = pivot_data.corr()

# Seleccionar solo indicadores con correlaciones significativas
# Filtrar indicadores con suficientes datos (al menos 15 años de datos)
valid_indicators = pivot_data.count()[pivot_data.count() >= 15].index
correlation_subset = correlation_matrix.loc[valid_indicators, valid_indicators]

if len(correlation_subset) > 1:
    plt.figure(figsize=(14, 12))
    
    # Crear heatmap de correlación
    mask = np.triu(np.ones_like(correlation_subset, dtype=bool))  # Máscara para mostrar solo la mitad inferior
    
    sns.heatmap(correlation_subset, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0, 
                square=True,
                fmt='.2f',
                cbar_kws={'label': 'Correlación'},
                xticklabels=[name[:20] + '...' if len(name) > 20 else name for name in correlation_subset.columns],
                yticklabels=[name[:20] + '...' if len(name) > 20 else name for name in correlation_subset.index])
    
    plt.title('Matriz de Correlación entre Indicadores Mundiales (2000-2023)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('results_images/wb_heatmap_paises.png', dpi=300, bbox_inches='tight')
    plt.close()
else:
    print(" No hay suficientes indicadores con datos completos para crear matriz de correlación")





# Análisis de scatter plots y tendencias temporales
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Scatter general: año vs todos los valores
axes[0, 0].scatter(df_long['Year'], df_long['Value'], alpha=0.6, s=20)
axes[0, 0].set_title('Distribución Temporal de Todos los Indicadores', fontweight='bold')
axes[0, 0].set_xlabel('Año')
axes[0, 0].set_ylabel('Valores')
axes[0, 0].grid(True, alpha=0.3)

# Tendencias de algunos indicadores específicos
key_indicators = ['GDP growth (annual %)', 'Life expectancy at birth, total (years)', 'Inflation, consumer prices (annual %)']
colors = ['blue', 'green', 'red']
for i, (indicator, color) in enumerate(zip(key_indicators, colors)):
    if indicator in df_long['Series Name'].values:
        data = df_long[df_long['Series Name'] == indicator]
        axes[0, 1].plot(data['Year'], data['Value'], marker='o', color=color, 
                       label=indicator[:20] + '...' if len(indicator) > 20 else indicator, linewidth=2)

axes[0, 1].set_title('Tendencias de Indicadores Clave', fontweight='bold')
axes[0, 1].set_xlabel('Año')
axes[0, 1].set_ylabel('Valores')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# Distribución de valores por década
df_long['Decade'] = (df_long['Year'] // 10) * 10
for decade in df_long['Decade'].unique():
    decade_data = df_long[df_long['Decade'] == decade]
    axes[1, 0].scatter(decade_data['Year'], decade_data['Value'], 
                       alpha=0.4, s=15, label=f'{int(decade)}s')
axes[1, 0].set_title('Distribución por Década', fontweight='bold')
axes[1, 0].set_xlabel('Año')
axes[1, 0].set_ylabel('Valores')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Boxplot de valores por año (sample de años)
sample_years = sorted(df_long['Year'].unique())[::4]  # Cada 4 años para legibilidad
yearly_data = [df_long[df_long['Year'] == year]['Value'].dropna() for year in sample_years]
axes[1, 1].boxplot(yearly_data, labels=sample_years)
axes[1, 1].set_title('Distribución de Valores por Año (muestra)', fontweight='bold')
axes[1, 1].set_xlabel('Año')
axes[1, 1].set_ylabel('Valores')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results_images/wb_scatterplots.png', dpi=300, bbox_inches='tight')
plt.close()


# Detección de outliers por indicador (más apropiado para múltiples indicadores)


outliers_by_indicator = []
for indicator in df_long['Series Name'].unique():
    indicator_data = df_long[df_long['Series Name'] == indicator]['Value'].dropna()
    
    if len(indicator_data) > 3:  # Solo si hay suficientes datos
        Q1 = indicator_data.quantile(0.25)
        Q3 = indicator_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = indicator_data[(indicator_data < lower_bound) | (indicator_data > upper_bound)]
        
        if len(outliers) > 0:
            outliers_by_indicator.append({
                'Indicator': indicator,
                'Count': len(outliers),
                'Percentage': (len(outliers) / len(indicator_data)) * 100,
                'Lower_Bound': lower_bound,
                'Upper_Bound': upper_bound,
                'Min_Outlier': outliers.min(),
                'Max_Outlier': outliers.max()
            })

outliers_df = pd.DataFrame(outliers_by_indicator)
if not outliers_df.empty:
    print("Outliers detectados por indicador:")
    print(outliers_df[['Indicator', 'Count', 'Percentage']].round(2))

# Método IQR general para todo el dataset
Q1 = df_long['Value'].quantile(0.25)
Q3 = df_long['Value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_long[(df_long['Value'] < lower_bound) | (df_long['Value'] > upper_bound)]
print(f"\nRango válido general (IQR): [{lower_bound:.4f}, {upper_bound:.4f}]")
print(f"Outliers generales detectados: {len(outliers)} ({(len(outliers) / len(df_long.dropna())) * 100:.2f}%)")

# Top valores extremos
if not df_long.empty:
    print(f"\nTop 10 valores más extremos (positivos):")
    top_outliers = df_long.nlargest(10, 'Value')[['Series Name', 'Year', 'Value']]
    print(top_outliers.to_string(index=False))

    print(f"\nTop 10 valores más extremos (negativos):")
    bottom_outliers = df_long.nsmallest(10, 'Value')[['Series Name', 'Year', 'Value']]
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
plt.savefig('results_images/wb_deteccion_outliers.png', dpi=300, bbox_inches='tight')
plt.close()


print(f"Filas originales (formato long): {len(df_long)}")

# Crear copia para curado
df_curado = df_long.copy()

# 1. Ya eliminamos valores nulos anteriormente
print(f"Filas con datos válidos: {len(df_curado)}")

# 2. Eliminar outliers extremos por indicador (más conservador para datos heterogéneos)
rows_before = len(df_curado)
cleaned_data = []

for indicator in df_curado['Series Name'].unique():
    indicator_data = df_curado[df_curado['Series Name'] == indicator].copy()
    
    if len(indicator_data) > 3:
        Q1 = indicator_data['Value'].quantile(0.25)
        Q3 = indicator_data['Value'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Usar 2.5*IQR para ser más conservador con datos heterogéneos
        lower_bound = Q1 - 2.5 * IQR
        upper_bound = Q3 + 2.5 * IQR
        
        # Filtrar outliers extremos
        indicator_cleaned = indicator_data[
            (indicator_data['Value'] >= lower_bound) & 
            (indicator_data['Value'] <= upper_bound)
        ]
        cleaned_data.append(indicator_cleaned)
    else:
        cleaned_data.append(indicator_data)

df_curado = pd.concat(cleaned_data, ignore_index=True)
removed = rows_before - len(df_curado)
print(f"Filas después de eliminar outliers extremos (2.5*IQR por indicador): {len(df_curado)} (eliminadas: {removed})")

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
df_curado_wide.to_csv('f42e0515_Data_curado.csv', index=False)
df_curado.to_csv('f42e0515_Data_curado_long.csv', index=False)

print("  - f42e0515_Data_curado.csv (formato wide)")
print("  - f42e0515_Data_curado_long.csv (formato long)")


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogramas comparativos (muestra)
sample_size = min(1000, len(df_long.dropna()))
df_sample_before = df_long.dropna().sample(n=sample_size, random_state=42)
df_sample_after = df_curado.sample(n=min(sample_size, len(df_curado)), random_state=42)

axes[0, 0].hist(df_sample_before['Value'], bins=50, alpha=0.7, 
                edgecolor='black', color='coral', label='Antes')
axes[0, 0].axvline(df_long['Value'].mean(), color='red', linestyle='--', 
                   linewidth=2, label='Media')
axes[0, 0].set_title(f'ANTES - Distribución (n={len(df_long)})', fontweight='bold')
axes[0, 0].set_xlabel('Valores')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(df_sample_after['Value'], bins=50, alpha=0.7, 
                edgecolor='black', color='skyblue', label='Después')
axes[0, 1].axvline(df_curado['Value'].mean(), color='red', linestyle='--', 
                   linewidth=2, label='Media')
axes[0, 1].set_title(f'DESPUÉS - Distribución (n={len(df_curado)})', fontweight='bold')
axes[0, 1].set_xlabel('Valores')
axes[0, 1].set_ylabel('Frecuencia')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Boxplots comparativos (muestra)
axes[1, 0].boxplot([df_sample_before['Value'], df_sample_after['Value']],
                    labels=['Antes', 'Después'], vert=True)
axes[1, 0].set_title('Comparación Boxplot', fontweight='bold')
axes[1, 0].set_ylabel('Valores')
axes[1, 0].grid(True, alpha=0.3)

# Estadísticas comparativas
stats_comparison = pd.DataFrame({
    'Antes': [
        len(df_long),
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
plt.savefig('results_images/wb_comparacion_antes_despues.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nEstadísticas comparativas:")
print(stats_comparison)


