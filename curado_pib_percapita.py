"""
Script para el curado y análisis exploratorio del dataset del Banco Mundial
Datos de crecimiento del PIB per cápita mundial (2000-2023)
Incluye visualizaciones, detección de outliers y limpieza de datos
Enfocado en el crecimiento económico per cápita global
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
    df_full = pd.read_csv('P_Popular_Indicators/TasaCrecimientoPIBpc.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        df_full = pd.read_csv('P_Popular_Indicators/TasaCrecimientoPIBpc.csv', encoding='latin-1')
    except UnicodeDecodeError:
        df_full = pd.read_csv('P_Popular_Indicators/TasaCrecimientoPIBpc.csv', encoding='iso-8859-1')

# Filtrar solo los datos del indicador de PIB per cápita
df = df_full[df_full['Series Name'] == 'GDP per capita growth (annual %)'].copy()

print(f"Dataset completo: {df_full.shape[0]} filas x {df_full.shape[1]} columnas")
print(f"Dataset filtrado (crecimiento PIB per cápita): {df.shape[0]} filas x {df.shape[1]} columnas")

print(f"\nDimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"\nIndicador analizado: {df['Series Name'].iloc[0]}")
print(f"País/Región: {df['Country Name'].iloc[0]}")
print(f"\nPrimeras columnas:")
print(df[['Series Name', 'Series Code', 'Country Name', 'Country Code']].head())

print(df.info())

# Identificar columnas de años (rango 2000-2023)
year_columns = [col for col in df.columns if '[YR' in col and any(str(year) in col for year in range(2000, 2024))]
print(f"\nColumnas de años encontradas: {len(year_columns)}")
print(f"Rango: {year_columns[0]} hasta {year_columns[-1]}")

# Crear DataFrame en formato long
id_vars = ['Series Name', 'Series Code', 'Country Name', 'Country Code']
df_long = df.melt(id_vars=id_vars, var_name='Year', value_name='Value')

# Limpiar la columna Year
df_long['Year'] = df_long['Year'].str.extract(r'(\d{4})').astype(int)

# Convertir valores a numérico (manejar '..' como NaN)
df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')

# Eliminar filas con valores nulos
df_long = df_long.dropna(subset=['Value'])

print(f"\nDimensiones formato long: {df_long.shape[0]} filas x {df_long.shape[1]} columnas")
print(f"\nPrimeras filas del formato long:")
print(df_long.head(10))
print(f"\nRango de años: {df_long['Year'].min()} - {df_long['Year'].max()}")

print(f"\nEstadísticas del crecimiento del PIB per cápita:")
print(df_long['Value'].describe())

print(f"\nValores nulos:")
print(df_long.isnull().sum())
print(f"\nPorcentaje de valores nulos: {(df_long['Value'].isnull().sum() / len(df_long)) * 100:.2f}%")

# Estadísticas por año
years_in_data = df_long['Year'].max() - df_long['Year'].min() + 1
year_stats = df_long.groupby('Year')['Value'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
print(f"\nEstadísticas por año ({years_in_data} años):")
print(year_stats.round(4))

# Crear directorio para imágenes si no existe
import os
os.makedirs('results_images', exist_ok=True)

# Visualización 1: Distribuciones
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histograma general
axes[0, 0].hist(df_long['Value'].dropna(), bins=15, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].set_title('Distribución del Crecimiento PIB per cápita (%)', fontweight='bold', fontsize=12)
axes[0, 0].set_xlabel('Crecimiento PIB per cápita (%)')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].axvline(df_long['Value'].mean(), color='red', linestyle='--', linewidth=2, label='Media')
axes[0, 0].axvline(df_long['Value'].median(), color='green', linestyle='--', linewidth=2, label='Mediana')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Boxplot
axes[0, 1].boxplot(df_long['Value'].dropna(), vert=True)
axes[0, 1].set_title('Boxplot del Crecimiento PIB per cápita', fontweight='bold', fontsize=12)
axes[0, 1].set_ylabel('Crecimiento PIB per cápita (%)')
axes[0, 1].grid(True, alpha=0.3)

# Serie temporal
axes[1, 0].plot(df_long['Year'], df_long['Value'], marker='o', linewidth=2, markersize=8, color='darkblue')
axes[1, 0].set_title('Evolución del Crecimiento PIB per cápita Mundial (2000-2023)', fontweight='bold', fontsize=12)
axes[1, 0].set_xlabel('Año')
axes[1, 0].set_ylabel('Crecimiento PIB per cápita (%)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)

# Estadísticas por década
df_long['Decade'] = (df_long['Year'] // 10) * 10
decade_stats = df_long.groupby('Decade')['Value'].agg(['mean', 'std']).round(2)
decades = decade_stats.index
means = decade_stats['mean']
stds = decade_stats['std']

axes[1, 1].bar(decades, means, yerr=stds, alpha=0.7, color='coral', 
               capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
axes[1, 1].set_title('Crecimiento Promedio por Década', fontweight='bold', fontsize=12)
axes[1, 1].set_xlabel('Década')
axes[1, 1].set_ylabel('Crecimiento Promedio (%)')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results_images/pib_distribucion_general.png', dpi=300, bbox_inches='tight')
   
plt.close()

# Visualización 2: Análisis temporal detallado
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Línea de tendencia con suavizado
from scipy.signal import savgol_filter
if len(df_long) > 5:
    smooth_values = savgol_filter(df_long['Value'], window_length=5, polyorder=2)
    axes[0, 0].plot(df_long['Year'], df_long['Value'], 'o-', alpha=0.7, label='Datos originales', color='steelblue')
    axes[0, 0].plot(df_long['Year'], smooth_values, '-', linewidth=3, label='Tendencia suavizada', color='red')
else:
    axes[0, 0].plot(df_long['Year'], df_long['Value'], 'o-', alpha=0.7, label='Datos originales', color='steelblue')

axes[0, 0].set_title('Tendencia del Crecimiento PIB per cápita', fontweight='bold')
axes[0, 0].set_xlabel('Año')
axes[0, 0].set_ylabel('Crecimiento (%)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

# Identificar crisis y recuperaciones
crisis_years = df_long[df_long['Value'] < 0]['Year'].tolist()
recovery_years = df_long[df_long['Value'] > 4]['Year'].tolist()

axes[0, 1].scatter(df_long['Year'], df_long['Value'], s=100, alpha=0.7, color='lightblue')
if crisis_years:
    crisis_data = df_long[df_long['Year'].isin(crisis_years)]
    axes[0, 1].scatter(crisis_data['Year'], crisis_data['Value'], s=150, color='red', 
                       label=f'Crisis ({len(crisis_years)} años)', marker='v')
if recovery_years:
    recovery_data = df_long[df_long['Year'].isin(recovery_years)]
    axes[0, 1].scatter(recovery_data['Year'], recovery_data['Value'], s=150, color='green', 
                       label=f'Fuerte crecimiento ({len(recovery_years)} años)', marker='^')

axes[0, 1].set_title('Identificación de Crisis y Recuperaciones', fontweight='bold')
axes[0, 1].set_xlabel('Año')
axes[0, 1].set_ylabel('Crecimiento (%)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)

# Análisis de volatilidad (ventana móvil)
window_size = 5
rolling_std = df_long.set_index('Year')['Value'].rolling(window=window_size).std()
axes[1, 0].plot(rolling_std.index, rolling_std.values, 'o-', color='orange', linewidth=2)
axes[1, 0].set_title(f'Volatilidad del Crecimiento (ventana móvil {window_size} años)', fontweight='bold')
axes[1, 0].set_xlabel('Año')
axes[1, 0].set_ylabel('Desviación Estándar Móvil')
axes[1, 0].grid(True, alpha=0.3)

# Distribución acumulativa
sorted_values = np.sort(df_long['Value'])
cumulative_prob = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
axes[1, 1].plot(sorted_values, cumulative_prob, 'o-', color='purple', linewidth=2)
axes[1, 1].set_title('Función de Distribución Acumulativa', fontweight='bold')
axes[1, 1].set_xlabel('Crecimiento PIB per cápita (%)')
axes[1, 1].set_ylabel('Probabilidad Acumulada')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Crecimiento = 0%')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('results_images/pib_analisis_temporal.png', dpi=300, bbox_inches='tight')

plt.close()

# Detección de outliers


# Método IQR
Q1 = df_long['Value'].quantile(0.25)
Q3 = df_long['Value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_long[(df_long['Value'] < lower_bound) | (df_long['Value'] > upper_bound)]
print(f"Rango válido (IQR): [{lower_bound:.4f}, {upper_bound:.4f}]")
print(f"Outliers detectados: {len(outliers)} de {len(df_long)} ({(len(outliers) / len(df_long)) * 100:.2f}%)")

if not outliers.empty:
    print(f"\nOutliers identificados:")
    for _, row in outliers.iterrows():
        print(f"  {row['Year']}: {row['Value']:.2f}%")

# Método Z-score
z_scores = np.abs(stats.zscore(df_long['Value']))
z_outliers = df_long[z_scores > 2]
print(f"\nOutliers por Z-score (|z| > 2): {len(z_outliers)} ({(len(z_outliers) / len(df_long)) * 100:.2f}%)")

# Visualización de outliers
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Boxplot con outliers marcados
normal_data = df_long[(df_long['Value'] >= lower_bound) & (df_long['Value'] <= upper_bound)]['Value']
outlier_data = df_long[(df_long['Value'] < lower_bound) | (df_long['Value'] > upper_bound)]['Value']

axes[0, 0].boxplot([normal_data, outlier_data], labels=['Normales', 'Outliers'], vert=True)
axes[0, 0].set_title('Comparación: Datos Normales vs Outliers', fontweight='bold')
axes[0, 0].set_ylabel('Crecimiento PIB per cápita (%)')
axes[0, 0].grid(True, alpha=0.3)

# Serie temporal con outliers marcados
axes[0, 1].plot(df_long['Year'], df_long['Value'], 'o-', alpha=0.7, color='steelblue', label='Datos normales')
if not outliers.empty:
    axes[0, 1].scatter(outliers['Year'], outliers['Value'], color='red', s=100, 
                       label=f'Outliers ({len(outliers)})', zorder=5, marker='s')
axes[0, 1].axhline(y=lower_bound, color='red', linestyle='--', alpha=0.5, label='Límites IQR')
axes[0, 1].axhline(y=upper_bound, color='red', linestyle='--', alpha=0.5)
axes[0, 1].set_title('Serie Temporal con Outliers Marcados', fontweight='bold')
axes[0, 1].set_xlabel('Año')
axes[0, 1].set_ylabel('Crecimiento (%)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Histograma con outliers
axes[1, 0].hist(normal_data, bins=10, alpha=0.7, color='steelblue', label='Datos normales')
if len(outlier_data) > 0:
    axes[1, 0].hist(outlier_data, bins=5, alpha=0.7, color='red', label='Outliers')
axes[1, 0].axvline(x=lower_bound, color='red', linestyle='--', alpha=0.7)
axes[1, 0].axvline(x=upper_bound, color='red', linestyle='--', alpha=0.7)
axes[1, 0].set_title('Distribución con Outliers Marcados', fontweight='bold')
axes[1, 0].set_xlabel('Crecimiento (%)')
axes[1, 0].set_ylabel('Frecuencia')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Q-Q plot para normalidad
stats.probplot(df_long['Value'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Normalidad)', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results_images/pib_deteccion_outliers.png', dpi=300, bbox_inches='tight')

plt.close()

# Proceso de curado

print(f"Filas originales: {len(df_long)}")

# Crear copia para curado
df_curado = df_long.copy()

# 1. Ya no hay valores nulos que eliminar
print(f"Filas sin valores nulos: {len(df_curado)}")

# 2. Eliminar outliers extremos (usando 2.5*IQR para ser más conservador)
Q1 = df_curado['Value'].quantile(0.25)
Q3 = df_curado['Value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound_extreme = Q1 - 2.5 * IQR
upper_bound_extreme = Q3 + 2.5 * IQR

before = len(df_curado)
df_curado = df_curado[(df_curado['Value'] >= lower_bound_extreme) & 
                      (df_curado['Value'] <= upper_bound_extreme)]
removed = before - len(df_curado)
print(f"Filas después de eliminar outliers extremos (2.5*IQR): {len(df_curado)} (eliminadas: {removed})")

print(f"\nDATASET FINAL: {len(df_curado)} filas ({len(df_long) - len(df_curado)} eliminadas)")
print(f"Porcentaje retenido: {(len(df_curado) / len(df_long)) * 100:.2f}%")

# Convertir de vuelta a formato wide para guardar
if len(df_curado) > 0:
    df_curado_wide = df_curado.pivot_table(
        index=['Series Name', 'Series Code', 'Country Name', 'Country Code'],
        columns='Year',
        values='Value'
    ).reset_index()

    # Renombrar columnas al formato original
    df_curado_wide.columns = [f'{col} [YR{col}]' if isinstance(col, int) else col 
                              for col in df_curado_wide.columns]

    # Guardar archivos curados
    df_curado_wide.to_csv('TasaCrecimientoPIBpc_curado.csv', index=False)
    df_curado.to_csv('TasaCrecimientoPIBpc_curado_long.csv', index=False)


    # Comparación antes/después
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Serie temporal comparativa
    axes[0, 0].plot(df_long['Year'], df_long['Value'], 'o-', alpha=0.7, color='coral', 
                    label=f'Antes (n={len(df_long)})', linewidth=2)
    axes[0, 0].plot(df_curado['Year'], df_curado['Value'], 'o-', alpha=0.8, color='steelblue', 
                    label=f'Después (n={len(df_curado)})', linewidth=2)
    axes[0, 0].set_title('Comparación Temporal: Antes vs Después', fontweight='bold')
    axes[0, 0].set_xlabel('Año')
    axes[0, 0].set_ylabel('Crecimiento PIB per cápita (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Histogramas comparativos
    axes[0, 1].hist(df_long['Value'], bins=10, alpha=0.7, color='coral', label='Antes', density=True)
    axes[0, 1].hist(df_curado['Value'], bins=10, alpha=0.7, color='steelblue', label='Después', density=True)
    axes[0, 1].axvline(df_long['Value'].mean(), color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 1].axvline(df_curado['Value'].mean(), color='blue', linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 1].set_title('Comparación de Distribuciones', fontweight='bold')
    axes[0, 1].set_xlabel('Crecimiento (%)')
    axes[0, 1].set_ylabel('Densidad')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Boxplots comparativos
    axes[1, 0].boxplot([df_long['Value'], df_curado['Value']], 
                       labels=['Antes', 'Después'], vert=True)
    axes[1, 0].set_title('Comparación Boxplot', fontweight='bold')
    axes[1, 0].set_ylabel('Crecimiento (%)')
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
    plt.savefig('results_images/pib_comparacion_antes_despues.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nEstadísticas comparativas:")
    print(stats_comparison.round(4))

  
    if len(crisis_years) > 0:
        print(f"Años de crisis (crecimiento negativo): {crisis_years}")
    if len(recovery_years) > 0:
        print(f"Años de fuerte crecimiento (>4%): {recovery_years}")
    
    print(f"Calidad del curado: {(len(df_curado) / len(df_long)) * 100:.1f}% de datos conservados")

else:
    print(" No hay datos válidos después del curado")

