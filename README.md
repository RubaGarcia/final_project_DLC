# Final Project DLC

## Dataset Spotify (playlist_2010to2023)

Análisis del dataset de Spotify con canciones del 2010 al 2023.

- **Archivo original**: `playlist_2010to2023.csv`
- **Archivo curado**: `datasets_curados/playlist_2010to2023_curado.csv`
- **Script de curado**: `curado_dataset.py`

## Dataset World Bank - Tasa de Crecimiento PIB per cápita

Análisis de indicadores del Banco Mundial.

- **Archivos curados**: 
  - `datasets_curados/TasaCrecimientoPIBpc_curado.csv`
  - `datasets_curados/TasaCrecimientoPIBpc_curado_long.csv`
- **Script de curado**: `curado_pib_percapita.py`

## Estructura del Proyecto

```
.
├── curado_dataset.py              # Script para curar dataset de Spotify
├── curado_pib_percapita.py       # Script para curar dataset del Banco Mundial
├── playlist_2010to2023.csv        # Dataset original de Spotify
├── datasets_curados/              # Datasets procesados y curados
│   ├── playlist_2010to2023_curado.csv
│   ├── TasaCrecimientoPIBpc_curado.csv
│   └── TasaCrecimientoPIBpc_curado_long.csv
├── P_Popular_Indicators/          # Indicadores populares del Banco Mundial
└── results_images/                # Imágenes de resultados y visualizaciones
```
