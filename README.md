# Estudio y comparación de modelos estocásticos para la simulación de series financieras

Repositorio asociado al Trabajo Fin de Grado del Grado en Matemática Computacional de la Universidad Internacional de La Rioja (UNIR).

## Descripción

Este proyecto estudia la aplicación de modelos estocásticos a la simulación y análisis de series temporales financieras.

El trabajo se centra inicialmente en el Movimiento Browniano Geométrico (GBM), utilizado habitualmente para modelizar la evolución de activos financieros. Se estudian sus propiedades teóricas, la estimación de sus parámetros a partir de datos reales y su implementación mediante simulación de Monte Carlo.

Posteriormente, se realiza una comparación entre diferentes enfoques. Por un lado, se compara la valoración obtenida mediante simulación de Monte Carlo con la solución analítica de Black--Scholes. Por otro lado, se compara el modelo GBM con un modelo GARCH(1,1), que permite modelizar una volatilidad variable en el tiempo.

El objetivo principal es analizar las ventajas y limitaciones del GBM y estudiar si la incorporación de volatilidad condicional mediante GARCH proporciona mejoras en la representación de los datos financieros.

## Estructura del proyecto

```text
TFG/
│
├── README.md
│
├── codigo/
│   ├── simulador_gbm.py
│   ├── analisis_gbm.py
│   ├── modelo_garch.py
│   └── comparacion_modelos.py
│
├── datos/
│   └── datos.csv
│
├── resultados/
│   ├── descripcion_datos.csv
│   ├── calibracion_gbm.csv
│   ├── validacion_gbm.csv
│   ├── sensibilidad_mu.csv
│   ├── sensibilidad_sigma.csv
│   ├── convergencia_monte_carlo.csv
│   ├── parametros_garch.csv
│   ├── comparacion_montecarlo_black_scholes.csv
│   └── comparacion_gbm_garch.csv
│
├── figuras/
│   └── *.png
│
└── memoria/
    └── TFG.pdf
