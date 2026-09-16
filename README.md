# Simulación de precios de activos financieros mediante el método de Monte Carlo: aplicación y comparación de modelos estocásticos

Repositorio asociado al Trabajo Fin de Grado del Grado en Matemática Computacional de la Universidad Internacional de La Rioja (UNIR).

## Descripción

Este proyecto desarrolla un estudio sobre la simulación de precios de activos financieros mediante métodos de Monte Carlo.

El trabajo se centra en la aplicación del Movimiento Browniano Geométrico (GBM) como modelo estocástico para representar la evolución temporal de los precios de activos financieros. A partir de datos históricos, se realiza la calibración de los parámetros del modelo y se generan trayectorias simuladas mediante técnicas de Monte Carlo.

Asimismo, se estudia la convergencia del método de Monte Carlo y se realiza un análisis de sensibilidad respecto a los parámetros de deriva y volatilidad.

Como parte del análisis, se compara la valoración obtenida mediante simulación de Monte Carlo con la solución analítica de Black--Scholes. Finalmente, se incorpora un modelo GARCH(1,1) como alternativa al GBM para analizar el efecto de una volatilidad variable en el tiempo.

El objetivo es estudiar las posibilidades y limitaciones de los modelos utilizados para la simulación de precios de activos financieros y evaluar el comportamiento de los diferentes enfoques considerados.

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
│   └── datos_sp500.csv
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
