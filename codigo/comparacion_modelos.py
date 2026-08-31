"""
comparacion_modelos.py
======================
Comparación GBM vs GARCH(1,1) usando:
    - el mismo S&P 500;
    - el mismo conjunto de entrenamiento;
    - el mismo conjunto de validación;
    - el mismo precio inicial;
    - el mismo horizonte temporal.

Para instalar dependencias:
    pip install numpy pandas matplotlib arch scipy
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from simulador_gbm import SimuladorGBM, calibrar_gbm
from modelo_garch import (
    cargar_datos,
    rendimientos_logaritmicos,
    ajustar_garch,
    extraer_parametros,
    simular_garch_precio,
)

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"
FIGURAS = BASE / "figuras"
RESULTADOS.mkdir(exist_ok=True)
FIGURAS.mkdir(exist_ok=True)

SESIONES_ANUALES = 252
SEED = 42
N_SIMULACIONES = 10_000


def metricas(y_real, y_pred):
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(np.mean((y_real - y_pred) ** 2))
    mae = np.mean(np.abs(y_real - y_pred))
    mape = 100.0 * np.mean(
        np.abs((y_real - y_pred) / y_real)
    )
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def cobertura(y_real, inferior, superior):
    y_real = np.asarray(y_real)
    inferior = np.asarray(inferior)
    superior = np.asarray(superior)
    return 100.0 * np.mean(
        (y_real >= inferior) & (y_real <= superior)
    )




def precio_black_scholes_call(S0, K, r, sigma, T):
    """Precio analítico de una call europea Black--Scholes."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def comparacion_black_scholes(S0, sigma, seed=SEED):
    """Compara valoración Monte Carlo bajo GBM con Black--Scholes."""
    K, r, T = S0, 0.04, 1.0
    Ns = np.array([100, 500, 1000, 5000, 10000, 50000])
    precio_bs = precio_black_scholes_call(S0, K, r, sigma, T)
    estimaciones = []
    rng = np.random.default_rng(seed)
    ST_final = None
    for N in Ns:
        Z = rng.standard_normal(N)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        precio_mc = np.exp(-r * T) * np.mean(np.maximum(ST - K, 0.0))
        estimaciones.append(precio_mc)
        if N == Ns[-1]: ST_final = ST
    estimaciones = np.asarray(estimaciones)
    tabla = pd.DataFrame({'N':Ns,'Monte_Carlo':estimaciones,'Black_Scholes':precio_bs,'Error_absoluto':np.abs(estimaciones-precio_bs)})
    tabla.to_csv(RESULTADOS/'comparacion_montecarlo_black_scholes.csv',index=False)

    plt.figure(figsize=(8,5)); plt.hist(ST_final,bins=60,density=True,alpha=0.7)
    plt.axvline(S0,linestyle='--',label='Precio inicial'); plt.axvline(K,linestyle=':',label='Strike')
    plt.xlabel('Precio terminal $S_T$'); plt.ylabel('Densidad'); plt.title('Distribución simulada del precio terminal'); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURAS/'distribucion_precio_terminal.png',dpi=220); plt.close()

    plt.figure(figsize=(8,5)); plt.semilogx(Ns,estimaciones,'o-',label='Monte Carlo')
    plt.axhline(precio_bs,linestyle='--',label='Black--Scholes'); plt.xlabel('Número de simulaciones N'); plt.ylabel('Precio de la opción'); plt.title('Convergencia Monte Carlo hacia Black--Scholes'); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURAS/'convergencia_black_scholes.png',dpi=220); plt.close()
    return tabla

def main():
    # ======================================================
    # 1. DATOS Y DIVISIÓN TEMPORAL
    # ======================================================
    df = cargar_datos()
    n_train = int(len(df) * 0.80)

    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].copy()

    precios_train = train["Price"].to_numpy()
    precios_test = test["Price"].to_numpy()
    fechas_test = test["Date"].to_numpy()

    S0 = float(precios_train[-1])
    horizonte = len(test)
    T = horizonte / SESIONES_ANUALES

    # ======================================================
    # 2. GBM
    # ======================================================
    cal_gbm = calibrar_gbm(precios_train, SESIONES_ANUALES)

    gbm = SimuladorGBM(
        S0=S0,
        mu=cal_gbm["mu"],
        sigma=cal_gbm["sigma"],
        T=T,
        pasos=horizonte,
    )

    _, S_gbm = gbm.simular(
        N=N_SIMULACIONES,
        seed=SEED,
    )

    gbm_media = S_gbm[:, 1:].mean(axis=0)
    gbm_p5 = np.percentile(S_gbm[:, 1:], 5, axis=0)
    gbm_p95 = np.percentile(S_gbm[:, 1:], 95, axis=0)

    # ======================================================
    # 3. GARCH(1,1)
    # ======================================================
    r_train = rendimientos_logaritmicos(train["Price"])
    resultado_garch = ajustar_garch(r_train)
    par_garch = extraer_parametros(resultado_garch)

    S_garch = simular_garch_precio(
        resultado_garch,
        S0=S0,
        horizonte=horizonte,
        n_simulaciones=N_SIMULACIONES,
        seed=SEED,
    )

    garch_media = S_garch[:, 1:].mean(axis=0)
    garch_p5 = np.percentile(S_garch[:, 1:], 5, axis=0)
    garch_p95 = np.percentile(S_garch[:, 1:], 95, axis=0)

    # ======================================================
    # 4. MÉTRICAS
    # ======================================================
    met_gbm = metricas(precios_test, gbm_media)
    met_garch = metricas(precios_test, garch_media)

    cov_gbm = cobertura(precios_test, gbm_p5, gbm_p95)
    cov_garch = cobertura(precios_test, garch_p5, garch_p95)

    tabla = pd.DataFrame(
        {
            "Métrica": ["RMSE", "MAE", "MAPE (%)", "Cobertura 5-95 (%)"],
            "GBM": [
                met_gbm["RMSE"],
                met_gbm["MAE"],
                met_gbm["MAPE"],
                cov_gbm,
            ],
            "GARCH(1,1)": [
                met_garch["RMSE"],
                met_garch["MAE"],
                met_garch["MAPE"],
                cov_garch,
            ],
        }
    )

    tabla.to_csv(
        RESULTADOS / "comparacion_gbm_garch.csv",
        index=False,
    )

    print("\n=== COMPARACIÓN GBM VS GARCH(1,1) ===")
    print(tabla.to_string(index=False))

    print("\n=== PARÁMETROS ===")
    print(f"GBM mu anual: {cal_gbm['mu']:.6f}")
    print(f"GBM sigma anual: {cal_gbm['sigma']:.6f}")
    for k, v in par_garch.items():
        print(f"GARCH {k}: {v:.8f}")

    # ======================================================
    # 5. GRÁFICO: PRECIO REAL Y MEDIAS SIMULADAS
    # ======================================================
    plt.figure(figsize=(11, 6))
    plt.plot(fechas_test, precios_test, label="Precio real")
    plt.plot(fechas_test, gbm_media, label="Media GBM")
    plt.plot(fechas_test, garch_media, label="Media GARCH(1,1)")

    plt.fill_between(
        fechas_test,
        gbm_p5,
        gbm_p95,
        alpha=0.15,
        label="Banda GBM 5-95 %",
    )

    plt.fill_between(
        fechas_test,
        garch_p5,
        garch_p95,
        alpha=0.15,
        label="Banda GARCH 5-95 %",
    )

    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.title("Comparación fuera de muestra: GBM vs GARCH(1,1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURAS / "comparacion_precios_gbm_garch.png",
        dpi=220,
    )
    plt.close()

    # ======================================================
    # 6. GRÁFICO: VOLATILIDAD
    # ======================================================
    # Volatilidad condicional estimada en entrenamiento
    fechas_vol = train["Date"].iloc[1:].to_numpy()
    vol_garch = (
        np.asarray(resultado_garch.conditional_volatility)
        / 100.0
        * np.sqrt(SESIONES_ANUALES)
    )

    plt.figure(figsize=(11, 5))
    plt.plot(
        fechas_vol,
        vol_garch,
        label="Volatilidad condicional GARCH",
    )
    plt.axhline(
        cal_gbm["sigma"],
        linestyle="--",
        label="Volatilidad constante GBM",
    )
    plt.xlabel("Fecha")
    plt.ylabel("Volatilidad anualizada")
    plt.title("Volatilidad: GARCH(1,1) frente a GBM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURAS / "comparacion_volatilidad_gbm_garch.png",
        dpi=220,
    )
    plt.close()

    # ======================================================
    # 7. GUARDAR SERIES PARA LATEX O ANÁLISIS POSTERIOR
    # ======================================================
    series = pd.DataFrame(
        {
            "Date": fechas_test,
            "Precio_real": precios_test,
            "GBM_media": gbm_media,
            "GBM_p5": gbm_p5,
            "GBM_p95": gbm_p95,
            "GARCH_media": garch_media,
            "GARCH_p5": garch_p5,
            "GARCH_p95": garch_p95,
        }
    )

    series.to_csv(
        RESULTADOS / "series_comparacion_gbm_garch.csv",
        index=False,
    )

    # 8. COMPARACIÓN MONTE CARLO (GBM) VS BLACK--SCHOLES
    tabla_bs = comparacion_black_scholes(S0, cal_gbm["sigma"])
    print("\n=== MONTE CARLO (GBM) VS BLACK--SCHOLES ===")
    print(tabla_bs.to_string(index=False))

    print("\nArchivos generados en:")
    print(RESULTADOS)
    print(FIGURAS)


if __name__ == "__main__":
    main()
