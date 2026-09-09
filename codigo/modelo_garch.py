"""
modelo_garch.py
===============
Ajuste, predicción y simulación de un modelo GARCH(1,1).

Dependencias:
    pip install arch pandas numpy matplotlib
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model


BASE = Path(__file__).resolve().parent
DATA_PATH = BASE.parent / "datos" / "datos_sp500.csv"
RESULTADOS = BASE.parent / "resultados"
FIGURAS = BASE.parent / "figuras"
RESULTADOS.mkdir(exist_ok=True)
FIGURAS.mkdir(exist_ok=True)

SESIONES_ANUALES = 252
SEED = 42


def cargar_datos(path=DATA_PATH):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    precio = (
        df["Price"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df["Price"] = pd.to_numeric(precio, errors="coerce")

    return (
        df.dropna(subset=["Date", "Price"])
        .sort_values("Date")
        .drop_duplicates(subset="Date")
        .reset_index(drop=True)
    )


def rendimientos_logaritmicos(precios):
    return np.log(precios).diff().dropna()


def ajustar_garch(rendimientos, distribucion="normal"):
    """
    Ajusta un GARCH(1,1) con media constante.

    Se multiplican los rendimientos por 100 para trabajar en porcentaje,
    práctica habitual al utilizar la librería arch.
    """
    r_pct = 100.0 * np.asarray(rendimientos, dtype=float)

    modelo = arch_model(
        r_pct,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist=distribucion,
        rescale=False,
    )
    resultado = modelo.fit(disp="off")
    return resultado


def extraer_parametros(resultado):
    p = resultado.params
    return {
        "mu_diario": float(p.get("mu", 0.0) / 100.0),
        "omega": float(p["omega"] / 10000.0),
        "alpha": float(p["alpha[1]"]),
        "beta": float(p["beta[1]"]),
        "alpha_mas_beta": float(p["alpha[1]"] + p["beta[1]"]),
    }


def pronostico_garch(resultado, horizonte):
    """
    Pronóstico condicional de media y volatilidad para 'horizonte' días.

    Devuelve la media y la desviación típica de los rendimientos en escala
    decimal, no en porcentaje.
    """
    forecast = resultado.forecast(horizon=horizonte, reindex=False)

    media = forecast.mean.values[-1] / 100.0
    varianza = forecast.variance.values[-1] / 10000.0
    sigma = np.sqrt(varianza)

    return media, sigma


def simular_garch_precio(
    resultado,
    S0,
    horizonte,
    n_simulaciones=10000,
    seed=SEED,
):
    """
    Genera trayectorias de rendimientos y precios usando la dinámica GARCH(1,1).

    La simulación se realiza en escala decimal:
        r_t = mu + eps_t
        eps_t = sigma_t z_t
        sigma_t^2 = omega + alpha eps_{t-1}^2 + beta sigma_{t-1}^2

    Para iniciar la simulación fuera de muestra se calcula previamente la
    varianza condicional del primer período futuro utilizando el último
    residuo observado y la última varianza condicional estimada.
    """
    par = extraer_parametros(resultado)
    rng = np.random.default_rng(seed)

    mu = par["mu_diario"]
    omega = par["omega"]
    alpha = par["alpha"]
    beta = par["beta"]

    # ============================================================
    # CONDICIÓN INICIAL PARA LA SIMULACIÓN FUERA DE MUESTRA
    # ============================================================

    # Última volatilidad condicional estimada (escala decimal)
    sigma_T = float(np.asarray(resultado.conditional_volatility)[-1]) / 100.0
    sigma2_T = sigma_T ** 2

    # Último residuo observado (escala decimal)
    eps_T = float(np.asarray(resultado.resid)[-1]) / 100.0

    # Varianza condicional del primer período futuro:
    # sigma_{T+1}^2 = omega + alpha * eps_T^2 + beta * sigma_T^2
    sigma2_0 = (
            omega
            + alpha * eps_T ** 2
            + beta * sigma2_T
    )

    # ============================================================
    # SIMULACIÓN
    # ============================================================

    precios = np.empty((n_simulaciones, horizonte + 1))
    precios[:, 0] = S0

    # La primera simulación utiliza sigma_{T+1}^2
    sigma2 = np.full(n_simulaciones, sigma2_0)

    for t in range(1, horizonte + 1):
        z = rng.standard_normal(n_simulaciones)

        sigma = np.sqrt(np.maximum(sigma2, 1e-16))
        eps = sigma * z
        r = mu + eps

        # Actualización del precio mediante rendimientos logarítmicos
        precios[:, t] = precios[:, t - 1] * np.exp(r)

        # Actualización de la varianza para el siguiente período
        sigma2 = (
            omega
            + alpha * eps ** 2
            + beta * sigma2
        )

    return precios


def main():
    df = cargar_datos()
    n_train = int(len(df) * 0.80)
    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].copy()

    r_train = rendimientos_logaritmicos(train["Price"])

    resultado = ajustar_garch(r_train)
    par = extraer_parametros(resultado)

    print(resultado.summary())
    print("\n=== PARÁMETROS GARCH(1,1) ===")
    for k, v in par.items():
        print(f"{k}: {v:.8f}")

    pd.DataFrame([par]).to_csv(
        RESULTADOS / "parametros_garch.csv", index=False
    )

    # Volatilidad condicional histórica
    fechas_r = train["Date"].iloc[1:].to_numpy()
    vol_anual = (
        np.asarray(resultado.conditional_volatility)
        / 100.0
        * np.sqrt(SESIONES_ANUALES)
    )

    plt.figure(figsize=(10, 5))
    plt.plot(fechas_r, vol_anual)
    plt.xlabel("Fecha")
    plt.ylabel("Volatilidad anualizada")
    plt.title("Volatilidad condicional estimada por GARCH(1,1)")
    plt.tight_layout()
    plt.savefig(FIGURAS / "volatilidad_garch.png", dpi=200)
    plt.close()

    media_f, sigma_f = pronostico_garch(resultado, len(test))
    tabla_forecast = pd.DataFrame(
        {
            "paso": np.arange(1, len(test) + 1),
            "media_rendimiento": media_f,
            "sigma_condicional": sigma_f,
        }
    )
    tabla_forecast.to_csv(
        RESULTADOS / "pronostico_garch.csv", index=False
    )

    S0 = float(train["Price"].iloc[-1])
    precios_sim = simular_garch_precio(
        resultado,
        S0=S0,
        horizonte=len(test),
        n_simulaciones=10000,
        seed=SEED,
    )

    np.save(RESULTADOS / "trayectorias_garch.npy", precios_sim)
    print("\nTrayectorias GARCH guardadas correctamente.")


if __name__ == "__main__":
    main()
