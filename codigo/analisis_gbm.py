"""
analisis_gbm.py
===============
Calibración, validación, convergencia Monte Carlo y análisis de sensibilidad
del GBM utilizando datos_sp500.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from simulador_gbm import SimuladorGBM, calibrar_gbm


BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "datos_sp500.csv"
RESULTADOS = BASE / "resultados"
FIGURAS = BASE / "figuras"
RESULTADOS.mkdir(exist_ok=True)
FIGURAS.mkdir(exist_ok=True)

SESIONES_ANUALES = 252
SEED = 42
N_SIMULACIONES = 10_000


def cargar_datos(path=DATA_PATH):
    """
    Carga el CSV de Investing.com.

    El archivo puede contener precios con separadores de miles o comas.
    La función intenta convertir la columna Price de forma robusta.
    """
    df = pd.read_csv(path)

    if "Date" not in df.columns or "Price" not in df.columns:
        raise ValueError("El CSV debe contener las columnas 'Date' y 'Price'.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Conversión robusta de precio, eliminando separadores de miles.
    precio = (
        df["Price"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df["Price"] = pd.to_numeric(precio, errors="coerce")

    df = (
        df.dropna(subset=["Date", "Price"])
        .sort_values("Date")
        .drop_duplicates(subset="Date")
        .reset_index(drop=True)
    )

    return df


def descripcion_datos(df):
    """Genera estadísticas básicas y comprobaciones de calidad."""
    resumen = {
        "fecha_inicial": df["Date"].min(),
        "fecha_final": df["Date"].max(),
        "n_observaciones": len(df),
        "fechas_duplicadas": int(df["Date"].duplicated().sum()),
        "faltantes_price": int(df["Price"].isna().sum()),
    }

    resumen_df = pd.DataFrame(
        [{"variable": k, "valor": v} for k, v in resumen.items()]
    )
    resumen_df.to_csv(RESULTADOS / "descripcion_datos.csv", index=False)
    return resumen


def dividir_train_test(df, proporcion_train=0.80):
    n_train = int(len(df) * proporcion_train)
    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].copy()
    return train, test


def metricas(y_real, y_pred):
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(np.mean((y_real - y_pred) ** 2))
    mae = np.mean(np.abs(y_real - y_pred))

    mascara = y_real != 0
    mape = 100.0 * np.mean(
        np.abs((y_real[mascara] - y_pred[mascara]) / y_real[mascara])
    )

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}



def graficas_descriptivas(df):
    precios = df["Price"].to_numpy(dtype=float)
    fechas = df["Date"]
    rendimientos = np.diff(np.log(precios))
    fechas_r = fechas.iloc[1:]

    plt.figure(figsize=(10, 5)); plt.plot(fechas, precios)
    plt.xlabel("Fecha"); plt.ylabel("Precio de cierre"); plt.title("Evolución histórica del S&P 500")
    plt.tight_layout(); plt.savefig(FIGURAS / "precio_historico.png", dpi=200); plt.close()

    plt.figure(figsize=(10, 5)); plt.plot(fechas_r, rendimientos)
    plt.xlabel("Fecha"); plt.ylabel("Rendimiento logarítmico"); plt.title("Rendimientos logarítmicos diarios")
    plt.tight_layout(); plt.savefig(FIGURAS / "rendimientos_logaritmicos.png", dpi=200); plt.close()

    plt.figure(figsize=(8, 5)); plt.hist(rendimientos, bins=50, density=True, alpha=0.7)
    plt.xlabel("Rendimiento logarítmico"); plt.ylabel("Densidad"); plt.title("Distribución de los rendimientos logarítmicos")
    plt.tight_layout(); plt.savefig(FIGURAS / "histograma_rendimientos.png", dpi=200); plt.close()


def graficar_trayectorias(simulador, N=50):
    """Genera una muestra de trayectorias simuladas del GBM."""
    tiempos, S = simulador.simular(N=N, seed=SEED)
    plt.figure(figsize=(10, 6))
    for trayectoria in S: plt.plot(tiempos, trayectoria, alpha=0.6)
    plt.xlabel("Tiempo (años)"); plt.ylabel("Precio"); plt.title("Trayectorias simuladas mediante GBM")
    plt.tight_layout(); plt.savefig(FIGURAS / "trayectorias_gbm.png", dpi=200); plt.close()

def validar_distribucion(simulador, N=N_SIMULACIONES):
    """Compara S_T simulado con sus momentos teóricos."""
    _, S = simulador.simular(N=N, seed=SEED)
    ST = S[:, -1]
    log_ST = np.log(ST)

    media_teo = simulador.esperanza_teorica()
    var_teo = simulador.varianza_teorica()

    media_sim = np.mean(ST)
    var_sim = np.var(ST, ddof=1)

    tabla = pd.DataFrame(
        {
            "estadistico": ["Media", "Varianza"],
            "teorico": [media_teo, var_teo],
            "simulado": [media_sim, var_sim],
        }
    )
    tabla["error_relativo_pct"] = (
        np.abs(tabla["simulado"] - tabla["teorico"])
        / np.abs(tabla["teorico"])
        * 100
    )
    tabla.to_csv(RESULTADOS / "validacion_gbm.csv", index=False)

    # QQ-plot de log(S_T)
    plt.figure(figsize=(7, 5))
    stats.probplot(log_ST, dist="norm", plot=plt)
    plt.title("QQ-plot de log(S_T)")
    plt.tight_layout()
    plt.savefig(FIGURAS / "qqplot_log_ST.png", dpi=200)
    plt.close()

    return tabla


def analizar_convergencia(simulador):
    """Estudia empíricamente el orden O(N^{-1/2})."""
    Ns = np.array([100, 500, 1000, 5000, 10000, 50000])
    media_teo = simulador.esperanza_teorica()

    errores = []
    for N in Ns:
        ST = simulador.precio_terminal_directo(N=N, seed=SEED + int(N))
        error = abs(np.mean(ST) - media_teo)
        errores.append(error)

    errores = np.asarray(errores)
    pendiente, intercepto = np.polyfit(np.log(Ns), np.log(errores), 1)

    tabla = pd.DataFrame({"N": Ns, "error_absoluto": errores})
    tabla["pendiente_estimacion"] = pendiente
    tabla.to_csv(RESULTADOS / "convergencia_monte_carlo.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.loglog(Ns, errores, "o-", label="Error empírico")
    referencia = np.exp(intercepto) * Ns ** (-0.5)
    plt.loglog(Ns, referencia, "--", label=r"Referencia $N^{-1/2}$")
    plt.xlabel("Número de simulaciones N")
    plt.ylabel("Error absoluto de la media")
    plt.title(f"Convergencia Monte Carlo. Pendiente = {pendiente:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURAS / "convergencia_monte_carlo.png", dpi=200)
    plt.close()

    return pendiente, tabla


def sensibilidad(S0, T=1.0, pasos=252):
    """
    Analiza explícitamente media y varianza terminal frente a mu y sigma.
    """
    mus = [0.02, 0.05, 0.08, 0.12]
    sigmas = [0.10, 0.18, 0.25, 0.35]
    sigma_fija = 0.18
    mu_fijo = 0.08

    filas_mu = []
    for mu in mus:
        sim = SimuladorGBM(S0=S0, mu=mu, sigma=sigma_fija,
                           T=T, pasos=pasos)
        _, S = sim.simular(N=N_SIMULACIONES, seed=SEED)
        ST = S[:, -1]
        filas_mu.append(
            {
                "mu": mu,
                "media_simulada": np.mean(ST),
                "media_teorica": sim.esperanza_teorica(),
                "varianza_simulada": np.var(ST, ddof=1),
                "varianza_teorica": sim.varianza_teorica(),
                "p5": np.percentile(ST, 5),
                "p95": np.percentile(ST, 95),
            }
        )

    filas_sigma = []
    for sigma in sigmas:
        sim = SimuladorGBM(S0=S0, mu=mu_fijo, sigma=sigma,
                           T=T, pasos=pasos)
        _, S = sim.simular(N=N_SIMULACIONES, seed=SEED)
        ST = S[:, -1]
        filas_sigma.append(
            {
                "sigma": sigma,
                "media_simulada": np.mean(ST),
                "media_teorica": sim.esperanza_teorica(),
                "varianza_simulada": np.var(ST, ddof=1),
                "varianza_teorica": sim.varianza_teorica(),
                "p5": np.percentile(ST, 5),
                "p95": np.percentile(ST, 95),
            }
        )

    df_mu = pd.DataFrame(filas_mu)
    df_sigma = pd.DataFrame(filas_sigma)

    df_mu.to_csv(RESULTADOS / "sensibilidad_mu.csv", index=False)
    df_sigma.to_csv(RESULTADOS / "sensibilidad_sigma.csv", index=False)

    # Varianza frente a mu
    plt.figure(figsize=(7, 5))
    plt.plot(df_mu["mu"], df_mu["varianza_simulada"], "o-",
             label="Simulada")
    plt.plot(df_mu["mu"], df_mu["varianza_teorica"], "s--",
             label="Teórica")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"Varianza de $S_T$")
    plt.title(r"Sensibilidad de la varianza respecto a $\mu$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURAS / "sensibilidad_mu_varianza.png", dpi=200)
    plt.close()

    # Varianza frente a sigma
    plt.figure(figsize=(7, 5))
    plt.plot(df_sigma["sigma"], df_sigma["varianza_simulada"], "o-",
             label="Simulada")
    plt.plot(df_sigma["sigma"], df_sigma["varianza_teorica"], "s--",
             label="Teórica")
    plt.xlabel(r"$\sigma$")
    plt.ylabel(r"Varianza de $S_T$")
    plt.title(r"Sensibilidad de la varianza respecto a $\sigma$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURAS / "sensibilidad_sigma_varianza.png", dpi=200)
    plt.close()

    return df_mu, df_sigma


def main():
    df = cargar_datos()
    info = descripcion_datos(df)
    graficas_descriptivas(df)

    print("=== DESCRIPCIÓN DE DATOS ===")
    for k, v in info.items():
        print(f"{k}: {v}")

    train, test = dividir_train_test(df)

    cal = calibrar_gbm(train["Price"].to_numpy(),
                       sesiones_anuales=SESIONES_ANUALES)

    S0 = float(train["Price"].iloc[-1])
    horizonte = len(test) / SESIONES_ANUALES

    simulador = SimuladorGBM(
        S0=S0,
        mu=cal["mu"],
        sigma=cal["sigma"],
        T=horizonte,
        pasos=len(test),
    )

    graficar_trayectorias(simulador)

    print("\n=== CALIBRACIÓN GBM ===")
    print(f"mu = {cal['mu']:.6f}")
    print(f"sigma = {cal['sigma']:.6f}")

    print("\n=== VALIDACIÓN TEÓRICA ===")
    print(validar_distribucion(simulador).to_string(index=False))

    pendiente, _ = analizar_convergencia(simulador)
    print(f"\nPendiente de convergencia estimada: {pendiente:.4f}")

    sensibilidad(S0=float(df["Price"].iloc[0]))

    # Predicción puntual esperada para validación
    t = np.arange(1, len(test) + 1) / SESIONES_ANUALES
    pred_media = S0 * np.exp(cal["mu"] * t)
    met = metricas(test["Price"].to_numpy(), pred_media)

    pd.DataFrame([met]).to_csv(
        RESULTADOS / "metricas_gbm_validacion.csv", index=False
    )

    print("\n=== MÉTRICAS GBM FUERA DE MUESTRA ===")
    for k, v in met.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
