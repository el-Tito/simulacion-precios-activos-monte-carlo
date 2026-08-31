"""
simulador_gbm.py
================
Funciones y clase para simular un Movimiento Browniano Geométrico (GBM).

Modelo:
    dS_t = mu S_t dt + sigma S_t dW_t

Solución exacta discretizada:
    S_{t+dt} = S_t exp((mu - 0.5 sigma^2) dt
                       + sigma sqrt(dt) Z)
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SimuladorGBM:
    S0: float
    mu: float
    sigma: float
    T: float = 1.0
    pasos: int = 252

    @property
    def dt(self):
        return self.T / self.pasos

    def simular(self, N=10000, seed=None):
        """
        Genera N trayectorias independientes del GBM.

        Devuelve
        --------
        tiempos : ndarray, shape (pasos + 1,)
        S : ndarray, shape (N, pasos + 1)
        """
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((N, self.pasos))

        incrementos_log = (
            (self.mu - 0.5 * self.sigma**2) * self.dt
            + self.sigma * np.sqrt(self.dt) * Z
        )

        log_relativos = np.cumsum(incrementos_log, axis=1)

        S = np.empty((N, self.pasos + 1))
        S[:, 0] = self.S0
        S[:, 1:] = self.S0 * np.exp(log_relativos)

        tiempos = np.linspace(0.0, self.T, self.pasos + 1)
        return tiempos, S

    def precio_terminal_directo(self, N=10000, seed=None):
        """Simula directamente S_T usando la solución cerrada."""
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal(N)

        ST = self.S0 * np.exp(
            (self.mu - 0.5 * self.sigma**2) * self.T
            + self.sigma * np.sqrt(self.T) * Z
        )
        return ST

    def esperanza_teorica(self, t=None):
        """E[S_t] = S0 exp(mu t)."""
        if t is None:
            t = self.T
        return self.S0 * np.exp(self.mu * np.asarray(t))

    def varianza_teorica(self, t=None):
        """
        Var(S_t) =
        S0^2 exp(2 mu t) [exp(sigma^2 t) - 1].
        """
        if t is None:
            t = self.T
        t = np.asarray(t)
        return (
            self.S0**2
            * np.exp(2.0 * self.mu * t)
            * (np.exp(self.sigma**2 * t) - 1.0)
        )

    @staticmethod
    def estadisticos_terminales(S):
        """Calcula estadísticos empíricos del precio terminal."""
        ST = S[:, -1]
        return {
            "media": float(np.mean(ST)),
            "mediana": float(np.median(ST)),
            "varianza": float(np.var(ST, ddof=1)),
            "desviacion_tipica": float(np.std(ST, ddof=1)),
            "p5": float(np.percentile(ST, 5)),
            "p50": float(np.percentile(ST, 50)),
            "p95": float(np.percentile(ST, 95)),
        }


def calibrar_gbm(precios, sesiones_anuales=252):
    """
    Calibra mu y sigma a partir de precios históricos.

    Para rendimientos logarítmicos diarios r:
        sigma = sqrt(252) * std(r)
        mu = 252 * mean(r) + 0.5 * sigma^2
    """
    precios = np.asarray(precios, dtype=float)
    if len(precios) < 2:
        raise ValueError("Se necesitan al menos dos precios.")

    r = np.diff(np.log(precios))
    sigma = np.sqrt(sesiones_anuales) * np.std(r, ddof=1)
    mu = sesiones_anuales * np.mean(r) + 0.5 * sigma**2

    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "rendimientos": r,
    }
