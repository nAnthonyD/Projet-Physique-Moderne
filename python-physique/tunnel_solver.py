"""
Unites reduites : hbar = 1, m = 1.
Dans ces unites :
    - la relation de dispersion est  omega = k^2 / 2
    - la vitesse de groupe d'un paquet centre sur k0 est  v_g = k0
    - l'energie moyenne (terme dominant) est  E = k0^2 / 2

Conventions de notation:
    - sigma : largeur spatiale du paquet d'ondes gaussien
    - L     : largeur (longueur) de la barriere de potentiel
    - V0    : hauteur de la barriere
"""

import numpy as np
from scipy.sparse import diags, identity
from scipy.sparse.linalg import splu

HBAR = 1.0
M = 1.0


# ----------------------------------------------------------------------
# 1. Paquet d'ondes gaussien initial
# ----------------------------------------------------------------------
def gaussian_packet(x, x0, k0, sigma):
    """Paquet d'ondes gaussien normalise a t=0.

    Psi(x,0) = (1/(2 pi sigma^2))^(1/4) * exp(-(x-x0)^2/(4 sigma^2)) * exp(i k0 x)

    x0     : position initiale du centre
    k0     : nombre d'onde central (-> vitesse de groupe v_g = hbar k0 / m)
    sigma  : largeur spatiale (ecart-type de |Psi|^2 vaut sigma)
    """
    norm = (1.0 / (2.0 * np.pi * sigma**2)) ** 0.25
    enveloppe = np.exp(-((x - x0) ** 2) / (4.0 * sigma**2))
    phase = np.exp(1j * k0 * x)
    return norm * enveloppe * phase


# ----------------------------------------------------------------------
# 2. Potentiel : barriere rectangulaire
# ----------------------------------------------------------------------
def barriere(x, V0, x_debut, L):
    """Barriere rectangulaire de hauteur V0, de x_debut a x_debut + L."""
    V = np.zeros_like(x)
    V[(x >= x_debut) & (x <= x_debut + L)] = V0
    return V


# ----------------------------------------------------------------------
# 3. Coeur : evolution Crank-Nicolson
# ----------------------------------------------------------------------
def evolue_CN(psi0, V, dx, dt, nt):
    """Fait evoluer psi0 pendant nt pas de temps par Crank-Nicolson.

    On resout, a chaque pas :   A . psi^{n+1} = B . psi^n
    avec  A = I + i dt/(2 hbar) H   et   B = I - i dt/(2 hbar) H
    H = -hbar^2/(2m) d2/dx2 + V   (matrice tridiagonale)

    A etant constante, on la factorise une seule fois (LU) -> rapide.

    Retourne un tableau 2D PSI de forme (nt+1, nx) : une ligne par instant.
    """
    nx = len(psi0)

    # Operateur energie cinetique : -hbar^2/(2m) * (deuxieme difference)/dx^2
    coeff = -(HBAR**2) / (2.0 * M * dx**2)
    diag_princ = -2.0 * coeff * np.ones(nx) + V
    diag_off = coeff * np.ones(nx - 1)
    H = diags([diag_off, diag_princ, diag_off], offsets=[-1, 0, 1], format="csc")

    I = identity(nx, format="csc")
    facteur = 1j * dt / (2.0 * HBAR)
    A = (I + facteur * H).tocsc()
    B = (I - facteur * H).tocsc()

    lu = splu(A)  # factorisation LU une seule fois

    PSI = np.zeros((nt + 1, nx), dtype=complex)
    PSI[0] = psi0
    psi = psi0.copy()
    for n in range(nt):
        psi = lu.solve(B.dot(psi))
        PSI[n + 1] = psi
    return PSI


# ----------------------------------------------------------------------
# 4. Diagnostics physiques
# ----------------------------------------------------------------------
def norme(psi, dx):
    """Norme ∫|psi|^2 dx (doit rester = 1)."""
    return np.sum(np.abs(psi) ** 2) * dx


def barycentre(psi, x, dx):
    """Position moyenne <x> = ∫ x |psi|^2 dx (suivi du centre du paquet)."""
    return np.sum(x * np.abs(psi) ** 2) * dx


def transmission_reflexion(psi, x, dx, x_fin_barriere):
    """Coefficients T et R apres separation du paquet.

    T = proba a droite de la barriere, R = proba a gauche.
    A appeler quand le paquet transmis et le paquet reflechi sont
    bien separes (mais avant qu'ils n'atteignent les bords du domaine).
    """
    densite = np.abs(psi) ** 2
    T = np.sum(densite[x > x_fin_barriere]) * dx
    R = np.sum(densite[x < x_fin_barriere]) * dx
    return T, R


def T_analytique(E, V0, L):
    """Coefficient de transmission ANALYTIQUE d'une onde plane d'energie E
    a travers une barriere rectangulaire (hbar = m = 1).

    Sert de reference pour valider la simulation.
    """
    if E < V0:  # effet tunnel : E < V0
        kappa = np.sqrt(2 * M * (V0 - E)) / HBAR
        return 1.0 / (1.0 + (V0**2 * np.sinh(kappa * L) ** 2)
                      / (4.0 * E * (V0 - E)))
    elif E > V0:  # au-dessus de la barriere
        k2 = np.sqrt(2 * M * (E - V0)) / HBAR
        return 1.0 / (1.0 + (V0**2 * np.sin(k2 * L) ** 2)
                      / (4.0 * E * (E - V0)))
    else:
        return 1.0 / (1.0 + (M * V0 * L**2) / (2.0 * HBAR**2))
